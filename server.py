"""AquaLens local JSON API and custom web dashboard (no Streamlit)."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import gzip
import hashlib
import math
import re
from pathlib import Path
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

import numpy as np
import pandas as pd
from aqualens.data import load_demo, load_data, zones_summary, PARAMETERS
from aqualens.alerts import detect_alerts, current_condition, ALERT_POLICY
from aqualens.assessment import assess_water
from aqualens.monitoring import monitor_satellite, project_satellite
from aqualens.prophet_model import prophet_available
from aqualens.forecast import dataset_fingerprint, forecast_zones, train_models
from aqualens.evidence import load_knowledge, parse_knowledge, retrieve_evidence, explain_alert, evaluate_retrieval

ROOT = Path(__file__).resolve().parent
MAX_BODY = 12 * 1024 * 1024

class APIError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def clean(value):
    """JSON strict conversion: no NaN/Infinity or pandas-specific scalars."""
    if isinstance(value, pd.DataFrame):
        return clean(value.to_dict('records'))
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [clean(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp,datetime,date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Dashboard:
    def __init__(self, root=ROOT, *, use_custom_models=True):
        self.root = Path(root)
        self.use_custom_models = use_custom_models
        self.datasets = OrderedDict()
        self.lock = threading.RLock()
        self.training_lock = threading.Lock()
        self.jobs = {}
        self.collections = OrderedDict()
        self.cached = {}
        self.bootstrap_cache = OrderedDict()
        self.historical_cache = OrderedDict()
        self.torch_ready = None
        data_path = self.root / 'data' / 'modis_observations.csv'
        self.modis = load_data(data_path)
        self.modis[1]['filename'] = 'MODIS_Lake_Features_NDWI_Turbidity_Chlorophyll_2020_2025_.csv'
        self.demo = self.modis  # Compatibility alias; never silently use synthetic data.
        supplemental_path = self.root / 'data' / 'esa_vembanad_monitoring.csv.gz'
        self.supplemental = (
            pd.read_csv(supplemental_path, parse_dates=['date'])
            if supplemental_path.is_file() else None
        )

    def dataset(self, key='modis'):
        if key in ('modis', 'demo'):
            return self.modis
        with self.lock:
            if key not in self.datasets:
                raise APIError('Dataset expired or unknown. Upload it again.',404)
            return self.datasets[key]

    def knowledge(self, evidence_id=None):
        if not evidence_id:
            return load_knowledge()
        with self.lock:
            if evidence_id not in self.collections:
                raise APIError('Evidence collection expired or unknown.',404)
            return self.collections[evidence_id]

    def checkpoint(self, df, *, bundled=False):
        fingerprint = dataset_fingerprint(df)
        candidates = [self.root/'models']
        if not bundled and self.use_custom_models:
            candidates.insert(0, self.root/'models'/'custom'/fingerprint)
        for directory in candidates:
            meta = directory/'evaluation.json'
            checkpoint = directory/'det_v2.pt'
            if meta.exists() and checkpoint.exists():
                try:
                    summary = json.loads(meta.read_text(encoding='utf-8'))
                    if summary.get('fingerprint') == fingerprint:
                        return checkpoint,summary
                except (ValueError,OSError):
                    pass
        return None,{}

    def has_torch(self):
        with self.lock:
            if self.torch_ready is None:
                try:
                    import torch
                    torch.set_num_threads(2)
                    self.torch_ready = True
                except (ImportError,OSError):
                    self.torch_ready = False
            return self.torch_ready

    def models(self, df):
        checkpoint,summary = self.checkpoint(df)
        matched = bool(checkpoint and self.has_torch())
        if matched:
            # Validate the actual checkpoint, not just its neighboring metadata.
            signature = (str(checkpoint),checkpoint.stat().st_mtime_ns)
            with self.lock:
                checked = self.cached.get(('checkpoint',signature))
            if checked is None:
                from aqualens.forecast import _load_checkpoint
                try:
                    _load_checkpoint(df,checkpoint)
                    checked = True
                except (ValueError,RuntimeError,OSError,KeyError):
                    checked = False
                with self.lock:
                    self.cached[('checkpoint',signature)] = checked
            matched = checked
        reason = '' if matched else 'Train DET on this dataset to enable forecasts. A compatible checkpoint and PyTorch are required.'
        return [dict(id='det',name='DET',available=matched,reason=reason,quality=self.model_quality(summary))]

    @staticmethod
    def model_quality(summary):
        metrics=summary.get('metrics',[])
        valid=[row for row in metrics if row.get('r2') is not None and np.isfinite(row.get('r2',np.nan))]
        weak=[row for row in valid if row['r2']<0]
        return dict(evaluated=bool(valid),negative_r2_count=len(weak),metric_count=len(valid),
            status='Needs validation' if weak else 'Evaluated' if valid else 'Not evaluated',
            warning='Some held-out DET R2 scores are negative. Forecasts are exploratory and require better source data or model improvement.' if weak else None)

    def evaluation(self,key='modis'):
        df,_ = self.dataset(key)
        path,summary = self.checkpoint(df)
        if path:
            try:
                by_zone_path = path.parent/'evaluation_by_zone.csv'
                return clean(dict(rows=pd.read_csv(path.parent/'evaluation.csv'),
                     by_zone=pd.read_csv(by_zone_path) if by_zone_path.exists() else [],metadata=summary))
            except (ValueError,OSError):
                pass
        return dict(rows=[],by_zone=[],metadata=dict(
            evaluation='DET chronological held-out evaluation',
            limitations=['Train DET on this dataset before reporting forecast accuracy.']))

    def historical_signature(self):
        """Only this separate evaluation directory can supply historical results."""
        directory=self.root/'models'/'historical_validation'
        paths=[directory/name for name in ('summary.json','heldout_predictions.csv','det_historical.pt')]
        return tuple((path.stat().st_mtime_ns,path.stat().st_size) if path.is_file() else None for path in paths)

    def benchmark_signature(self):
        directory = self.root / 'models' / 'model_selection'
        return tuple(
            (path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else None
            for path in (directory / 'benchmark.json', directory / 'predictions.csv')
        )

    def model_benchmark(self, key='modis', fingerprint=None):
        df,_ = self.dataset(key)
        fingerprint = fingerprint or dataset_fingerprint(df)
        signature = self.benchmark_signature()
        cache_key = ('model_benchmark', fingerprint, signature)
        with self.lock:
            if cache_key in self.historical_cache:
                return self.historical_cache[cache_key]
        try:
            directory = self.root / 'models' / 'model_selection'
            saved = json.loads((directory / 'benchmark.json').read_text(encoding='utf-8')) if all(signature) else {}
            if saved.get('available') and saved.get('fingerprint') == fingerprint:
                benchmark = dict(saved, saved_history=True)
            elif key in ('modis', 'demo'):
                benchmark = dict(available=False, reason='The bundled model comparison is unavailable.')
            else:
                from aqualens.model_selection import fit_forecast_policy
                benchmark = dict(fit_forecast_policy(df), saved_history=False)
        except (ValueError, OSError, RuntimeError) as exc:
            benchmark = dict(available=False, reason=str(exc))
        benchmark = clean(benchmark)
        with self.lock:
            self.historical_cache[cache_key] = benchmark
            while len(self.historical_cache) > 10:
                self.historical_cache.popitem(last=False)
        return benchmark

    def historical_status(self,key='modis',fingerprint=None):
        """Read provenance without importing Torch or running a forecast."""
        df,_=self.dataset(key)
        fingerprint=fingerprint or dataset_fingerprint(df)
        signature=self.historical_signature()
        cache_key=('metadata',fingerprint,signature)
        with self.lock:
            cached=self.historical_cache.get(cache_key)
            if cached is not None:
                self.historical_cache.move_to_end(cache_key)
                return cached
        status=dict(available=False,metadata={},reason='Historical validation has not been trained for this dataset. Run scripts/train_backtest.py to create a separate pre-2022 evaluation model.')
        directory=self.root/'models'/'historical_validation'
        if all(item is not None for item in signature):
            try:
                metadata=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
                if metadata.get('fingerprint')!=fingerprint:
                    status['reason']='The saved historical validation belongs to different observations. Results are not reused for this uploaded dataset.'
                else:
                    train_end=pd.Timestamp(metadata['train_end']).normalize()
                    validation_end=pd.Timestamp(metadata['validation_end']).normalize()
                    test_start=pd.Timestamp(metadata['test_start']).normalize()
                    test_end=pd.Timestamp(metadata['test_end']).normalize()
                    latest=pd.to_datetime(df.date).max().normalize()
                    if not (train_end<=pd.Timestamp('2021-06-30') and train_end<validation_end<=pd.Timestamp('2021-12-31') and validation_end<test_start<=test_end<=latest and test_start>=pd.Timestamp('2022-01-01')):
                        raise ValueError('Training, selection and test boundaries must keep all 2022-2023 targets out of fitting and selection.')
                    observed_dates=pd.DatetimeIndex(sorted(pd.to_datetime(df.date).dt.normalize().unique()))
                    selection_dates=observed_dates[(observed_dates>train_end)&(observed_dates<=validation_end)]
                    if not len(selection_dates):raise ValueError('Model selection has no observed validation dates.')
                    selection_lock=selection_dates[-1]
                    declared_lock=metadata.get('selection_lock_date')
                    if declared_lock and pd.Timestamp(declared_lock).normalize()!=selection_lock:
                        raise ValueError('The selection lock must equal the last observed validation date.')
                    metadata=dict(metadata,selection_lock_date=selection_lock.date().isoformat())
                    expected_hash=metadata.get('checkpoint_sha256')
                    if expected_hash and hashlib.sha256((directory/'det_historical.pt').read_bytes()).hexdigest()!=expected_hash:
                        raise ValueError('The historical checkpoint audit hash does not match.')
                    status=dict(available=True,metadata=metadata,reason=None)
            except (OSError,ValueError,TypeError,KeyError,OverflowError) as exc:
                status['reason']='Saved historical validation provenance is invalid: '+str(exc)
        with self.lock:
            self.historical_cache[cache_key]=status
            while len(self.historical_cache)>10:
                self.historical_cache.popitem(last=False)
        return clean(status)

    @staticmethod
    def historical_metrics(rows,by_zone=False):
        """Scores are calculated from exact saved pairs, never smoothed graph lines."""
        frame=pd.DataFrame(rows)
        if frame.empty:return []
        keys=(['zone_id'] if by_zone else [])+['parameter','horizon_days']
        records=[]
        for values,group in frame.groupby(keys,dropna=False,sort=True):
            actual=pd.to_numeric(group.actual,errors='coerce').to_numpy(float)
            predicted=pd.to_numeric(group.prediction,errors='coerce').to_numpy(float)
            valid=np.isfinite(actual)&np.isfinite(predicted)
            a,b=actual[valid],predicted[valid]
            error=b-a
            denominator=float(((a-a.mean())**2).sum()) if len(a) else 0
            values=values if isinstance(values,tuple) else (values,)
            records.append(dict(zip(keys,values),mae=float(np.abs(error).mean()) if len(a) else None,
                rmse=float(np.sqrt((error**2).mean())) if len(a) else None,
                r2=float(1-(error**2).sum()/denominator) if denominator>0 else None,n=int(len(a))))
        return clean(records)

    def backtest(self,key='modis',zone=None,horizon_days=None,parameter=None,model='det'):
        """Serve frozen pre-2022 DET predictions; production weights are never used."""

        if model != 'det':
            return self.comparison_backtest(key, zone, horizon_days, parameter, model)
        if horizon_days is not None and (isinstance(horizon_days,bool) or not isinstance(horizon_days,int) or horizon_days not in (8,16,32)):
            raise APIError('Historical validation horizon_days must be 8, 16 or 32 (approximate native observation horizons).')
        if parameter not in (None,'',*PARAMETERS):raise APIError('Unknown water parameter.')
        df,_=self.dataset(key)
        if zone is not None and zone not in ('all',*df.zone_id.astype(str).unique()):raise APIError('Unknown zone.')
        fingerprint=dataset_fingerprint(df)
        status=self.historical_status(key,fingerprint)
        empty=dict(status,rows=[],metrics=[],metrics_by_zone=[],available_zones=[],default_zone='all')
        if not status['available']:return empty
        signature=self.historical_signature()
        cache_key=('results',fingerprint,signature)
        with self.lock:
            stored=self.historical_cache.get(cache_key)
        if stored is None:
            try:
                frame=pd.read_csv(self.root/'models'/'historical_validation'/'heldout_predictions.csv',dtype={'zone_id':str})
                required={'origin_date','target_date','horizon_days','actual_elapsed_days','zone_id','parameter','actual','prediction'}
                if not required.issubset(frame.columns) or frame.empty:raise ValueError('Prediction pairs are missing required fields.')
                origins=pd.to_datetime(frame.origin_date,errors='raise').dt.normalize()
                targets=pd.to_datetime(frame.target_date,errors='raise').dt.normalize()
                elapsed=pd.to_numeric(frame.actual_elapsed_days,errors='raise')
                observed_dates=pd.DatetimeIndex(sorted(pd.to_datetime(df.date).dt.normalize().unique()))
                valid_dates=origins.lt(targets)&targets.between(pd.Timestamp(status['metadata']['test_start']),pd.Timestamp(status['metadata']['test_end']))
                if not valid_dates.all() or not elapsed.eq((targets-origins).dt.days).all():raise ValueError('A saved prediction uses an invalid forecast origin or target date.')
                if not origins.ge(pd.Timestamp(status['metadata']['selection_lock_date'])).all():raise ValueError('A prediction origin precedes the date when validation-based model selection was available.')
                origin_positions=observed_dates.get_indexer(origins)
                target_positions=observed_dates.get_indexer(targets)
                if (origin_positions<0).any() or (target_positions<0).any():raise ValueError('Prediction dates are not dates observed in this uploaded dataset.')
                if not frame.horizon_days.isin([8,16,32]).all() or not frame.parameter.isin(PARAMETERS).all():raise ValueError('Invalid saved horizon or parameter.')
                native_steps=frame.horizon_days.map({8:1,16:2,32:4}).to_numpy()
                if not np.equal(target_positions-origin_positions,native_steps).all():raise ValueError('A saved prediction does not match its native observation-step horizon.')
                if 'horizon_steps' in frame and not pd.to_numeric(frame.horizon_steps,errors='raise').eq(native_steps).all():raise ValueError('Saved horizon step metadata does not match the prediction dates.')
                frame['horizon_days']=frame.horizon_days.astype(int)
                frame['actual_elapsed_days']=elapsed.astype(int)
                if not frame.zone_id.isin(['all',*df.zone_id.astype(str).unique()]).all():raise ValueError('Invalid saved zone.')
                if frame.duplicated(['target_date','horizon_days','zone_id','parameter']).any():raise ValueError('Duplicate historical prediction pairs.')
                for column in ('actual','prediction'):
                    frame[column]=pd.to_numeric(frame[column],errors='raise')
                    if np.isinf(frame[column].to_numpy(float)).any():raise ValueError('Infinite saved readings are invalid.')
                index_rows=frame[frame.parameter.eq('ndwi') & frame.actual.notna()]
                if not index_rows.actual.between(-1,1).all():raise ValueError('Invalid NDWI targets cannot be scored as observations.')
                frame['origin_date']=origins.dt.strftime('%Y-%m-%d')
                frame['target_date']=targets.dt.strftime('%Y-%m-%d')
                frame['error']=frame.prediction-frame.actual
                frame['absolute_error']=frame.error.abs()
                # The dedicated experiment does not claim calibrated intervals.
                frame['lower']=None;frame['upper']=None
                frame=frame.sort_values(['target_date','horizon_days','zone_id','parameter'])
                rows=clean(frame)
                zones=sorted(frame.zone_id.unique(),key=lambda value:(value!='all',value))
                pooled=[row for row in rows if row['zone_id']!='all']
                stored=dict(rows=rows,metrics=self.historical_metrics(pooled),metrics_by_zone=self.historical_metrics(rows,True),
                    available_zones=zones,default_zone='all' if 'all' in zones else zones[0])
                with self.lock:
                    self.historical_cache[cache_key]=stored
                    while len(self.historical_cache)>10:self.historical_cache.popitem(last=False)
            except (OSError,ValueError,TypeError,KeyError,OverflowError) as exc:
                return dict(empty,available=False,reason='Saved historical prediction pairs are invalid: '+str(exc))
        if zone is not None and zone not in stored['available_zones']:raise APIError('No saved historical predictions for this zone.')
        rows=[row for row in stored['rows'] if (zone is None or row['zone_id']==zone) and (horizon_days is None or row['horizon_days']==horizon_days) and (not parameter or row['parameter']==parameter)]
        metric_rows=[row for row in rows if row['zone_id']!='all'] if zone is None else rows
        return clean(dict(available=True,metadata=status['metadata'],reason=None,rows=rows,
            metrics=self.historical_metrics(metric_rows),metrics_by_zone=self.historical_metrics(rows,True),
            available_zones=stored['available_zones'],default_zone=stored['default_zone'],
            selection=dict(zone=zone,horizon_days=horizon_days,parameter=parameter or None),
            metric_scope='Selected zone' if zone else 'Pooled observed zone-target pairs; whole-lake rows excluded from pooling'))

    def comparison_backtest(self, key, zone, horizon_days, parameter, model):
        reference = self.backtest(key, zone, horizon_days, parameter)
        benchmark = self.model_benchmark(key)
        if not benchmark.get('available') or not benchmark.get('saved_history'):
            return dict(reference, available=False, rows=[], metrics=[], metrics_by_zone=[],
                        reason='No saved comparison traces for this dataset.')
        selected = benchmark['selected_model'] if model == 'auto' else model
        candidates = {item['id']: item['name'] for item in benchmark['models']}
        if selected not in candidates:
            raise APIError('Unknown comparison model.')
        if not reference['available']:
            return reference
        keys = ['zone_id', 'parameter', 'origin_date', 'target_date', 'horizon_days']
        signature = ('comparison_rows', self.benchmark_signature())
        with self.lock:
            pairs = self.historical_cache.get(signature)
        try:
            if pairs is None:
                pairs = pd.read_csv(self.root / 'models' / 'model_selection' / 'predictions.csv', dtype={'zone_id': str})
                if pairs.duplicated(['model', *keys]).any():
                    raise ValueError('Duplicate forecast pairs.')
                with self.lock:
                    self.historical_cache[signature] = pairs
                    while len(self.historical_cache) > 10:
                        self.historical_cache.popitem(last=False)
            if not reference['rows']:
                return dict(reference, model=selected, model_name=candidates[selected])
            expected = pd.DataFrame(reference['rows']).set_index(keys)
            indexed = pairs.loc[pairs.model.eq(selected)].set_index(keys)
            aligned = indexed.reindex(expected.index)
            if not np.allclose(aligned['actual'], expected['actual'], rtol=1e-10, atol=1e-12) or not np.isfinite(aligned['prediction']).all():
                raise ValueError('Comparison readings do not match the recorded DET benchmark targets.')
            expected['det_prediction'] = expected['prediction']
            expected['prediction'] = aligned['prediction']
            expected['model'] = selected
            expected['model_name'] = candidates[selected]
            expected['error'] = expected['prediction'] - expected['actual']
            expected['absolute_error'] = expected['error'].abs()
            rows = clean(expected.reset_index())
            metric_rows = rows if zone else [row for row in rows if row['zone_id'] != 'all']
            metadata = dict(reference['metadata'], model_family=candidates[selected],
                            selected_model=selected, selected_model_name=candidates[selected],
                            comparison_method=benchmark.get('method'), retrospective=True)
            return dict(reference, model=selected, model_name=candidates[selected], metadata=metadata,
                        rows=rows, metrics=self.historical_metrics(metric_rows),
                        metrics_by_zone=self.historical_metrics(rows, True))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(reference, available=False, rows=[], metrics=[], metrics_by_zone=[],
                        reason='The model comparison could not be aligned with recorded targets: ' + str(exc))

    def assessment(self,key='modis',zone='all',evidence_id=None,as_of=None):
        df,_ = self.dataset(key)
        try:
            if as_of and pd.Timestamp(as_of) > pd.to_datetime(df.date).max():
                raise ValueError('Assessment date cannot be after the latest recorded observation.')
            return clean(assess_water(df, knowledge=self.knowledge(evidence_id), zone_id=zone, as_of=as_of))
        except (ValueError,TypeError) as exc:
            raise APIError(str(exc)) from None

    def monitoring(self, key='modis', zone='all', as_of=None, model=None, evidence_id=None):
        df, _ = self.dataset(key)
        try:
            observation = monitor_satellite(df, zone=zone, as_of=as_of,
                                            supplemental=self.supplemental if key in ('modis', 'demo') else None,
                                            knowledge=self.knowledge(evidence_id))
        except (ValueError, TypeError) as exc:
            raise APIError(str(exc)) from None
        result = dict(observation=observation)
        if not model:
            return clean(result)
        if not observation['date'] or pd.Timestamp(observation['date']) != pd.to_datetime(df.date).max().normalize():
            return clean(dict(result, forecast_error='Condition forecasts begin at the latest recorded source date.'))
        try:
            forecast = self.forecast(key, model, 32)
        except APIError as exc:
            if exc.status != 409:
                raise
            return clean(dict(result, forecast_error=str(exc)))
        result['projection'] = self.monitoring_projection(key, df, observation, forecast)
        return clean(result)

    def monitoring_projection(self, key, df, observation, forecast):
        effective = forecast['metadata'].get('selected_model', forecast['model'])
        rows = forecast['rows']
        if observation['zone_id'] == 'all' and not any(row['zone_id'] == 'all' for row in rows):
            rows = rows + self.lake_forecasts(df, rows)
        benchmark = self.model_benchmark(key)
        return project_satellite(observation, rows, metrics=benchmark.get('metrics', []), model=effective)

    def bootstrap(self,key='modis',evidence_id=None):
        df,report = self.dataset(key)
        knowledge = self.knowledge(evidence_id)
        checkpoint,_=self.checkpoint(df)
        fingerprint=dataset_fingerprint(df)
        signature=(key,fingerprint,checkpoint.stat().st_mtime_ns if checkpoint else None,evidence_id,self.historical_signature(),self.benchmark_signature())
        with self.lock:
            cached=self.bootstrap_cache.get(signature)
            if cached is not None:
                self.bootstrap_cache.move_to_end(signature)
                return cached
        evaluation = self.evaluation(key)
        zone_ids = ['all'] + sorted(df.zone_id.astype(str).unique())
        water_assessments = [assess_water(df, knowledge=knowledge, zone_id=zone) for zone in zone_ids]
        supplemental = self.supplemental if key in ('modis', 'demo') else None
        satellite_monitoring = {
            zone: monitor_satellite(df, zone=zone, knowledge=knowledge, supplemental=supplemental)
            for zone in zone_ids
        }
        columns=[c for c in ['date','lake_id','zone_id','zone_name','cell_id','lat','lon',*PARAMETERS,*[p+'_raw' for p in PARAMETERS],*[p+'_is_observed' for p in PARAMETERS],'ndwi_domain_invalid','valid_pixel_fraction','is_observed','source_kind','sensor','rainfall_mm','temp_c'] if c in df]
        benchmark = self.model_benchmark(key, fingerprint)
        models = self.models(df)
        needs_det = benchmark.get('selected_model') in ('det', 'consensus')
        has_prophet = any(row['id'] == 'prophet' for row in benchmark.get('models', []))
        prophet_ready = has_prophet and prophet_available()
        if has_prophet:
            models.append(dict(id='prophet', name='Prophet', available=prophet_ready,
                               reason='' if prophet_ready else 'Install the project requirements to run Prophet.'))
        auto_ready = (benchmark.get('available') and (not needs_det or models[0]['available'])
                      and (benchmark.get('selected_model') != 'prophet' or prophet_ready))
        models.append(dict(id='auto', name=benchmark.get('selected_model_name') or 'Auto model selection',
                           available=bool(auto_ready),
                           reason='' if auto_ready else benchmark.get('reason') or 'No compatible selected forecaster.'))
        result=clean(dict(dataset='modis' if key=='demo' else key,report=report,observations=df[columns],zones=zones_summary(df),
             alerts=detect_alerts(df),condition=current_condition(df),models=models,model_benchmark=benchmark,evaluation=evaluation['rows'],
             evaluation_meta=evaluation['metadata'],evaluation_by_zone=evaluation.get('by_zone',[]),knowledge=knowledge,evidence_id=evidence_id,
             retrieval_metrics=evaluate_retrieval(knowledge),alert_policy=ALERT_POLICY,water_assessments=water_assessments,
             satellite_monitoring=satellite_monitoring,
             historical_validation=self.historical_status(key,fingerprint)))
        with self.lock:
            self.bootstrap_cache[signature]=result
            while len(self.bootstrap_cache)>10:
                self.bootstrap_cache.popitem(last=False)
        return result

    def upload(self, body):
        raw = body.get('csv')
        if not isinstance(raw,str) or not raw.strip():
            raise APIError('Provide CSV text in the csv field.')
        if len(raw.encode('utf-8')) > MAX_BODY:
            raise APIError('CSV is limited to 12 MB.',413)
        frame,report = load_data(raw.encode('utf-8'))
        if len(frame)>100000:
            raise APIError('Prototype uploads are limited to 100,000 observations.')
        report['filename'] = Path(str(body.get('filename','upload.csv'))).name[:200]
        key = str(uuid.uuid4())
        with self.lock:
            self.datasets[key] = (frame,report)
            while len(self.datasets)>5:
                self.datasets.popitem(last=False)
        return self.bootstrap(key)

    def add_knowledge(self,body):
        records = parse_knowledge(json.dumps(body.get('records'),allow_nan=False).encode('utf-8'))
        key = str(uuid.uuid4())
        with self.lock:
            self.collections[key] = records
            while len(self.collections)>5:
                self.collections.popitem(last=False)
        return dict(evidence_id=key,records=records,knowledge=records,count=len(records))

    def forecast(self,key='modis',model='det',horizon=32,*,bundled=False):
        """Serve a selected forecaster or an explicit DET reference run."""
        df,_ = self.dataset(key)
        if isinstance(horizon,bool) or not isinstance(horizon,int) or not 1 <= horizon <= 730:
            raise APIError('Forecast horizon must be an integer from 1 to 730 days.')
        if model != 'det':
            return self.selected_forecast(key, df, horizon, model)
        checkpoint,summary = self.checkpoint(df,bundled=bundled)
        if not checkpoint or not self.has_torch():
            raise APIError('Train DET on this dataset before forecasting; no compatible checkpoint is available.',409)
        signature=(str(checkpoint),checkpoint.stat().st_mtime_ns,horizon)
        with self.lock:
            cached=self.cached.get(('forecast',signature))
        if cached is not None:
            return cached
        try:
            rows = forecast_zones(df,horizon_days=horizon,model='det',checkpoint_path=checkpoint)
        except (ValueError,RuntimeError,OSError,KeyError) as exc:
            raise APIError('The DET checkpoint is incompatible with this dataset. Train DET again. '+str(exc),409) from exc
        metadata=dict(training=summary,max_validated_horizon_days=32,
            latest_observation=pd.to_datetime(df.date).max().isoformat(),
            uncertainty='Empirical 80% validation residual intervals transferred from the evaluation model to the all-history production refit. Coverage is not guaranteed; inspect held-out interval coverage. Daily values interpolate between anchors. Long horizons are experimental.',
            model_quality=self.model_quality(summary),
            extrapolated=horizon>32)
        result=clean(dict(rows=rows,model='det',horizon_days=horizon,metadata=metadata))
        with self.lock:
            self.cached[('forecast',signature)]=result
        return result

    def selected_forecast(self, key, df, horizon, model='auto'):
        benchmark = self.model_benchmark(key)
        if not benchmark.get('available'):
            raise APIError(benchmark.get('reason') or 'A forecast cannot be selected for this dataset.', 409)
        selected = benchmark['selected_model'] if model == 'auto' else model
        candidates = {item['id']: item['name'] for item in benchmark['models']}
        if selected not in candidates:
            raise APIError('Unknown forecasting model.')
        name = candidates[selected]
        bundled = benchmark.get('saved_history', False)
        checkpoint,_ = self.checkpoint(df, bundled=bundled) if selected in ('det', 'consensus') else (None, {})
        if selected == 'consensus' and (not checkpoint or not self.has_torch()):
            raise APIError('The selected consensus model needs a compatible DET checkpoint and PyTorch.', 409)
        cache_key = ('selected_forecast', benchmark['fingerprint'], self.benchmark_signature(), model, horizon,
                     checkpoint.stat().st_mtime_ns if checkpoint else None)
        with self.lock:
            cached = self.cached.get(cache_key)
        if cached is not None:
            return cached
        if selected == 'det':
            reference = self.forecast(key, 'det', horizon, bundled=bundled)
            zone_rows = reference['rows']
            metadata = dict(reference['metadata'])
        else:
            from aqualens.model_selection import forecast_selected
            try:
                zone_rows = clean(forecast_selected(df, horizon, benchmark, det_checkpoint_path=checkpoint, model=selected))
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                raise APIError('The selected model could not forecast these observations: ' + str(exc), 409) from exc
            if not zone_rows:
                raise APIError('No usable predictions could be produced from these observations.', 409)
            overall = [row for row in benchmark.get('metrics', [])
                       if row['model'] == selected and row['zone_id'] == 'all' and row.get('r2') is not None]
            weak = sum(row['r2'] < 0 for row in overall)
            daily_basis = ('Prophet predicts each calendar day directly.' if selected == 'prophet'
                           else 'Daily values interpolate observation-step anchors.')
            metadata = dict(training={'fit_partition': 'Forecast model fitted using available historical observations'},
                            max_validated_horizon_days=32, latest_observation=pd.to_datetime(df.date).max().isoformat(),
                            uncertainty=f'Zone bands summarize historical validation errors; refit coverage is unverified. {daily_basis} Beyond 32 days, results are experimental and have no bands.',
                            model_quality=dict(evaluated=bool(overall), metric_count=len(overall), negative_r2_count=weak,
                                               status='Mixed historical skill' if weak else 'Historically evaluated' if overall else 'Not scored for the whole lake',
                                               warning='Some whole-lake retrospective R² scores are below zero.' if weak else None),
                            extrapolated=horizon > 32)
        zone_rows = [dict(row, model=selected, model_name=name) for row in zone_rows if row['zone_id'] != 'all']
        metadata.update(selected_model=selected, selected_model_name=name,
                        selection_method=benchmark.get('method'), validation_end=benchmark.get('validation_end'))
        result = clean(dict(model=model, rows=zone_rows + self.lake_forecasts(df, zone_rows), horizon_days=horizon, metadata=metadata))
        with self.lock:
            self.cached[cache_key] = result
            auto_keys = [item for item in self.cached if item[0] == 'selected_forecast']
            for expired in auto_keys[:-18]:
                self.cached.pop(expired, None)
        return result

    @staticmethod
    def lake_forecasts(df, rows):
        from aqualens.backtest import _source_values
        latest = df.loc[pd.to_datetime(df.date).eq(pd.to_datetime(df.date).max())]
        valid = _source_values(latest)
        weights = {p: latest.loc[valid[p].notna()].groupby('zone_id').size().to_dict() for p in PARAMETERS}
        groups = {}
        for row in rows:
            weight = weights[row['parameter']].get(row['zone_id'], 0)
            value = row['prediction']
            if weight and value is not None and np.isfinite(value):
                groups.setdefault((row['date'], row['parameter']), []).append((row, weight))
        combined = []
        for entries in groups.values():
            mass = sum(weight for _, weight in entries)
            average = sum(row['prediction'] * weight for row, weight in entries) / mass
            combined.append(dict(entries[0][0], zone_id='all', prediction=average, lower=None, upper=None,
                                 origin_observed_count=mass, zones_in_summary=len(entries),
                                 interval_basis='Whole-lake forecasts use the latest parameter-valid source observation counts. No calibrated lake interval.'))
        return combined

    def predict_date(self,key,target,model='det'):
        df,_=self.dataset(key)
        try:
            if not isinstance(target,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',target):
                raise ValueError('Invalid date')
            when=pd.Timestamp(target)
            if when.tzinfo is not None: when=when.tz_localize(None)
            when=when.normalize()
        except (ValueError,TypeError):
            raise APIError('date must use YYYY-MM-DD.') from None
        horizon=int((when-pd.to_datetime(df.date).max().normalize()).days)
        result=self.forecast(key,model,horizon)
        rows=[row for row in result['rows'] if pd.Timestamp(row['date']).normalize()==when]
        return dict(date=when.date().isoformat(),rows=rows,model=model,horizon_days=horizon,metadata=result['metadata'])

    def explain(self,key,zone,parameter,evidence_id=None,horizon_days=32,forecast_date=None,model='det'):
        if parameter not in PARAMETERS:
            raise APIError('Unknown water parameter.')
        df,_ = self.dataset(key)
        explanation_df=df
        if zone == 'all':
            explanation_df=df.copy()
            explanation_df['zone_id']='all'
            explanation_df['zone_name']='Whole lake'
        alerts = detect_alerts(explanation_df)
        selection = alerts[(alerts.zone_id==zone)&(alerts.parameter==parameter)]
        if selection.empty:
            raise APIError('Unknown zone.',404)
        forecast_summary=dict(available=False,reason='Forecast unavailable for the selected model or location.')
        checkpoint,_=self.checkpoint(df)
        try:
            forecasts = None
            if model != 'det' or (checkpoint and self.has_torch()):
                forecasts=self.predict_date(key,forecast_date,model) if forecast_date else self.forecast(key,model,horizon_days)
        except APIError as exc:
            if exc.status != 409:
                raise
            forecast_summary['reason'] = str(exc)
        if forecasts:
            if model != 'det':
                selected = [row for row in forecasts['rows'] if row['zone_id'] == zone and row['parameter'] == parameter]
                if selected:
                    forecast_summary = dict(selected[-1], available=True, current_value=selection.iloc[0]['value'],
                                            max_validated_horizon_days=forecasts['metadata']['max_validated_horizon_days'])
                return clean(explain_alert(selection.iloc[0].to_dict(), explanation_df, self.knowledge(evidence_id), forecast_summary=forecast_summary))
            selected=[row for row in forecasts['rows'] if (zone=='all' or row['zone_id']==zone) and row['parameter']==parameter]
            if selected and zone=='all':
                last_date=max(row['date'] for row in selected)
                selected=[row for row in selected if row['date']==last_date]
                latest=df[pd.to_datetime(df.date).eq(pd.to_datetime(df.date).max())]
                valid=latest[parameter].notna()
                if parameter+'_is_observed' in latest:
                    valid &= latest[parameter+'_is_observed'].fillna(False)
                if 'valid_pixel_fraction' in latest:
                    quality=pd.to_numeric(latest.valid_pixel_fraction,errors='coerce')
                    valid &= quality.isna() | quality.ge(.5)
                weights=latest[valid].groupby('zone_id').cell_id.nunique().to_dict()
                mass=sum(weights.get(row['zone_id'],0) for row in selected)
                if mass:
                    average=sum(float(row['prediction'])*weights.get(row['zone_id'],0) for row in selected)/mass
                    selected=[dict(selected[0],zone_id='all',prediction=average,lower=None,upper=None,
                        interval_basis='Lake summary weights DET zone predictions by latest parameter-valid observed cell counts; aggregate intervals are not calibrated.')]
            if selected:
                forecast_summary=dict(selected[-1],available=True,model='det',
                    current_value=selection.iloc[0]['value'],
                    max_validated_horizon_days=32)
        return clean(explain_alert(selection.iloc[0].to_dict(),explanation_df,self.knowledge(evidence_id),forecast_summary=forecast_summary))

    def train(self,body):
        key = body.get('dataset','modis')
        df,_ = self.dataset(key)
        epochs = body.get('epochs',200)
        if isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=300:
            raise APIError('epochs must be an integer from 1 to 300.')
        if not self.has_torch():
            raise APIError('Install requirements-training.txt to enable training.')
        from aqualens.forecast import split_dates
        split_dates(df)
        if pd.to_datetime(df.date).nunique()<35:
            raise APIError('Refined DET requires at least 35 distinct observation dates; 80 or more are recommended.')
        if not self.training_lock.acquire(blocking=False):
            raise APIError('A training job is already running. Wait for it to finish.',409)
        job_id = str(uuid.uuid4())
        job = dict(id=job_id,dataset=key,status='queued',progress=0,
                   message='Preparing chronological training and validation windows.',summary=None,error=None,logs=[])
        with self.lock:
            self.jobs[job_id] = job
        snapshot = df.copy(deep=True)
        def run():
            try:
                self.update_job(job_id,status='running',progress=1,message='Training DET with chronological validation and early stopping.')
                fingerprint = dataset_fingerprint(snapshot)
                output = self.root/'models'/'custom'/fingerprint
                def progress_callback(update):
                    with self.lock:
                        active=self.jobs[job_id]
                        active['logs'].append(clean(dict(update)))
                        active['logs']=active['logs'][-600:]
                        progress=float(update.get('progress',active['progress']))
                        if 0 <= progress <= 1: progress *= 100
                        active['progress']=min(99,progress)
                        active['message']=update.get('message',f"Epoch {update.get('epoch', '?')} / {epochs}")
                        for name in ('epoch','epochs','train_loss','val_loss'):
                            if name in update: active[name]=clean(update[name])
                summary = train_models(snapshot,output,epochs=epochs,progress_callback=progress_callback)
                self.update_job(job_id,status='complete',progress=100,message='Training and held-out evaluation complete.',summary=summary)
            except Exception as exc:
                self.update_job(job_id,status='failed',message='Training failed.',error=str(exc))
            finally:
                self.training_lock.release()
        threading.Thread(target=run,name='aqualens-training',daemon=True).start()
        return self.job(job_id)

    def update_job(self,job_id,**values):
        with self.lock:
            self.jobs[job_id].update(values)

    def job(self,job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise APIError('Unknown job.',404)
            return clean(dict(self.jobs[job_id]))

    def get(self,path,query):
        def arg(name,default=None):
            return query.get(name,[default])[0]
        key = arg('dataset','modis')
        if path=='/api/bootstrap': return self.bootstrap(key,arg('evidence_id'))
        if path=='/api/assessment': return self.assessment(key,arg('zone','all'),arg('evidence_id'),arg('as_of'))
        if path=='/api/monitoring': return self.monitoring(key,arg('zone','all'),arg('as_of'),arg('model'),arg('evidence_id'))
        if path=='/api/evaluation': return self.evaluation(key)
        if path=='/api/backtest':
            try: horizon=int(arg('horizon_days')) if arg('horizon_days') is not None else None
            except (ValueError,TypeError):raise APIError('Historical validation horizon_days must be 8, 16 or 32.') from None
            return self.backtest(key,arg('zone'),horizon,arg('parameter'),arg('model','det'))
        if path=='/api/model-benchmark': return self.model_benchmark(key)
        if path=='/api/forecast':
            try: horizon=int(arg('horizon_days')) if arg('horizon_days') is not None else int(arg('horizon','4'))*7
            except (ValueError,TypeError): raise APIError('horizon_days must be an integer.') from None
            return self.forecast(key,arg('model','det'),horizon)
        if path in ('/api/predict-date','/api/predict'): return self.predict_date(key,arg('date'),arg('model','det'))
        if path=='/api/explain':
            try: horizon=int(arg('horizon_days','32'))
            except (ValueError,TypeError): raise APIError('horizon_days must be an integer.') from None
            return self.explain(key,arg('zone','Z1'),arg('parameter','turbidity_proxy'),arg('evidence_id'),horizon,arg('forecast_date'),arg('model','det'))
        if path=='/api/evidence':
            parameter=arg('parameter')
            if parameter not in (None,'',*PARAMETERS): raise APIError('Unknown water parameter.')
            return clean(dict(results=retrieve_evidence(str(arg('q',''))[:2000],parameter=parameter or None,top_k=8,knowledge=self.knowledge(arg('evidence_id')),sensor=arg('sensor'))))
        if path.startswith('/api/jobs/'): return self.job(path.rsplit('/',1)[-1])
        raise APIError('Unknown API endpoint.',404)


class Handler(BaseHTTPRequestHandler):
    server_version = 'AquaLens/1.0'
    @property
    def app(self): return self.server.app

    def log_message(self, fmt, *args):
        # Do not put uploaded records or evidence queries in server logs.
        pass

    def response_headers(self,status,content_type,length,encoding=None):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(length))
        self.send_header('Vary','Accept-Encoding')
        if encoding: self.send_header('Content-Encoding',encoding)
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','strict-origin-when-cross-origin')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https://tile.openstreetmap.org https://server.arcgisonline.com; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def accepts_gzip(self):
        for entry in self.headers.get('Accept-Encoding','').lower().split(','):
            pieces=[p.strip() for p in entry.split(';')]
            if pieces[0]=='gzip' and not any(p in ('q=0','q=0.0','q=0.00','q=0.000') for p in pieces[1:]):
                return True
        return False
    def json(self,value,status=200):
        data=json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
        compressed=self.accepts_gzip() and len(data)>16384
        if compressed: data=gzip.compress(data,compresslevel=4)
        self.response_headers(status,'application/json; charset=utf-8',len(data),'gzip' if compressed else None)
        self.wfile.write(data)

    def valid_host(self):
        host=self.headers.get('Host','')
        port=self.server.server_address[1]
        return host in (f'127.0.0.1:{port}',f'localhost:{port}')

    def do_GET(self):
        if not self.valid_host():
            return self.json({'error':'Unexpected host.'},403)
        parsed=urlsplit(self.path)
        try:
            if parsed.path=='/api/export':
                key=parse_qs(parsed.query).get('dataset',['modis'])[0]
                df,_=self.app.dataset(key)
                data=df.to_csv(index=False,date_format='%Y-%m-%d').encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type','text/csv; charset=utf-8')
                self.send_header('Content-Disposition','attachment; filename="aqualens-observations.csv"')
                self.send_header('Content-Length',str(len(data)))
                self.send_header('X-Content-Type-Options','nosniff')
                self.end_headers();self.wfile.write(data)
            elif parsed.path.startswith('/api/'):
                self.json(self.app.get(parsed.path,parse_qs(parsed.query)))
            else:
                self.static(parsed.path)
        except (APIError,ValueError,KeyError) as exc:
            self.json({'error':str(exc)},getattr(exc,'status',400))
        except Exception:
            self.json({'error':'The operation could not be completed. Check the dataset and installed dependencies.'},500)

    def static(self,path):
        import mimetypes
        web=(self.app.root/'web').resolve()
        decoded=unquote(path)
        target=(web/('index.html' if decoded=='/' else decoded.lstrip('/'))).resolve()
        if not target.is_relative_to(web) or not target.is_file():
            raise APIError('File not found.',404)
        data=target.read_bytes()
        mime=mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        compressed=self.accepts_gzip() and len(data)>16384 and (mime.startswith('text/') or target.suffix in ('.json','.js','.css','.svg'))
        if compressed: data=gzip.compress(data,compresslevel=4)
        self.response_headers(200,mime,len(data),'gzip' if compressed else None)
        self.wfile.write(data)

    def do_POST(self):
        if not self.valid_host(): return self.json({'error':'Unexpected host.'},403)
        port=self.server.server_address[1]
        origin=self.headers.get('Origin')
        if origin is not None and origin not in (f'http://127.0.0.1:{port}',f'http://localhost:{port}'):
            return self.json({'error':'Cross-origin mutations are not permitted.'},403)
        if self.headers.get('Content-Type','').split(';')[0].strip()!='application/json':
            return self.json({'error':'Content-Type must be application/json.'},415)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0: raise APIError('A JSON request body is required.')
            if length>MAX_BODY: raise APIError('Request body is limited to 12 MB.',413)
            body=json.loads(self.rfile.read(length))
            if not isinstance(body,dict): raise APIError('Request body must be an object.')
            path=urlsplit(self.path).path
            if path=='/api/upload': result=self.app.upload(body)
            elif path=='/api/knowledge': result=self.app.add_knowledge(body)
            elif path=='/api/train': result=self.app.train(body)
            else: raise APIError('Unknown API endpoint.',404)
            self.json(result,202 if path=='/api/train' else 200)
        except (ValueError,KeyError,UnicodeDecodeError) as exc:
            self.json({'error':str(exc)},getattr(exc,'status',400))
        except Exception:
            self.json({'error':'The operation could not be completed. Check the dataset and installed dependencies.'},500)


def make_server(port=8765,app=None):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    server.app=app or Dashboard()
    return server

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Run the AquaLens custom dashboard locally.')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    with make_server(args.port) as server:
        print(f'AquaLens dashboard: http://localhost:{args.port}',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass


