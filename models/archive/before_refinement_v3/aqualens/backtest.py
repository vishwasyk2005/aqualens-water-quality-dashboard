"""Frozen pre-2022 DET evaluation on rolling historical forecast origins.

This module writes a separate evaluation checkpoint. Production forecasting
checkpoints are deliberately not read, replaced, or refitted here.
"""
from pathlib import Path
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from .forecast import PARAMETERS, STEPS, HORIZONS, _panel, _scaler, _features, dataset_fingerprint

TRAIN_END='2021-06-30'
VALIDATION_END='2021-12-31'
TEST_START='2022-01-01'
SEED=42

def _safe(value):
    if isinstance(value,dict): return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [_safe(v) for v in value]
    if value is pd.NA or value is pd.NaT: return None
    if isinstance(value,(pd.Timestamp,)): return value.isoformat()
    if isinstance(value,np.generic): return _safe(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def _write_json(path,value):
    Path(path).write_text(json.dumps(_safe(value),indent=2,allow_nan=False),encoding='utf-8')

def prepare_backtest_panel(df,train_end=TRAIN_END):
    """Freeze the input grid and cell list using training-period coordinates.

    Existing zone identifiers are fixed study-area metadata. Source measurements
    outside the training coordinate range map to the closest grid edge and are
    counted in the audit. Future values never determine the grid or scaler.
    """
    d=df.copy()
    d['date']=pd.to_datetime(d.date).dt.normalize()
    training=d[d.date<=pd.Timestamp(train_end)]
    if training.date.nunique()<24: raise ValueError('At least 24 pre-cutoff observation dates are required.')
    zones=sorted(training.zone_id.astype(str).unique())
    unknown_zone=~d.zone_id.astype(str).isin(zones)
    excluded_unknown=int(unknown_zone.sum())
    d=d[~unknown_zone].copy()
    original=d.cell_id.astype(str).nunique()
    training_source_cells=int(training.cell_id.astype(str).nunique())
    bounds={c:[float(training[c].min()),float(training[c].max())] for c in ('lat','lon')}
    outside=((d.lat<bounds['lat'][0])|(d.lat>bounds['lat'][1])|(d.lon<bounds['lon'][0])|(d.lon>bounds['lon'][1]))
    if training_source_cells>96:
        lat=((d.lat-bounds['lat'][0])/max(bounds['lat'][1]-bounds['lat'][0],1e-9)*6).clip(0,5.999999).astype(int)
        lon=((d.lon-bounds['lon'][0])/max(bounds['lon'][1]-bounds['lon'][0],1e-9)*6).clip(0,5.999999).astype(int)
        d['cell_id']=d.zone_id.astype(str)+'_past_grid_'+lat.astype(str)+'_'+lon.astype(str)
    training_cells=sorted(d[d.date<=pd.Timestamp(train_end)].cell_id.astype(str).unique())
    # Keep future-only cell IDs from changing the hierarchy or triggering the
    # legacy panel helper's own coarsening decision. Their values remain valid
    # zone targets, but they are not assimilated into a new model input cell.
    unknown_input=~d.cell_id.astype(str).isin(training_cells)
    ignored_cells=sorted(d.loc[unknown_input,'cell_id'].astype(str).unique())
    input_frame=d.copy()
    placeholders=d[d.date<=pd.Timestamp(train_end)].groupby('zone_id').cell_id.first().to_dict()
    input_frame.loc[unknown_input,'cell_id']=input_frame.loc[unknown_input,'zone_id'].map(placeholders)
    for parameter in PARAMETERS:
        input_frame.loc[unknown_input,parameter]=np.nan
        if parameter+'_raw' in input_frame: input_frame.loc[unknown_input,parameter+'_raw']=np.nan
        if parameter+'_is_observed' in input_frame: input_frame.loc[unknown_input,parameter+'_is_observed']=False
    if 'is_observed' in input_frame: input_frame.loc[unknown_input,'is_observed']=False
    p=_panel(input_frame)
    observed=_source_values(d).assign(date=d.date,zone_id=d.zone_id)
    targets=observed.groupby(['date','zone_id'])[PARAMETERS].mean()
    p['targets']=targets.reindex(pd.MultiIndex.from_product([p['dates'],p['zones']],names=['date','zone_id'])).to_numpy(float).reshape(len(p['dates']),len(p['zones']),len(PARAMETERS))
    p['audit'].update(training_coordinate_bounds=bounds,geometry_fit_through=train_end,
        source_cells=int(original),original_cells=int(original),training_source_cells=training_source_cells,model_cells=len(p['cells']),source_zone_ids='Fixed supplied/inferred study-area metadata; no water values used to derive zone IDs.',
        training_cell_ids=p['cells'],future_only_input_cells_ignored=ignored_cells,
        outside_training_coordinate_rows=int(outside.sum()),unknown_zone_rows_excluded=excluded_unknown,
        spatial_aggregation='6 by 6 training-coordinate grid intersected with fixed zones' if training_source_cells>96 else 'Original training-period cells only',
        target_definition='Parameter-valid observed source values averaged within each fixed zone. Targets are never filled.',
        source_units='NDWI index and unverified turbidity/chlorophyll proxies; source pixel QA is unknown.')
    return p,d

def build_examples(p,mean,scale,window=8,train_end=TRAIN_END,validation_end=VALIDATION_END,test_start=TEST_START):
    """Separate target masks per horizon, including all eligible final 1-step dates."""
    features=_features(p,mean,scale)
    origins=np.arange(window-1,len(features)-1)
    if not len(origins): raise ValueError('Insufficient dated histories for DET.')
    x=np.stack([features[i-window+1:i+1] for i in origins])
    shape=(len(origins),len(STEPS),len(p['zones']),len(PARAMETERS))
    target=np.full(shape,np.nan,dtype=np.float32)
    partition={key:np.zeros(shape,dtype=bool) for key in ('train','val','test')}
    selection_lock=p['dates'][p['dates']<=pd.Timestamp(validation_end)][-1]
    training_lock=p['dates'][p['dates']<=pd.Timestamp(train_end)][-1]
    for n,origin in enumerate(origins):
        for hi,step in enumerate(STEPS):
            ti=origin+step
            if ti>=len(p['dates']): continue
            when=p['dates'][ti]
            target[n,hi]=(p['targets'][ti]-mean)/scale
            key='train' if when<=pd.Timestamp(train_end) else 'val' if when<=pd.Timestamp(validation_end) else 'test' if when>=pd.Timestamp(test_start) else None
            if key=='val' and p['dates'][origin]<training_lock: key=None
            if key=='test' and p['dates'][origin]<selection_lock: key=None
            if key: partition[key][n,hi]=np.isfinite(target[n,hi])
    return x,target,origins,partition

def rows_metrics(rows):
    if not len(rows): return []
    frame=pd.DataFrame(rows)
    records=[]
    for (horizon,parameter),group in frame.groupby(['horizon_days','parameter'],sort=True):
        actual=group.actual.to_numpy(float); predicted=group.prediction.to_numpy(float)
        error=predicted-actual; denominator=float(((actual-actual.mean())**2).sum())
        records.append(dict(model='det',parameter=parameter,horizon_days=int(horizon),mae=float(np.abs(error).mean()),
            rmse=float(np.sqrt((error**2).mean())),r2=float(1-(error**2).sum()/denominator) if denominator>0 else None,
            n=len(group),target_dates=int(group.target_date.nunique())))
    return records

def _source_values(d):
    values=pd.DataFrame(index=d.index)
    for parameter in PARAMETERS:
        raw=pd.to_numeric(d.get(parameter+'_raw',d[parameter]),errors='coerce')
        observed=d.get(parameter+'_is_observed',d.get('is_observed',pd.Series(True,index=d.index))).astype(str).str.lower().isin(['true','1','1.0','yes'])
        valid=observed & raw.notna() & np.isfinite(raw)
        if parameter+'_was_missing' in d: valid &= ~d[parameter+'_was_missing'].astype(bool)
        if parameter=='ndwi': valid &= raw.between(-1,1)
        if 'valid_pixel_fraction' in d:
            quality=pd.to_numeric(d.valid_pixel_fraction,errors='coerce');valid &= quality.isna()|quality.ge(.5)
        values[parameter]=raw.where(valid)
    return values

def _observed_zone_counts(d,p):
    values=_source_values(d)
    counts=[]
    for parameter in PARAMETERS:
        indexed=d[values[parameter].notna()].groupby(['date','zone_id']).cell_id.count().reindex(pd.MultiIndex.from_product([p['dates'],p['zones']],names=['date','zone_id']),fill_value=0)
        counts.append(indexed.to_numpy(float).reshape(len(p['dates']),len(p['zones'])))
    return np.stack(counts,-1)

def make_test_rows(p,predictions,origins,mask,counts):
    """Only scored native targets are returned; no daily interpolation."""
    rows=[]
    for n,origin in enumerate(origins):
        for hi,step in enumerate(STEPS):
            target_index=origin+step
            if target_index>=len(p['dates']): continue
            base=dict(origin_date=str(p['dates'][origin].date()),target_date=str(p['dates'][target_index].date()),
                horizon_days=HORIZONS[hi],horizon_steps=step,
                actual_elapsed_days=int((p['dates'][target_index]-p['dates'][origin]).days),model='det',lower=None,upper=None)
            for ki,parameter in enumerate(PARAMETERS):
                eligible=mask[n,hi,:,ki]&np.isfinite(predictions[n,hi,:,ki])
                for zi,zone in enumerate(p['zones']):
                    if not eligible[zi]: continue
                    actual=float(p['targets'][target_index,zi,ki]); predicted=float(predictions[n,hi,zi,ki]); error=predicted-actual
                    rows.append(dict(base,zone_id=zone,parameter=parameter,actual=actual,prediction=predicted,error=error,
                        absolute_error=abs(error),origin_observed_count=int(counts[origin,zi,ki]),
                        target_observed_count=int(counts[target_index,zi,ki]),zones_in_summary=1))
                # Both aggregate lines use exactly the same past-origin weights.
                # Renormalize over common scored zones; target-date counts never set weights.
                weights=np.where(eligible,counts[origin,:,ki],0.)
                if weights.sum()>0:
                    weights=weights/weights.sum()
                    actual=float(np.sum(p['targets'][target_index,eligible,ki]*weights[eligible]))
                    predicted=float(np.sum(predictions[n,hi,eligible,ki]*weights[eligible])); error=predicted-actual
                    rows.append(dict(base,zone_id='all',parameter=parameter,actual=actual,prediction=predicted,error=error,
                        absolute_error=abs(error),origin_observed_count=int(counts[origin,eligible,ki].sum()),
                        target_observed_count=int(counts[target_index,eligible,ki].sum()),zones_in_summary=int((weights>0).sum())))
    return rows

def train_historical_backtest(df,output_dir,epochs=160,progress_callback=None):
    import torch
    from .tree_model import HierarchicalForecaster
    if not 1<=epochs<=300: raise ValueError('Maximum epochs must be 1..300.')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    p,mapped=prepare_backtest_panel(df)
    if p['dates'][-1]<pd.Timestamp(TEST_START): raise ValueError('Dataset requires observed targets in 2022 or later.')
    mean,scale=_scaler(p,pd.Timestamp(TRAIN_END))
    fingerprint=dataset_fingerprint(df)
    configs=[dict(window=12,hidden=24,lr=.003),dict(window=8,hidden=32,lr=.001)]
    histories={};tuning=[];selected=None;selected_loss=float('inf')
    def emit(data):
        _write_json(out/'training_progress.json',data)
        if progress_callback:progress_callback(data)
    def prepared(cfg):
        x,target,origins,partitions=build_examples(p,mean,scale,cfg['window'])
        if any(not partitions[key].any() for key in partitions): raise ValueError('Training, validation, and test each need observed targets.')
        return torch.from_numpy(x),torch.from_numpy(np.nan_to_num(target)),{key:torch.from_numpy(value) for key,value in partitions.items()},origins,partitions
    def objective(pred,target,mask):return torch.nn.functional.smooth_l1_loss(pred[mask],target[mask])
    for ci,cfg in enumerate(configs,1):
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        x,target,masks,origins,_=prepared(cfg)
        train_indices=np.flatnonzero(masks['train'].any(dim=(1,2,3)).numpy())
        val_indices=np.flatnonzero(masks['val'].any(dim=(1,2,3)).numpy())
        net=HierarchicalForecaster(p['cell_zone'],cfg['hidden'])
        optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.001)
        best=float('inf');best_epoch=0;best_state=None;stale=0;history=[]
        for epoch in range(1,epochs+1):
            net.train();losses=[]
            for batch in np.array_split(rng.permutation(train_indices),max(1,math.ceil(len(train_indices)/16))):
                optimizer.zero_grad();loss=objective(net(x[batch]),target[batch],masks['train'][batch]);loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():val=float(objective(net(x[val_indices]),target[val_indices],masks['val'][val_indices]))
            history.append(dict(candidate=ci,epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val))
            if val<best-1e-5:
                best=val;best_epoch=epoch;stale=0;best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            else:stale+=1
            if epoch==1 or epoch%5==0:
                emit(dict(stage='validation_search',candidate=ci,candidates=len(configs),epoch=epoch,epochs=epochs,
                    train_loss=history[-1]['train_loss'],val_loss=val,progress=(ci-1+epoch/epochs)/len(configs)*.9,
                    message=f'Historical DET candidate {ci}, epoch {epoch}; pre-2022 validation {val:.6f}'))
                print(f'Historical DET candidate={ci} epoch={epoch} train={history[-1]["train_loss"]:.6f} validation={val:.6f}',flush=True)
            if stale>=18:break
        histories[str(ci)]=history
        tuning.append(dict(cfg,candidate=ci,best_epoch=best_epoch,epochs_run=len(history),validation_loss=best,train_origins=len(train_indices),validation_origins=len(val_indices)))
        if best<selected_loss:selected_loss=best;selected=(cfg.copy(),best_state,best_epoch,ci)
    cfg,state,best_epoch,candidate=selected
    x,target,masks,origins,raw_masks=prepared(cfg)
    net=HierarchicalForecaster(p['cell_zone'],cfg['hidden']);net.load_state_dict(state);net.eval()
    emit(dict(stage='held_out_evaluation',progress=.95,message='Configuration locked. Scoring only 2022–2023 held-out targets.'))
    # One immutable checkpoint is scored at all rolling origins; no test-period fitting.
    with torch.no_grad():predictions=net(x).numpy()*scale+mean
    counts=_observed_zone_counts(mapped,p)
    rows=make_test_rows(p,predictions,origins,raw_masks['test'],counts)
    zone_rows=[row for row in rows if row['zone_id']!='all']
    metrics=rows_metrics(zone_rows)
    per_zone=[]
    for zone in [*p['zones'],'all']:
        per_zone.extend(dict(row,zone_id=zone) for row in rows_metrics([r for r in rows if r['zone_id']==zone]))
    selected_history=histories[str(candidate)]
    observed_partitions={
        'training':p['dates'][p['dates']<=pd.Timestamp(TRAIN_END)],
        'validation':p['dates'][(p['dates']>pd.Timestamp(TRAIN_END))&(p['dates']<=pd.Timestamp(VALIDATION_END))],
        'test':p['dates'][p['dates']>=pd.Timestamp(TEST_START)]}
    partition_info={name:dict(first_observation=str(values[0].date()),last_observation=str(values[-1].date()),observation_dates=len(values)) for name,values in observed_partitions.items()}
    test_heads={str(horizon):len({row['target_date'] for row in rows if row['horizon_days']==horizon}) for horizon in HORIZONS}
    summary=dict(version=1,experiment='historical_validation_pre2022',model='det',fingerprint=fingerprint,
        train_start=str(p['dates'][0].date()),train_end=TRAIN_END,validation_start='2021-07-01',validation_end=VALIDATION_END,
        test_start=TEST_START,test_end=str(p['dates'][-1].date()),partitions=partition_info,
        selection_uses_targets_through=VALIDATION_END,weights_fit_through=TRAIN_END,scaler_fit_through=TRAIN_END,
        horizon_steps=STEPS,horizons_days=HORIZONS,forecast_horizons_days=HORIZONS,
        config=cfg,best_epoch=best_epoch,selected_candidate=candidate,seed=SEED,max_epochs=epochs,patience=18,
        history={'det':selected_history},candidate_histories=histories,tuning=tuning,
        train_windows=int(raw_masks['train'].any(axis=(1,2,3)).sum()),validation_windows=int(raw_masks['val'].any(axis=(1,2,3)).sum()),
        test_windows=int(raw_masks['test'].any(axis=(1,2,3)).sum()),test_dates_by_horizon=test_heads,
        first_test_target_by_horizon={str(h):min(row['target_date'] for row in rows if row['horizon_days']==h) for h in HORIZONS},
        train_observed_targets=int(raw_masks['train'].sum()),validation_observed_targets=int(raw_masks['val'].sum()),test_observed_targets=int(raw_masks['test'].sum()),
        data_audit=p['audit'],mean=mean.tolist(),scale=scale.tolist(),metrics=metrics,metrics_by_zone=per_zone,
        evaluation='Frozen pre-2022 DET weights; training targets through June 2021, model/epoch selection using July–December 2021 only. Rolling forecasts score each native 1/2/4-step target from 2022 onward. Each horizon is scored independently.',
        weights_frozen=True,production_refit=False,checkpoint='det_historical.pt',
        selection_lock_date=partition_info['validation']['last_observation'],model_available_from=partition_info['validation']['last_observation'],
        test_origin_rule='Forecast origin must be on or after the final validation observation used for checkpoint selection; target must be in 2022 or later.',
        lake_aggregation='Use parameter-valid observed source-row counts by zone at each forecast origin as weights for BOTH actual and predicted lake summaries. Renormalize over zones with a valid scored target and positive origin count. This is a common-weight evaluation summary, not the target-date changing-sample raw lake mean.',
        interval_method='Not estimated in this historical experiment; lower/upper bounds are unavailable.',
        errors='error = prediction - actual; absolute_error = abs(error). Metrics.csv pools scored zone pairs only; metrics_by_zone.csv includes individual zones and common-weight whole-lake summaries.',
        limitations=['No weights, scaler, hyperparameters, or early-stopping decision use 2022–2023 target values.',
            'Later rolling origins may use measurements that have already arrived in 2022–2023; this evaluates repeated short-horizon forecasts, not one forecast of the entire year from 2021.',
            'Horizon labels ~8/~16/~32 days refer to 1/2/4 native observations; actual elapsed days vary and are recorded for every point.',
            'Every plotted pair is an observed target date. There is no daily interpolation or curve adjustment to make the lines agree.',
            'Fixed source zone identifiers are geographic metadata. Input grid bounds and modeled cells are frozen from training-period coordinates; future-only input cells are ignored.',
            'Parameter units, extraction algorithms, source-product provenance, and satellite pixel QA remain unverified.',
            'Observed errors may be substantial; matching is measured, never forced. Results concern this historical lake dataset only.'])
    torch.save(dict(version=1,experiment=summary['experiment'],fingerprint=fingerprint,state=state,config=cfg,
        mean=mean.tolist(),scale=scale.tolist(),cells=p['cells'],zones=p['zones'],cell_zone=p['cell_zone'],
        train_end=TRAIN_END,validation_end=VALIDATION_END,best_epoch=best_epoch,weights_frozen=True),out/'det_historical.pt')
    summary['checkpoint_sha256']=hashlib.sha256((out/'det_historical.pt').read_bytes()).hexdigest()
    pd.DataFrame(rows).to_csv(out/'heldout_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(per_zone).to_csv(out/'metrics_by_zone.csv',index=False)
    pd.DataFrame(selected_history).to_csv(out/'training_history.csv',index=False)
    pd.DataFrame([row for history in histories.values() for row in history]).to_csv(out/'candidate_history.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    _write_json(out/'summary.json',summary);_write_json(out/'predictions.json',rows)
    emit(dict(stage='complete',progress=1,message='Frozen historical DET evaluation complete; actual/predicted pairs saved.'))
    return summary


def rescore_historical_checkpoint(df,output_dir):
    """Apply stricter eligibility/audit checks to unchanged selected weights.

    This operation never fits a scaler, selects a configuration, or updates
    weights from the scored test values. It can repair evaluation eligibility.
    """
    import torch
    from .tree_model import HierarchicalForecaster
    torch.set_num_threads(2)
    out=Path(output_dir)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    ck=torch.load(out/'det_historical.pt',map_location='cpu',weights_only=True)
    if ck['fingerprint']!=dataset_fingerprint(df):raise ValueError('Historical checkpoint does not match this source dataset.')
    p,mapped=prepare_backtest_panel(df)
    mean=np.asarray(ck['mean']);scale=np.asarray(ck['scale'])
    check_mean,check_scale=_scaler(p,pd.Timestamp(TRAIN_END))
    np.testing.assert_allclose(mean,check_mean,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(scale,check_scale,rtol=1e-12,atol=1e-12)
    if p['cells']!=ck['cells'] or p['cell_zone']!=ck['cell_zone']:raise ValueError('Frozen input hierarchy changed.')
    x,_,origins,masks=build_examples(p,mean,scale,ck['config']['window'])
    net=HierarchicalForecaster(p['cell_zone'],ck['config']['hidden']);net.load_state_dict(ck['state']);net.eval()
    with torch.no_grad():predictions=net(torch.from_numpy(x)).numpy()*scale+mean
    rows=make_test_rows(p,predictions,origins,masks['test'],_observed_zone_counts(mapped,p))
    metrics=rows_metrics([row for row in rows if row['zone_id']!='all'])
    per_zone=[]
    for zone in [*p['zones'],'all']:per_zone.extend(dict(row,zone_id=zone) for row in rows_metrics([r for r in rows if r['zone_id']==zone]))
    lock=str(p['dates'][p['dates']<=pd.Timestamp(VALIDATION_END)][-1].date())
    summary.update(metrics=metrics,metrics_by_zone=per_zone,data_audit=p['audit'],
        selection_lock_date=lock,model_available_from=lock,
        test_origin_rule='Forecast origin must be on or after the final validation observation used for checkpoint selection; target must be in 2022 or later.',
        test_windows=int(masks['test'].any(axis=(1,2,3)).sum()),test_observed_targets=int(masks['test'].sum()),
        test_dates_by_horizon={str(h):len({r['target_date'] for r in rows if r['horizon_days']==h}) for h in HORIZONS},
        first_test_target_by_horizon={str(h):min(r['target_date'] for r in rows if r['horizon_days']==h) for h in HORIZONS},
        checkpoint_sha256=hashlib.sha256((out/'det_historical.pt').read_bytes()).hexdigest(),
        eligibility_correction='The final validation observation availability boundary is enforced at every forecast origin. Existing validation-selected checkpoint weights, scaler, configuration, epoch count, and validation losses are unchanged; no test-score tuning or retraining was performed.')
    summary['limitations'].append('Longer-horizon test curves start later in January 2022 because each forecast origin must be after validation-based checkpoint selection became available.')
    pd.DataFrame(rows).to_csv(out/'heldout_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(per_zone).to_csv(out/'metrics_by_zone.csv',index=False)
    _write_json(out/'summary.json',summary);_write_json(out/'predictions.json',rows)
    return summary
