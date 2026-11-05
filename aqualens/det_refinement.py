"""Past-only DET refinement; no held-out 2022--2023 scoring in this module.

Six configurations are declared before fitting. Validation uses a frozen model
that is available at each origin; the 2022+ frame is rejected by training.
"""
from pathlib import Path
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from .backtest import prepare_backtest_panel,build_examples,TRAIN_END,VALIDATION_END
from .forecast import PARAMETERS,HORIZONS,STEPS,_features,_scaler,dataset_fingerprint

SEED=42
FORMAT_VERSION=3
CONFIGS=[
    dict(candidate=1,mode='anchored',window=8,hidden=16,dropout=0.,lr=.001,weight_decay=.01,loss='mse'),
    dict(candidate=2,mode='anchored',window=12,hidden=24,dropout=.1,lr=.001,weight_decay=.01,loss='mse'),
    dict(candidate=3,mode='anchored',window=8,hidden=24,dropout=.1,lr=.003,weight_decay=.001,loss='smooth_l1'),
    dict(candidate=4,mode='direct',window=8,hidden=16,dropout=.1,lr=.001,weight_decay=.01,loss='mse'),
    dict(candidate=5,mode='direct',window=12,hidden=24,dropout=.1,lr=.001,weight_decay=.01,loss='mse'),
    dict(candidate=6,mode='direct',window=12,hidden=32,dropout=0.,lr=.001,weight_decay=.001,loss='smooth_l1'),
]
CRITERION='Mean of nine validation RMSE values (3 parameters x 3 native horizons), each divided by its training-only pooled within-zone temporal standard deviation; every task has equal weight.'


def _json_safe(value):
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [_json_safe(v) for v in value]
    if isinstance(value,np.generic):return _json_safe(value.item())
    if isinstance(value,pd.Timestamp):return value.isoformat()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def _write_json(path,value):
    Path(path).write_text(json.dumps(_json_safe(value),indent=2,allow_nan=False),encoding='utf-8')


def ensure_past_only(df,validation_end=VALIDATION_END):
    if df.empty:raise ValueError('Historical refinement requires dated observations.')
    if pd.to_datetime(df.date).max()>pd.Timestamp(validation_end):
        raise ValueError(f'Truncate to observations through {pd.Timestamp(validation_end).date()} BEFORE calling the refinement search; held-out readings are forbidden.')


def causal_zone_anchors(panel,mean,scale,max_gap_days=16):
    """Exact source-zone means, causal short carry-forward, then training mean.

    The anchor is an observed target definition at the ORIGIN, not the mean of
    differently weighted model-grid cells. Missing means are never backfilled.
    """
    values=panel['targets'];dates=panel['dates']
    mean=np.asarray(mean,dtype=float);scale=np.asarray(scale,dtype=float)
    prior=np.broadcast_to(mean,(len(panel['zones']),len(PARAMETERS))).copy()
    last_day=np.full(prior.shape,np.nan)
    anchors=np.empty_like(values,dtype=float)
    for index,date in enumerate(dates):
        day=date.value/86400e9
        valid=np.isfinite(values[index]);prior[valid]=values[index][valid];last_day[valid]=day
        recent=np.isfinite(last_day)&((day-last_day)<=max_gap_days)
        anchors[index]=np.where(recent,prior,np.asarray(mean))
    return ((anchors-np.asarray(mean))/np.asarray(scale)).astype('float32')


def training_target_scales(panel,train_end=TRAIN_END):
    """Pool temporal variation within zones; spatial mean differences don't dominate."""
    values=panel['targets'][panel['dates']<=pd.Timestamp(train_end)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        zone_variance=np.nanvar(values,axis=0)
        pooled=np.sqrt(np.nanmean(zone_variance,axis=0))
        global_scale=np.nanstd(values,axis=(0,1))
    fallback=np.where(np.isfinite(global_scale)&(global_scale>1e-6),global_scale,1.)
    return np.where(np.isfinite(pooled)&(pooled>1e-6),pooled,fallback)


def make_examples(panel,mean,scale,window,train_end=TRAIN_END,validation_end=VALIDATION_END):
    x,target,origins,masks=build_examples(panel,mean,scale,window,train_end,validation_end)
    training_dates=panel['dates'][panel['dates']<=pd.Timestamp(train_end)]
    if not len(training_dates):raise ValueError('No training dates.')
    fit_available=training_dates[-1]
    masks['val'] &= (panel['dates'][origins]>=fit_available)[:,None,None,None]
    anchors=causal_zone_anchors(panel,mean,scale)[origins]
    return x,anchors,target,origins,masks


def metric_rows(actual,prediction,mask,target_scales):
    records=[]
    for hi,horizon in enumerate(HORIZONS):
        for ki,parameter in enumerate(PARAMETERS):
            valid=mask[:,hi,:,ki]&np.isfinite(actual[:,hi,:,ki])&np.isfinite(prediction[:,hi,:,ki])
            a=actual[:,hi,:,ki][valid];b=prediction[:,hi,:,ki][valid]
            if not len(a):raise ValueError('Every parameter/horizon needs observed validation targets.')
            error=b-a;rmse=float(np.sqrt(np.mean(error**2)))
            records.append(dict(parameter=parameter,horizon_days=horizon,n=int(len(a)),mae=float(np.mean(np.abs(error))),
                rmse=rmse,bias=float(np.mean(error)),normalization_scale=float(target_scales[ki]),normalized_rmse=rmse/float(target_scales[ki])))
    return records


def validation_score(records):return float(np.mean([row['normalized_rmse'] for row in records]))


def _network_class():
    # Keep data preparation/imports independent of Torch until a model is needed.
    import torch
    from torch import nn
    from .tree_model import ChildSumTreeLSTM
    class RefinedDET(nn.Module):
        def __init__(self,cell_zone,hidden=24,dropout=0.,mode='anchored',features=12):
            super().__init__()
            if mode not in ('anchored','direct'):raise ValueError('DET mode must be anchored or direct.')
            self.cell_zone=list(cell_zone);self.zone_count=max(cell_zone)+1;self.mode=mode
            self.zone_indices=[[i for i,parent in enumerate(cell_zone) if parent==zone] for zone in range(self.zone_count)]
            self.encoder=nn.GRU(features,hidden,batch_first=True)
            self.encoder_dropout=nn.Dropout(dropout)
            self.leaf_memory=nn.Linear(hidden,hidden)
            self.zone_tree=ChildSumTreeLSTM(hidden);self.lake_tree=ChildSumTreeLSTM(hidden)
            self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.Tanh(),nn.Dropout(dropout),nn.Linear(hidden,9))
            # Anchored starts at the exact causal zone reading. Direct starts at
            # the training feature mean. Both remain a single learned DET.
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        def forward(self,x,anchors=None):
            batch,steps,cells,features=x.shape
            _,encoded=self.encoder(x.permute(0,2,1,3).reshape(batch*cells,steps,features))
            leaves=self.encoder_dropout(encoded[-1].reshape(batch,cells,-1))
            memories=torch.tanh(self.leaf_memory(leaves))
            states=[self.zone_tree(leaves[:,indices],memories[:,indices]) for indices in self.zone_indices]
            zh=torch.stack([value[0] for value in states],1);zc=torch.stack([value[1] for value in states],1)
            lake,_=self.lake_tree(zh,zc)
            joined=torch.cat([zh,lake[:,None].expand(-1,self.zone_count,-1)],-1)
            result=self.head(joined).reshape(batch,self.zone_count,3,3).permute(0,2,1,3)
            if self.mode=='anchored':
                if anchors is None or tuple(anchors.shape)!=(batch,self.zone_count,3):raise ValueError('Exact causal zone anchors are required for anchored DET.')
                result=result+anchors[:,None]
            return result
    return RefinedDET


def build_refined_model(cell_zone,config):
    return _network_class()(cell_zone,hidden=config['hidden'],dropout=config.get('dropout',0.),mode=config['mode'])


def load_refined_checkpoint(path):
    import torch
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('format_version')!=FORMAT_VERSION or checkpoint.get('model_family')!='DET':raise ValueError('Not a supported refined DET checkpoint.')
    return checkpoint


def model_from_checkpoint(checkpoint):
    net=build_refined_model(checkpoint['cell_zone'],checkpoint['config'])
    net.load_state_dict(checkpoint['state']);net.eval()
    return net


def predict_normalized(checkpoint,x,anchors):
    import torch
    net=model_from_checkpoint(checkpoint)
    with torch.no_grad():return net(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(anchors,dtype=torch.float32)).numpy()


def _validation_pairs(panel,origins,actual,prediction,mask):
    rows=[]
    for n,origin in enumerate(origins):
        for hi,step in enumerate(STEPS):
            target_index=origin+step
            if target_index>=len(panel['dates']):continue
            for zi,zone in enumerate(panel['zones']):
                for ki,parameter in enumerate(PARAMETERS):
                    if not mask[n,hi,zi,ki]:continue
                    rows.append(dict(origin_date=str(panel['dates'][origin].date()),target_date=str(panel['dates'][target_index].date()),
                        actual_elapsed_days=int((panel['dates'][target_index]-panel['dates'][origin]).days),zone_id=zone,
                        parameter=parameter,horizon_days=HORIZONS[hi],actual=float(actual[n,hi,zi,ki]),prediction=float(prediction[n,hi,zi,ki])))
    return rows


def _reference_scores(df,panel,target_scales,original_checkpoint,train_end=TRAIN_END,validation_end=VALIDATION_END,frozen_reference=None):
    import torch
    refs={}
    mean,scale=_scaler(panel,pd.Timestamp(train_end))
    x,anchor,target,origins,masks=make_examples(panel,mean,scale,12,train_end,validation_end)
    actual=target*scale+mean
    no_change=np.broadcast_to(anchor[:,None],target.shape)*scale+mean
    metrics=metric_rows(actual,no_change,masks['val'],target_scales)
    refs['last_observed_diagnostic']=dict(score=validation_score(metrics),metrics=metrics,
        description='Diagnostic only: exact latest causal observed zone value, with no learned adjustment. This is not a deployed forecasting model.')
    if original_checkpoint and Path(original_checkpoint).is_file():
        from .tree_model import HierarchicalForecaster
        ck=torch.load(original_checkpoint,map_location='cpu',weights_only=True)
        if ck['cells']!=panel['cells'] or ck['cell_zone']!=panel['cell_zone']:
            if not frozen_reference:raise ValueError('Original historical reference hierarchy differs; a frozen pre-repair validation audit is required.')
            audit=json.loads(Path(frozen_reference).read_text(encoding='utf-8')) if not isinstance(frozen_reference,dict) else frozen_reference
            reference=audit['reference']
            if audit['fit_data_fingerprint']!=dataset_fingerprint(df) or pd.Timestamp(audit['train_end'])!=pd.Timestamp(train_end) or pd.Timestamp(audit['validation_end'])!=pd.Timestamp(validation_end):
                raise ValueError('Frozen reference is not the same past-only source/partitions.')
            if reference['checkpoint_sha256']!=hashlib.sha256(Path(original_checkpoint).read_bytes()).hexdigest():raise ValueError('Frozen reference checkpoint changed.')
            metrics=[]
            for record in reference['metrics']:
                record=dict(record);hi=HORIZONS.index(record['horizon_days']);ki=PARAMETERS.index(record['parameter'])
                if record['n']!=int(masks['val'][:,hi,:,ki].sum()):raise ValueError('Frozen reference target-pair count differs.')
                record['normalization_scale']=float(target_scales[ki]);record['normalized_rmse']=record['rmse']/float(target_scales[ki]);metrics.append(record)
            refs['original_frozen_det']=dict(reference,score=validation_score(metrics),metrics=metrics,
                description='Frozen original DET pre-2022 validation audit computed using its own unsnapped preprocessing. Same source-zone target pairs and corrected origin availability; raw errors retained and normalized with current training-zone temporal scales.',
                frozen_reference_fit_fingerprint=audit['fit_data_fingerprint'])
            return refs
        old_mean=np.asarray(ck['mean']);old_scale=np.asarray(ck['scale'])
        x,anchor,target,origins,masks=make_examples(panel,old_mean,old_scale,ck['config']['window'],train_end,validation_end)
        old=HierarchicalForecaster(ck['cell_zone'],ck['config']['hidden']);old.load_state_dict(ck['state']);old.eval()
        with torch.no_grad():prediction=old(torch.from_numpy(x)).numpy()*old_scale+old_mean
        metrics=metric_rows(target*old_scale+old_mean,prediction,masks['val'],target_scales)
        refs['original_frozen_det']=dict(score=validation_score(metrics),metrics=metrics,
            checkpoint_sha256=hashlib.sha256(Path(original_checkpoint).read_bytes()).hexdigest(),
            description='Existing separate DET checkpoint evaluated only on the same corrected pre-2022 validation-origin mask.')
    return refs


def train_refinement(df,output_dir,epochs=200,patience=25,configs=None,original_checkpoint=None,progress_callback=None,train_end=TRAIN_END,validation_end=VALIDATION_END,frozen_reference=None):
    """Fit/search only 2020--2021 data; held-out 2022+ input is an error."""
    import torch
    ensure_past_only(df,validation_end)
    if not 1<=epochs<=300:raise ValueError('Maximum epochs must be 1..300.')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    configs=[dict(value) for value in (CONFIGS if configs is None else configs)]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    panel,_=prepare_backtest_panel(df,train_end=train_end)
    mean,scale=_scaler(panel,pd.Timestamp(train_end));target_scales=training_target_scales(panel,train_end)
    last_train=panel['dates'][panel['dates']<=pd.Timestamp(train_end)][-1]
    last_validation=panel['dates'][panel['dates']<=pd.Timestamp(validation_end)][-1]
    plan=dict(format_version=FORMAT_VERSION,model_family='DET',fit_data_fingerprint=dataset_fingerprint(df),
        train_end=str(pd.Timestamp(train_end).date()),validation_end=str(pd.Timestamp(validation_end).date()),fit_available_from=str(last_train.date()),selection_lock_date=str(last_validation.date()),
        selection_criterion=CRITERION,seed=SEED,max_epochs=epochs,patience=patience,configs=configs,
        held_out_after_validation_loaded=False,held_out_after_validation_scored=False,
        held_out_2022_loaded=False if pd.Timestamp(validation_end)<pd.Timestamp('2022-01-01') else None,held_out_2022_scored=False if pd.Timestamp(validation_end)<pd.Timestamp('2022-01-01') else None,
        objective='MSE or SmoothL1 over errors divided by training-only within-zone temporal standard deviation, as predeclared per candidate.',
        predictions='Unconstrained raw DET outputs; no clipping, smoothing or post-test curve adjustment.')
    _write_json(out/'search_plan.json',plan)
    def emit(data):
        _write_json(out/'progress.json',data)
        if progress_callback:progress_callback(data)
    frozen_path=out/'frozen_original_reference.json'
    references=_reference_scores(df,panel,target_scales,original_checkpoint,train_end,validation_end,frozen_reference or (frozen_path if frozen_path.exists() else None))
    all_metrics=[];histories=[];tuning=[];selected=None;best_score=float('inf')
    for ci,cfg in enumerate(configs,1):
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        x,anchor,target,origins,masks=make_examples(panel,mean,scale,cfg['window'],train_end,validation_end)
        tx=torch.from_numpy(x);ta=torch.from_numpy(anchor);ty=torch.from_numpy(np.nan_to_num(target))
        tm={key:torch.from_numpy(value) for key,value in masks.items()}
        training=np.flatnonzero(masks['train'].any(axis=(1,2,3)));validation=np.flatnonzero(masks['val'].any(axis=(1,2,3)))
        if not len(training) or not len(validation):raise ValueError('Observed training and validation examples are required.')
        ratio=torch.as_tensor(scale/target_scales,dtype=torch.float32)
        net=build_refined_model(panel['cell_zone'],cfg)
        optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        candidate_best=float('inf');candidate_state=None;best_epoch=0;stale=0;history=[]
        for epoch in range(1,epochs+1):
            net.train();losses=[]
            for batch in np.array_split(rng.permutation(training),max(1,math.ceil(len(training)/16))):
                optimizer.zero_grad();prediction=net(tx[batch],ta[batch]);valid=tm['train'][batch]
                error=(prediction-ty[batch])*ratio
                objective=(error[valid]**2).mean() if cfg['loss']=='mse' else torch.nn.functional.smooth_l1_loss(error[valid],torch.zeros_like(error[valid]))
                objective.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(objective.detach()))
            net.eval()
            with torch.no_grad():raw_prediction=net(tx[validation],ta[validation]).numpy()*scale+mean
            metrics=metric_rows(target[validation]*scale+mean,raw_prediction,masks['val'][validation],target_scales)
            score=validation_score(metrics)
            row=dict(candidate=cfg['candidate'],epoch=epoch,train_loss=float(np.mean(losses)),validation_score=score)
            history.append(row)
            if score<candidate_best-1e-5:
                candidate_best=score;best_epoch=epoch;stale=0;candidate_state={key:value.detach().clone() for key,value in net.state_dict().items()}
            else:stale+=1
            if epoch==1 or epoch%10==0:
                emit(dict(stage='past_only_search',candidate=ci,candidates=len(configs),epoch=epoch,max_epochs=epochs,validation_score=score,best_score=candidate_best))
                print(f'DET refinement candidate={ci}/{len(configs)} epoch={epoch} validation_normalized_rmse={score:.6f}',flush=True)
            if stale>=patience:break
        net.load_state_dict(candidate_state);net.eval()
        with torch.no_grad():predictions=net(tx,ta).numpy()*scale+mean
        metrics=metric_rows(target*scale+mean,predictions,masks['val'],target_scales)
        all_metrics.extend(dict(metric,candidate=cfg['candidate']) for metric in metrics)
        histories.extend(history)
        tuning.append(dict(cfg,best_epoch=best_epoch,epochs_run=len(history),validation_score=candidate_best,
            train_origins=len(training),validation_origins=len(validation)))
        if candidate_best<best_score:
            best_score=candidate_best;selected=dict(config=cfg,state=candidate_state,best_epoch=best_epoch,
                metrics=metrics,rows=_validation_pairs(panel,origins,target*scale+mean,predictions,masks['val']),history=history)
    ck=dict(format_version=FORMAT_VERSION,model_family='DET',architecture='GRU -> Child-Sum Tree-LSTM cells/zones/lake -> regularized multi-horizon heads',
        config=selected['config'],state=selected['state'],mean=mean.tolist(),scale=scale.tolist(),target_scales=target_scales.tolist(),
        cells=panel['cells'],zones=panel['zones'],cell_zone=panel['cell_zone'],features=12,
        fit_data_fingerprint=plan['fit_data_fingerprint'],train_end=str(pd.Timestamp(train_end).date()),validation_end=str(pd.Timestamp(validation_end).date()),
        fit_available_from=str(last_train.date()),selection_lock_date=str(last_validation.date()),best_epoch=selected['best_epoch'],
        anchor_definition='Parameter-valid observed source-zone mean at origin; causal carry-forward up to 16 days, otherwise training mean.',
        raw_output_constraints='none',weights_frozen=True)
    path=out/'det_refined.pt';torch.save(ck,path)
    summary=dict(plan,selected_candidate=selected['config']['candidate'],config=selected['config'],best_epoch=selected['best_epoch'],
        validation_score=best_score,validation_metrics=selected['metrics'],target_scales=target_scales.tolist(),mean=mean.tolist(),scale=scale.tolist(),
        references=references,tuning=tuning,history=selected['history'],data_audit=panel['audit'],
        checkpoint='det_refined.pt',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_locked=True,test_evaluation_authorized=False,
        improves_original_validation_reference=best_score<references.get('original_frozen_det',{}).get('score',float('-inf')),
        beats_last_observed_validation_diagnostic=best_score<references['last_observed_diagnostic']['score'],
        limitations=[f"Training uses {int((panel['dates']<=pd.Timestamp(train_end)).sum())} dates; validation uses {int(((panel['dates']>pd.Timestamp(train_end))&(panel['dates']<=pd.Timestamp(validation_end))).sum())} dates.",
            'The 2022--2023 period was reviewed in earlier development; future evaluation is not a newly untouched external test.',
            'This search does not access or score water values after the declared validation cutoff. Later scoring requires an explicit root lock and report.',
            'A lower pre-2022 validation error does not guarantee improved 2022--2023 or future accuracy.',
            'Source proxy units and retrieval algorithms remain unverified; no field-water-safety classification is established.'])
    _write_json(out/'summary.json',summary)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(out/'candidate_validation_metrics.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'candidate_histories.csv',index=False)
    pd.DataFrame(selected['history']).to_csv(out/'training_history.csv',index=False)
    pd.DataFrame(selected['rows']).to_csv(out/'validation_predictions.csv',index=False)
    for name,value in references.items():pd.DataFrame(value['metrics']).to_csv(out/(name+'_metrics.csv'),index=False)
    emit(dict(stage='complete',selected_candidate=summary['selected_candidate'],validation_score=best_score,test_scored=False))
    return summary
