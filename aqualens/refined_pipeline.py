"""DET v3 integration: past-only selection, honest evaluation and separate refit."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
from .forecast import PARAMETERS,STEPS,HORIZONS,_features,_scaler,dataset_fingerprint,split_dates
from .backtest import prepare_backtest_panel,build_examples,make_test_rows,_observed_zone_counts,rows_metrics,_safe
from .det_refinement import train_refinement,load_refined_checkpoint,model_from_checkpoint,build_refined_model,causal_zone_anchors,training_target_scales,CONFIGS

def _write(path,value):Path(path).write_text(json.dumps(_safe(value),indent=2,allow_nan=False),encoding='utf-8')

def _evaluate(df,ck,train_end,validation_end):
 import torch
 past=df[pd.to_datetime(df.date)<=pd.Timestamp(validation_end)].copy()
 if dataset_fingerprint(past)!=ck.get('fit_data_fingerprint'):raise ValueError('The frozen DET selection belongs to different training/validation observations.')
 panel,mapped=prepare_backtest_panel(df,train_end=train_end)
 if panel['cells']!=ck['cells'] or panel['cell_zone']!=ck['cell_zone']:raise ValueError('Frozen DET input hierarchy differs from its checkpoint.')
 mean,scale=np.asarray(ck['mean']),np.asarray(ck['scale']);cfg=ck['config']
 x,target,origins,masks=build_examples(panel,mean,scale,cfg['window'],train_end,validation_end,str((pd.Timestamp(validation_end)+pd.Timedelta(days=1)).date()))
 anchors=causal_zone_anchors(panel,mean,scale)[origins]
 net=model_from_checkpoint(ck)
 with torch.no_grad():pred=net(torch.from_numpy(x),torch.from_numpy(anchors)).numpy()*scale+mean
 actual=target*scale+mean
 rows=make_test_rows(panel,pred,origins,masks['test'],_observed_zone_counts(mapped,panel))
 pooled=rows_metrics([row for row in rows if row['zone_id']!='all'])
 per_zone=[]
 for zone in [*panel['zones'],'all']:per_zone.extend(dict(row,zone_id=zone) for row in rows_metrics([r for r in rows if r['zone_id']==zone]))
 radii=np.zeros((3,3));coverage=[]
 for hi,horizon in enumerate(HORIZONS):
  for ki,parameter in enumerate(PARAMETERS):
   residual=np.abs(actual[:,hi,:,ki]-pred[:,hi,:,ki]);valid=masks['val'][:,hi,:,ki]&np.isfinite(residual)
   errors=residual[valid]
   if len(errors):radii[hi,ki]=float(np.quantile(errors,.8,method='higher'))
   test_valid=masks['test'][:,hi,:,ki]&np.isfinite(residual)
   coverage.append(dict(horizon_days=horizon,parameter=parameter,nominal_coverage=.8,test_coverage=float((residual[test_valid]<=radii[hi,ki]).mean()) if test_valid.any() else None,n=int(test_valid.sum()),radius=float(radii[hi,ki])))
 return panel,rows,pooled,per_zone,radii,coverage,masks

def export_historical(df,selection_dir,output_dir):
 """Score a checkpoint already locked using pre-2022 validation, once selected."""
 import torch
 selection=Path(selection_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 audit=json.loads((selection/'summary.json').read_text(encoding='utf-8'))
 ck=load_refined_checkpoint(selection/'det_refined.pt')
 if hashlib.sha256((selection/'det_refined.pt').read_bytes()).hexdigest()!=audit.get('checkpoint_sha256'):raise ValueError('The locked DET checkpoint hash does not match its selection audit.')
 if ck['train_end']!='2021-06-30' or ck['validation_end']!='2021-12-31':raise ValueError('Historical selection must use the declared pre-2022 cutoffs.')
 panel,rows,metrics,by_zone,_,_,masks=_evaluate(df,ck,ck['train_end'],ck['validation_end'])
 fingerprint=dataset_fingerprint(df);dates=panel['dates'];lock=str(dates[dates<=pd.Timestamp(ck['validation_end'])][-1].date())
 partitions={}
 for name,mask in [('training',dates<=pd.Timestamp(ck['train_end'])),('validation',(dates>pd.Timestamp(ck['train_end']))&(dates<=pd.Timestamp(ck['validation_end']))),('test',dates>pd.Timestamp(ck['validation_end']))]:
  values=dates[mask];partitions[name]=dict(first_observation=str(values[0].date()),last_observation=str(values[-1].date()),observation_dates=len(values))
 history=[dict(row,validation_score=row['validation_score']) for row in audit['history']]
 summary=dict(version=3,experiment='historical_validation_pre2022',model='det',fingerprint=fingerprint,
  train_start=str(dates[0].date()),train_end=ck['train_end'],validation_start='2021-07-01',validation_end=ck['validation_end'],test_start='2022-01-01',test_end=str(dates[-1].date()),partitions=partitions,
  weights_fit_through=ck['train_end'],scaler_fit_through=ck['train_end'],selection_uses_targets_through=ck['validation_end'],selection_lock_date=lock,model_available_from=lock,
  horizon_steps=STEPS,horizons_days=HORIZONS,forecast_horizons_days=HORIZONS,config=ck['config'],best_epoch=ck['best_epoch'],selected_candidate=ck['config']['candidate'],seed=42,max_epochs=audit['max_epochs'],patience=audit['patience'],history={'det':history},tuning=audit['tuning'],
  train_windows=int(masks['train'].any(axis=(1,2,3)).sum()),validation_windows=int(masks['val'].any(axis=(1,2,3)).sum()),test_windows=int(masks['test'].any(axis=(1,2,3)).sum()),
  test_dates_by_horizon={str(h):len({r['target_date'] for r in rows if r['horizon_days']==h}) for h in HORIZONS},first_test_target_by_horizon={str(h):min(r['target_date'] for r in rows if r['horizon_days']==h) for h in HORIZONS},
  data_audit=panel['audit'],mean=ck['mean'],scale=ck['scale'],metrics=metrics,metrics_by_zone=by_zone,weights_frozen=True,production_refit=False,checkpoint='det_historical.pt',
  evaluation='Frozen refined DET; settings and epoch selected only with origin-safe 2021 validation; native rolling forecast targets in 2022–2023.',
  refinement=dict(validation_score=audit['validation_score'],selection_criterion=audit['selection_criterion'],references=audit['references'],fixes=['Six-decimal coordinate canonicalization prevents numerical grid churn.','Validation origins cannot precede training-model availability.','Exact causal observed-zone anchors for residual candidates; learned direct DET heads also evaluated.','Training-only within-zone temporal scales balance the three parameters.'],selected_before_test=True,held_out_previously_reviewed=True),
  lake_aggregation='Both actual and predicted summaries use identical parameter-valid source-row count weights at the forecast origin, renormalized over common scored zones. This is not the changing-sample target-date raw lake mean.',
  interval_method='No uncertainty bands are claimed for historical comparison.',errors='error = prediction - actual; metrics.csv pools zones only.',
  limitations=['The 2022–2023 period was reviewed during earlier development. This remains a retrospective benchmark, not fresh external validation.','Current configuration and epoch were locked using pre-2022 validation without scoring post-2021 targets during selection.','Unknown source QA, retrieval formulas and proxy units limit environmental interpretation.','Later observed readings enter rolling input histories only after becoming available. This is not a single year-long forecast.','Negative scores and errors remain visible; no curve adjustment or test-based model selection is performed.'])
 historical=dict(ck,version=3,fingerprint=fingerprint,experiment=summary['experiment'])
 torch.save(historical,out/'det_historical.pt');summary['checkpoint_sha256']=hashlib.sha256((out/'det_historical.pt').read_bytes()).hexdigest()
 pd.DataFrame(rows).to_csv(out/'heldout_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(by_zone).to_csv(out/'metrics_by_zone.csv',index=False)
 pd.DataFrame(history).to_csv(out/'training_history.csv',index=False);pd.DataFrame(audit['tuning']).to_csv(out/'tuning.csv',index=False)
 _write(out/'predictions.json',rows);_write(out/'summary.json',summary)
 return summary

def refit_production(df,selection_dir,output_dir,progress_callback=None):
 """Refit only the locked DET configuration/epoch on all available past data."""
 import torch
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 selection=Path(selection_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 ck=load_refined_checkpoint(selection/'det_refined.pt');audit=json.loads((selection/'summary.json').read_text(encoding='utf-8'))
 if hashlib.sha256((selection/'det_refined.pt').read_bytes()).hexdigest()!=audit.get('checkpoint_sha256'):raise ValueError('The locked DET checkpoint hash does not match its selection audit.')
 panel,rows,metrics,by_zone,radii,coverage,masks=_evaluate(df,ck,ck['train_end'],ck['validation_end'])
 fingerprint=dataset_fingerprint(df);dates=panel['dates']
 history=[dict(row,validation_score=row['validation_score']) for row in audit['history']]
 summary=dict(version=3,fingerprint=fingerprint,config=ck['config'],best_epoch=ck['best_epoch'],selected_candidate=ck['config']['candidate'],seed=42,
  cells=panel['cells'],zones=panel['zones'],data_audit=panel['audit'],train_end=ck['train_end'],validation_end=ck['validation_end'],test_start=str((pd.Timestamp(ck['validation_end'])+pd.Timedelta(days=1)).date()),
  train_windows=int(masks['train'].any(axis=(1,2,3)).sum()),validation_windows=int(masks['val'].any(axis=(1,2,3)).sum()),test_windows=int(masks['test'].any(axis=(1,2,3)).sum()),history={'det':history},tuning=audit['tuning'],metrics=metrics,
  evaluation='Origin-safe chronological validation; each native horizon scored independently after model selection is available. Evaluation weights and all-history refit are separate.',
  refinement=dict(validation_score=audit['validation_score'],selection_criterion=audit['selection_criterion'],references=audit['references'],selected_before_test=True),
  calibration=dict(nominal_coverage=.8,absolute_error_radius=radii.tolist(),test_coverage=coverage,method='Empirical same-selection-validation absolute residual quantiles; not independent calibration or guaranteed coverage. Widths transfer to separately refitted production weights.'),
  production_refit=dict(epochs=ck['best_epoch'],observations_through=str(dates[-1].date()),fit='All historical eligible targets using settings and epoch selected before held-out scoring.'),
  limitations=['Turbidity/chlorophyll source algorithms, physical units and satellite QA remain unverified.','The 2022–2023 historical benchmark has already been reviewed during development; fresh independent data are needed for confirmation.','Accuracy is reported separately from successful training; negative R² is retained.','Daily display interpolates native DET outputs. Beyond32days projections are experimental and have no uncertainty bounds.','Zone intervals are empirical and can under-cover; lake intervals are not calibrated.'])
 evaluation=dict(ck,version=3,fingerprint=fingerprint,calibration=summary['calibration'],fit_partition='Train-only refined evaluation weights; validation-selected epoch')
 torch.save(evaluation,out/'det_evaluation_v2.pt')
 final_panel,_=prepare_backtest_panel(df,train_end=str(dates[-1].date()));mean,scale=_scaler(final_panel,dates[-1]);target_scales=training_target_scales(final_panel,dates[-1])
 x,target,origins,part=build_examples(final_panel,mean,scale,ck['config']['window'],str(dates[-1].date()),str(dates[-1].date()),str((dates[-1]+pd.Timedelta(days=1)).date()))
 anchor=causal_zone_anchors(final_panel,mean,scale)[origins]
 tx,ta,ty=torch.from_numpy(x),torch.from_numpy(anchor),torch.from_numpy(np.nan_to_num(target));mask=torch.from_numpy(part['train']);indices=np.flatnonzero(part['train'].any(axis=(1,2,3)))
 torch.manual_seed(42);rng=np.random.default_rng(42);net=build_refined_model(final_panel['cell_zone'],ck['config']);cfg=ck['config'];optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);ratio=torch.tensor(scale/target_scales,dtype=torch.float32)
 for epoch in range(1,ck['best_epoch']+1):
  net.train();losses=[]
  for batch in np.array_split(rng.permutation(indices),max(1,math.ceil(len(indices)/16))):
   optimizer.zero_grad();error=(net(tx[batch],ta[batch])-ty[batch])*ratio;valid=mask[batch]
   loss=(error[valid]**2).mean() if cfg['loss']=='mse' else torch.nn.functional.smooth_l1_loss(error[valid],torch.zeros_like(error[valid]));loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
  if epoch==1 or epoch%10==0:
   update=dict(stage='production_refit',epoch=epoch,epochs=ck['best_epoch'],train_loss=float(np.mean(losses)),progress=.8+.19*epoch/ck['best_epoch'],message=f'Refined DET refit: epoch {epoch}/{ck["best_epoch"]}')
   if progress_callback:progress_callback(update)
   print(update['message'],flush=True)
 production=dict(format_version=3,version=3,model_family='DET',fingerprint=fingerprint,config=cfg,state=net.state_dict(),mean=mean.tolist(),scale=scale.tolist(),target_scales=target_scales.tolist(),cells=final_panel['cells'],zones=final_panel['zones'],cell_zone=final_panel['cell_zone'],geometry_fit_through=str(dates[-1].date()),calibration=summary['calibration'],summary=summary,fit_partition='Production all-history refined DET refit')
 torch.save(production,out/'det_v2.pt')
 for name,values in [('evaluation.csv',metrics),('evaluation_by_zone.csv',by_zone),('heldout_predictions.csv',rows),('interval_coverage.csv',coverage),('training_history.csv',history),('tuning.csv',audit['tuning'])]:pd.DataFrame(values).to_csv(out/name,index=False)
 _write(out/'evaluation.json',summary);_write(out/'training_audit.json',final_panel['audit'])
 return summary

def train_current_det(df,output_dir,epochs=160,window=None,progress_callback=None):
 cut=split_dates(df)
 if pd.to_datetime(df.date).nunique()<35:raise ValueError('Refined DET requires at least 35 distinct observation dates so the chronological training partition contains at least 24 dates; 80 or more are recommended.')
 train_end=str(cut['train_end'].date());validation_end=str(cut['validation_end'].date());out=Path(output_dir);selection=out/'selection_v3'
 past=df[pd.to_datetime(df.date)<=cut['validation_end']].copy();configs=[dict(cfg) for cfg in CONFIGS]
 if window is not None:
  for cfg in configs:cfg['window']=window
 def emit(update):
  if progress_callback:progress_callback(dict(update,progress=min(.79,((update.get('candidate',1)-1)+update.get('epoch',1)/epochs)/len(configs)*.79)))
 train_refinement(past,selection,epochs=epochs,configs=configs,progress_callback=emit,train_end=train_end,validation_end=validation_end)
 return refit_production(df,selection,out,progress_callback)

def forecast_refined(df,ck,days):
 import torch
 torch.set_num_threads(2)
 cutoff=ck.get('geometry_fit_through') or ck.get('train_end')
 panel,_=prepare_backtest_panel(df,train_end=cutoff)
 if panel['cells']!=ck['cells'] or panel['cell_zone']!=ck['cell_zone']:raise ValueError('Refined DET frozen hierarchy differs from the uploaded observations.')
 mean,scale=np.asarray(ck['mean']),np.asarray(ck['scale']);window=ck['config']['window'];features=_features(panel,mean,scale)
 if len(features)<window:raise ValueError('Not enough history for the refined DET.')
 history=features[-window:].copy();anchor=causal_zone_anchors(panel,mean,scale)[-1].copy();latest=panel['dates'][-1];net=model_from_checkpoint(ck)
 offsets=[0];values=[anchor*scale+mean];origin=0
 while origin<days:
  with torch.no_grad():pred=net(torch.from_numpy(history[None]),torch.from_numpy(anchor[None])).numpy()[0]*scale+mean
  for hi,horizon in enumerate(HORIZONS):offsets.append(origin+horizon);values.append(pred[hi])
  if origin+32>=days:break
  base=history[-1,:,:3].copy();zbase=anchor.copy()
  for step in range(1,5):
   projected=np.array([[np.interp(step*8,[0,*HORIZONS],np.r_[zbase[z,k]*scale[k]+mean[k],pred[:,z,k]]) for k in range(3)] for z in range(len(panel['zones']))])
   nextrow=history[-1].copy();normalized=(projected-mean)/scale;nextrow[:,:3]=base+(normalized-zbase)[panel['cell_zone']]
   nextrow[:,3:6]=0;nextrow[:,6:9]=np.minimum(nextrow[:,6:9]+8/64,1);when=latest+pd.Timedelta(days=origin+step*8);phase=when.day_of_year/365.25*2*np.pi;nextrow[:,9:11]=[np.sin(phase),np.cos(phase)];nextrow[:,11]=1
   history=np.concatenate([history[1:],nextrow[None]]);anchor=normalized.astype('float32')
  origin+=32
 order=np.argsort(offsets);offsets=np.asarray(offsets)[order];values=np.asarray(values)[order];unique=np.r_[True,np.diff(offsets)>0];offsets,values=offsets[unique],values[unique]
 radii=np.asarray(ck['calibration']['absolute_error_radius']);rows=[]
 for horizon in range(1,days+1):
  for zi,zone in enumerate(panel['zones']):
   for ki,parameter in enumerate(PARAMETERS):
    value=float(np.interp(horizon,offsets,values[:,zi,ki]));radius=float(np.interp(horizon,HORIZONS,radii[:,ki])) if horizon<=32 else np.nan
    lower=value-radius if np.isfinite(radius) else np.nan;upper=value+radius if np.isfinite(radius) else np.nan
    if parameter=='ndwi':
     value=float(np.clip(value,-1,1));lower=float(np.clip(lower,-1,1)) if np.isfinite(lower) else lower;upper=float(np.clip(upper,-1,1)) if np.isfinite(upper) else upper
    rows.append(dict(date=latest+pd.Timedelta(days=horizon),zone_id=zone,parameter=parameter,prediction=value,lower=lower,upper=upper,model='det',horizon_days=horizon,is_extrapolation=horizon>32,interval_basis='Empirical validation residual width transferred to production refit; coverage unverified' if horizon<=32 else 'Unavailable beyond32days',prediction_basis='Refined DET native anchors; daily interpolation; future NDWI display and empirical bounds clipped to index domain [-1,1]' if horizon<=32 else 'Experimental recursive refined DET'))
 return pd.DataFrame(rows)
