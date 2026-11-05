"""Real-data DET-only training, chronological audit and calibrated forecasts.
Native MODIS cadence is approximately eight days. Daily display values are
interpolations; forecasts beyond 32 days are experimental recursive projections.
"""
from pathlib import Path
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd

PARAMETERS = ['ndwi', 'turbidity_proxy', 'chlorophyll_proxy']
WINDOW = 12
VERSION = 2
STEPS = [1, 2, 4]
HORIZONS = [8, 16, 32]

def dataset_fingerprint(df):
    cols = [c for c in ['date','lake_id','zone_id','cell_id','lat','lon',*PARAMETERS,*[k+'_raw' for k in PARAMETERS],*[k+'_is_observed' for k in PARAMETERS],*[k+'_was_missing' for k in PARAMETERS],'is_observed','valid_pixel_fraction','sensor','source_kind','algorithm_version'] if c in df]
    d = df[cols].copy()
    if 'is_observed' in d:
        valid = d.is_observed.astype(str).str.lower().isin(['true','1','1.0','yes'])
        d['is_observed'] = valid
        d.loc[~valid, PARAMETERS] = np.nan
    d['date'] = pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
    d = d.sort_values(['date','zone_id','cell_id']).reset_index(drop=True)
    return hashlib.sha256(d.to_csv(index=False,float_format='%.17g').encode()).hexdigest()

def split_dates(df):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df.date).dropna().unique()))
    if len(dates)<32:
        raise ValueError('DET requires at least 32 distinct observation dates; 80 or more recommended.')
    a,b = int(len(dates)*.70),int(len(dates)*.85)
    return {'train_end':dates[a-1], 'validation_end':dates[b-1], 'test_start':dates[b]}

def _panel(df):
    needed = {'date','zone_id','cell_id',*PARAMETERS}
    if not needed.issubset(df):
        raise ValueError('Missing forecast columns: '+', '.join(sorted(needed-set(df))))
    d=df.copy()
    if 'lake_id' in d and d.lake_id.nunique()>1:
        raise ValueError('Select one lake before training DET.')
    d['date']=pd.to_datetime(d.date).dt.normalize()
    d['zone_id']=d.zone_id.astype(str)
    d['cell_id']=d.cell_id.astype(str)
    if (d.groupby('cell_id').zone_id.nunique()>1).any():
        raise ValueError('Each cell_id must belong to exactly one zone.')
    original_cells=int(d.cell_id.nunique())
    # Spatial coarsening reduces sparse sampling noise; it is data preparation,
    # not a second forecasting model. Group boundaries depend on geography only.
    if original_cells>96 and {'lat','lon'}.issubset(d):
        lat=(d.lat-d.lat.min())/max(float(d.lat.max()-d.lat.min()),1e-9)
        lon=(d.lon-d.lon.min())/max(float(d.lon.max()-d.lon.min()),1e-9)
        a=np.minimum((lat*6).astype(int),5).astype(str)
        b=np.minimum((lon*6).astype(int),5).astype(str)
        d['cell_id']=d.zone_id+'_grid_'+a+'_'+b
    zones,cells=sorted(d.zone_id.unique()),sorted(d.cell_id.unique())
    parent=d.groupby('cell_id').zone_id.first()
    parents=[zones.index(parent[c]) for c in cells]
    for k in PARAMETERS:
        source=k+'_raw' if k+'_raw' in d else k
        d[k]=pd.to_numeric(d[source],errors='coerce').replace([np.inf,-np.inf],np.nan)
        flag=k+'_was_missing'
        if flag in d: d.loc[d[flag].astype(bool),k]=np.nan
    invalid_ndwi=int(d.ndwi_domain_invalid.astype(bool).sum()) if 'ndwi_domain_invalid' in d else int((d.ndwi.notna() & ~d.ndwi.between(-1,1)).sum())
    d.loc[~d.ndwi.between(-1,1),'ndwi']=np.nan
    for parameter in PARAMETERS:
        flag=parameter+'_is_observed'
        if flag in d:
            valid=d[flag].astype(str).str.lower().isin(['true','1','1.0','yes'])
            d.loc[~valid,parameter]=np.nan
        elif 'is_observed' in d:
            valid=d.is_observed.astype(str).str.lower().isin(['true','1','1.0','yes'])
            d.loc[~valid,parameter]=np.nan
    excluded_coverage=0
    if 'valid_pixel_fraction' in d:
        coverage=pd.to_numeric(d.valid_pixel_fraction,errors='coerce')
        bad=coverage.notna() & (coverage<.5)
        excluded_coverage=int(bad.sum()); d.loc[bad,PARAMETERS]=np.nan
    dates=pd.DatetimeIndex(sorted(d.date.unique()))
    grouped=d.groupby(['date','cell_id'])[PARAMETERS].mean()
    grid=pd.MultiIndex.from_product([dates,cells],names=['date','cell_id'])
    raw=grouped.reindex(grid).to_numpy(float).reshape(len(dates),len(cells),3)
    # Zone targets are observed source measurements (never filled targets).
    zg=d.groupby(['date','zone_id'])[PARAMETERS].mean()
    zg=zg.reindex(pd.MultiIndex.from_product([dates,zones],names=['date','zone_id']))
    targets=zg.to_numpy(float).reshape(len(dates),len(zones),3)
    filled=raw.copy(); age=np.ones_like(raw)*32
    previous=np.full((len(cells),3),np.nan); previous_day=np.full((len(cells),3),np.nan)
    for t,date in enumerate(dates):
        day=date.value/86400e9
        valid=np.isfinite(raw[t]); previous[valid]=raw[t][valid]; previous_day[valid]=day
        days=day-previous_day
        age[t]=np.where(np.isfinite(days),np.minimum(days,64),64)
        filled[t]=np.where(days<=16,previous,np.nan)
    return {'dates':dates,'raw':raw,'filled':filled,'age':age,'targets':targets,'cells':cells,'zones':zones,'cell_zone':parents,
            'audit':{'original_cells':original_cells,'model_cells':len(cells),'zones':len(zones),'observation_dates':len(dates),
                     'date_start':str(dates[0].date()),'date_end':str(dates[-1].date()),'median_cadence_days':float(np.median(np.diff(dates.values).astype('timedelta64[D]').astype(int))),
                     'invalid_ndwi_excluded':invalid_ndwi,'known_low_coverage_excluded':excluded_coverage,
                     'spatial_aggregation':'6 by 6 geographic grid intersected with zones' if original_cells>96 else 'original cells',
                     'missing_cell_parameter_fraction':float((~np.isfinite(raw)).mean()),
                     'negative_proxy_values_retained':'Algorithm and units are unverified; no physical concentration assumption is made.'}}

def _scaler(p,train_end):
    values=p['raw'][p['dates']<=train_end]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(values,axis=(0,1)); scale=np.nanstd(values,axis=(0,1))
    if not np.isfinite(mean).all(): raise ValueError('Each parameter needs observed training values.')
    return mean,np.where(np.isfinite(scale)&(scale>1e-6),scale,1.)

def _features(p,mean,scale):
    values=np.where(np.isfinite(p['filled']),p['filled'],mean)
    phase=p['dates'].day_of_year.to_numpy()/365.25*2*np.pi
    seasonal=np.broadcast_to(np.stack([np.sin(phase),np.cos(phase)],-1)[:,None,:],(len(phase),len(p['cells']),2))
    gaps=np.r_[8,np.diff(p['dates'].values).astype('timedelta64[D]').astype(float)]/8
    gap_feature=np.broadcast_to(gaps[:,None,None],(len(gaps),len(p['cells']),1))
    return np.concatenate([(values-mean)/scale,np.isfinite(p['raw']).astype(float),p['age']/64,seasonal,gap_feature],-1).astype('float32')

def _examples(p,mean,scale,window=WINDOW):
    features=_features(p,mean,scale)
    origins=np.arange(window-1,len(features)-max(STEPS))
    x=np.stack([features[i-window+1:i+1] for i in origins])
    y=np.stack([(p['targets'][i+np.array(STEPS)]-mean)/scale for i in origins])
    return x,y.astype('float32'),origins

def _metrics(actual,predicted,horizons=HORIZONS):
    rows=[]
    for hi,h in enumerate(horizons):
        for k,name in enumerate(PARAMETERS):
            valid=np.isfinite(actual[:,hi,:,k])&np.isfinite(predicted[:,hi,:,k])
            a,b=actual[:,hi,:,k][valid],predicted[:,hi,:,k][valid]
            e=a-b; denom=float(((a-a.mean())**2).sum()) if len(a) else 0
            rows.append({'model':'det','parameter':name,'horizon_days':h,'mae':float(np.abs(e).mean()) if len(a) else None,
                         'rmse':float(np.sqrt((e**2).mean())) if len(a) else None,
                         'r2':float(1-(e**2).sum()/denom) if denom>0 else None,'n':len(a)})
    return rows

def _load_checkpoint(df,path):
    import torch
    if path is None or not Path(path).exists(): raise ValueError('Train DET on this dataset before forecasting.')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('version')!=VERSION or ck.get('fingerprint')!=dataset_fingerprint(df):
        raise ValueError('This DET checkpoint belongs to a different dataset. Train DET for these observations.')
    return ck

def forecast_zones(df,horizon_weeks=4,model='det',checkpoint_path=None,horizon_days=None):
    if model!='det': raise ValueError('DET is the only supported forecasting model.')
    days=int(horizon_days if horizon_days is not None else horizon_weeks*7)
    if not 1<=days<=730: raise ValueError('Choose a forecast horizon between 1 and 730 days.')
    ck=_load_checkpoint(df,checkpoint_path)
    from .tree_model import HierarchicalForecaster
    import torch
    p=_panel(df); mean=np.asarray(ck['mean']); scale=np.asarray(ck['scale'])
    if p['cells']!=ck['cells'] or p['zones']!=ck['zones']: raise ValueError('The uploaded spatial hierarchy differs from the trained DET.')
    cfg=ck['config']; window=cfg['window']; f=_features(p,mean,scale)
    if len(f)<window: raise ValueError('Insufficient history for DET.')
    net=HierarchicalForecaster(p['cell_zone'],cfg['hidden']); net.load_state_dict(ck['state']); net.eval()
    history=f[-window:].copy(); latest=p['dates'][-1]
    anchors=np.stack([history[-1,np.array(p['cell_zone'])==z,:3].mean(0)*scale+mean for z in range(len(p['zones']))])
    anchors=np.where(np.isfinite(p['targets'][-1]),p['targets'][-1],anchors)
    offsets=[0]; values=[anchors]; origin=0
    # Recursive blocks are exposed as experimental after the validated direct horizons.
    while origin<days:
        with torch.no_grad(): predicted=net(torch.from_numpy(history[None])).numpy()[0]*scale+mean
        predicted[:,:,0]=np.clip(predicted[:,:,0],-1,1)
        for hi,h in enumerate(HORIZONS): offsets.append(origin+h); values.append(predicted[hi])
        if origin+32>=days: break
        base=history[-1,:,:3].copy()
        zbase=np.stack([base[np.array(p['cell_zone'])==z].mean(0) for z in range(len(p['zones']))])
        for step in range(1,5):
            projected=np.stack([np.interp(step*8,[0,*HORIZONS],np.r_[zbase[z,k]*scale[k]+mean[k],predicted[:,z,k]]) for z in range(len(p['zones'])) for k in range(3)]).reshape(len(p['zones']),3)
            nextrow=history[-1].copy()
            nextrow[:,:3]=base+((projected-mean)/scale-zbase)[p['cell_zone']]
            nextrow[:,3:6]=0; nextrow[:,6:9]=np.minimum(nextrow[:,6:9]+8/64,1)
            date=latest+pd.Timedelta(days=origin+step*8); phase=date.day_of_year/365.25*2*np.pi
            nextrow[:,9:11]=[np.sin(phase),np.cos(phase)]; nextrow[:,11]=1
            history=np.concatenate([history[1:],nextrow[None]])
        origin+=32
    order=np.argsort(offsets); offsets=np.asarray(offsets)[order]; values=np.asarray(values)[order]
    # Remove repeated block boundary offsets while keeping its previously predicted value.
    unique=np.r_[True,np.diff(offsets)>0]; offsets,values=offsets[unique],values[unique]
    radii=np.asarray(ck['calibration']['absolute_error_radius'])
    records=[]
    for day in range(1,days+1):
        for z,zone in enumerate(p['zones']):
            for k,name in enumerate(PARAMETERS):
                value=float(np.interp(day,offsets,values[:,z,k])); radius=float(np.interp(day,HORIZONS,radii[:,k])) if day<=32 else np.nan
                lower=value-radius if np.isfinite(radius) else np.nan; upper=value+radius if np.isfinite(radius) else np.nan
                if k==0:
                    value=float(np.clip(value,-1,1)); lower=max(-1,lower) if np.isfinite(lower) else lower; upper=min(1,upper) if np.isfinite(upper) else upper
                records.append({'date':latest+pd.Timedelta(days=day),'zone_id':zone,'parameter':name,'prediction':value,
                                'lower':lower,'upper':upper,'model':'det','horizon_days':day,'is_extrapolation':day>32,
                                'interval_basis':'Empirical band at nominal 80%; shared selection-validation residuals; interpolated between native horizons' if day<=32 else 'Unavailable beyond validated 32-day horizon',
                                'prediction_basis':'Direct DET anchors; daily linear interpolation' if day<=32 else 'Experimental recursive DET projection'})
    return pd.DataFrame(records)

def train_models(df,output_dir,epochs=160,window=None,progress_callback=None):
    import torch
    from .tree_model import HierarchicalForecaster
    if epochs<1: raise ValueError('epochs must be positive.')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    p,cut=_panel(df),split_dates(df); mean,scale=_scaler(p,cut['train_end'])
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    fingerprint=dataset_fingerprint(df)
    configs=[{'window':window or 12,'hidden':24,'lr':.003}, {'window':window or 8,'hidden':32,'lr':.001}]
    if len(p['dates'])<60: configs=configs[:1]; configs[0]['window']=min(configs[0]['window'],6)
    def emit(data):
        (out/'training_progress.json').write_text(json.dumps(data),encoding='utf-8')
        if progress_callback: progress_callback(data)
    def setup(cfg,scaler_mean=mean,scaler_scale=scale):
        x,y,origins=_examples(p,scaler_mean,scaler_scale,cfg['window'])
        last=p['dates'][origins+max(STEPS)]; first=p['dates'][origins+min(STEPS)]
        masks={'train':last<=cut['train_end'],'val':(first>cut['train_end'])&(last<=cut['validation_end']), 'test':first>cut['validation_end']}
        if any(mask.sum()<2 for mask in masks.values()): raise ValueError('Insufficient chronological windows; provide more observation dates.')
        return torch.from_numpy(x),torch.from_numpy(np.nan_to_num(y)),torch.from_numpy(np.isfinite(y)),origins,masks,y
    def loss(pred,target,mask):
        if not mask.any(): raise ValueError('No observed targets in this partition.')
        return torch.nn.functional.smooth_l1_loss(pred[mask],target[mask])
    tuning=[]; selected=None; selected_score=float('inf'); histories={}
    for ci,cfg in enumerate(configs):
        torch.manual_seed(42); rng=np.random.default_rng(42)
        x,y,mask,origins,masks,raw_y=setup(cfg)
        net=HierarchicalForecaster(p['cell_zone'],cfg['hidden'])
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=1e-3)
        best=float('inf'); best_state=None; stale=0; best_epoch=0; history=[]
        ix=np.flatnonzero(masks['train'])
        for epoch in range(1,epochs+1):
            net.train(); losses=[]
            for batch in np.array_split(rng.permutation(ix),max(1,math.ceil(len(ix)/16))):
                opt.zero_grad(); pred=net(x[batch]); objective=loss(pred,y[batch],mask[batch]); objective.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); losses.append(float(objective.detach()))
            net.eval()
            with torch.no_grad(): val=float(loss(net(x[masks['val']]),y[masks['val']],mask[masks['val']]))
            row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':val}; history.append(row)
            if val<best-1e-5:
                best,best_epoch,stale=val,epoch,0; best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
            else: stale+=1
            if epoch==1 or epoch%5==0:
                emit({'stage':'validation_search','candidate':ci+1,'candidates':len(configs),'epoch':epoch,'epochs':epochs,
                      'train_loss':row['train_loss'],'val_loss':val,'progress':min(.7,(ci+epoch/epochs)/len(configs)*.7),
                      'message':f'DET candidate {ci+1}/{len(configs)}, epoch {epoch}: validation {val:.5f}'})
                print(f'DET candidate={ci+1} epoch={epoch} train={row["train_loss"]:.6f} val={val:.6f}',flush=True)
            if stale>=18: break
        histories[str(ci+1)]=history
        tuning.append({**cfg,'candidate':ci+1,'best_epoch':best_epoch,'validation_loss':best,'epochs_run':len(history)})
        if best<selected_score:
            selected_score=best; selected=(cfg.copy(),best_state,best_epoch,ci+1)
    cfg,state,best_epoch,candidate=selected
    x,y,mask,origins,masks,raw_y=setup(cfg)
    evaluation_net=HierarchicalForecaster(p['cell_zone'],cfg['hidden']); evaluation_net.load_state_dict(state); evaluation_net.eval()
    with torch.no_grad():
        validation_pred=evaluation_net(x[masks['val']]).numpy()*scale+mean
        test_pred=evaluation_net(x[masks['test']]).numpy()*scale+mean
    validation_actual=raw_y[masks['val']]*scale+mean; actual=raw_y[masks['test']]*scale+mean
    radii=np.zeros((3,3)); counts=np.zeros((3,3),int)
    for h in range(3):
        for k in range(3):
            errors=np.abs(validation_actual[:,h,:,k]-validation_pred[:,h,:,k]); errors=errors[np.isfinite(errors)]
            if len(errors)<5: raise ValueError('Insufficient validation residuals for uncertainty calibration.')
            q=min(1,math.ceil((len(errors)+1)*.8)/len(errors))
            radii[h,k]=np.quantile(errors,q,method='higher'); counts[h,k]=len(errors)
    metrics=_metrics(actual,test_pred); per_zone=[]; predictions=[]; coverage=[]
    for z,zone in enumerate(p['zones']):
        per_zone.extend([{**r,'zone_id':zone} for r in _metrics(actual[:,:,z:z+1],test_pred[:,:,z:z+1])])
    test_origins=origins[masks['test']]
    for n,i in enumerate(test_origins):
        for h,step in enumerate(STEPS):
            for z,zone in enumerate(p['zones']):
                for k,parameter in enumerate(PARAMETERS):
                    if np.isfinite(actual[n,h,z,k]):
                        predictions.append({'origin_date':str(p['dates'][i].date()),'target_date':str(p['dates'][i+step].date()),
                          'horizon_days':HORIZONS[h],'actual_elapsed_days':int((p['dates'][i+step]-p['dates'][i]).days),'zone_id':zone,'parameter':parameter,
                          'actual':float(actual[n,h,z,k]),'prediction':float(test_pred[n,h,z,k]),'lower':float(test_pred[n,h,z,k]-radii[h,k]),'upper':float(test_pred[n,h,z,k]+radii[h,k])})
    for h,horizon in enumerate(HORIZONS):
        for k,parameter in enumerate(PARAMETERS):
            errors=np.abs(actual[:,h,:,k]-test_pred[:,h,:,k]); valid=np.isfinite(errors)
            coverage.append({'horizon_days':horizon,'parameter':parameter,'nominal_coverage':.8,'test_coverage':float((errors[valid]<=radii[h,k]).mean()),'n':int(valid.sum()),'radius':float(radii[h,k])})
    calibration={'nominal_coverage':.8,'absolute_error_radius':radii.tolist(),'validation_counts':counts.tolist(),'test_coverage':coverage,
                 'method':'Nominal 80% absolute validation residual quantile pooled over zones per horizon and parameter. The SAME validation partition selected the model and estimated residual widths; this is an empirical band, not independent conformal calibration. Temporal dependence, observed held-out undercoverage, and transfer to all-history production refit preclude guaranteed coverage.'}
    common={'version':VERSION,'fingerprint':fingerprint,'config':cfg,'cells':p['cells'],'zones':p['zones'],'cell_zone':p['cell_zone'],'calibration':calibration,
            'horizons_days':HORIZONS,'architecture':'GRU temporal encoder -> Child-Sum Tree-LSTM cells to zones -> Child-Sum Tree-LSTM zones to lake -> residual multihorizon zone heads'}
    torch.save({**common,'state':state,'mean':mean.tolist(),'scale':scale.tolist(),'fit_partition':'training dates only; validation-selected epoch'},out/'det_evaluation_v2.pt')
    # Production refit is separate: no claims of holdout evaluation for these weights.
    final_mean,final_scale=_scaler(p,p['dates'][-1]); fx,fy,fmask,_,_,_=setup(cfg,final_mean,final_scale)
    torch.manual_seed(42); rng=np.random.default_rng(42)
    final=HierarchicalForecaster(p['cell_zone'],cfg['hidden']); opt=torch.optim.AdamW(final.parameters(),lr=cfg['lr'],weight_decay=1e-3)
    for epoch in range(1,best_epoch+1):
        final.train(); losses=[]
        for batch in np.array_split(rng.permutation(len(fx)),max(1,math.ceil(len(fx)/16))):
            opt.zero_grad(); value=loss(final(fx[batch]),fy[batch],fmask[batch]); value.backward()
            torch.nn.utils.clip_grad_norm_(final.parameters(),1.); opt.step(); losses.append(float(value.detach()))
        if epoch==1 or epoch%5==0:
            emit({'stage':'production_refit','epoch':epoch,'epochs':best_epoch,'train_loss':float(np.mean(losses)),'val_loss':None,
                  'progress':.7+.29*epoch/best_epoch,'message':f'Final DET refit on all history: epoch {epoch}/{best_epoch}'})
            print(f'Production refit epoch={epoch}/{best_epoch}',flush=True)
    summary={**{k:v for k,v in common.items() if k!='cell_zone'},'seed':42,'data_audit':p['audit'],'train_end':str(cut['train_end'].date()),
             'validation_end':str(cut['validation_end'].date()),'test_start':str(cut['test_start'].date()),
             'train_windows':int(masks['train'].sum()),'validation_windows':int(masks['val'].sum()),'test_windows':int(masks['test'].sum()),
             'window_observations':cfg['window'],'window_weeks':round(cfg['window']*p['audit']['median_cadence_days']/7,2),
             'history':{'det':histories[str(candidate)]},'candidate_histories':histories,'tuning':tuning,'best_epoch':best_epoch,
             'selected_candidate':candidate,'metrics':metrics,'evaluation':'Chronological 70/15/15; no target window crosses a partition boundary. Rolling observed-history origins; direct 1/2/4 native observation steps (~8/16/32 days).',
             'production_refit':{'epochs':best_epoch,'observations_through':str(p['dates'][-1].date()),'fit':'All available historical windows; configuration and epoch count selected before final held-out scoring.',
                                  'interval_transfer':'Empirical residual widths from the same validation set used for model selection are transferred from evaluation model to production refit; production coverage is not independently validated.'},
             'limitations':['MODIS source algorithm, acquisition provenance and proxy units have not been independently verified.',
                            'Spatial grid aggregation is documented preprocessing; no second forecasting model is used.',
                            'Unknown pixel quality is permitted and reported; satellite proxies do not establish drinking-water safety.',
                            'Reported test scores belong to det_evaluation_v2.pt; det_v2.pt is a separate production refit.',
                            'Native observation gaps vary; elapsed target days are retained in heldout_predictions.csv.',
                            'Daily curves interpolate native predictions. Beyond 32 days forecasts are experimental recursive projections, with no calibrated interval.',
                            'Intervals are validation-calibrated empirical bands; temporal dependence and production refit limit guarantees.',
                            'Negative and extreme proxy values remain because their algorithms and units are unverified; invalid NDWI outside [-1,1] is masked.']}
    torch.save({**common,'state':final.state_dict(),'mean':final_mean.tolist(),'scale':final_scale.tolist(),'fit_partition':'production all-history refit','summary':summary},out/'det_v2.pt')
    pd.DataFrame(metrics).to_csv(out/'evaluation.csv',index=False)
    pd.DataFrame(per_zone).to_csv(out/'evaluation_by_zone.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'heldout_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    pd.DataFrame(histories[str(candidate)]).to_csv(out/'training_history.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'interval_coverage.csv',index=False)
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    emit({'stage':'complete','progress':1,'message':'DET trained on the uploaded historical dataset; held-out results saved.'})
    return summary

