"""Export the real MODIS workspace with selected and reference forecasts."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server import Dashboard,clean
from aqualens.forecast import dataset_fingerprint

def json_safe(value):return clean(value)

def build_snapshot():
    app=Dashboard(use_custom_models=False);df,report=app.dataset('modis')
    payload=app.bootstrap('modis')
    fingerprint=dataset_fingerprint(df)
    if payload['evaluation_meta'].get('fingerprint')!=fingerprint:raise ValueError('Train DET on this real dataset before exporting.')
    benchmark=payload['model_benchmark']
    if not benchmark.get('available'):
        raise ValueError('Generate the bundled forecast comparison before exporting.')
    payload['static_bundle']={'version':3,'title':'AquaLens supplied MODIS dataset and forecast comparison','synthetic':False,'source':'legacy_upload','fingerprint':fingerprint,'as_of':report['end'],'generated_at':datetime.now(timezone.utc).isoformat(),'description':'Computed from the user-provided historical CSV, not a live satellite feed. Source acquisition and proxy units are unverified.'}
    results={name:app.forecast('modis',name,730) for name in ('auto','det','consensus','seasonal_kernel','prophet')}
    payload['forecasts']={name:result['rows'] for name,result in results.items()}
    payload['forecast_metadata']={name:result['metadata'] for name,result in results.items()}
    payload['monitoring_projections'] = {}
    for name, result in results.items():
        native = dict(result, rows=[row for row in result['rows'] if row['horizon_days'] in (8, 16, 32)])
        payload['monitoring_projections'][name] = {
            zone: app.monitoring_projection('modis', df, observed, native)
            for zone, observed in payload['satellite_monitoring'].items()
        }
    payload['backtest']=app.backtest('modis')
    payload['backtests']={'auto':app.backtest('modis',model='auto'),'det':payload['backtest']}
    for model in benchmark['models']:
        if model['id']!='det':
            payload['backtests'][model['id']]=app.backtest('modis',model=model['id'])
    if not all(result['available'] for result in payload['backtests'].values()):
        raise ValueError('A candidate could not be aligned with the recorded benchmark targets.')
    payload['explanations']={}
    for zone in ['all']+sorted(df.zone_id.unique()):
        for parameter in ['ndwi','turbidity_proxy','chlorophyll_proxy']:
            payload['explanations'][f'{zone}:{parameter}']=app.explain('modis',zone,parameter,horizon_days=32,model='auto')
    return clean(payload)

def main():
    snapshot=build_snapshot();path=ROOT/'web'/'demo-state.json'
    path.write_text(json.dumps(snapshot,allow_nan=False,separators=(',',':')),encoding='utf-8')
    print(f"Exported real MODIS workspace: {len(snapshot['observations'])} observations; {len(snapshot['forecasts']['auto'])} selected forecasts; {len(snapshot['explanations'])} evidence reports; {path.stat().st_size:,} bytes",flush=True)
if __name__=='__main__':main()
