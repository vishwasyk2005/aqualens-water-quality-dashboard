"""Export the real MODIS/DET workspace for local offline viewing."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server import Dashboard,clean
from aqualens.forecast import dataset_fingerprint

def json_safe(value):return clean(value)

def build_snapshot():
    app=Dashboard();df,report=app.dataset('modis')
    payload=app.bootstrap('modis')
    fingerprint=dataset_fingerprint(df)
    if payload['evaluation_meta'].get('fingerprint')!=fingerprint:raise ValueError('Train DET on this real dataset before exporting.')
    payload['static_bundle']={'version':2,'title':'AquaLens supplied MODIS dataset and trained DET','synthetic':False,'source':'legacy_upload','fingerprint':fingerprint,'as_of':report['end'],'generated_at':datetime.now(timezone.utc).isoformat(),'description':'Computed from the user-provided historical CSV, not a live satellite feed. Source acquisition and proxy units are unverified.'}
    payload['forecasts']={'det':app.forecast('modis','det',730)['rows']}
    payload['backtest']=app.backtest('modis')
    payload['explanations']={}
    for zone in ['all']+sorted(df.zone_id.unique()):
        for parameter in ['ndwi','turbidity_proxy','chlorophyll_proxy']:
            payload['explanations'][f'{zone}:{parameter}']=app.explain('modis',zone,parameter,horizon_days=32)
    return clean(payload)

def main():
    snapshot=build_snapshot();path=ROOT/'web'/'demo-state.json'
    path.write_text(json.dumps(snapshot,allow_nan=False,separators=(',',':')),encoding='utf-8')
    print(f"Exported real MODIS workspace: {len(snapshot['observations'])} observations; {len(snapshot['forecasts']['det'])} DET predictions; {len(snapshot['explanations'])} evidence reports; {path.stat().st_size:,} bytes",flush=True)
if __name__=='__main__':main()
