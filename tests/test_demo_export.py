"""Real-data DET snapshot provenance and horizon checks."""
from pathlib import Path
import json,sys,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aqualens.data import load_data
from aqualens.forecast import dataset_fingerprint
from scripts.export_dashboard_demo import json_safe
class RealWorkspaceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.b=json.loads((ROOT/'web'/'demo-state.json').read_text(encoding='utf-8'))
 def test_real_data_and_fingerprint(self):
  b=self.b;d,r=load_data(ROOT/'data'/'modis_observations.csv')
  self.assertEqual(b['dataset'],'modis');self.assertFalse(b['static_bundle']['synthetic'])
  self.assertEqual(b['static_bundle']['fingerprint'],dataset_fingerprint(d))
  self.assertEqual(b['evaluation_meta']['fingerprint'],dataset_fingerprint(d))
  self.assertEqual(len(b['observations']),46755);self.assertEqual(b['report']['ndwi_invalid_count'],2476)
  self.assertEqual(b['report']['end'],'2023-02-10')
 def test_det_reference_and_seasonal_forecasts(self):
  b=self.b;rows=b['forecasts']['det']
  self.assertEqual(len(rows),730*6*3);self.assertEqual(len({r['date'] for r in rows}),730)
  self.assertTrue(all(r['model']=='det' and r['prediction'] is not None and pd.Timestamp(r['date'])>pd.Timestamp('2023-02-10') for r in rows))
  self.assertTrue(all(r['lower'] is None and r['upper'] is None for r in rows if r['horizon_days']>32))
  seasonal=b['forecasts']['seasonal_kernel']
  self.assertEqual(len(seasonal),730*7*3)
  self.assertTrue(all(r['model']=='seasonal_kernel' and r['prediction'] is not None for r in seasonal))
  self.assertEqual({r['parameter'] for r in seasonal},{'ndwi','turbidity_proxy','chlorophyll_proxy'})
 def test_prophet_has_daily_forecasts_and_historical_pairs(self):
  b=self.b
  forecast=b['forecasts']['prophet']
  self.assertEqual(len(forecast),730*7*3)
  self.assertTrue(all(r['prediction'] is not None and r['model']=='prophet' for r in forecast))
  self.assertEqual(len({r['date'] for r in forecast}),730)
  metrics=b['model_benchmark']['metrics']
  self.assertEqual(sum(m['n'] for m in metrics if m['model']=='prophet' and m['zone_id']=='all'),447)
 def test_lake_summary_archive_matches_its_exact_pairs(self):
  results=json.loads((ROOT/'web'/'station'/'lake-summary-results.json').read_text(encoding='utf-8'))
  self.assertEqual(results['fingerprint'],self.b['model_benchmark']['fingerprint'])
  self.assertEqual(len(results['series']),3)
  for series in results['series']:
   self.assertEqual({row['horizon_days'] for row in series['forecasts']},{8,16,32})
   for metric in series['metrics']:
    pairs=pd.DataFrame([p for p in series['pairs'] if p['horizon_days']==metric['horizon_days']])
    self.assertEqual(len(pairs),metric['n'])
    self.assertFalse(pairs[['origin_date','target_date']].duplicated().any())
    self.assertTrue((pairs['origin_date']<pairs['target_date']).all())
    r2=1-((pairs.actual-pairs.prediction)**2).sum()/((pairs.actual-pairs.actual.mean())**2).sum()
    self.assertAlmostEqual(r2,metric['r2'])
 def test_evidence_uses_real_sensor_context(self):
  self.assertEqual(len(self.b['explanations']),21)
  for e in self.b['explanations'].values():
   self.assertEqual(pd.Timestamp(e['provenance']['as_of']),pd.Timestamp('2023-02-10'))
   self.assertNotIn('synthetic_demo',e['provenance']['source_kinds'])
   self.assertNotIn('gee-sentinel-quality',[s['source_id'] for s in e['sources']])
 def test_strict_json(self):
  import numpy as np
  self.assertEqual(json.loads(json.dumps(json_safe([np.nan,np.inf,pd.NaT,np.int64(2)]),allow_nan=False)),[None,None,None,2])
if __name__=='__main__':unittest.main()
