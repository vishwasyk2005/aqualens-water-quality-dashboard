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
 def test_only_det_daily_predictions(self):
  b=self.b;self.assertEqual(set(b['forecasts']),{'det'});rows=b['forecasts']['det']
  self.assertEqual(len(rows),730*6*3);self.assertEqual(len({r['date'] for r in rows}),730)
  self.assertTrue(all(r['model']=='det' and r['prediction'] is not None and pd.Timestamp(r['date'])>pd.Timestamp('2023-02-10') for r in rows))
  self.assertTrue(all(r['lower'] is None and r['upper'] is None for r in rows if r['horizon_days']>32))
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
