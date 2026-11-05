"""Delivery checks against the real historical predictions and source measurements."""
from pathlib import Path
import hashlib,json,sys,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aqualens.data import load_data,PARAMETERS
from aqualens.forecast import dataset_fingerprint
class HistoricalDeliveryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.path=ROOT/'models'/'historical_validation'
  cls.meta=json.loads((cls.path/'summary.json').read_text(encoding='utf-8'))
  cls.rows=pd.read_csv(cls.path/'heldout_predictions.csv')
  cls.source,_=load_data(ROOT/'data'/'modis_observations.csv')
 def test_saved_model_period_identity_and_integrity(self):
  m=self.meta
  self.assertEqual(m['fingerprint'],dataset_fingerprint(self.source))
  self.assertLessEqual(pd.Timestamp(m['weights_fit_through']),pd.Timestamp('2021-06-30'))
  self.assertLessEqual(pd.Timestamp(m['selection_uses_targets_through']),pd.Timestamp('2021-12-31'))
  self.assertTrue(m['weights_frozen']);self.assertFalse(m['production_refit'])
  self.assertEqual(m['checkpoint_sha256'],hashlib.sha256((self.path/'det_historical.pt').read_bytes()).hexdigest())
  self.assertGreater(m['best_epoch'],0)
 def test_every_target_follows_available_model_and_native_origin(self):
  f=self.rows;m=self.meta;calendar=sorted(pd.to_datetime(self.source.date).unique());indices={pd.Timestamp(value):index for index,value in enumerate(calendar)}
  self.assertTrue((pd.to_datetime(f.origin_date)>=pd.Timestamp(m['selection_lock_date'])).all())
  self.assertTrue(pd.to_datetime(f.target_date).between('2022-01-01','2023-02-10').all())
  self.assertFalse(f.duplicated(['target_date','horizon_days','zone_id','parameter']).any())
  for row in f.itertuples():
   origin,target=pd.Timestamp(row.origin_date),pd.Timestamp(row.target_date)
   self.assertEqual(indices[target]-indices[origin],{8:1,16:2,32:4}[row.horizon_days])
   self.assertEqual((target-origin).days,row.actual_elapsed_days)
  expected={8:('2022-01-01',51),16:('2022-01-09',50),32:('2022-01-25',48)}
  for horizon,(start,count) in expected.items():
   selected=f[f.horizon_days==horizon]
   self.assertEqual(selected.target_date.min(),start);self.assertEqual(selected.target_date.nunique(),count)
   self.assertEqual(selected.target_date.max(),'2023-02-10')
 def test_zone_actuals_are_real_parameter_valid_source_means(self):
  d=self.source.copy();d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
  for parameter in PARAMETERS:
   raw=pd.to_numeric(d[parameter+'_raw'],errors='coerce')
   valid=d[parameter+'_is_observed'].astype(bool)&d[parameter].notna()&raw.notna()
   if parameter=='ndwi':valid &= raw.between(-1,1)
   truth=d.assign(raw_value=raw)[valid].groupby(['date','zone_id']).raw_value.mean()
   selected=self.rows[(self.rows.parameter==parameter)&self.rows.zone_id.ne('all')]
   expected=np.array([truth.loc[(row.target_date,row.zone_id)] for row in selected.itertuples()])
   np.testing.assert_allclose(selected.actual,expected,atol=1e-10,rtol=1e-10)
 def test_scores_reproduce_exact_pairs_without_clipping_negative_r2(self):
  saved=pd.read_csv(self.path/'metrics_by_zone.csv')
  for row in saved.itertuples():
   f=self.rows[(self.rows.zone_id==row.zone_id)&(self.rows.parameter==row.parameter)&(self.rows.horizon_days==row.horizon_days)]
   error=f.prediction-f.actual
   self.assertEqual(len(f),row.n)
   self.assertAlmostEqual(error.abs().mean(),row.mae,places=10)
   self.assertAlmostEqual(np.sqrt((error**2).mean()),row.rmse,places=10)
   denom=((f.actual-f.actual.mean())**2).sum()
   if denom:self.assertAlmostEqual(1-(error**2).sum()/denom,row.r2,places=10)
  self.assertTrue((saved.r2<0).any())  # Preserve actual weak scores; never clamp them into apparent success.
  self.assertTrue(self.rows[['lower','upper']].isna().all().all())
 def test_saved_workspace_contains_same_historical_results(self):
  snapshot=json.loads((ROOT/'web'/'demo-state.json').read_text(encoding='utf-8'))
  backtest=snapshot.get('backtest',{})
  self.assertTrue(backtest.get('available'))
  self.assertEqual(backtest['metadata']['fingerprint'],self.meta['fingerprint'])
  self.assertEqual(len(backtest['rows']),len(self.rows))
  self.assertEqual(backtest['metadata']['selection_lock_date'],self.meta['selection_lock_date'])
if __name__=='__main__':unittest.main()
