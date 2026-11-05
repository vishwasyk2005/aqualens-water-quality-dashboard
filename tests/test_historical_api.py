"""Historical-validation artifact isolation, chronology and exact-pair API tests."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server import Dashboard,APIError
from aqualens.forecast import dataset_fingerprint


class HistoricalAPITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work'/'train-temp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.directory=self.root/'models'/'historical_validation'
        self.directory.mkdir(parents=True)
        records=[]
        for when in ['2021-06-24','2021-12-08','2021-12-24','2022-01-01','2022-01-09','2022-01-17','2022-01-25']:
            for zone in ['Z1','Z2']:
                records.append(dict(date=pd.Timestamp(when),lake_id='lake',zone_id=zone,zone_name=zone,
                    cell_id=zone+'-cell',lat=9.5,lon=76.4,ndwi=.2,turbidity_proxy=1.,chlorophyll_proxy=2.,is_observed=True))
        self.frame=pd.DataFrame(records)
        self.app=Dashboard.__new__(Dashboard)
        self.app.root=self.root
        self.app.modis=(self.frame,dict(end='2022-01-25'))
        self.app.demo=self.app.modis
        self.app.datasets=OrderedDict()
        self.app.lock=threading.RLock()
        self.app.historical_cache=OrderedDict()
        self.checkpoint=self.directory/'det_historical.pt'
        self.checkpoint.write_bytes(b'separate-pre-2022-evaluation-weights')
        self.summary=dict(fingerprint=dataset_fingerprint(self.frame),train_end='2021-06-24',validation_end='2021-12-24',
            test_start='2022-01-01',test_end='2022-01-25',selection_lock_date='2021-12-24',config=dict(window=8,hidden=32,lr=.001),best_epoch=10,
            checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.rows=[]
        for zone in ['Z1','Z2','all']:
            for parameter in ['ndwi','turbidity_proxy','chlorophyll_proxy']:
                for origin,target,horizon,elapsed,actual,prediction in [
                    ('2021-12-24','2022-01-01',8,8,0.,0.),
                    ('2022-01-01','2022-01-09',8,8,.2,.1),
                    ('2021-12-24','2022-01-09',16,16,.2,.1),
                    ('2021-12-24','2022-01-25',32,32,.2,.1)]:
                    self.rows.append(dict(origin_date=origin,target_date=target,horizon_days=horizon,actual_elapsed_days=elapsed,
                        zone_id=zone,parameter=parameter,actual=actual,prediction=prediction,lower=99,upper=100))
        self.save()

    def save(self):
        (self.directory/'summary.json').write_text(json.dumps(self.summary),encoding='utf-8')
        pd.DataFrame(self.rows).to_csv(self.directory/'heldout_predictions.csv',index=False)

    def test_saved_pairs_are_loaded_without_torch_or_forecasting(self):
        with patch.object(self.app,'has_torch',side_effect=AssertionError('Historical plots need no Torch')), patch('server.forecast_zones',side_effect=AssertionError('Never use production weights')):
            result=self.app.backtest()
        self.assertTrue(result['available'])
        self.assertEqual(len(result['rows']),36)
        self.assertEqual(result['available_zones'],['all','Z1','Z2'])
        self.assertEqual(result['default_zone'],'all')
        first=next(r for r in result['rows'] if r['target_date']=='2022-01-01')
        self.assertEqual(first['actual'],0.)
        self.assertEqual(first['prediction'],0.)
        self.assertEqual(first['error'],0.)
        self.assertIsNone(first['lower'])
        self.assertIsNone(first['upper'])
        json.dumps(result,allow_nan=False)

    def test_api_filters_keep_matching_target_and_origin_pairs(self):
        result=self.app.get('/api/backtest',{'zone':['all'],'horizon_days':['16'],'parameter':['ndwi']})
        self.assertTrue(result['available'])
        self.assertEqual(len(result['rows']),1)
        self.assertEqual(result['rows'][0]['target_date'],'2022-01-09')
        self.assertEqual(result['rows'][0]['origin_date'],'2021-12-24')
        self.assertEqual(result['rows'][0]['actual_elapsed_days'],16)
        self.assertEqual(result['metrics'][0]['n'],1)
        self.assertIsNone(result['metrics'][0]['r2'])

    def test_metrics_exclude_whole_lake_duplicates_from_pooled_zone_score(self):
        result=self.app.backtest()
        metric=next(r for r in result['metrics'] if r['parameter']=='ndwi' and r['horizon_days']==8)
        self.assertEqual(metric['n'],4)
        self.assertAlmostEqual(metric['mae'],.05)
        self.assertAlmostEqual(metric['rmse'],.1*2**-.5)
        self.assertEqual(metric['r2'],.5)
        lake=next(r for r in result['metrics_by_zone'] if r['zone_id']=='all' and r['parameter']=='ndwi' and r['horizon_days']==8)
        self.assertEqual(lake['n'],2)

    def test_modified_or_unknown_upload_cannot_reuse_results(self):
        changed=self.frame.copy()
        changed.loc[0,'turbidity_proxy']+=.01
        self.app.datasets['changed']=(changed,{})
        result=self.app.backtest('changed')
        self.assertFalse(result['available'])
        self.assertEqual(result['rows'],[])
        self.assertIn('different observations',result['reason'])
        self.app.datasets['same']=(self.frame.copy(),{})
        self.assertTrue(self.app.backtest('same')['available'])
        with self.assertRaises(APIError):self.app.backtest('unknown')

    def test_production_weights_never_substitute_for_missing_historical_experiment(self):
        (self.root/'models'/'det_v2.pt').write_bytes(b'production')
        self.checkpoint.unlink()
        result=self.app.backtest()
        self.assertFalse(result['available'])
        self.assertEqual(result['rows'],[])
        self.assertIn('separate pre-2022',result['reason'])

    def test_post_2021_training_or_selection_is_rejected(self):
        for key,value in [('train_end','2022-01-01'),('validation_end','2022-01-01'),('test_start','2021-12-24')]:
            with self.subTest(key=key):
                original=self.summary[key]
                self.summary[key]=value
                self.save()
                result=self.app.backtest()
                self.assertFalse(result['available'])
                self.assertEqual(result['rows'],[])
                self.summary[key]=original

    def test_checkpoint_hash_and_prediction_chronology_are_audited(self):
        self.checkpoint.write_bytes(b'wrong weights')
        self.assertFalse(self.app.backtest()['available'])
        self.checkpoint.write_bytes(b'separate-pre-2022-evaluation-weights')
        for field,value in [('origin_date','2022-01-01'),('actual_elapsed_days',9),('target_date','2021-12-24')]:
            with self.subTest(field=field):
                original=self.rows[0][field]
                self.rows[0][field]=value
                self.save()
                result=self.app.backtest()
                self.assertFalse(result['available'])
                self.assertIn('invalid',result['reason'])
                self.rows[0][field]=original

    def test_changed_artifacts_invalidate_the_result_cache(self):
        before=self.app.backtest(zone='Z1',horizon_days=8,parameter='ndwi')
        self.rows[0]['prediction']=.5
        self.save()
        after=self.app.backtest(zone='Z1',horizon_days=8,parameter='ndwi')
        self.assertEqual(before['rows'][0]['prediction'],0.)
        self.assertEqual(after['rows'][0]['prediction'],.5)

    def test_prediction_cannot_use_a_model_selected_after_its_origin(self):
        # This origin and elapsed day count are valid calendar values, but model
        # selection had not finished on 8 December.
        self.rows[3].update(origin_date='2021-12-08',target_date='2022-01-01',actual_elapsed_days=24)
        self.save()
        result=self.app.backtest()
        self.assertFalse(result['available'])
        self.assertIn('selection was available',result['reason'])

    def test_native_step_horizon_must_match_real_observation_calendar(self):
        # Relabeling an 8-day/one-observation pair as 16-day/two-observation is invalid.
        self.rows[0]['horizon_days']=16
        self.save()
        result=self.app.backtest()
        self.assertFalse(result['available'])
        self.assertIn('native observation-step',result['reason'])

    def test_selection_lock_is_derived_and_cannot_be_misdeclared(self):
        del self.summary['selection_lock_date']
        self.save()
        self.assertEqual(self.app.backtest()['metadata']['selection_lock_date'],'2021-12-24')
        self.summary['selection_lock_date']='2021-12-08'
        self.save()
        self.assertFalse(self.app.backtest()['available'])

    def test_duplicate_target_horizon_zone_parameter_is_rejected(self):
        self.rows.append(dict(self.rows[0]))
        self.save()
        result=self.app.backtest()
        self.assertFalse(result['available'])
        self.assertIn('Duplicate',result['reason'])

    def test_invalid_selection_is_an_api_error(self):
        for horizon in [0,7,33,True,8.,'8']:
            with self.assertRaises(APIError):self.app.backtest(horizon_days=horizon)
        with self.assertRaises(APIError):self.app.backtest(zone='Z99')
        with self.assertRaises(APIError):self.app.backtest(parameter='oxygen')
        with self.assertRaises(APIError):self.app.get('/api/backtest',{'horizon_days':['no']})


if __name__=='__main__':unittest.main()
