"""End-to-end contract checks for current DET v3 training and inference.

The fixture is synthetic and outside the historical experiment's calendar. One
small, predeclared DET configuration and one epoch exercise integration only;
these tests do not search real data or assess 2022--2023 model accuracy.
"""
from contextlib import redirect_stdout
from pathlib import Path
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aqualens.forecast import PARAMETERS,forecast_zones,split_dates,dataset_fingerprint
from aqualens.refined_pipeline import train_current_det
from aqualens.det_refinement import CONFIGS


def synthetic_observations():
    records=[]
    for index,when in enumerate(pd.date_range('2024-01-01',periods=48,freq='8D')):
        for cell in range(4):
            zone='Z1' if cell<2 else 'Z2'
            records.append(dict(date=when,lake_id='integration-fixture',zone_id=zone,zone_name=zone,
                cell_id='cell'+str(cell),lat=9.2+cell*.01,lon=76.2+cell*.01,
                ndwi=.15+.18*np.sin(index/5)+cell*.01,
                turbidity_proxy=.8+.2*np.cos(index/7)+cell*.03,
                chlorophyll_proxy=2.+.4*np.sin(index/9)+cell*.1,is_observed=True))
    return pd.DataFrame(records)


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
class RefinedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)
        cls.frame=synthetic_observations()
        cls.temporary=tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work'/'train-temp')
        cls.output=Path(cls.temporary.name)
        configuration=dict(CONFIGS[0],candidate=1,window=4,hidden=8,dropout=0.)
        cls.progress=[]
        with patch('aqualens.refined_pipeline.CONFIGS',[configuration]),redirect_stdout(io.StringIO()):
            cls.summary=train_current_det(cls.frame,cls.output,epochs=1,progress_callback=cls.progress.append)
        cls.checkpoint=cls.output/'det_v2.pt'
        cls.production=torch.load(cls.checkpoint,map_location='cpu',weights_only=True)
        cls.evaluation=torch.load(cls.output/'det_evaluation_v2.pt',map_location='cpu',weights_only=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'temporary'):cls.temporary.cleanup()

    def temporary_checkpoint(self,checkpoint,name):
        import torch
        path=self.output/name
        torch.save(checkpoint,path)
        return path

    def test_generic_training_uses_dataset_cutoffs_and_separate_fit_partitions(self):
        cut=split_dates(self.frame)
        expected_train=str(cut['train_end'].date())
        expected_validation=str(cut['validation_end'].date())
        self.assertEqual(self.summary['version'],3)
        self.assertEqual(self.summary['train_end'],expected_train)
        self.assertEqual(self.summary['validation_end'],expected_validation)
        self.assertNotEqual(self.summary['train_end'],'2021-06-30')
        self.assertEqual(self.evaluation['train_end'],expected_train)
        self.assertEqual(self.evaluation['validation_end'],expected_validation)
        self.assertEqual(self.production['geometry_fit_through'],str(self.frame.date.max().date()))
        self.assertIn('Train-only',self.evaluation['fit_partition'])
        self.assertIn('all-history',self.production['fit_partition'])
        self.assertEqual(self.production['format_version'],3)
        self.assertEqual(self.production['model_family'],'DET')
        self.assertEqual(self.production['fingerprint'],dataset_fingerprint(self.frame))
        self.assertNotEqual((self.output/'det_evaluation_v2.pt').read_bytes(),self.checkpoint.read_bytes())
        self.assertTrue((self.output/'selection_v3'/'det_refined.pt').is_file())
        plan=json.loads((self.output/'selection_v3'/'search_plan.json').read_text(encoding='utf-8'))
        self.assertFalse(plan['held_out_after_validation_loaded'])
        self.assertFalse(plan['held_out_after_validation_scored'])

    def test_saved_summary_supports_dashboard_metrics_history_and_audit(self):
        saved=json.loads((self.output/'evaluation.json').read_text(encoding='utf-8'))
        for key in ('config','best_epoch','history','metrics','calibration','production_refit','refinement','data_audit'):
            self.assertIn(key,saved)
        self.assertEqual({row['model'] for row in saved['metrics']},{'det'})
        self.assertEqual({row['horizon_days'] for row in saved['metrics']},{8,16,32})
        self.assertEqual({row['parameter'] for row in saved['metrics']},set(PARAMETERS))
        self.assertEqual(saved['production_refit']['observations_through'],str(self.frame.date.max().date()))
        self.assertGreater(saved['best_epoch'],0)
        self.assertTrue(saved['history']['det'])
        self.assertIn('validation_score',saved['history']['det'][0])
        self.assertTrue(any(event.get('stage')=='production_refit' for event in self.progress))
        pairs=pd.read_csv(self.output/'heldout_predictions.csv')
        self.assertTrue(pd.to_datetime(pairs.origin_date).ge(pd.Timestamp(saved['validation_end'])).all())
        self.assertTrue(pd.to_datetime(pairs.target_date).gt(pd.to_datetime(pairs.origin_date)).all())
        self.assertEqual(np.asarray(saved['calibration']['absolute_error_radius']).shape,(3,3))

    def test_forecasts_use_saved_v3_weights_and_experimental_long_range_contract(self):
        for horizon in (30,40):
            rows=forecast_zones(self.frame,model='det',horizon_days=horizon,checkpoint_path=self.checkpoint)
            self.assertEqual(len(rows),horizon*2*3)
            self.assertEqual(rows.date.min(),self.frame.date.max()+pd.Timedelta(days=1))
            self.assertEqual(rows.date.max(),self.frame.date.max()+pd.Timedelta(days=horizon))
            self.assertTrue(np.isfinite(rows.prediction).all())
            self.assertEqual(set(rows.model),{'det'})
            direct=rows[rows.horizon_days<=32]
            self.assertTrue(direct[['lower','upper']].notna().all().all())
            self.assertTrue((direct.lower<=direct.prediction).all())
            self.assertTrue((direct.prediction<=direct.upper).all())
            extended=rows[rows.horizon_days>32]
            self.assertTrue(extended[['lower','upper']].isna().all().all())
            self.assertTrue(extended.is_extrapolation.all())

    def test_zero_residual_daily_curve_is_anchored_at_latest_observed_zone_mean(self):
        import torch
        checkpoint=dict(self.production,config=dict(self.production['config'],mode='anchored'))
        checkpoint['state']={key:value.clone() for key,value in self.production['state'].items()}
        checkpoint['state']['head.3.weight'].zero_()
        checkpoint['state']['head.3.bias'].zero_()
        path=self.temporary_checkpoint(checkpoint,'zero-residual.pt')
        rows=forecast_zones(self.frame,horizon_days=8,checkpoint_path=path)
        latest=self.frame[self.frame.date.eq(self.frame.date.max())].groupby('zone_id')[PARAMETERS].mean()
        for zone in latest.index:
            for parameter in PARAMETERS:
                values=rows[(rows.zone_id==zone)&(rows.parameter==parameter)].prediction.to_numpy()
                np.testing.assert_allclose(values,latest.loc[zone,parameter],atol=1e-6,rtol=1e-6)

    def test_ndwi_intervals_remain_ordered_when_raw_outputs_exceed_index_domain(self):
        for direction in (-1.,1.):
            checkpoint=dict(self.production,config=dict(self.production['config'],mode='direct'))
            checkpoint['state']={key:value.clone() for key,value in self.production['state'].items()}
            checkpoint['state']['head.3.weight'].zero_()
            checkpoint['state']['head.3.bias'].zero_()
            checkpoint['state']['head.3.bias'][[0,3,6]]=direction*100.
            checkpoint['calibration']=dict(self.production['calibration'],absolute_error_radius=np.full((3,3),.01).tolist())
            path=self.temporary_checkpoint(checkpoint,'outside-domain-'+str(int(direction))+'.pt')
            rows=forecast_zones(self.frame,horizon_days=30,checkpoint_path=path)
            index=rows[rows.parameter.eq('ndwi')]
            self.assertTrue(index.prediction.between(-1,1).all())
            self.assertTrue(index.lower.between(-1,1).all())
            self.assertTrue(index.upper.between(-1,1).all())
            self.assertTrue((index.lower<=index.prediction).all())
            self.assertTrue((index.prediction<=index.upper).all())

    def test_incompatible_or_missing_checkpoint_and_other_models_never_fall_back(self):
        changed=self.frame.copy()
        changed.loc[changed.index[0],'ndwi']+=.01
        with patch('aqualens.refined_pipeline.forecast_refined',side_effect=AssertionError('Rejected input must not reach inference')):
            with self.assertRaisesRegex(ValueError,'different dataset'):
                forecast_zones(changed,horizon_days=30,checkpoint_path=self.checkpoint)
            with self.assertRaisesRegex(ValueError,'Train DET'):
                forecast_zones(self.frame,horizon_days=30,checkpoint_path=self.output/'missing.pt')
            for other in ('ridge','prophet','mean_pool','persistence'):
                with self.assertRaisesRegex(ValueError,'only supported'):
                    forecast_zones(self.frame,model=other,horizon_days=30,checkpoint_path=self.checkpoint)


if __name__=='__main__':
    unittest.main()
