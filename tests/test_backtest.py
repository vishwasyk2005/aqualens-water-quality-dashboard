"""Scientific safeguards for the frozen historical DET experiment."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aqualens.backtest import prepare_backtest_panel,build_examples,make_test_rows,rows_metrics,TRAIN_END,VALIDATION_END
from aqualens.forecast import _scaler

def frame():
    records=[]
    for t,when in enumerate(pd.date_range('2020-01-01',periods=75,freq='16D')):
        for c in range(4):
            records.append(dict(date=when,zone_id='Z1' if c<2 else 'Z2',zone_name='Zone '+str(c//2+1),cell_id='c'+str(c),
                lake_id='lake',lat=9+c*.02,lon=76+c*.02,ndwi=.1+.2*np.sin(t/8+c),turbidity_proxy=.5+t*.01+c*.1,
                chlorophyll_proxy=1+t*.03+c*.2,is_observed=True))
    return pd.DataFrame(records)

class HistoricalBacktestTests(unittest.TestCase):
    def test_partitions_are_per_head_and_before_future_target(self):
        p,_=prepare_backtest_panel(frame());mean,scale=_scaler(p,pd.Timestamp(TRAIN_END))
        x,y,origins,masks=build_examples(p,mean,scale,8)
        for n,origin in enumerate(origins):
            for hi,step in enumerate([1,2,4]):
                target=origin+step
                if target>=len(p['dates']):
                    self.assertFalse(any(mask[n,hi].any() for mask in masks.values()));continue
                when=p['dates'][target]
                self.assertLess(p['dates'][origin],when)
                if masks['train'][n,hi].any():self.assertLessEqual(when,pd.Timestamp(TRAIN_END))
                if masks['val'][n,hi].any():self.assertGreater(when,pd.Timestamp(TRAIN_END));self.assertLessEqual(when,pd.Timestamp(VALIDATION_END))
                if masks['test'][n,hi].any():self.assertGreaterEqual(when,pd.Timestamp('2022-01-01'))

    def test_future_values_cannot_change_scaler_or_train_validation_arrays(self):
        original=frame();changed=original.copy()
        future=changed.date>=pd.Timestamp('2022-01-01')
        changed.loc[future,['ndwi','turbidity_proxy','chlorophyll_proxy']]=[.9,999,777]
        first,_=prepare_backtest_panel(original);second,_=prepare_backtest_panel(changed)
        m1,s1=_scaler(first,pd.Timestamp(TRAIN_END));m2,s2=_scaler(second,pd.Timestamp(TRAIN_END))
        np.testing.assert_array_equal(m1,m2);np.testing.assert_array_equal(s1,s2)
        x1,y1,o1,masks1=build_examples(first,m1,s1,8);x2,y2,o2,masks2=build_examples(second,m2,s2,8)
        np.testing.assert_array_equal(o1,o2)
        for key in ('train','val'):
            np.testing.assert_array_equal(masks1[key],masks2[key])
            eligible=masks1[key].any(axis=(1,2,3))
            np.testing.assert_array_equal(x1[eligible],x2[eligible])
            np.testing.assert_array_equal(y1[masks1[key]],y2[masks2[key]])

    def test_test_origins_follow_checkpoint_selection_availability(self):
        p,_=prepare_backtest_panel(frame());mean,scale=_scaler(p,pd.Timestamp(TRAIN_END))
        _,_,origins,masks=build_examples(p,mean,scale,8)
        lock=p['dates'][p['dates']<=pd.Timestamp(VALIDATION_END)][-1]
        for origin,eligible in zip(origins,masks['test']):
            if eligible.any():self.assertGreaterEqual(p['dates'][origin],lock)

    def test_future_only_cells_cannot_change_coarsening_threshold(self):
        original=frame();future=[]
        for i in range(100):
            row=original.iloc[-1].to_dict();row.update(cell_id='future'+str(i),zone_id='Z1',lat=9.01+i*.0001,lon=76.01+i*.0001)
            future.append(row)
        changed=pd.concat([original,pd.DataFrame(future)],ignore_index=True)
        first,_=prepare_backtest_panel(original);second,_=prepare_backtest_panel(changed)
        self.assertEqual(first['cells'],second['cells']);self.assertEqual(len(second['cells']),4)
        self.assertEqual(second['audit']['training_source_cells'],4)
        self.assertEqual(len(second['audit']['future_only_input_cells_ignored']),100)
        old=first['dates']<=pd.Timestamp(VALIDATION_END)
        np.testing.assert_array_equal(first['raw'][old],second['raw'][old])

    def test_last_observed_date_remains_one_step_target(self):
        p,_=prepare_backtest_panel(frame());mean,scale=_scaler(p,pd.Timestamp(TRAIN_END))
        _,_,origins,masks=build_examples(p,mean,scale,8)
        self.assertEqual(origins[-1],len(p['dates'])-2)
        self.assertTrue(masks['test'][-1,0].any())
        self.assertFalse(masks['test'][-1,1:].any())

    def test_training_geometry_ignores_future_coordinate_extremes(self):
        original=frame();changed=original.copy()
        changed.loc[changed.date>=pd.Timestamp('2022-01-01'),'lat']+=.2
        p1,_=prepare_backtest_panel(original);p2,_=prepare_backtest_panel(changed)
        self.assertEqual(p1['audit']['training_coordinate_bounds'],p2['audit']['training_coordinate_bounds'])
        self.assertEqual(p1['cells'],p2['cells'])
        self.assertGreater(p2['audit']['outside_training_coordinate_rows'],p1['audit']['outside_training_coordinate_rows'])

    def test_missing_targets_are_not_scored_or_filled(self):
        d=frame();when=d.date.max();d.loc[(d.date==when)&(d.zone_id=='Z1'),'ndwi']=np.nan
        p,_=prepare_backtest_panel(d);mean,scale=_scaler(p,pd.Timestamp(TRAIN_END))
        _,target,origins,masks=build_examples(p,mean,scale,8)
        self.assertFalse(masks['test'][-1,0,0,0]);self.assertTrue(np.isnan(target[-1,0,0,0]))
        self.assertTrue(masks['test'][-1,0,1,0])

    def test_lake_truth_and_forecast_use_identical_origin_weights(self):
        p=dict(dates=pd.date_range('2022-01-01',periods=5,freq='8D'),zones=['Z1','Z2'],targets=np.zeros((5,2,3)))
        p['targets'][1,:,0]=[.2,.8]
        predictions=np.zeros((1,3,2,3));predictions[0,0,:,0]=[.4,.6]
        mask=np.zeros((1,3,2,3),bool);mask[0,0,:,0]=True
        counts=np.ones((5,2,3));counts[0,:,0]=[3,1];counts[1,:,0]=[1,3]
        rows=make_test_rows(p,predictions,np.array([0]),mask,counts)
        lake=next(r for r in rows if r['zone_id']=='all')
        self.assertAlmostEqual(lake['actual'],.35);self.assertAlmostEqual(lake['prediction'],.45)
        self.assertAlmostEqual(lake['error'],.1);self.assertEqual(lake['zones_in_summary'],2)
        self.assertIsNone(lake['lower']);self.assertIsNone(lake['upper'])

    def test_metrics_keep_negative_r2_and_exact_pairs(self):
        rows=[dict(target_date='2022-01-01',horizon_days=8,parameter='ndwi',actual=0.,prediction=1.),dict(target_date='2022-01-09',horizon_days=8,parameter='ndwi',actual=1.,prediction=0.)]
        result=rows_metrics(rows)[0]
        self.assertEqual(result['mae'],1);self.assertEqual(result['rmse'],1);self.assertEqual(result['r2'],-3)
        self.assertEqual(result['n'],2)

class SavedHistoricalArtifactTests(unittest.TestCase):
    def test_saved_native_predictions_obey_lock_and_dates(self):
        import json
        output=ROOT/'models/historical_validation'
        summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        rows=pd.read_csv(output/'heldout_predictions.csv')
        self.assertEqual(summary['selection_lock_date'],'2021-12-27')
        self.assertTrue((rows.origin_date>='2021-12-27').all())
        self.assertTrue((rows.target_date>='2022-01-01').all())
        self.assertTrue((rows.origin_date<rows.target_date).all())
        self.assertFalse(rows.duplicated(['target_date','horizon_days','zone_id','parameter']).any())
        self.assertEqual(rows.groupby('horizon_days').target_date.nunique().to_dict(),{8:51,16:50,32:48})
        self.assertEqual(rows.groupby('horizon_days').target_date.min().to_dict(),{8:'2022-01-01',16:'2022-01-09',32:'2022-01-25'})
        self.assertTrue(rows.lower.isna().all());self.assertTrue(rows.upper.isna().all())

    def test_saved_checkpoint_identity_matches_summary(self):
        import hashlib,json
        output=ROOT/'models/historical_validation'
        summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['checkpoint_sha256'],hashlib.sha256((output/'det_historical.pt').read_bytes()).hexdigest())
        self.assertTrue(summary['weights_frozen']);self.assertFalse(summary['production_refit'])
        self.assertLess(summary['train_end'],summary['validation_start'])
        self.assertLess(summary['validation_end'],summary['test_start'])

if __name__=='__main__':unittest.main()
