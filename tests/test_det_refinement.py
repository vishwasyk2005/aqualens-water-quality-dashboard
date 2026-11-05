"""Past-only refinement safeguards and exact-origin DET semantics."""
from pathlib import Path
import importlib.util
import sys,tempfile,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aqualens.backtest import prepare_backtest_panel,TRAIN_END,VALIDATION_END
from aqualens.forecast import _scaler
from aqualens.det_refinement import (ensure_past_only,causal_zone_anchors,training_target_scales,
    make_examples,metric_rows,validation_score,build_refined_model,load_refined_checkpoint,
    model_from_checkpoint,train_refinement)


def fixture():
    rows=[]
    for t,when in enumerate(pd.date_range('2020-01-01','2021-12-31',freq='8D')):
        for cell in range(4):
            rows.append(dict(date=when,lake_id='lake',zone_id='Z1' if cell<2 else 'Z2',cell_id='c'+str(cell),
                lat=9.5+cell*.01,lon=76.4+cell*.01,ndwi=.1+.1*np.sin(t/8+cell),
                turbidity_proxy=1+.2*np.sin(t/5+cell),chlorophyll_proxy=2+.3*np.cos(t/10+cell),is_observed=True))
    return pd.DataFrame(rows)


class RefinementDataTests(unittest.TestCase):
    def test_search_rejects_any_held_out_readings(self):
        data=fixture();ensure_past_only(data)
        data.loc[data.index[-1],'date']=pd.Timestamp('2022-01-01')
        with self.assertRaisesRegex(ValueError,'held-out'):ensure_past_only(data)

    def test_anchor_uses_exact_zone_target_not_equal_model_cell_weights(self):
        panel=dict(dates=pd.DatetimeIndex(['2021-01-01']),zones=['Z1'],
            targets=np.array([[[.9,9.,90.]]]),raw=np.array([[[0.,0.,0.],[1.,10.,100.]]]))
        anchor=causal_zone_anchors(panel,[0,0,0],[1,1,1])
        np.testing.assert_allclose(anchor[0,0],[.9,9,90])
        self.assertFalse(np.allclose(anchor[0,0],panel['raw'][0].mean(0)))

    def test_anchor_carry_forward_is_causal_and_expires(self):
        values=np.array([[[.1,1,2]],[[np.nan]*3],[[np.nan]*3],[[np.nan]*3],[[.8,8,80]]])
        panel=dict(dates=pd.DatetimeIndex(['2021-01-01','2021-01-09','2021-01-17','2021-01-25','2021-02-10']),zones=['Z1'],targets=values)
        original=causal_zone_anchors(panel,[0,0,0],[1,1,1])
        np.testing.assert_array_equal(original[1],original[0]);np.testing.assert_array_equal(original[2],original[0])
        np.testing.assert_array_equal(original[3],np.zeros((1,3)))
        panel['targets'][-1]=999
        changed=causal_zone_anchors(panel,[0,0,0],[1,1,1])
        np.testing.assert_array_equal(original[:-1],changed[:-1])

    def test_validation_forecast_origins_follow_final_training_observation(self):
        panel,_=prepare_backtest_panel(fixture());mean,scale=_scaler(panel,pd.Timestamp(TRAIN_END))
        _,_,target,origins,masks=make_examples(panel,mean,scale,8)
        available=panel['dates'][panel['dates']<=pd.Timestamp(TRAIN_END)][-1]
        for n,origin in enumerate(origins):
            for hi,step in enumerate([1,2,4]):
                if masks['val'][n,hi].any():
                    self.assertGreaterEqual(panel['dates'][origin],available)
                    self.assertGreater(panel['dates'][origin+step],pd.Timestamp(TRAIN_END))
                    self.assertLessEqual(panel['dates'][origin+step],pd.Timestamp(VALIDATION_END))
                if masks['train'][n,hi].any():self.assertLessEqual(panel['dates'][origin+step],pd.Timestamp(TRAIN_END))
        self.assertFalse(masks['test'].any())

    def test_scaler_temporal_scale_and_training_arrays_ignore_validation_values(self):
        original=fixture();changed=original.copy()
        changed.loc[changed.date>pd.Timestamp(TRAIN_END),['ndwi','turbidity_proxy','chlorophyll_proxy']]=[.8,99,77]
        p1,_=prepare_backtest_panel(original);p2,_=prepare_backtest_panel(changed)
        m1,s1=_scaler(p1,pd.Timestamp(TRAIN_END));m2,s2=_scaler(p2,pd.Timestamp(TRAIN_END))
        np.testing.assert_array_equal(m1,m2);np.testing.assert_array_equal(s1,s2)
        np.testing.assert_array_equal(training_target_scales(p1),training_target_scales(p2))
        x1,a1,y1,o1,mask1=make_examples(p1,m1,s1,8);x2,a2,y2,o2,mask2=make_examples(p2,m2,s2,8)
        training=mask1['train'].any(axis=(1,2,3))
        np.testing.assert_array_equal(x1[training],x2[training]);np.testing.assert_array_equal(a1[training],a2[training])
        np.testing.assert_array_equal(y1[mask1['train']],y2[mask2['train']])

    def test_each_parameter_horizon_has_equal_weight_in_validation_score(self):
        actual=np.zeros((2,3,1,3));prediction=np.ones_like(actual);mask=np.ones_like(actual,dtype=bool)
        rows=metric_rows(actual,prediction,mask,np.array([1.,2.,4.]))
        self.assertEqual(len(rows),9)
        self.assertAlmostEqual(validation_score(rows),(1+.5+.25)/3)
        self.assertTrue(all(row['n']==2 for row in rows))


@unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
class RefinementModelTests(unittest.TestCase):
    def test_zero_initialized_anchored_det_starts_at_exact_observation(self):
        import torch
        torch.manual_seed(42)
        config=dict(hidden=8,dropout=.1,mode='anchored')
        net=build_refined_model([0,0,1,1],config);net.eval()
        x=torch.randn(2,8,4,12);anchor=torch.tensor([[[.1,.2,.3],[.4,.5,.6]]]*2)
        result=net(x,anchor)
        torch.testing.assert_close(result,anchor[:,None].expand(-1,3,-1,-1))
        with self.assertRaisesRegex(ValueError,'anchors'):net(x)

    def test_direct_det_remains_distinct_from_origin_anchor(self):
        import torch
        net=build_refined_model([0,0,1,1],dict(hidden=8,dropout=0.,mode='direct'));net.eval()
        result=net(torch.randn(2,8,4,12),torch.ones(2,2,3)*100)
        torch.testing.assert_close(result,torch.zeros_like(result))

    def test_hierarchy_order_invariance_and_all_cell_gradients(self):
        import torch
        torch.manual_seed(42)
        net=build_refined_model([0,0,1,1],dict(hidden=8,dropout=.1,mode='anchored'));net.eval()
        torch.nn.init.normal_(net.head[-1].weight,std=.1)
        x=torch.randn(2,8,4,12,requires_grad=True);anchor=torch.randn(2,2,3)
        prediction=net(x,anchor)
        torch.testing.assert_close(prediction,net(x[:,:,[1,0,3,2]],anchor))
        prediction.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad.abs().sum((0,1,3))>0).all())

    def test_custom_cutoffs_are_used_throughout_training_and_checkpoint(self):
        config=dict(candidate=1,mode='direct',window=8,hidden=8,dropout=0.,lr=.001,weight_decay=.01,loss='mse')
        data=fixture();data=data[data.date<=pd.Timestamp('2021-06-30')].copy()
        with tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work'/'train-temp') as folder:
            result=train_refinement(data,folder,epochs=1,patience=1,configs=[config],train_end='2020-12-31',validation_end='2021-06-30')
            checkpoint=load_refined_checkpoint(Path(folder)/'det_refined.pt')
            self.assertEqual(result['train_end'],'2020-12-31');self.assertEqual(result['validation_end'],'2021-06-30')
            self.assertEqual(checkpoint['train_end'],'2020-12-31');self.assertEqual(checkpoint['validation_end'],'2021-06-30')
            self.assertLessEqual(pd.Timestamp(result['selection_lock_date']),pd.Timestamp('2021-06-30'))
            panel,_=prepare_backtest_panel(data,train_end='2020-12-31')
            mean,scale=_scaler(panel,pd.Timestamp('2020-12-31'))
            np.testing.assert_allclose(checkpoint['mean'],mean);np.testing.assert_allclose(checkpoint['scale'],scale)

    def test_checkpoint_contains_only_locked_past_only_model(self):
        import torch
        config=dict(candidate=1,mode='anchored',window=8,hidden=8,dropout=0.,lr=.001,weight_decay=.01,loss='mse')
        with tempfile.TemporaryDirectory(dir=ROOT.parents[1]/'work'/'train-temp') as folder:
            result=train_refinement(fixture(),folder,epochs=1,patience=1,configs=[config])
            self.assertTrue(result['checkpoint_locked']);self.assertFalse(result['held_out_2022_scored'])
            self.assertFalse(result['test_evaluation_authorized'])
            self.assertEqual(result['selected_candidate'],1)
            checkpoint=load_refined_checkpoint(Path(folder)/'det_refined.pt')
            self.assertEqual(checkpoint['model_family'],'DET');self.assertLessEqual(pd.Timestamp(checkpoint['validation_end']),pd.Timestamp('2021-12-31'))
            net=model_from_checkpoint(checkpoint)
            p,_=prepare_backtest_panel(fixture());x,a,_,_,_=make_examples(p,np.array(checkpoint['mean']),np.array(checkpoint['scale']),8)
            with torch.no_grad():prediction=net(torch.from_numpy(x[:2]),torch.from_numpy(a[:2]))
            self.assertEqual(tuple(prediction.shape),(2,3,2,3));self.assertTrue(torch.isfinite(prediction).all())

class RefinementArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json
        cls.out=ROOT/'models/refinement_v3'
        cls.summary=json.loads((cls.out/'summary.json').read_text(encoding='utf-8'))
    def test_selected_candidate_minimizes_declared_past_only_criterion(self):
        summary=self.summary;tuning=pd.read_csv(self.out/'tuning.csv')
        chosen=tuning.loc[tuning.validation_score.idxmin()]
        self.assertEqual(int(chosen.candidate),summary['selected_candidate'])
        self.assertAlmostEqual(chosen.validation_score,summary['validation_score'],places=8)
        self.assertEqual(len(tuning),6)
        self.assertFalse(summary['held_out_after_validation_scored'])
        self.assertLessEqual(pd.Timestamp(summary['validation_end']),pd.Timestamp('2021-12-31'))
        self.assertEqual(summary['data_audit']['coordinate_rounding_decimals'],6)
        self.assertEqual(summary['data_audit']['model_cells'],28)
    def test_locked_checkpoint_hash_and_validation_rows(self):
        import hashlib,json
        summary=self.summary;lock=json.loads((self.out/'selection_lock.json').read_text(encoding='utf-8'))
        sha=hashlib.sha256((self.out/'det_refined.pt').read_bytes()).hexdigest()
        self.assertEqual(sha,summary['checkpoint_sha256']);self.assertEqual(sha,lock['checkpoint_sha256'])
        self.assertEqual(lock['candidate'],summary['selected_candidate'])
        rows=pd.read_csv(self.out/'validation_predictions.csv')
        self.assertTrue((pd.to_datetime(rows.target_date)<=pd.Timestamp('2021-12-31')).all())
        self.assertTrue((pd.to_datetime(rows.target_date)>pd.Timestamp('2021-06-30')).all())
        self.assertTrue((pd.to_datetime(rows.origin_date)>=pd.Timestamp('2021-06-26')).all())
        self.assertFalse(rows.duplicated(['origin_date','target_date','zone_id','parameter','horizon_days']).any())

if __name__=='__main__':unittest.main()
