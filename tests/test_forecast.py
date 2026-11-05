"""Scientific behavior checks for DET chronology, masks, reload and horizons."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from aqualens.forecast import PARAMETERS, _panel, _scaler, _examples, split_dates, dataset_fingerprint, train_models, forecast_zones

def fixture(periods=64):
    rows=[]
    for t,date in enumerate(pd.date_range('2020-01-01',periods=periods,freq='8D')):
        for z in range(2):
            for c in range(2):
                rows.append({'date':date,'lake_id':'test','zone_id':f'z{z}','cell_id':f'z{z}c{c}',
                 'lat':12+z*.01,'lon':77+c*.01,'ndwi':.2+.05*np.sin(t/5)+z*.01,
                 'turbidity_proxy':.5+.02*t+c*.01,'chlorophyll_proxy':1+.1*np.cos(t/4)+z*.1,'is_observed':True})
    return pd.DataFrame(rows)

class ForecastTests(unittest.TestCase):
    def test_scaler_and_training_windows_ignore_future(self):
        df=fixture(); cut=split_dates(df); p=_panel(df); m,s=_scaler(p,cut['train_end'])
        altered=df.copy(); altered.loc[altered.date>cut['train_end'],'turbidity_proxy']=100000
        q=_panel(altered); mm,ss=_scaler(q,cut['train_end']); np.testing.assert_array_equal(m,mm); np.testing.assert_array_equal(s,ss)
        x,y,origins=_examples(p,m,s); xx,yy,other=_examples(q,mm,ss)
        train=p['dates'][origins+4]<=cut['train_end']
        np.testing.assert_array_equal(x[train],xx[train]); np.testing.assert_array_equal(y[train],yy[train])
        self.assertTrue((p['dates'][origins[train]+4]<=cut['train_end']).all())
    def test_targets_not_filled_or_invalid_ndwi(self):
        df=fixture(); latest=df.date.max(); selected=(df.date==latest)&(df.zone_id=='z0')
        df.loc[selected,PARAMETERS]=np.nan; p=_panel(df)
        self.assertTrue(np.isnan(p['targets'][-1,0]).all()); self.assertTrue(np.isfinite(p['filled'][-1,0]).all())
        df.loc[selected,'ndwi']=100; p=_panel(df)
        self.assertTrue(np.isnan(p['targets'][-1,0,0])); self.assertEqual(p['audit']['invalid_ndwi_excluded'],2)
    def test_coverage_exclusion_and_no_backward_fill(self):
        df=fixture(); df['valid_pixel_fraction']=np.nan; first=df.date.min()
        df.loc[(df.date==first)&(df.cell_id=='z0c0'),PARAMETERS]=np.nan
        df.loc[(df.date==df.date.max())&(df.zone_id=='z0'),'valid_pixel_fraction']=.1
        p=_panel(df); self.assertTrue(np.isnan(p['filled'][0,0]).all()); self.assertTrue(np.isnan(p['targets'][-1,0]).all())
    def test_fingerprint(self):
        df=fixture(); self.assertEqual(dataset_fingerprint(df),dataset_fingerprint(df.sample(frac=1,random_state=7)))
        other=df.copy();other.loc[0,'ndwi']+=.01
        self.assertNotEqual(dataset_fingerprint(df),dataset_fingerprint(other))
        moved=df.copy(); moved.loc[0,'lat']+=.001
        self.assertNotEqual(dataset_fingerprint(df),dataset_fingerprint(moved))
        observed=df.copy(); observed['ndwi_is_observed']=True
        hidden=observed.copy(); hidden.loc[0,'ndwi_is_observed']=False
        self.assertNotEqual(dataset_fingerprint(observed),dataset_fingerprint(hidden))
    def test_only_det_and_horizon_limits(self):
        for mode in ['mean_pool','tree','prophet','ridge','persistence']:
            with self.assertRaisesRegex(ValueError,'only supported'): forecast_zones(fixture(),model=mode)
        with self.assertRaises(ValueError): forecast_zones(fixture(),horizon_days=731)
        with self.assertRaises(ValueError): split_dates(fixture(8))
    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
    def test_train_reload_extrapolation_and_saved_audit(self):
        df=fixture()
        with tempfile.TemporaryDirectory() as folder:
            summary=train_models(df,folder,epochs=2); ck=Path(folder)/'det_v2.pt'
            self.assertEqual({r['model'] for r in summary['metrics']},{'det'})
            self.assertEqual({r['horizon_days'] for r in summary['metrics']},{8,16,32})
            self.assertTrue((Path(folder)/'det_evaluation_v2.pt').exists())
            result=forecast_zones(df,horizon_days=40,checkpoint_path=ck)
            self.assertEqual(len(result),40*2*3); self.assertEqual(result.date.min(),df.date.max()+pd.Timedelta(days=1))
            self.assertTrue(np.isfinite(result.prediction).all())
            self.assertTrue(result.loc[result.horizon_days>32,['lower','upper']].isna().all().all())
            self.assertTrue(result.loc[result.horizon_days>32,'is_extrapolation'].all())
            changed=df.copy(); changed.loc[0,'ndwi']+=.1
            with self.assertRaisesRegex(ValueError,'different dataset'): forecast_zones(changed,checkpoint_path=ck)
    @unittest.skipUnless(importlib.util.find_spec('torch'),'torch unavailable')
    def test_child_order_invariance_and_gradients(self):
        import torch
        from aqualens.tree_model import HierarchicalForecaster
        torch.manual_seed(7); net=HierarchicalForecaster([0,0,1,1]); x=torch.randn(2,8,4,12,requires_grad=True)
        output=net(x); torch.testing.assert_close(output,net(x[:,:,[1,0,3,2]]))
        output.square().sum().backward(); self.assertTrue(torch.isfinite(x.grad).all()); self.assertTrue((x.grad.abs().sum((0,1,3))>0).all())
        self.assertFalse(hasattr(net,'mean_pool'))
if __name__=='__main__': unittest.main()


