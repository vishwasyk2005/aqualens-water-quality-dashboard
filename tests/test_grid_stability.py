"""Regressions for stable geographic binning of repeated pre-2022 sites.

Six decimal degrees is a numerical canonicalization policy, not a claim about
sensor resolution or measurement accuracy. Sub-nanodegree representation noise
must not create new model cells; genuine cross-bin movement must still count.
"""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from aqualens.backtest import prepare_backtest_panel,TRAIN_END
from aqualens.forecast import _scaler,_features


def stable_sites():
    records=[]
    # The dense site lattice includes 6x6-bin boundaries and exceeds the
    # spatial-coarsening threshold. Every observation precedes 2022.
    dates=pd.date_range('2020-01-01',periods=40,freq='16D')
    for time,when in enumerate(dates):
        for latitude_index in range(13):
            for longitude_index in range(9):
                site=latitude_index*9+longitude_index
                records.append(dict(date=when,lake_id='synthetic-grid-lake',
                    zone_id='Z1' if latitude_index<6 else 'Z2',
                    zone_name='Fixed geographic zone',
                    cell_id=f'P_{latitude_index:02}_{longitude_index:02}',
                    lat=9.+latitude_index*.01,lon=76.+longitude_index*.01,
                    ndwi=.1+.001*site+.0001*time,
                    turbidity_proxy=.5+.02*(site%7)+.001*time,
                    chlorophyll_proxy=1.+.1*(site%11)+.003*time,
                    is_observed=True))
    return pd.DataFrame(records)


class GeographicGridStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=stable_sites()

    def assert_same_inputs(self,original,changed):
        first,mapped_first=prepare_backtest_panel(original)
        second,mapped_second=prepare_backtest_panel(changed)
        pd.testing.assert_series_equal(mapped_first.cell_id,mapped_second.cell_id)
        self.assertEqual(first['cells'],second['cells'])
        self.assertEqual(first['cell_zone'],second['cell_zone'])
        self.assertEqual(first['zones'],second['zones'])
        self.assertEqual(first['audit']['training_coordinate_bounds'],second['audit']['training_coordinate_bounds'])
        for name in ('raw','filled','age','targets'):
            np.testing.assert_array_equal(first[name],second[name],err_msg=name)
        mean1,scale1=_scaler(first,pd.Timestamp(TRAIN_END))
        mean2,scale2=_scaler(second,pd.Timestamp(TRAIN_END))
        np.testing.assert_array_equal(mean1,mean2)
        np.testing.assert_array_equal(scale1,scale2)
        np.testing.assert_array_equal(_features(first,mean1,scale1),_features(second,mean2,scale2))

    def test_microjitter_in_validation_does_not_reassign_fixed_sites(self):
        original=self.source.copy(deep=True)
        changed=original.copy(deep=True)
        future=changed.date>pd.Timestamp(TRAIN_END)
        signs=np.where(np.arange(len(changed))%2,1.,-1.)
        changed.loc[future,'lat']+=signs[future]*1e-10
        changed.loc[future,'lon']-=signs[future]*1e-10
        self.assertGreater(original.cell_id.nunique(),96)
        self.assertLess(changed.date.max(),pd.Timestamp('2022-01-01'))
        before=changed.copy(deep=True)
        self.assert_same_inputs(original,changed)
        pd.testing.assert_frame_equal(changed,before)
        pd.testing.assert_frame_equal(original,self.source)

    def test_microjitter_in_training_bounds_preserves_the_same_grid(self):
        changed=self.source.copy(deep=True)
        signs=np.where(np.arange(len(changed))%3,1.,-1.)
        changed['lat']+=signs*1e-10
        changed['lon']-=signs*1e-10
        self.assert_same_inputs(self.source,changed)

    def test_real_cross_bin_movement_changes_input_location_not_target_values(self):
        original=self.source.copy(deep=True)
        changed=original.copy(deep=True)
        moved=(changed.date>pd.Timestamp(TRAIN_END))&changed.cell_id.eq('P_02_02')
        self.assertTrue(moved.any())
        changed.loc[moved,'lon']+=.02
        first,mapped_first=prepare_backtest_panel(original)
        second,mapped_second=prepare_backtest_panel(changed)
        self.assertTrue(mapped_first.loc[moved,'cell_id'].ne(mapped_second.loc[moved,'cell_id']).all())
        self.assertEqual(first['cells'],second['cells'])
        self.assertEqual(first['cell_zone'],second['cell_zone'])
        np.testing.assert_array_equal(first['targets'],second['targets'])
        training=first['dates']<=pd.Timestamp(TRAIN_END)
        np.testing.assert_array_equal(first['raw'][training],second['raw'][training])
        self.assertFalse(np.array_equal(first['raw'][~training],second['raw'][~training],equal_nan=True))


if __name__=='__main__':
    unittest.main()
