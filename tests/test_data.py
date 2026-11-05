import json
import unittest
import numpy as np
import pandas as pd
from aqualens.data import load_data,load_demo,generate_demo,zones_summary

class DataTests(unittest.TestCase):
    def read(self,rows):
        return load_data(pd.DataFrame(rows).to_csv(index=False).encode())
    def row(self,date='2025-01-05',cell='A',value=.2,lat=9.65,lon=76.4):
        return dict(date=date,lake_id='lake',zone_id='Z1',zone_name='Inlet',cell_id=cell,lat=lat,lon=lon,ndwi=value,turbidity_proxy=value,chlorophyll_proxy=value)
    def test_legacy_dates_and_coordinates(self):
        csv='system:index,Chlorophyll,NDWI,Turbidity,.geo\n2020_01_01_0,1.5,-0.4,0.26,"{""type"":""Point"",""coordinates"":[76.4,9.65]}"\n'
        data,report=load_data(csv.encode())
        self.assertEqual(data.date.iloc[0],pd.Timestamp('2020-01-01'))
        self.assertAlmostEqual(data.lon.iloc[0],76.4)
        self.assertEqual(report['source'],'legacy_upload')
        self.assertTrue(data.cell_id.iloc[0].startswith('cell_'))
    def test_duplicate_first_and_anomaly_retained(self):
        data,report=self.read([self.row(value=100),self.row(value=.1)])
        self.assertEqual(report['duplicate_rows'],1)
        self.assertEqual(len(data),1)
        self.assertEqual(data.turbidity_proxy.iloc[0],100)
    def test_no_cross_cell_or_future_fill(self):
        rows=[self.row('2025-01-05','A',.2),self.row('2025-01-12','A',np.nan),self.row('2025-01-19','A',.8),self.row('2025-01-05','B',np.nan,lat=9.66),self.row('2025-01-12','B',.9,lat=9.66)]
        data,report=self.read(rows)
        a=data[data.cell_id.eq('A')].sort_values('date')
        b=data[data.cell_id.eq('B')].sort_values('date')
        self.assertEqual(a.ndwi.iloc[1],.2)
        self.assertTrue(a.ndwi_was_missing.iloc[1])
        self.assertFalse(a.is_observed.iloc[1])
        self.assertTrue(np.isnan(b.ndwi.iloc[0]))
        self.assertTrue(np.isnan(a.ndwi_raw.iloc[1]))
    def test_fill_max_two_weeks(self):
        dates=['2025-01-05','2025-01-12','2025-01-19','2025-01-26','2025-02-02']
        data,_=self.read([self.row(d,value=.2 if i==0 else np.nan) for i,d in enumerate(dates)])
        self.assertEqual(data.ndwi.iloc[1],.2)
        self.assertEqual(data.ndwi.iloc[2],.2)
        self.assertTrue(np.isnan(data.ndwi.iloc[3]))
        self.assertTrue(np.isnan(data.ndwi.iloc[4]))
        data,_=self.read([self.row('2025-01-05',value=.2),self.row('2025-02-02',value=np.nan)])
        self.assertTrue(np.isnan(data.ndwi.iloc[1]))
    def test_invalid_coordinates_and_empty_values(self):
        for rows in [[self.row(lat=91)],[self.row(lon=181)],[self.row(value=np.nan)]]:
            with self.assertRaises(ValueError):
                self.read(rows)
        with self.assertRaises(ValueError):
            load_data(b'')
        with self.assertRaises(ValueError):
            load_data(b'hello,world\n1,2\n')
    def test_stable_hierarchy_required(self):
        rows=[self.row(),self.row('2025-01-12')]
        rows[1]['zone_id']='Z2'
        with self.assertRaisesRegex(ValueError,'multiple zones'):
            self.read(rows)
        rows[1]['zone_id']='Z1'
        rows[1]['lat']=9.8
        with self.assertRaisesRegex(ValueError,'changing coordinates'):
            self.read(rows)
    def test_no_mixed_sensors(self):
        rows=[self.row(),self.row('2025-01-12')]
        rows[0]['sensor']='MODIS'
        rows[1]['sensor']='Sentinel-2'
        with self.assertRaisesRegex(ValueError,'Multiple sensors'):
            self.read(rows)
    def test_invalid_ndwi_preserved_but_not_filled(self):
        rows=[self.row('2025-01-05',value=.2),self.row('2025-01-12',value=99)]
        d,r=self.read(rows)
        self.assertTrue(np.isnan(d.ndwi.iloc[1]))
        self.assertEqual(d.ndwi_raw.iloc[1],99)
        self.assertTrue(d.ndwi_domain_invalid.iloc[1])
        self.assertFalse(d.ndwi_is_observed.iloc[1])
        self.assertTrue(d.turbidity_proxy_is_observed.iloc[1])
        self.assertEqual(r['ndwi_invalid_count'],1)
    def test_processed_csv_reload_preserves_quality_and_raw_values(self):
        d,_=self.read([self.row('2025-01-05',value=.2),self.row('2025-01-12',value=99)])
        other,_=load_data(d.to_csv(index=False).encode())
        self.assertEqual(other.ndwi_raw.tolist(),d.ndwi_raw.tolist())
        self.assertEqual(other.ndwi_is_observed.tolist(),d.ndwi_is_observed.tolist())
        self.assertTrue(np.isnan(other.ndwi.iloc[1]))
    def test_unknown_coverage_is_unknown(self):
        data,report=self.read([self.row()])
        self.assertTrue(np.isnan(data.valid_pixel_fraction.iloc[0]))
        self.assertIsNone(report['coverage'])
        json.dumps(report,allow_nan=False)
    def test_synthetic_demo(self):
        data,report=load_demo()
        self.assertEqual((report['cells'],report['zones'],report['periods']),(24,6,156))
        self.assertEqual(report['end'],'2025-12-28')
        self.assertEqual(report['source'],'synthetic_demo')
        self.assertTrue(any('SYNTHETIC' in s for s in report['warnings']))
        summary=zones_summary(data)
        self.assertEqual(len(summary),6)
        self.assertTrue(summary.date.eq(data.date.max()).all())
        self.assertTrue(summary.cells.eq(4).all())
        self.assertGreater(summary.loc[summary.zone_id.eq('Z1'),'turbidity_proxy'].iloc[0],.5)
        first,_=generate_demo()
        second,_=generate_demo()
        pd.testing.assert_frame_equal(first,second)

if __name__=='__main__':
    unittest.main()
