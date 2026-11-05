"""Observed-step screening stays separate from safety and anomaly scores."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from aqualens.assessment import assess_water
from test_assessment import measurements, guidance

class PreliminaryAssessmentTests(unittest.TestCase):
    def setUp(self):
        catalog=patch('aqualens.assessment.load_knowledge',return_value=guidance())
        catalog.start();self.addCleanup(catalog.stop)

    def assess(self,df=None,**kwargs):
        return assess_water(measurements() if df is None else df,**kwargs)

    def by(self,result):
        return {r['parameter']:r for r in result['observations']}

    def test_rising_falling_steady_compare_previous_observed_step(self):
        df=measurements();latest=df.date.eq(df.date.max()) & df.zone_id.eq('z1')
        df.loc[latest,['turbidity_proxy','chlorophyll_proxy']]=[11.,1.]
        result=self.assess(df,zone_id='z1');rows=self.by(result)
        self.assertEqual([rows[p]['trend']['direction'] for p in ['turbidity_proxy','chlorophyll_proxy','ndwi']],['rising','falling','steady'])
        self.assertEqual(rows['turbidity_proxy']['trend']['change'],1.)
        self.assertEqual(rows['turbidity_proxy']['trend']['previous_date'][:10],'2024-04-01')
        self.assertEqual('Higher turbidity-proxy reading',result['preliminary']['title'])
        json.dumps(result,allow_nan=False)

    def test_stable_historical_status_can_have_rising_last_step(self):
        df=measurements();df.loc[df.date.eq(df.date.max()) & df.zone_id.eq('z1'),'turbidity_proxy']=10.1
        row=self.by(self.assess(df,zone_id='z1'))['turbidity_proxy']
        self.assertEqual(row['status'],'Stable');self.assertEqual(row['trend']['direction'],'rising')
        self.assertIn('separate historical',row['trend']['basis'])
        self.assertEqual(self.assess(df,zone_id='z1')['preliminary']['title'],'Higher turbidity-proxy reading')

    def test_review_headline_requires_matching_historical_flag(self):
        df=measurements();latest=df.date.eq(df.date.max()) & df.zone_id.eq('z1')
        df.loc[latest,'turbidity_proxy']=30.
        result=self.assess(df,zone_id='z1')
        self.assertEqual(self.by(result)['turbidity_proxy']['status'],'Unusual')
        self.assertEqual(result['preliminary']['title'],'Clarity-related signal needs review')
        df.loc[latest,'turbidity_proxy']=10.01
        df.loc[latest,'ndwi']=.95
        result=self.assess(df,zone_id='z1')
        self.assertEqual(self.by(result)['ndwi']['status'],'Unusual')
        self.assertEqual(result['preliminary']['title'],'Higher turbidity-proxy reading')

    def test_confidence_explains_missing_quality_and_calibration(self):
        df=measurements();df['valid_pixel_fraction']=np.nan
        confidence=self.assess(df)['preliminary']['confidence']
        self.assertEqual(confidence['label'],'Limited')
        self.assertIn('not a validated probability',confidence['meaning'])
        self.assertIn('pixel-quality',' '.join(confidence['basis']))
        self.assertIn('calibration',' '.join(confidence['basis']))

    def test_low_coverage_latest_does_not_create_change_signals(self):
        df=measurements();df.loc[df.date.eq(df.date.max()),'valid_pixel_fraction']=.1
        result=self.assess(df)
        self.assertTrue(all(r['value'] is None for r in result['observations']))
        self.assertTrue(all(r['trend']['direction']=='unavailable' for r in result['observations']))
        self.assertEqual(result['preliminary']['confidence']['label'],'Insufficient')

    def test_unobserved_previous_date_is_skipped(self):
        df=measurements();before=df.date.max()-pd.Timedelta(days=7)
        df.loc[df.date.eq(before),'turbidity_proxy_is_observed']=False
        trend=self.by(self.assess(df,zone_id='z1'))['turbidity_proxy']['trend']
        self.assertEqual(trend['previous_date'][:10],'2024-03-25')
        self.assertEqual(trend['direction'],'steady')

    def test_invalid_direct_ndwi_preserves_other_valid_history(self):
        df=measurements().drop(columns=['ndwi_is_observed']);df.loc[df.date.eq(df.date.max()),'ndwi']=1.6
        row=self.by(self.assess(df))['ndwi']
        self.assertIsNone(row['value']);self.assertEqual(row['trend']['direction'],'unavailable')
        self.assertAlmostEqual(row['previous'],.3)

    def test_absent_latest_zone_is_not_backfilled(self):
        df=measurements();df=df[~(df.date.eq(df.date.max()) & df.zone_id.eq('z2'))]
        result=self.assess(df,zone_id='z2')
        self.assertTrue(all(r['trend']['direction']=='unavailable' for r in result['observations']))
        self.assertEqual(result['preliminary']['confidence']['label'],'Insufficient')

    def test_future_values_do_not_change_cutoff_assessment(self):
        df=measurements();cutoff=df.date.max();before=self.assess(df,as_of=cutoff)
        future=df.iloc[[-1]].copy();future['date']+=pd.Timedelta(days=7)
        future[['ndwi','turbidity_proxy','chlorophyll_proxy']]=[.99,10000.,8000.]
        future['cell_id']='future_cell';future['zone_id']='future-zone';future['zone_name']='future zone'
        self.assertEqual(before,self.assess(pd.concat([df,future]),as_of=cutoff))

    def test_missing_references_withhold_interpretations_and_use_guidance(self):
        result=self.assess(knowledge=[])
        for signal in result['preliminary']['signals']:
            self.assertFalse(signal['evidence_ids']);self.assertIn('withheld',signal['interpretation'].lower())
            self.assertIn('withheld',signal['action'].lower())
        for use in result['uses']:
            self.assertFalse(use['preliminary']['safety_clearance'])
            self.assertIn('withheld',use['preliminary']['action'].lower())

    def test_low_proxy_readings_never_approve_a_physical_use(self):
        df=measurements();df.loc[df.date.eq(df.date.max()),['turbidity_proxy','chlorophyll_proxy']]=[.001,.001]
        result=self.assess(df);sources={r['source_id'] for r in result['sources']}
        for use in result['uses']:
            self.assertFalse(use['preliminary']['safety_clearance'])
            self.assertTrue(set(use['preliminary']['evidence_ids'])<=sources)
        self.assertTrue(all(u['status']=='not_established' for u in result['uses'][:-1]))
        self.assertIn('not establish',' '.join(r['interpretation'] for r in result['preliminary']['signals']).lower())

    def test_unchanged_lake_mean_preserves_opposing_zone_flags(self):
        df=measurements();latest=df.date.eq(df.date.max())
        df.loc[latest & df.zone_id.eq('z1'),'turbidity_proxy']=0.
        df.loc[latest & df.zone_id.eq('z2'),'turbidity_proxy']=30.
        result=self.assess(df);row=self.by(result)['turbidity_proxy']
        self.assertEqual(row['trend']['direction'],'steady');self.assertEqual(row['status'],'Unusual')
        self.assertIn('historical',result['preliminary']['title'])
        self.assertIn('hide different zone changes',result['preliminary']['scope_note'])

    def test_steady_readings_do_not_claim_clean_water(self):
        result=self.assess()
        self.assertIn('No step change',result['preliminary']['title'])
        self.assertTrue(all(r['trend']['direction']=='steady' for r in result['observations']))
        self.assertIn('not measured concentrations or a water-use approval',result['preliminary']['summary'])

if __name__=='__main__':unittest.main()
