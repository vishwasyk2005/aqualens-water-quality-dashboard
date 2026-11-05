"""Behavioral checks for alert chronology, data quality and evidence grounding."""
import json
import unittest
import numpy as np
import pandas as pd
from aqualens.alerts import detect_alerts, current_condition
from aqualens.evidence import retrieve_evidence, explain_alert, parse_knowledge, evaluate_retrieval


def observations(periods=15):
    dates = pd.date_range('2024-01-01', periods=periods, freq='7D')
    return pd.DataFrame([dict(date=date, lake_id='lake', zone_id='z1', zone_name='Inlet', cell_id='c1', lat=12.9, lon=77.5, ndwi=.4, turbidity_proxy=10., chlorophyll_proxy=2., rainfall_mm=5., temp_c=24., valid_pixel_fraction=.9, is_observed=True, sensor='synthetic', source_kind='synthetic') for date in dates])


class AlertEvidenceTests(unittest.TestCase):
    def test_use_guidance_is_not_retrieved_as_optical_cause_evidence(self):
        df=observations()
        df['sensor']='MODIS'
        df['valid_pixel_fraction']=np.nan
        alert=detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0].to_dict()
        result=explain_alert(alert,df)
        source_ids={row['source_id'] for row in result['sources']}
        self.assertIn('gee-modis-quality',source_ids)
        self.assertFalse(source_ids & {'who-drinking-assessment','cpcb-designated-uses','fao-irrigation-quality','cdc-bloom-precautions'})

    def test_current_spike_not_in_own_baseline(self):
        df = observations()
        df.loc[df.index[-1], 'turbidity_proxy'] = 30.
        alert = detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0]
        self.assertEqual(alert.baseline, 10.)
        self.assertEqual(alert.severity, 'Critical')
        self.assertGreater(alert.score, 4.)

    def test_low_coverage_missing_and_imputed_are_withheld(self):
        for field, value in [('valid_pixel_fraction', .1), ('is_observed', False), ('turbidity_proxy', np.nan)]:
            df = observations()
            df.loc[df.index[-1], field] = value
            row = detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0]
            self.assertEqual(row.severity, 'Insufficient data')
            self.assertTrue(np.isnan(row.score))

    def test_insufficient_history(self):
        self.assertTrue(detect_alerts(observations(4)).severity.eq('Insufficient data').all())

    def test_stale_zone_is_not_normal(self):
        df = observations()
        other = df.iloc[:-1].copy()
        other['zone_id'] = 'z2'
        other['zone_name'] = 'Old zone'
        rows = detect_alerts(pd.concat([df, other]))
        self.assertTrue(rows[rows.zone_id.eq('z2')].severity.eq('Insufficient data').all())

    def test_ndwi_change_does_not_establish_pollution(self):
        df = observations()
        df.loc[df.index[-1], 'ndwi'] = .8
        row = detect_alerts(df).query("parameter == 'ndwi'").iloc[0]
        self.assertIn('does not establish pollution', row.reason)

    def test_unknown_quality_keeps_observed_condition_with_qualification(self):
        df=observations()
        df['valid_pixel_fraction']=np.nan
        df.loc[df.index[-1],'turbidity_proxy']=30.
        row=detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0]
        self.assertEqual(row.status,'Unusual')
        self.assertTrue(row.coverage_unknown)
        self.assertTrue(np.isnan(row.coverage))
        self.assertIn('unverified',row.reason)
        condition=current_condition(df)
        self.assertEqual(condition['as_of'][:10],'2024-04-08')
        self.assertTrue(condition['coverage_unknown'])
        entry=next(r for r in condition['parameters'] if r['parameter']=='turbidity_proxy')
        self.assertEqual(entry['value'],30.)
        self.assertEqual(entry['change'],20.)

    def test_changing_spatial_samples_not_confused_with_missing_readings(self):
        df=observations()
        # A different coordinate is sampled at each date: available measurements
        # are complete even though each snapshot is a small historical subset.
        df['cell_id']=['c'+str(i%3) for i in range(len(df))]
        df['valid_pixel_fraction']=np.nan
        rows=detect_alerts(df)
        self.assertTrue(rows.status.eq('Stable').all())
        self.assertTrue(rows.observation_fraction.eq(1).all())
        self.assertTrue(rows.spatial_fraction.lt(.5).all())
        self.assertTrue(rows.reason.str.contains('changing spatial sampling').all())

    def test_parameter_observed_flags_control_condition(self):
        df=observations()
        df['ndwi_is_observed']=True
        df.loc[df.index[-1],'ndwi_is_observed']=False
        row=detect_alerts(df).query("parameter == 'ndwi'").iloc[0]
        self.assertEqual(row.status,'Insufficient data')
        self.assertTrue(np.isnan(row.value))

    def test_forecast_context_is_det_output_not_retrieval_prediction(self):
        df=observations();df['rainfall_mm']=np.nan;df['valid_pixel_fraction']=np.nan
        row=detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0].to_dict()
        future=dict(available=True,prediction=12.345,date='2024-05-01',model='det',is_extrapolation=True)
        result=explain_alert(row,df,forecast_summary=future)
        self.assertEqual(result['forecast_summary'],future)
        self.assertIn('12.3450',result['summary'])
        self.assertIn('experimental',result['summary'])
        self.assertEqual(result['hypotheses'][0]['status'],'Insufficient local support')
        self.assertIn('unavailable',result['hypotheses'][0]['support'][0])
        self.assertTrue(result['sources'])

    def test_retrieval_filter_and_no_overlap(self):
        hits = retrieve_evidence('cloud shadow', parameter='ndwi', sensor='Sentinel-2')
        self.assertEqual([x['source_id'] for x in hits], ['gee-sentinel-quality'])
        self.assertEqual(retrieve_evidence('quantum cryptocurrency'), [])
        self.assertEqual(retrieve_evidence('rainfall', parameter='unknown'), [])
        knowledge = [dict(source_id='one', title='Rain', url='https://example.org', date=None, text='rainfall sediment', applicability='test', parameters=['turbidity_proxy'], seasons=['winter'])]
        self.assertEqual(retrieve_evidence('rainfall', season='summer', knowledge=knowledge), [])

    def test_sensor_filter_excludes_sentinel_sources_for_modis(self):
        df=observations();df['sensor']='MODIS';df['source_kind']='legacy_upload';df['valid_pixel_fraction']=np.nan
        row=detect_alerts(df).query("parameter == 'ndwi'").iloc[0].to_dict()
        result=explain_alert(row,df)
        self.assertEqual(result['provenance']['sensor_filter'],'MODIS')
        self.assertNotIn('gee-sentinel-quality',{source['source_id'] for source in result['sources']})
        hits=retrieve_evidence('cloud shadow',parameter='ndwi',sensor='MODIS')
        self.assertNotIn('gee-sentinel-quality',{source['source_id'] for source in hits})
        source=dict(source_id='general',title='Cloud',url='https://example.org/report',text='Cloud shadow affects reflectance.',applicability='General observations')
        parsed=parse_knowledge(json.dumps([source]))
        self.assertEqual(parsed[0]['sensors'],['all'])
        self.assertEqual(retrieve_evidence('cloud',sensor='MODIS',knowledge=parsed)[0]['source_id'],'general')
        source['sensors']='MODIS'
        with self.assertRaises(ValueError):parse_knowledge(json.dumps([source]))

    def test_unknown_date_and_upload_validation(self):
        source = dict(source_id='custom', title='Local study', url='https://example.org/study', text='Sediment', applicability='Local context', date=None)
        self.assertIsNone(parse_knowledge(json.dumps([source]))[0]['date'])
        source['url'] = 'javascript:alert(1)'
        with self.assertRaises(ValueError):
            parse_knowledge(json.dumps([source]))

    def test_explanation_uses_no_future_weather(self):
        df = observations()
        df.loc[df.index[-1], 'turbidity_proxy'] = 30.
        row = detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0].to_dict()
        future = df.iloc[[-1]].copy()
        future['date'] += pd.Timedelta(weeks=1)
        future['rainfall_mm'] = 500.
        base = explain_alert(row, df)
        appended = explain_alert(row, pd.concat([df, future]))
        self.assertEqual(base, appended)
        self.assertEqual(base['hypotheses'][0]['status'], 'Insufficient local support')
        self.assertIn('not causal proof', base['hypotheses'][0]['limitations'])

    def test_missing_knowledge_cannot_claim_reference_support(self):
        df = observations()
        df.loc[df.index[-1], ['turbidity_proxy', 'rainfall_mm']] = [30., 50.]
        row = detect_alerts(df).query("parameter == 'turbidity_proxy'").iloc[0].to_dict()
        result = explain_alert(row, df, knowledge=[])
        self.assertEqual(result['sources'], [])
        self.assertEqual(result['hypotheses'][0]['status'], 'Observation present; insufficient reference evidence')

    def test_retrieval_benchmark_is_labeled_demo(self):
        result = evaluate_retrieval()
        self.assertEqual(len(result), 3)
        self.assertTrue(result.benchmark.str.contains('not independent validation').all())
        self.assertTrue(result.recall_at_2.between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
