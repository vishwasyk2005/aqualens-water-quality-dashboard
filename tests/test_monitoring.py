import json
import unittest

import pandas as pd

from aqualens.monitoring import monitor_satellite, project_satellite
from server import Dashboard


def observations():
    dates = pd.date_range("2024-01-01", periods=11, freq="8D")
    return pd.DataFrame([
        dict(date=date, zone_id="Z1", zone_name="North inlet", cell_id=cell,
             ndwi=.3, ndwi_raw=.3, turbidity_proxy=.5, chlorophyll_proxy=1.,
             is_observed=True, source_kind="legacy_upload", sensor="MODIS",
             valid_pixel_fraction=.75)
        for date in dates for cell in ("p1", "p2")
    ])


class SatelliteMonitoringTests(unittest.TestCase):
    def test_observed_values_match_native_forecast_target_filter(self):
        data = observations()
        latest = data.date.max()
        data.loc[data.date.eq(latest), "ndwi"] = .75
        data.loc[data.date.eq(latest) & data.cell_id.eq("p2"), "ndwi_raw"] = 1.1
        data.loc[data.date.eq(latest) & data.cell_id.eq("p1"), "ndwi_raw"] = .12
        data.loc[data.date.eq(latest) & data.cell_id.eq("p2"), "valid_pixel_fraction"] = .1
        summary = monitor_satellite(data, zone="Z1", observed_on="2025-01-01")
        water = summary["indicators"][0]
        self.assertEqual(water["value"], .12)
        self.assertEqual(water["previous"], .3)
        self.assertEqual(water["status"], "Unusual")
        self.assertEqual(water["sample_count"], 1)
        self.assertEqual(water["reported_count"], 2)
        self.assertEqual(water["coverage"], .75)
        self.assertTrue(summary["source"]["stale"])
        self.assertIn("gee-modis-quality", water["evidence_ids"])
        json.dumps(summary, allow_nan=False)

    def test_explicit_date_leaves_a_missing_observation_visible(self):
        data = observations()
        latest = data.date.max()
        data["chlorophyll_proxy_is_observed"] = data.date.ne(latest)
        summary = monitor_satellite(data)
        algae = summary["indicators"][-1]
        self.assertEqual(algae["date"], latest.date().isoformat())
        self.assertEqual(algae["status"], "Insufficient data")
        self.assertIsNone(algae["value"])
        self.assertLess(algae["previous_date"], algae["date"])
        self.assertIsNone(monitor_satellite(data, as_of="2023-01-01")["date"])

    def test_projection_uses_native_dates_and_exact_scope_accuracy(self):
        data = observations()
        observed = monitor_satellite(data, zone="Z1")
        target = (data.date.max() + pd.Timedelta(days=8)).date().isoformat()
        predictions = [
            dict(parameter=parameter, model="det", zone_id="Z1", horizon_days=8, date=target, prediction=value)
            for parameter, value in [("ndwi", .7), ("turbidity_proxy", .8), ("chlorophyll_proxy", 1.4)]
        ]
        metrics = [
            dict(parameter="ndwi", model="det", zone_id="Z1", horizon_days=8, r2=-.01, n=40),
            dict(parameter="turbidity_proxy", model="det", zone_id="all", horizon_days=8, r2=.8, n=40),
            dict(parameter="chlorophyll_proxy", model="det", zone_id="Z1", horizon_days=8, r2=.35, n=40),
        ]
        report = project_satellite(observed, predictions, metrics=metrics, model="det")
        forecast = report["horizons"][0]
        self.assertEqual(forecast["target_date"], target)
        self.assertEqual(forecast["status"], "Unusual")
        self.assertEqual(forecast["indicators"][0]["prediction"], .7)
        self.assertEqual([item["status"] for item in forecast["indicators"]],
                         ["Insufficient data", "Insufficient data", "Unusual"])
        self.assertEqual(report["horizons"][1]["status"], "Insufficient data")
        json.dumps(report, allow_nan=False)

    def test_bundled_api_keeps_separate_source_dates_and_verifies_the_outlook(self):
        app = Dashboard(use_custom_models=False)
        result = app.get("/api/monitoring", {"model": ["seasonal_kernel"], "zone": ["all"]})
        observed = result["observation"]
        self.assertEqual(observed["date"], "2023-02-10")
        self.assertTrue(observed["source"]["stale"])
        dated = {row["parameter"]: row["date"] for row in observed["indicators"]}
        self.assertEqual(dated["lake_surface_temp_c"], "2023-12-26")
        self.assertEqual(dated["tsm_mg_l"], "2023-12-31")
        projection = result["projection"]
        self.assertEqual(projection["origin_date"], observed["date"])
        self.assertEqual([row["target_date"] for row in projection["horizons"]],
                         ["2023-02-18", "2023-02-26", "2023-03-14"])
        for lead in projection["horizons"]:
            readings = {row["parameter"]: row for row in lead["indicators"]}
            self.assertEqual(readings["ndwi"]["status"], "Insufficient data")
            self.assertEqual(readings["chlorophyll_proxy"]["status"], "Stable")
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
