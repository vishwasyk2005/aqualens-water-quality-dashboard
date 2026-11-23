"""Satellite observation summaries and historically supported forecast outlooks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alerts import ALERT_POLICY, PARAMETERS, detect_alerts, zone_series
from .assessment import _number, _reference_collection
from .backtest import _source_values
from .evidence import retrieve_evidence


INDICATORS = {
    "ndwi": ("Water reflectance (NDWI)", "index",
             "NDWI tracks water contrast in the imagery. An unusual value is a cue to compare water extent, shorelines, and scene quality.",
             "Compare imagery and cloud or shadow flags for the selected zone.",
             "water reflectance cloud shadow pixel quality", ("gee-modis-quality", "gee-sentinel-quality")),
    "turbidity_proxy": ("Water clarity signal", "proxy units",
                        "This optical signal can help prioritize a check for changes in clarity or suspended material. The uploaded proxy scale is uncalibrated.",
                        "Inspect the color and clarity of the same area in the next clear image.",
                        "turbidity optical clarity suspended sediment proxy", ("usgs-turbidity", "usgs-sediment")),
    "chlorophyll_proxy": ("Algae-related signal", "proxy units",
                          "A chlorophyll proxy helps track possible changes in algae. Bloom species and toxins require separate evidence.",
                          "Compare the next clear image for persistent discoloration or surface patches.",
                          "chlorophyll algae cyanobacteria", ("epa-cyanobacteria",)),
    "lake_surface_temp_c": ("Lake surface temperature", "°C",
                            "Thermal satellite observations describe changes at the lake surface.",
                            "Compare the next usable thermal image for persistence and location of the change.",
                            "lake surface water temperature satellite", ("esa-lake-observations",)),
    "tsm_mg_l": ("Suspended matter", "mg/L",
                 "The satellite estimate tracks particles in the observed surface water.",
                 "Compare the next usable optical image, particularly near inlets and shorelines.",
                 "lake water satellite suspended matter", ("esa-lake-observations",)),
}
STATUS_RANK = {"Insufficient data": -1, "Stable": 0, "Watch": 1, "Unusual": 2}
FORECAST_HORIZONS = (8, 16, 32)
FORECAST_SKILL_GOAL = 0.2


def _day(value):
    parsed = pd.to_datetime(value, errors="raise", utc=True)
    if pd.isna(parsed):
        raise ValueError("A valid monitoring date is required.")
    return parsed.tz_localize(None).normalize()


def _date(value):
    return value.date().isoformat() if value is not None and pd.notna(value) else None


def _score(value, center, scale):
    if value is None or center is None or scale is None:
        return "Insufficient data", None
    distance = abs(value - center) / scale
    direction = "higher" if value > center else "lower" if value < center else "steady"
    if distance >= ALERT_POLICY["critical_score"]:
        return "Unusual", direction
    if distance >= ALERT_POLICY["warning_score"]:
        return "Watch", direction
    return "Stable", direction


def _indicator(parameter, scoped_frame, when, evidence):
    label, unit, meaning, action, _, _ = INDICATORS[parameter]
    minimum_fraction = ALERT_POLICY["minimum_coverage"]
    history_start = when - pd.Timedelta(weeks=ALERT_POLICY["history_weeks"])
    series = zone_series(scoped_frame, parameter)
    now = series[series.date.eq(when)]
    prior = series[series.date.lt(when) & series.observation_fraction.ge(minimum_fraction)].dropna(subset=["value"])
    recent = prior[prior.date.ge(history_start)]
    value = _number(now.value.iloc[0]) if len(now) else None
    fraction = float(now.observation_fraction.iloc[0]) if len(now) else 0.
    reported = int(now.sampled_cells.iloc[0]) if len(now) else 0
    coverage = _number(now.coverage.iloc[0]) if len(now) else None
    coverage_unknown = not len(now) or bool(now.coverage_unknown.iloc[0]) or coverage is None
    previous = _number(prior.value.iloc[-1]) if len(prior) else None
    center = _number(recent.value.median()) if len(recent) >= ALERT_POLICY["minimum_periods"] else None
    scale = None
    if center is not None:
        scale_floor = .1 if unit == "°C" else max(abs(center) * .05, .01)
        scale = max(1.4826 * float(np.median(abs(recent.value - center))), scale_floor)
    status, direction = _score(value, center, scale)
    if value is None:
        reason = "No usable observed value on this sampling date."
    elif fraction < minimum_fraction:
        reason = "Fewer than half of the reported samples had a usable value."
    elif center is None:
        reason = f"Only {len(recent)} usable dates in the previous 12 weeks. At least 6 are needed."
    else:
        reason = (f"Within the recent reference range from {len(recent)} usable dates." if status == "Stable"
                  else f"{direction.capitalize()} than the recent reference range from {len(recent)} usable dates.")
    if fraction < minimum_fraction:
        status, direction = "Insufficient data", None
    width = ALERT_POLICY["warning_score"] * scale if scale is not None else None
    return dict(
        parameter=parameter, label=label, value=value, unit=unit, date=_date(when),
        previous=previous, previous_date=_date(prior.date.iloc[-1]) if len(prior) else None,
        change=value - previous if value is not None and previous is not None else None,
        baseline=center, baseline_scale=scale, range_low=center - width if width is not None else None,
        range_high=center + width if width is not None else None, history_periods=len(recent),
        status=status, direction=direction, sample_count=int(round(reported * fraction)), reported_count=reported,
        coverage=coverage, coverage_unknown=coverage_unknown, reason=reason,
        interpretation=meaning if value is not None else "A usable satellite reading is needed to describe this signal.",
        action=action if value is not None else "Check the next usable pass.",
        evidence_ids=[row["source_id"] for row in evidence],
    )


def _condition(indicators):
    classified = [row for row in indicators if row["status"] in {"Watch", "Unusual"}]
    if classified:
        return max((row["status"] for row in classified), key=STATUS_RANK.get)
    if not indicators or any(row["status"] == "Insufficient data" for row in indicators):
        return "Insufficient data"
    return "Stable"


def monitor_satellite(frame, *, zone="all", as_of=None, observed_on=None, supplemental=None, knowledge=None):
    """Summarize recorded satellite values at one scope and observation date."""
    zone = str(zone)
    if frame.empty:
        return dict(date=None, zone_id=zone, zone_name="All sampled zones", status="Insufficient data",
                    headline="No satellite observations available", summary="Upload a dated satellite series to assess changes.",
                    source=dict(kind="uploaded", label="Uploaded satellite observations", sensor=None,
                                latest_date=None, age_days=None, stale=None),
                    indicators=[], priority_zones=[], sources=[], comparison=None)
    if zone != "all" and zone not in set(frame.zone_id.astype(str)):
        raise ValueError("Select an available monitoring zone.")
    dates = pd.to_datetime(frame.date)
    cutoff = _day(as_of) if as_of is not None else None
    observations = frame.loc[dates.le(cutoff)] if cutoff is not None else frame
    when = pd.to_datetime(observations.date).max()
    selected = observations if zone == "all" else observations.loc[observations.zone_id.astype(str).eq(zone)]
    selected = selected.assign(**_source_values(selected))
    zone_name = "All sampled zones" if zone == "all" else str(frame.loc[frame.zone_id.astype(str).eq(zone), "zone_name"].iloc[0])
    primary = selected.assign(zone_id=zone, zone_name=zone_name)
    available = [(parameter, primary, when) for parameter in PARAMETERS]
    for parameter in ("lake_surface_temp_c", "tsm_mg_l"):
        source = supplemental if supplemental is not None and parameter in supplemental else selected
        if parameter not in source:
            continue
        if zone != "all":
            source = source.loc[source.zone_id.astype(str).eq(zone)]
        if cutoff is not None:
            source = source.loc[pd.to_datetime(source.date).le(cutoff)]
        usable = pd.to_numeric(source[parameter], errors="coerce")
        usable = usable.where(np.isfinite(usable) & ((usable >= 0) if parameter == "tsm_mg_l" else True))
        source = source.assign(**{parameter: usable})
        supplemental_date = pd.to_datetime(source.loc[usable.notna(), "date"]).max()
        if pd.notna(supplemental_date):
            available.append((parameter, source.assign(zone_id=zone, zone_name=zone_name), supplemental_date))
    if pd.isna(when):
        available = [(parameter, data, date) for parameter, data, date in available if pd.notna(date)]
    sensors = pd.unique(selected["sensor"].dropna().astype(str)) if "sensor" in selected else []
    sensor = str(sensors[0]) if len(sensors) == 1 else None
    source_kinds = selected["source_kind"].dropna().astype(str).unique() if "source_kind" in selected else []
    kind = str(source_kinds[0]) if len(source_kinds) == 1 else "mixed" if len(source_kinds) else "uploaded"
    name = {"legacy_upload": "Uploaded MODIS-style archive", "synthetic_demo": "Synthetic demonstration"}.get(kind, "Uploaded satellite observations")
    today = _day(observed_on) if observed_on is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    age = max((today - when).days, 0) if pd.notna(when) else None
    source_info = dict(kind=kind, label=name, sensor=sensor, latest_date=_date(when), age_days=age,
                       stale=age > 30 if age is not None else None)
    reference = _reference_collection(knowledge)
    if sensor and "modis" in sensor.lower():
        sensor_filter = "MODIS"
    elif sensor and ("sentinel-2" in sensor.lower() or "s2" in sensor.lower()):
        sensor_filter = "Sentinel-2"
    else:
        sensor_filter = "unspecified"
    sources = {}
    indicators = []
    for parameter, data, date in available:
        *_, query, source_ids = INDICATORS[parameter]
        refs = retrieve_evidence(query, knowledge=[row for row in reference if row["source_id"] in source_ids],
                                 top_k=2, sensor=sensor_filter)
        sources.update({row["source_id"]: row for row in refs})
        indicator = _indicator(parameter, data, date, refs)
        indicator["age_days"] = max((today - date).days, 0)
        indicator["stale"] = indicator["age_days"] > 30
        indicators.append(indicator)
    if zone == "all" and pd.notna(when):
        alerts = detect_alerts(selected)
        flagged = alerts[alerts.status.isin(["Watch", "Unusual"])]
        priorities = [
            dict(zone_id=str(identifier), zone_name=str(group.zone_name.iloc[0]),
                 status=max(group.status, key=STATUS_RANK.get),
                 indicators=[INDICATORS[name][0] for name in group.parameter],
                 reason="Check the indicated satellite signals and compare with the next clear pass.")
            for identifier, group in flagged.groupby("zone_id", sort=False, observed=True)
        ]
        priorities.sort(key=lambda item: (-STATUS_RANK[item["status"]], -len(item["indicators"]), item["zone_id"]))
    else:
        priorities = []
    primary_indicators = [row for row in indicators if row["parameter"] in PARAMETERS]
    status = _condition(primary_indicators)
    headline = {
        "Unusual": "Unusual satellite changes to investigate",
        "Watch": "Satellite changes to watch",
        "Stable": "Satellite signals within their recent range",
        "Insufficient data": "Limited satellite evidence for this date",
    }[status]
    if priorities and STATUS_RANK[priorities[0]["status"]] > STATUS_RANK[status]:
        headline = "Localized satellite changes to review"
        status = priorities[0]["status"]
    descriptive = [
        f"{row['label']} is {row['direction']} than its recent range" if row["status"] in {"Watch", "Unusual"}
        else f"{row['label']} is within its recent range" if row["status"] == "Stable"
        else f"{row['label']} needs more usable readings"
        for row in primary_indicators
    ]
    summary = ". ".join(descriptive) + "." if descriptive else "Upload a dated satellite series to assess changes."
    if priorities:
        summary += (" One mapped zone merits a closer look." if len(priorities) == 1
                    else f" {len(priorities)} mapped zones merit a closer look.")
    return dict(date=_date(when), zone_id=zone, zone_name=zone_name, status=status,
                headline=headline, summary=summary, source=source_info, indicators=indicators,
                priority_zones=priorities, sources=list(sources.values()),
                comparison="Reference range from usable observations in the preceding 12 weeks.")


def project_satellite(report, forecast_rows, *, metrics=(), model=None):
    """Assess native forecasts against the same observed reference range."""
    issue = report.get("date")
    horizons = []
    if not issue:
        return dict(origin_date=issue, model=model, horizons=horizons)
    zone = report["zone_id"]
    scores = {(row["parameter"], row["horizon_days"]): row for row in metrics
              if str(row.get("zone_id")) == zone and (model is None or row.get("model") == model)}
    predictions = {(row["parameter"], row["horizon_days"]): row for row in forecast_rows
                   if str(row.get("zone_id")) == zone and not row.get("is_extrapolation", False)
                   and (model is None or row.get("model") in {None, model})}
    for days in FORECAST_HORIZONS:
        target = _date(_day(issue) + pd.Timedelta(days=days))
        indicators = []
        for source in report["indicators"]:
            parameter = source["parameter"]
            if parameter not in PARAMETERS:
                continue
            row = predictions.get((parameter, days), {})
            row_date = row.get("date")
            estimate = _number(row.get("prediction")) if row_date is not None and _date(_day(row_date)) == target else None
            accuracy = scores.get((parameter, days), {})
            r2, pairs = _number(accuracy.get("r2")), _number(accuracy.get("n"))
            status, direction = _score(estimate, source["baseline"], source["baseline_scale"])
            supported = False
            if estimate is None:
                reason = "No matching projection is available for this indicator and date."
            elif r2 is None or pairs is None or pairs < ALERT_POLICY["minimum_periods"]:
                reason = "Historical accuracy is unavailable for this indicator, location, and lead time."
            elif r2 <= FORECAST_SKILL_GOAL:
                reason = "Historical accuracy falls short of the 0.20 R² monitoring goal."
            elif status == "Insufficient data" or source["status"] == "Insufficient data":
                reason = "The latest observation or its reference history is too limited to classify the outlook."
            else:
                supported = True
                reason = ("Projection stays within the observation reference range." if status == "Stable"
                          else f"Projection is {direction} than the observation reference range.")
            if not supported:
                status, direction = "Insufficient data", None
            action = {
                "Stable": "Continue regular satellite monitoring.",
                "Insufficient data": "Check the next usable satellite pass before acting on the estimate.",
            }.get(status, INDICATORS[parameter][3])
            indicators.append(dict(parameter=parameter, label=source["label"], prediction=estimate,
                                   unit=source["unit"], baseline=source["baseline"], range_low=source["range_low"],
                                   range_high=source["range_high"], status=status, direction=direction,
                                   r2=r2, pairs=int(pairs) if pairs is not None else None, reason=reason,
                                   action=action, prediction_low=_number(row.get("lower")), prediction_high=_number(row.get("upper")),
                                   origin_value=source["value"],
                                   change=estimate - source["value"] if estimate is not None and source["value"] is not None else None))
        status = _condition(indicators)
        supported = sum(row["status"] != "Insufficient data" for row in indicators)
        headline = {
            "Unusual": "Unusual change projected",
            "Watch": "Change projected to watch",
            "Stable": "Signals projected to stay near the recent range",
            "Insufficient data": "Partial condition outlook" if supported else "Condition outlook has limited support",
        }[status]
        statements = [
            f"{row['label']} is projected {row['direction']} than its recent range" if row["status"] in {"Watch", "Unusual"}
            else f"{row['label']} is projected to stay within its recent range"
            for row in indicators if row["status"] != "Insufficient data"
        ]
        summary = ". ".join(statements) + ". " if statements else ""
        summary += f"{supported} of {len(indicators)} satellite signals support a condition outlook at this lead."
        next_step = ("Compare the flagged location with the next usable satellite pass." if status in {"Watch", "Unusual"}
                     else "Continue regular satellite monitoring." if supported else "Use the next usable satellite pass to reassess the location.")
        horizons.append(dict(horizon_days=days, target_date=target, status=status, headline=headline,
                             summary=summary, next_step=next_step, indicators=indicators))
    return dict(origin_date=issue, model=model, horizons=horizons)
