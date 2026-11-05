"""Causal exploratory monitoring alerts; not water-safety categories."""
from __future__ import annotations
import numpy as np
import pandas as pd

PARAMETERS = ("ndwi", "turbidity_proxy", "chlorophyll_proxy")
ALERT_POLICY = {
    "history_weeks": 12, "minimum_periods": 6, "minimum_coverage": 0.5,
    "warning_score": 2.5, "critical_score": 4.0,
    "description": "Absolute robust z-score against preceding 12 weeks; exploratory thresholds, not regulatory or validated health limits.",
}
ALERT_COLUMNS = ["zone_id", "zone_name", "date", "parameter", "value", "baseline", "score", "severity", "reason", "coverage", "observation_fraction", "coverage_unknown", "status", "spatial_fraction"]


def _observed(frame: pd.DataFrame, parameter: str | None = None) -> pd.Series:
    if parameter and parameter + "_is_observed" in frame:
        return frame[parameter + "_is_observed"].fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])
    if "is_observed" not in frame:
        return pd.Series(True, index=frame.index)
    return frame["is_observed"].fillna(False).astype(str).str.lower().isin(["true", "1", "1.0"])


def zone_series(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Aggregate finite observed cells; absent cells must be explicit input rows."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    raw = pd.to_numeric(frame[parameter], errors="coerce").replace([np.inf, -np.inf], np.nan)
    quality = pd.to_numeric(frame.get("valid_pixel_fraction", pd.Series(np.nan, index=frame.index)), errors="coerce").clip(0, 1)
    # Missing quality metadata is different from a missing measurement. Retain
    # observed values, report quality as unknown, and never invent 100% pixels.
    valid = _observed(frame, parameter) & raw.notna() & (quality.isna() | quality.ge(ALERT_POLICY["minimum_coverage"]))
    frame["_value"] = raw.where(valid)
    frame["_coverage"] = quality.where(valid)
    frame["_observed"] = valid.astype(float)
    frame["_unknown"] = valid & quality.isna()
    expected = frame.groupby("zone_id", observed=True).cell_id.nunique()
    result = frame.groupby(["zone_id", "zone_name", "date"], as_index=False, observed=True).agg(
        value=("_value", "mean"), coverage=("_coverage", "mean"), observation_fraction=("_observed", "sum"), sampled_cells=("cell_id", "size"), coverage_unknown=("_unknown", "any"))
    result["spatial_fraction"] = result.observation_fraction / result.zone_id.map(expected).clip(lower=1)
    result["observation_fraction"] /= result.sampled_cells.clip(lower=1)
    return result.sort_values("date")


def detect_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Latest global-date snapshot per zone and parameter.

    Baseline excludes current value, imputed and low coverage observations.
    Scale floor is max(5% absolute median, 0.01); minimum 6 prior periods.
    """
    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    latest = pd.to_datetime(df["date"]).max()
    rows = []
    for parameter in PARAMETERS:
        series = zone_series(df, parameter)
        for (zone_id, zone_name), group in series.groupby(["zone_id", "zone_name"], observed=True):
            now = group[group.date.eq(latest)]
            value = float(now.value.iloc[0]) if len(now) else np.nan
            coverage = float(now.coverage.iloc[0]) if len(now) else np.nan
            observation_fraction = float(now.observation_fraction.iloc[0]) if len(now) else 0.0
            coverage_unknown = bool(now.coverage_unknown.iloc[0]) if len(now) else True
            spatial_fraction = float(now.spatial_fraction.iloc[0]) if len(now) else 0.0
            prior = group[(group.date < latest) & (group.date >= latest - pd.Timedelta(weeks=ALERT_POLICY["history_weeks"])) & group.observation_fraction.ge(ALERT_POLICY["minimum_coverage"])].value.dropna()
            baseline = float(prior.median()) if len(prior) else np.nan
            score = np.nan
            severity = "Insufficient data"
            if not len(now):
                reason = "No observations for this zone at the latest dataset date."
            elif not np.isfinite(value):
                latest_rows = df[(df.zone_id == zone_id) & (pd.to_datetime(df.date) == latest)]
                latest_raw = pd.to_numeric(latest_rows[parameter], errors="coerce")
                if not latest_raw.replace([np.inf, -np.inf], np.nan).notna().any():
                    reason = "Latest parameter measurements are missing; classification withheld."
                elif not _observed(latest_rows, parameter).any():
                    reason = "Latest values are imputed or marked unobserved; classification withheld."
                else:
                    reason = "Latest measured pixel coverage is low; classification withheld."
            elif observation_fraction < ALERT_POLICY["minimum_coverage"]:
                reason = "Less than 50% of the reported sample has usable observed measurements; classification withheld."
            elif len(prior) < ALERT_POLICY["minimum_periods"]:
                reason = f"Only {len(prior)} usable prior periods; at least 6 are required."
            else:
                mad = float(np.median(np.abs(prior.to_numpy() - baseline)))
                scale = max(1.4826 * mad, abs(baseline) * 0.05, 0.01)
                score = (value - baseline) / scale
                severity = "Critical" if abs(score) >= ALERT_POLICY["critical_score"] else "Warning" if abs(score) >= ALERT_POLICY["warning_score"] else "Normal"
                direction = "above" if score >= 0 else "below"
                reason = f"{abs(score):.2f} robust scale units {direction} the prior {len(prior)}-period median. Exploratory monitoring alert."
                if parameter == "ndwi":
                    reason += " NDWI change concerns water reflectance/extent; it does not establish pollution."
            if spatial_fraction < .5:
                reason += " Available sample covers less than half of historically reported coordinates; changing spatial sampling can affect these averages."
            if coverage_unknown:
                reason += " Satellite pixel quality metadata is unavailable; measurement quality remains unverified."
            status = {"Normal": "Stable", "Warning": "Watch", "Critical": "Unusual"}.get(severity, severity)
            rows.append(dict(zone_id=zone_id, zone_name=zone_name, date=latest, parameter=parameter, value=value, baseline=baseline, score=score, severity=severity, reason=reason, coverage=coverage, observation_fraction=observation_fraction, coverage_unknown=coverage_unknown, status=status, spatial_fraction=spatial_fraction))
    return pd.DataFrame(rows, columns=ALERT_COLUMNS)




def current_condition(df: pd.DataFrame) -> dict:
    """Describe latest uploaded observations, without environmental safety labels."""
    if df.empty:
        return dict(as_of=None, summary="No observations available.", parameters=[], zones=[])
    latest = pd.to_datetime(df.date).max()
    alerts = detect_alerts(df)
    parameters = []
    rank = {"Stable": 0, "Watch": 1, "Unusual": 2, "Insufficient data": 3}
    for parameter in PARAMETERS:
        series = zone_series(df, parameter)
        # Lake-level summaries weight observed cell counts, not arbitrary zone size.
        usable = df.copy()
        usable["date"] = pd.to_datetime(usable.date)
        raw = pd.to_numeric(usable.get(parameter + "_raw", usable[parameter]), errors="coerce")
        quality = pd.to_numeric(usable.get("valid_pixel_fraction", pd.Series(np.nan, index=usable.index)), errors="coerce")
        valid = _observed(usable, parameter) & np.isfinite(raw) & pd.to_numeric(usable[parameter], errors="coerce").notna() & (quality.isna() | quality.ge(.5))
        history = raw.where(valid).groupby(usable.date).mean().sort_index()
        value = float(history.get(latest, np.nan))
        previous_values = history[history.index < latest].dropna()
        previous = float(previous_values.iloc[-1]) if len(previous_values) else np.nan
        delta = value - previous
        selected = alerts[alerts.parameter.eq(parameter)]
        statuses = selected.status.tolist()
        status = max(statuses, key=lambda x: rank.get(x, 3)) if statuses else "Insufficient data"
        parameters.append(dict(parameter=parameter, value=value, previous=previous,
            previous_date=previous_values.index[-1].isoformat() if len(previous_values) else None,
            change=delta, change_pct=100 * delta / abs(previous) if np.isfinite(previous) and abs(previous)>1e-12 else None,
            status=status, coverage_unknown=bool(selected.coverage_unknown.any()),
            units="Uploaded index; physical units unverified", date=latest.isoformat()))
    zones = []
    for (zone_id, zone_name), selected in alerts.groupby(["zone_id", "zone_name"], observed=True):
        zones.append(dict(zone_id=zone_id, zone_name=zone_name,
            status=max(selected.status, key=lambda x: rank.get(x, 3)),
            coverage_unknown=bool(selected.coverage_unknown.any()),
            parameters=selected.to_dict("records")))
    flagged = sum(item["status"] in ("Watch", "Unusual") for item in zones)
    summary = f"Latest uploaded observations: {latest.date()}. {flagged} of {len(zones)} zones show unusual historical changes requiring review."
    if not flagged:
        summary = f"Latest uploaded observations: {latest.date()}. No zone exceeds the statistical change thresholds among available observations where sufficient history is available."
    if alerts.spatial_fraction.lt(.5).any():
        summary += " Spatial sampling differs across dates, so zone means may reflect changes in sampled locations."
    if alerts.coverage_unknown.any():
        summary += " Satellite pixel quality is unknown."
    summary += " Monitoring status describes change relative to history; it does not establish water safety or a cause."
    return dict(as_of=latest.isoformat(), summary=summary, parameters=parameters, zones=zones,
        coverage_unknown=bool(alerts.coverage_unknown.any()), method=ALERT_POLICY["description"])
