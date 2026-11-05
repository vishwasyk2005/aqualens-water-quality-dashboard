"""Offline lexical RAG and auditable hypothesis comparison.

TF-IDF vectors are lexical representations, not neural semantic embeddings.
Rules compare observable support for candidates; this is not an LLM ToT search
or model attribution. Uploaded passages are treated as data, never instructions.
"""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from .alerts import zone_series, _observed, ALERT_POLICY

KNOWLEDGE_PATH = Path(__file__).resolve().parent.parent / "data" / "evidence.json"
STOP_WORDS = set("a an the of to in for and or is are on by with that this it as at from be can not does its".split())


def parse_knowledge(raw: bytes | str) -> list[dict]:
    """Validate a user JSON evidence collection; never fetch or execute contents."""
    if isinstance(raw, bytes):
        if len(raw) > 2_000_000:
            raise ValueError("Evidence JSON is limited to 2 MB.")
        raw = raw.decode("utf-8-sig")
    records = json.loads(raw)
    if not isinstance(records, list) or len(records) > 200:
        raise ValueError("Evidence must be a JSON list with at most 200 passages.")
    ids = set()
    result = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Each passage must be an object.")
        for key in ("source_id", "title", "url", "text", "applicability"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Each passage needs a nonempty {key}.")
        if item["source_id"] in ids:
            raise ValueError("source_id values must be unique.")
        ids.add(item["source_id"])
        if urlparse(item["url"]).scheme != "https" or not urlparse(item["url"]).netloc:
            raise ValueError("Source URLs must be HTTPS links.")
        if len(item["text"]) > 15000:
            raise ValueError("Each passage must be at most 15,000 characters.")
        params = item.get("parameters", ["all"])
        seasons = item.get("seasons", ["all"])
        if not isinstance(params, list) or not all(isinstance(x, str) for x in params):
            raise ValueError("parameters must be a list of strings.")
        if not isinstance(seasons, list) or not all(isinstance(x, str) for x in seasons):
            raise ValueError("seasons must be a list of strings.")
        sensors = item.get("sensors", ["all"])
        if not isinstance(sensors, list) or not all(isinstance(x, str) for x in sensors):
            raise ValueError("sensors must be a list of strings.")
        topics = item.get("assessment_topics", [])
        if not isinstance(topics, list) or not all(isinstance(x, str) for x in topics):
            raise ValueError("assessment_topics must be a list of strings.")
        date = item.get("date")
        if date is not None:
            try:
                pd.Timestamp(date)
            except (ValueError, TypeError):
                raise ValueError("date must be a valid date or null.") from None
        result.append({key: item.get(key) for key in ("source_id", "title", "url", "date", "text", "applicability")} | {"parameters": params, "seasons": seasons, "sensors": sensors, "assessment_topics": topics, "verified_on": item.get("verified_on")})
    return result


def load_knowledge() -> list[dict]:
    return parse_knowledge(KNOWLEDGE_PATH.read_bytes())


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z][a-z0-9]+", text.lower().replace("_", " ")) if token not in STOP_WORDS]


def retrieve_evidence(query: str, parameter: str | None = None, season: str | None = None, top_k: int = 3, knowledge: list[dict] | None = None, sensor: str | None = None) -> list[dict]:
    """Metadata-filtered TF-IDF cosine retrieval; no result on zero overlap."""
    if top_k <= 0 or not query.strip():
        return []
    records = load_knowledge() if knowledge is None else knowledge
    candidates = [row for row in records if (parameter is None or parameter in row.get("parameters", ["all"]) or "all" in row.get("parameters", [])) and (season is None or season in row.get("seasons", ["all"]) or "all" in row.get("seasons", []))]
    if sensor:
        sensor_key = sensor.lower()
        candidates = [row for row in candidates if "all" in [str(s).lower() for s in row.get("sensors", ["all"])] or sensor_key in [str(s).lower() for s in row.get("sensors", ["all"])]]
    if not candidates:
        return []
    terms = [Counter(_tokens(row["title"] + " " + row["text"])) for row in candidates]
    query_terms = Counter(_tokens(query))
    vocabulary = set().union(*(set(t) for t in terms))
    weights = {term: math.log((1 + len(terms)) / (1 + sum(term in counter for counter in terms))) + 1 for term in vocabulary}
    query_vec = {t: count * weights[t] for t, count in query_terms.items() if t in weights}
    norm_query = math.sqrt(sum(x*x for x in query_vec.values()))
    if norm_query == 0:
        return []
    results = []
    for row, counts in zip(candidates, terms):
        vector = {term: count * weights[term] for term, count in counts.items()}
        norm_doc = math.sqrt(sum(x*x for x in vector.values()))
        score = sum(weight * vector.get(term, 0) for term, weight in query_vec.items()) / (norm_query * norm_doc) if norm_doc else 0
        if score > 0:
            results.append(dict(row, score=round(score, 5)))
    return sorted(results, key=lambda row: (-row["score"], row["source_id"]))[:top_k]


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (ValueError, TypeError):
        return False


def explain_alert(alert: dict, df: pd.DataFrame, knowledge: list[dict] | None = None, forecast_summary: dict | None = None) -> dict:
    """Compare hypotheses using observations up to alert date only."""
    when = pd.Timestamp(alert["date"])
    parameter = alert["parameter"]
    context = df[(df.zone_id == alert["zone_id"]) & (pd.to_datetime(df.date) <= when)].copy()
    context["date"] = pd.to_datetime(context["date"])
    current = context[context.date.eq(when)]
    prior = context[(context.date < when) & (context.date >= when - pd.Timedelta(weeks=12))]
    observed = _observed(context, parameter)
    quality_values = pd.to_numeric(context.get("valid_pixel_fraction", pd.Series(np.nan, index=context.index)), errors="coerce")
    usable = context[observed & (quality_values.isna() | quality_values.ge(ALERT_POLICY["minimum_coverage"]))]
    now_usable = usable[usable.date.eq(when)]
    before_usable = usable[(usable.date < when) & (usable.date >= when - pd.Timedelta(weeks=12))]
    sources = {}
    hypotheses = []
    reference_book = load_knowledge() if knowledge is None else knowledge
    # Use-assessment guidance is not evidence for an optical-change cause.
    reference_book = [row for row in reference_book if not row.get("assessment_topics") or "monitoring" in row["assessment_topics"]]
    sensor_values = context.get("sensor", pd.Series(dtype=str)).dropna().astype(str).str.upper()
    source_values = context.get("source_kind", pd.Series(dtype=str)).dropna().astype(str).str.lower()
    sensor = "MODIS" if sensor_values.str.contains("MODIS").any() or source_values.str.contains("legacy").any() else "Sentinel-2" if sensor_values.str.contains("SENTINEL|S2").any() else None

    def add(name, supported, support, limitations, query, evidence_parameter=parameter):
        evidence = retrieve_evidence(query, parameter=evidence_parameter, knowledge=reference_book, top_k=2, sensor=sensor)
        for row in evidence:
            sources[row["source_id"]] = row
        if supported and evidence:
            status = "Plausible; unconfirmed"
        elif supported:
            status = "Observation present; insufficient reference evidence"
        else:
            status = "Insufficient local support"
        hypotheses.append(dict(name=name, status=status, support=support, limitations=limitations, evidence=evidence))

    rising = _finite(alert.get("value")) and _finite(alert.get("baseline")) and alert["value"] > alert["baseline"]
    rain_now = float(now_usable.rainfall_mm.mean()) if "rainfall_mm" in now_usable and len(now_usable) else np.nan
    rain_before = float(before_usable.groupby("date").rainfall_mm.mean().median()) if "rainfall_mm" in before_usable and len(before_usable) else np.nan
    rain_support = _finite(rain_now) and _finite(rain_before) and rain_now > max(rain_before * 1.5, rain_before + 5)
    add("Runoff or suspended sediment", parameter == "turbidity_proxy" and rising and rain_support,
        [f"Latest rainfall context: {rain_now:.1f} mm; previous-period median: {rain_before:.1f} mm." if _finite(rain_now) and _finite(rain_before) else "Aligned measured rainfall context is unavailable.", "Turbidity is above its historical median." if parameter == "turbidity_proxy" and rising else "No matching turbidity increase established."],
        "Co-occurrence is not causal proof. Catchment flow and sediment samples are missing. Rainfall context is used only when supplied in the uploaded measurements.", "rainfall runoff sediment storms", "turbidity_proxy")
    chl_now = float(now_usable.chlorophyll_proxy.mean()) if len(now_usable) else np.nan
    chl_before = float(before_usable.groupby("date").chlorophyll_proxy.mean().median()) if len(before_usable) else np.nan
    chl_support = _finite(chl_now) and _finite(chl_before) and chl_now > chl_before + max(abs(chl_before) * 0.15, 0.01)
    add("Algal biomass change", chl_support and parameter in ("chlorophyll_proxy", "turbidity_proxy"),
        [f"Chlorophyll proxy: {chl_now:.3f}; previous-period median: {chl_before:.3f}." if _finite(chl_now) and _finite(chl_before) else "Insufficient observed chlorophyll history."],
        "A proxy cannot identify cyanobacteria, toxins or nutrient sources. Nutrient and field measurements are required.", "chlorophyll algae nutrients temperature cyanobacteria", "chlorophyll_proxy")
    quality = float(alert.get("coverage")) if _finite(alert.get("coverage")) else np.nan
    low_quality = not _finite(quality) or quality < ALERT_POLICY["minimum_coverage"]
    add("Measurement or coverage artefact", low_quality,
        [f"Effective coverage is {quality:.0%}." if _finite(quality) else "Satellite pixel coverage is unknown; valid uploaded measurements are retained with this qualification."],
        "Low coverage suggests inspecting imagery; it does not establish that the observed change is an artefact. Adequate coverage does not rule out atmospheric errors.", "cloud shadow reflectance pixels quality", parameter)
    older = usable[(usable.date >= when - pd.Timedelta(days=395)) & (usable.date <= when - pd.Timedelta(days=335))]
    add("Seasonal variation", False,
        [f"{older.date.nunique()} usable periods near the same season last year."],
        "A single seasonal window is not enough to establish a recurring seasonal cause. Compare several years and model residuals.", "seasonal variation", parameter)
    zone = alert.get("zone_name", str(alert["zone_id"]))
    if alert.get("severity") == "Insufficient data":
        summary = f"{zone}: classification withheld for {parameter} on {when.date()}. {alert.get('reason', '')} Causes cannot be established."
    else:
        summary = f"{zone}: {alert.get('status', alert.get('severity', 'Monitoring'))} signal for {parameter} on {when.date()}. {alert.get('reason', '')} Candidate explanations below remain unconfirmed; these categories do not assess drinking-water safety."
    if forecast_summary and forecast_summary.get("available"):
        predicted = forecast_summary.get("prediction")
        if _finite(predicted):
            summary += f" DET forecasts {float(predicted):.4f} for {forecast_summary.get('date', 'the selected date')} in the uploaded parameter scale."
        if forecast_summary.get("is_extrapolation"):
            summary += " This selected horizon extends beyond the trained horizon and is experimental."
    elif forecast_summary:
        summary += " DET forecasting requires a compatible trained checkpoint for this dataset."
    kinds = sorted(context.source_kind.dropna().astype(str).unique().tolist()) if "source_kind" in context else []
    return dict(summary=summary, current_condition=dict(as_of=when.isoformat(),value=alert.get("value"),baseline=alert.get("baseline"),status=alert.get("status",alert.get("severity")),reason=alert.get("reason"),coverage_unknown=bool(alert.get("coverage_unknown",not _finite(quality)))), forecast_summary=forecast_summary, hypotheses=hypotheses, sources=list(sources.values()), provenance={"as_of": when.isoformat(), "zone_id": str(alert["zone_id"]), "parameter": parameter, "source_kinds": kinds, "sensor_filter": sensor, "retrieval": "metadata-filtered TF-IDF cosine (lexical RAG)", "explanation_method": "deterministic evidence-linked hypothesis comparison; no LLM ToT search or causal attribution", "policy": dict(ALERT_POLICY), "evidence_status": "General reference paraphrases; publication dates unknown unless supplied. Local causes unconfirmed."})


def evaluate_retrieval(knowledge: list[dict] | None = None) -> pd.DataFrame:
    """Tiny hand-authored retrieval smoke benchmark; not scientific validation."""
    cases = [
        ("rainfall runoff suspended sediment", "turbidity_proxy", {"usgs-turbidity", "usgs-sediment"}),
        ("chlorophyll algae nutrients cyanobacteria", "chlorophyll_proxy", {"epa-cyanobacteria"}),
        ("cloud shadows scene classification", "ndwi", {"gee-sentinel-quality"}),
    ]
    rows = []
    for query, parameter, relevant in cases:
        hits = retrieve_evidence(query, parameter=parameter, top_k=2, knowledge=knowledge)
        found = {row["source_id"] for row in hits}
        true_positive = len(found & relevant)
        rows.append(dict(query=query, parameter=parameter, precision_at_2=true_positive / 2, recall_at_2=true_positive / len(relevant), retrieved=", ".join(sorted(found)), benchmark="3 hand-authored demo queries; not independent validation"))
    return pd.DataFrame(rows)

