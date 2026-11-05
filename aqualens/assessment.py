"""Past-only water monitoring and reference-backed use/precaution assessment.

Satellite proxies never establish fitness for drinking, recreation, irrigation,
or aquatic life. General guidance is retrieved from a verified local reference
collection, then applied with fixed transparent rules; it is not a safety
certificate or an LLM-generated diagnosis.
"""
from __future__ import annotations

import math
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .alerts import ALERT_POLICY, PARAMETERS, _observed, current_condition, detect_alerts, zone_series
from .evidence import load_knowledge, retrieve_evidence

LABELS = {"ndwi": "Water index (NDWI)", "turbidity_proxy": "Turbidity proxy", "chlorophyll_proxy": "Chlorophyll proxy"}
TOPIC_IDS = {
    "drinking": {"who-drinking-assessment", "cpcb-designated-uses"},
    "recreation": {"cpcb-designated-uses", "cdc-bloom-precautions"},
    "irrigation": {"fao-irrigation-quality", "cpcb-designated-uses"},
    "aquatic_life": {"cpcb-designated-uses", "epa-cyanobacteria"},
    "precautions": {"who-drinking-assessment", "cpcb-designated-uses", "fao-irrigation-quality", "cdc-bloom-precautions"},
    "monitoring": {"usgs-turbidity", "epa-cyanobacteria", "gee-modis-quality", "gee-sentinel-quality"},
}
OFFICIAL_DOMAINS = {"who.int", "cpcb.nic.in", "cpcb.gov.in", "fao.org", "cdc.gov", "usgs.gov", "epa.gov", "developers.google.com"}
RANK = {"Stable": 0, "Watch": 1, "Unusual": 2, "Insufficient data": 3}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _official(url):
    host = (urlparse(str(url)).hostname or "").lower()
    return urlparse(str(url)).scheme == "https" and any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_DOMAINS)


def _timestamp(value):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("as_of must be a valid observation date.")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return _number(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _reference_collection(knowledge):
    """Do not let custom passages impersonate a curated policy reference.

    User evidence can be searched elsewhere in the dashboard. Policy advice is
    supported only by matching source passages in the installed reference book.
    Empty/unmatched collections withhold advice rather than silently falling
    back to a different collection.
    """
    bundled = load_knowledge()
    supplied = bundled if knowledge is None else knowledge
    expected = {row["source_id"]: row for row in bundled}
    eligible = []
    for row in supplied:
        if not isinstance(row, dict):
            continue
        original = expected.get(row.get("source_id"))
        if original is None or not _official(row.get("url", "")):
            continue
        if any(row.get(key) != original.get(key) for key in ("title", "url", "text", "applicability", "assessment_topics", "parameters", "seasons", "sensors", "verified_on", "date")):
            continue
        eligible.append(row)
    return eligible


def _preliminary_assessment(observations, selected, current_rows, latest, zone_id, alerts):
    """Previous observed-step screening; never a physical-unit or safety model."""
    scope = "Lake-wide observed mean" if zone_id == "all" else "Selected-zone observed mean"
    scope_note = ("Directions compare lake-wide observed means with their previous usable readings. They can hide different zone changes; historical labels still reflect the most flagged zone."
        if zone_id == "all" else "Directions compare this zone's observed means with their previous usable readings.")
    signals = []
    for row in observations:
        value, previous = row["value"], row["previous"]
        if value is None or previous is None:
            direction, label = "unavailable", "Comparison unavailable"
        elif math.isclose(value, previous, rel_tol=1e-12, abs_tol=1e-12):
            direction, label = "steady", "No numerical step change"
        elif value > previous:
            direction, label = "rising", "Higher than previous reading"
        else:
            direction, label = "falling", "Lower than previous reading"
        row["trend"] = dict(direction=direction, label=label, previous_date=row.get("previous_date"), change=row["change"], comparison_scope=scope,
            basis="Direct comparison with the preceding usable observed reading; numerical equality tolerance only. Stable/Watch/Unusual labels use a separate historical-deviation policy.")
        name = row["label"]
        if direction == "unavailable":
            observation = f"{name}: a current and preceding usable observed reading are both needed to describe a change."
            title = f"{name}: comparison unavailable"
        elif direction == "steady":
            observation = f"{name} is numerically unchanged from its preceding usable observed reading."
            title = f"{name}: no step change"
        else:
            word = "higher" if direction == "rising" else "lower"
            observation = f"{name} is {word} than its preceding usable observed reading ({row['previous_date'][:10]})."
            title = f"{name}: {word} reading"
        parameter = row["parameter"]
        if direction == "unavailable":
            interpretation = "A relative-change interpretation is withheld because a usable comparison is missing."
            action = "Obtain another usable observation and inspect quality metadata before interpreting a trend."
        elif parameter == "turbidity_proxy":
            interpretation = "The optical-proxy change is a reason to check water clarity and suspended material. Its undocumented formula does not establish the direction or amount of physical turbidity."
            if direction == "steady":
                interpretation = "No step change is recorded in this proxy. This does not establish clear water or rule out hazards."
            action = "Inspect water clarity and suspended material on site, and check the imagery and sampled locations."
        elif parameter == "chlorophyll_proxy":
            interpretation = "The proxy change may flag an algae-related difference worth inspecting. It does not establish a change in calibrated algal biomass, identify harmful species, or confirm toxins."
            if direction == "steady":
                interpretation = "No step change is recorded in the chlorophyll proxy. Harmful algae and toxin presence remain unassessed."
            action = "Inspect algae-related changes and confirm the signal with appropriate field sampling. Follow the separately stated precautions if visible bloom signs or an advisory are present."
        else:
            interpretation = "The index change is a reason to inspect water reflectance and observation quality. It does not independently establish pollution or water-use suitability."
            if direction == "steady":
                interpretation = "No step change is recorded in the water index. Water-quality or safety conditions cannot be inferred from an unchanged index."
            action = "Compare the dated imagery and quality flags to check whether water coverage, reflectance or observation conditions changed."
        refs = list(row["evidence_ids"])
        if not refs:
            interpretation = "Interpretation withheld: no matching verified reference passage was retrieved."
            action = "Reference-backed guidance is withheld until a matching verified reference is available."
        signals.append(dict(parameter=parameter, title=title, observation=observation, interpretation=interpretation, action=action, evidence_ids=refs,
            evidence_note="Numeric direction comes from the uploaded readings. References support a cautious screening check; they do not calibrate the proxy or prove a local cause.",
            status=direction))
    comparable = [row for row in observations if row["trend"]["direction"] != "unavailable"]
    rising = {row["parameter"] for row in comparable if row["trend"]["direction"] == "rising" and row["evidence_ids"]}
    flagged_parameters = {row["parameter"] for row in observations if row["status"] in {"Watch", "Unusual"}}
    flagged = bool(flagged_parameters)
    if not comparable:
        title = "More usable observations needed for a change assessment"
    elif {"turbidity_proxy", "chlorophyll_proxy"} <= rising:
        title = "Clarity and algae-related signals need review" if flagged_parameters & {"turbidity_proxy", "chlorophyll_proxy"} else "Turbidity and chlorophyll proxy readings increased"
    elif "turbidity_proxy" in rising:
        title = "Clarity-related signal needs review" if "turbidity_proxy" in flagged_parameters else "Higher turbidity-proxy reading"
    elif "chlorophyll_proxy" in rising:
        title = "Algae-related signal needs review" if "chlorophyll_proxy" in flagged_parameters else "Higher chlorophyll-proxy reading"
    elif flagged:
        title = "Unusual historical index changes need review"
    elif any(row["parameter"] == "ndwi" and row["trend"]["direction"] != "steady" for row in comparable):
        title = "Water-index change to inspect"
    elif any(row["trend"]["direction"] == "falling" for row in comparable):
        title = "Lower proxy readings to verify"
    elif all(row["trend"]["direction"] == "steady" for row in comparable):
        title = "No step change in the available indicators"
    else:
        title = "Recorded index changes to review"
    if comparable and not any(row["evidence_ids"] for row in comparable):
        title = "Recorded changes await interpretation evidence"
    summary = " ".join(signal["observation"] for signal in signals)
    summary += " These are preliminary screening signals, not measured concentrations or a water-use approval."
    confidence_basis = [f"{len(comparable)} of 3 indicators have both current and preceding usable observed readings.",
        "Turbidity and chlorophyll proxy formulas and physical-unit calibration are unverified; their numeric direction is not a calibrated change in physical water quality."]
    if not len(current_rows) or "valid_pixel_fraction" not in current_rows or current_rows.valid_pixel_fraction.isna().any():
        confidence_basis.append("Satellite pixel-quality information is missing for some or all latest readings.")
    history_counts = []
    changed_samples = False
    for row in observations:
        parameter = row["parameter"]
        values = pd.to_numeric(selected.get(parameter, pd.Series(index=selected.index, dtype=float)), errors="coerce")
        quality = pd.to_numeric(selected.get("valid_pixel_fraction", pd.Series(np.nan, index=selected.index)), errors="coerce")
        valid = _observed(selected, parameter) & values.notna() & np.isfinite(values) & (quality.isna() | quality.ge(ALERT_POLICY["minimum_coverage"]))
        usable = selected[valid]
        prior = usable[usable.date.lt(latest) & usable.date.ge(latest - pd.Timedelta(weeks=ALERT_POLICY["history_weeks"]))] if latest is not None else usable.iloc[:0]
        history_counts.append(int(prior.date.nunique()))
        if row.get("previous_date") and "cell_id" in usable:
            before_cells = set(usable.loc[usable.date.eq(pd.Timestamp(row["previous_date"])), "cell_id"].astype(str))
            now_cells = set(usable.loc[usable.date.eq(latest), "cell_id"].astype(str))
            if now_cells and before_cells and now_cells != before_cells:
                changed_samples = True
    if history_counts and min(history_counts) < ALERT_POLICY["minimum_periods"]:
        confidence_basis.append(f"At least one indicator has fewer than {ALERT_POLICY['minimum_periods']} usable prior dates in the historical comparison window.")
    if changed_samples:
        confidence_basis.append("The observed locations differ from the preceding reading; changes in the means can partly reflect the sampled locations.")
    if len(alerts) and alerts.spatial_fraction.lt(.5).any():
        confidence_basis.append("Some latest zone samples cover less than half of historically reported locations.")
    if any(not row["evidence_ids"] for row in observations):
        confidence_basis.append("Interpretation evidence is unavailable for one or more indicators; those interpretations are withheld.")
    confidence = dict(label="Insufficient" if not comparable else "Limited", basis=confidence_basis,
        meaning="Qualitative confidence in this relative-change interpretation, not a validated probability, measurement-accuracy estimate or water-safety score.")
    return dict(title=title, summary=summary, confidence=confidence, signals=signals, scope_note=scope_note)


def _use_preliminary(uses, preliminary):
    signals = preliminary["signals"]
    by_parameter = {row["parameter"]: row for row in signals}
    for use in uses:
        relevant = signals if use["id"] == "monitoring" else [by_parameter[p] for p in ("turbidity_proxy", "chlorophyll_proxy", "ndwi")]
        available = [row for row in relevant if row["status"] != "unavailable"]
        grounded = [row for row in available if row["evidence_ids"]]
        summary = " ".join(row["observation"] for row in relevant)
        if not available:
            label = "Preliminary comparison unavailable"
            action = "Relative-change interpretation is withheld until current and preceding usable observed readings are available."
        elif not grounded:
            label = "Recorded signals; interpretation withheld"
            action = "Reference-backed preliminary guidance is withheld because matching verified interpretation evidence is unavailable."
        else:
            label = "Preliminary three-indicator screening"
            if use["id"] == "monitoring":
                action = "Use the dated index changes to prioritize imagery review and field follow-up. " + " ".join(row["action"] for row in grounded[:2])
            elif use["id"] == "irrigation":
                action = "Review the optical changes on site as an initial irrigation-source screening step. This does not assess crop, soil or salinity suitability. " + grounded[0]["action"]
            elif use["id"] == "aquatic_life":
                action = "Review the available index changes as an initial habitat-screening step; oxygen and ammonia conditions remain unmeasured. " + grounded[0]["action"]
            elif use["id"] == "recreation":
                action = "Use the optical changes to prioritize an on-site check before considering contact with the water. " + grounded[0]["action"]
            else:
                action = "Use the optical changes to prioritize source inspection before considering drinking or cooking use. " + grounded[0]["action"]
        refs = list(dict.fromkeys([identifier for row in grounded for identifier in row["evidence_ids"]] + (use["evidence_ids"] if grounded else [])))
        use["preliminary"] = dict(label=label, summary=summary, action=action, evidence_ids=refs, safety_clearance=False,
            evidence_note="A screening interpretation of the supplied indicators, not a suitability decision. Reference guidance does not provide missing local measurements.")


def assess_water(df: pd.DataFrame, knowledge=None, zone_id="all", as_of=None) -> dict:
    """Assess the latest available snapshot on/before ``as_of`` without forecasts.

    ``as_of`` is an observation cutoff, not an assertion about today's water.
    A zone absent on the global snapshot is withheld, never backfilled with its
    own older readings. Measurements in additional unvalidated columns are not
    silently treated as laboratory results.
    """
    zone_id = str(zone_id)
    frame = df.copy()
    if "date" not in frame:
        if not frame.empty:
            raise ValueError("Observations need a date column.")
        frame = pd.DataFrame(columns=["date", "zone_id", "zone_name", "cell_id", *PARAMETERS])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame = frame[frame.date.notna()].copy()
    if "ndwi" in frame:
        ndwi_numeric = pd.to_numeric(frame.ndwi, errors="coerce")
        ndwi_raw = pd.to_numeric(frame.get("ndwi_raw", frame.ndwi), errors="coerce")
        invalid_ndwi = (ndwi_numeric.notna() & ~ndwi_numeric.between(-1, 1)) | (ndwi_raw.notna() & ~ndwi_raw.between(-1, 1))
        if invalid_ndwi.any():
            if "ndwi_is_observed" not in frame:
                frame["ndwi_is_observed"] = _observed(frame, "ndwi")
            frame.loc[invalid_ndwi, "ndwi"] = np.nan
            frame.loc[invalid_ndwi, "ndwi_is_observed"] = False
    if as_of is not None:
        frame = frame[frame.date.le(_timestamp(as_of))].copy()
    available_zones = set(frame.zone_id.astype(str)) if "zone_id" in frame else set()
    if zone_id != "all" and zone_id not in available_zones:
        raise ValueError("No observations for the selected zone on or before the assessment date.")
    latest = frame.date.max() if len(frame) else None
    zone_name = "Whole lake" if zone_id == "all" else str(frame.loc[frame.zone_id.astype(str).eq(zone_id), "zone_name"].iloc[0])
    selected = frame if zone_id == "all" else frame[frame.zone_id.astype(str).eq(zone_id)]
    current_rows = selected[selected.date.eq(latest)] if latest is not None else selected
    reference_book = _reference_collection(knowledge)
    sensor_values = selected.get("sensor", pd.Series(dtype=str)).dropna().astype(str).str.upper()
    kinds = selected.get("source_kind", pd.Series(dtype=str)).dropna().astype(str).str.lower()
    sensor = "MODIS" if sensor_values.str.contains("MODIS").any() or kinds.str.contains("legacy").any() else "Sentinel-2" if sensor_values.str.contains("SENTINEL|S2").any() else None
    sources = {}
    retrieval_log = []

    def evidence(topic, query, required_ids=None):
        identifiers = TOPIC_IDS.get(topic, set())
        if required_ids is not None:
            identifiers = identifiers & set(required_ids)
        candidates = [row for row in reference_book if row["source_id"] in identifiers and (not row.get("assessment_topics") or topic in row["assessment_topics"])]
        if sensor is None:
            candidates = [row for row in candidates if "all" in [str(value).lower() for value in row.get("sensors", ["all"])]]
        hits = retrieve_evidence(query, knowledge=candidates, top_k=2, sensor=sensor)
        for row in hits:
            sources[row["source_id"]] = row
        retrieval_log.append(dict(topic=topic, query=query, source_ids=[row["source_id"] for row in hits]))
        return [row["source_id"] for row in hits]

    observations = []
    all_condition = current_condition(frame) if len(frame) else {"parameters": [], "summary": "No observations available."}
    alerts = detect_alerts(frame) if len(frame) else pd.DataFrame()
    if zone_id != "all" and len(alerts):
        alerts = alerts[alerts.zone_id.astype(str).eq(zone_id)]
    observation_refs = {
        "ndwi": evidence("monitoring", "water reflectance cloud shadow quality pixels observation reliability", {"gee-modis-quality", "gee-sentinel-quality"}),
        "turbidity_proxy": evidence("monitoring", "turbidity optical clarity suspended sediment proxy", {"usgs-turbidity"}),
        "chlorophyll_proxy": evidence("monitoring", "chlorophyll algae cyanobacteria toxins sampling", {"epa-cyanobacteria"}),
    }
    for parameter in PARAMETERS:
        parameter_alerts = alerts[alerts.parameter.eq(parameter)] if len(alerts) else pd.DataFrame()
        if zone_id == "all":
            row = next((item for item in all_condition["parameters"] if item["parameter"] == parameter), {})
            value, previous, change = (_number(row.get(key)) for key in ("value", "previous", "change"))
            status = row.get("status", "Insufficient data")
            previous_date = row.get("previous_date")
            flagged = parameter_alerts[parameter_alerts.status.isin(["Watch", "Unusual"])] if len(parameter_alerts) else pd.DataFrame()
            missing = int(parameter_alerts.status.eq("Insufficient data").sum()) if len(parameter_alerts) else 0
            names = ", ".join(flagged.zone_name.astype(str)) if len(flagged) else ""
            reason = ("The number is the lake-wide observed mean; the label reflects the highest-priority zone status, based on each zone's own history. "
                f"{len(flagged)} of {len(parameter_alerts)} zones show Watch/Unusual changes" + (f" ({names})" if names else "") + f"; {missing} have insufficient data.") if len(parameter_alerts) else "No usable observation history is available."
        else:
            history = zone_series(frame, parameter)
            history = history[history.zone_id.astype(str).eq(zone_id)].sort_values("date")
            now = history[history.date.eq(latest)]
            past = history[history.date.lt(latest)].dropna(subset=["value"])
            value = _number(now.value.iloc[0]) if len(now) else None
            previous = _number(past.value.iloc[-1]) if len(past) else None
            change = value - previous if value is not None and previous is not None else None
            previous_date = past.date.iloc[-1].isoformat() if len(past) else None
            alert = parameter_alerts.iloc[0].to_dict() if len(parameter_alerts) else {}
            status = alert.get("status", "Insufficient data")
            reason = alert.get("reason", "No observations for this zone at the latest dataset date.")
        if value is None:
            status = "Insufficient data"
        observations.append(dict(parameter=parameter, label=LABELS[parameter], value=value, previous=previous, previous_date=previous_date, change=change,
            units="Dimensionless index" if parameter == "ndwi" else "Supplied proxy scale; physical units unverified", status=status, reason=reason,
            evidence_ids=observation_refs[parameter], evidence_note="The value and monitoring status come from uploaded observations and the disclosed change policy. References explain interpretation limits; they do not verify the local water.",
            evidence_status="reference_available" if observation_refs[parameter] else "no_reference",
            zone_details=parameter_alerts.to_dict("records") if zone_id == "all" and len(parameter_alerts) else []))
    preliminary = _preliminary_assessment(observations, selected, current_rows, latest, zone_id, alerts)
    worst = max((row["status"] for row in observations), key=lambda item: RANK.get(item, 3), default="Insufficient data")
    status_code = {"Stable": "stable", "Watch": "watch", "Unusual": "unusual", "Insufficient data": "insufficient"}.get(worst, "insufficient")
    status_label = {"Stable": "Stable observed signals", "Watch": "Changes need review", "Unusual": "Unusual changes detected", "Insufficient data": "Assessment limited by missing data"}[worst]
    if latest is None:
        summary = "No dated observations are available for this assessment."
    elif not len(current_rows):
        summary = f"{zone_name} has no observations on the latest dataset date ({latest.date()}); its present monitoring status is withheld."
    elif zone_id == "all":
        flagged_zones = int(alerts[alerts.status.isin(["Watch", "Unusual"])].zone_id.nunique()) if len(alerts) else 0
        total_zones = int(alerts.zone_id.nunique()) if len(alerts) else 0
        summary = f"Latest uploaded observations: {latest.date()}. {flagged_zones} of {total_zones} zones show Watch/Unusual changes. Monitoring labels describe zone-level historical changes, not safety or local causes."
    else:
        summary = f"Latest uploaded observations for {zone_name}: {latest.date()}. Monitoring status: {worst}. This describes historical change, not suitability for a water use."
    uses = []
    use_rules = [
        ("drinking", "Drinking & cooking", "drinking", "drinking water microbial chemical safety testing laboratory E coli", {"who-drinking-assessment"},
         "Satellite indices cannot establish microbial or chemical drinking-water safety.", ["Microbial indicators, including E. coli", "Chemical analysis appropriate to the source", "Treatment and local drinking-water assessment"],
         "Use a supply confirmed suitable for drinking and cooking while appropriate laboratory and treatment checks are arranged."),
        ("recreation", "Bathing & recreation", "recreation", "bathing recreation microbial coliform pH dissolved oxygen biochemical oxygen demand swimming local advisories", {"cpcb-designated-uses", "cdc-bloom-precautions"},
         "Bathing suitability needs use-specific measurements and local advisories that this proxy CSV does not contain.", ["Microbial indicators for bathing water", "pH and dissolved oxygen", "Biochemical oxygen demand (BOD)", "Current local recreational-water advisories"],
         "Check current local bathing advisories and obtain the missing use-specific tests before treating this location as suitable for bathing."),
        ("irrigation", "Irrigation", "irrigation", "irrigation salinity electrical conductivity sodium adsorption ratio boron chloride soil crop", {"fao-irrigation-quality"},
         "Turbidity and chlorophyll proxies do not measure salinity or sodium-related irrigation hazards.", ["Electrical conductivity / salinity", "Sodium adsorption ratio (SAR)", "Boron and chloride", "pH", "Crop, soil and irrigation-method context"],
         "Obtain irrigation-water tests and assess crop, soil and irrigation-method requirements before selecting this source."),
        ("aquatic_life", "Aquatic life & fisheries", "aquatic_life", "aquatic life fisheries dissolved oxygen pH ammonia", {"cpcb-designated-uses"},
         "Satellite signals cannot establish habitat suitability, dissolved oxygen or ammonia conditions. Fish-consumption safety is a separate assessment.", ["Dissolved oxygen", "pH and free ammonia"],
         "Measure dissolved oxygen, pH and ammonia and review local habitat conditions before concluding that the water supports aquatic life or fisheries."),
    ]
    for identifier, title, topic, query, allowed, description, missing, action in use_rules:
        identifiers = evidence(topic, query, allowed)
        uses.append(dict(id=identifier, title=title, status="not_established", label="Suitability not established", summary=description, missing_tests=missing,
            action=action if identifiers else "Reference-backed guidance is withheld because no matching verified passage was retrieved.", evidence_ids=identifiers,
            evidence_note="General guidance identifies the measurements needed for this use. The uploaded observations do not supply those measurements, so no suitability clearance is issued.",
            evidence_status="reference_available" if identifiers else "no_reference"))
    monitoring_ids = evidence("monitoring", "satellite reflectance quality observation turbidity optical proxy monitoring", {"usgs-turbidity", "gee-modis-quality", "gee-sentinel-quality"})
    monitoring_supported = any(row["value"] is not None for row in observations)
    uses.append(dict(id="monitoring", title="Satellite monitoring & research", status="monitoring_only" if monitoring_supported else "not_established",
        label="Monitoring only" if monitoring_supported else "No usable latest readings", summary="The available observed indices can support exploratory tracking of relative changes. They do not establish suitability for consuming or contacting the water." if monitoring_supported else "No usable latest observed values are available for monitoring this location.",
        missing_tests=["Satellite pixel quality flags" ] if "valid_pixel_fraction" not in current_rows or current_rows.valid_pixel_fraction.isna().any() else [],
        action="Inspect the dated imagery, quality flags and sampling pattern alongside the index history." if monitoring_ids else "Reference-backed interpretation guidance is withheld because no matching verified passage was retrieved.",
        evidence_ids=monitoring_ids, evidence_note="References describe optical signals and observation quality. Monitoring availability is based on the uploaded data, not on a water-safety determination.", evidence_status="reference_available" if monitoring_ids else "no_reference"))
    _use_preliminary(uses, preliminary)
    precautions = []

    def precaution(identifier, title, priority, action, why, trigger, topic, query, allowed):
        identifiers = evidence(topic, query, allowed)
        precautions.append(dict(id=identifier, title=title, priority=priority,
            action=action if identifiers else "Recommendation withheld: no matching verified reference passage was retrieved.",
            why=why, trigger=trigger, evidence_ids=identifiers,
            evidence_note="Conditional general guidance, not evidence that a local hazard is present." if priority == "conditional" else "General guidance applied to the identified information gap; it is not proof of local contamination.",
            evidence_status="reference_available" if identifiers else "no_reference"))
    precaution("confirm-drinking", "Confirm drinking-water quality", "high", "Use a confirmed suitable drinking-water supply; arrange microbial and chemical checks before relying on this lake for drinking or cooking.",
        "The upload contains optical indices and proxies, without a drinking-water test panel.", "Drinking or cooking use is being considered; required tests are absent.", "drinking", "drinking water microbial chemical risk assessment testing", {"who-drinking-assessment"})
    precaution("conditional-bloom", "If bloom signs or an advisory are present", "conditional", "If you see scum, unusual discoloration or a local bloom advisory, avoid contact and keep people and animals out of affected water. Do not rely on boiling to remove algal toxins.",
        "A chlorophyll proxy cannot identify a harmful bloom or confirm toxins. This precaution activates only with visible signs or an advisory; neither is established by this CSV.",
        "Only if visible scum, unusual discoloration or a current official bloom advisory is present; not triggered by a proxy value alone.", "precautions", "algal bloom scum discoloration avoid water pets animals boiling toxins", {"cdc-bloom-precautions"})
    precaution("verify-irrigation", "Test before irrigation decisions", "routine", "Test salinity, sodium-related hazards and relevant ions, then review crop and soil requirements before using the water for irrigation.",
        "The upload does not contain EC, SAR, boron or chloride results.", "Irrigation use is being considered; the required water and crop/soil information is absent.", "irrigation", "irrigation salinity sodium boron chloride crop soil tests", {"fao-irrigation-quality"})
    precaution("field-check", "Check water on site", "routine", "Arrange field measurements for the intended use, including microbial indicators where relevant, pH, dissolved oxygen and ammonia for habitat assessment.",
        "A stable satellite signal cannot establish use-specific water quality.", "Use suitability is unresolved by the uploaded proxy measurements.", "precautions", "designated water use bathing coliform pH dissolved oxygen ammonia", {"cpcb-designated-uses"})
    if any(row["status"] in {"Watch", "Unusual"} for row in observations) or any("quality" in row["reason"].lower() for row in observations):
        precaution("check-optical", "Verify unusual optical changes", "routine", "Review image quality and spatial sampling, then check the observed changes with field turbidity and water-quality measurements.",
            "Optical changes can reflect suspended material or measurement conditions; the cause is not established.", "An optical change needs review or image-quality information is missing.", "monitoring", "turbidity optical suspended material cloud quality reflectance observation", {"usgs-turbidity", "gee-modis-quality", "gee-sentinel-quality"})
    no_evidence = any(not row["evidence_ids"] for row in uses + precautions)
    limitations = [
        "This assessment describes the latest uploaded observation date, not live water conditions today.",
        "NDWI concerns water reflectance/extent and does not independently measure pollution or drinking-water safety.",
        "Turbidity and chlorophyll use the supplied uncalibrated proxy scale; no NTU or concentration conversion is assumed.",
        "Microbial, chemical and use-specific laboratory measurements are absent or have not been validated by this prototype. No physical water use is cleared.",
        "Monitoring categories describe change against past observations; stable signals do not establish safety and unusual signals do not prove contamination.",
        "General reference guidance is not local hazard confirmation. RAG cannot supply missing measurements.",
    ]
    if "valid_pixel_fraction" not in current_rows or current_rows.valid_pixel_fraction.isna().any():
        limitations.append("Satellite pixel quality is missing for some or all latest readings; the observation quality is unverified.")
    if len(alerts) and alerts.spatial_fraction.lt(.5).any():
        limitations.append("The latest sample covers less than half of historically reported coordinates in some locations; changing spatial sampling can affect the displayed means.")
    if no_evidence:
        limitations.append("Some matching verified reference passages were not retrieved; the affected recommendations are explicitly withheld.")
    return _clean(dict(schema_version=1, as_of=latest.isoformat() if latest is not None else None, zone_id=zone_id, zone_name=zone_name,
        status=dict(code=status_code, label=status_label, summary=summary), preliminary=preliminary, observations=observations, uses=uses, precautions=precautions, limitations=limitations,
        sources=list(sources.values()), evidence_status="partial" if no_evidence and sources else "unavailable" if not sources else "available",
        provenance=dict(observation_cutoff=_timestamp(as_of).isoformat() if as_of is not None else None, assessment_basis="Latest uploaded observed measurements; no DET forecast used in the present assessment", source_rows=int(len(current_rows)),
            history_rows=int(len(selected)), sensor_filter=sensor, monitoring_policy=dict(ALERT_POLICY), retrieval="Topic-filtered TF-IDF cosine over matching bundled official passages",
            generation="Deterministic evidence-linked assessment rules; no LLM diagnosis", preliminary_method="Previous usable observed-step comparison, separate from historical-deviation labels; qualitative interpretation confidence only", policy_reference_check="Custom passages must match the installed verified reference text and URL to support use or precaution guidance", retrieval_queries=retrieval_log)))
