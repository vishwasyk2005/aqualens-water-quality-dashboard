# Present-water assessment

The primary dashboard page is **Water assessment** (`#assessment`). Uploading a CSV opens this page. The flow is recorded condition, a **three-indicator preliminary interpretation**, possible uses, precautions, inspectable evidence, and then the DET outlook.

## What is implemented

- Whole-lake and zone assessments for the latest observation date. The supplied CSV ends on **10 February 2023**; these results are not a measurement of water today.
- Three observed indicators with previous observed readings, absolute changes, and exploratory monitoring labels. The displayed whole-lake values are observed means; whole-lake labels reflect the most flagged zone, not a separate test of that mean. Zone details remain in the audit output.
- Five use categories: drinking/cooking, bathing/recreation, irrigation, aquatic life/fisheries, and satellite monitoring/research. Each card gives a verdict, explanation, next action, missing tests and citations.
- Conditional, cited precautions for drinking decisions, visible bloom signs or an official advisory, irrigation checks and field verification. A satellite chlorophyll proxy does not trigger a harmful-bloom diagnosis.
- Numbered evidence buttons open the stored reference passage, its application limits, original HTTPS source and checked date. The assessment can be downloaded as a dated Markdown report with its citations.
- Zone changes, use tabs and keyboard navigation, source inspection, and links to the geographic map and separate DET forecast.

## Preliminary screening from the three indicators

The page now gives a useful initial interpretation from NDWI, turbidity proxy and chlorophyll proxy instead of leading with missing laboratory tests. Each indicator has a previous-observation direction, a tentative meaning, a practical next check, and relevant retrieved citations. The combined preliminary summary identifies which optical signals warrant attention, and each use card applies those signals to that use before its confirmation limits.

Trend direction compares the latest usable observed value with the preceding usable observed value; it is separate from the Stable/Watch/Unusual historical-change category. A rise can remain within the historical range. Missing, imputed, poor-coverage or invalid values do not become a change signal. Whole-lake changes concern the observed mean and may reflect changing spatial sampling; they do not establish a change at every location.

Qualitative confidence describes the available information for this screening. It is not a calibrated probability, model accuracy or a water-safety score. Unknown pixel quality, unverified proxy calibration, short comparison history and changing sampled locations limit interpretation. Falling proxy values do not establish cleaner or safer water. When reference evidence is absent, the observed trend remains visible and reference-backed interpretation/actions are withheld.

The screen may suggest checking suspended material or clarity when the turbidity proxy rises, or investigating a possible algal-biomass change when the chlorophyll proxy rises. NDWI changes suggest inspecting water reflectance/extent and image quality. These are checks motivated by the observed indices, not measured concentrations, confirmed blooms or local causes.

## What the supplied data can establish

NDWI, uncalibrated turbidity proxy and uncalibrated chlorophyll proxy support exploratory monitoring. They do not establish pathogen levels, chemical contamination, dissolved oxygen, salinity, sodium hazards or toxin presence. No physical use is cleared from these proxies. The prototype returns **Suitability not established** and identifies the required measurements, rather than labelling unknown water safe or unsafe.

The test lists are screening requirements, not a complete certification panel. Drinking decisions require a suitable laboratory and local water-safety assessment. CPCB surface-water source classes are not an approval to drink untreated water. Irrigation interpretation also needs crop, soil and management context. Habitat suitability and fish-consumption safety are separate questions.

Extra CSV columns are not silently treated as validated laboratory measurements. A future implementation can add a separate, validated laboratory-data workflow with explicit units, sampling dates, methods and applicable criteria.

## Evidence and RAG method

The assessment retrieves relevant stored passages using topic and sensor metadata plus TF-IDF cosine similarity. Fixed, transparent rules combine observed data with retrieved guidance. There is no cloud LLM call and no automatic inference of absent measurements. A general reference supports interpretation or precautions; the CSV supports local measured findings. Neither is presented as local safety proof.

Policy citations are accepted only when the supplied source ID, URL, passage and metadata match the installed verified reference collection. Custom reference collections remain searchable in the existing Evidence page, but altered or unrelated passages cannot authorize safety advice. An empty or unmatched policy collection explicitly withholds recommendations, without silently substituting a different library. Source passages are data, never executable instructions.

Official guidance checked **27 October 2026**:

- [WHO: Guidelines for drinking-water quality (2022)](https://www.who.int/publications/i/item/9789240045064): drinking-water risk assessment, microbial and chemical aspects, treatment and surveillance.
- [CPCB: Water quality criteria](https://cpcb.gov.in/water-quality-criteria/): designated-use requirements, including coliforms, pH, oxygen/BOD, ammonia and irrigation indicators. No numerical proxy-to-standard conversion is performed.
- [FAO: Water quality for agriculture, chapter 1/Table 1](https://www.fao.org/4/t0234e/T0234E01.htm): salinity, EC/SAR, specific-ion hazards, and local crop/soil context.
- [CDC: Preventing illnesses caused by harmful algal blooms](https://www.cdc.gov/harmful-algal-blooms/prevention/index.html): conditional exposure precautions and official advisories. These do not establish that a bloom exists locally.

Existing USGS, EPA and sensor-quality references provide optical and observation-quality context. Stored passages are short paraphrases, not direct quotations or live searches. Publication and verification dates have different meanings.

## API and saved workspace

`GET /api/assessment?dataset=modis&zone=all` returns the latest assessment. `zone=Z1` selects a zone; optional `as_of=2021-12-31` restricts the input to earlier observations. Future dates and unknown zones are rejected. The bootstrap response includes `water_assessments`, one report for the lake and each zone, so offline viewing uses the same saved assessment.

No DET checkpoint, training policy, historical test target or source measurement is changed by this feature. Forecasts remain separate from present assessment and retain their reported accuracy limitations.
