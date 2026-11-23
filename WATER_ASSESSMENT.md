# Satellite condition monitoring

The **Monitor** page (`#assessment`) opens at the recorded satellite condition. It shows the observation date, available coverage, recent changes, zones to inspect, supporting evidence, and projections from that observation. Water-use tabs below the readings show the relevant satellite findings, an inspection step and evidence. Open **How to verify this use** for the field checks.

The supplied MODIS-style series ends on **10 February 2023**. Supplementary [ESA Lakes CCI observations](https://climate.esa.int/en/projects/lakes/data/) add suspended matter and lake surface temperature where usable, through December 2023. Each indicator shows its own acquisition date. A newer compatible satellite CSV can update the monitor.

The forecast condition cards compare the 8-, 16-, and 32-day values with their observed 12-week reference range. A condition label requires enough recorded history and matching historical accuracy above the project’s R² goal of 0.20. Forecasts from the archive retain their original issue and target dates. Lake median forecasts are evaluated and displayed on their own observation series.

## What is implemented

- Whole-lake and zone assessments for the latest observation date. The supplied CSV ends on **10 February 2023**; these results are not a measurement of water today.
- Observed indicators with previous usable readings, absolute changes, and monitoring labels. Whole-lake means and flagged zones have their own evidence.
- Four water-use tabs cover drinking and cooking, bathing and recreation, irrigation, and aquatic life and fisheries. The assessment API also includes satellite monitoring and research.
- Conditional, cited precautions for drinking decisions, visible bloom signs or an official advisory, irrigation checks and field verification. A satellite chlorophyll proxy does not trigger a harmful-bloom diagnosis.
- Numbered evidence buttons open the stored reference passage, its application limits, original HTTPS source and checked date. The recorded condition can be exported with its citations.
- Zone changes, use tabs and keyboard navigation, source inspection, and links to the geographic map and separate DET forecast.

## Preliminary screening from the three indicators

The page now gives a useful initial interpretation from NDWI, turbidity proxy and chlorophyll proxy instead of leading with missing laboratory tests. Each indicator has a previous-observation direction, a tentative meaning, a practical next check, and relevant retrieved citations. The combined preliminary summary identifies which optical signals warrant attention, and each use card applies those signals to that use before its confirmation limits.

Trend direction compares the latest usable observed value with the preceding usable observed value; it is separate from the Stable/Watch/Unusual historical-change category. A rise can remain within the historical range. Missing, imputed, poor-coverage or invalid values do not become a change signal. Whole-lake changes concern the observed mean and may reflect changing spatial sampling; they do not establish a change at every location.

Qualitative confidence describes the available information for this screening. It is not a calibrated probability, model accuracy or a water-safety score. Unknown pixel quality, unverified proxy calibration, short comparison history and changing sampled locations limit interpretation. Falling proxy values do not establish cleaner or safer water. When reference evidence is absent, the observed trend remains visible and reference-backed interpretation/actions are withheld.

The screen may suggest checking suspended material or clarity when the turbidity proxy rises, or investigating a possible algal-biomass change when the chlorophyll proxy rises. NDWI changes suggest inspecting water reflectance/extent and image quality. These are checks motivated by the observed indices, not measured concentrations, confirmed blooms or local causes.

## What the supplied data can establish

Each use shows **Use checks pending** alongside the satellite findings. NDWI and the turbidity and chlorophyll proxies help prioritize source inspections, bloom checks, irrigation-intake inspections and habitat visits. Field measurements establish the facts needed to assess the intended use.

The test lists are screening requirements, not a complete certification panel. Drinking decisions require a suitable laboratory and local water-safety assessment. CPCB surface-water source classes are not an approval to drink untreated water. Irrigation interpretation also needs crop, soil and management context. Habitat suitability and fish-consumption safety are separate questions.

For Vembanad, start with dated station results near the satellite locations. Match the station location and sample date to nearby satellite observations and keep the sampling method, units and source with each field result. Use missing measurements to plan targeted sampling. The assessment currently reads the satellite CSV, so verified field results need a separate import workflow. [Kerala water-quality portal](https://kspcbipam.kerala.gov.in/Dashboard/waterQuality) · [KSPCB monitoring reports](https://kspcb.kerala.gov.in/water-and-air-quality-directory)

CPCB's 2022 report lists Vembanad stations including Pathiramanal (station 2312) and publishes ranges for pH, dissolved oxygen, conductivity, BOD and bacteria. Use these ranges as historical context and request individual sample dates and locations for matching with satellite scenes. [CPCB 2022 lake monitoring report](https://cpcb.nic.in/wqm/2022/Water_pond_tanks_2022.pdf)

## Evidence and RAG method

The assessment retrieves relevant stored passages using topic and sensor metadata plus TF-IDF cosine similarity. Fixed, transparent rules combine observed data with retrieved guidance. There is no cloud LLM call and no automatic inference of absent measurements. A general reference supports interpretation or precautions; the CSV supports local measured findings. Neither is presented as local safety proof.

Policy citations are accepted only when the supplied source ID, URL, passage and metadata match the installed verified reference collection. Custom reference collections remain searchable in the existing Evidence page, but altered or unrelated passages cannot authorize safety advice. An empty or unmatched policy collection explicitly withholds recommendations, without silently substituting a different library. Source passages are data, never executable instructions.

Official guidance and reference passages:

- [WHO: Guidelines for drinking-water quality (2022)](https://www.who.int/publications/i/item/9789240045064): drinking-water risk assessment, microbial and chemical aspects, treatment and surveillance.
- Annexure I lists designated-use requirements, including bacteria, pH, oxygen, BOD, ammonia and irrigation measurements. [CPCB: Water quality criteria](https://cpcb.nic.in/wqm/Ind-Guidelines-RestWaterBodies.pdf)
- [FAO: Water quality for agriculture, chapter 1/Table 1](https://www.fao.org/4/t0234e/T0234E01.htm): salinity, EC/SAR, specific-ion hazards, and local crop/soil context.
- [CDC: Preventing illnesses caused by harmful algal blooms](https://www.cdc.gov/harmful-algal-blooms/prevention/index.html): conditional exposure precautions and official advisories. These do not establish that a bloom exists locally.

Existing USGS, EPA and sensor-quality references provide optical and observation-quality context. Stored passages are short paraphrases, not direct quotations or live searches. Publication and verification dates have different meanings.

## API and saved workspace

`GET /api/assessment?dataset=modis&zone=all` returns the latest assessment. `zone=Z1` selects a zone; optional `as_of=2021-12-31` restricts the input to earlier observations. Future dates and unknown zones are rejected. The bootstrap response includes `water_assessments`, one report for the lake and each zone, so offline viewing uses the same saved assessment.

No DET checkpoint, training policy, historical test target or source measurement is changed by this feature. Forecasts remain separate from present assessment and retain their reported accuracy limitations.
