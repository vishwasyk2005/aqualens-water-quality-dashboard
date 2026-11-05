# Validation record

Updated 24 October 2026. AquaLens is a custom local dashboard with a Python API and a DET-only forecasting pipeline.

## Data and model provenance

The bundled training source is the user's `MODIS_Lake_Features_NDWI_Turbidity_Chlorophyll_2020_2025_.csv`, copied unchanged to `data/modis_observations.csv`. It contains 46,755 rows and 143 observation dates from 2020-01-01 to 2023-02-10. The filename's 2025 endpoint is not treated as data coverage. Satellite extraction provenance, proxy formulas, physical units and pixel QA are unverified; no authenticated Earth Engine acquisition was performed in this implementation.

The loader preserves raw data, excludes 2,476 NDWI values outside [-1,1], tracks observation flags per parameter, and leaves missing satellite-quality metadata unknown. Negative/extreme turbidity and chlorophyll values remain explicitly unverified. Changing coordinate samples are distinguished from missing readings.

The DET run is reproducible with seed 42 and the settings in `models/evaluation.json`. A 6x6 spatial grid intersected with inferred zones reduces 981 reported coordinates to 28 model cells. Native observation dates are retained; median cadence is eight days.

| Partition | Boundary / size |
|---|---|
| Training | Through 2022-02-26; 89 windows |
| Validation | Through 2022-08-13; 18 windows |
| Test | From 2022-08-21; 19 windows |
| Selected DET | History 8 observations, hidden size 32, learning rate 0.001 |
| Selected epoch | 51; separate production refit also 51 epochs |

Target windows do not cross partition boundaries. Scaling is fitted on the training partition for evaluation. Two configurations of the same DET are selected using validation loss; the final chronological test is not used for selection. Rolling test origins may use observations available by that origin. Actual elapsed target dates are recorded, so nominal 8/16/32-day heads are not misrepresented as uniformly spaced measurements.

Published metrics belong to `det_evaluation_v2.pt`, fitted only on training history with validation-selected settings. The dashboard uses `det_v2.pt`, a separate refit on all available historical windows. These checkpoints must match the dataset fingerprint and hierarchy; incompatible uploads do not silently use a different forecasting method.

## Measured forecasting limits

- Only DET supplies forecasts. Direct heads represent 1, 2 and 4 native steps, approximately 8, 16 and 32 days.
- Daily display points interpolate DET anchors. They are not separately trained daily forecasts.
- Above 32 days, projections are recursive, experimental and have no calibrated bounds.
- Turbidity test R2 is negative at all three evaluated horizons; NDWI and chlorophyll results are stronger but remain dataset-specific. The dashboard exposes the quality warning rather than hiding these results.
- Nominal 80% zone bands use residual quantiles from the same validation partition used for model selection. Test coverage is approximately 47%-78%, not the nominal 80%.
- Validation residual widths transfer from the evaluation checkpoint to the all-history production refit. Production coverage and daily interpolated coverage are not independently established.
- Lake summaries combine zone forecasts using latest parameter-valid cell counts; aggregate uncertainty is withheld.
- Current-condition categories reflect historical deviation. Unknown QA, inferred boundaries and changing spatial samples qualify interpretation. The system cannot establish drinking-water safety or a pollution cause.

Detailed results remain in `evaluation.csv`, `evaluation_by_zone.csv`, `heldout_predictions.csv`, `interval_coverage.csv`, `training_history.csv`, `tuning.csv` and `training_audit.json` under `models/`.

## Backend verification

Direct end-to-end checks with the saved real-data checkpoint verified:

- DET availability and dataset-fingerprint matching.
- 32-day inference: 576 rows across six zones and three parameters, dated 2023-02-11 through 2023-03-14.
- 730-day inference: 13,140 finite predictions; rows beyond 32 days are marked extrapolations and have null bounds.
- Future-date selection, including 2023-03-01, returns the matching day rather than a rounded weekly date.
- Whole-lake and zone explanations retain current observations, numerical DET output, evidence and quality caveats.
- Bootstrap output is strict JSON without NaN or Infinity. Full observations remain available; gzip and a bounded checkpoint-aware cache reduce repeated loading costs.

The automated suites cover data ingestion, chronological windows, DET checkpoint rejection, inference shapes and horizons, uploads, validation errors, asynchronous training jobs, condition chronology, parameter-specific observed flags, unknown quality, changing spatial samples, retrieval filtering and provenance, sensor applicability, gzip negotiation, JSON safety and HTTP boundaries. Loopback-dependent checks can be skipped when the execution sandbox prohibits socket access; a direct-handler gzip test also verifies response content and encoding headers without sockets.

Run the complete suite in the target environment:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The test count is intentionally not frozen in this document; the command's final summary is the authoritative run result. Interactive browser checks and startup verification should accompany code-level checks after changes to the frontend or launcher.

## Evidence validation scope

RAG uses local TF-IDF passage ranking and deterministic evidence-linked explanations, not a trained generative language model or full Tree-of-Thought search. Metadata filters include parameter and optional sensor/season. MODIS explanations exclude Sentinel-2-specific quality references; custom passages without sensor metadata are general by default.

The MODIS QA source supplies general inspection context only. It does not prove the source CSV was extracted from MOD09GA or that its pixels were cloud-masked. Missing rainfall remains unavailable; references do not become local measurements. Source-linked hypotheses are explicitly unconfirmed.

The included retrieval smoke benchmark uses a small hand-authored query set. It checks integration and labels its scope; it is not an independent scientific evaluation of explanation correctness or causal inference.

## Local and saved-workspace behavior

The local launcher starts the Python API at `http://localhost:8765`. A fresh machine needs compatible Python dependencies, including PyTorch for DET inference and training. The dashboard supports CSV uploads, dataset-specific training, saved-model inference, latest-condition analysis and reference imports in local API mode.

The saved workspace contains computed outputs from the same supplied real CSV and DET checkpoint. Browsing a snapshot cannot process a new dataset or train a model; those actions require the local engine. The map uses real geographic coordinates over optional Esri satellite or OpenStreetMap street tiles, with a plain-background fallback when tiles are unavailable. Leaflet JavaScript and styles are bundled locally. Only the two tile image providers are permitted by the local server's image policy; scripts and API connections remain same-origin. Tile-provider attribution is retained. Date filtering uses an exact observed date; markers are never mixed across observation dates. Invalid or missing readings are omitted for the selected parameter. Inferred zone labels do not imply surveyed water-body or management boundaries.

## Final workspace verification: 24 October 2026

All 55 automated tests passed in the prepared Python 3.12/PyTorch environment, including the loopback HTTP tests. Browser checks covered the responsive mobile and desktop views, real observation loading, zone selection, 30- and 90-day outlooks, a selected future date, matching RAG forecast evidence, data search, and reference search. A 730-day live API request returned all 13,140 rows in 4.74 seconds. An in-memory upload of the complete supplied CSV preserved the trained fingerprint and reused its DET checkpoint without retraining.

The final launcher starts the local engine successfully. Static source syntax and archive integrity were checked. These checks establish application behavior; they do not establish stronger scientific forecast accuracy than the recorded held-out scores.

## Interactive geographic map update

The dedicated Map navigation view and embedded overview/zone maps share pan/zoom controls, basemap switching, fit-to-data, expandable viewing, exact observation-date controls, parameter and zone filters, and clickable measurements. The default satellite basemap is provided by Esri; OpenStreetMap streets and a plain background are also available. A tile-load warning identifies network or provider failures without hiding local measurement markers. Provider imagery is contextual: it is not asserted to match the historical CSV observation date. The latest supplied observation date reports 326 coordinate records; 311 are valid for NDWI. Basemap imagery and uploaded measurements retain separate provenance.

Regression checks cover the locally bundled map dependency order, map navigation, and restrictive tile image policy with a suitable cross-origin referrer policy. Browser validation of map interaction is recorded with the current delivery checks.

## Map delivery verification: 26 October 2026

The local Leaflet dependency and license, Map navigation, exact-date filters, map initialization/removal, and tile security policy were checked. Map data checks passed for the supplied observations: 326 valid latest-date turbidity points and 311 valid latest-date NDWI points. Historical-date filtering, zone filtering, invalid readings and missing coordinates were also checked.

Browser verification confirmed Esri satellite imagery and OpenStreetMap streets loading, zoom changing from level 11 to 12, dragging changing the map center, exact historical-date selection, parameter switching, point selection opening all three recorded values and coordinates, and navigation from the selected point to its zone analysis. Responsive and expanded views were inspected. Map images are contextual and do not determine the CSV measurement date.

The non-HTTP suite passed 52 executed checks (with the loopback class skipped under restricted permissions). The four HTTP checks then passed separately with loopback permission, for 56 Python checks in total. The additional JavaScript map-data checks and source syntax checks passed. The archive was rebuilt to include water-map.js and all local Leaflet assets and its integrity was verified.

## Historical validation delivery: 26 October 2026

A separately trained frozen DET fits only training targets through June 2021, selects configuration and epoch using July–December 2021 targets, and scores 2022–10 February 2023 observations. Candidate 1, epoch 42 was selected. Every scored origin is on or after the last validation observation (27 December 2021); longer-horizon curves therefore begin later in January. The saved experiment has 3,129 exact comparison rows and 51/50/48 eligible target dates for the 1/2/4-observation heads. Production checkpoint hashes remained unchanged.

New safeguards test future-value and coordinate invariance of training preprocessing, per-head target partitions, model-availability chronology, actual native-calendar steps, frozen-checkpoint integrity, common-origin aggregation weights, correspondence with source zone means, exact metric recomputation, saved-workspace parity and incompatible-dataset rejection. Historical API loads saved results without importing Torch or invoking production inference.

The full Python suite ran 86 tests; 85 passed on that run and one existing HTTP rejection check encountered Windows connection-aborted error 10053. All four HTTP checks passed on an isolated recheck, completing all 86 distinct checks. JavaScript syntax, real-artifact UI/metric checks and geographic-map regression checks passed. The historical UI reproduces each selected metric from exact paired values, preserves negative R², and reports an empty comparison where no forecast is eligible.

The main training report remains the record for the original future-forecast model. HISTORICAL_VALIDATION.md and models/historical_validation/REPORT.md describe the separate historical checkpoint, source periods, exact scores and limitations.

Browser interaction verification completed: all three comparison graphs rendered; the future-only 30–730-day control was hidden on Historical validation; selecting the approximately 32-day horizon showed 48 dates and the first eligible target (25 January 2022) traced to a 27 December 2021 origin with 29 actual elapsed days. An early-January empty range showed zero comparisons and disabled its selected-data download. Zone filtering worked. Keyboard selection and a direct mouse click on a graph date both updated the forecast inspector to the correct recorded date and values. Transparent hit areas improve graph-point selection without changing lines or measurements. The completed view was captured in the delivery preview.

## DET refinement v3 verification: 26 October 2026

Pre-2022 audits identified numerical coordinate-bin churn, a source-zone/grid-anchor mismatch, early validation origins preceding model availability, and unequal task weighting. Coordinates are now canonicalized before fitting the grid; validation origins respect fitting availability; exact source-zone anchors are supported for residual DET candidates; training and selection balance temporal parameter scales. Six predefined DET configurations were searched without post-2021 targets. Direct-head candidate 5, epoch 54 was locked before the updated historical score was computed.

The 2022–2023 before/after comparison uses the same 3,129 targets and dates; actual values were checked unchanged within numerical tolerance. Results are reported without choosing a different model after inspecting test scores. Historical metrics and separately refitted future production weights remain distinct. The benchmark period has already been reviewed during development and is not fresh external validation.

The full Python suite passes **108 tests** in the prepared runtime. New tests cover coordinate microjitter invariance, genuine movement, past-only search enforcement, generic dates/cutoffs, grid/zone anchors, balanced objectives, checkpoint separation, format-v3 inference, NDWI band ordering/domain, long-horizon bounds, and checkpoint/data rejection. Existing data, API, map and historical validation checks pass. Source/selection fingerprint and locked-checkpoint SHA checks run before evaluation/refit.

Training curves distinguish the balanced training objective from validation normalized RMSE. Future metrics use the selected zone or origin-weighted lake summary when available. Future display defaults to 30 days; beyond 32 days remains experimental. Original reports and weights are archived, and FORECAST_REFINEMENT_REPORT.md records improvements and regressions.

Final browser checks verified v3 metrics, 32-day horizon and zone filtering (48 recorded target dates), and the date inspector tracing 25 January 2022 to a 27 December 2021 origin. The validation hero now explicitly labels the previously reviewed retrospective comparison, and its title contrast is corrected. JavaScript syntax validation passed after these presentation changes.
Future 30-day inference and RAG evidence loaded successfully in the browser. Activating the 9 January 2022 turbidity graph point updated the inspector to the 1 January origin and the exact saved actual/prediction values. The final comparison preview is saved outside the packaged source as outputs/AquaLens-Refined-DET.jpg.

## Present-water assessment delivery: 27 October 2026

Added a primary Water assessment page with lake/zone selection, latest recorded condition, five use categories, missing tests, conditional precautions, numbered source inspection and a Markdown report download. Uploading CSV observations opens this page. The map handoff preserves location scope and forecasting remains a separate next step.

Nine reference passages now include checked official WHO, CPCB, FAO and CDC guidance. Topic/sensor-filtered lexical retrieval supports deterministic assessment rules; citation text and applicability are inspectable. Custom or altered passages cannot impersonate verified policy references. The prototype explicitly withholds actions when matching evidence is absent. Satellite proxies never clear a physical water use or diagnose a harmful bloom. Missing readings and missing latest-zone observations remain withheld, and all responses identify the recorded observation date rather than claiming live conditions.

The full Python suite passes **125 tests**, including 11 assessment-engine and five API/evidence integration checks. They cover causal observation cutoffs, future-data invariance, observed/imputed flags, absent-zone latest dates, invalid NDWI, strict JSON, source substitution, reference absence, per-use evidence and conditional precautions. JavaScript syntax checks pass for the main application and assessment module. All original DET weights, historical prediction targets and source observations are preserved byte-for-byte.

The launcher now prefers the installed bundled data libraries over the development-only copies, avoiding the blocked development-library path observed on this computer. Startup and the selected-zone assessment endpoint were checked successfully on localhost.

Browser verification: lake/zone selection updates actual signals and dated condition; all five use categories render, arrow-key navigation works, irrigation citations open the correct FAO passage and applicability, recreation shows BOD and two supporting references, and the map handoff preserves selected zone Z1. A Markdown assessment was downloaded through the browser and checked for the date, local observations, missing tests, precautions and full references. An additional regression check ensures use-specific policy passages are not misused as optical-change cause evidence.

Saved-workspace verification completed through a static server without the Python API: all seven assessment locations and five use tabs load from the regenerated real-data snapshot. Citation integrity and physical-use withholding were checked across all saved reports; use-specific policy references do not appear as optical-cause evidence. Desktop and narrow-panel layouts were inspected with no horizontal overflow. Updated assessment screenshots are included as separate delivery previews.

## Three-indicator preliminary screening: 27 October 2026

Added a data-driven preliminary layer with observed-step direction, possible interpretation, next checks, and source citations for NDWI, turbidity proxy and chlorophyll proxy. Use cards now lead with screening guidance; the existing confirmation limits and missing measurements are secondary expandable details. Qualitative confidence states information limitations and is not a safety probability or forecast accuracy. Exported reports include the preliminary findings, confidence basis and use-specific guidance.

All **139 Python tests pass**. Thirteen new engine checks cover rising/falling/unchanged comparisons, current/previous observed flags, direct invalid-NDWI inputs, missing references, no comparison history, zone/lake direction differences, opposing zone changes, temporal cutoffs, and no physical-use clearance. Stable-only increases have observational headlines; stronger review wording requires a matching indicator's historical flag. An added API check verifies real-data preliminary content, citations and unchanged physical-use status. JavaScript syntax and frontend rendering/escaping checks pass.

The saved workspace contains all seven updated preliminary assessments. Saved forecasts and backtest data are retained exactly. All original source data, active DET weights and historical prediction pairs match the recorded pre-change SHA256 digests.

Browser verification completed on the local engine: whole-lake screening renders all three rising indicators and its review headline; selecting Z1 changes the turbidity direction to lower and uses a descriptive chlorophyll headline while the historical labels remain Stable. Irrigation screening and USGS evidence expansion work, the layout has no horizontal overflow, and the browser-downloaded Markdown report includes the preliminary findings, qualitative-confidence explanation, practical checks, per-use screening and references.

Final narrow-panel visual check found and corrected an intrinsic grid-width overflow in the new screening section. Explicit zero-minimum grid tracks now keep the page and cards within the available width; browser measurements confirm body scroll width equals client width (532 px) and the assessment card fits its 496 px container. The corrected preview was recaptured.

## Interface clarity redesign: 27 October 2026

Added a navy/teal visual system with a light mist background, larger text, readable notices/tables and consistent controls across every page. Plain-language navigation names and descriptions group monitoring, outlook and data/evidence pages. Narrow layouts use a page selector plus an accessible menu drawer. The drawer has one event owner, correctly overrides legacy hidden-menu rules, closes with Escape, restores focus to its trigger, and keeps its off-screen controls inert while closed.

Condition and overview pages provide four workflow shortcuts. Assessment copy is shorter, with detailed comparison/rationale available in expandable sections. References, numeric findings and suitability limits are preserved. The forecast page now has a clear location/indicator header, numbered chart/date/error/evidence sections, 8/16/30-day date shortcuts where applicable, a selected-shortcut state, plain error-metric explanations and expandable technical records. The training dialog remains available; no training was started during UI testing.

JavaScript syntax and the new frontend presentation contracts pass. The contracts verify required hook uniqueness, retained negative-score warnings and metric values, date-shortcut bounds, escaped user-controlled labels, safe source links, all five assessment uses and all nine navigation routes. The five historical-delivery checks and two map integration/security checks pass. The map-navigation contract now covers a desktop route and a native narrow-screen selector instead of two button lists.

Browser verification passed for all nine routes at both desktop and narrow-panel sizes with no horizontal overflow. Actual date shortcuts update the forecast input, selected state and all three prediction cards; zone changes update the outlook scope; the training dialog opens and closes without fitting a model; water-use guidance remains accessible. The menu opens visibly, Escape closes it and restores focus. Desktop previews were saved, and both connected and saved-workspace modes rendered the new interface.

Source observations, active bundled DET weights, saved forecast/historical data and assessment rules remain byte-identical to the recorded pre-redesign digests. The interface introduces no external fonts/scripts or new model dependency. UI_GUIDE.md documents the new page names and workflow.
