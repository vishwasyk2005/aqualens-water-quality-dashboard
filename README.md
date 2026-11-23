# AquaLens: Satellite water observatory

A custom interactive dashboard with a local Python analysis engine. Upload satellite-derived CSV observations, assess the latest recorded condition and possible water uses, review precautions and their retrieved evidence, then explore forecasts selected by chronological comparison. Forecast candidates include Prophet from the original prototype and DET, which can be retrained on an uploaded CSV. See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) and [WATER_ASSESSMENT.md](WATER_ASSESSMENT.md).

The included DET was trained on the supplied MODIS-named CSV. This workspace uses real supplied observations, not synthetic demo measurements. The file's satellite acquisition provenance and parameter formulas have not been independently verified.

## Start the dashboard

On Windows, extract the complete folder and double-click **start.bat**. Open **http://localhost:8765** and keep the launcher window open. The launcher prefers the bundled runtime data libraries on the development computer; elsewhere it creates `.venv` and installs the training requirements, including PyTorch, which is needed for saved-model forecasts and retraining. The first setup needs an internet connection.

For a fresh installation, use Python 3.12 and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-training.txt
.\.venv\Scripts\python server.py --port 8765
```

`requirements-training.txt` includes NumPy, Pandas and Prophet through `requirements.txt`, plus PyTorch. CPU PyTorch is sufficient. The frontend has no build step or required CDN libraries. The API binds to `127.0.0.1`.

## Interactive workspace

The new frontend uses top navigation for Monitor, Map, Forecast, Accuracy, Sources, and Data. See [UI_GUIDE.md](UI_GUIDE.md) for the workflow.

- **Water assessment:** latest recorded lake/zone condition, preliminary interpretation of all three indicator trends with qualitative confidence, five interactive use categories with practical screening guidance, confirmation requirements, conditional precautions, numbered source evidence, and a downloadable assessment. Proxy-only data does not clear physical water uses.
- **Overview:** current-condition cards, historical trends, parameter controls, a geographic measurement map, zone comparisons, and evidence for the latest observations.
- **Interactive map:** a dedicated Leaflet map with satellite and street basemaps, drag/pan, wheel or button zoom, full-screen expansion, exact observation-date selection, parameter and zone filters, and clickable measurement details.
- **Future outlook:** choose Auto, Prophet, DET, or another evaluated model. Explore forecasts by date and location, download estimates, and review historical errors for the selected model. Projections beyond 32 days are experimental.
- **Check accuracy:** compare each benchmark candidate against the same recorded 2022–early-2023 readings. Filter by location, indicator and observation horizon, and inspect actual readings, predictions, errors and forecast origins.
- **Zone intelligence:** selected-zone measurements, history, monitoring status, forecasts where available, and source-backed interpretation.
- **Alerts:** filters for exploratory historical-change signals and links to the affected zones.
- **Evidence:** parameter-aware reference search, source passages and applicability, and import of a custom JSON reference collection.
- **Data:** CSV upload, processed records, quality and provenance warnings, hierarchy details, and observation downloads.
- **Training:** background DET training, actual epoch/loss updates, saved learning curves, chronological test metrics, and checkpoint compatibility checks.

The map uses actual uploaded latitude/longitude values over Esri satellite imagery or OpenStreetMap streets. A plain-background option works without basemap imagery. Leaflet is bundled locally; basemap tiles require an internet connection. Tile requests expose the viewed map area to the selected provider, but the application does not send CSV files or parameter values to those providers. Provider attribution stays visible.

Only measurements from the selected observation date and selected parameter are drawn: the latest date in the supplied CSV has 326 reported measurements, of which 311 have valid NDWI. Missing or invalid values are not drawn as valid measurements, and no invented lake outline or zone boundary is shown. Marker colors represent the selected parameter's observed range, not drinking-water safety thresholds. Zones inferred from coordinates are a geographic grid, not verified inlet/outlet or management boundaries.

The full API supports uploads and retraining. A saved real-data snapshot in `web/demo-state.json` supports browsing when the Python API is unavailable; its legacy filename does not mean synthetic data. Snapshot mode cannot train or process a new uploaded dataset. Serve `web/` over HTTP rather than opening it with `file://`.

## Included data and CSV upload

The supplied file contains **46,755 observations on 143 dates, from 1 January 2020 through 10 February 2023**, across 981 distinct reported coordinates and six inferred zones. Its filename mentions 2020-2025, but its actual observations end in 2023. Median observation spacing is eight days.

The original project format is accepted directly:

```text
system:index,Chlorophyll,NDWI,Turbidity,.geo
```

`system:index` must contain a `YYYY_MM_DD` date. `.geo` is a GeoJSON Point with coordinates in `[longitude, latitude]` order. The parser handles suffixes after the date.

Canonical uploads require:

```text
date,lat,lon,ndwi,turbidity_proxy,chlorophyll_proxy
```

Recommended metadata: `lake_id,zone_id,zone_name,cell_id,valid_pixel_fraction,is_observed,sensor,source_kind,algorithm_version`. Optional weather fields are `rainfall_mm,temp_c`. Upload one lake and one compatible sensor/algorithm at a time. Supply stable zone and cell identifiers when available.

Processing validates dates and coordinates, removes duplicate date/cell rows, preserves raw parameter values, and handles only short causal gaps within the same cell. Invalid NDWI outside [-1,1] is preserved in raw fields and excluded from analysis: **2,476 such readings occur in the included CSV**. Negative/extreme turbidity and chlorophyll values are retained with warnings because their retrieval formulas and units are unknown. No percentile clipping hides anomalies. Parameter-specific observation flags distinguish measured values from filled values.

Missing pixel-quality metadata remains unknown. Valid measurements can still support statistical change analysis, qualified by unknown quality and changing spatial sampling. The system never treats missing QA as verified clear imagery.

The browser accepts files below 11 MB to fit the server's 12 MB JSON-request limit. Up to five uploaded datasets are held in memory; restarting the API discards those sessions. Saved DET checkpoints remain under `models/custom/<fingerprint>/` and only match the dataset they were trained on. Auto compares lightweight candidates for sufficiently long new uploads using that upload's own chronological split. An upload with too little usable history can still be assessed and mapped.

## What "present condition" means

Present means the **latest observation date in the selected CSV**, not today's date or a live satellite feed. For the included file, that is **10 February 2023**.

Stable, Watch, and Unusual describe robust deviation from the preceding 12 weeks of available observations. At least six usable prior periods are required; otherwise the system reports Insufficient data. These are statistical monitoring categories, not regulatory thresholds, confirmed pollution diagnoses, or drinking-water safety assessments. Unknown satellite quality and changes in the sampled locations can affect interpretation. Overview parameter status can reflect a flagged zone even when the whole-lake average is comparatively stable.

## Current DET architecture and training

DET passes ordered cell histories through a GRU and Child-Sum Tree-LSTM cell/zone/lake hierarchy. Six predefined regularized DET configurations were tested, using either exact observed-zone residual anchors or learned direct output heads. The selected DET uses **direct heads, 12 historical observations, 24 hidden units, dropout 0.1, learning rate 0.001, weight decay 0.01 and epoch 54**.

Coordinates are canonicalized to six decimal degrees before fitting grid boundaries. This prevents numerical jitter from moving the same physical sites between training cells. The supplied data now uses 28 stable model cells; zone targets remain the unchanged, valid observed source means.

The bundled refined model was selected using training targets through June 2021 and origin-safe validation targets from July–December 2021. Selection balances the three parameters and three native horizons using training-only within-zone temporal scales. The checkpoint was locked before scoring 2022–February 2023. This period has already been reviewed during development, so it is a retrospective benchmark rather than new independent external validation.

A separate production DET is refitted on all available historical observations for the selected 54 epochs to forecast after the latest CSV date. It is distinct from the frozen evaluation weights. The original models and results remain in `models/archive/before_refinement_v3/`.

Training through the dashboard or `scripts/train.py` supports other uploaded data using chronological 70/15/15 cutoffs. Only the earlier training and validation frame enters configuration selection; later test targets are scored afterwards. Each forecast origin must follow availability of the fitted/selected model. At least **35 distinct observation dates** are required so the fitting partition has at least 24 dates; 80 or more are recommended. Training can stop before the requested epoch limit.

```powershell
.\.venv\Scripts\python scripts/train.py --epochs 180
.\.venv\Scripts\python scripts/train.py --data path/to/lake.csv --output models/custom_run --epochs 180
```

Dashboard-created custom checkpoints are discovered under `models/custom/<fingerprint>/`. Standalone CLI output folders are for inspection unless placed there or used as the bundled `models/` folder. Historical validation uses its separate pre-2022 experiment and is not overwritten by a future-model retraining job.

## Accuracy and uncertainty

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the current comparison across forecasting families and the Auto selection. The paragraphs below describe the archived DET refinement and the explicit DET option.

See [FORECAST_REFINEMENT_REPORT.md](FORECAST_REFINEMENT_REPORT.md) for all before/after results, unchanged-target checks and remaining weaknesses. The 8-day origin-weighted whole-lake turbidity RMSE fell from approximately **0.6404 to 0.3435**, but R² remains only about **0.112**. Whole-lake NDWI still has negative R² at 8/16 days. Some tasks worsen. The dashboard displays the selected location's own historical metrics, rather than substituting pooled zone scores for a lake-summary forecast.

The validation selection score improved by only about **0.98%** over the archived DET on the same origin-safe pre-2022 comparison. This is not an accuracy percentage. Reliable future water-quality forecasting has not been established; the unknown retrieval algorithms, extreme proxy values and absent satellite QA remain limitations.

The native heads predict 1/2/4 observation steps, approximately 8/16/32 days, with actual elapsed days retained in historical results. The future page defaults to 30 days. Daily display values interpolate native outputs. Longer projections remain available, explicitly experimental, and have no uncertainty bounds beyond32 days.

Zone bands are empirical nominal-80% absolute residual quantiles from the same validation used for selection; they are not independent confidence guarantees. Widths transfer to separately refitted production weights, whose coverage is unverified. Whole-lake aggregate intervals are omitted. Historical metrics use raw DET outputs; future NDWI display values and band endpoints are limited to the index domain and labelled in forecast exports.

## Saved model and audit files

- `models/det_evaluation_v2.pt`: active format-v3, train-only refined evaluation weights; filename retained for compatibility.
- `models/det_v2.pt`: active format-v3 all-history production refit.
- `models/model_selection/`: chronological comparison report and exact recorded/predicted pairs by candidate.
- `models/historical_validation/`: frozen historical checkpoint, exact comparison pairs and current scores.
- `models/refinement_v3/`: locked pre-2022 six-candidate plan, validation scores, selected checkpoint and before/after audit.
- `models/evaluation.json`, `evaluation.csv`, `evaluation_by_zone.csv`: active configuration, history, limitations, fingerprint and scores.
- `models/heldout_predictions.csv`, `training_history.csv`, `tuning.csv`, `interval_coverage.csv`, `training_audit.json`: exact targets, forecast origins, training and uncertainty records.
- `models/production_forecast_730days.csv`: current production projections, with explicit long-horizon limits.
- `models/archive/before_refinement_v3/`: original models, preprocessing, reports and scores retained for comparison.

## RAG evidence and explanations

The evidence layer uses local, metadata-filtered **TF-IDF lexical retrieval** over curated source paraphrases, followed by deterministic evidence-linked explanations and candidate hypotheses. The selected forecast model supplies the prediction; retrieval supplies supporting context.

Explanations connect the latest observation, its historical change, the selected forecast and relevant source passages. They never invent rainfall when the CSV lacks weather data or assert that a general reference proves a local cause. Source titles, URLs, dates where known, applicability and method provenance are retained. MODIS explanations filter out Sentinel-2-specific references. The MODIS QA reference describes general quality checks; it does not establish which MODIS product generated this CSV.

Custom evidence imports are JSON lists of up to 200 passages. Each passage needs `source_id,title,url,text,applicability`; URLs must use HTTPS. Optional metadata includes `date,verified_on,parameters,seasons,sensors`. Omitted sensor tags default to `all`. Text is treated as evidence data, never executed as instructions. Imported collections are isolated to their local session.

## API examples

```text
GET /api/bootstrap?dataset=modis
GET /api/model-benchmark?dataset=modis
GET /api/forecast?dataset=modis&model=auto&horizon_days=30
GET /api/forecast?dataset=modis&model=det&horizon_days=30
GET /api/predict-date?dataset=modis&date=2023-03-01
GET /api/explain?dataset=modis&zone=all&parameter=turbidity_proxy&horizon_days=32
GET /api/evidence?q=cloud%20shadow&parameter=ndwi&sensor=MODIS
GET /api/evaluation?dataset=modis
GET /api/export?dataset=modis
POST /api/upload       { "csv": "...", "filename": "lake.csv" }
POST /api/knowledge    { "records": [...] }
POST /api/train        { "dataset": "modis", "epochs": 180 }
GET /api/jobs/<job_id>
```

Use the returned upload dataset ID for its later requests, and the returned `evidence_id` to select an imported evidence collection. The `demo` dataset alias points to the same real bundled observations for compatibility. Forecast horizons are 1-730 days; the historical `horizon` query parameter remains a weeks-based compatibility option.

## Optional Earth Engine extraction

No authenticated GEE download was performed for the supplied dataset. A separate extraction helper remains available:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-gee.txt
.\.venv\Scripts\python scripts/export_gee.py --project YOUR_PROJECT --area path/to/lake.geojson --lake-id study_lake --start 2020-01-01 --end 2025-01-01 --authenticate
```

That helper queues a **Sentinel-2** export with its own reflectance proxies. Supply an authorized Earth Engine project, lake boundary and appropriate projected CRS. Its outputs are not interchangeable with the current MODIS-named measurements; use consistent algorithms and train a separate compatible DET before comparing them.

## Verify or regenerate the saved workspace

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts/export_dashboard_demo.py
```

The snapshot exporter requires a matching trained real-data checkpoint. See `VALIDATION.md` for the validation scope and `models/evaluation.json` for the exact scientific record.

Original project: [Satellite-Driven Water Quality Monitoring System](https://github.com/vishwasyk2005/Satellite-Driven-Water-Quality-Monitoring-System-)

`data/demo.csv` is an isolated synthetic fixture for ingestion tests; the application defaults to `data/modis_observations.csv`, and the bundled trained weights and saved workspace use the supplied real observations.

## Historical validation: actual versus predicted

Choose **Historical validation** in the dashboard for three actual-versus-DET graphs, zone/horizon/date controls, error scores, a forecast-date inspector and CSV/audit downloads. A separately trained frozen model fits only 2020–June 2021, selects settings on July–December 2021, and scores 2022–10 February 2023 targets after selection is available. The original production model remains for future forecasting. See [HISTORICAL_VALIDATION.md](HISTORICAL_VALIDATION.md) for the recorded results and exact method.

To reproduce the historical training run:

```powershell
.\.venv\Scripts\python scripts/train_backtest.py --epochs 160
```
