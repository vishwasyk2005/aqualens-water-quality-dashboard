# Historical validation: actual readings versus DET predictions

The prototype now has a **Historical validation** page showing actual observations and independent DET predictions across 2022 and early 2023. This experiment uses a separate checkpoint from the all-history model used for future forecasts.

## How to use the page

Open the dashboard and choose **Historical validation**. Select a zone or the origin-weighted whole-lake summary, then choose approximately 8, 16 or 32 days ahead. Three graphs show NDWI, turbidity proxy and chlorophyll proxy together with MAE, RMSE, R² and the number of paired targets. Blue points/lines are actual observations; orange dashed lines are DET predictions.

Select a graph point or a target date to inspect when the forecast was made, its actual elapsed days, its observed value, its prediction and its signed error. Date filters recompute scores using the visible pairs. Download the selected comparisons, all results or the training audit from the page.

## Data separation

| Stage | Policy period | Actual observation dates | Date count |
|---|---|---|---:|
| Fit weights and scaler | 1 Jan 2020–30 Jun 2021 | 1 Jan 2020–26 Jun 2021 | 69 |
| Select configuration and stopping epoch | 1 Jul–31 Dec 2021 | 4 Jul–27 Dec 2021 | 23 |
| Score unseen targets | 1 Jan 2022–10 Feb 2023 | 1 Jan 2022–10 Feb 2023 | 51 |

The model becomes available after the final validation observation, **27 December 2021**. Every scored forecast origin is on or after that date, and every target is later than its origin. No 2022–2023 target values fit weights, fit the scaler, select hyperparameters or select the stopping epoch.

This is a rolling historical test: at each origin the model uses only measurements already available on that date. The same frozen weights issue each prediction. Later 2022 observations can enter later input histories after they arrive. This tests repeated short forecasts through the period, rather than a single year-long forecast made in 2021.

| Native horizon | First eligible target | Last eligible target | Dates per parameter/zone |
|---|---|---|---:|
| 1 observation, approximately 8 days | 1 Jan 2022 | 10 Feb 2023 | 51 |
| 2 observations, approximately 16 days | 9 Jan 2022 | 10 Feb 2023 | 50 |
| 4 observations, approximately 32 days | 25 Jan 2022 | 10 Feb 2023 | 48 |

Calendar gaps vary, especially at year boundaries. The CSV and date inspector retain each actual elapsed duration. Only real recorded target dates are plotted; no daily target values are interpolated.

## Actual training run

DET uses the existing GRU history encoder and Child-Sum Tree-LSTM geographic hierarchy. Grid bounds, modeled cells and the input scaler are frozen from the training period. Parameter-specific masks exclude invalid or missing targets; input filling is causal and carries missingness/age features. Fixed source zone identifiers are study-area metadata.

Two DET configurations were trained using seed 42, AdamW, a maximum of 160 epochs and early stopping after 18 epochs without validation improvement. Validation selected **candidate 1, epoch 42**, with history window 12 observations, hidden size 24 and learning rate 0.003. Training histories and both candidates' selection scores are saved. No production refit is used in this historical experiment.

The availability audit excluded early long-horizon forecasts whose origins preceded completion of model selection. This eligibility correction rescored the unchanged selected checkpoint; it did not retrain or tune the model against test scores.

## Recorded results across the six zones

These scores pool valid zone-target pairs. Whole-lake summary rows are excluded from this pooling.

| Approximate horizon | Parameter | MAE | RMSE | R² | Pairs |
|---|---|---:|---:|---:|---:|
| 8 days | Chlorophyll proxy | 2.1894 | 3.7442 | 0.0783 | 306 |
| 8 days | NDWI | 0.0976 | 0.1289 | 0.4551 | 306 |
| 8 days | Turbidity proxy | 0.6659 | 1.1019 | -0.7649 | 306 |
| 16 days | Chlorophyll proxy | 2.5440 | 4.0953 | -0.0851 | 300 |
| 16 days | NDWI | 0.1012 | 0.1345 | 0.3839 | 300 |
| 16 days | Turbidity proxy | 0.6651 | 1.1236 | -0.9370 | 300 |
| 32 days | Chlorophyll proxy | 2.3844 | 3.7946 | 0.0358 | 288 |
| 32 days | NDWI | 0.0936 | 0.1272 | 0.4217 | 288 |
| 32 days | Turbidity proxy | 0.5735 | 1.0456 | -0.9087 | 288 |

NDWI has the strongest pooled zone scores. Turbidity has negative R² across all three horizons; chlorophyll performance is mixed. The plots retain errors and divergence. R² is not an accuracy percentage, and successful execution does not establish reliable environmental forecasting.

## Whole-lake summary results

Both the actual and predicted summary use identical zone weights derived from parameter-valid source observation counts **at the forecast origin**. Weights are renormalized over common scored zones. This is a defined origin-weighted evaluation summary, rather than the changing-sample mean of all target-date CSV cells. Its actual curve can differ between horizons because those horizons have different origins and weights. Zone-level and lake-summary R² values describe different target series and should not be compared as equivalent scores.

| Approximate horizon | Parameter | MAE | RMSE | R² | Pairs |
|---|---|---:|---:|---:|---:|
| 8 days | Chlorophyll proxy | 1.3734 | 1.9491 | 0.3606 | 51 |
| 8 days | NDWI | 0.0644 | 0.0818 | -0.3064 | 51 |
| 8 days | Turbidity proxy | 0.5360 | 0.6404 | -2.0869 | 51 |
| 16 days | Chlorophyll proxy | 1.8089 | 2.3566 | 0.0725 | 50 |
| 16 days | NDWI | 0.0720 | 0.0950 | -0.7985 | 50 |
| 16 days | Turbidity proxy | 0.5932 | 0.6935 | -2.6688 | 50 |
| 32 days | Chlorophyll proxy | 1.1217 | 1.7037 | 0.4592 | 48 |
| 32 days | NDWI | 0.0650 | 0.0810 | -0.4345 | 48 |
| 32 days | Turbidity proxy | 0.4055 | 0.5115 | -1.2147 | 48 |

## Saved evidence and limits

The experiment saves 3,129 exact actual/predicted comparison rows across six zones, the whole-lake summary, three parameters and three horizons. Artifacts are in `models/historical_validation/`: `det_historical.pt`, `summary.json`, `heldout_predictions.csv`, `predictions.json`, `metrics.csv`, `metrics_by_zone.csv`, `training_history.csv`, `candidate_history.csv` and `tuning.csv`.

No uncertainty bands are claimed for this historical run. Satellite pixel QA, retrieval algorithms and physical proxy units remain unverified. These comparisons evaluate forecast agreement and do not determine drinking-water safety or the causes of water-quality changes.

The historical API rejects mismatched source data and invalid prediction chronology. The saved workspace includes the same real comparisons for viewing without the Python engine. Retraining the future-forecast model does not replace these independent historical results.

Checkpoint SHA-256: `cde9747841facf361ba59f4d886e639ac0f967b3577b6ad3739a1584fb001bce`

Dataset fingerprint: `9157b8c74454d4a0180e0abacdc366cabc1bf9ab7d9cd61df751af12ef930d9b`
