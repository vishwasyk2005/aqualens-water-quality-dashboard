# Historical DET validation report

This separate DET checkpoint was fitted and selected with pre-2022 observations. Production forecasting weights were not replaced. Graphs show genuine errors without adjusting curves to make them match.

## Chronological protocol

| Partition | Allowed calendar period | Actual observations | Dates |
|---|---|---|---:|
| Training | 2020-01-01 through 2021-06-30 | 2020-01-01 to 2021-06-26 | 69 |
| Validation | 2021-07-01 through 2021-12-31 | 2021-07-04 to 2021-12-27 | 23 |
| Test | 2022-01-01 through 2023-02-10 | 2022-01-01 to 2023-02-10 | 51 |

The scaler, learned weights, input-grid coordinate bounds, and modeled cell IDs use the training period only. The last 2021 validation observation selects configuration and stopping epoch. Test forecasts are allowed only from **27 December 2021** onward, after the final validation observation has arrived.

The checkpoint stays frozen. Later rolling origins can use observations already available at those dates as inputs. This evaluates repeated short forecasts during 2022–2023, not one forecast of the entire period issued in 2021.

## Actual training

| Candidate | Window | Hidden size | Learning rate | Best epoch | Epochs run | Validation loss |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 24 | 0.003 | 42 | 60 | 0.206773 |
| 2 | 8 | 32 | 0.001 | 13 | 31 | 0.226776 |

Selected candidate: **1**. Selected epoch: **42**. Seed 42, AdamW, observed-target Smooth L1 loss, 160 maximum epochs and 18-epoch early stopping. No all-history refit was performed.

Input histories use the existing GRU and Child-Sum Tree-LSTM DET architecture, with 37 training-frozen model cells and six fixed geographic zones. Unknown future-only input cells are ignored. Targets retain all valid observed values within the fixed zones. Invalid NDWI and missing targets are masked; turbidity/chlorophyll remain unverified source proxies.

## Test coverage

| Approximate horizon | Native steps | First target | Last target | Dates |
|---|---:|---|---|---:|
| 8 days | 1 | 2022-01-01 | 2023-02-10 | 51 |
| 16 days | 2 | 2022-01-09 | 2023-02-10 | 50 |
| 32 days | 4 | 2022-01-25 | 2023-02-10 | 48 |

Actual elapsed days vary and are recorded for each prediction. Graphs use actual native observation dates, with no daily target interpolation. Longer horizons start later because their forecast origins must follow checkpoint selection.

## Actual errors

Pooled-zone metrics count zone/target pairs. Whole-lake summaries apply parameter-valid observed source-row counts at the forecast origin as the same weights to both actual and predicted zone values, using common valid zones. This differs from a target-date mean with a changing spatial sample.

### Pooled zone-target pairs

| Horizon | Parameter | MAE | RMSE | R² | Pairs |
|---|---|---:|---:|---:|---:|
| 8 days | chlorophyll_proxy | 2.1894 | 3.7442 | 0.0783 | 306 |
| 8 days | ndwi | 0.0976 | 0.1289 | 0.4551 | 306 |
| 8 days | turbidity_proxy | 0.6659 | 1.1019 | -0.7649 | 306 |
| 16 days | chlorophyll_proxy | 2.5440 | 4.0953 | -0.0851 | 300 |
| 16 days | ndwi | 0.1012 | 0.1345 | 0.3839 | 300 |
| 16 days | turbidity_proxy | 0.6651 | 1.1236 | -0.9370 | 300 |
| 32 days | chlorophyll_proxy | 2.3844 | 3.7946 | 0.0358 | 288 |
| 32 days | ndwi | 0.0936 | 0.1272 | 0.4217 | 288 |
| 32 days | turbidity_proxy | 0.5735 | 1.0456 | -0.9087 | 288 |

### Whole-lake common-weight pairs

| Horizon | Parameter | MAE | RMSE | R² | Pairs |
|---|---|---:|---:|---:|---:|
| 8 days | chlorophyll_proxy | 1.3734 | 1.9491 | 0.3606 | 51 |
| 8 days | ndwi | 0.0644 | 0.0818 | -0.3064 | 51 |
| 8 days | turbidity_proxy | 0.5360 | 0.6404 | -2.0869 | 51 |
| 16 days | chlorophyll_proxy | 1.8089 | 2.3566 | 0.0725 | 50 |
| 16 days | ndwi | 0.0720 | 0.0950 | -0.7985 | 50 |
| 16 days | turbidity_proxy | 0.5932 | 0.6935 | -2.6688 | 50 |
| 32 days | chlorophyll_proxy | 1.1217 | 1.7037 | 0.4592 | 48 |
| 32 days | ndwi | 0.0650 | 0.0810 | -0.4345 | 48 |
| 32 days | turbidity_proxy | 0.4055 | 0.5115 | -1.2147 | 48 |

Scores are mixed. Pooled NDWI shows signal; turbidity has negative R² at every horizon. Whole-lake aggregation has separate scores and must not borrow the pooled-zone R². Negative R² indicates larger squared error than a constant equal to the scored target mean. R² is not percentage accuracy.

No uncertainty bands are claimed in this experiment. All lower/upper bounds are unavailable. Source pixel QA, retrieval formulas and physical units remain unverified. The file ends on 10 February 2023 despite its filename.

## Audit files

- `det_historical.pt`: immutable pre-2022 validation-selected weights.
- `summary.json`: split boundaries, histories, configuration, metrics and fingerprint.
- `heldout_predictions.csv` / `predictions.json`: 3,129 native actual/predicted pairs including lake summaries.
- `metrics.csv`: pooled-zone scores; `metrics_by_zone.csv`: per-zone and lake scores.
- `training_history.csv`, `candidate_history.csv`, `tuning.csv`: recorded learning and selection.
- Reproduce with `python scripts/train_backtest.py` (separate output folder).

The forecast-origin availability guard was strengthened during review. Existing selected weights, scaler, configuration, best epoch and validation losses were retained. Only ineligible early long-horizon test pairs were removed and the remaining pairs rescored. There was no tuning on the test scores. Input tensor compatibility was verified exactly after the training-grid threshold guard repair.

Checkpoint SHA-256: `cde9747841facf361ba59f4d886e639ac0f967b3577b6ad3739a1584fb001bce`

Source dataset fingerprint: `9157b8c74454d4a0180e0abacdc366cabc1bf9ab7d9cd61df751af12ef930d9b`
