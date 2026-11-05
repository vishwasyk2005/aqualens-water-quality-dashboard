# DET forecast refinement report

The original forecasts were weak. This update corrects reproducible preprocessing and evaluation defects, retrains DET, and keeps errors visible. It does not establish operational water-quality prediction accuracy.

## What changed

- Tiny coordinate jitter (around 1e−10 degrees) previously moved physically identical sites across numerical grid boundaries. Coordinates are now canonicalized to six decimal degrees before fitting the training grid. This is numerical grouping, not sensor calibration. The frozen model grid is now 28 stable cells instead of 37 spurious input nodes.
- Validation forecasts can originate only after fitting is complete. Historical test forecasts can originate only after model selection is complete.
- Residual DET candidates use exact causal source-zone observations as their baseline, instead of unweighted, filled grid-cell means. The selected candidate uses learned direct DET heads and therefore does not add that mismatched grid baseline.
- Training errors and selection scores are balanced using training-only within-zone temporal scales. All candidates remain GRU + Child-Sum Tree-LSTM DET models; no mean-pooling predictor or alternative model is deployed.

## Selection before test scoring

Six fixed DET configurations were compared using only 2020–2021 observations. The selected model is candidate 5, direct heads, history window 12, hidden size 24, dropout 0.1, learning rate 0.001, weight decay 0.01, and epoch **54**.

The mean normalized validation RMSE decreased from **1.752870** to **1.735670**, a modest **0.98%** improvement. Each of the three parameters and three native horizons has equal weight after training-only temporal normalization. This is a validation score change, not an accuracy percentage.

Fitting uses 69 observed dates through 26 June 2021. Selection uses 23 observed dates from 4 July through 27 December 2021. No 2022–2023 readings enter this search. The selected checkpoint hash and search plan were locked before the updated historical score was computed.

## Recorded whole-lake results

Both actual and predicted summaries use identical source-observation-count weights from the forecast origin. All before/after targets and dates were checked to be identical. These lake-summary scores differ from pooled zone scores.

| Horizon | Parameter | Old RMSE | New RMSE | Old R² | New R² |
|---|---|---:|---:|---:|---:|
| ≈ 8 days | Chlorophyll proxy | 1.9491 | 1.9869 | 0.3606 | 0.3356 |
| ≈ 8 days | NDWI | 0.0818 | 0.0745 | -0.3064 | -0.0848 |
| ≈ 8 days | Turbidity proxy | 0.6404 | 0.3435 | -2.0869 | 0.1117 |
| ≈ 16 days | Chlorophyll proxy | 2.3566 | 2.0321 | 0.0725 | 0.3104 |
| ≈ 16 days | NDWI | 0.0950 | 0.0737 | -0.7985 | -0.0837 |
| ≈ 16 days | Turbidity proxy | 0.6935 | 0.3408 | -2.6688 | 0.1140 |
| ≈ 32 days | Chlorophyll proxy | 1.7037 | 1.9691 | 0.4592 | 0.2776 |
| ≈ 32 days | NDWI | 0.0810 | 0.0644 | -0.4345 | 0.0934 |
| ≈ 32 days | Turbidity proxy | 0.5115 | 0.3287 | -1.2147 | 0.0854 |

The 8-day whole-lake turbidity RMSE decreases substantially, but its R² is still only about 0.11. Whole-lake NDWI remains negative at the 8- and 16-day horizons. Some individual tasks worsen, including the 8-day whole-lake chlorophyll score. These remaining weaknesses are displayed rather than hidden.

## Pooled zone results

| Horizon | Parameter | Old RMSE | New RMSE | Old R² | New R² |
|---|---|---:|---:|---:|---:|
| ≈ 8 days | Chlorophyll proxy | 3.7442 | 3.5160 | 0.0783 | 0.1872 |
| ≈ 8 days | NDWI | 0.1289 | 0.1202 | 0.4551 | 0.5266 |
| ≈ 8 days | Turbidity proxy | 1.1019 | 0.8198 | -0.7649 | 0.0232 |
| ≈ 16 days | Chlorophyll proxy | 4.0953 | 3.5679 | -0.0851 | 0.1764 |
| ≈ 16 days | NDWI | 0.1345 | 0.1179 | 0.3839 | 0.5266 |
| ≈ 16 days | Turbidity proxy | 1.1236 | 0.7994 | -0.9370 | 0.0195 |
| ≈ 32 days | Chlorophyll proxy | 3.7946 | 3.5263 | 0.0358 | 0.1673 |
| ≈ 32 days | NDWI | 0.1272 | 0.1126 | 0.4217 | 0.5466 |
| ≈ 32 days | Turbidity proxy | 1.0456 | 0.7524 | -0.9087 | 0.0117 |

## What the experiment can establish

The 2022–2023 period has already been reviewed during prototype development, so it is a retrospective benchmark rather than fresh external validation. The current search did not use its values to select configuration or epoch. Confirmation of future generalization requires additional independent observations.

Only 69 fitting dates and 23 selection dates are available. The pre-2022 source already contains extreme, negative turbidity/chlorophyll proxy values and unknown satellite quality. Those readings were not silently deleted, clipped, or altered to improve the graphs. Their retrieval formulas and physical units remain unverified.

The historical graphs contain real recorded target dates, with 51/50/48 eligible dates for approximately 8/16/32 days. The model was available from 27 December 2021. Later rolling origins can use measurements only after they arrive. This is repeated short-term forecasting, not one prediction of an entire year.

A separate production DET is refitted on all available history for the selected 54 epochs to forecast after the latest CSV observation. The reported historical test scores belong to the frozen evaluation checkpoint, not that all-history refit. Daily future curves interpolate native heads; projections beyond 32 days remain experimental.

Historical scores use raw network outputs. Future NDWI display values and interval endpoints are clipped to the valid index domain; this is labelled in the exported forecast basis. The empirical future bands reuse selection-validation residuals, transfer to a different all-history refit, and have no guaranteed coverage. Historical comparison has no claimed uncertainty bands.

## Artifacts

- `models/refinement_v3/`: locked pre-2022 search plan, six candidates, validation metrics, checkpoint, history, original-reference audit and before/after historical comparison.
- `models/historical_validation/`: active frozen historical weights, 3,129 paired readings and current metrics.
- `models/`: separate active evaluation weights and all-history production weights for future forecasts.
- `models/archive/before_refinement_v3/`: original model weights, metrics and preprocessing retained as an audit trail.

Locked selection checkpoint SHA-256: `7ebaae4a227edb21dd4cd4259cf7bd48df7a918d1c62bf61012f3912363b6aae`

Source dataset fingerprint: `9157b8c74454d4a0180e0abacdc366cabc1bf9ab7d9cd61df751af12ef930d9b`
