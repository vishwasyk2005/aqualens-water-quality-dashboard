# AquaLens DET training report

This report covers the original future-forecast model and its 70/15/15 evaluation. The newer pre-2022 historical experiment is separate: see [HISTORICAL_VALIDATION.md](HISTORICAL_VALIDATION.md) for the 2022–2023 actual-versus-predicted graphs, epoch-42 checkpoint and recorded scores.

The prototype now contains a DET trained on the supplied historical MODIS CSV. It does not use synthetic demo observations for these saved weights. DET is the only forecasting model. The results below come from saved training and evaluation artifacts, and show that accuracy is mixed: turbidity forecasting remains weak.

## Data used

- Source: `data/modis_observations.csv`, copied from the user-provided `MODIS_Lake_Features_NDWI_Turbidity_Chlorophyll_2020_2025_.csv`.
- Rows: **46,755**. Distinct observation dates: **143**.
- Actual date range: **1 January 2020 to 10 February 2023**. The filename does not establish coverage through 2025.
- Original geographic sample locations: **981**.
- Median gap between observation dates: **8 days**. Gaps vary, including shorter year-boundary gaps and an occasional longer gap. These are not daily observations.
- Spatial structure: six zones inferred from a 3-by-2 geographic grid. For training, locations are aggregated into a 6-by-6 geographic grid intersected with the zone boundaries, producing **28 model cells across six zones**.
- These grid zones are inferred geographic groups, not verified ecological boundaries or field-surveyed management zones.

The provided file was used directly. This training run did not fetch new observations from Google Earth Engine. Satellite acquisition provenance, retrieval algorithms and parameter units have not been independently verified.

Source file SHA-256:

`d34f086580a512d91a95f629657e7fe8d0ff637e606711f43a46e16f39e28a6b`

## Quality checks and missing measurements

The ingestion pipeline retains the original measurements alongside processed values. **2,476 NDWI values outside [-1, 1] are excluded** from analysis and targets, with raw values preserved. This uses the documented NDWI column's index domain; the upstream calculation still needs verification. Values are not clipped into an apparently valid range.

There are **1,785 rows containing a negative turbidity or chlorophyll source value**. Those proxy values are retained and flagged because the formulas and units are unknown. Their names alone do not justify treating them as physical concentrations or assigning drinking-water safety labels. Extreme proxy values are not automatically removed to improve model scores.

Every parameter has its own observation mask. Missing or invalid target measurements are excluded from the loss and evaluation rather than replaced with invented labels. Model input histories may carry a preceding valid cell measurement forward for up to 16 days, accompanied by missingness and observation-age features. Leading gaps and older gaps fall back to the training mean for numerical input only; the missingness information remains visible to the model. Future observations are never used to fill earlier inputs.

Pixel coverage is unknown for this entire supplied file. No rows are excluded by the known-coverage threshold because no coverage values are available. Unknown coverage is not interpreted as perfect image quality. The model panel has approximately **3.77% missing cell-parameter entries** before causal input filling.

## DET architecture

Each model cell supplies an ordered historical sequence. Each time step has 12 features: three standardized parameter values, three observation masks, three measurement ages, two seasonal features and the elapsed gap since the previous observation.

A GRU learns the time pattern for each cell. A genuine Child-Sum Tree-LSTM then combines cell representations into zone representations. A second Child-Sum Tree-LSTM combines the zones into a lake representation. Forecast heads use the learned zone and lake information to predict NDWI, turbidity proxy and chlorophyll proxy for each zone.

The heads predict residual changes from the latest causal cell-level parameter values. The three direct targets are **1, 2 and 4 native observations ahead**, approximately **8, 16 and 32 days**. The exact elapsed days are retained in `models/heldout_predictions.csv`, because native observation gaps are not perfectly constant.

There is no mean-pooling forecasting model, Prophet model, ridge model or trained risk-classification model in this implementation. Geographic aggregation and calculation of observed zone summaries are data preparation operations. RAG retrieves supporting information for explanations; it does not produce the numerical forecasts.

## Training and evaluation separation

Observation dates are split chronologically before fitting the scaler or constructing eligible target windows:

| Partition | Dates | Eligible windows for selected configuration |
|---|---|---:|
| Training | 1 January 2020 to 26 February 2022 | 89 |
| Validation | 6 March 2022 to 13 August 2022 | 18 |
| Test | 21 August 2022 to 10 February 2023 | 19 |

The first validation date follows the training cutoff in the actual observation calendar. Each eligible window contains the required preceding history and all three forecast targets. A training window's latest target must fall within training. Validation and test targets must stay inside their respective partitions. Consequently, not every date becomes an eligible forecast origin, and no target window crosses a partition boundary.

Only the training partition fits the evaluation scaler and evaluation model weights. Validation chooses the architecture settings and stopping epoch. The held-out test period is scored after selecting the model, and its results are not used to tune another candidate.

Evaluation uses rolling origins: observations that have become available by a later test origin may be used in that origin's input history. These results are not a single long forecast made at the beginning of the entire test period.

Training uses seed 42, deterministic PyTorch operations, mini-batches of up to 16 windows, AdamW, weight decay .001, Smooth L1 loss over observed standardized targets, and gradient clipping at norm 1. Each candidate allows up to 160 epochs, stopping after 18 epochs without sufficient validation improvement.

## Configuration selection and final refit

Both candidates are DET architectures; no alternative forecasting family was trained.

| Candidate | History length | Hidden size | Learning rate | Best epoch | Epochs run | Best validation loss |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 12 observations | 24 | .003 | 37 | 55 | .057331 |
| **2, selected** | **8 observations** | **32** | **.001** | **51** | **69** | **.054833** |

These are normalized training-objective losses, not accuracy percentages. Eight observations represent approximately 64 days at the median cadence, with actual spans varying.

Two checkpoints serve distinct purposes:

- `models/det_evaluation_v2.pt` contains the training-partition-only weights selected using validation. **All test scores in this report belong to this checkpoint.**
- `models/det_v2.pt` is a separate production model retrained from initialization for **51 epochs on all available eligible historical windows**. Its scaler is also fitted on all available history. This gives future forecasting access to the full uploaded dataset, but there is no remaining independent held-out test period for these production weights.

The epoch count and settings for the production refit were selected before held-out scoring. Reported evaluation scores must not be presented as an independent test of the all-history production checkpoint.

## Actual held-out results

Each row pools **114 observed zone-target pairs**: 19 forecast origins across six zones. MAE and RMSE use the parameter's supplied proxy scale; verified concentration units are unavailable. R² is a statistical score, not a percentage accuracy.

| Approximate horizon | Parameter | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| 8 days | NDWI | .097010 | .127471 | .470554 |
| 8 days | Turbidity proxy | .546841 | 1.076666 | **-.314453** |
| 8 days | Chlorophyll proxy | 1.914460 | 3.038646 | .262222 |
| 16 days | NDWI | .100716 | .131092 | .472023 |
| 16 days | Turbidity proxy | .534290 | 1.017443 | **-.176683** |
| 16 days | Chlorophyll proxy | 2.333870 | 3.491945 | .104277 |
| 32 days | NDWI | .105702 | .137554 | .462641 |
| 32 days | Turbidity proxy | .570052 | 1.028664 | **-.202239** |
| 32 days | Chlorophyll proxy | 2.354269 | 3.559663 | .076453 |

NDWI has the strongest scores among these three outputs. Chlorophyll performance is modest and declines at longer horizons. **Turbidity has negative R² at every evaluated horizon.** Under the R² definition, its squared error exceeds that of a constant equal to the held-out target mean; this is a diagnostic interpretation, not another trained model. The turbidity forecasts should not be described as reliable environmental predictions.

Proper training and honest evaluation do not guarantee good accuracy. Improving the source algorithms, quality flags, spatial boundaries and historical coverage is a substantive next step. More epochs alone do not resolve these data and generalization limits.

## Empirical interval performance

Displayed bands use an absolute-residual quantile with a nominal 80% level, pooled across zones for each parameter and native horizon. **The same validation partition both selected the model and estimated these widths.** There is no independent calibration partition and no guaranteed conformal coverage. Observations are temporally dependent.

Actual held-out coverage is:

| Approximate horizon | NDWI | Turbidity proxy | Chlorophyll proxy |
|---|---:|---:|---:|
| 8 days | 71.9% | 58.8% | 77.2% |
| 16 days | 78.1% | 56.1% | 68.4% |
| 32 days | 73.7% | 47.4% | 61.4% |

All are below the nominal 80%, with particularly poor turbidity coverage. These should be called **empirical uncertainty bands**, not guaranteed 80% confidence intervals. The production refit reuses evaluation-model residual widths; its coverage is not independently validated and can differ further.

## Daily display and long-range limits

Forecasts start after **10 February 2023**, the latest actual observation in this file. A dashboard date in 2026 does not make these observations current.

The interface supports a future-date picker and up to 730 days. Daily values within the direct horizon interpolate the native DET predictions; they do not represent daily satellite observations or independently evaluated daily forecasts. Uncertainty widths are also interpolated between native horizons, without independent validation at intermediate days.

**Beyond approximately 32 days, outputs are experimental recursive DET projections.** Synthetic projected input histories are fed back to the same model. No held-out long-range accuracy claim is made, and lower/upper interval bounds are absent beyond 32 days. These long-range outputs are intended for prototype exploration, not established operational forecasts.

A technical check produced 13,140 finite prediction rows for six zones, three parameters and 730 dates, from 11 February 2023 to 9 February 2025. Passing this check establishes that the feature runs; it does not establish prediction accuracy.

## Audit files and verification

- `models/training_audit.json`: source file identity, ingestion report, grid structure and model-input checksum.
- `models/evaluation.json`: configuration, split boundaries, candidate histories, limitations and calibration information.
- `models/evaluation.csv` and `models/evaluation_by_zone.csv`: overall and zone-specific held-out metrics.
- `models/heldout_predictions.csv`: each scored prediction, its actual value, origin date, target date and exact elapsed days.
- `models/tuning.csv` and `models/training_history.csv`: configuration selection and loss curves.
- `models/interval_coverage.csv`: empirical held-out band coverage.
- `models/production_forecast_730days.csv`: the checked exploratory production forecast.

Seven focused tests pass for temporal separation and scaling, missing-target handling, invalid NDWI, coverage filtering, tree child-order invariance and gradients, model/data compatibility, forecasting horizons and long-range interval exclusions.

The dataset compatibility fingerprint includes coordinates, raw values and observation flags, so changing the inputs requires compatible retraining. A metadata-only fingerprint migration was audited: the original training identity matched, and raw, filled, target and age arrays plus the geographic hierarchy were identical before and after the fingerprint update. Trained weights were unchanged by that metadata correction.

Current compatibility fingerprint:

`9157b8c74454d4a0180e0abacdc366cabc1bf9ab7d9cd61df751af12ef930d9b`
