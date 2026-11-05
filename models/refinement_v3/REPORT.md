# Past-only DET refinement selection

This run selects one DET using only 2020–2021 measurements. It does not evaluate 2022–2023 targets. The source is parsed with its fixed study-area zone identifiers, then truncated before construction of any training/search arrays. The input panel contains 69 training dates and 23 validation dates. Training targets and feature/target scales stop in June 2021; validation origins are no earlier than the last training observation, 26 June 2021.

Six configurations were declared in search_plan.json before fitting. They share the temporal GRU and Child-Sum Tree-LSTM hierarchy. Three use exact observed-zone residual anchors; three predict zone values directly. Zero output initialization, dropout, weight decay, and parameter-balanced objectives were predeclared. Selection uses the mean of nine validation RMSE values, divided by training-only pooled within-zone temporal standard deviations. This gives equal weight to each parameter/horizon task and does not optimize any 2022 score.

A concrete input bug was repaired before the final search: coordinate jitter near geographic grid boundaries created spurious model cells. Coordinates are rounded to six decimal places before training-grid bounds and assignments are frozen. The repaired panel has 28 model cells. The earlier unsnapped search is retained only as an audit in audit_unsnapped/ and is not the selected run.

Selected candidate: **5**, direct DET, history window **12**, hidden size **24**, dropout **0.1**, learning rate **0.001**, weight decay **0.01**, loss **mse**. Validation selects **epoch 54**. The model becomes available after validation selection on 27 December 2021.

The selected validation score is **1.735670**, compared with **1.752870** for the original frozen DET under the same origin-safe validation targets: approximately **0.98% lower**. This is a modest aggregate validation improvement, not a claim of reliable forecasting. NDWI and some task scores worsen while several chlorophyll/turbidity scores improve. The latest-observation diagnostic is **1.919328** and is not deployed as another forecasting model.

| Parameter | Approximate horizon | Original DET validation RMSE | Selected DET validation RMSE |
|---|---|---:|---:|
| ndwi | 8 days | 0.163065 | 0.166041 |
| turbidity_proxy | 8 days | 1.379549 | 1.381245 |
| chlorophyll_proxy | 8 days | 2.316794 | 2.077519 |
| ndwi | 16 days | 0.154462 | 0.170156 |
| turbidity_proxy | 16 days | 1.513999 | 1.408141 |
| chlorophyll_proxy | 16 days | 2.205005 | 2.087341 |
| ndwi | 32 days | 0.170842 | 0.184741 |
| turbidity_proxy | 32 days | 1.460386 | 1.465029 |
| chlorophyll_proxy | 32 days | 2.469214 | 2.189949 |

The original historical model has 37 unsnapped cells and is evaluated using its own frozen preprocessing audit. It is never forced into the repaired 28-cell hierarchy. Its checkpoint SHA, source fingerprint, raw task errors, corrected origin eligibility and pair counts are retained in frozen_original_reference.json. The reference's raw errors are normalized using the current training-zone target scales; the target definition is unchanged by coordinate rounding.

The checkpoint is a plain safe-loadable state dictionary and configuration, not a serialized model object. Public helpers are build_refined_model, load_refined_checkpoint, model_from_checkpoint, predict_normalized, causal_zone_anchors and make_examples in aqualens/det_refinement.py. Inputs have shape [batch, history, cells, 12], normalized anchors [batch, zones, 3], and output [batch, 3 native horizons, zones, 3 parameters]. Raw outputs multiply by checkpoint scale and add checkpoint mean. No prediction clipping or smoothing is applied by this module.

The 2022–2023 period has already been reviewed during earlier development. Any subsequent score of this locked model is a development-period evaluation, not a newly untouched external test. Root explicitly locked the selected checkpoint before scoring it on that period. Future accuracy still needs fresh observations and verified source algorithms/units.

Selected checkpoint SHA-256: `7ebaae4a227edb21dd4cd4259cf7bd48df7a918d1c62bf61012f3912363b6aae`

Past-only source fingerprint: `5171c73277d0f479c2ddabc3d84d44dc888563d8362f6df78bb195e95bfc8004`
