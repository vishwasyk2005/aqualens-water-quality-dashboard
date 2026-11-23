# Choosing the AquaLens forecast model

**AquaLens Auto selects DET for the supplied MODIS CSV.** DET had the lowest whole-lake error in the July–December 2021 selection window. Seasonal similarity had lower overall error in the later retrospective comparison and is available as a dashboard option.

## What was measured

The CSV contains 46,755 rows across 143 observation dates, with 69 for fitting, 23 for selection, and 51 in the later period. Each of eight candidates used the same 3,129 recorded historical targets, with exact matching on observation date, forecast origin, location, indicator and horizon. A horizon of approximately 8, 16 or 32 days means 1, 2 or 4 observation steps in the benchmark. Each forecast uses only readings available at its origin, following [rolling forecast origin evaluation](https://otexts.com/fpptr/tscv.html).

Scores average normalized RMSE across all three indicators and horizons, using training-period scale factors. Lower scores are better. Whole-lake and pooled zone scores use different scales and should be compared only within their own columns.

| Model | 2021 whole lake | Later whole lake | 2021 pooled zones | Later pooled zones |
|---|---:|---:|---:|---:|
| DET | **1.9910** | 1.1734 | 1.7357 | 1.3216 |
| Lagged ridge regression | 1.9941 | 1.2918 | **1.7266** | 1.3884 |
| DET + seasonal + ridge | 2.0267 | 1.1854 | 1.7283 | 1.3158 |
| Seasonal similarity | 2.0528 | **1.1554** | 1.7391 | **1.2941** |
| Recent median | 2.0698 | 1.4146 | 1.7456 | 1.4889 |
| Seasonal regression | 2.1618 | 1.1767 | 1.7854 | 1.3091 |
| Last observation | 2.2413 | 1.6511 | 1.9193 | 2.0043 |
| Prophet | 2.4436 | 1.9541 | 2.2161 | 2.2179 |

The dashboard uses whole-lake selection for the bundled lake because that is the primary view. This scope choice and the addition of a fixed, equally weighted combination were made during project development after reviewing an earlier retrospective comparison. The 2022–February 2023 scores are development evidence. They cannot stand in for future, independent observations.

Seasonal similarity is now selectable in Forecast and Forecast Accuracy. It weights previous observations by time-of-year similarity and achieved chlorophyll proxy R² of 0.400, 0.376, and 0.371. DET has lower overall error in the pre-2022 selection window and remains the Auto choice.

Prophet uses the original prototype's multiplicative seasonality, yearly seasonality, changepoint prior of 0.1 and seasonality prior of 5.0. Historical fits use observed values through the common training cutoff. The fit has about 18 months of history, which limits how well it can estimate a yearly cycle. On the later whole-lake records, Prophet's chlorophyll proxy R² was about 0.32, 0.31 and 0.30 at the three horizons. NDWI and turbidity proxy R² were negative at all three horizons.

## Where the trade-off matters

See [Forecast audit: R² above 0.20](R2_TARGET_REPORT.md) for the nine-score target and the additional spatial, temporal and weather experiments.

DET's later whole-lake error score was about **29% lower than repeating the last observation**. The three-model combination performed better for whole-lake NDWI at shorter horizons: 8-day RMSE fell from **0.0745 to 0.0671** (about 10%) and R² increased from **−0.085 to +0.121**. At 16 days R² moved from **−0.084 to +0.012**. These are still weak scores. The same combination increased chlorophyll proxy error at all three horizons. Check Accuracy lets you see the exact numbers for each location and indicator.

The latest bundled observation is **10 February 2023**. Future forecasts in this workspace are relative to that date. Zone bands, where shown, illustrate earlier validation error widths. Coverage on the production refit is unknown, and no whole-lake band is calibrated. Prophet forecasts each calendar date directly. The other candidates interpolate between evaluated horizons. Projections beyond 32 days are experimental. Unverified satellite quality, sparse dates and unknown proxy formulas limit what any of these models can learn.

New uploads use their own chronological split and compare the lightweight candidates when enough usable history exists. DET can be trained separately on the upload. The bundled lake's scores do not carry over to another dataset.

## Reproduce

From the dashboard directory with the training requirements installed:

```powershell
python scripts/compare_forecast_models.py
python scripts/export_dashboard_demo.py
```

The comparison saves the chosen model, scope scores, provenance and limitations in `models/model_selection/benchmark.json` and the exact candidate forecasts in `models/model_selection/predictions.csv`.
