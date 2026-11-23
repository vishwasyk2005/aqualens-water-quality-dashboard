# Forecast audit: R² above 0.20

**Result: the target was not met.** No candidate evaluated in this pass reached R² strictly greater than 0.20 for all nine whole-lake indicator and horizon combinations. The prototype retains DET as its production default. The Check Accuracy page now shows the complete scorecard for the selected model and location.

## The original comparison

The recorded data, forecast origins, eligibility rules, and scores were preserved. The standard benchmark has 51, 50 and 48 paired whole-lake dates at approximately 8, 16 and 32 days, respectively. These correspond to 1, 2 and 4 observed steps. R² summarizes predicted versus recorded values across a series; a single reading has no meaningful R². A negative score means the squared errors were larger than those of a constant prediction equal to the recorded period's mean.

| Indicator | Approx. 8 days | Approx. 16 days | Approx. 32 days |
|---|---:|---:|---:|
| Water index (NDWI) | −0.085 | −0.084 | 0.093 |
| Turbidity proxy | 0.112 | 0.114 | 0.085 |
| Chlorophyll proxy | **0.336** | **0.310** | **0.278** |

**DET goal count: 3 of 9.** The counts above 0.20 use the unrounded scores.

## Additional forecast methods tried

- **Spatial models:** compared regularized, robust and kernel regressions with zone readings, sample coverage, quantiles, gridded summaries, and historical and seasonal inputs. The strongest tested model policy for 8-day turbidity reached 0.164, below the goal. The locked spatial model policies also missed the target for NDWI.
- **Seasonal and past-pattern models:** fit and selected on earlier history and used previous observations already available at the time of each forecast. A selected variant produced chlorophyll R² of 0.383, 0.367 and 0.334, but none of its NDWI or turbidity horizons cleared 0.20.
- **Rolling corrections:** tested adapting forecasts using errors from earlier targets only after those readings arrived. The variants evaluated did not reach the target across all indicators.
- **Regional weather:** combined precipitation, temperature, wind and solar radiation with calendar and earlier lake measurements. The best validation choices for each task were fixed before scoring the later period. Weather improved all three chlorophyll scores, to 0.410, 0.375 and 0.513, but did not lift any NDWI or turbidity score above 0.20. This was an exploratory comparison with published [ERA5 reanalysis via Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api), using weather ending at least seven days before each forecast origin. Finalized archived ERA5 is a later revision of the [initial ERA5T data](https://www.ecmwf.int/en/about/media-centre/news/2026/era5t-reanalysis-data), so this experiment does not establish a deployable weather model with identical inputs.

Model settings were selected using pre-2022 dates. Rolling methods could then use a reading only after its date. The 2022–February 2023 scores still represent a retrospective development comparison: this time period had already been examined earlier in the project. Its results are not a guarantee on future measurements.

## Evaluation with the new Vembanad measurements

The combined CSV preserves all 46,755 original rows and fields. Surface temperature matched 1,711 rows across 18 original dates. Total suspended matter (TSM) matched 9,650 rows across 68 original dates. Empty cells mark dates and locations without a usable reading. The separate daily extraction from [ESA Lakes CCI v3](https://climate.esa.int/en/projects/lakes/data/) has usable measurements on 164 temperature dates and 674 TSM dates for the study area, from 2020 through 2023.

### Original satellite indicators

The additional models used the same whole-lake values, forecast origins, and target dates as the existing benchmark. Model selection used readings up to December 2021. None of these policies cleared the threshold across the original three indicators. R² scores use 51, 50, and 48 matched dates.

| Approach | Indicator | ~8 days | ~16 days | ~32 days |
|---|---|---:|---:|---:|
| Pooled tree models | NDWI | −0.139 | −0.131 | −0.126 |
| Pooled tree models | Turbidity proxy | 0.083 | 0.089 | −0.016 |
| Pooled tree models | Chlorophyll proxy | 0.337 | 0.393 | 0.379 |
| Seasonal estimates with updates from prior readings | Chlorophyll proxy | 0.400 | 0.376 | 0.371 |
| Regression and DET blends with ESA inputs | NDWI | −0.184 | −0.059 | 0.143 |
| Regression and DET blends with ESA inputs | Turbidity proxy | 0.047 | −2.385 | −0.042 |
| Regression and DET blends with ESA inputs | Chlorophyll proxy | 0.202 | 0.275 | 0.114 |

An ensemble that adjusted the model weights based on past forecast errors also missed the complete target. The ESA-input experiment used observations ending at least eight days before the forecast origin. These observations came from a finalized archive, so their historical availability to an operational system was not established. DET remains the dashboard default.

### Separate temperature and suspended-matter series

These scores use medians over eight-day intervals with at least one usable ESA reading. They describe those interval summaries. The three original MODIS targets above retain their original observation dates and aggregation. Model selection uses dates before 2022 and the later comparison extends through December 2023.

| Indicator | ~8 days | ~16 days | ~32 days | Observed pairs |
|---|---:|---:|---:|---|
| Surface temperature, initial seasonal regression | 0.195 | −0.089 | 0.186 | 33, 32, 31 |
| Surface temperature, seasonal model with a recent-temperature adjustment | **0.356** | **0.356** | **0.327** | 33, 32, 31 |
| Suspended matter, lagged weather and earlier satellite readings | 0.078 | 0.157 | −0.072 | 84, 83, 81 |

The selected temperature model averages past daily temperatures by time of year and adjusts for recent departures from that seasonal pattern. The model choice uses three chronological validation windows before 2022 and a common rule across horizons. Recent input periods end at least eight days before issue time. Each scored interval has an observed target. The weather-based suspended-matter trial also uses archived weather, whose historical availability to an operational system was not established. Two observed suspended-matter peaks in May and September 2022 account for about 79% of the variation measured by the R² denominator. The forecasts missed those peaks, and both remain in the comparison.

These temperature and suspended-matter experiments preceded the dedicated dashboard views described below.

## Lake median forecasts in the dashboard

The Forecast and Forecast Accuracy pages now include separate lake median results. Available records were grouped by target date or eight-day interval. These are different forecast targets with their own recorded actual values. The original mean-based comparisons remain available and use the same targets, origin dates, and pair counts as before.

| Target and model | ~8 days | ~16 days | ~32 days | Recorded pairs |
|---|---:|---:|---:|---|
| Chlorophyll proxy, Seasonal similarity on original mean | 0.400 | 0.376 | 0.371 | 51, 50, 48 |
| Turbidity proxy, seasonal model for median of observed points | 0.426 | 0.310 | 0.290 | 51, 50, 48 |
| Surface temperature, seasonal model for eight-day summaries | 0.356 | 0.356 | 0.327 | 33, 32, 31 |
| Suspended matter, seasonal and hydrologic model for pooled eight-day median | 0.2025 | 0.2067 | 0.2561 | 84, 83, 81 |

The pooled suspended-matter median gives each usable grid-cell reading the same weight across the eight-day interval. In the earlier median-of-daily-medians calculation, a sparsely observed day received the same weight as a day with many valid grid cells. Both suspended-matter benchmarks retain the same evaluation dates.

NDWI still falls below 0.20 in the whole-lake comparison. Additional tests using geometrically aligned pixels, water-only locations, regional discharge, modeled tides, and ESA optical covariates did not clear the threshold. The original turbidity mean still falls below 0.20. The scores above 0.20 for median turbidity apply to the median target.

## What would improve the next attempt

The 46,755 input rows span only 143 dates. Spatial samples on the same date provide lake coverage but do not provide more independent moments to learn how the lake changes. Validation also showed changes in the water index and large shifts in turbidity variability. The source's cloud and pixel quality and the proxy formulas are unverified. Collect more recent, quality-screened readings with consistent locations and matching water level, inflow or rainfall measurements. Keep a later span untouched until a candidate is frozen so it can serve as a genuinely new evaluation.
