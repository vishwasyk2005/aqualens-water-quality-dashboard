# AquaLens interface guide

The new frontend starts with the lake's observation history, then separates water condition, the interactive map, forecasts, and accuracy into their own views. You can move between them with the top navigation. On a small screen, the navigation scrolls sideways.

## Find your way

| Page | Purpose |
|---|---|
| Monitor | Latest recorded condition, three indicators, potential uses, precautions and cited evidence |
| Map | Actual observation points, date and zone filters, and measurement inspection |
| Forecast | Date and horizon controls, estimates from the selected method, and downloads |
| Accuracy | Actual and predicted readings, the R² goal scorecard and exact paired values |
| Sources | Searchable references, applicability, and optional custom sources |
| Data | CSV upload and export, quality checks, observation explorer, and DET training |

The opening image illustrates a fictional freshwater landscape. Use **Explore the map** for the actual locations and **Read the assessment** to review the latest recorded finding. **Review change alerts** opens signals that departed from recent history.

## Read the outlook

1. Open **Forecast**, then pick a location, indicator, and forecast length.
2. Keep **Auto** for the model selected using earlier observations, or choose a named method to compare.
3. Use the date picker or timeline to inspect all three estimates for the same day. Save the selected location's outlook as CSV.
4. Open **Accuracy** to switch among the nine indicator and horizon combinations, compare observed and predicted values, and inspect each matched date or export its pairs.

The date controls change the selected display date. They do not change model weights. Experimental projections, interval limitations and negative R² values remain visible. See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) for the results and data limits.

The supplied MODIS dataset can open its saved Auto forecasts through day 32 and its frozen historical benchmark directly. AquaLens checks the dataset fingerprint and selected model before using these results. New uploads use their own data. For maintainers, `python scripts/export_station_reference.py` rebuilds these small bundles after the complete saved workspace is exported.

## Local workspace and saved dataset

Open the project with `start.bat` for uploads, training, and interactive API queries. When the local engine cannot connect, the saved MODIS workspace loads instead. It includes historical observations and saved forecasts for the supplied CSV. Forecasts and assessment summaries are calculated from the recorded period; they are not a live feed.

## Hero image

`web/station/lake-basin.png` was created with the built-in image generation tool. The brief was a wide, top-down aerial view of a fictional tropical freshwater lake, with curving wetland shores, moss and jade tones, and no text, people, or identifiable location. It is labelled as an illustration in the interface.

The local UI remains buildless and uses no external font or application-script service. The existing map tile providers and security policy are unchanged.
