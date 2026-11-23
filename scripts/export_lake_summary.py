"""Package the measured Vembanad lake-summary forecasts for the dashboard."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aqualens.alerts import ALERT_POLICY
from aqualens.data import load_data
from aqualens.forecast import dataset_fingerprint
from aqualens.monitoring import FORECAST_SKILL_GOAL, _score
from aqualens.robust_turbidity import robust_turbidity_report


HORIZONS = (8, 16, 32)
TIME_COLUMNS = ["origin_date", "target_date", "horizon_days", "actual", "prediction"]


def temperature_forecasts(daily, periods):
    periods = periods["value"]
    origin = periods.dropna().index[-1]
    available = origin - pd.Timedelta(days=8)
    history = daily.loc[:available].dropna()
    seasonal_center = history.mean()
    history_day = history.index.day_of_year.to_numpy(dtype=float)

    def seasonal(date):
        distance = (history_day - date.day_of_year + 182.625) % 365.25 - 182.625
        weights = np.exp(-.5 * np.square(distance / 20))
        return (weights @ history.to_numpy() + 3 * seasonal_center) / (weights.sum() + 3)

    recent = periods.loc[:available].tail(2).dropna()
    anomalies = [value - seasonal(date) for date, value in recent.items()]
    correction = .15 * float(np.median(anomalies)) if anomalies else 0.
    forecasts = []
    for days in HORIZONS:
        target = origin + pd.Timedelta(days=days)
        forecasts.append(dict(origin_date=origin.date().isoformat(), target_date=target.date().isoformat(),
                              horizon_days=days, prediction=float(seasonal(target) + correction)))
    return forecasts


def result(label, basis, unit, pairs, outlook, observations, sample_unit):
    paired = pd.DataFrame(pairs)[TIME_COLUMNS].sort_values(["horizon_days", "target_date"])
    scores = []
    for days, rows in paired.groupby("horizon_days"):
        actual = rows.actual.to_numpy()
        errors = rows.prediction.to_numpy() - actual
        denominator = np.square(actual - actual.mean()).sum()
        if denominator <= 0 or not np.isfinite(errors).all():
            raise ValueError(f"No usable historical score for {label} at {days} days.")
        scores.append(dict(horizon_days=int(days), r2=float(1 - np.square(errors).sum() / denominator),
                           n=int(len(rows))))
    if tuple(point["horizon_days"] for point in scores) != HORIZONS:
        raise ValueError(f"Missing forecast horizon for {label}.")
    observations = observations.dropna(subset=["value"]).sort_index()
    if observations.empty:
        raise ValueError(f"No observed medians are available for {label}.")
    origin = observations.index[-1]
    if any(pd.Timestamp(row["origin_date"]) != origin for row in outlook):
        raise ValueError(f"{label} projections do not match the latest observed median.")
    observed = observations.iloc[-1]
    sample_date = observed.get("last_sample_date", origin)
    latest = dict(date=origin.date().isoformat(), last_sample_date=sample_date.date().isoformat(),
                  value=float(observed.value), sample_count=int(observed.sample_count), sample_unit=sample_unit)
    weeks = ALERT_POLICY["history_weeks"]
    prior = observations.loc[(observations.index >= origin - pd.Timedelta(weeks=weeks)) &
                             (observations.index < origin), "value"]
    center = scale = low = high = None
    if len(prior) >= ALERT_POLICY["minimum_periods"]:
        center = float(prior.median())
        scale_floor = .1 if unit == "°C" else max(abs(center) * .05, .01)
        scale = max(1.4826 * float(np.median(abs(prior - center))), scale_floor)
        width = ALERT_POLICY["warning_score"] * scale
        low, high = center - width, center + width
    observed_status, observed_direction = _score(latest["value"], center, scale)
    latest.update(status=observed_status, direction=observed_direction)
    reference = dict(baseline=center, range_low=low, range_high=high, history_periods=len(prior),
                     window_weeks=weeks, first_date=prior.index[0].date().isoformat() if len(prior) else None,
                     last_date=prior.index[-1].date().isoformat() if len(prior) else None)
    by_days = {row["horizon_days"]: row for row in scores}
    forecasts = []
    for row in outlook:
        score = by_days[row["horizon_days"]]
        prediction = row["prediction"]
        status, direction = _score(prediction, center, scale)
        if score["n"] < ALERT_POLICY["minimum_periods"] or score["r2"] <= FORECAST_SKILL_GOAL:
            status, direction = "Insufficient data", None
            reason = "Historical accuracy is too limited for a condition label at this lead."
        elif status == "Insufficient data":
            count = "One observed median" if len(prior) == 1 else f"{len(prior)} observed medians"
            reason = f"{count} in the preceding {weeks} weeks. At least {ALERT_POLICY['minimum_periods']} are needed."
        elif status == "Stable":
            reason = "The projected median stays within the recent reference range."
        else:
            reason = f"The projected median is {direction} than the recent reference range."
        forecasts.append(dict(row, status=status, direction=direction, reason=reason,
                              r2=score["r2"], pairs=score["n"], baseline=center, range_low=low, range_high=high,
                              change=prediction - latest["value"]))
    return dict(label=label, basis=basis, unit=unit, metrics=scores,
                pairs=paired.to_dict("records"), forecasts=forecasts,
                latest_observation=latest, reference=reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="Directory holding the ESA source and archived prediction pairs.")
    args = parser.parse_args()
    files = args.analysis_dir
    frame, _ = load_data(ROOT / "data/modis_observations.csv")
    turbidity = robust_turbidity_report(frame)
    esa = pd.read_csv(files / "vembanad_esa_daily_observations.csv", parse_dates=["date"])
    turbidity_observations = pd.DataFrame(turbidity["observations"]).rename(columns={"observed_count": "sample_count"})
    turbidity_observations["date"] = pd.to_datetime(turbidity_observations["date"])
    turbidity_observations = turbidity_observations.set_index("date")
    interval = dict(freq="8D", label="right", closed="right", origin="start")
    daily_temp = esa.groupby("date").lake_surface_temp_c.median()
    temperature_observations = daily_temp.groupby(pd.Grouper(**interval)).agg(value="median", sample_count="count").loc[:daily_temp.index.max()]
    temperature_observations["last_sample_date"] = daily_temp.index.to_series().where(daily_temp.notna()).groupby(
        pd.Grouper(**interval)
    ).max()
    suspended_dates = esa["date"].where(esa.tsm_mg_l.notna())
    suspended_observations = esa.assign(last_sample_date=suspended_dates).groupby(pd.Grouper(key="date", **interval)).agg(
        value=("tsm_mg_l", "median"), sample_count=("tsm_mg_l", "count"), last_sample_date=("last_sample_date", "max")
    ).loc[:esa.date.max()]
    temperature = pd.read_csv(files / "forecast_exog_now_esa_temp_predictions.csv")
    temperature = temperature.loc[temperature.model.eq("shared-rolling") & temperature["count"].eq(1)]
    temperature = temperature.drop(columns="prediction").rename(
        columns={"target_period_end": "target_date", "new_prediction": "prediction"}
    )
    suspended = pd.read_csv(files / "forecast_tsm_pooled_selected_pairs.csv").rename(
        columns={"target_period_end": "target_date"}
    )
    suspended_outlook = pd.read_csv(files / "forecast_tsm_pooled_future.csv").rename(
        columns={"target_period_end": "target_date", "prediction_mg_l": "prediction"}
    )
    forecast_columns = ["origin_date", "target_date", "horizon_days", "prediction"]
    series = [
        result("Turbidity proxy", "Median of observed points on each satellite date", "proxy units",
               turbidity["historical"]["rows"], pd.DataFrame(turbidity["forecasts"])[forecast_columns].to_dict("records"),
               turbidity_observations, "observed points"),
        result("Lake surface temperature", "Eight-day median of daily ESA lake summaries", "°C",
               temperature, temperature_forecasts(daily_temp, temperature_observations),
               temperature_observations, "usable observation days"),
        result("Suspended matter", "Eight-day median of all usable ESA grid-cell readings", "mg/L",
               suspended, suspended_outlook[forecast_columns].to_dict("records"),
               suspended_observations, "usable grid-cell readings"),
    ]
    bundle = dict(fingerprint=dataset_fingerprint(frame), series=series)
    path = ROOT / "web/station/lake-summary-results.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Saved {sum(len(item['pairs']) for item in series)} forecast pairs and nine projections.")


if __name__ == "__main__":
    main()
