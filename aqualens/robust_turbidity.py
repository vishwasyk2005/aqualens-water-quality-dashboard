"""Seasonal forecasts for the lake-wide median of observed turbidity proxies."""
from collections import defaultdict

import numpy as np
import pandas as pd

from .backtest import _source_values


MODEL = "seasonal_turbidity_median"
PARAMETER = "turbidity_proxy"
AGGREGATION = "observed_spatial_median"
SELECTION_END = pd.Timestamp("2021-12-31")
YEAR_DAYS = 365.25
MINIMUM_OBSERVATION_DATES = 5
# These recent-median weights won validation on the bundled Vembanad series
# through December 2021. The seasonal bandwidth was 30 days for each horizon.
HORIZONS = ((8, 1, .65), (16, 2, .35), (32, 4, 0.))


def _estimate(days, values, horizon, recent_weight):
    observed = np.isfinite(values)
    if observed.sum() < MINIMUM_OBSERVATION_DATES:
        return None

    available = values[observed]
    center = np.median(available)
    mad = max(float(np.median(np.abs(available - center))), 1e-6)
    clipped = np.clip(available, center - 3 * mad, center + 3 * mad)
    phase_distance = (days[observed] - (days[-1] + horizon) + YEAR_DAYS / 2) % YEAR_DAYS
    weights = np.exp(-.5 * np.square((phase_distance - YEAR_DAYS / 2) / 30))
    seasonal = np.average(clipped, weights=weights)

    recent = values[-2:]
    recent_median = np.median(recent[np.isfinite(recent)]) if np.isfinite(recent).any() else center
    return float((1 - recent_weight) * seasonal + recent_weight * recent_median)


def _score(rows):
    by_horizon = defaultdict(list)
    for row in rows:
        by_horizon[row["horizon_days"]].append(row)

    metrics = []
    for horizon, entries in sorted(by_horizon.items()):
        actual = np.array([row["actual"] for row in entries])
        errors = np.array([row["error"] for row in entries])
        sst = float(np.square(actual - actual.mean()).sum())
        metrics.append(dict(
            model=MODEL, parameter=PARAMETER, scope="all", horizon_days=horizon,
            rmse=float(np.sqrt(np.mean(np.square(errors)))), mae=float(np.mean(np.abs(errors))),
            r2=1. - float(np.square(errors).sum()) / sst if sst > 0 else None,
            n=len(entries),
        ))
    return metrics


def robust_turbidity_report(frame):
    """Return live forecasts and retrospective observed-median pairs.

    The input is a single lake normalized by `load_data`. Observation dates
    and valid source rows are preserved. Historical medians have their own
    estimand and should never be compared to the legacy mean as actual values.
    """
    observed = _source_values(frame)[PARAMETER]
    grouped = observed.groupby(frame["date"]).agg(["median", "count"]).sort_index()
    if grouped.empty:
        raise ValueError("No dates were found for the turbidity median.")
    dates = pd.DatetimeIndex(grouped.index)
    days = (dates - dates[0]).days.to_numpy()
    values = grouped["median"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy()
    past_observed = dates[(dates <= SELECTION_END) & np.isfinite(values)]
    records = [
        dict(date=date.date().isoformat(), value=float(value), observed_count=int(count))
        for date, value, count in zip(dates, values, counts)
        if np.isfinite(value)
    ]

    latest = dates[-1]
    forecasts, historical = [], []
    for horizon, step, recent_weight in HORIZONS:
        estimate = _estimate(days, values, horizon, recent_weight)
        if estimate is not None:
            date = latest + pd.Timedelta(days=horizon)
            forecasts.append(dict(model=MODEL, parameter=PARAMETER, zone_id="all",
                                  origin_date=latest.date().isoformat(), target_date=date.date().isoformat(),
                                  horizon_days=horizon, horizon_steps=step, prediction=estimate))

        if past_observed.empty:
            continue
        for origin in np.flatnonzero(dates >= past_observed[-1]):
            target = origin + step
            if target >= len(dates) or dates[target] <= SELECTION_END or not np.isfinite(values[target]):
                continue
            estimate = _estimate(days[:origin+1], values[:origin+1], horizon, recent_weight)
            if estimate is None:
                continue
            actual = float(values[target])
            error = estimate - actual
            historical.append(dict(
                model=MODEL, parameter=PARAMETER, zone_id="all",
                origin_date=dates[origin].date().isoformat(), target_date=dates[target].date().isoformat(),
                horizon_days=horizon, horizon_steps=step,
                actual_elapsed_days=int((dates[target] - dates[origin]).days),
                actual=actual, prediction=estimate, error=error, absolute_error=abs(error),
                origin_observed_count=int(counts[origin]), target_observed_count=int(counts[target]),
            ))

    return dict(
        available=bool(forecasts), model=MODEL, parameter=PARAMETER, scope="all",
        aggregation=AGGREGATION, selection_end=SELECTION_END.date().isoformat(),
        observations=records, forecasts=forecasts,
        historical=dict(rows=historical, metrics=_score(historical)),
    )
