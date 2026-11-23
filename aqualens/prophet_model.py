"""Forecast observed zone series using the original AquaLens Prophet settings."""

from importlib.util import find_spec

import numpy as np
import pandas as pd


PROPHET_SETTINGS = {
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.1,
    "seasonality_prior_scale": 5.0,
    "yearly_seasonality": True,
    # AquaLens applies its own empirical validation intervals across models.
    "uncertainty_samples": 0,
}


def prophet_available():
    """Check for the optional Prophet package without loading its Stan runtime."""
    try:
        return find_spec("prophet") is not None
    except (ImportError, ValueError):
        return False


def _predict(panel, cutoff, requested):
    dates = pd.DatetimeIndex(panel["dates"])
    training = dates <= pd.Timestamp(cutoff)
    targets = np.asarray(panel["targets"], dtype=float)[training]
    training_dates = dates[training]
    finite = np.isfinite(targets)
    if not finite.any(axis=(0, 1)).all():
        raise ValueError("Prophet needs recorded training observations for every indicator.")

    try:
        from prophet import Prophet
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
        raise RuntimeError(
            "Prophet and its Stan backend could not be loaded. Install or repair the Prophet dependency."
        ) from error

    unique_dates = requested.unique().sort_values()
    future = pd.DataFrame({"ds": unique_dates})
    predictions = np.empty((len(unique_dates), *targets.shape[1:]))
    fallback = np.nanmedian(np.where(finite, targets, np.nan), axis=(0, 1))
    for zone, indicator in np.ndindex(targets.shape[1:]):
        valid = finite[:, zone, indicator]
        series = targets[valid, zone, indicator]
        if len(series) < 2 or np.all(series == series[0]):
            predictions[:, zone, indicator] = series[0] if len(series) else fallback[indicator]
            continue
        history = pd.DataFrame({"ds": training_dates[valid], "y": series})
        try:
            model = Prophet(**PROPHET_SETTINGS)
            model.fit(history)
            forecast = model.predict(future)
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
            raise RuntimeError(
                "Prophet could not run its forecasting backend. Check the data and the Prophet installation."
            ) from error
        values = forecast.set_index("ds").yhat.reindex(unique_dates).to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError("Prophet did not return finite predictions for all requested dates.")
        predictions[:, zone, indicator] = values
    return predictions[unique_dates.get_indexer(requested)]


def prophet_native(panel, cutoff, horizons):
    """Use a fixed training cutoff and predict each nominal origin-plus-lead date."""
    dates = pd.DatetimeIndex(panel["dates"])
    horizons = pd.to_timedelta(list(horizons), unit="D").to_numpy()
    targets = (dates.to_numpy()[:, None] + horizons[None, :]).ravel()
    predictions = _predict(panel, cutoff, pd.DatetimeIndex(targets))
    return predictions.reshape(len(dates), len(horizons), *predictions.shape[1:])


def prophet_daily(panel, cutoff, days):
    """Predict every calendar day after the newest observation in the panel."""
    first = pd.DatetimeIndex(panel["dates"])[-1] + pd.Timedelta(days=1)
    return _predict(panel, cutoff, pd.date_range(first, periods=days, freq="D"))
