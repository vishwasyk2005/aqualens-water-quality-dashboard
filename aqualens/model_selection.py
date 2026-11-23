"""Causal comparison of short-horizon forecasts on common observation pairs."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .backtest import (
    _observed_zone_counts, _source_values, _write_json, build_examples,
    make_test_rows, prepare_backtest_panel,
)
from .det_refinement import (
    causal_zone_anchors, load_refined_checkpoint, predict_normalized,
    training_target_scales,
)
from .forecast import HORIZONS, PARAMETERS, STEPS, _features, _load_checkpoint, dataset_fingerprint, forecast_zones, split_dates
from .prophet_model import PROPHET_SETTINGS, prophet_available, prophet_daily, prophet_native


VERSION = 1
MODELS = {
    'persistence': 'Last observation',
    'recent_median': 'Recent median',
    'climatology': 'Seasonal regression',
    'seasonal_kernel': 'Seasonal similarity',
    'prophet': 'Prophet',
    'ridge': 'Lagged ridge regression',
    'det': 'DET',
    'consensus': 'DET + seasonal + ridge',
}
MIN_ORIGIN = 11
SEASON_YEAR = 365.25
CONSENSUS = ('det', 'climatology', 'ridge')


def _day_number(dates):
    return (pd.DatetimeIndex(dates) - pd.Timestamp('1970-01-01')).days.to_numpy()


def _panel(df, cutoff):
    frame = df.assign(date=pd.to_datetime(df.date).dt.normalize(), zone_id=df.zone_id.astype(str))
    if 'lake_id' in frame and frame.lake_id.nunique() > 1:
        raise ValueError('Choose observations from one lake before forecasting.')
    zones = sorted(frame.loc[frame.date.le(cutoff), 'zone_id'].unique())
    if not zones:
        raise ValueError('No zones are available in the training period.')
    if len(zones) > 48:
        raise ValueError('Choose a study area with at most 48 zones for the model comparison.')
    frame = frame[frame.zone_id.isin(zones)]
    dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    raw = _source_values(frame).assign(date=frame.date, zone_id=frame.zone_id)
    grouped = raw.groupby(['date', 'zone_id'])[PARAMETERS].mean()
    shape = (len(dates), len(zones), len(PARAMETERS))
    targets = grouped.reindex(pd.MultiIndex.from_product([dates, zones])).to_numpy(float).reshape(shape)
    return {'dates': dates, 'zones': zones, 'targets': targets}, frame


def _season(days):
    phase = 2 * np.pi * np.asarray(days, dtype=float) / SEASON_YEAR
    return np.stack([np.ones_like(phase), np.sin(phase), np.cos(phase)], axis=-1)


def _ridge(design, target, strength, robust=False, penalty=None):
    penalty = np.eye(design.shape[1]) if penalty is None else penalty
    weights = np.ones(len(target))
    for _ in range(5 if robust else 1):
        weighted_design = design * weights[:, None]
        coeffs = np.linalg.solve(design.T @ weighted_design + strength * penalty, weighted_design.T @ target)
        if robust:
            residuals = np.abs(target - design @ coeffs)
            cap = max(float(np.median(residuals) * 2.5), 1e-6)
            weights = np.minimum(1., cap / np.maximum(residuals, 1e-12))
    return coeffs


def _fit_season(panel, cutoff):
    mask = panel['dates'] <= cutoff
    targets = panel['targets'][mask]
    design = _season(_day_number(panel['dates'][mask]))
    coefficients = np.empty((*targets.shape[1:], design.shape[-1]))
    if not np.isfinite(targets).any(axis=(0, 1)).all():
        raise ValueError('Each indicator needs recorded observations in the training period.')
    for indicator, fallback in enumerate(np.nanmedian(targets, axis=(0, 1))):
        for zone in range(targets.shape[1]):
            series = targets[:, zone, indicator]
            valid = np.isfinite(series)
            if valid.sum() < 8:
                coefficients[zone, indicator] = [np.median(series[valid]) if valid.any() else fallback, 0, 0]
                continue
            coefficients[zone, indicator] = _ridge(design[valid], series[valid], 5., robust=True,
                                                    penalty=np.diag([1e-8, 1., 1.]))
    return coefficients


def _history_features(panel, season):
    """Summarize the newest available readings; right edge is always the origin."""
    targets = panel['targets']
    days = _day_number(panel['dates'])
    shape = targets.shape[1:]
    zone_ids = np.eye(shape[0])
    prior = np.full(shape, np.nan)
    last_day = np.full(shape, np.nan)
    latest_values, recent_medians, rows = [], [], []
    for index, day in enumerate(days):
        now = targets[index]
        observed = np.isfinite(now)
        prior = np.where(observed, now, prior)
        last_day = np.where(observed, day, last_day)
        seasonal = season @ _season(day)
        current = np.where(np.isfinite(prior), prior, seasonal)
        previous = latest_values[max(0, index - 2)] if latest_values else current
        history = targets[max(0, index - 3):index + 1]
        rolling = current.copy()
        for zone, parameter in zip(*np.where(np.isfinite(history).any(axis=0))):
            readings = history[:, zone, parameter]
            rolling[zone, parameter] = np.median(readings[np.isfinite(readings)])
        age = np.where(np.isfinite(last_day), np.minimum(day - last_day, 96.), 96.) / 96.
        rows.append(np.concatenate((zone_ids, current, rolling, current - previous, age), axis=1))
        latest_values.append(current)
        recent_medians.append(rolling)
    return np.stack(latest_values), np.stack(recent_medians), np.stack(rows)


def _lagged_design(history, days, horizon):
    nominal_dates = _season(days + horizon)[..., 1:]
    seasonal = np.broadcast_to(nominal_dates[:, None, :], (*history.shape[:2], 2))
    return np.concatenate((history, seasonal), axis=2)


def _seasonal_similarity(days, targets, latest):
    """Average observed readings near the forecast's time of year."""
    bandwidth = 36.
    predictions = np.empty((len(days), len(HORIZONS), *targets.shape[1:]))
    observed = np.isfinite(targets)
    values = np.where(observed, targets, 0.)
    for origin, day in enumerate(days):
        elapsed = day + np.asarray(HORIZONS)[:, None] - days[None, :origin + 1]
        distance = (elapsed + SEASON_YEAR / 2) % SEASON_YEAR - SEASON_YEAR / 2
        weights = np.exp(-.5 * np.square(distance / bandwidth))
        mass = np.einsum('ht,tzp->hzp', weights, observed[:origin + 1])
        total = np.einsum('ht,tzp->hzp', weights, values[:origin + 1])
        fallback = np.broadcast_to(latest[origin], total.shape).copy()
        predictions[origin] = np.divide(total, mass, out=fallback, where=mass > 1e-7)
    return predictions


def _predict_candidates(panel, cutoff, candidates):
    dates, targets = panel['dates'], panel['targets']
    days = _day_number(dates)
    season = _fit_season(panel, cutoff)
    latest, medians, history = _history_features(panel, season)
    dimensions = (len(dates), len(STEPS), *latest.shape[1:])
    estimates = {name: np.zeros(dimensions) for name in candidates}
    if 'seasonal_kernel' in candidates:
        estimates['seasonal_kernel'] = _seasonal_similarity(days, targets, latest)
    for h_index, (step, horizon) in enumerate(zip(STEPS, HORIZONS)):
        if 'persistence' in candidates:
            estimates['persistence'][:, h_index] = latest
        if 'recent_median' in candidates:
            estimates['recent_median'][:, h_index] = medians
        seasonal = np.einsum('zpk,tk->tzp', season, _season(days + horizon))
        if 'climatology' in candidates:
            estimates['climatology'][:, h_index] = seasonal
        if 'ridge' not in candidates:
            continue
        estimates['ridge'][:, h_index] = seasonal
        design = _lagged_design(history, days, horizon)
        origin = np.arange(MIN_ORIGIN, len(dates) - step)
        origin = origin[dates[origin + step] <= cutoff]
        predictors = design[origin].reshape(-1, design.shape[-1])
        for indicator in range(targets.shape[-1]):
            outcome = targets[origin + step, :, indicator].ravel()
            valid = np.isfinite(outcome)
            if valid.sum() < 20:
                continue
            x_train, y_train = predictors[valid], outcome[valid]
            center, scale = x_train.mean(axis=0), x_train.std(axis=0)
            scale = np.where(scale > 1e-8, scale, 1.)
            target_center = np.median(y_train)
            coeffs = _ridge((x_train - center) / scale, y_train - target_center, 20., robust=True)
            estimates['ridge'][:, h_index, :, indicator] = (design - center) / scale @ coeffs + target_center
    return estimates, latest


def _date_cutoff(value):
    return pd.Timestamp(value).normalize()


def _partition_masks(panel, train_end, validation_end, test_end):
    dates, targets = panel['dates'], panel['targets']
    origins = np.arange(MIN_ORIGIN, len(dates) - 1)
    mask_shape = (len(origins), len(STEPS), *targets.shape[1:])
    validation, test = np.zeros(mask_shape, dtype=bool), np.zeros(mask_shape, dtype=bool)
    training_lock = dates[dates <= train_end][-1]
    selection_lock = dates[dates <= validation_end][-1]
    for h_index, step in enumerate(STEPS):
        candidates = np.flatnonzero(origins + step < len(dates))
        origin = origins[candidates]
        target = origin + step
        eligible = np.isfinite(targets[target])
        validation[candidates, h_index] = eligible & ((dates[origin] >= training_lock) & (dates[target] > train_end) & (dates[target] <= validation_end))[:, None, None]
        test[candidates, h_index] = eligible & ((dates[origin] >= selection_lock) & (dates[target] > validation_end) & (dates[target] <= test_end))[:, None, None]
    return origins, validation, test


def _det_predictions(df, panel, checkpoint_path, train_end, validation_end):
    import torch

    checkpoint = load_refined_checkpoint(checkpoint_path)
    if _date_cutoff(checkpoint['train_end']) != train_end or _date_cutoff(checkpoint['validation_end']) != validation_end:
        raise ValueError('Benchmark cutoffs do not match the frozen DET checkpoint.')
    past = df[pd.to_datetime(df.date).dt.normalize().le(validation_end)].copy()
    if dataset_fingerprint(past) != checkpoint.get('fit_data_fingerprint'):
        raise ValueError('The frozen DET checkpoint belongs to different training observations.')
    det_panel, _ = prepare_backtest_panel(df, train_end=train_end)
    if det_panel['cells'] != checkpoint['cells'] or det_panel['cell_zone'] != checkpoint['cell_zone']:
        raise ValueError('The frozen DET checkpoint has a different input hierarchy.')
    if det_panel['zones'] != panel['zones'] or not det_panel['dates'].equals(panel['dates']):
        raise ValueError('The model candidates do not share the frozen DET zone/date keys.')
    mean, scale = np.asarray(checkpoint['mean']), np.asarray(checkpoint['scale'])
    window = checkpoint['config']['window']
    features, _, origins, _ = build_examples(det_panel, mean, scale, window, train_end, validation_end,
                                             validation_end + pd.Timedelta(days=1))
    torch.set_num_threads(2)
    anchors = causal_zone_anchors(det_panel, mean, scale)[origins]
    predictions = np.full((len(panel['dates']), len(STEPS), *panel['targets'].shape[1:]), np.nan)
    predictions[origins] = predict_normalized(checkpoint, features, anchors) * scale + mean
    return predictions


def _metrics(frame):
    if frame.empty:
        return []
    zone_rows = frame[frame.zone_id.ne('all')].assign(zone_id='pooled')
    combined = pd.concat([frame, zone_rows], ignore_index=True)
    rows = []
    for (model, zone, parameter, horizon), group in combined.groupby(['model', 'zone_id', 'parameter', 'horizon_days'], sort=True):
        actual, predicted = group.actual.to_numpy(float), group.prediction.to_numpy(float)
        error = predicted - actual
        denominator = np.square(actual - actual.mean()).sum()
        rows.append(dict(model=model, zone_id=zone, parameter=parameter, horizon_days=int(horizon),
                         mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                         r2=float(1. - np.square(error).sum() / denominator) if denominator > 0 else None,
                         n=len(group), target_dates=int(group.target_date.nunique())))
    baseline = {(row['zone_id'], row['parameter'], row['horizon_days']): row['rmse'] for row in rows if row['model'] == 'persistence'}
    for row in rows:
        anchor = baseline[(row['zone_id'], row['parameter'], row['horizon_days'])]
        row['rmse_vs_persistence_pct'] = (1. - row['rmse'] / anchor) * 100. if anchor > 0 else None
    return rows


def _score(metrics, zone, scales):
    tasks = [item['rmse'] / scales[PARAMETERS.index(item['parameter'])] for item in metrics if item['zone_id'] == zone]
    return float(np.mean(tasks)) if len(tasks) == len(PARAMETERS) * len(HORIZONS) else None


def _lake_scales(panel, counts, cutoff, fallback):
    fit_dates = panel['dates'] <= cutoff
    observed = panel['targets'][fit_dates]
    weights = np.where(np.isfinite(observed), counts[fit_dates], 0.)
    total = weights.sum(axis=1)
    lake = np.divide(np.nansum(observed * weights, axis=1), total,
                     out=np.full_like(total, np.nan), where=total > 0)
    scale = np.array([np.std(series[np.isfinite(series)]) if np.isfinite(series).any() else 0. for series in lake.T])
    return np.where(scale > 1e-6, scale, fallback)


def _scope_scores(metrics, pooled_scales, lake_scales):
    pooled = _score(metrics, 'pooled', pooled_scales)
    lake = _score(metrics, 'all', lake_scales)
    return dict(pooled=pooled, all=lake, balanced=(pooled + lake) / 2 if pooled is not None and lake is not None else None)


def _rows(panel, origins, mask, counts, predictions):
    result = []
    for model, values in predictions.items():
        if not np.isfinite(values[origins][mask]).all():
            raise ValueError(f'{MODELS[model]} is missing predictions on common observed targets.')
        result.extend(dict(row, model=model) for row in make_test_rows(panel, values[origins], origins, mask, counts))
    return pd.DataFrame(result)


def benchmark_models(df, evaluation_checkpoint=None, *, train_end=None, validation_end=None, test_end=None,
                     selection_scope='pooled', require_prophet=False):
    """Fit before validation, lock the winner on validation, then score later pairs."""
    if selection_scope not in ('all', 'pooled'):
        raise ValueError('Model selection scope must be all or pooled.')
    if (train_end is None) != (validation_end is None):
        raise ValueError('Provide both benchmark cutoff dates, or neither.')
    if train_end is None:
        cuts = split_dates(df)
        train_end, validation_end = cuts['train_end'], cuts['validation_end']
    train_end, validation_end = _date_cutoff(train_end), _date_cutoff(validation_end)
    test_end = _date_cutoff(pd.to_datetime(df.date).max() if test_end is None else test_end)
    if not train_end < validation_end < test_end:
        raise ValueError('Model comparison needs ordered training, validation and later evaluation periods.')
    panel, mapped = _panel(df, train_end)
    dates = panel['dates']
    if (dates <= train_end).sum() < 22 or ((dates > train_end) & (dates <= validation_end)).sum() < 5 or ((dates > validation_end) & (dates <= test_end)).sum() < 5:
        raise ValueError('Model comparison needs 22 training dates and at least five dates in each later period. More dates produce a more useful comparison.')
    origins, validation, retrospective = _partition_masks(panel, train_end, validation_end, test_end)
    candidates = [name for name in MODELS if name not in ('det', 'consensus', 'prophet')]
    predictions, _ = _predict_candidates(panel, train_end, candidates)
    prophet_note = None
    if prophet_available() or require_prophet:
        try:
            predictions['prophet'] = prophet_native(panel, train_end, HORIZONS)
        except RuntimeError as error:
            if require_prophet:
                raise
            prophet_note = str(error)
    if evaluation_checkpoint is not None:
        predictions['det'] = _det_predictions(df, panel, evaluation_checkpoint, train_end, validation_end)
        predictions['consensus'] = np.mean([predictions[component] for component in CONSENSUS], axis=0)
    counts = _observed_zone_counts(mapped, panel)
    scales = training_target_scales(panel, train_end)
    lake_scales = _lake_scales(panel, counts, train_end, scales)
    validation_rows = _rows(panel, origins, validation, counts, predictions)
    validation_metrics = _metrics(validation_rows)
    validation_scope_scores = {name: _scope_scores([row for row in validation_metrics if row['model'] == name], scales, lake_scales) for name in predictions}
    pooled_scores = {name: scores['pooled'] for name, scores in validation_scope_scores.items()}
    if any(scores[selection_scope] is None or scores['pooled'] is None for scores in validation_scope_scores.values()):
        raise ValueError('Each indicator and horizon needs recorded validation targets before selecting a model.')
    labels = {'all': 'the whole-lake summary', 'pooled': 'pooled zone observations'}
    ranking = sorted(pooled_scores, key=lambda name: (validation_scope_scores[name][selection_scope], pooled_scores[name], name))
    winner = ranking[0]

    paired = _rows(panel, origins, retrospective, counts, predictions)
    metrics = _metrics(paired)
    retrospective_scope_scores = {name: _scope_scores([row for row in metrics if row['model'] == name], scales, lake_scales) for name in predictions}
    models = [dict(id=name, name=MODELS[name], validation_score=pooled_scores[name], retrospective_score=retrospective_scope_scores[name]['pooled'],
                   selection_score=validation_scope_scores[name][selection_scope],
                   validation_scope_scores=validation_scope_scores[name], retrospective_scope_scores=retrospective_scope_scores[name])
              for name in ranking]
    radii = {}
    for name in predictions:
        zone_selection = validation_rows[validation_rows.zone_id.ne('all') & validation_rows.model.eq(name)]
        radii[name] = [[float(np.quantile(zone_selection.loc[zone_selection.parameter.eq(parameter) & zone_selection.horizon_days.eq(horizon), 'absolute_error'], .8, method='higher'))
                       for parameter in PARAMETERS] for horizon in HORIZONS]
    limitations = [
        'Later periods may already have been reviewed during project development; fresh observations are needed for external validation.',
        'Horizon labels correspond to 1, 2 and 4 satellite reading steps, which may occur at irregular intervals.',
        'Whole-lake comparison lines share origin observation weights on the same eligible zones.',
        'The active forecast model is refitted using all available readings after selection. Its future performance and interval coverage are unknown.',
        'Observations beyond 32 days have not been validated. Prophet models calendar days directly; other models interpolate between the native horizons.',
    ]
    if 'consensus' in predictions:
        limitations.append('The equal-weight consensus was added after seeing the first retrospective comparison. Its current scores are development evidence, not an unseen test.')
    if 'seasonal_kernel' in predictions:
        limitations.append('Seasonal similarity was added during retrospective development and updates only with observations already recorded at each forecast origin.')
    if 'prophet' in predictions:
        limitations.append('Prophet was fitted to observed dates through the fitting cutoff. Its historical predictions use nominal calendar leads, and the same recorded targets as the other candidates.')
    if prophet_note:
        limitations.append(prophet_note)
    if evaluation_checkpoint is not None and selection_scope == 'all':
        limitations.append('Whole-lake was chosen as the product selection scope after the initial model comparison, because the dashboard opens to the whole lake. The later period is retrospective model development, not a fresh test of this choice.')
    return dict(
        version=VERSION, available=True, fingerprint=dataset_fingerprint(df), selected_model=winner,
        selected_model_name=MODELS[winner], models=models, metrics=metrics, validation_metrics=validation_metrics,
        pooled_validation_winner=min(pooled_scores, key=pooled_scores.get),
        train_end=str(train_end.date()), validation_end=str(validation_end.date()),
        test_start=str((validation_end + pd.Timedelta(days=1)).date()), test_end=str(test_end.date()),
        observation_dates=dict(training=int((dates <= train_end).sum()),
                               validation=int(((dates > train_end) & (dates <= validation_end)).sum()),
                               retrospective=int(((dates > validation_end) & (dates <= test_end)).sum())),
        intervals=radii, normalized_scales=dict(zip(PARAMETERS, scales.tolist())), lake_normalized_scales=dict(zip(PARAMETERS, lake_scales.tolist())),
        method=f'The default uses the earlier validation period and minimizes mean normalized RMSE across the three indicators and three native horizons for {labels[selection_scope]}. Pooled zone validation breaks ties. Lower is better. Later scores use the same forecast pairs for every candidate.',
        normalization='Pooled zone RMSE uses training-only within-zone temporal deviation; whole-lake RMSE uses training-only whole-lake temporal deviation. All nine tasks have equal weight within each scope.',
        comparison_baseline='persistence', selection_scope=selection_scope, limitations=limitations,
        model_settings={'prophet': PROPHET_SETTINGS.copy()} if 'prophet' in predictions else {},
    ), paired


def fit_forecast_policy(df, evaluation_checkpoint=None):
    """Compare only on this dataset, with its own chronological cutoffs."""
    return benchmark_models(df, evaluation_checkpoint)[0]


def _production_det_native(df, zone_panel, checkpoint_path):
    checkpoint = _load_checkpoint(df, checkpoint_path)
    if checkpoint.get('format_version') != 3:
        raise ValueError('Consensus requires the current version of the DET checkpoint.')
    cutoff = checkpoint.get('geometry_fit_through') or checkpoint.get('train_end')
    panel, _ = prepare_backtest_panel(df, train_end=cutoff)
    if panel['cells'] != checkpoint['cells'] or panel['cell_zone'] != checkpoint['cell_zone'] or panel['zones'] != zone_panel['zones']:
        raise ValueError('The DET and statistical candidates have different production hierarchies.')
    mean, scale = np.asarray(checkpoint['mean']), np.asarray(checkpoint['scale'])
    window = checkpoint['config']['window']
    features = _features(panel, mean, scale)
    if len(features) < window:
        raise ValueError('There are too few readings to use the DET part of the consensus.')
    anchors = causal_zone_anchors(panel, mean, scale)[-1:]
    return predict_normalized(checkpoint, features[None, -window:], anchors)[0] * scale + mean


def forecast_selected(df, days, report, det_checkpoint_path=None, model=None, *, trust_dataset=False):
    """Use a locked model (or an explicit benchmark candidate) after the newest reading."""
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 730:
        raise ValueError('Choose a forecast horizon between 1 and 730 days.')
    if not report.get('available') or report.get('version') != VERSION or (not trust_dataset and report.get('fingerprint') != dataset_fingerprint(df)):
        raise ValueError('A model comparison for these observations is not available.')
    if model is None:
        model = report['selected_model']
    if model not in {item['id'] for item in report['models']}:
        raise ValueError('This model was not evaluated on these observations.')
    if model == 'det':
        return forecast_zones(df, horizon_days=days, model='det', checkpoint_path=det_checkpoint_path)
    if model not in MODELS:
        raise ValueError('Unknown forecast model.')
    latest = _date_cutoff(pd.to_datetime(df.date).max())
    panel, _ = _panel(df, latest)
    if model == 'prophet':
        calendar = prophet_daily(panel, latest, days)
    else:
        components = ['climatology', 'ridge'] if model == 'consensus' else [model]
        candidates, anchors = _predict_candidates(panel, latest, components)
        if model == 'consensus':
            values = _production_det_native(df, panel, det_checkpoint_path)
            if not np.isfinite(values).all():
                raise ValueError('DET has missing estimates for consensus zones or parameters.')
            raw_predictions = np.mean([values, candidates['climatology'][-1], candidates['ridge'][-1]], axis=0)
        else:
            raw_predictions = candidates[model][-1]
    radii = np.asarray(report['intervals'][model], dtype=float)
    rows = []
    for horizon in range(1, days + 1):
        for zone_index, zone in enumerate(panel['zones']):
            for indicator, parameter in enumerate(PARAMETERS):
                if model == 'prophet':
                    prediction = calendar[horizon - 1, zone_index, indicator]
                    basis = 'Daily Prophet calendar forecast'
                    if horizon > 32:
                        basis += ' beyond the evaluated horizon'
                    if parameter == 'ndwi':
                        basis += '; NDWI clipped to [-1, 1]'
                else:
                    prediction = np.interp(horizon, [0, *HORIZONS], np.r_[anchors[-1, zone_index, indicator], raw_predictions[:, zone_index, indicator]])
                    basis = ('Native 8/16/32-day values; daily interpolation; future NDWI clipped to [-1, 1]'
                             if horizon <= 32 else 'Experimental scenario carrying forward the 32-day estimate')
                radius = np.interp(horizon, HORIZONS, radii[:, indicator]) if horizon <= 32 else np.nan
                low, high = prediction - radius, prediction + radius
                if parameter == 'ndwi':
                    prediction, low, high = np.clip([prediction, low, high], -1., 1.)
                rows.append(dict(date=latest+pd.Timedelta(days=horizon), zone_id=zone, parameter=parameter,
                                 prediction=float(prediction), lower=float(low), upper=float(high),
                                 model=model, horizon_days=horizon, is_extrapolation=horizon > 32,
                                 interval_basis='Empirical validation error width transferred to the refit; coverage unverified' if horizon <= 32 else 'Unavailable beyond 32 days',
                                 prediction_basis=basis))
    return pd.DataFrame(rows)


def write_benchmark(report, predictions, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / 'benchmark.json', report)
    predictions.to_csv(destination / 'predictions.csv', index=False)


def load_benchmark(path, fingerprint):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if not report.get('available') or report.get('version') != VERSION or report.get('fingerprint') != fingerprint:
        raise ValueError('Saved model comparison does not match these observations.')
    return report
