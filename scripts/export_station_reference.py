"""Prepare the small, fingerprinted default outlook payloads from the saved workspace."""
import json
from pathlib import Path


def main():
    web = Path(__file__).resolve().parents[1] / 'web'
    snapshot = json.loads((web / 'demo-state.json').read_text(encoding='utf-8'))
    benchmark = snapshot['model_benchmark']
    fingerprint = benchmark['fingerprint']
    selected = benchmark['selected_model']
    if snapshot['dataset'] != 'modis' or fingerprint != snapshot['static_bundle']['fingerprint']:
        raise ValueError('Saved observations do not match the bundled benchmark.')
    forecast_metadata = snapshot['forecast_metadata']['auto']
    backtest = snapshot['backtests']['auto']
    if (selected != forecast_metadata.get('selected_model')
        or selected != backtest['metadata'].get('selected_model')
        or not backtest['available']
        or fingerprint != backtest['metadata'].get('fingerprint')):
        raise ValueError('Saved outlook and historical results do not match the bundled selection.')

    keys = ('horizon_days', 'origin_date', 'target_date', 'actual_elapsed_days', 'zone_id',
            'parameter', 'model', 'actual', 'prediction', 'det_prediction', 'error')
    cases = {
        'forecast': {
            'rows': [row for row in snapshot['forecasts']['auto'] if 1 <= row['horizon_days'] <= 32],
            'metadata': {key: forecast_metadata[key] for key in ('selected_model_name', 'selected_model', 'uncertainty')},
            'maximum_days': 32,
        },
        'backtest': {
            'available': True, 'model': selected, 'model_name': forecast_metadata['selected_model_name'],
            'rows': [{key: row[key] for key in keys if key in row} for row in backtest['rows']],
            'metadata': {key: backtest['metadata'][key] for key in ('train_end', 'validation_end', 'test_start', 'test_end')},
        }
    }
    for kind, payload in cases.items():
        asset = {'schema': 1, 'dataset': 'modis', 'fingerprint': fingerprint, 'selected_model': selected,
                 'kind': kind, 'payload': payload}
        (web / 'station' / f'{kind}-reference.json').write_text(
            json.dumps(asset, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        print(f'Prepared {kind}: {len(payload["rows"]):,} existing rows.')


if __name__ == '__main__':
    main()
