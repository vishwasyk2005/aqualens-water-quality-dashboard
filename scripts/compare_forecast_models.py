"""Generate the AquaLens model comparison from the archived MODIS CSV."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from aqualens.data import load_data
from aqualens.model_selection import benchmark_models, write_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/modis_observations.csv')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/historical_validation/det_historical.pt')
    parser.add_argument('--reference', type=Path, default=ROOT / 'models/historical_validation/heldout_predictions.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/model_selection')
    args = parser.parse_args()
    observations, _ = load_data(args.input)
    comparison, predictions = benchmark_models(observations, args.checkpoint, train_end='2021-06-30',
                                               validation_end='2021-12-31', selection_scope='all',
                                               require_prophet=True)
    reference = pd.read_csv(args.reference)
    keys = ['origin_date', 'target_date', 'horizon_days', 'zone_id', 'parameter']
    reference = reference.set_index(keys).sort_index()
    for name, frame in predictions.groupby('model'):
        pairs = frame.set_index(keys).sort_index()
        if not pairs.index.equals(reference.index) or not np.allclose(pairs.actual, reference.actual, rtol=1e-7, atol=1e-8):
            raise ValueError(f'{name} does not use the same date, zone, horizon and actual records as the frozen DET reference.')
        if name == 'det' and not np.allclose(pairs.prediction, reference.prediction, rtol=1e-6, atol=1e-7):
            raise ValueError('Recomputed DET predictions differ from the archived frozen DET predictions.')
    comparison['archived_det_pair_count'] = len(reference)
    write_benchmark(comparison, predictions, args.output)
    print(f"Selected on whole-lake validation: {comparison['selected_model_name']}")
    for model in comparison['models']:
        lake = model['retrospective_scope_scores']['all']
        print(f"{model['name']}: whole-lake validation={model['selection_score']:.5f}; whole-lake retrospective={lake:.5f}; pooled retrospective={model['retrospective_score']:.5f}")
    print(f'{len(reference)} identical archived pairs per candidate; {len(predictions)} total predictions')


if __name__ == '__main__':
    main()
