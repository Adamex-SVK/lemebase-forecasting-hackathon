"""Generate the tested 50/50 peer blend on the full target-shop history."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from analogues import peer_forecast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from features import ShopMatrix
from make_submission import validate_submission
from validate import load_weekly


def main():
    folder = ROOT / 'submissions'
    baseline = folder / 'ensemble_top3_blend_v1.csv'
    validate_submission(baseline)
    matrix = ShopMatrix(load_weekly('test'))
    frame = pd.read_csv(baseline).set_index('article_id').loc[matrix.keys.article_id]
    champion = frame.to_numpy(dtype=float)
    peers, matches = peer_forecast(matrix.U, champion)
    blend = np.rint((peers + champion) / 2)
    matched = {row['target'] for row in matches if row['used']}
    changed = blend != champion
    assert set(np.flatnonzero(changed.any(axis=1))) <= matched
    target = folder / 'ensemble_early_sales_peers_50_50_v1.csv'
    output = pd.DataFrame(blend.astype(int), columns=frame.columns, index=frame.index)
    output.to_csv(target, mode='x')
    validate_submission(target)
    np.testing.assert_array_equal(pd.read_csv(target).set_index('article_id').loc[frame.index], blend)
    print(f'{len(matches)} recent, {len(matched)} matched, {changed.any(axis=1).sum()} changed products, {changed.sum()} changed weekly forecasts')


if __name__ == '__main__':
    main()
