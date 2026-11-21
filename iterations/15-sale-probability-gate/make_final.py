"""Generate the full-history 70/30 sale-gated candidate CSV locally."""
import io
import numpy as np
import pandas as pd

from run import ROOT, features, review, sale_probabilities, validate
from make_submission import validate_submission


def main():
    matrix = features.ShopMatrix(validate.load_weekly('test'))
    ids = matrix.keys.article_id
    weeks = pd.date_range(pd.Timestamp(matrix.weeks[-1]) + pd.Timedelta(days=7), periods=13, freq='7D')
    assert weeks[0] == pd.Timestamp('2023-04-24') and weeks[-1] == pd.Timestamp('2023-07-17')
    columns = [f'w{i}' for i in range(1, 14)]
    hashes = {}
    foundation = []
    for model in review.MODELS:
        path = f'iterations/03-foundation-model/forecasts_final/{model}__test__final__padded.parquet'
        data = review.load_blob(review.SOURCE, path, hashes)
        foundation.append(review.matrix(pd.read_parquet(io.BytesIO(data)), matrix.keys, weeks))
    data = review.load_blob(review.SOURCE, 'submissions/lgbm_own_v1.csv', hashes)
    regression = pd.read_csv(io.BytesIO(data)).set_index('article_id', verify_integrity=True).loc[ids, columns].to_numpy()
    dead = matrix.U[:, -8:].sum(axis=1) == 0
    champion = pd.read_csv(ROOT/'submissions/ensemble_top3_blend_v1.csv').set_index('article_id').loc[ids, columns].to_numpy()
    np.testing.assert_array_equal(review.compose(foundation, regression, dead), champion)
    probability, rounds = sale_probabilities(matrix)
    result = review.compose(foundation, np.where(probability < .5, 0, regression), dead)
    assert np.isfinite(result).all() and (result >= 0).all() and (result <= champion).all()
    frame = pd.DataFrame(result.astype(np.int64), columns=columns)
    frame.insert(0, 'article_id', ids.to_numpy())
    target = ROOT/'submissions/ensemble_top3_lgbm30_sale_gate_v1.csv'
    frame.to_csv(target, index=False, mode='x')
    validate_submission(target)
    changed = result != champion
    print(f'{target}: {len(frame)} products, {rounds} classifier trees, {changed.sum()} changed cells in {changed.any(axis=1).sum()} products')


if __name__ == '__main__':
    main()
