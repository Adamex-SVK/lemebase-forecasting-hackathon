"""Reload each model and reproduce its saved target-shop forecasts and scores."""
import json

import lightgbm
import numpy as np
import pandas as pd

from backtest import HERE, features, read_components, review, train_gbm, validate


def main():
    out = HERE / "results/backtest"
    rows = pd.read_csv(out / "paired.csv").set_index("offset", verify_integrity=True)
    assert set(rows.index) == {0, 13, 26, 39, 16, 32}
    full = validate.load_weekly("test")
    for offset, row in rows.iterrows():
        prefix = full[full.week <= full.week.max() - pd.Timedelta(days=7*offset)]
        history, hold, cut = validate.split(prefix)
        matrix = features.ShopMatrix(prefix[prefix.week < cut])
        weeks = pd.date_range(cut, periods=13, freq="7D")
        parts, own = read_components(offset, matrix.keys, weeks, row.primary, {})
        model = lightgbm.Booster(model_file=str(out / "fits" / f"model_w{offset}.txt"))
        forecast = train_gbm.forecast_matrix(model, matrix, weeks)
        saved = pd.read_parquet(out / "fits" / f"paired_w{offset}.parquet")
        for col, expected in [("pooled_tree", forecast), ("forecast", forecast), ("control", own)]:
            values = saved[validate.KEYS + ["week", col]].rename(columns={col: "forecast"})
            if col != "pooled_tree":
                expected = review.compose(parts, expected, matrix.U[:, -8:].sum(axis=1) == 0)
            np.testing.assert_array_equal(review.matrix(values, matrix.keys, weeks), expected)
            score_key = {"pooled_tree": "pooled_tree_score", "forecast": "pooled_score", "control": "own_score"}[col]
            official = validate.score(history, hold, values)
            np.testing.assert_allclose(official["score"], row[score_key], rtol=0, atol=1e-10)
            assert official["n_scored"] == row.n_scored
    summary = json.loads((out / "summary.json").read_text())
    primary = rows.loc[[0, 13, 26, 39]]
    np.testing.assert_allclose(primary.pooled_score.mean()-primary.own_score.mean(),
                               summary["primary_mean_gain"], rtol=0, atol=1e-10)
    passed = (primary.gain.mean() >= .5 and (primary.gain > 0).sum() >= 3
              and primary.late_gain.mean() >= 0 and (rows.loc[[16, 32], "gain"] >= 0).all())
    assert bool(passed) == summary["review_pass"]
    print("Verified six reloaded models, paired ensembles, scores and review decision.")


if __name__ == "__main__":
    main()
