"""Export one iteration 22 challenger using all 143 pre-forecast weeks."""
import argparse
import json

import numpy as np
import pandas as pd

from finalize import base_forecasts, digest
from lightgbm_trial import f, train
from score_trial import HERE, ROOT, component, review, v
from make_submission import validate_submission, write_submission

FILES = {
    "daily": "ensemble_top3_daily_lgbm30_v1.csv",
    "own": "ensemble_top3_chronos_own_finetuned_v1.csv",
    "pooled": "ensemble_top3_chronos_pooled_finetuned_v1.csv",
}


def export(arm):
    history = v.load_weekly("test")
    matrix = f.ShopMatrix(history)
    keys = matrix.keys
    ids = keys.article_id.to_numpy()
    assert matrix.W == 143 and history.week.max() == pd.Timestamp("2023-04-17")
    weeks = pd.date_range("2023-04-24", periods=13, freq="7D")
    parts, tree, hashes = base_forecasts(keys, weeks)
    dead = matrix.U[:, -8:].sum(axis=1) == 0
    control = review.compose(parts, tree, dead)
    incumbent = pd.read_csv(ROOT/"submissions/ensemble_top3_blend_v1.csv").set_index("article_id")
    np.testing.assert_array_equal(control, incumbent.loc[ids].to_numpy())
    folder = HERE/"results/challenger_exports"
    folder.mkdir(exist_ok=True)
    if arm == "daily":
        with np.load(HERE/"results/daily_features.npz") as daily:
            np.testing.assert_array_equal(daily["ids"], ids)
            model_path = folder/"daily_model.txt"
            tree, rounds = train(matrix, daily["features"], model_path)
        training = {"rounds": rounds, "model_sha256": digest(model_path)}
    else:
        assert review.MODELS[-1] == "chronos-t5-small"
        parts[-1], training = component(HERE/f"results/gpu/test-final-{arm}", keys, history.week.max())
        assert training["arm"] == arm and training["shop"] == "test"
    forecast = review.compose(parts, tree, dead)
    path = ROOT/"submissions"/FILES[arm]
    if not path.exists():
        write_submission(forecast, ids, path)
    assert validate_submission(path)
    actual = pd.read_csv(path).set_index("article_id", verify_integrity=True).loc[ids].to_numpy()
    np.testing.assert_array_equal(actual, forecast)
    changed = int(np.count_nonzero(actual != control))
    assert changed > 0
    metadata = dict(file=str(path.relative_to(ROOT)), sha256=digest(path), changed_cells=changed,
                    training_end=str(history.week.max().date()), training=training, base_hashes=hashes)
    (folder/f"{arm}.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"{path.name}: verified {forecast.shape}, {changed} changed cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=FILES)
    export(parser.parse_args().arm)
