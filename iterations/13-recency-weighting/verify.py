"""Verify locked inputs, results and one saved-model prediction round trip."""
import argparse
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import run as experiment
import recency


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    out = ap.parse_args().output
    meta = json.loads((out/'metadata.json').read_text())
    assert experiment.sha(experiment.v.IM/'weekly_test.parquet') == meta['input_sha256']['data']
    assert experiment.sha(experiment.ROOT/'submissions/ensemble_top3_blend_v1.csv') == meta['input_sha256']['submission']
    for p in [experiment.HERE/'run.py', experiment.HERE/'recency.py',
              experiment.ROOT/'src/features.py', experiment.ROOT/'src/train_gbm.py',
              experiment.ROOT/'src/validate.py']:
        assert experiment.sha(p) == meta['code_sha256'][p.name]
    scores = pd.read_csv(out/'scores.csv')
    assert len(scores) == 18
    assert not scores.duplicated(['offset','variant']).any()
    control = scores[scores.variant == 'control']
    assert (control.lgbm_max_abs_drift_vs_saved == 0).all()
    primary = scores[scores.primary].pivot(index='offset',columns='variant',values='score')
    assert np.isclose(primary.incumbent.mean(), -13.66594696689804)
    assert np.isclose((primary.recency-primary.incumbent).mean(), meta['primary_gain'])
    groups = pd.read_csv(out/'groups.csv')
    contributions = groups.groupby(['offset','variant','dimension']).agg(
        mase=('total_mase_contribution','sum'), n_scored=('n_scored','sum')).reset_index()
    checked = contributions.merge(scores[['offset','variant','mase','n_scored']],on=['offset','variant'],suffixes=('_group','_all'))
    np.testing.assert_allclose(checked.mase_group, checked.mase_all, atol=1e-10)
    np.testing.assert_array_equal(checked.n_scored_group.to_numpy(), checked.n_scored_all.to_numpy())
    # Serialization check uses the largest candidate fit, no training or tuning.
    full = experiment.v.load_weekly('test')
    _, _, cut = experiment.v.split(full)
    M = experiment.f.ShopMatrix(full[full.week < cut])
    X, scale, valid = experiment.f.inference_features(M)
    model = lgb.Booster(model_file=str(out/'runs/w0_recency/model.txt'))
    pred = np.maximum(model.predict(X).reshape(M.n,13)*scale[:,None], 0)
    pred[~valid] = 0
    saved = np.load(out/'runs/w0_recency/forecast.npy',allow_pickle=False)
    np.testing.assert_allclose(pred,saved,atol=1e-10,rtol=0)
    result = dict(locked_inputs_unchanged=True, code_hashes_match=True, all_six_controls_exact=True,
                  score_rows=18, group_partitions_valid=True, model_reload_max_abs_difference=float(np.abs(pred-saved).max()))
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
