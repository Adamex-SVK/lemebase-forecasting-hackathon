"""Train the UCI-augmented LightGBM and evaluate the frozen ensemble on six cuts."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import lightgbm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "iterations/15-sale-probability-gate"))
from run import diagnose, features, read_components, review, train_gbm, validate
from run_arms import merge, OTHER_STEP

CONTRACT = ROOT / "iterations/10-benchmark-diagnosis/validation.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def donor_windows():
    audit = json.loads((HERE / "results/source_audit.json").read_text())
    path = ROOT / "data/external/uci_online_retail_weekly.parquet"
    assert digest(path) == audit["weekly_sha256"]
    df = pd.read_parquet(path)
    df = df.loc[df.week >= "2011-01-03"].assign(
        store_id="UCI", on_hand=np.nan, on_hand_missing=True, out_of_stock_flag=False)
    matrix = features.ShopMatrix(df)
    assert (matrix.W, matrix.n) == (48, 3709)
    windows = features.build_windows(matrix, features.make_origins(matrix.W, step=OTHER_STEP))
    assert len(windows["y"]) == 251654 and (windows["w"] == 1).all()
    return windows, pd.Timestamp(matrix.weeks[-1])


def fit_pooled(matrix, donor, weeks, model_path):
    origins = features.make_origins(matrix.W)
    inner = origins[-7:]
    earlier = origins[origins <= inner[0] - 13]
    assert earlier[-1] + 13 <= inner[0] and origins[-1] + 13 <= matrix.W

    def pooled(selected):
        own = features.build_windows(matrix, selected)
        return merge([("target", own), ("UCI", donor)], weighting="unweighted")

    initial, best = train_gbm.fit(pooled(earlier), features.build_windows(matrix, inner))
    del initial
    rounds = int(best * 1.1) + 1
    train = pooled(origins)
    model, _ = train_gbm.fit(train, rounds=rounds)
    model.save_model(str(model_path))
    forecast = train_gbm.forecast_matrix(model, matrix, weeks)
    assert np.isfinite(forecast).all()
    return forecast, dict(best_iteration=best, rounds=rounds, pooled_rows=len(train["y"]),
                          donor_rows=len(donor["y"]), training_end=str(pd.Timestamp(matrix.weeks[-1]).date()))


def main(out):
    cfg = json.loads(CONTRACT.read_text())
    primary = cfg["primary_development_offsets"]
    extra = [i for i in cfg["purged_forward_check_offsets"] if i not in primary]
    locked = pd.read_csv(CONTRACT.parent / "results/scores.csv").set_index("offset")
    donor, donor_end = donor_windows()
    full = validate.load_weekly("test")
    assert donor_end < full.week.min()
    out.mkdir(parents=True, exist_ok=False)
    saved = out / "fits"
    saved.mkdir()
    rows, groups, hashes = [], [], {}
    for offset in primary + extra:
        prefix = full[full.week <= full.week.max() - pd.Timedelta(days=7*offset)]
        history, hold, cut = validate.split(prefix)
        matrix = features.ShopMatrix(prefix[prefix.week < cut])
        weeks = pd.date_range(cut, periods=13, freq="7D")
        parts, own = read_components(offset, matrix.keys, weeks, offset in primary, hashes)
        dead = matrix.U[:, -8:].sum(axis=1) == 0
        control = review.compose(parts, own, dead)
        def score(forecast):
            return validate.score(history, hold, train_gbm.to_long(matrix, forecast, weeks))
        base = score(control)
        np.testing.assert_allclose(base["score"], locked.loc[offset, "score"], rtol=0, atol=1e-8)
        print(f"Offset {offset}: fitting through {pd.Timestamp(matrix.weeks[-1]).date()}", flush=True)
        pooled, info = fit_pooled(matrix, donor, weeks, saved / f"model_w{offset}.txt")
        forecast = review.compose(parts, pooled, dead)
        result = score(forecast)
        check = train_gbm.to_long(matrix, forecast, weeks)
        check["control"] = control.reshape(-1)
        check["pooled_tree"] = pooled.reshape(-1)
        check.to_parquet(saved / f"paired_w{offset}.parquet", index=False)
        actual = review.matrix(hold.rename(columns={"units": "forecast"}), matrix.keys, weeks)
        scale = validate.history_scale(history).reindex(pd.MultiIndex.from_frame(matrix.keys)).to_numpy()
        ok = np.isfinite(scale) & (scale > 0)
        labels = diagnose.history_groups(matrix.U, matrix.OH, parts + [own], scale)
        row = dict(offset=offset, primary=offset in primary, holdout_start=str(cut.date()),
                   holdout_end=str(weeks[-1].date()), n_scored=result["n_scored"],
                   excluded=matrix.n-result["n_scored"], unit_coverage=result["holdout_units_share_scored"],
                   own_tree_score=score(own)["score"], pooled_tree_score=score(pooled)["score"], **info)
        for name, F, official in [("own", control, base), ("pooled", forecast, result)]:
            err = np.abs(actual[ok] - F[ok]) / scale[ok, None]
            np.testing.assert_allclose(official["mase"], err.mean(), rtol=0, atol=1e-10)
            row.update({f"{name}_score": official["score"], f"{name}_mase": official["mase"],
                        f"{name}_early": 100*(1-err[:, :7].mean()),
                        f"{name}_late": 100*(1-err[:, 7:].mean())})
            for dim in ("observed_activation_age", "activity13", "last_stock", "stock_missing13"):
                for label in np.unique(labels[dim][ok]):
                    mask = labels[dim][ok] == label
                    groups.append(dict(offset=offset, variant=name, dimension=dim, group=label,
                                       n_scored=mask.sum(), mase=err[mask].mean()))
        row["gain"] = row["pooled_score"] - row["own_score"]
        row["late_gain"] = row["pooled_late"] - row["own_late"]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "paired.csv", index=False)
        pd.DataFrame(groups).to_csv(out / "groups.csv", index=False)
        print(f"Offset {offset}: {row['gain']:+.4f} score points", flush=True)
    stats = pd.DataFrame(rows).set_index("offset")
    gain = stats.loc[primary, "gain"]
    late = stats.loc[primary, "late_gain"].mean()
    threshold = cfg["promotion_review"]
    passed = (gain.mean() >= threshold["minimum_primary_mean_gain_score_points"]
              and (gain > 0).sum() >= threshold["minimum_primary_periods_improved"]
              and late >= 0 and (stats.loc[extra, "gain"] >= 0).all())
    summary = dict(primary_mean_gain=gain.mean(), primary_wins=int((gain > 0).sum()),
                   primary_late_gain=late, review_pass=bool(passed),
                   python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   lightgbm=lightgbm.__version__, training_params=train_gbm.PARAMS, source_blobs=hashes,
                   source_commits=[review.SOURCE, diagnose.ADDITIONAL_SOURCE],
                   target_sha256=digest(validate.IM / "weekly_test.parquet"),
                   runner_sha256=digest(Path(__file__)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Primary mean gain {gain.mean():+.4f}, wins {(gain > 0).sum()}/4, pass {passed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/backtest")
    main(parser.parse_args().output)
