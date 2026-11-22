"""Score paired historical forecasts and reconstruct the target submission."""
import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from train import OUT, TARGET
import make_submission as submission
import validate as v

ROOT = v.ROOT
PR = "ca3a0ed111ecec8ef55ceb9c408a2988ee892149"
FROZEN = "71b82f33f0fe5647a13d3786cd17d11b4c4e5493"
BATCH = "5592b3593c794dd21b441b0fdd564eea90f78248"
FM = ["moirai-2.0-small", "tirex", "chronos-t5-small"]
KEY = v.KEYS + ["week"]
SOURCE = OUT / "source"
HASHES = {}


def blob(commit, path):
    data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
    HASHES[f"{commit}:{path}"] = hashlib.sha256(data).hexdigest()
    return io.BytesIO(data)


def matrix(src, grid):
    if isinstance(src, Path):
        HASHES[str(src.relative_to(ROOT))] = hashlib.sha256(src.read_bytes()).hexdigest()
    raw = pd.read_parquet(src).set_index(KEY).sort_index()
    assert raw.index.is_unique and raw.index.equals(grid)
    pred = raw.forecast.to_numpy(dtype=float).reshape(-1, 13)
    assert np.isfinite(pred).all()
    return pred


def combine(parts, tree, inactive):
    avg = np.mean(parts[:3], axis=0)
    if len(parts) == 4:
        avg = (3 * avg + parts[3]) / 4
    avg[inactive] = 0
    return np.maximum(0, np.rint(.7 * avg + .3 * tree))


def evaluate(shop, fold, arm, train, hold, parts, reference, candidate, published, groups):
    grid = hold.set_index(KEY).sort_index()
    hist = train.pivot(index=v.KEYS, columns="week", values="units").sort_index()
    assert hist.index.equals(grid.index.droplevel("week").unique()) and hist.notna().all().all()
    u = hist.to_numpy()
    dead = u[:, -8:].sum(axis=1) == 0
    actual = grid.units.to_numpy().reshape(-1, 13)
    rows = []
    for method, tree in [("reference", reference), ("distribution", candidate)]:
        forecast = combine(parts, tree, dead)
        long = grid.index.to_frame(index=False).assign(forecast=forecast.ravel())
        res = v.score(train, hold, long, detail=True)
        scale = res["detail"].scale.reindex(hist.index).to_numpy()
        ok = np.isfinite(scale)
        abs_scaled = np.abs(actual[ok] - forecast[ok]) / scale[ok, None]
        mean = abs_scaled.mean()
        np.testing.assert_allclose(mean, res["mase"], rtol=1e-10)
        if method == "reference" and published is not None:
            assert abs(res["score"] - published) <= .00501, (shop, fold, arm, res["score"], published)
        row = dict(shop=shop, offset=fold, cut=str(grid.index.get_level_values("week").min().date()),
                   arm=arm, components=len(parts)+1, model=method, mase=mean, score=res["score"],
                   early_mase=abs_scaled[:, :7].mean(), late_mase=abs_scaled[:, 7:].mean(),
                   scored=res["n_scored"], excluded=res["n_series"]-res["n_scored"],
                   all_zero_history=int((u.max(axis=1)==0).sum()),
                   constant_nonzero=int((~ok & (u.max(axis=1)>0)).sum()),
                   unit_coverage=res["holdout_units_share_scored"])
        rows.append(row)
        if shop == "test":
            recent = u[ok, -13:] > 0
            first = (u[ok] > 0).argmax(axis=1)
            stock = train[train.week == train.week.max()].set_index(v.KEYS).reindex(hist.index[ok])
            for group, mask in {"first_sale_last13": u.shape[1]-first <= 13,
                                "rare_1to3": (recent.sum(axis=1)>=1) & (recent.sum(axis=1)<=3),
                                "inactive13": ~recent.any(axis=1),
                                "last_stock_missing": stock.on_hand_missing.to_numpy(bool),
                                "last_stock_zero": (stock.on_hand.eq(0) & ~stock.on_hand_missing).to_numpy(),
                                "last_stock_positive": (stock.on_hand.gt(0) & ~stock.on_hand_missing).to_numpy()}.items():
                groups.append(dict(offset=fold, model=method, group=group, n=int(mask.sum()),
                                   mase=abs_scaled[mask].mean() if mask.any() else np.nan))
    return rows


def target():
    full = v.load_weekly("test")
    scores = pd.read_csv(ROOT / "iterations/25-presentation/results/hackathon_scores.csv").ensemble
    rows, groups = [], []
    saved = sorted(SOURCE.glob("chronos-2-dist31__test__w*__padded.parquet"))
    if saved:
        assert len(saved) == 3
        anchor = SOURCE / "chronos-2-ft__test__cut0__padded.parquet"
        _, hold0, _ = v.split(full)
        grid0 = hold0.set_index(KEY).sort_index().index
        np.testing.assert_array_equal(matrix(anchor, grid0), matrix(
            SOURCE / "chronos-2-dist31__test__w0__padded.parquet", grid0))
    for fold in TARGET[:-1]:
        frame = full[full.week <= full.week.max() - pd.Timedelta(weeks=fold)]
        train, hold, cut = v.split(frame)
        grid = hold.set_index(KEY).sort_index().index
        original = fold in [0, 13, 26, 39]
        commit, folder, suffix = ((FROZEN, "forecasts", f"cut{fold//13}") if original
                                  else (BATCH, "forecasts_cuts", f"w{fold}"))
        tree_path = (f"iterations/02-robust-eval/forecasts/base_cut{fold//13}_seed0.npy" if original
                     else f"iterations/05-meta-ensemble/base_forecasts/lgbm_w{fold}.npy")
        tree = np.load(blob(commit, tree_path), allow_pickle=False)
        assert tree.shape == (len(grid)//13,13) and np.isfinite(tree).all()
        parts = [matrix(blob(commit, f"iterations/03-foundation-model/{folder}/{m}__test__{suffix}__padded.parquet"), grid) for m in FM]
        if original:
            parts.append(matrix(SOURCE / f"chronos-2-ft__test__{suffix}__padded.parquet", grid))
        else:
            extra = SOURCE / f"chronos-2-dist31__test__{suffix}__padded.parquet"
            if extra.exists():
                parts.append(matrix(extra, grid))
        rows += evaluate("test", fold, "target_tree", train, hold, parts, tree,
                         matrix(OUT / f"test_{fold}.parquet", grid), scores[fold//13] if original else None, groups)
    paired = report(rows, "target").reset_index()
    primary = paired.offset.isin([0, 13, 26, 39])
    numbers = [c for c in paired if c.endswith(("_mase", "_points"))]
    paired.groupby(np.where(primary, "primary", "supplementary"))[numbers].mean().rename_axis(
        "periods").to_csv(OUT / "target_average.csv")
    pd.DataFrame(groups).to_csv(OUT / "target_groups.csv", index=False)


def report(rows, name):
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"{name}_metrics.csv", index=False)
    index = ["shop","arm","cut","offset","components"]
    wide = table.pivot(index=index, columns="model", values=["mase","early_mase","late_mase"])
    result = pd.DataFrame(index=wide.index)
    for metric in ["mase","early_mase","late_mase"]:
        for model in ["reference","distribution"]:
            result[f"{model}_{metric}"] = wide[metric,model]
        result[f"{metric}_gain_points"] = 100 * (wide[metric,"reference"] - wide[metric,"distribution"])
    result.to_csv(OUT / f"{name}_paired.csv")
    print(result.round(5).to_string(), flush=True)
    return result


def shops():
    base = ROOT / "iterations/28-pr20-four-shops/results"
    adapters = ROOT / "iterations/29-borrowing/results"
    refs = pd.read_csv(adapters / "comparison.csv").set_index("shop")
    rows = []
    for shop in "ABCD":
        train, hold, cut = v.split(v.load_weekly(shop))
        grid = hold.set_index(KEY).sort_index().index
        fixed = [matrix(base / f"{m}_{shop}.parquet", grid) for m in FM]
        tree = matrix(base / f"lgbm_{shop}.parquet", grid)
        for arm, donor in [("own", shop), ("pooled", "ABCD")]:
            adapter = matrix(adapters / donor / f"chronos-2-ft_{shop}.parquet", grid)
            pred = matrix(OUT / f"{shop}_{arm}.parquet", grid)
            rows += evaluate(shop, 0, arm, train, hold, fixed + [adapter], tree, pred, refs.loc[shop,arm], [])
    result = report(rows, "shops")
    result.groupby(level="arm").mean().to_csv(OUT / "equal_shop_average.csv")


def final():
    ids = submission.history_ids()
    hist = v.load_weekly("test")
    keys = hist[v.KEYS].drop_duplicates().sort_values(v.KEYS)
    assert keys.article_id.tolist() == ids
    weeks = pd.date_range(submission.FIRST_FORECAST_WEEK, periods=13, freq="7D")
    grid = pd.MultiIndex.from_frame(keys.merge(pd.DataFrame({"week":weeks}), how="cross"))
    parts = [matrix(blob(PR, f"iterations/03-foundation-model/forecasts_final/{m}__test__final__padded.parquet"), grid) for m in FM]
    parts.append(matrix(blob(PR, "iterations/03-foundation-model/forecasts_oos-ft/chronos-2-ft__test__final__padded.parquet"), grid))
    inactive = hist[hist.week >= weeks[0] - pd.Timedelta(weeks=8)].groupby("article_id").units.sum().reindex(ids).eq(0).to_numpy()
    tree = pd.read_csv(blob(PR, "submissions/lgbm_own_v1.csv")).set_index("article_id").loc[ids].to_numpy()
    old = pd.read_csv(blob(PR, "submissions/ensemble_d40_plus_chronos2ft_v1.csv")).set_index("article_id").loc[ids].to_numpy()
    np.testing.assert_array_equal(combine(parts, tree, inactive), old)
    pred = combine(parts, matrix(OUT / "test_final.parquet", grid), inactive)
    path = ROOT / "submissions/ensemble_pr20_weekly_distribution_v1.csv"
    if path.exists():
        np.testing.assert_array_equal(pd.read_csv(path).set_index("article_id").loc[ids].to_numpy(), pred)
    else:
        submission.write_submission(pred, ids, path)
    submission.validate_submission(path)
    print("Changed cells", (pred != old).sum(), "changed articles", (pred != old).any(axis=1).sum(), "SHA-256", hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["target","shops","final"])
    stage = p.parse_args().stage
    {"target":target, "shops":shops, "final":final}[stage]()
    info = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                lightgbm=lightgbm.__version__, seed=0, source_sha256=HASHES)
    (OUT / f"{stage}_provenance.json").write_text(json.dumps(info, indent=2))
