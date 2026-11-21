"""Steps 1.2-1.3: global LightGBM (L1 loss on units / MASE scale, D29), local cut evaluation (D14, D26).

Step 1.2 = the `own` arm on the test shop:  .venv/bin/python src/train_gbm.py test

Early stopping: the last 7 window origins of the training period are validation windows;
training windows use origins at least 13 weeks earlier, so no target week is shared.
The number of trees found is then reused (x1.1) for a final fit on all windows.
"""
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as f  # noqa: E402
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
H = f.H
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0, num_threads=8)
OUT = v.ROOT / "iterations" / "01-global-gbm"


def fit(win_train, win_val=None, rounds=None, params=PARAMS):
    dtr = lgb.Dataset(win_train["X"], win_train["y"], weight=win_train["w"], feature_name=f.FEATURES, free_raw_data=True)
    if win_val is not None:
        dva = lgb.Dataset(win_val["X"], win_val["y"], weight=win_val["w"], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return m, m.best_iteration
    return lgb.train(params, dtr, num_boost_round=rounds), rounds


def forecast_matrix(model, M, weeks):
    """Forecast units for the 13 weeks after the last week of M; series with scale 0 get 0."""
    X, scale, valid = f.inference_features(M)
    p = model.predict(X).reshape(M.n, H) * scale[:, None]
    p = np.clip(p, 0, None)
    p[~valid] = 0.0
    return p


def to_long(M, F, weeks):
    return pd.DataFrame({
        "store_id": np.repeat(M.keys["store_id"].to_numpy(), H), "article_id": np.repeat(M.keys["article_id"].to_numpy(), H),
        "week": np.tile(np.asarray(weeks), M.n), "forecast": F.reshape(-1)})


def complete(fc, hold):
    m = hold[v.KEYS + ["week"]].merge(fc, on=v.KEYS + ["week"], how="left")
    m["forecast"] = m["forecast"].fillna(0.0).clip(lower=0.0)
    return m


def own_arm(shop, sample_inactive=False, df=None):
    """`df` lets a caller pass a shorter history (used for rolling-origin cuts)."""
    t0 = time.time()
    df = v.load_weekly(shop) if df is None else df
    train, hold, cut = v.split(df)
    M = f.ShopMatrix(df[df["week"] < cut])
    weeks = pd.date_range(cut, periods=H, freq="7D")
    origins = f.make_origins(M.W)
    val_o = origins[-7:]
    tr_o = origins[origins <= val_o[0] - H]
    w_tr = f.build_windows(M, tr_o, sample_inactive=sample_inactive)
    w_va = f.build_windows(M, val_o, sample_inactive=sample_inactive)
    print(f"{shop}: train windows {w_tr['n_windows']:,} (origins {tr_o[0]}..{tr_o[-1]}), val windows {w_va['n_windows']:,}", flush=True)
    m1, best = fit(w_tr, w_va)
    print(f"early stopping: best iteration {best}, val L1 {m1.best_score['valid_0']['l1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    w_all = f.build_windows(M, origins, sample_inactive=sample_inactive)
    n_final = int(best * 1.1) + 1
    m2, _ = fit(w_all, None, rounds=n_final)
    print(f"final fit on {w_all['n_windows']:,} windows, {n_final} trees  ({time.time()-t0:.0f}s)", flush=True)
    F = forecast_matrix(m2, M, weeks)
    return {"df": df, "train": train, "hold": hold, "cut": cut, "M": M, "weeks": weeks, "F": F,
            "model": m2, "best_iter": best, "seconds": time.time() - t0}


def evaluate(shop, res):
    M, hold, train = res["M"], res["hold"], res["train"]
    rows = {}
    fc = complete(to_long(M, res["F"], res["weeks"]), hold)
    rows["plain"] = (fc, v.score(train, hold, fc, detail=True))
    F2 = res["F"].copy()
    F2[M.U[:, -8:].sum(axis=1) == 0] = 0.0          # D20: zero rule, no sale in the last 8 weeks -> 0
    fc2 = complete(to_long(M, F2, res["weeks"]), hold)
    rows["zero_rule"] = (fc2, v.score(train, hold, fc2, detail=True))
    return rows


if __name__ == "__main__":
    shop = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = own_arm(shop)
    ev = evaluate(shop, res)
    out = []
    for variant, (fc, r) in ev.items():
        d = {k: x for k, x in r.items() if k != "detail"}
        out.append({"iteration": "01-global-gbm", "step": "1.2", "convention": "D26", "shop": shop, "model": "lgbm_l1",
                    "training_data": "own", "variant": variant, **d, "best_iter": res["best_iter"], "seconds": round(res["seconds"])})
        print(f"{shop} lgbm_l1 own [{variant:9s}] score {r['score']:7.2f}  MASE {r['mase']:.3f}  scored {r['n_scored']}")
    pd.DataFrame(out).to_csv(OUT / f"results_{shop}_own.csv", index=False)
    # diagnostics
    M = res["M"]; det = ev["plain"][1]["detail"]
    key = pd.MultiIndex.from_frame(M.keys)
    last13 = pd.Series(M.U[:, -13:].sum(axis=1) > 0, index=key).reindex(det.index)
    last8 = pd.Series(M.U[:, -8:].sum(axis=1) > 0, index=key).reindex(det.index)
    print("\nMASE by group (plain model):")
    for name, mask in [("sold in last 13 wks", last13), ("no sale in last 13 wks", ~last13), ("no sale in last 8 wks", ~last8)]:
        print(f"  {name:24s} n={int(mask.sum()):5d}  mean MASE {det.loc[mask, 'mase'].mean():.3f}  (score contribution {100*(1-det.loc[mask,'mase'].mean()):.1f})")
    hold, fc = res["hold"], ev["plain"][0]
    mm = hold.merge(fc, on=v.KEYS + ["week"]).merge(det[["scale"]].reset_index(), on=v.KEYS)
    mm["h"] = ((mm["week"] - res["weeks"][0]).dt.days // 7) + 1
    mm["mase_h"] = (mm["units"] - mm["forecast"]).abs() / mm["scale"]
    ph = mm.groupby("h")["mase_h"].mean()
    print("\nMASE by horizon week (1..13):", " ".join(f"{x:.2f}" for x in ph))
    imp = pd.Series(res["model"].feature_importance("gain"), index=f.FEATURES).sort_values(ascending=False)
    print("\nTop 12 features by gain (% of total):")
    print((100 * imp / imp.sum()).head(12).round(1).to_string())
    pd.DataFrame({"mase_by_h": ph}).to_csv(OUT / f"mase_by_horizon_{shop}_own.csv")
    print("\nbaselines for reference (local cut, D26):")
    b = pd.read_csv(v.ROOT / "iterations/00-baselines/results.csv")
    b = b[(b["shop"] == shop) & (b["forecast_level"] == "series") & (b["scale_mode"] == "full")]
    print(b[["model", "score"]].round(1).to_string(index=False))
