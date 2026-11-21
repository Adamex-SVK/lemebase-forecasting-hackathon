"""Step 0.6: baselines on the local cuts (D14), scored with src/validate.py.

Baselines (as described on the brief's Data page, plus simple references):
  zero           forecast 0
  naive          the last training week repeated 13 times
  mean13/median13  mean / median of the last 13 training weeks
  seasonal_naive the value of the same week one year (52 weeks) earlier (needs >= 65 training weeks)
  autoarima      one AutoARIMA (no seasonality) per series; no sale in the last 8 weeks -> 0
  topdown_mstl   MSTL on the shop total (season 52 with >= 2 years of history, else 4),
                 split to series by each series' share of the last 13 training weeks;
                 no sale in the last 8 weeks -> 0

Forecasts are made at two levels (F8): 'series' (store x article) and 'article'
(summed over stores). The test shop has no stores, so only 'series'.

Run: .venv/bin/python src/baselines.py [shop ...]
Writes iterations/00-baselines/results.csv
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate as v  # noqa: E402

warnings.filterwarnings("ignore")
from statsforecast import StatsForecast  # noqa: E402
from statsforecast.models import MSTL, AutoARIMA  # noqa: E402

H = v.HORIZON
RESULTS = v.ROOT / "iterations" / "00-baselines" / "results.csv"
BRIEF = {  # published on the brief's Data / Brief page (A-D baselines)
    ("A", "autoarima"): 8.7, ("A", "topdown_mstl"): 10.2, ("B", "autoarima"): 20.4,
    ("B", "topdown_mstl"): 20.4, ("C", "autoarima"): 41.8, ("C", "topdown_mstl"): 37.7,
    ("D", "autoarima"): 57.2, ("D", "topdown_mstl"): 44.9,
}


def wide(train):
    return train.set_index(v.KEYS + ["week"])["units"].unstack("week").fillna(0.0).sort_index(axis=1)


def to_long(index, F, weeks):
    n = len(index)
    return pd.DataFrame({
        "store_id": np.repeat(index.get_level_values(0).to_numpy(), len(weeks)),
        "article_id": np.repeat(index.get_level_values(1).to_numpy(), len(weeks)),
        "week": np.tile(np.asarray(weeks), n),
        "forecast": np.asarray(F, dtype=float).reshape(-1),
    })


def complete(fc, hold):
    """One forecast row for every holdout row; series without a model get 0."""
    m = hold[v.KEYS + ["week"]].merge(fc, on=v.KEYS + ["week"], how="left")
    m["forecast"] = m["forecast"].fillna(0.0).clip(lower=0.0)
    return m


def autoarima(P, weeks):
    active = P.index[P.iloc[:, -8:].sum(axis=1).to_numpy() > 0]
    if len(active) == 0:
        return to_long(P.index[:0], np.zeros((0, H)), weeks)
    sub = P.loc[active]
    long = sub.stack().rename("y").reset_index()
    long["unique_id"] = long["store_id"].astype(str) + "|" + long["article_id"].astype(str)
    long = long.rename(columns={"week": "ds"})[["unique_id", "ds", "y"]]
    sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq="W-MON", n_jobs=-1)
    f = sf.forecast(df=long, h=H).reset_index()
    f[["store_id", "article_id"]] = f["unique_id"].str.split("|", n=1, expand=True)
    f["week"] = f["ds"]
    f["forecast"] = f["AutoARIMA"]
    return f[v.KEYS + ["week", "forecast"]]


def topdown(P, weeks):
    n = P.shape[1]
    sl = 52 if n >= 104 else 4
    total = pd.DataFrame({"unique_id": "total", "ds": P.columns, "y": P.sum(axis=0).to_numpy()})
    sf = StatsForecast(models=[MSTL(season_length=[sl])], freq="W-MON")
    tf = sf.forecast(df=total, h=H)["MSTL"].clip(lower=0).to_numpy()
    share = P.iloc[:, -13:].sum(axis=1).to_numpy() / max(P.iloc[:, -13:].to_numpy().sum(), 1e-9)
    F = share[:, None] * tf[None, :]
    F[P.iloc[:, -8:].sum(axis=1).to_numpy() == 0] = 0.0
    return to_long(P.index, F, weeks)


def baselines(P, weeks):
    A = P.to_numpy()
    n = A.shape[1]
    out = {"zero": lambda: to_long(P.index, np.zeros((len(P), H)), weeks),
           "naive": lambda: to_long(P.index, np.repeat(A[:, -1:], H, axis=1), weeks),
           "mean13": lambda: to_long(P.index, np.repeat(A[:, -13:].mean(axis=1, keepdims=True), H, axis=1), weeks),
           "median13": lambda: to_long(P.index, np.repeat(np.median(A[:, -13:], axis=1, keepdims=True), H, axis=1), weeks)}
    if n >= 65:
        out["seasonal_naive"] = lambda: to_long(P.index, A[:, n - 52:n - 52 + H], weeks)
    out["autoarima"] = lambda: autoarima(P, weeks)
    out["topdown_mstl"] = lambda: topdown(P, weeks)
    return out


def run(shops):
    rows = []
    for shop in shops:
        df = v.load_weekly(shop)
        train, hold, cut = v.split(df)
        weeks = pd.date_range(cut, periods=H, freq="7D")
        for level in (["series", "article"] if shop != "test" else ["series"]):
            tr = v.to_article_level(train) if level == "article" else train
            P = wide(tr[v.KEYS + ["week", "units"]])
            for name, make in baselines(P, weeks).items():
                t0 = time.time()
                fc = complete(make(), v.to_article_level(hold) if level == "article" else hold)
                sec = time.time() - t0
                for sm in ["full", "from_first_sale"]:
                    r = v.score(train, hold, fc, level=level, scale_mode=sm)
                    rows.append({"iteration": "00-baselines", "convention": "D26 leading-zero grid",
                                 "shop": shop, "forecast_level": level,
                                 "scale_mode": sm, "model": name, "training_data": "own", **r,
                                 "seconds": round(sec, 1)})
                last = rows[-2]
                print(f"{shop:4s} {level:7s} {name:15s} score(full)={last['score']:8.1f}  scored={last['n_scored']:6d}  {sec:6.1f}s", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    shops = sys.argv[1:] or ["A", "C", "test", "B", "D"]
    res = run(shops)
    pd.set_option("display.width", 250)
    print("\nBrief's published baselines vs ours (score, full scale; level in brackets):")
    for (s, m), b in BRIEF.items():
        if s in shops:
            sub = res[(res.shop == s) & (res.model == m) & (res.scale_mode == "full")]
            print(f"  {s} {m:13s} brief {b:5.1f} | ours " + " | ".join(f"{r.forecast_level}: {r.score:6.1f}" for r in sub.itertuples()))
