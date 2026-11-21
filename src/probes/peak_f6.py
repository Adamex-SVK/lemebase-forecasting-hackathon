"""F6 investigation: why does the test shop's yearly peak move?  Run: .venv/bin/python src/probes/peak_f6.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import validate as v  # noqa: E402


def totals(shop):
    df = v.load_weekly(shop)
    P = df.pivot_table(index=["store_id", "article_id"], columns="week", values="units", aggfunc="sum", fill_value=0)
    return P


def acf_table(T, name, lags=range(44, 60)):
    x = np.log1p(T)
    x = x - x.rolling(27, center=True, min_periods=10).median()      # remove the slow trend
    x = (x - x.mean()) / x.std()
    ac = {k: float(x.autocorr(k)) for k in lags}
    top = sorted(ac, key=ac.get, reverse=True)[:4]
    print(f"[{name}] autocorrelation of detrended log weekly total at lags 48..56:", {k: round(ac[k], 2) for k in range(48, 57)})
    print(f"[{name}] strongest lags in 44..59:", [(k, round(ac[k], 2)) for k in top])


for shop in ["test", "A"]:
    P = totals(shop)
    T = P.sum(axis=0)
    print(f"\n===== {shop}: {P.shape[0]} series, {P.shape[1]} weeks =====")
    best = {}
    for y in sorted(set(T.index.year)):
        s = T[T.index.year == y]
        best[y] = s.idxmax()
    print("top week per calendar year:", {y: (str(w.date()), int(T[w])) for y, w in best.items()})
    ws = list(best.values())
    print("gaps between yearly top weeks (weeks):", [int((b - a).days / 7) for a, b in zip(ws, ws[1:])])
    acf_table(T, shop)

# peak composition in the test shop
P = totals("test"); T = P.sum(axis=0)
peaks = [pd.Timestamp(x) for x in ["2020-11-16", "2021-11-01", "2022-10-17"]]
tops = []
print("\n===== test shop: who drives the peaks =====")
for p in peaks:
    i = T.index.get_loc(p)
    ctx = np.r_[i - 8:i - 1, i + 2:i + 9]
    base = np.median(P.to_numpy()[:, ctx], axis=1)
    excess = P.to_numpy()[:, i] - base
    pos = np.clip(excess, 0, None)
    order = np.argsort(-pos)
    top20 = pos[order[:20]].sum() / pos.sum()
    n_rise = int((excess > 0).sum())
    tops.append(set(order[:50]))
    print(f"{p.date()}: total {int(T[p])} vs baseline {int(base.sum())} (+{100*(T[p]/base.sum()-1):.0f}%), "
          f"{n_rise} of {len(P)} articles above their baseline, top-20 articles = {100*top20:.0f}% of the excess")
j = lambda a, b: len(a & b) / len(a | b)
print("overlap (Jaccard) of the top-50 excess articles: 2020-21 %.2f, 2021-22 %.2f, 2020-22 %.2f" % (j(tops[0], tops[1]), j(tops[1], tops[2]), j(tops[0], tops[2])))
df = v.load_weekly("test")
oh = df.pivot_table(index="week", values="on_hand", aggfunc="sum")["on_hand"]
for p in peaks:
    i = oh.index.get_loc(p)
    print(f"total on_hand around {p.date()}: ", [int(x) for x in oh.iloc[i - 3:i + 4].to_numpy()])

# what does the forecast window look like in earlier years?
print("\n===== the 13 forecast weeks (2023-04-24 .. 2023-07-17) vs the same calendar weeks before =====")
cut = T.index[-1]
for lag, label in [(52, "52 weeks earlier (2022)"), (104, "104 weeks earlier (2021)"), (50, "50 weeks earlier"), (100, "100 weeks earlier")]:
    idx = [T.index[-1] + pd.Timedelta(weeks=h) - pd.Timedelta(weeks=lag) for h in range(1, 14)]
    vals = [int(T.get(w, np.nan)) if w in T.index else None for w in idx]
    print(f"{label:28s} {vals}")
last13 = [int(x) for x in T.iloc[-13:]]
print(f"{'last 13 known weeks':28s} {last13}")
