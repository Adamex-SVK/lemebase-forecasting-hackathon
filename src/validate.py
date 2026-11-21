"""Step 0.5: the local metric and the validation cuts (D14, D15).

Score as stated on the brief's page: for each article, the mean over the 13
forecast weeks of |actual - forecast| / (how much the article usually moves
from one week to the next in its history); average over articles (MASE);
score = (1 - MASE) * 100.

Conventions the page does not pin down are explicit parameters, so that
step 0.7 can find the ones that reproduce the brief's published baselines (F1, F8):
  level      'series'  one series per store x article (the test shop has no stores)
             'article' sum over stores first
  scale_mode 'full'            history = all training weeks of the series
             'from_first_sale' history starts at the first non-zero week
  zero scale (constant history) and series without history cannot be scored
             and are excluded and counted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IM = ROOT / "data" / "interim"
HORIZON = 13
KEYS = ["store_id", "article_id"]


def load_weekly(shop):
    return pd.read_parquet(IM / f"weekly_{shop}.parquet")


def cut_week(df, horizon=HORIZON):
    """First holdout week: the last `horizon` complete weeks are held out (D14)."""
    return df["week"].max() - pd.Timedelta(weeks=horizon - 1)


def pad_to_shop_start(train, series):
    """D26: every series is zero-filled from the shop's first week (leading zeros),
    as the brief's published baselines imply. `series` = all series of the shop."""
    weeks = pd.DataFrame({"week": np.sort(train["week"].unique())})
    g = series.merge(weeks, how="cross").merge(train, on=KEYS + ["week"], how="left")
    g["units"] = g["units"].fillna(0.0)
    for c, fill in [("on_hand_missing", True), ("on_hand_was_negative", False), ("out_of_stock_flag", False)]:
        if c in g:
            g[c] = g[c].fillna(fill)
    if "shop" in g:
        g["shop"] = g["shop"].ffill().bfill()
    return g


def split(df, horizon=HORIZON, pad=True):
    """Return (train, holdout, cut). Holdout is one row per series per holdout week;
    weeks before a series' start count as 0 units (no sale, no stock record).
    With pad=True (D26, default) the training frame has every series from the
    shop's first week; series that only start in the holdout get an all-zero history."""
    cut = cut_week(df, horizon)
    train = df[df["week"] < cut]
    if pad:
        train = pad_to_shop_start(train, df[KEYS].drop_duplicates())
    weeks = pd.DataFrame({"week": pd.date_range(cut, periods=horizon, freq="7D")})
    series = df[KEYS].drop_duplicates()
    hold = series.merge(weeks, how="cross").merge(
        df.loc[df["week"] >= cut, KEYS + ["week", "units"]], on=KEYS + ["week"], how="left")
    hold["units"] = hold["units"].fillna(0.0)
    return train, hold, cut


def to_article_level(df, cols=("units",)):
    """Sum over stores (level='article'). store_id becomes '*'."""
    g = df.groupby(["article_id", "week"], as_index=False)[list(cols)].sum()
    g["store_id"] = "*"
    return g


def history_scale(train, scale_mode="full"):
    """Mean absolute week-to-week change of each series in its training history."""
    t = train.sort_values(KEYS + ["week"])[KEYS + ["week", "units"]]
    if scale_mode == "from_first_sale":
        t = t[t.groupby(KEYS)["units"].cumsum() > 0]
    elif scale_mode != "full":
        raise ValueError(scale_mode)
    t = t.assign(d=t.groupby(KEYS)["units"].diff().abs())
    return t.groupby(KEYS)["d"].mean().rename("scale")  # NaN when fewer than 2 weeks


def score(train, hold, fc, level="series", scale_mode="full", horizon=HORIZON, detail=False):
    """fc: DataFrame with store_id, article_id, week, forecast (one row per series-week).

    Returns a dict with the score and how many series were scored / excluded."""
    if level == "article":
        train, hold, fc = (to_article_level(train), to_article_level(hold),
                           to_article_level(fc.rename(columns={"forecast": "units"})).rename(columns={"units": "forecast"}))
    elif level != "series":
        raise ValueError(level)
    m = hold.merge(fc[KEYS + ["week", "forecast"]], on=KEYS + ["week"], how="left", validate="one_to_one")
    if m["forecast"].isna().any():
        raise ValueError(f"{int(m['forecast'].isna().sum())} holdout rows have no forecast")
    m["ae"] = (m["units"] - m["forecast"]).abs()
    err = m.groupby(KEYS)["ae"].mean().rename("err").to_frame()
    err["n_weeks"] = m.groupby(KEYS).size()
    assert (err["n_weeks"] == horizon).all()
    err = err.join(history_scale(train, scale_mode))
    n = len(err)
    no_hist = err["scale"].isna()
    zero = err["scale"] == 0
    ok = err[~no_hist & ~zero].copy()
    ok["mase"] = ok["err"] / ok["scale"]
    mase = float(ok["mase"].mean()) if len(ok) else float("nan")
    hu = m.groupby(KEYS)["units"].sum()
    res = {
        "score": (1 - mase) * 100, "mase": mase,
        "n_series": n, "n_scored": len(ok),
        "n_no_history": int(no_hist.sum()), "n_zero_scale": int(zero.sum()),
        "holdout_units_share_scored": float(hu.loc[ok.index].sum() / hu.sum()) if hu.sum() else float("nan"),
    }
    if detail:
        res["detail"] = ok  # per series: err, scale, mase
    return res


def summary():
    rows = []
    for shop in ["A", "B", "C", "D", "test"]:
        df = load_weekly(shop)
        train, hold, cut = split(df)
        for level in ["series", "article"] if shop != "test" else ["series"]:
            for sm in ["full", "from_first_sale"]:
                tr, ho = (to_article_level(train), to_article_level(hold)) if level == "article" else (train, hold)
                sc = history_scale(tr, sm)
                allk = ho[KEYS].drop_duplicates().set_index(KEYS).index
                sc = sc.reindex(allk)
                hu = ho.groupby(KEYS)["units"].sum().reindex(allk)
                ok = sc.notna() & (sc > 0)
                rows.append({"shop": shop, "level": level, "scale_mode": sm,
                             "cut_first_holdout_week": str(cut.date()),
                             "train_weeks": int(train["week"].nunique()), "n_series": len(allk),
                             "no_history": int(sc.isna().sum()), "zero_scale": int((sc == 0).sum()),
                             "scored": int(ok.sum()),
                             "holdout_units_in_scored_%": round(100 * hu[ok].sum() / hu.sum(), 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(summary().to_string(index=False))
