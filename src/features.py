"""Step 1.1: features and training windows for the global LightGBM (iteration 01).

Decisions: D11/D20/D29 (target = units / the series' MASE scale, L1 loss), D26 (history
is zero-padded from the shop's first week, so the scale is exactly the metric's scale),
D28 (weighted sampling of inactive windows for big shops), F6 (no calendar features).

Only things the test shop has are used: units, on_hand, out-of-stock flag, plus the
horizon h. Every feature of a window with origin `o` uses ONLY weeks before `o`.

A "window" is (series, origin o); the model predicts units in week o+h-1 for h = 1..13.
"""
import warnings

import numpy as np
import pandas as pd

KEYS = ["store_id", "article_id"]
H = 13
N_LAGS = 26
MIN_HIST = 13
DEAD_CAP = 104

FEATURES = (
    [f"lag_{k}" for k in range(1, N_LAGS + 1)]
    + ["mean4", "mean8", "mean13", "mean26", "mean52", "median13", "median26", "max13",
       "zshare13", "zshare26", "weeks_since_sale", "series_age", "log_scale", "log_mean13",
       "oh_last_log", "oh_missing_last", "oos_last", "oos_share4", "oh_zero_share4",
       "shop_ratio4_26", "h"]
)


FEATURES_D33 = FEATURES[:-1] + ["cover_last", "cover_avg4", "h"]   # D33: stock relative to recent sales

# Iteration 04: rich columns (price, discounts, promotions, markdowns), only for shops that have them.
RICH_FEATURES = ["price_ratio4", "disc4", "disc13", "promo4", "promo13", "markdown_last", "markdown_max4",
                 "weeks_since_promo", "rich_missing"]


def feature_names(rich=False, d33=False):
    """Column names of the model matrix, `h` last; the rich columns sit before `h`."""
    base = FEATURES_D33 if d33 else FEATURES
    return base[:-1] + RICH_FEATURES + ["h"] if rich else base


class ShopMatrix:
    """Series x weeks matrices of one shop (units, on_hand, out-of-stock flag).

    mask_oh_before (D33): `on_hand` before this date is treated as missing (the test shop's
    old stock regime, which has a different scale).
    rich: optional frame (store_id, article_id, week + rich.RICH_COLS); fills M.R[col] (n x W, NaN where
    missing). Rows whose key or week is not in df are ignored."""

    def __init__(self, df, mask_oh_before=None, rich=None):
        self.keys = df[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
        self.weeks = np.sort(df["week"].unique())
        steps = np.diff(self.weeks).astype("timedelta64[D]").astype(int)
        assert (steps == 7).all(), "weeks must be contiguous"
        n, w = len(self.keys), len(self.weeks)
        r = pd.MultiIndex.from_frame(self.keys).get_indexer(pd.MultiIndex.from_frame(df[KEYS]))
        c = pd.Index(self.weeks).get_indexer(df["week"])
        self.U = np.zeros((n, w), dtype=np.float32)
        self.U[r, c] = df["units"].to_numpy()
        self.OH = np.full((n, w), np.nan, dtype=np.float32)
        known = ~df["on_hand_missing"].to_numpy()
        self.OH[r[known], c[known]] = df["on_hand"].to_numpy()[known]
        if mask_oh_before is not None:
            self.OH[:, self.weeks < np.datetime64(pd.Timestamp(mask_oh_before))] = np.nan
        self.OOS = np.zeros((n, w), dtype=bool)
        self.OOS[r, c] = df["out_of_stock_flag"].to_numpy()
        self.n, self.W = n, w
        self.has_rich = rich is not None
        self.R = {}
        if self.has_rich:
            from rich import RICH_COLS
            rr = pd.MultiIndex.from_frame(self.keys).get_indexer(pd.MultiIndex.from_frame(rich[KEYS]))
            rc = pd.Index(self.weeks).get_indexer(rich["week"])
            ok = (rr >= 0) & (rc >= 0)
            for col in RICH_COLS:
                a = np.full((n, w), np.nan, dtype=np.float32)
                a[rr[ok], rc[ok]] = rich[col].to_numpy(dtype=np.float32)[ok]
                self.R[col] = a


def features_at(M, o, d33=False, rich=False):
    """Features of every series at origin o (first forecast week index), using weeks < o only.

    Returns (X [n x len(FEATURES)-1 float32, no `h`], scale [n float64], valid [n bool])."""
    assert MIN_HIST <= o <= M.W
    U = M.U[:, :o].astype(np.float64)
    n = M.n
    scale = np.abs(np.diff(U, axis=1)).mean(axis=1)
    valid = scale > 0
    s = np.where(valid, scale, 1.0)
    cols = {}
    for k in range(1, N_LAGS + 1):
        cols[f"lag_{k}"] = U[:, o - k] / s if o - k >= 0 else np.full(n, np.nan)

    def roll(w, fn):
        return fn(U[:, o - w:o], axis=1) / s if o >= w else np.full(n, np.nan)

    for w in (4, 8, 13, 26, 52):
        cols[f"mean{w}"] = roll(w, np.mean)
    cols["median13"] = roll(13, np.median)
    cols["median26"] = roll(26, np.median)
    cols["max13"] = roll(13, np.max)
    cols["zshare13"] = (U[:, o - 13:o] == 0).mean(axis=1)
    cols["zshare26"] = (U[:, max(0, o - 26):o] == 0).mean(axis=1)
    rev = U[:, ::-1] > 0
    any_sale = rev.any(axis=1)
    cols["weeks_since_sale"] = np.where(any_sale, np.minimum(rev.argmax(axis=1), DEAD_CAP), DEAD_CAP)
    first = (U > 0).argmax(axis=1)
    cols["series_age"] = np.where(any_sale, o - first, 0)
    cols["log_scale"] = np.log1p(scale)
    cols["log_mean13"] = np.log1p(U[:, o - 13:o].mean(axis=1))
    oh = M.OH[:, o - 1].astype(np.float64)
    cols["oh_last_log"] = np.log1p(np.clip(oh, 0, None))
    cols["oh_missing_last"] = np.isnan(oh).astype(float)
    cols["oos_last"] = M.OOS[:, o - 1].astype(float)
    cols["oos_share4"] = M.OOS[:, o - 4:o].mean(axis=1)
    oh4 = M.OH[:, o - 4:o].astype(np.float64)
    known = ~np.isnan(oh4)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["oh_zero_share4"] = np.where(known.sum(axis=1) > 0, ((oh4 == 0) & known).sum(axis=1) / known.sum(axis=1), np.nan)
    T = M.U[:, :o].sum(axis=0).astype(np.float64)
    cols["shop_ratio4_26"] = np.full(n, T[o - 4:o].mean() / (T[max(0, o - 26):o].mean() + 1e-9))
    names = FEATURES
    if d33:   # weeks of stock cover: on_hand relative to the recent sales level (raw units, not scaled)
        level = U[:, o - 13:o].mean(axis=1) + 1.0
        cols["cover_last"] = np.log1p(np.clip(oh, 0, None) / level)
        with np.errstate(invalid="ignore"):
            avg4 = np.where(known.sum(axis=1) > 0, np.nansum(np.where(known, oh4, 0.0), axis=1) / np.maximum(known.sum(axis=1), 1), np.nan)
        cols["cover_avg4"] = np.log1p(np.clip(avg4, 0, None) / level)
        names = FEATURES_D33
    if rich:   # only weeks < o of M.R are read
        nan = np.full(n, np.nan)
        if M.has_rich:
            R = {c: M.R[c][:, :o].astype(np.float64) for c in M.R}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                num = np.nanmean(R["price"][:, o - 4:o], axis=1)
                med = np.nanmedian(R["price"][:, max(0, o - 26):o], axis=1)
                cols["price_ratio4"] = np.where(med > 0, num / np.where(med > 0, med, 1.0), np.nan)   # never inf
                cols["disc4"] = np.nanmean(R["disc_share"][:, o - 4:o], axis=1)
                cols["disc13"] = np.nanmean(R["disc_share"][:, o - 13:o], axis=1)
                cols["promo4"] = np.nanmean(R["promo_flag"][:, o - 4:o], axis=1)
                cols["promo13"] = np.nanmean(R["promo_flag"][:, o - 13:o], axis=1)
                cols["markdown_max4"] = np.nanmax(R["markdown_level"][:, o - 4:o], axis=1)
            cols["markdown_last"] = R["markdown_level"][:, o - 1]
            # DEAD_CAP also when promo_flag is all NaN; rich_missing tells the two cases apart
            prev = (R["promo_flag"] == 1)[:, ::-1]
            cols["weeks_since_promo"] = np.where(prev.any(axis=1), np.minimum(prev.argmax(axis=1), DEAD_CAP), DEAD_CAP)
            cols["rich_missing"] = np.all([np.isnan(R[c][:, o - 13:o]).all(axis=1) for c in R], axis=0).astype(float)
        else:
            for name in RICH_FEATURES:
                cols[name] = nan
            cols["rich_missing"] = np.ones(n)
        names = names[:-1] + RICH_FEATURES + ["h"]
    X = np.column_stack([cols[f] for f in names if f != "h"]).astype(np.float32)
    return X, scale, valid


def make_origins(W, step=2, horizon=H, min_hist=MIN_HIST):
    """Origins with all `horizon` target weeks inside the data; the latest possible origin is
    always included (so the most recent behaviour is represented)."""
    last = W - horizon
    if last < min_hist:
        return np.array([], dtype=int)
    return np.arange(last, min_hist - 1, -step)[::-1]


def build_windows(M, origins, sample_inactive=False, keep_inactive=0.10, weight_inactive=10.0,
                  rng=None, horizon=H, d33=False, rich=False, rich_drop=0.0):
    """Training rows: one row per (kept series-origin) x h. Returns dict with
    X [rows x len(FEATURES)] (last column h), y (units / scale), w (sample weight),
    series (row index into M.keys), origin (week index), plus n_windows.

    rich=True appends the RICH_FEATURES columns (before h). rich_drop: share of windows (one random
    decision per series-origin, all h rows alike) whose rich columns are masked: all NaN except
    rich_missing = 1. No random draw is made when rich_drop is 0 or rich is False."""
    rng = rng or np.random.default_rng(0)
    names = feature_names(rich, d33)
    rcols = [i for i, nm in enumerate(names) if nm in RICH_FEATURES and nm != "rich_missing"]
    miss_col = names.index("rich_missing") if rich else None
    Xs, ys, ws, ser, org = [], [], [], [], []
    for o in origins:
        assert o + horizon <= M.W
        X, scale, valid = features_at(M, o, d33, rich)
        active = M.U[:, max(0, o - 26):o].sum(axis=1) > 0
        w = np.ones(M.n)
        keep = valid.copy()
        if sample_inactive:
            lucky = rng.random(M.n) < keep_inactive
            keep &= active | lucky
            w = np.where(active, 1.0, weight_inactive)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        Y = M.U[idx, o:o + horizon].astype(np.float64) / scale[idx, None]
        Xi = X[idx]
        if rich and rich_drop > 0:
            drop = rng.random(len(idx)) < rich_drop
            Xi = Xi.copy()
            Xi[np.ix_(drop, rcols)] = np.nan
            Xi[drop, miss_col] = 1.0
        for h in range(1, horizon + 1):
            Xh = np.column_stack([Xi, np.full(len(idx), h, dtype=np.float32)])
            Xs.append(Xh); ys.append(Y[:, h - 1].astype(np.float32)); ws.append(w[idx].astype(np.float32))
            ser.append(idx); org.append(np.full(len(idx), o))
    if not Xs:
        return {"X": np.zeros((0, len(names)), np.float32), "y": np.zeros(0, np.float32), "w": np.zeros(0, np.float32),
                "series": np.zeros(0, int), "origin": np.zeros(0, int), "n_windows": 0}
    return {"X": np.vstack(Xs), "y": np.concatenate(ys), "w": np.concatenate(ws),
            "series": np.concatenate(ser), "origin": np.concatenate(org),
            "n_windows": int(sum(len(i) for i in ser) // horizon)}


def inference_features(M, horizon=H, d33=False, rich=False):
    """Rows for forecasting the `horizon` weeks after the last week of M.

    Returns (X [n*horizon x len(FEATURES)], scale [n], valid [n]); rows are ordered
    series-major (series 0 h=1..13, series 1 ...)."""
    X, scale, valid = features_at(M, M.W, d33, rich)
    Xh = np.concatenate([np.column_stack([X, np.full(M.n, h, dtype=np.float32)])[:, None, :] for h in range(1, horizon + 1)], axis=1)
    return Xh.reshape(M.n * horizon, -1), scale, valid
