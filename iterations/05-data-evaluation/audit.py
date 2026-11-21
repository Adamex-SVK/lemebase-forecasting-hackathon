"""Read-only input audit; outputs stay in this iteration. No model training.

Run: .venv/bin/python iterations/05-data-evaluation/audit.py
An explicit --output DIRECTORY can preserve a separate rerun.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import prepare as prep
import validate as val

KEYS = ["store_id", "article_id"]
SHOPS = ["A", "B", "C", "D", "test"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(out, name, rows):
    pd.DataFrame(rows).to_csv(out / name, index=False)


def raw_audit(out):
    files, cols = [], []
    for path in sorted((ROOT / "hackathon-retail-sets").rglob("*.parquet")):
        df = pd.read_parquet(path)
        rel = str(path.relative_to(ROOT))
        files.append(dict(file=rel, rows=len(df), columns=len(df.columns), sha256=digest(path)))
        for c in df:
            s = df[c]
            row = dict(file=rel, column=c, dtype=str(s.dtype), missing=int(s.isna().sum()),
                       distinct=int(s.nunique()))
            if c in ("units", "on_hand", "on_hand_start", "on_hand_end"):
                num = pd.to_numeric(s, errors="coerce").astype(float)
                row.update(negative=int((num < 0).sum()), zero=int((num == 0).sum()),
                           invalid_numeric=int((s.notna() & num.isna()).sum()),
                           infinite=int(np.isinf(num).sum()), minimum=num.min(), maximum=num.max())
            cols.append(row)
    table(out, "raw_files.csv", files)
    table(out, "raw_columns.csv", cols)
    return files


def raw_sales(shop):
    raw = ROOT / "hackathon-retail-sets"
    if shop == "B":
        d = pd.read_parquet(raw / "set-b/sales_lines.parquet")
        b = pd.read_parquet(raw / "set-b/baskets.parquet")
        assert b.transaction_id.is_unique
        d = d.merge(b[["transaction_id", "store_id", "timestamp"]], on="transaction_id",
                    how="left", validate="many_to_one")
        assert d.store_id.notna().all()
        date = "timestamp"
    else:
        path, date = {
            "A": ("set-a/weekly.parquet", "week_start"),
            "C": ("set-c/daily.parquet", "date"),
            "D": ("set-d/sales.parquet", "date"),
            "test": ("test-shop/history.parquet", "week_start"),
        }[shop]
        d = pd.read_parquet(raw / path)
    if shop == "test":
        d["store_id"] = "T"
    d["store_id"] = d.store_id.astype(str)
    d["article_id"] = d.article_id.astype(str)
    d["week"] = prep.monday(d[date])
    d["units"] = pd.to_numeric(d.units).astype(float)
    grain = {"A": KEYS + [date], "B": ["transaction_id", "line_no"],
             "C": KEYS + [date], "test": ["article_id", date]}.get(shop)
    dup = int(d.duplicated(grain).sum()) if grain else None
    first, last = prep.RANGES[shop]
    kept = d[d.week.between(first, last)]
    grouped = kept.groupby(KEYS + ["week"], as_index=False).units.sum()
    neg = grouped.units < 0
    rec = dict(shop=shop, raw_rows=len(d), natural_key_duplicates=dup,
               raw_negative_rows=int((d.units < 0).sum()), raw_net_units=d.units.sum(),
               excluded_boundary_rows=len(d) - len(kept), complete_week_net_units=kept.units.sum(),
               negative_net_series_weeks=int(neg.sum()),
               units_added_by_clipping=float(-grouped.loc[neg, "units"].sum()))
    grouped["units"] = grouped.units.clip(lower=0)
    return grouped, rec


def shop_audit(shop, out):
    path = ROOT / "data/interim" / f"weekly_{shop}.parquet"
    df = pd.read_parquet(path)
    for k in KEYS:
        df[k] = df[k].astype(str)
    assert not df.duplicated(KEYS + ["week"]).any(), shop
    assert df[KEYS + ["week", "units"]].notna().all().all(), shop
    assert np.isfinite(df.units).all() and (df.units >= 0).all(), shop
    assert (df.week.dt.weekday == 0).all(), shop
    assert (df.on_hand.dropna() >= 0).all(), shop
    assert np.isfinite(df.on_hand.dropna()).all(), shop
    assert (df.on_hand_missing == df.on_hand.isna()).all(), shop
    g = df.groupby(KEYS).week.agg(["min", "max", "size"])
    assert (((g["max"] - g["min"]).dt.days // 7 + 1) == g["size"]).all(), shop
    assert (g["max"] == df.week.max()).all(), shop
    assert df.week.nunique() == (df.week.max() - df.week.min()).days // 7 + 1
    raw, rec = raw_sales(shop)
    merged = df[KEYS + ["week", "units"]].merge(raw, on=KEYS + ["week"], how="outer",
                                                  suffixes=("_prepared", "_raw"), validate="one_to_one")
    # Raw activity can include all-zero or boundary-only series excluded by the historical universe rule.
    gap = (merged.units_prepared.fillna(0) - merged.units_raw.fillna(0)).abs()
    rec.update(prepared_units=df.units.sum(), reconciliation_max_abs_error=gap.max(),
               reconciliation_mismatched_rows=int((gap > 1e-8).sum()),
               raw_week_rows_without_prepared_row=int(merged.units_prepared.isna().sum()))
    assert gap.max() < 1e-8, (shop, rec)
    weeks = np.sort(df.week.unique())
    U = df.pivot(index=KEYS, columns="week", values="units").reindex(columns=weeks).fillna(0)
    a = U.to_numpy(dtype=float)
    recent = a[:, -13:]
    allzero = (a.sum(axis=1) == 0)
    since = np.where((a > 0).any(axis=1), (a[:, ::-1] > 0).argmax(axis=1), a.shape[1])
    nz = (a > 0).sum(axis=1)
    posmean = np.divide(a.sum(axis=1), nz, out=np.zeros(len(a)), where=nz > 0)
    posvar = np.divide((a*a).sum(axis=1), nz, out=np.zeros(len(a)), where=nz > 0) - posmean**2
    cv2 = np.divide(np.maximum(posvar, 0), posmean**2, out=np.full(len(a), np.nan), where=posmean > 0)
    totals = a.sum(axis=0)
    recent_mean, prev_mean = totals[-13:].mean(), totals[-26:-13].mean()
    scale = np.abs(np.diff(a, axis=1)).mean(axis=1)
    active = nz > 0
    series = U.index.to_frame(index=False)
    series["shop"] = shop
    series["positive_weeks"] = nz
    series["mean_units_full_shop_grid"] = a.mean(axis=1)
    series["zero_share_full_shop_grid"] = (a == 0).mean(axis=1)
    series["weeks_since_sale"] = since
    series["positive_size_cv_squared"] = cv2
    series["mase_scale_full_history"] = scale
    series["recent13_units"] = recent.sum(axis=1)
    series.to_csv(out / f"series_{shop}.csv", index=False)
    totalrows = pd.DataFrame(dict(shop=shop, week=pd.to_datetime(weeks), total_units=totals,
                                 products_selling=(a > 0).sum(axis=0)))
    totalrows.to_csv(out / f"weekly_totals_{shop}.csv", index=False)
    lags = []
    # Descriptive aggregate correlations, not proof of product seasonality or future predictability.
    for lag in (1, 4, 13, 26, 50, 52):
        n = len(totals) - lag
        corr = np.corrcoef(totals[:-lag], totals[lag:])[0, 1] if n >= 26 else np.nan
        lags.append(dict(shop=shop, lag=lag, paired_weeks=max(n, 0), aggregate_correlation=corr))
    known = df.on_hand.notna()
    zstock = known & (df.on_hand == 0)
    pstock = known & (df.on_hand > 0)
    row = dict(shop=shop, rows=len(df), stores=df.store_id.nunique(), articles=df.article_id.nunique(),
               series=len(a), weeks=len(weeks), first_week=str(pd.Timestamp(weeks[0]).date()),
               last_week=str(pd.Timestamp(weeks[-1]).date()), zero_share_prepared=float((df.units == 0).mean()),
               zero_share_full_shop_grid=float((a == 0).mean()), stock_missing_share=float((~known).mean()),
               negative_stock_clipped=int(df.on_hand_was_negative.sum()),
               stockout_flag_share=float(df.out_of_stock_flag.mean()), allzero_series=int(allzero.sum()),
               median_weekly_units_per_series=float(np.median(a.mean(axis=1))),
               mean_weekly_units=float(a.mean()), no_sale_last8=int((since >= 8).sum()),
               no_sale_last13=int((since >= 13).sum()),
               last13_zero_share=float((recent == 0).mean()),
               last13_vs_previous13_total_change=float(recent_mean / prev_mean - 1) if prev_mean else np.nan,
               intermittent_share_adi_ge_1_32=float((len(weeks) / nz[active] >= 1.32).mean()),
               irregular_positive_size_share_cv2_ge_0_49=float((cv2[active] >= .49).mean()),
               units_mean_stock_zero=float(df.loc[zstock, "units"].mean()),
               units_mean_stock_positive=float(df.loc[pstock, "units"].mean()),
               stock_missing_first13=float(df.loc[df.week.isin(weeks[:13]), "on_hand_missing"].mean()),
               stock_missing_last13=float(df.loc[df.week.isin(weeks[-13:]), "on_hand_missing"].mean()),
               sha256=digest(path))
    cuts = []
    # Nonoverlapping 13-week targets; require >=13 weeks of inputs. Training origins use step=2.
    for cut_index in range(4):
        train_weeks = len(weeks) - 13 * (cut_index + 1)
        if train_weeks < 13:
            cuts.append(dict(shop=shop, cut=cut_index, feasible=False, train_weeks=train_weeks))
            continue
        history, future = a[:, :train_weeks], a[:, train_weeks:train_weeks+13]
        sc = np.abs(np.diff(history, axis=1)).mean(axis=1)
        ok = sc > 0
        cold = history.sum(axis=1) == 0
        total_units = future.sum()
        pred = np.repeat(np.median(history[:, -13:], axis=1)[:, None], 13, axis=1)
        scored_mase = np.abs(future[ok] - pred[ok]).mean(axis=1) / sc[ok]
        zero_mase = future[ok].mean(axis=1) / sc[ok]
        n_origins = len(range(train_weeks - 13, 12, -2)) if train_weeks >= 26 else 0
        cuts.append(dict(shop=shop, cut=cut_index, feasible=True, train_weeks=train_weeks,
                         holdout_start=str(pd.Timestamp(weeks[train_weeks]).date()),
                         holdout_end=str(pd.Timestamp(weeks[train_weeks+12]).date()),
                         training_origins_step2=n_origins, n_series=len(a), scored=int(ok.sum()),
                         zero_scale_excluded=int((~ok).sum()),
                         cold_series=int(cold.sum()), cold_series_sold_after=int((cold & (future.sum(axis=1) > 0)).sum()),
                         cold_holdout_units_share=float(future[cold].sum()/total_units) if total_units else 0,
                         scored_holdout_units_share=float(future[ok].sum()/total_units) if total_units else 0,
                         median13_score=float(100*(1-scored_mase.mean())),
                         zero_score=float(100*(1-zero_mase.mean())),
                         scale_p10=float(np.quantile(sc[ok], .1)), scale_median=float(np.median(sc[ok]))))
    return df, U, row, rec, cuts, lags


def saved_error_audit(out, df, U):
    rows, horizon_rows, per_series = [], [], []
    for k in range(4):
        path = ROOT / f"logs/chronos_four/LightGBM_{k}.parquet"
        if not path.exists():
            continue
        limited = df[df.week <= df.week.max() - pd.Timedelta(weeks=13*k)]
        train, hold, cut = val.split(limited)
        fc = pd.read_parquet(path)
        for c in KEYS:
            fc[c] = fc[c].astype(str)
        assert np.isfinite(fc.forecast).all() and (fc.forecast >= 0).all()
        result = val.score(train, hold, fc, detail=True)
        hist = U.loc[:, U.columns < cut]
        med = hist.iloc[:, -13:].median(axis=1).rename("median13")
        m = hold.merge(fc, on=KEYS+["week"], validate="one_to_one").merge(
            med.reset_index(), on=KEYS, validate="many_to_one")
        scales = val.history_scale(train).rename("scale").reset_index()
        m = m.merge(scales, on=KEYS, validate="many_to_one")
        m = m[m.scale > 0].copy()
        m["lgbm_mase"] = abs(m.units-m.forecast)/m.scale
        m["median_mase"] = abs(m.units-m.median13)/m.scale
        m["h"] = ((m.week-cut).dt.days//7)+1
        for h, part in m.groupby("h"):
            horizon_rows.append(dict(cut=k, horizon=h, n_series=len(part),
                                     lgbm_mase=part.lgbm_mase.mean(), median_mase=part.median_mase.mean()))
        d = m.groupby(KEYS)[["lgbm_mase", "median_mase"]].mean()
        d["recent13_units"] = hist.iloc[:, -13:].sum(axis=1)
        zshare = (hist.iloc[:, -13:] == 0).mean(axis=1)
        stock = train[train.week == train.week.max()].set_index(KEYS).on_hand
        masks = {"all": np.ones(len(d), bool), "recent13_active": d.recent13_units > 0,
                 "recent13_inactive": d.recent13_units == 0,
                 "recent13_zero_share_ge_75pct": zshare.reindex(d.index) >= .75,
                 "stock_last_zero": stock.reindex(d.index) == 0,
                 "stock_last_positive": stock.reindex(d.index) > 0,
                 "stock_last_missing": stock.reindex(d.index).isna()}
        for name, mask in masks.items():
            part = d.loc[mask]
            rows.append(dict(cut=k, group=name, n_series=len(part),
                             lgbm_mase=part.lgbm_mase.mean(), median_mase=part.median_mase.mean(),
                             gain_score_points=100*(part.median_mase-part.lgbm_mase).mean(),
                             share_of_lgbm_scaled_error=float(part.lgbm_mase.sum()/d.lgbm_mase.sum())))
        assert np.isclose(d.lgbm_mase.mean(), result["mase"])
        d["cut"] = k
        per_series.append(d.reset_index())
    table(out, "saved_lgbm_error_groups.csv", rows)
    table(out, "saved_lgbm_error_horizons.csv", horizon_rows)
    if per_series:
        pd.concat(per_series).to_csv(out / "saved_lgbm_per_series.csv", index=False)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rawfiles = raw_audit(out)
    summaries, reconciliation, cuts, lags, prepared_paths = [], [], [], [], []
    for shop in SHOPS:
        df, U, row, rec, cs, ls = shop_audit(shop, out)
        summaries.append(row); reconciliation.append(rec); cuts.extend(cs); lags.extend(ls)
        prepared_paths.append(ROOT / f"data/interim/weekly_{shop}.parquet")
        print(shop, {k: row[k] for k in ("series", "weeks", "zero_share_prepared", "stock_missing_share")}, flush=True)
        if shop == "test":
            saved_error_audit(out, df, U)
            yearly = df.assign(year=df.week.dt.year).groupby("year").agg(
                weeks=("week", "nunique"), mean_stock=("on_hand", "mean"),
                median_stock=("on_hand", "median"), missing_stock=("on_hand_missing", "mean"),
                mean_units=("units", "mean"))
            yearly.to_csv(out / "test_stock_by_year.csv")
    table(out, "shop_summary.csv", summaries)
    table(out, "sales_reconciliation.csv", reconciliation)
    table(out, "validation_feasibility.csv", cuts)
    table(out, "aggregate_lag_correlations.csv", lags)
    manifest = {r["file"]: r["sha256"] for r in rawfiles}
    manifest.update({str(p.relative_to(ROOT)): digest(p) for p in prepared_paths})
    for rel, sha in manifest.items():
        assert digest(ROOT / rel) == sha, f"Input changed during audit: {rel}"
    metadata = dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                    pyarrow=pyarrow.__version__, git_head=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    script_sha256=digest(Path(__file__)), input_sha256=manifest,
                    note="Existing preprocessing and D26 metric audited, not changed or independently certified official. No training.")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("PASS: input hashes unchanged; all prepared keys/grids and sales reconciliations verified.", flush=True)
    print(pd.DataFrame(cuts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
