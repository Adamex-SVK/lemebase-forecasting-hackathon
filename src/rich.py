"""Iteration 04: rich-column tables for the donor shops C and D (weekly grain).

Mapping follows iterations/04-rich-columns/notes_adamex-svk_rich_schema.md.
Output: data/interim/rich_{C,D}.parquet with columns
  store_id, article_id, week, price, disc_share, promo_flag, markdown_level (float32),
exactly one row per key of the shop's weekly table, same order (D7).

Run: .venv/bin/python -c "import sys; sys.path.insert(0,'src'); import rich; rich.build_all()"
"""
import numpy as np
import pandas as pd

import prepare

RAW = prepare.RAW
OUT = prepare.OUT
KEYS = ["store_id", "article_id", "week"]
RICH_COLS = ["price", "disc_share", "promo_flag", "markdown_level"]
TIER = {"FULL_PRICE": 0, "MARKDOWN_TIER_1": 1, "MARKDOWN_TIER_2": 2, "CLEARANCE": 3, "PROMO": 0}


def _check_keys(wk):
    assert not wk.duplicated(KEYS).any(), "weekly_keys has duplicate keys"


def _finish(wk, parts, shop):
    """Left-join the per-week frames onto the keys, fill zero-unit defaults, assert grain."""
    out = wk[KEYS].reset_index(drop=True)
    n = len(out)
    for p in parts:
        assert not p.duplicated(KEYS).any()
        out = out.merge(p, on=KEYS, how="left", validate="one_to_one")
        assert len(out) == n, f"row count changed in merge ({shop})"
    out["disc_share"] = out["disc_share"].fillna(0.0)  # no units -> 0
    out["promo_flag"] = out["promo_flag"].fillna(0.0)
    for c in RICH_COLS:
        out[c] = out[c].astype("float32")
    out.loc[out["price"] <= 0, "price"] = np.nan  # zero/negative price is not a price
    assert not (out["price"] <= 0).any()
    out = out[KEYS + RICH_COLS]
    assert len(out) == len(wk) and not out.duplicated(KEYS).any()
    assert (out[KEYS].to_numpy() == wk[KEYS].to_numpy()).all(), "order differs from weekly_keys"
    return out


def _in_range(week, shop):
    first, last = prepare.RANGES[shop]
    return (week >= first) & (week <= last)


def rich_c(daily, weekly_keys):
    _check_keys(weekly_keys)
    c = pd.DataFrame({
        "store_id": daily["store_id"].astype(str), "article_id": daily["article_id"].astype(str),
        "week": prepare.monday(daily["date"]),
        "is_return": pd.to_numeric(daily["is_return"]), "units": pd.to_numeric(daily["units"]),
        "list": pd.to_numeric(daily["unit_price"]), "disc": pd.to_numeric(daily["discount_pct"]).fillna(0),
        "type": daily["sales_type"].astype(str),
    })
    c = c[_in_range(c["week"], "C")]
    list_price = c.groupby(KEYS, as_index=False)["list"].mean().rename(columns={"list": "list_price"})
    s = c[(c["is_return"] == 0) & (c["units"] > 0)].copy()
    s["real"] = s["list"] * (1 - s["disc"])
    s["ru"] = s["real"] * s["units"]
    s["du"] = s["units"].where(s["disc"] > 0, 0.0)
    s["promo"] = (s["type"] == "PROMO").astype(float)
    g = s.groupby(KEYS, as_index=False).agg(ru=("ru", "sum"), u=("units", "sum"), du=("du", "sum"),
                                            promo_flag=("promo", "max"))
    g["real_price"] = g["ru"] / g["u"]
    g["disc_share"] = g["du"] / g["u"]
    g = g[KEYS + ["real_price", "disc_share", "promo_flag"]]
    wk = weekly_keys[KEYS].reset_index(drop=True)
    m = wk.merge(list_price, on=KEYS, how="left", validate="one_to_one")
    assert len(m) == len(wk)
    m = m.merge(g, on=KEYS, how="left", validate="one_to_one")
    assert len(m) == len(wk)
    m["price"] = m["real_price"].fillna(m["list_price"])
    m["markdown_level"] = np.nan  # C: no tiers (provisional, see notes)
    return _finish(wk, [m[KEYS + ["price", "disc_share", "promo_flag", "markdown_level"]]], "C")


def _stock_ranges(stock):
    s = stock[["store_id", "article_id", "start_date", "end_date", "stock_status", "unit_price"]].copy()
    s["store_id"] = s["store_id"].astype(str)
    s["article_id"] = s["article_id"].astype(str)
    s["s"] = pd.to_datetime(s["start_date"])
    e = s["end_date"].astype(str).str[:10]
    e = e.where(e < "2200-01-01", "2200-01-01")  # clip open-ended 9999-12-31 before to_datetime
    s["e"] = pd.to_datetime(e)
    s["unit_price"] = s["unit_price"].astype(float)
    s["tier"] = s["stock_status"].map(TIER)
    assert s["tier"].notna().all(), "unknown stock_status"
    s["promo"] = (s["stock_status"] == "PROMO").astype(float)
    s = s.sort_values(["s", "store_id", "article_id"]).reset_index(drop=True)
    # ranges of one series must not overlap, so the covering range is unique
    t = s.sort_values(["store_id", "article_id", "s"])
    same = (t["store_id"] == t["store_id"].shift()) & (t["article_id"] == t["article_id"].shift())
    assert not (same & (t["s"] <= t["e"].shift())).any(), "overlapping stock ranges"
    return s


def _cover(left, date_col, s):
    """For each left row, the stock range of its series covering `date_col` (NaN if none)."""
    n = len(left)
    l = left.sort_values(date_col, kind="stable")
    m = pd.merge_asof(l, s[["store_id", "article_id", "s", "e", "tier", "promo", "unit_price"]],
                      left_on=date_col, right_on="s", by=["store_id", "article_id"])
    assert len(m) == n
    m.index = l.index
    miss = m["e"] < m[date_col]
    for c in ("tier", "promo", "unit_price"):
        m.loc[miss, c] = np.nan
    return m.drop(columns=["s", "e"]).loc[left.index]  # restore left order


def rich_d(sales, stock, weekly_keys):
    _check_keys(weekly_keys)
    wk = weekly_keys[KEYS].reset_index(drop=True)
    s = _stock_ranges(stock)
    d = pd.DataFrame({
        "store_id": sales["store_id"].astype(str), "article_id": sales["article_id"].astype(str),
        "date": pd.to_datetime(sales["date"]), "type": sales["sales_type"].astype(str),
        "is_return": sales["is_return"], "units": sales["units"].astype(float),
        "amount": sales["sales_amount"].astype(float),
    })
    d["week"] = prepare.monday(d["date"])
    d = d[_in_range(d["week"], "D") & (d["is_return"] == 0) & (d["units"] > 0)].reset_index(drop=True)
    assert d["type"].isin(TIER).all(), "unknown sales_type"
    d["lvl"] = d["type"].map(TIER).astype(float)
    # PROMO rows take the tier of the stock range covering the date, else 0
    pr = d["type"] == "PROMO"
    if pr.any():
        cv = _cover(d.loc[pr, ["store_id", "article_id", "date"]], "date", s)
        d.loc[pr, "lvl"] = cv["tier"].fillna(0).to_numpy()
    d["du"] = d["units"].where(d["type"] != "FULL_PRICE", 0.0)
    d["promo"] = pr.astype(float)
    g = d.groupby(KEYS, as_index=False).agg(amount=("amount", "sum"), u=("units", "sum"), du=("du", "sum"),
                                            promo_flag=("promo", "max"), sale_level=("lvl", "max"))
    g["sale_price"] = g["amount"] / g["u"]
    g["disc_share"] = g["du"] / g["u"]
    # stock range covering the week's Monday (for weeks without sales and the PROMO stock status)
    mon = _cover(wk.assign(date=wk["week"]), "date", s)
    assert len(mon) == len(wk)
    m = wk.copy()
    m["stock_level"] = mon["tier"].to_numpy()
    m["stock_promo"] = mon["promo"].to_numpy()
    m["stock_price"] = mon["unit_price"].to_numpy()
    m = m.merge(g[KEYS + ["sale_price", "disc_share", "promo_flag", "sale_level"]], on=KEYS, how="left",
                validate="one_to_one")
    assert len(m) == len(wk)
    m["price"] = m["sale_price"].fillna(m["stock_price"])
    m["markdown_level"] = m["sale_level"].fillna(m["stock_level"])
    m["promo_flag"] = np.maximum(m["promo_flag"].fillna(0), m["stock_promo"].fillna(0))
    return _finish(wk, [m[KEYS + ["price", "disc_share", "promo_flag", "markdown_level"]]], "D")


def _promo_keys(promotions, promotion_articles, promotion_stores, articles, stores):
    """Unique (store_id, article_id, week) keys touched by at least one promotion (D7: no multiplication)."""
    p = promotions[["promotion_id", "scope_type", "category", "start_date", "end_date"]].copy()
    p["start_date"] = pd.to_datetime(p["start_date"])
    p["end_date"] = pd.to_datetime(p["end_date"])
    rows = []
    for pid, a, b in zip(p["promotion_id"], p["start_date"], p["end_date"]):
        for w in prepare.monday(pd.Series(pd.date_range(a, b))).unique():
            rows.append((pid, w))
    pw = pd.DataFrame(rows, columns=["promotion_id", "week"])
    # article scope: SKU via promotion_articles, CATEGORY via articles.category
    sku = p.loc[p["scope_type"] == "SKU", ["promotion_id"]].merge(
        promotion_articles[["promotion_id", "article_id"]].astype(str), on="promotion_id")
    cat = p.loc[p["scope_type"] == "CATEGORY", ["promotion_id", "category"]].merge(
        articles[["article_id", "category"]].astype({"article_id": str}), on="category")[["promotion_id", "article_id"]]
    assert set(p["scope_type"]) <= {"SKU", "CATEGORY"}, "unknown promotion scope"
    pa = pd.concat([sku, cat], ignore_index=True).drop_duplicates()
    # store scope: rows restrict to that store, no rows = chain-wide
    ps = promotion_stores[["promotion_id", "store_id"]].astype({"store_id": str}).drop_duplicates()
    restricted = set(ps["promotion_id"])
    allst = p.loc[~p["promotion_id"].isin(restricted), ["promotion_id"]].merge(
        pd.DataFrame({"store_id": sorted(set(stores))}), how="cross")
    pst = pd.concat([ps, allst], ignore_index=True)
    k = pw.merge(pa, on="promotion_id").merge(pst, on="promotion_id")
    k = k[KEYS].drop_duplicates().reset_index(drop=True)  # at most 2 promotions per article-week: a flag suffices
    k["promo_flag"] = 1.0
    return k


def rich_b(sales_lines, baskets, promotions, promotion_articles, promotion_stores, articles, weekly_keys):
    _check_keys(weekly_keys)
    wk = weekly_keys[KEYS].reset_index(drop=True)
    bk = baskets[["transaction_id", "store_id", "timestamp"]]
    assert bk["transaction_id"].is_unique
    m = sales_lines[["transaction_id", "article_id", "units", "unit_price", "line_discount"]].merge(
        bk, on="transaction_id", how="left", validate="many_to_one")
    assert m["store_id"].notna().all()
    m["store_id"] = m["store_id"].astype(str)
    m["article_id"] = m["article_id"].astype(str)
    m["week"] = prepare.monday(pd.to_datetime(m["timestamp"]))
    m = m[m["units"] > 0]
    m["pu"] = m["unit_price"] * m["units"]
    m["du"] = m["units"].where(m["line_discount"] > 0, 0).astype(float)
    g = m.groupby(KEYS, as_index=False).agg(pu=("pu", "sum"), u=("units", "sum"), du=("du", "sum"))
    g["sale_price"] = g["pu"] / g["u"]
    g["disc_share"] = g["du"] / g["u"]
    g = g[KEYS + ["sale_price", "disc_share"]]
    list_price = articles[["article_id", "price"]].astype({"article_id": str}).rename(columns={"price": "list_price"})
    assert list_price["article_id"].is_unique
    out = wk.merge(g, on=KEYS, how="left", validate="one_to_one")
    assert len(out) == len(wk)
    out = out.merge(list_price, on="article_id", how="left", validate="many_to_one")
    assert len(out) == len(wk)
    out["price"] = out["sale_price"].fillna(out["list_price"])
    out["markdown_level"] = np.nan  # B: no tiers
    pk = _promo_keys(promotions, promotion_articles, promotion_stores, articles, wk["store_id"].unique())
    out = out.merge(pk, on=KEYS, how="left", validate="one_to_one")
    assert len(out) == len(wk)
    return _finish(wk, [out[KEYS + ["price", "disc_share", "promo_flag", "markdown_level"]]], "B")


def load_rich(shop):
    if shop not in ("B", "C", "D"):
        return None
    return pd.read_parquet(OUT / f"rich_{shop}.parquet")


def _report(shop, out, n):
    print(f"{shop}: rows {len(out)} (weekly {n})")
    print((out[RICH_COLS].isna().mean()).round(4).to_string())


def build_all():
    OUT.mkdir(parents=True, exist_ok=True)
    wb = pd.read_parquet(OUT / "weekly_B.parquet", columns=KEYS)
    rb = RAW / "set-b"
    ob = rich_b(pd.read_parquet(rb / "sales_lines.parquet"), pd.read_parquet(rb / "baskets.parquet"),
                pd.read_parquet(rb / "promotions.parquet"), pd.read_parquet(rb / "promotion_articles.parquet"),
                pd.read_parquet(rb / "promotion_stores.parquet"), pd.read_parquet(rb / "articles.parquet"), wb)
    assert len(ob) == len(wb)
    ob.to_parquet(OUT / "rich_B.parquet", index=False)
    _report("B", ob, len(wb))
    wc = pd.read_parquet(OUT / "weekly_C.parquet", columns=KEYS)
    daily = pd.read_parquet(RAW / "set-c/daily.parquet",
                            columns=["date", "store_id", "article_id", "is_return", "units", "unit_price",
                                     "discount_pct", "sales_type"])
    oc = rich_c(daily, wc)
    assert len(oc) == len(wc)
    oc.to_parquet(OUT / "rich_C.parquet", index=False)
    _report("C", oc, len(wc))
    wd = pd.read_parquet(OUT / "weekly_D.parquet", columns=KEYS)
    sales = pd.read_parquet(RAW / "set-d/sales.parquet",
                            columns=["date", "store_id", "article_id", "sales_type", "is_return", "units",
                                     "sales_amount"])
    stock = pd.read_parquet(RAW / "set-d/stock.parquet",
                            columns=["start_date", "end_date", "stock_status", "store_id", "article_id",
                                     "unit_price"])
    od = rich_d(sales, stock, wd)
    assert len(od) == len(wd)
    od.to_parquet(OUT / "rich_D.parquet", index=False)
    _report("D", od, len(wd))
