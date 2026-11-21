"""Checks for src/rich.py. Run: .venv/bin/python tests/test_rich.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rich  # noqa: E402

W = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])


WD = pd.to_datetime(["2025-06-02", "2025-06-09", "2025-06-16"])


def keys(series=("s1", "s2"), W=W):
    return pd.DataFrame([{"store_id": "S", "article_id": a, "week": w} for a in series for w in W])


def crow(a, d, ret, units, price, disc, typ):
    return {"date": d, "store_id": "S", "article_id": a, "is_return": str(ret), "units": str(units),
            "unit_price": str(price), "discount_pct": str(disc), "sales_type": typ}


def test_rich_c_grain_and_values():
    rows = [
        # s1 week 1: plain sale 2 units at 10, plus stock-only row
        crow("s1", "2024-01-01", 0, 2, "10.0", "0", "R"), crow("s1", "2024-01-02", -1, 0, "10.0", "0", "R"),
        # s1 week 2: promo day 1 unit at list 10, 50% off, and 1 plain unit at 10
        crow("s1", "2024-01-09", 0, 1, "10.0", "0.5", "PROMO"), crow("s1", "2024-01-10", 0, 1, "10.0", "0", "R"),
        # s1 week 3: zero-unit week, stock-only rows at list 12
        crow("s1", "2024-01-15", -1, 0, "12.0", "0", "R"), crow("s1", "2024-01-16", -1, 0, "12.0", "0", "R"),
        # s2: plain sales in all weeks, a return in week 1 (ignored)
        crow("s2", "2024-01-01", 0, 1, "5.0", "0", "C"), crow("s2", "2024-01-02", 1, -1, "5.0", "0", "C_RETURN"),
        crow("s2", "2024-01-08", 0, 1, "5.0", "0", "C"), crow("s2", "2024-01-15", 0, 1, "5.0", "0", "C"),
    ]
    k = keys()
    out = rich.rich_c(pd.DataFrame(rows), k)
    assert len(out) == len(k)
    assert (out[["store_id", "article_id", "week"]].values == k.values).all()
    assert list(out.columns) == ["store_id", "article_id", "week"] + rich.RICH_COLS
    o = out.set_index(["article_id", "week"])
    z = o.loc[("s1", W[2])]
    assert z["disc_share"] == 0 and z["price"] == 12 and z["promo_flag"] == 0
    assert o.loc[("s1", W[1]), "promo_flag"] == 1
    assert o.loc[("s1", W[1]), "disc_share"] == 0.5
    assert abs(o.loc[("s1", W[1]), "price"] - 7.5) < 1e-6  # (1*5 + 1*10) / 2
    assert o["promo_flag"].sum() == 1
    assert o.loc[("s2", W[0]), "price"] == 5  # return ignored
    assert (o["price"] > 0).all() and o["disc_share"].between(0, 1).all()
    assert o["markdown_level"].isna().all()
    assert all(out[c].dtype == np.float32 for c in rich.RICH_COLS)
    print("OK test_rich_c_grain_and_values")


def srow(a, d, typ, units, amount, ret=0):
    return {"date": d, "store_id": "S", "article_id": a, "sales_type": typ, "is_return": ret,
            "units": float(units), "sales_amount": float(amount)}


def test_rich_d_tiers():
    sales = pd.DataFrame([
        srow("s1", "2025-06-02", "MARKDOWN_TIER_2", 1, 8),
        srow("s1", "2025-06-09", "MARKDOWN_TIER_1", 1, 9), srow("s1", "2025-06-10", "CLEARANCE", 1, 5),
        srow("s1", "2025-06-10", "FULL_PRICE", 2, 20),
        srow("s2", "2025-06-02", "PROMO", 2, 12), srow("s2", "2025-06-03", "FULL_PRICE", 1, 10, ret=1),
    ])
    stock = pd.DataFrame([
        {"store_id": "S", "article_id": "s2", "start_date": "2025-05-01", "end_date": "9999-12-31",
         "stock_status": "MARKDOWN_TIER_1", "unit_price": 7.0},
        {"store_id": "S", "article_id": "s1", "start_date": "2025-06-14", "end_date": "2025-06-30",
         "stock_status": "CLEARANCE", "unit_price": 4.0},
    ])
    k = keys(W=WD)
    out = rich.rich_d(sales, stock, k)
    assert len(out) == len(k) and (out[["store_id", "article_id", "week"]].values == k.values).all()
    o = out.set_index(["article_id", "week"])
    assert o.loc[("s1", WD[0]), "markdown_level"] == 2
    assert o.loc[("s1", WD[1]), "markdown_level"] == 3  # tier 1 and clearance: max
    assert abs(o.loc[("s1", WD[1]), "disc_share"] - 0.5) < 1e-6  # 2 of 4 units
    assert abs(o.loc[("s1", WD[1]), "price"] - 34 / 4) < 1e-6
    assert o.loc[("s1", WD[2]), "markdown_level"] == 3  # no sales: stock range covering the Monday
    assert o.loc[("s1", WD[2]), "price"] == 4 and o.loc[("s1", WD[2]), "disc_share"] == 0
    # PROMO row takes the tier of the covering stock range; promo flag set
    assert o.loc[("s2", WD[0]), "promo_flag"] == 1 and o.loc[("s2", WD[0]), "markdown_level"] == 1
    assert o.loc[("s2", WD[0]), "price"] == 6 and o.loc[("s2", WD[0]), "disc_share"] == 1
    assert o.loc[("s2", WD[2]), "markdown_level"] == 1  # open-ended range (9999-12-31)
    assert o["promo_flag"].sum() == 1
    assert not o[["disc_share", "promo_flag"]].isna().any().any()
    print("OK test_rich_d_tiers")


def test_rich_zero_price_is_nan():
    c = pd.DataFrame([
        crow("s1", "2024-01-01", 0, 1, "0", "0", "R"),            # zero list price
        crow("s1", "2024-01-08", 0, 1, "10.0", "1", "PROMO"),     # 100% discount -> realised 0
        crow("s1", "2024-01-15", 0, 1, "10.0", "0", "R"),
        crow("s2", "2024-01-01", 0, 1, "5.0", "0", "R")])
    oc = rich.rich_c(c, keys()).set_index(["article_id", "week"])
    assert pd.isna(oc.loc[("s1", W[0]), "price"]) and pd.isna(oc.loc[("s1", W[1]), "price"])
    assert oc.loc[("s1", W[2]), "price"] == 10
    assert (oc["price"].dropna() > 0).all()
    d = pd.DataFrame([srow("s1", "2025-06-02", "FULL_PRICE", 2, 0), srow("s1", "2025-06-09", "FULL_PRICE", 2, 20)])
    st = pd.DataFrame(columns=["store_id", "article_id", "start_date", "end_date", "stock_status", "unit_price"])
    od = rich.rich_d(d, st, keys(W=WD)).set_index(["article_id", "week"])
    assert pd.isna(od.loc[("s1", WD[0]), "price"]) and od.loc[("s1", WD[1]), "price"] == 10
    assert (od["price"].dropna() > 0).all()
    print("OK test_rich_zero_price_is_nan")


def test_rich_duplicate_keys_rejected():
    k = pd.concat([keys(), keys().iloc[:1]], ignore_index=True)
    c = pd.DataFrame([crow("s1", "2024-01-01", 0, 1, "10.0", "0", "R")])
    d = pd.DataFrame([srow("s1", "2024-01-01", "FULL_PRICE", 1, 10)])
    st = pd.DataFrame(columns=["store_id", "article_id", "start_date", "end_date", "stock_status", "unit_price"])
    for fn in (lambda: rich.rich_c(c, k), lambda: rich.rich_d(d, st, k)):
        try:
            fn()
        except AssertionError:
            continue
        raise SystemExit("duplicate keys not rejected")
    print("OK test_rich_duplicate_keys_rejected")


def _b_inputs():
    """2 stores, 3 weeks (Mon 2024-01-01/08/15), articles a1,a2 in CAT-1, a3 in CAT-2."""
    arts = pd.DataFrame({"article_id": ["a1", "a2", "a3"], "category": ["CAT-1", "CAT-1", "CAT-2"],
                         "price": [10.0, 4.0, 0.0]})
    prom = pd.DataFrame([
        # P1 SKU a1, chain-wide, Sun 2024-01-07 .. Fri 2024-01-12: touches weeks 1 and 2
        {"promotion_id": "P1", "scope_type": "SKU", "category": None, "start_date": "2024-01-07", "end_date": "2024-01-12"},
        # P2 CATEGORY CAT-1 (a1, a2), store 1 only, week 2 (Mon-Sat)
        {"promotion_id": "P2", "scope_type": "CATEGORY", "category": "CAT-1", "start_date": "2024-01-08", "end_date": "2024-01-13"},
        # P3 SKU a1, store 2 only, week 3: no overlap with a1 in week 2 on other store
        {"promotion_id": "P3", "scope_type": "SKU", "category": None, "start_date": "2024-01-15", "end_date": "2024-01-20"},
    ])
    pa = pd.DataFrame({"promotion_id": ["P1", "P3"], "article_id": ["a1", "a1"]})
    ps = pd.DataFrame({"promotion_id": ["P2", "P3"], "store_id": [1, 2]})
    k = pd.DataFrame([{"store_id": s, "article_id": a, "week": w} for s in ("1", "2")
                      for a in ("a1", "a2", "a3") for w in W])
    return arts, prom, pa, ps, k


def _lines(rows):
    sl = pd.DataFrame([{"transaction_id": t, "line_no": 1, "article_id": a, "units": u, "unit_price": p,
                        "line_discount": d} for t, a, u, p, d in rows])
    return sl


def _baskets(rows):
    return pd.DataFrame([{"transaction_id": t, "store_id": s, "timestamp": ts} for t, s, ts in rows])


def test_rich_b_promo_expansion():
    arts, prom, pa, ps, k = _b_inputs()
    sl = _lines([("t1", "a1", 1, 10.0, 0.0)])
    bk = _baskets([("t1", 1, "2024-01-02 10:00:00")])
    out = rich.rich_b(sl, bk, prom, pa, ps, arts, k)
    assert len(out) == len(k)
    o = out.set_index(["store_id", "article_id", "week"])["promo_flag"]
    exp = {}
    # P1: chain-wide a1, weeks 1 and 2, both stores
    for s in ("1", "2"):
        exp[(s, "a1", W[0])] = 1
        exp[(s, "a1", W[1])] = 1
    # P2: store 1, a1 and a2, week 2 (a1 also in P1: still one row, flag 1)
    exp[("1", "a2", W[1])] = 1
    # P3: store 2, a1, week 3
    exp[("2", "a1", W[2])] = 1
    for key, v in o.items():
        assert v == exp.get(key, 0), (key, v)
    assert o.sum() == len(exp) == 6  # P2 on a1 duplicates P1
    assert o.loc[("2", "a2", W[1])] == 0  # P2 is store 1 only
    assert o.loc[("1", "a3", W[1])] == 0  # CAT-2 not covered
    assert list(out.columns) == ["store_id", "article_id", "week"] + rich.RICH_COLS
    assert out["markdown_level"].isna().all()
    print("OK test_rich_b_promo_expansion")


def test_rich_b_disc_share():
    arts, prom, pa, ps, k = _b_inputs()
    sl = _lines([("t1", "a1", 2, 10.0, 0.0), ("t2", "a1", 1, 10.0, 1.0), ("t3", "a1", 1, 10.0, 2.0),
                 ("t4", "a2", 3, 4.0, 0.0)])
    bk = _baskets([("t1", 1, "2024-01-02 10:00:00"), ("t2", 1, "2024-01-03 10:00:00"),
                   ("t3", 2, "2024-01-03 10:00:00"), ("t4", 1, "2024-01-16 10:00:00")])
    out = rich.rich_b(sl, bk, prom, pa, ps, arts, k).set_index(["store_id", "article_id", "week"])
    r = out.loc[("1", "a1", W[0])]
    assert abs(r["disc_share"] - 1 / 3) < 1e-6 and r["price"] == 10  # 1 of 3 units discounted
    assert out.loc[("2", "a1", W[0]), "disc_share"] == 1
    z = out.loc[("1", "a2", W[0])]  # zero-unit week: no discount, list price
    assert z["disc_share"] == 0 and z["price"] == 4
    assert out.loc[("1", "a2", W[2]), "price"] == 4
    assert pd.isna(out.loc[("1", "a3", W[0]), "price"])  # list price 0 -> NaN
    assert out["disc_share"].between(0, 1).all()
    print("OK test_rich_b_disc_share")


def test_rich_b_rowcount():
    arts, prom, pa, ps, k = _b_inputs()
    sl = _lines([("t1", "a1", 1, 10.0, 0.0), ("t2", "a1", 1, 10.0, 0.0)])
    bk = _baskets([("t1", 1, "2024-01-09 10:00:00"), ("t2", 1, "2024-01-10 10:00:00")])
    out = rich.rich_b(sl, bk, prom, pa, ps, arts, k)
    assert len(out) == len(k) and not out.duplicated(["store_id", "article_id", "week"]).any()
    assert (out[["store_id", "article_id", "week"]].values == k.values).all()
    kd = pd.concat([k, k.iloc[:1]], ignore_index=True)
    try:
        rich.rich_b(sl, bk, prom, pa, ps, arts, kd)
    except AssertionError:
        print("OK test_rich_b_rowcount")
        return
    raise SystemExit("duplicate keys not rejected")


def test_load_rich_none():
    assert rich.load_rich("A") is None and rich.load_rich("test") is None
    print("OK test_load_rich_none")


test_rich_c_grain_and_values()
test_rich_d_tiers()
test_rich_zero_price_is_nan()
test_rich_duplicate_keys_rejected()
test_load_rich_none()
test_rich_b_promo_expansion()
test_rich_b_disc_share()
test_rich_b_rowcount()
