"""Throwaway probe (iteration 04, step 4.0): inspect status/sales/promotion codes of donor shops C, D, B."""
from pathlib import Path

import pandas as pd

R = str(Path(__file__).resolve().parents[2] / "hackathon-retail-sets") + "/"
pd.set_option("display.width", 200)

print("=== C ===")
c = pd.read_parquet(R + "set-c/daily.parquet")
for k in ["units", "sales_amount", "discount_pct", "unit_price", "unit_cost", "on_hand"]:
    c[k] = pd.to_numeric(c[k], errors="coerce")
print(c.sales_type.value_counts(dropna=False))
print(c.stock_status.value_counts(dropna=False))
print(pd.crosstab(c.sales_type, c.stock_status, dropna=False))
print(c.is_return.value_counts(dropna=False))
print("share discount_pct>0:", (c.discount_pct > 0).mean())
for g in ["stock_status", "sales_type", "is_phasing_out"]:
    print(c.groupby(g).agg(n=("units", "size"), med_price=("unit_price", "median"), med_disc=("discount_pct", "median"),
                           disc_pos=("discount_pct", lambda s: (s > 0).mean()),
                           med_dsr=("days_since_receipt", lambda s: pd.to_numeric(s, errors="coerce").median()),
                           units_sum=("units", "sum")))
# price relative to the article's own median list price, per stock_status
c["pr"] = c.unit_price / c.groupby("article_id").unit_price.transform("median")
print("median price ratio to article median by stock_status:\n", c.groupby("stock_status").pr.median())
print("by sales_type:\n", c.groupby("sales_type").pr.median())
print("C is_return by units==0 (stock-only rows):\n", pd.crosstab(c.is_return, c.units == 0))
print("phasing x status:\n", pd.crosstab(c.is_phasing_out, c.stock_status))
ns = c.groupby("article_id").stock_status.nunique()
print("C articles with >1 stock_status over time:", (ns > 1).sum(), "of", len(ns))
print("C price endings of .98/.99 share by status:\n", c.groupby("stock_status").unit_price.apply(lambda s: ((s*100).round() % 100).isin([98, 99]).mean()))
mdc = c[c.sales_type.isin(["MD", "PROMO"])]
print("C MD/PROMO article price quantiles (list):", mdc.groupby("sales_type").unit_price.describe())

print("\n=== D ===")
s = pd.read_parquet(R + "set-d/sales.parquet")
st = pd.read_parquet(R + "set-d/stock.parquet")
print(s.sales_type.value_counts(dropna=False))
print(st.stock_status.value_counts(dropna=False))
# stock tier per sales row: sales date within stock [start,end] for the same store/article
s["date"] = pd.to_datetime(s["date"])
st["start_date"] = pd.to_datetime(st.start_date.clip(upper="2262-01-01"))
st["end_date"] = pd.to_datetime(st.end_date.clip(upper="2262-01-01"))  # open-ended ranges use 9999-12-31
s = s.reset_index(drop=True); s["sid"] = s.index
m = s.merge(st[["store_id", "article_id", "start_date", "end_date", "stock_status", "unit_price"]],
            on=["store_id", "article_id"], how="left")
m = m[(m.date >= m.start_date) & (m.date <= m.end_date)]
print("sales rows matched to a stock range:", m.sid.nunique(), "of", len(s), "; matched pairs", len(m))
# D7 grain checks
ov = m.groupby("sid").size()
print("D7 stock ranges overlapping per sales row (store, article, date):", ov.value_counts().sort_index().to_dict())
print("D7 stock overlap tier conflicts (rows with >1 distinct status):", int((m.groupby("sid").stock_status.nunique() > 1).sum()))
print("D7 stock duplicate (store, article, start_date):", int(st.duplicated(["store_id", "article_id", "start_date"]).sum()))
ar_b = pd.read_parquet(R + "set-b/articles.parquet")
print("D7 B articles.article_id unique:", ar_b.article_id.is_unique, len(ar_b))
un = s[~s.sid.isin(m.sid)].copy()
pairs = st.groupby(["store_id", "article_id"]).agg(smin=("start_date", "min"), smax=("end_date", "max")).reset_index()
un = un.merge(pairs, on=["store_id", "article_id"], how="left")
reason = pd.Series("date inside pair span but in a gap", index=un.index)
reason[un.smin.isna()] = "store-article pair absent from stock"
reason[un.smin.notna() & (un.date < un.smin)] = "sale before first stock range"
reason[un.smin.notna() & (un.date > un.smax)] = "sale after last stock range"
print("D7 unmatched sales rows by reason:", reason.value_counts().to_dict(), "; by sales_type:", un.sales_type.value_counts().to_dict(), "; returns share", un.is_return.mean())
print("D7 stock date span:", st.start_date.min(), st.end_date[st.end_date < "2262-01-01"].max(), "sales date span:", s.date.min(), s.date.max())
print(pd.crosstab(m.sales_type, m.stock_status))
sale = m[(m.is_return == 0) & (m.units > 0)].copy()
sale["real"] = sale.sales_amount / sale.units
sale["below"] = sale.real < sale.unit_price - 1e-6
sale["ratio"] = sale.real / sale.unit_price
print(sale.groupby("sales_type").agg(n=("below", "size"), share_below=("below", "mean"), med_ratio=("ratio", "median")))
print(sale.groupby("stock_status").agg(n=("below", "size"), share_below=("below", "mean"), med_ratio=("ratio", "median")))
# is the tier cut in stock unit_price? stock list price by tier, relative to the article median stock price
st["pr"] = st.unit_price / st.groupby("article_id").unit_price.transform("median")
print("D stock unit_price / article median stock price by tier:\n", st.groupby("stock_status").pr.describe()[["count", "25%", "50%", "75%"]])
print("returns by sales_type:\n", s.groupby("sales_type").is_return.mean())

print("\n=== B ===")
p = pd.read_parquet(R + "set-b/promotions.parquet")
pa = pd.read_parquet(R + "set-b/promotion_articles.parquet")
ps = pd.read_parquet(R + "set-b/promotion_stores.parquet")
ar = pd.read_parquet(R + "set-b/articles.parquet")
print(p.mechanic.value_counts(), p.scope_type.value_counts())
print("promos with article rows:", p.promotion_id.isin(pa.promotion_id).sum(), "with store rows:", p.promotion_id.isin(ps.promotion_id).sum())
print(pd.crosstab(p.scope_type, p.promotion_id.isin(pa.promotion_id)))
print(pd.crosstab(p.scope_type, p.promotion_id.isin(ps.promotion_id)))
p["start_date"] = pd.to_datetime(p.start_date); p["end_date"] = pd.to_datetime(p.end_date)
print("duration days:", (p.end_date - p.start_date).dt.days.describe().to_dict())
sku = p[p.scope_type == "SKU"].merge(pa, on="promotion_id")[["promotion_id", "article_id", "start_date", "end_date"]]
cat = p[p.scope_type == "CATEGORY"].merge(ar[["article_id", "category"]], on="category")[["promotion_id", "article_id", "start_date", "end_date"]]
ex = pd.concat([sku, cat], ignore_index=True)
print("expanded promo-article rows:", len(ex), "SKU", len(sku), "CAT", len(cat))
ex["week"] = [pd.date_range(a, b, freq="D").to_period("W-SUN").start_time.unique() for a, b in zip(ex.start_date, ex.end_date)]
ew = ex.explode("week")[["promotion_id", "article_id", "week"]]
cnt = ew.groupby(["article_id", "week"]).promotion_id.nunique()
print("promos per article-week:", cnt.value_counts().sort_index().to_dict(), "max", cnt.max())
print("promos per week:", ew.groupby("week").promotion_id.nunique().describe().to_dict())
