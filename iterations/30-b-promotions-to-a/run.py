from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as f
import prepare
import train_gbm as gb
import validate as v

OUT = HERE / "results"
KEY = v.KEYS + ["week"]
BASE = ROOT / "iterations/28-pr20-four-shops/results"
OWN = ROOT / "iterations/29-borrowing/results/A"


def promotion_share(matrix, windows):
    directory = ROOT / "hackathon-retail-sets/set-b"
    lines = ["transaction_id", "line_no"]
    sales = pd.read_parquet(directory / "sales_lines.parquet", columns=lines + ["article_id", "units"])
    receipts = pd.read_parquet(directory / "baskets.parquet", columns=["transaction_id", "store_id", "timestamp"])
    promos = pd.read_parquet(directory / "basket_promotions.parquet", columns=lines).drop_duplicates()
    assert len(promos.merge(sales[lines], on=lines, validate="one_to_one")) == len(promos)
    data = sales.merge(receipts, on="transaction_id", validate="many_to_one")
    data = data.merge(promos.assign(promoted=True), on=lines, how="left", validate="one_to_one")
    data["week"] = prepare.monday(data.timestamp)
    data["store_id"] = data.store_id.astype(str)
    data = data[data.week.isin(matrix.weeks)].copy()
    data["promo_units"] = data.units.where(data.promoted.eq(True), 0)
    weekly = data.groupby(KEY)[["units", "promo_units"]].sum()
    def wide(col):
        return (weekly[col].unstack("week").reindex(index=pd.MultiIndex.from_frame(matrix.keys),
                columns=matrix.weeks).fillna(0).to_numpy())
    total = wide("units")
    np.testing.assert_array_equal(total, matrix.U)
    promoted = wide("promo_units")
    assert ((promoted >= 0) & (promoted <= total)).all()
    def preceding_four(values):
        cumulative = np.pad(values.cumsum(axis=1), ((0,0),(1,0)))
        i, o = windows["series"], windows["origin"]
        return cumulative[i,o] - cumulative[i,o-4]
    den = preceding_four(total)
    return np.divide(preceding_four(promoted), den, out=np.zeros_like(den,dtype=float), where=den>0)


def read_fc(path, grid):
    series = pd.read_parquet(path).set_index(KEY).sort_index().forecast
    assert series.index.is_unique and series.index.equals(grid) and np.isfinite(series).all()
    return series.to_numpy()


train, hold, cut = v.split(v.load_weekly("A"))
b_train, _, b_cut = v.split(v.load_weekly("B"))
a, b = f.ShopMatrix(train), f.ShopMatrix(b_train)
wa, wb = [f.build_windows(m, f.make_origins(m.W)) for m in (a,b)]
promo = promotion_share(b, wb)
assert np.ptp(promo)>0
print("Cuts",cut,b_cut,"training rows",len(wa["y"]),len(wb["y"]),flush=True)
X = np.column_stack((np.vstack((wa["X"],wb["X"])), np.r_[np.full(len(wa["y"]),np.nan),np.zeros(len(promo))])).astype(np.float32)
y = np.r_[wa["y"], wb["y"]]
w = np.r_[wa["w"], wb["w"]]
X_test, scale, valid = f.inference_features(a)
X_test = np.column_stack((X_test, np.full(len(X_test),np.nan))).astype(np.float32)
grid = hold.set_index(KEY).sort_index().index
zero = a.U[:,-8:].sum(axis=1)==0
assert pd.MultiIndex.from_frame(a.keys).equals(grid.droplevel("week").unique())
foundation = np.mean([read_fc(BASE / (n+"_A.parquet"), grid) for n in ("moirai-2.0-small","tirex","chronos-t5-small")], axis=0)
foundation = (3 * foundation + read_fc(OWN / "chronos-2-ft_A.parquet", grid))/4
foundation[np.repeat(zero,13)] = 0
results = []
for arm in ("A only","A+B without promotions","A+B with promotions"):
    splits = 0
    if arm == "A only":
        tree = read_fc(BASE / "lgbm_A.parquet",grid)
    else:
        if arm.endswith("with promotions"):
            X[len(wa["y"]):,-1] = promo
        dataset = lgb.Dataset(X, y, weight=w, feature_name=f.FEATURES+["prior_4w_promo_unit_share"])
        model = lgb.train(gb.PARAMS,dataset,num_boost_round=276)
        splits = int(model.feature_importance()[-1])
        prediction = np.maximum(0, model.predict(X_test).reshape(a.n,13) * scale[:,None])
        prediction[~valid] = 0
        tree = prediction.ravel()
    forecasts = grid.to_frame(index=False).assign(forecast=np.maximum(0,np.round(.3*tree+.7*foundation)))
    metrics = v.score(train,hold,forecasts)
    assert metrics["n_scored"]==599 and metrics["holdout_units_share_scored"]==1
    if arm=="A only":
        assert np.isclose(metrics["score"],21.419128354166283)
    results.append(dict(arm=arm,score=metrics["score"],MASE=metrics["mase"],scored=metrics["n_scored"],promotion_column_splits=splits))
    print(results[-1],flush=True)
pd.DataFrame(results).to_csv(OUT/"scores.csv",index=False)
