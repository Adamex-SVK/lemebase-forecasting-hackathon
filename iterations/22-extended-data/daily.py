"""Create pharmacy transaction features from exact sales prefixes at each origin."""
from collections import defaultdict
from pathlib import Path
import argparse
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SOURCE_START = pd.Timestamp("2023-01-02")
NAMES = ["selling_days28", "days_since_positive_sale"]


def inputs(path):
    target = ROOT / "hackathon-retail-sets/test-shop/history.parquet"
    frame = pd.read_parquet(target)
    hist = frame.pivot(index="article_id",columns="week_start",values="units").sort_index(axis=1)
    assert len(hist.columns) == 143 and pd.Timestamp(hist.columns[0])+pd.Timedelta(weeks=127) == SOURCE_START
    with zipfile.ZipFile(path) as z:
        member = next(name for name in z.namelist() if name.endswith("/raw_dataset/sales_raw.csv"))
        with z.open(member) as handle:
            sales = pd.read_csv(handle, usecols=["DateTime", "DrugCode", "Quantity"])
    date = pd.to_datetime(sales.DateTime,format="%d %b %Y %H:%M:%S").dt.normalize()
    day = (date-SOURCE_START).dt.days
    valid = day.between(0, hist.shape[1]*7-1)
    return hist, sales.loc[valid,["DrugCode","Quantity"]].assign(day=day[valid])


def asof_arrays(sales, weeks):
    codes = pd.Index(sales.DrugCode.unique())
    calendar = sales.assign(week=sales.day//7).groupby(["DrugCode","week"]).Quantity.sum()
    weekly = calendar.unstack(fill_value=0).reindex(index=codes,columns=range(weeks),fill_value=0)
    active = sales[sales.Quantity>0]
    positive = np.zeros((len(codes),weeks*7),bool)
    positive[codes.get_indexer(active.DrugCode),active.day] = True
    return weekly.to_numpy(dtype=np.float64), positive


def at(target, sales_weeks, positive_days, origin):
    end = 7*origin
    seen = positive_days[:, :end]
    vals = np.column_stack((seen[:, -28:].sum(axis=1),
                           np.where(seen.any(axis=1),1+seen[:, ::-1].argmax(axis=1),np.nan)))
    matches = defaultdict(list)
    for key, val in zip(sales_weeks[:, :origin],vals):
        matches[key.tobytes()].append(val)
    # A source code first appearing after the origin has known zero sales to date.
    matches[np.zeros(origin, np.float64).tobytes()].append([0., np.nan])
    resolved = {}
    for key, candidates in matches.items():
        a = np.asarray(candidates)
        same = np.all((a == a[0]) | (np.isnan(a) & np.isnan(a[0])), axis=0)
        resolved[key] = np.where(same,a[0],np.nan)
    return np.asarray([resolved[key.tobytes()] for key in target[:, :origin]],dtype=np.float32)


def build(path):
    hist,sales = inputs(path)
    target = hist.to_numpy(dtype=np.float64)
    weekly, days = asof_arrays(sales,hist.shape[1])
    origins = range(13,hist.shape[1]+1)
    rows = np.stack([at(target,weekly,days,o) for o in origins])
    for o in (13,65,91,117,130,143):
        past_weekly,past_days = asof_arrays(sales[sales.day<7*o],o)
        np.testing.assert_array_equal(rows[o-13], at(target[:,:o],past_weekly,past_days,o))
    path = HERE / "results/daily_features.npz"
    np.savez_compressed(path,features=rows,ids=hist.index.to_numpy(dtype=str))
    print(f"Built {len(rows)} origin-specific feature matrices; six truncated-source checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("historical_archive", type=Path)
    build(parser.parse_args().historical_archive)
