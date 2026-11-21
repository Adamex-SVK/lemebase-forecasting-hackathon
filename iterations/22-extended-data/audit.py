"""Compare source sales with packaged keys and amounts before any model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "hackathon-retail-sets"
KEY = ["store_id", "article_id", "date"]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare(shop, original, packaged):
    tables = []
    for df in (original, packaged):
        df = df[KEY + ["units"]].copy()
        df["date"] = pd.to_datetime(df.date)
        df["units"] = pd.to_numeric(df.units)
        assert df.notna().all().all()
        tables.append(df.groupby(KEY).units.sum().sort_index())
    left, right = tables
    shared = left.index.intersection(right.index)
    delta = left.loc[shared] - right.loc[shared]
    if (delta.abs() > 1e-9).any():
        raise ValueError(f"{shop} has conflicting quantities on {int((delta.abs()>1e-9).sum())} shared keys")
    def weeks(s):
        return s.index.get_level_values("date").to_period("W-SUN").unique()
    return dict(shop=shop, source_rows=len(original), packaged_rows=len(packaged),
                shared_keys=len(shared), source_only_keys=len(left.index.difference(right.index)),
                packaged_only_keys=len(right.index.difference(left.index)),
                source_only_series=len(left.index.droplevel("date").unique().difference(
                    right.index.droplevel("date").unique())),
                source_only_weeks=len(weeks(left).difference(weeks(right))),
                max_shared_units_difference=delta.abs().max())


def run(folder):
    paths = {key: folder / ("lb22-" + name) for key, name in
             dict(A="A.zip", B_lines="B-lines.csv", B_headers="B-headers.csv",
                  C="C.parquet", D="D-sales.csv", D_stock="D-stock.csv").items()}
    hashes = {key: digest(path) for key, path in paths.items()}
    for label in ["D", "D_stock"]:
        manifest = json.loads((folder / "lb22-D.json").read_text())
        entry = next(x for x in manifest if ("inventory" in x["filename"]) == (label == "D_stock"))
        assert hashes[label] == entry["content_details"]["sha256_hash"]
    for label in ["A", "C"]:
        records = json.loads((folder / f"lb22-{label}.json").read_text())["files"]
        entry = next(x for x in records if x["key"].endswith(paths[label].suffix))
        assert "md5:" + hashlib.md5(paths[label].read_bytes()).hexdigest() == entry["checksum"]
    for label, name in [("B_lines","transaction_line_items.csv"), ("B_headers","transactions.csv")]:
        content = paths[label].read_bytes()
        expected = next(x["sha"] for x in json.loads((folder / "lb22-B.json").read_text()) if x["name"] == name)
        assert hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest() == expected
    with ZipFile(paths["A"]) as archive:
        snapshots = sorted(n for n in archive.namelist() if "/Data/Week " in n and n.endswith(" - Sales.csv"))
        assert len(snapshots) == 9
        last = pd.read_csv(archive.open(snapshots[-1])).set_index(["Store", "Product"]).sort_index()
        for path in snapshots[:-1]:
            prev = pd.read_csv(archive.open(path)).set_index(["Store", "Product"]).sort_index()
            pd.testing.assert_frame_equal(prev, last[prev.columns], check_dtype=False)
        a = last.rename_axis(KEY[:2]).rename_axis("date", axis=1).stack().rename("units").reset_index()
    def packaged(name):
        return pd.read_parquet(RAW / name)
    rows = [compare("A", a, packaged("set-a/weekly.parquet").rename(columns={"week_start":"date"}))]
    bh = pd.read_csv(paths["B_headers"])
    bl = pd.read_csv(paths["B_lines"])
    def b_daily(lines, headers):
        assert not lines.duplicated(["transaction_id","line_no"]).any()
        merged = lines.merge(headers[["transaction_id","store_id","timestamp"]], on="transaction_id", validate="many_to_one")
        assert len(merged) == len(lines)
        return merged.assign(date=pd.to_datetime(merged.timestamp).dt.normalize())
    b = b_daily(bl.rename(columns={"product_id":"article_id","qty":"units"}),bh.rename(columns={"txn_ts":"timestamp"}))
    bp = b_daily(packaged("set-b/sales_lines.parquet"), packaged("set-b/baskets.parquet"))
    rows.append(compare("B", b, bp))
    c = pd.read_parquet(paths["C"]).rename(columns={"TXN_DT":"date","STORE":"store_id","PRODUCT_NO":"article_id","QTY_SOLD":"units"})
    rows.append(compare("C", c, packaged("set-c/daily.parquet")))
    d = pd.read_csv(paths["D"]).rename(columns={"Transaction Date":"date","Store":"store_id","Product No":"article_id","Qty Sold":"units"})
    rows.append(compare("D", d, packaged("set-d/sales.parquet")))
    stock = pd.read_csv(paths["D_stock"], usecols=["Start Date","End Date","Product No","Store","Qty on hand"]).rename(
        columns={"Start Date":"start_date","End Date":"end_date","Product No":"article_id","Store":"store_id","Qty on hand":"on_hand"})
    keys = ["store_id","article_id","start_date","end_date","on_hand"]
    dsp = packaged("set-d/stock.parquet")
    for col in ["start_date","end_date"]:
        stock[col] = stock[col].astype(str)
        dsp[col] = dsp[col].astype(str)
    pd.testing.assert_frame_equal(stock[keys].sort_values(keys).reset_index(drop=True),
                                  dsp[keys].sort_values(keys).reset_index(drop=True), check_dtype=False)
    out = HERE / "results"
    pd.DataFrame(rows).to_csv(out/"source_comparison.csv",index=False)
    (out/"source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp"))
    run(parser.parse_args().source_dir)
