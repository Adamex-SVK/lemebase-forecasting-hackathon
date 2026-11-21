"""Audit UCI Online Retail and produce a weekly merchandise-sales donor table."""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_HASH = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != SOURCE_HASH:
        raise ValueError("Source does not match the reviewed release")
    with ZipFile(args.source) as archive:
        book = openpyxl.load_workbook(io.BytesIO(archive.read("Online Retail.xlsx")),
                                      read_only=True, data_only=True)
        records = book.active.values
        columns = next(records)
        raw = pd.DataFrame.from_records(records, columns=columns)
        book.close()
    repeated = raw.duplicated()
    lines = raw.loc[~repeated].copy()
    dates = pd.to_datetime(lines.InvoiceDate).dt.normalize()
    lines["week"] = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    # The workbook starts on Wednesday and ends on Friday.
    full = (lines.week > lines.week.min()) & (lines.week < lines.week.max())
    code = lines.StockCode.astype(str).str.strip().str.upper()
    merchandise = code.str.fullmatch(r"[0-9]{5}[A-Z]*")
    sale = (~lines.InvoiceNo.astype(str).str.upper().str.startswith("C")
            & (lines.Quantity > 0) & (lines.UnitPrice > 0))
    kept = lines.loc[full & merchandise & sale, ["week", "Quantity"]].copy()
    kept["article_id"] = code.loc[kept.index]
    weekly = kept.groupby(["article_id", "week"], as_index=False).Quantity.sum()
    weekly = weekly.rename(columns={"Quantity": "units"})
    assert weekly.units.sum() == kept.Quantity.sum()
    assert not weekly.duplicated(["article_id", "week"]).any()
    weeks = weekly.week.drop_duplicates().sort_values()
    # A few entire weeks can have no invoices. Preserve the complete calendar.
    first_week = lines.week.min() + pd.Timedelta(weeks=1)
    last_week = lines.week.max() - pd.Timedelta(weeks=1)
    calendar = pd.date_range(first_week, last_week, freq="7D")
    shops = []
    for name in ["A", "B", "C", "D", "test"]:
        published = pd.read_parquet(ROOT / "data" / "interim" / f"weekly_{name}.parquet",
                                    columns=["week"])
        other_weeks = pd.DatetimeIndex(published.week.unique())
        overlap = calendar.intersection(other_weeks)
        shops.append(dict(shop=name, first_week=str(other_weeks.min().date()),
                          last_week=str(other_weeks.max().date()), common_weeks=len(overlap)))
        if len(overlap) or calendar.max() >= other_weeks.min():
            raise ValueError(f"Donor date range requires manual overlap review for {name}")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    donor = ROOT / "data/external/uci_online_retail_weekly.parquet"
    donor.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(donor, index=False)
    meta = dict(source="Daqing Chen, UCI Online Retail, DOI 10.24432/C5BW33, CC BY 4.0",
                source_sha256=digest, python=platform.python_version(), pandas=pd.__version__,
                openpyxl=openpyxl.__version__, raw_lines=len(raw), exact_repeat_lines=int(repeated.sum()),
                first_transaction=str(dates.min().date()), last_transaction=str(dates.max().date()),
                retained_lines=len(kept), distinct_articles=weekly.article_id.nunique(),
                observed_product_weeks=len(weekly), units=int(weekly.units.sum()),
                first_full_week=str(first_week.date()), last_full_week=str(last_week.date()),
                calendar_weeks=len(calendar), weeks_with_merchandise_sales=len(weeks),
                empty_weeks=[str(d.date()) for d in calendar.difference(weeks)],
                weekly_sha256=hashlib.sha256(donor.read_bytes()).hexdigest())
    pd.DataFrame(shops).to_csv(out / "date_overlap.csv", index=False)
    (out / "source_audit.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
