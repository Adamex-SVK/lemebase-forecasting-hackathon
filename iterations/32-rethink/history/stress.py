"""Fit frozen median trees using pre-origin rows for the older stress period."""
from pathlib import Path
import importlib.util
import pandas as pd

import train

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "31-demand-distribution"
spec = importlib.util.spec_from_file_location("iteration31_tree", BASE / "distribution.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
output = HERE / "stress"
output.mkdir(exist_ok=True)
matrix, cut = train.history("test", 65)
assert str(cut.date()) == '2021-10-25'
assert pd.Timestamp(matrix.weeks[-1]) < cut
for name, dist in [('incumbent', old), ('history', train.dist)]:
    model = dist.fit([dist.examples(matrix)])
    median = dist.medians(model, matrix, dist.support([matrix]))
    weeks = pd.date_range(cut, periods=13, freq="7D")
    frame = train.gb.to_long(matrix, median, weeks)
    assert len(frame) == 2904 * 13 and not frame.duplicated(['store_id','article_id','week']).any()
    path = output / f'{name}.parquet'
    frame.to_parquet(path,index=False)
    assert pd.read_parquet(path).equals(frame)
    print(name, len(frame), str(cut.date()), flush=True)
