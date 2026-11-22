from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import features as f
import train_gbm as gb
import validate as v

for shop in "ABCD":
    out = HERE / "results" / f"lgbm_{shop}.parquet"
    train, hold, cut = v.split(v.load_weekly(shop))
    assert train.week.max() < cut == hold.week.min()
    matrix = f.ShopMatrix(train)
    weeks = pd.date_range(cut, periods=13, freq="7D")
    windows = f.build_windows(matrix, f.make_origins(matrix.W))
    model, _ = gb.fit(windows, rounds=276)
    forecasts = gb.to_long(matrix, gb.forecast_matrix(model, matrix, weeks), weeks)
    keys = v.KEYS + ["week"]
    assert len(forecasts.merge(hold[keys], on=keys, validate="one_to_one")) == len(hold) == len(forecasts)
    assert np.isfinite(forecasts.forecast).all()
    tmp = out.with_suffix(".tmp")
    forecasts.to_parquet(tmp, index=False)
    tmp.replace(out)
    print(shop,cut,len(forecasts),flush=True)
