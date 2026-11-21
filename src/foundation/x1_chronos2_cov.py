"""X1b: Chronos-2 with the test shop's stock columns as past covariates (on_hand, out-of-stock flag).

    python src/foundation/x1_chronos2_cov.py chronos-2 --cut-index K [--cov none|stock]

Same contexts as run_model.py (padded, cut K). Covariates are known only for the past (no future covariates), and only from
weeks before the cut. Output: iterations/03-foundation-model/forecasts_oos-cov-<cov>/{model}__test__cut{K}__padded.parquet
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "foundation"))
import validate as v  # noqa: E402
import models as fm  # noqa: E402
import run_model as rm  # noqa: E402

H = v.HORIZON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--cut-index", type=int, default=0); ap.add_argument("--cov", default="stock", choices=["none", "stock"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    index, ctx, active, hold, cut, train = rm.contexts_for("test", a.cut_index, "padded", final=a.final)
    full = v.load_weekly("test")
    hist = full[full["week"] < cut]
    weeks = pd.date_range(end=cut - pd.Timedelta(weeks=1), periods=len(ctx[0]), freq="W-MON")
    def piv(col):
        return hist.pivot_table(index=v.KEYS, columns="week", values=col, aggfunc="max").reindex(index=index, columns=weeks)
    oh = piv("on_hand").clip(lower=0).to_numpy(float)
    oos = piv("out_of_stock_flag").fillna(0).astype(float).to_numpy()
    pipe = fm.load(a.model, a.device)
    idx = np.flatnonzero(active)
    rows = []
    for k in idx:
        d = {"id": int(k), "timestamp": weeks, "target": ctx[k].astype("float32")}
        if a.cov == "stock":
            d["log_on_hand"] = np.log1p(oh[k]).astype("float32"); d["out_of_stock"] = oos[k].astype("float32")
        rows.append(pd.DataFrame(d))
    t0 = time.time()
    pred = pipe.predict_df(pd.concat(rows, ignore_index=True), prediction_length=H, quantile_levels=[0.5],
                           id_column="id", timestamp_column="timestamp", target="target").sort_values(["id", "timestamp"])
    F = np.zeros((len(index), H)); F[idx] = pred["0.5" if "0.5" in pred.columns else "predictions"].to_numpy().reshape(len(idx), H)
    fw = pd.date_range(cut, periods=H, freq="7D")
    fc = pd.DataFrame({"store_id": np.repeat(index.get_level_values(0).to_numpy(), H), "article_id": np.repeat(index.get_level_values(1).to_numpy(), H),
                       "week": np.tile(fw, len(index)), "forecast": F.reshape(-1)})
    out = ROOT / "iterations" / "03-foundation-model" / f"forecasts_oos-cov-{a.cov}"; out.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out / f"{a.model}__test__{'final' if a.final else 'cut' + str(a.cut_index)}__padded.parquet", index=False)
    print(f"{a.model} cut{a.cut_index} cov={a.cov}: {len(idx)} series, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
