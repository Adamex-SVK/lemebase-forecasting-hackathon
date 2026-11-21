"""X1c screen: LoRA fine-tune Chronos-2 on real sibling shops (Sets A, C, D only), then forecast the test shop zero-shot.

    python src/foundation/x1_chronos2_ft.py --steps 300 --per-shop 2000 [--cuts 0 1 2 3]

Training data: weekly units of Sets A, C, D (never the test shop, never Set B which is synthetic), each series zero-padded from
its shop's first week (D26 convention), sampled at random (seed 0) among series with at least 8 sales weeks. Cheap screen, not a tuned fine-tune.
Output: iterations/03-foundation-model/forecasts_oos-ft/chronos-2-ft__test__cut{K}__padded.parquet
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "foundation"))
import validate as v  # noqa: E402
import run_model as rm  # noqa: E402

H = v.HORIZON


def donor_series(per_shop, seed):
    rng = np.random.default_rng(seed); out = []
    for shop in ["A", "C", "D"]:
        d = pd.read_parquet(ROOT / "data" / "interim" / f"weekly_{shop}.parquet")
        W = d.pivot_table(index=["store_id", "article_id"], columns="week", values="units", aggfunc="sum", fill_value=0.0).sort_index(axis=1)
        A = W.to_numpy(np.float32)
        ok = np.flatnonzero((A > 0).sum(axis=1) >= 8)
        pick = rng.choice(ok, size=min(per_shop, len(ok)), replace=False)
        out += [{"target": A[i]} for i in pick]
        print(f"donor {shop}: {len(ok)} usable series, {len(pick)} sampled, {A.shape[1]} weeks", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--per-shop", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--cuts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--final", action="store_true", help="forecast the real future weeks (2023-04-24..) from the full history"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--with-test", type=int, default=0, help="also train on the test shop own history before the cut, repeated N times (fine-tune is then done per cut)"); ap.add_argument("--tag", default="ft"); ap.add_argument("--base", default="amazon/chronos-2")
    a = ap.parse_args()
    from chronos import Chronos2Pipeline
    import torch
    out = ROOT / "iterations" / "03-foundation-model" / f"forecasts_oos-{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    donors = donor_series(a.per_shop, a.seed)
    cache = {}

    def tuned(extra):
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        t0 = time.time()
        pipe = Chronos2Pipeline.from_pretrained(a.base, device_map="cuda")
        ft = pipe.fit(inputs=donors + extra, prediction_length=H, finetune_mode="lora", learning_rate=a.lr,
                      num_steps=a.steps, batch_size=32, context_length=104)
        print(f"fine-tuned in {time.time() - t0:.0f}s ({len(extra)} test-shop series added)", flush=True)
        return ft

    for k in ([0] if a.final else a.cuts):
        index, ctx, active, hold, cut, train = rm.contexts_for("test", k, "padded", final=a.final)
        extra = [{"target": ctx[i].astype("float32")} for i in np.flatnonzero(active)] * a.with_test if a.with_test else []
        key = k if a.with_test else 0
        if key not in cache:
            cache = {key: tuned(extra)}                      # sibling-only: one fine-tune serves every cut
        ft = cache[key]
        weeks = pd.date_range(end=cut - pd.Timedelta(weeks=1), periods=len(ctx[0]), freq="W-MON")
        idx = np.flatnonzero(active)
        rows = [pd.DataFrame({"id": int(i), "timestamp": weeks, "target": ctx[i].astype("float32")}) for i in idx]
        pred = ft.predict_df(pd.concat(rows, ignore_index=True), prediction_length=H, quantile_levels=[0.5],
                             id_column="id", timestamp_column="timestamp", target="target").sort_values(["id", "timestamp"])
        F = np.zeros((len(index), H)); F[idx] = pred["0.5" if "0.5" in pred.columns else "predictions"].to_numpy().reshape(len(idx), H)
        fw = pd.date_range(cut, periods=H, freq="7D")
        name = f"chronos-2-{a.tag}__test__" + ("final" if a.final else f"cut{k}") + "__padded.parquet"
        pd.DataFrame({"store_id": np.repeat(index.get_level_values(0).to_numpy(), H), "article_id": np.repeat(index.get_level_values(1).to_numpy(), H),
                      "week": np.tile(fw, len(index)), "forecast": F.reshape(-1)}).to_parquet(out / name, index=False)
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
