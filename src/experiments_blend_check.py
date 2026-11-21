"""Follow-up to 2.3/2.4: lgbm+median13+gate (no CatBoost) and paired per-article intervals against the lgbm reference.
Uses saved forecasts (seed-averaged). Run: .venv/bin/python src/experiments_blend_check.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_02 as e, experiments_loss as el, validate as v  # noqa: E401
FC, H = el.FC, 13
full = v.load_weekly("test")
rng = np.random.default_rng(0)
res, paired = {}, {}
for k in range(4):
    P = el.prep(full, k); M = P["M"]
    Fl = np.mean([np.load(FC / f"loss_F_cut{k}_seed{s}.npy") for s in range(3)], 0)
    Pr = np.mean([np.load(FC / f"loss_P_cut{k}_seed{s}.npy") for s in range(3)], 0)
    Fc = np.mean([np.load(FC / f"cb_F_cut{k}_seed{s}.npy") for s in range(3)], 0)
    med = np.repeat(np.median(M.U[:, -13:], axis=1, keepdims=True), H, axis=1).astype(float)
    gate = lambda F: np.where(Pr < 0.5, 0.0, F)
    V = {"lgbm": Fl, "median13": med, "lgbm+gate": gate(Fl), "lgbm+median13 (1/2)": (Fl + med) / 2,
         "lgbm+median13+gate (1/2)": gate((Fl + med) / 2), "lgbm+cb+median13+gate (1/3)": gate((Fl + Fc + med) / 3)}
    for n, F in V.items():
        r = e.score_F(M, F, P["train"], P["hold"], P["weeks"], ) if False else e.score_F(M, F, P["train"], P["hold"], P["weeks"])
        res.setdefault(n, []).append(r["score"]); paired[(k, n)] = r["detail"]["mase"]
pt = pd.DataFrame(res, index=range(4)).T; pt["mean"] = pt.mean(axis=1)
print(pt.sort_values("mean", ascending=False).round(2).to_string())
print("\nPaired over articles vs lgbm (score points, 95% bootstrap):")
for n in [x for x in res if x not in ("lgbm", "median13")]:
    out = []
    for k in range(4):
        d = (paired[(k, n)] - paired[(k, "lgbm")]).dropna().to_numpy()
        bs = [-100 * rng.choice(d, len(d)).mean() for _ in range(500)]
        out.append(f"{-100*d.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    print(f"  {n:28s} " + "  ".join(out))
