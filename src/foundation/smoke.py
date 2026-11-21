"""Smoke test of the model adapters on 3 toy weekly series. Run this FIRST on the GPU machine,
inside the environment of the model family:

    python src/foundation/smoke.py chronos-bolt-base chronos-2 ...

For every model it prints the forecast shape and values, or the error (so we see which adapter
needs a fix). Nothing from this file touches the project data."""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as fm  # noqa: E402

rng = np.random.default_rng(0)
ctx = [rng.poisson(3, 80).astype(float), np.r_[np.zeros(60), rng.poisson(1, 20)].astype(float), np.zeros(80)]
end = pd.Timestamp("2025-12-29")
device = "cuda"
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
except Exception:
    device = "cpu"
print("device:", device)
for name in sys.argv[1:] or ["dummy-median13"]:
    try:
        h = fm.load(name, device)
        F = fm.forecast(name, h, ctx, 13, end)
        assert F.shape == (3, 13) and np.isfinite(F).all()
        print(f"OK   {name:20s} shape {F.shape}  row0 {np.round(F[0, :4], 2)}  row1 {np.round(F[1, :4], 2)}")
    except Exception:
        print(f"FAIL {name}")
        traceback.print_exc(limit=3)
