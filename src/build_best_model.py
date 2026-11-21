"""Build the FINAL model end to end and check it against the recorded submission (D40 + sibling-shop fine-tuned Chronos-2).

    .venv/bin/python src/build_best_model.py            # rebuild LightGBM + ensemble, verify, validate
    .venv/bin/python src/build_best_model.py --skip-lgbm  # reuse submissions/lgbm_own_v1.csv (about 1 minute faster)

The recipe (D40, see MODEL.md): 0.3 x LightGBM (own arm, all history) + 0.7 x mean of the zero-shot forecasts of
moirai-2.0-small, tirex and chronos-t5-small; 0 where an article has no sale in the last 8 weeks; round; clip at 0.
Inputs: data/interim/weekly_test.parquet (run src/prepare.py) and the committed final forecasts in
iterations/03-foundation-model/forecasts_final/. Output: submissions/ensemble_d40_plus_chronos2ft_v1.csv (live -4.5). D40 alone: ensemble_top3_blend_v1.csv (live -4.8).
Uploading is done by a person, never by this script.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
TARGET = SUB / "ensemble_d40_plus_chronos2ft_v1.csv"
EXPECTED_SHA256 = "424efba0fc50e3ce49869f7f4129d0e1752a608088b2dd0931c569ce1f0aa936"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(script):
    print(f"-> {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)


def main():
    if "--skip-lgbm" not in sys.argv:
        run("make_lgbm_final.py")             # checks itself against submissions/lgbm_own_v1.csv (max difference 0)
    run("make_submission_foundation.py")       # D40 files (the foundation part of the final model)
    run("make_submission_ft.py")               # final file: D40 + the fine-tuned Chronos-2 forecast (committed, GPU not needed)
    got = sha(TARGET)
    ok = got == EXPECTED_SHA256
    print(f"\n{TARGET.name} sha256 {got[:16]}... {'MATCHES the recorded live-scored file' if ok else 'DIFFERS from the recorded file'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
