Cuts 2024-03-11 00:00:00 2025-07-28 00:00:00 training rows 466466 195000
{'arm': 'A only', 'score': 21.419128354166283, 'MASE': 0.7858087164583372, 'scored': 599, 'promotion_column_splits': 0}
{'arm': 'A+B without promotions', 'score': 21.54192126342277, 'MASE': 0.7845807873657723, 'scored': 599, 'promotion_column_splits': 109}
{'arm': 'A+B with promotions', 'score': 21.43996146537679, 'MASE': 0.7856003853462321, 'scored': 599, 'promotion_column_splits': 132}
Commands: .venv/bin/python iterations/30-b-promotions-to-a/run.py and slide.py. Seed 0. Versions: [('lightgbm', '4.7.0'), ('pandas', '2.3.3'), ('numpy', '2.5.5')]
The no-promotion column distinguishes B (constant zero) from A (missing), giving both arms access to donor identity. All A prediction rows have the promotion feature missing. Original A-only score reproduced. Review checkpoint reached.
