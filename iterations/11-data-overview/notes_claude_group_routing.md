# Per-group model routing (2026-11-21)
Groups by mean weekly units over the last 26 weeks of each cut (dead = no sale in 8 weeks, zeroed by all models).
1. Per-group scores of D40 / LightGBM / Moirai / TiRex / Chronos-T5-small / median13 (4 cuts): no model wins a group on
   all cuts; spreads are 0-3 points except fast (median13 and LightGBM clearly worse). Slow group (<0.5/wk) carries 34-43% of loss at about -110 for every model.
2. Hurdle median model (src/improve/hurdle_median.py, GPU machine, 20 threads, seed 0 only; seeds 1-2 stopped) vs D40:
   hurdle alone -14.93 vs -13.67 (worse on all 4 cuts). Routed to hurdle for one group, rest D40:
   slow -13.48 (+0.19, 3/4 cuts), low -13.91, mid -13.72, fast -14.82, slow+low -13.73, 50/50 blend -13.87.
   Nothing clears the bar (+0.5 mean, >=3 of 4 cuts). Not adopted.
