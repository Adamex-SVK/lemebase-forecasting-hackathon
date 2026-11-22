# Older-regime input forecasts

2026-11-22. The reserved target origin is 2021-10-25, offset 65. An isolated project GPU directory, dist32-stress-X4MTIZ, held a copy of the target parquet filtered by the predicate week earlier than 2021-10-25. It contained 65 weeks ending on 2021-10-18. The model runners used their --final entry point on that training-only table, yielding predictions through 2022-01-17. Each retained predictor sent 2,395 articles with some previous positive sales to inference. All 2,904 article keys appear in each output. Only historical donor parquets for A, C and D were linked into the isolated directory. No target holdout outcomes or scores were needed.

The SSH connector used the project account. Automatic approval review declined copying local project source to the server, citing the sensitivity of the payload and unverified external destination. The completed alternative used only source and data already installed on the server. A preliminary pass used the shared installation. Its first three outputs were superseded by canonical reruns using the archived PR 20 copy already on that server, with NumPy and torch seed zero. Verified against Git ca3a0ed, models.py SHA-256 2ca6f0d62bd11e6c4d13eea5543b2719168c180cac79ff7c2e6d1aac0a51b863 and run_model.py 97cb9f74fe5dd62454868c0abb50f80a21a9b1e3ce94fe34e57a57d6421be04a. Historical Moirai, TiRex and Chronos-T5 package environments, CUDA, HF_HUB_OFFLINE=1. Each used run_model.py MODEL test --final. The sibling Chronos-2 routine used x1_chronos2_ft.py --final --steps 300 --seed 0. Its code SHA-256 009a91c5e9438fabd14f944a921ebf1b1000e83203f546d869a838f49bd625c2 and src/validate.py 0bc5f162e09324c45c03586b6cfb1d6f1c42fafe367dd6875f64071c6eb54921 both match PR 20. The sibling arm used the original project training/inference installation as in iteration 31.

Validation uses unique store, article and Monday keys, complete dates, 37,752 finite predictions per component and exactly the 2,904 article keys from pre-origin history. Foundation forecasts are shared identically by the paired candidate and reference.

All four files passed these checks. Chronos-2 completed 300 steps in 102 seconds including setup.

    chronos-2-ft__test__w65__padded.parquet  3f4305698faeb20c397f4c4663858df5874c82ea9712c4e38d7b573ba6ac06a4
    chronos-t5-small__test__w65__padded.parquet  1ef5183bc396d2a16937bf64684c8cdc0c02aff507a4a59ecdffc4c59783796f
    moirai-2.0-small__test__w65__padded.parquet  39c12a2dc3389523b95fed1dec3a057625a2b20fb1208374acc287ae089f898e
    tirex__test__w65__padded.parquet  fcd1eb8aa6052274b3cb24063a27c04eeb27d84ea64ea45bf322dc3468786491
