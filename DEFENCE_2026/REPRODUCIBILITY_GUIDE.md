# Reproducibility and provenance

## Recovered experiment

Original input: repository-root `RECOVERED_RESEARCH_RESULTS.zip`, unchanged. Its SHA256 is `1aa45792f1a0704f312c81153a9e60ff051d5bea87873df717646125c54953c8`. Archive CRC check passed. `results/internal_defence_v1_1_2026-11-26/RECOVERED_RESULTS_MANIFEST.json` lists every extracted filename, byte size and SHA256. `outer_deadline/` holds final episode results; `nested_partial_deadline/` holds nested partial-stage files. Never combine them.

The final held-out table is `outer_deadline/final_episode_metrics.csv`, with 450 rows across 9 policy IDs and the same seeds 140000–140049. The corrected local analysis is reproducible with:

```powershell
python -m experiments.v1_1.reconstruct_defence_results
```

The run metadata and dependency versions are in `outer_deadline/experiment_config.json`; seed/config records are in `outer_deadline/seed_manifest.csv`. Colab device was CUDA on x86_64. V1.1 task and energy assumptions are recorded there.

## Local live demonstration

Use the project's `.conda-training` Python. Model files in `DEFENCE_2026/models` are separate state-dict exports. Their source hashes and provenance are in `models/model_provenance.json`. `defence_demo.py` accepts a controller and scenario seed; same seed recreates the same initial joints and targets. The one-episode result is a demonstration, not a 50-scenario evaluation.

## Data sources

The existing steel CSV is `notebooks/steel_data.csv`; it was analyzed only for demand patterns. SDU collaborative-robot data were not downloaded because file access required login. The Mendeley README is in `results/internal_defence_v1_1_2026-11-26/outer_deadline/dataset_documentation/readme.txt`; the large measurement archive was not downloaded.

## Model export

PyTorch and dynamic-batch ONNX files are preserved in `DEFENCE_2026/models`. ONNX checker and ONNX Runtime parity were tested at batch sizes 1, 2 and 8 using recovered PyBullet demonstration observations. Maximum absolute output difference was 7.63e-6 for each model; greedy actions agreed on all test observations. The declared tolerance is 5e-5. Source/model hashes are in `models/model_provenance.json`. This does not test Raspberry Pi performance.

## Demonstration check

`demo_headless_verification.csv` records one full 200-step-maximum headless episode for each controller on seed 140000. The original and improved students behave differently: the original student did not finish that scenario; the selected student did. This is a single illustrative scenario and is kept separate from the 50-seed evaluation.
