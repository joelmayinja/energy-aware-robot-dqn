# Internal Research Defence — V1.1

Prepared 26 November 2026 for the 28 November internal defence. This folder consolidates the recovered outer Colab deadline run, its independently recalculated results, a reproducible local PyBullet demonstration, and defence notes.

## Start here

1. Read [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md), then [VERIFIED_RESULTS.md](VERIFIED_RESULTS.md) and [LIMITATIONS.md](LIMITATIONS.md).
2. Follow [DEMONSTRATION_GUIDE.md](DEMONSTRATION_GUIDE.md). The default demo seed is 140000; try the same seed for each controller.
3. Use [DEFENCE_QUESTIONS_AND_ANSWERS.md](DEFENCE_QUESTIONS_AND_ANSWERS.md) to practise explaining the method and the negative energy result.

## Verified experiment state

The recovered archive was preserved unchanged at the repository root. Its 46 outer-run files and 15 nested partial-run files were extracted separately under `results/internal_defence_v1_1_2026-11-26/`. The manifest records the source archive hash and per-file hashes. Only the outer final episode table is used for the 50-seed results in this folder.

The independently verified held-out results are IK 44/50 (88%), retained teacher 46/50 (92%), and selected student 45/50 (90%). The student retains 97.83% of teacher completion, but **both learned policies used more simulated energy than fixed IK**. The internal evaluation archive provides no successful episodes for the fine-tuned teacher, so its success-conditional energy saving is unavailable.

## Important boundaries

- This is a PyBullet target-reaching simulation, not physical grasp-and-place.
- Energy is estimated from simulated joint mechanical work with assumed buffer efficiencies. It is not measured mains electricity, tariff cost, or proof of industrial energy savings.
- The industrial steel dataset is demand context only. No additional robot measurement archive was downloaded or used.
- ONNX export and PyTorch parity were verified at batch sizes 1, 2 and 8. Raspberry Pi deployment has not been tested.
- A one-seed demo is illustrative and must not be described as the 50-scenario evaluation.

## Main files

- `corrected_policy_summary.csv`, `corrected_paired_comparisons.csv`, `independent_verification.json`
- `figures/` — corrected, publication-style graphs
- `models/` — separate defence copies of the selected teacher/student and original student
- `model_provenance.json` — source file/checkpoint hashes and ONNX validation details
- `demo_runs.csv` and `demo_headless_verification.csv` — one selected live run and four headless demonstrations, each clearly separate from evaluation
- `../defence_demo.py` — live GUI/headless demonstration entry point
- `../INTERNAL_DEFENCE_V1_1.zip` — packaged copy of the essential defence artifacts
