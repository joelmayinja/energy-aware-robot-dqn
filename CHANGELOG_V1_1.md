# Version 1.1 development log

## Internal defence integration — 26 November 2026 (supersedes earlier pending notes)

- Preserved `RECOVERED_RESEARCH_RESULTS.zip` unchanged and recorded its SHA256; extracted outer deadline and nested partial outputs into separate versioned folders with per-file hashes.
- Independently recalculated the 450 outer episode rows. Corrected paired-energy comparisons use the actual `fixed_original` policy ID; the archived paired report's zero-pair output is retained but not used as the corrected result.
- Added Wilson completion intervals, paired-success bootstrap intervals, exact McNemar outcomes, failed-task energy, and N/A handling for zero-completion energy/task metrics. Generated separate corrected tables and figures in `DEFENCE_2026/`.
- Fixed teacher fine-tuning metadata writing (removed undefined student-only variables and wrong marker filename), added stale-config/data checks for resumed student/teacher training, stable notebook output path, JSON-safe action serialization, matched-seed cache validation, correct fixed-IK pairing, N/A aggregate metrics, and dynamic-batch ONNX export.
- Added `defence_demo.py` and separate teacher/student PyTorch and ONNX exports. ONNX checker and numerical/action parity passed for batch sizes 1, 2 and 8 on actual PyBullet demonstration observations.
- Added V1.1 regression tests; all 38 repository tests passed, and the bounded local smoke experiment passed. Four complete headless demo episodes on seed 140000 passed for IK, retained teacher, and selected student; the original student failed that illustrative seed. The live PyBullet GUI path also completed the selected-student seed 140000 demonstration. These are demonstration checks, not additional evaluation results.
- Built internal defence notes and `INTERNAL_DEFENCE_V1_1.zip`. No full retraining, commit, push, dashboard redesign, or modification of historical V1.0 result files was performed.

## 2026-11-26

- Recorded repository starting commit `e69a3d57855f95e252c2e8bac22ecac56008c5cc` and current working tree before V1.1 edits. Existing user changes were already present and have been preserved.
- Created branch `research/v1.1`; did not commit or push.
- Added an isolated environment wrapper, preserving V1.0 files and task geometry. The wrapper reports interval energy in joules, derives power from the engine timestep, applies a magnitude-sensitive peak penalty, and models discharge efficiency as an explicit 0.90 assumption. This assumption is not measured and needs sensitivity analysis.
- Added a V1.1 teacher update using Double DQN target selection, Huber loss, gradient clipping, and finite-horizon terminal handling. It is experimental, separate from the original agent.
- Added teacher-Q trajectory collection, trajectory-disjoint validation, complete DataLoader epochs, KL/CE/Q regression, early stopping, and greedy closed-loop evaluation utilities.
- Inspected `notebooks/steel_data.csv`; generated descriptive CSV/JSON/PNG outputs under the dated experiment folder. Analysis does not supply robot training labels.
- Inspected official metadata for SDU and Mendeley robot datasets; raw downloads were not confirmed accessible in this session.
- Added the first local V1.1 regression tests. Results are pending until executed in the local training environment.
- AI coding assistance was used to draft V1.1 code and documentation. The student must review, understand, test, and accurately disclose this assistance under course policy.

## Pre-recovery pending note (superseded on 26 November 2026)

This was the recorded state before the user supplied `RECOVERED_RESEARCH_RESULTS.zip`. The notebook, local smoke/tests, student distillation/evaluation, bounded teacher fine-tuning, baseline comparisons, ONNX validation, and defence package are now completed as described in the internal-defence integration entry above. No energy improvement is claimed.

## Validation update — 26 November 2026

- Five V1.1 unit tests passed, covering energy-ledger efficiency/capacity, one PyBullet step, the teacher checkpoint architecture, and a Double-DQN update.
- The local smoke pipeline passed: three teacher trajectories, 60 transitions, two student epochs, two student closed-loop episodes, two fixed-IK baseline episodes, one learner update, and checkpoint/replay restoration. Output is explicitly marked smoke.
- The existing repository suite passed: 35 tests total, including the five new tests.
- The generated notebook has valid nbformat JSON; both code cells parse; its embedded zip CRC and required source/model/data files were checked.
- The existing steel dataset analysis ran and generated outputs. Its source CSV hash is recorded in `DATASET_MANIFEST.json`.
- Direct public retrieval attempts were made. IEEE DataPort lists a 101.47 MB dataset ZIP behind a login gate; its README request returned HTTP 403. Mendeley’s public file list exposed a 5.5 KB README, which was downloaded and hashed. Its Download All archive is 2.709 GB in 26 split volumes; those bulk measurement files were not fetched. No access gate was bypassed.
- A local pre-Colab package is generated with a status file that says full research execution remains pending.

## Remaining research work

For April research, run multiple independent training seeds, improve the energy model with measured robot electrical data, establish successful adaptive-IK baselines, and test physical/edge deployment. The 90% teacher completion and positive-energy-saving targets remain distinct objectives; this run did not meet the energy objective.
