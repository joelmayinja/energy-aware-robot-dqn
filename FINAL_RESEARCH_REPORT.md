# Research implementation report

**Title:** Optimising Energy Costs of Robots Using a Lightweight Deep Q-Network  
**Researcher:** Mayinja Joel, Uganda Christian University  
**Status date:** 26 November 2026

## 1. Research question and actual scope

This project studies whether a discrete Deep Q-Network can control a simulated KUKA iiwa arm to reach end-effector targets while reducing a simplified estimate of grid energy. The implemented task is two-phase target reaching. It does not include a workpiece, gripper, grasp, or pick-and-place operation.

The current energy calculation sums positive joint mechanical work per simulation step, `max(0, torque × angular_velocity × dt)`, and treats negative work as recoverable braking energy. That is a simulation proxy. It is not mains electricity, measured motor input power, an energy tariff, or a validated battery model. The original UCI steel dataset gives industrial demand context; it contains no robot control labels and is not used to train the DQN.

## 2. Architecture and implemented artifacts

The V1.0 environment is in `src/environment.py`. Its observation contains 23 values: six joint positions, six velocities, end-effector and target positions, phase, buffer level, last grid draw, distance, and episode fraction. The DQN action table contains 89 joint-velocity combinations. The teacher network is 23–256–256–89, with 94,809 trainable parameters. The student is 23–64–64–89, with 11,481 parameters (about 87.9% fewer parameters). Parameter reduction alone does not establish closed-loop competence or edge speed.

The original environment, dashboard, checkpoints, and historical results were retained. New experimental code is isolated under `experiments/v1_1/`. It adds a time-step-aware joule/watt ledger, explicit 80% charge and assumed 90% discharge efficiency, a magnitude-sensitive normalized peak penalty, Double-DQN target selection, Huber loss, gradient clipping, terminal timeout handling, trajectory-level teacher/student data splits, full DataLoader epochs, KL/CE/Q losses, early stopping, and bounded corrective teacher labels on student-visited states.

## 3. Independently verified historical evaluation

The preserved 24 November results report:

| Controller | Held-out completion | Mean energy on completed episodes |
|---|---:|---:|
| Fixed-speed IK | 45/50 (90%) | 250.5545 Ws |
| Selected IK-assisted DQN teacher | 44/50 (88%) | 274.0852 Ws |
| Distilled student | 6/50 (12%) | 242.4473 Ws over only six successes |

On 44 matched scenarios where both IK and teacher succeeded, the mean paired energy saving `(IK energy − teacher energy) / IK energy × 100` is **−10.82%**. The negative value means that the teacher used more simulated grid energy on average. Thirty-five paired cases were negative and nine positive; median saving was −8.96%. The paired success outcomes were 44 both-success, five both-fail, one IK-only, and zero DQN-only. Exact paired McNemar p=1.0. This sample does not show a completion advantage. The lower student energy mean is conditional on only six completions and cannot establish an efficiency advantage.

The historical online-training log records 3 successful episodes out of 500 (0.6%). This is distinct from the selected teacher's held-out 44/50 evaluation; they are separate policy phases and must not be merged into a single training-success claim. Source CSVs and historical checkpoints remain under `results/colab_run_2026-11-24/`.

## 4. V1.1 work and validation to date

The new energy ledger and network update passed three focused unit tests plus analytical buffer-balance tests after the implementation was added. A bounded local end-to-end smoke run collected 3 greedy teacher trajectories (60 transitions at a 20-step smoke horizon), trained a student for 2 complete epochs, evaluated student and fixed-IK baseline on two smoke episodes, ran one Double-DQN update, and round-tripped a full learner checkpoint with its replay buffer. These are plumbing checks, not research results.

The existing steel CSV was fully inspected: 35,040 rows, 11 columns, 2018 coverage, exact 15-minute spacing, zero missing cells, and zero duplicate rows. Average interval power is derived from recorded kWh divided by 0.25 hours. Demand charts and CSV summaries are in `experiments/v1_1/runs/2026-11-26/dataset_analysis/`.

The raw SDU collaborative-robot and Mendeley high-payload datasets were not downloaded in this session. Their official metadata pages were inspected; provenance and attempted access are documented. No electrical power regression or real-robot validation has been claimed.

## 5. Main limitations and next evidence required

1. Replace or calibrate mechanical-work estimates with measured electrical data before claiming actual electricity or cost savings.
2. Run the generated Colab notebook in deadline mode; retain its exact configuration, seed files, checkpoint, logs, package versions, per-episode results, and ZIP.
3. Use the fresh 50-seed test only after choices are fixed on separate validation seeds. Report failures, energy on failures, energy per successful completion, paired uncertainty, and completion uncertainty.
4. Compare the selected student in independent closed-loop control; never substitute teacher actions during student evaluation.
5. Verify ONNX numerically on real simulator observations. Raspberry Pi readiness requires an actual Pi benchmark.
6. Keep the target-reaching scope explicit; pick-and-place and physical industrial deployment are not implemented.

## 6. Reproducibility

The notebook is `FINAL_RESEARCH_TRAINING.ipynb`. The local V1.1 smoke command is `\.conda-training\python.exe -m experiments.v1_1.smoke_validate`; the energy dataset analysis command is `python experiments/v1_1/analyze_steel.py`. The Colab notebook embeds the needed source modules, the original teacher and student weights, the steel data, and historical summary files. Deadline and full mode are separately logged. The notebook creates `FINAL_RESEARCH_RESULTS.zip` only after its evaluation and export cells complete.

**Conclusion:** the historical prototype demonstrates target reaching, but does not demonstrate energy savings or a capable student. V1.1 adds a more explicit and testable research pipeline. Its performance remains unverified until a full Colab run completes.
