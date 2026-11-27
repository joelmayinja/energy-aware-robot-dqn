# Project Audit: IK-Assisted Energy-Aware Robot Control

**Prepared:** 26 November 2026  
**Repository audited:** `C:\Users\JOEL\Desktop\DQN\Energy_consumption_DQN`  
**Scope:** repository structure, Python source, Colab notebook, test definitions, configuration, project documentation, saved CSVs and model artifacts. No implementation or experiment output was changed. The only file created for this request is this audit.

## Executive assessment

This is a working **simulation research prototype** with a PyBullet KUKA iiwa model, IK demonstrations, a DQN policy, an IK baseline, same-seed evaluation files, a Flask dashboard, and saved teacher/student artifacts. It has demonstrated target reaching for one selected IK-assisted teacher checkpoint in held-out simulation. It has **not demonstrated energy saving, real energy-cost reduction, physical-robot operation, actual pick-and-place, or edge readiness**.

The main reported results check out against the files in `results/colab_run_2026-11-24/`:

| Evidence | Independently checked result |
|---|---:|
| Held-out IK-assisted DQN | 44/50 = 88% |
| Held-out fixed-speed IK baseline | 45/50 = 90% |
| Shared successful seeds | 44/50 |
| Mean per-seed energy saving on shared successes | -10.8227% (rounds to -10.82%) |
| Median per-seed energy saving | -8.955% |
| Per-seed savings below zero / above zero | 35 / 9 |
| Mean / median peak reduction on shared successes | 0.22% / 3.24% |
| Online teacher training successes | 3/500 = 0.6% |
| Student held-out completion | 6/50 = 12% |

The negative energy result means the teacher used **more** energy on average than IK on the matched completed targets. The online `3/500` and held-out `44/50` are from different policy phases and must remain separately labelled. The teacher and IK completion difference is one seed only: 44 both completed, 5 both failed, 1 IK-only success, and 0 DQN-only successes. A two-sided exact McNemar comparison of these paired binary outcomes gives p=1.0; this sample does not establish a completion-rate advantage. Individual Wilson 95% intervals are approximately 76.2-94.4% for DQN and 78.6-95.7% for IK. The paired outcome is the more appropriate comparison here.

The principal scientific weakness is not that the results are “zero progress”; it is that the evidence currently supports a narrower conclusion than the project's energy-optimisation claim. A defensible conclusion is: **the selected IK-assisted DQN reaches targets in this simulation at about the IK baseline rate, but the energy objective has not been achieved.**

## 1. Repository structure and architecture

| Area | Files and role | Status |
|---|---|---|
| Simulation | `src/environment.py`: Gymnasium/PyBullet environment; loads plane and KUKA iiwa URDF, target phases, energy bookkeeping. | Implemented simulation. |
| Agent | `src/dqn_teacher.py`: Q-network, discrete action table, replay buffer, DQN update, weights and training checkpoints. | Implemented. |
| Training/evaluation | `src/training.py`: IK demonstration collection, behaviour cloning, local fine-tuning, local baseline and paired evaluation, CSV logging. `train_colab.ipynb`: Colab training, distillation, evaluation and export path. | Implemented in two related but not identical runners. |
| Web app | `src/backend.py`, `src/launcher.py`, `src/dashboard.py`, `frontend/login.html`, `frontend/web.html`. Flask serves a presentation dashboard and local training API. | Implemented demonstration interface; not an authentication product or robot controller. |
| Data | `notebooks/steel_data.csv`, `notebooks/dataset_prototype.ipynb`; `data/raw/` and `data/processed/`. | Steel dataset is exploratory context, not DQN input. |
| Results/artifacts | `results/colab_run_2026-11-24/`: summary, paired and matched-seed CSVs, training history, distillation losses, plot, and model files. `models/`: current and older checkpoints. | Useful evidence exists; notebook execution provenance is incomplete. |
| Tests/config/docs | `tests/`, `requirements.txt`, `environment-training.yml`, README and PDFs in `docs/`. | Tests are defined; this audit did not run them. |
| Hardware-facing concept | `tinkercad/` Arduino sketch and circuit illustration. | Threshold/rule mock-up only, not the trained DQN. |

The web and ML architecture are reasonably separated: `src/dashboard.py` re-exports the same Flask app from `src/backend.py` (`src/dashboard.py:1-5`); the launcher prepares/selects Python and starts that app (`src/launcher.py:12-23,66-120,143-166`); the API starts `train()` on a background thread (`src/backend.py:31-40,81-100,165-183`). The frontend login is explicitly a demo gate: the project README says it stores a display name and does not authenticate users or protect the API (`README.md:123,153-159`).

The runtime config `src/config.py:1-5` only sets local host and port. Important experimental values are hard-coded across `src/environment.py`, `src/training.py`, and notebook cells rather than captured in a versioned experiment manifest. Dependency files use minimum-version ranges (`requirements.txt:1-19`; `environment-training.yml:1-10`) rather than an exact lock, so a future install is not byte-for-byte reproducible. The Conda file also omits the ONNX packages that the Colab notebook installs separately (`train_colab.ipynb:47-48`).

## 2. DQN implementation and architecture

### Implemented

- The teacher Q-network is a fully connected MLP: 23 inputs, two 256-unit ReLU hidden layers, 89 Q-values (`src/dqn_teacher.py:66-80`).
- The discrete action table has 89 entries: one all-zero action, 24 one-joint commands (six joints times four non-zero velocity levels), and 64 coordinated all-joint +/-1 directions (`src/dqn_teacher.py:38-63`).
- Replay is a 10,000-entry deque; transitions are sampled uniformly (`src/dqn_teacher.py:83-104`).
- The agent uses epsilon-greedy selection, Adam, gamma 0.99 by default, batch size 64 in training, and a target network (`src/dqn_teacher.py:113-147,149-200`; `src/training.py:420-424,518,533-539`).
- The DQN target is the standard max-target form `r + gamma * max_a Q_target(s',a) * (1-done)` and uses MSE (`src/dqn_teacher.py:167-200`).
- Local full checkpoints include online and target weights, optimizer, replay, RNG states, gamma and metadata (`src/dqn_teacher.py:211-280`).

### Technical criticisms

1. **This is an IK-assisted policy, not autonomous DQN learning from scratch.** The notebook collects 100 IK demonstration episodes, behavior-clones for 3,000 optimizer updates, then fine-tunes while retaining an imitation loss (`train_colab.ipynb:122-167,147-157,184-185`). Use “IK-assisted DQN” consistently.
2. **The action discretization is coarse and structurally biased.** It permits a half-speed command on one joint at a time, or a full-speed +/-1 command on all six joints, but not arbitrary mixed multi-joint velocities. The table construction is at `src/dqn_teacher.py:44-63`. This can limit how closely it imitates smooth IK motion.
3. **The algorithm is basic DQN.** It uses `max Q_target` rather than Double DQN, MSE rather than a robust Huber loss, and has no prioritized replay, gradient clipping, or state normalization (`src/dqn_teacher.py:167-200`). These are possible stability/overestimation concerns, not proof of the observed failure cause.
4. **Truncation is not stored as terminal in the Colab and local DQN transitions.** Both loops store `terminated` only (`train_colab.ipynb:181-185`; `src/training.py:524-532`), while the environment can set `truncated=True` at 200 steps (`src/environment.py:293-294`). The learner then bootstraps from time-limit states. For a finite-horizon task this can bias values unless the horizon is deliberately treated as a continuing process; the observation does include step fraction (`src/environment.py:327-339`).
5. **The saved evidence represents one training run and one selected checkpoint.** The Colab notebook does not set global Python/NumPy/PyTorch seeds in the visible training cell; resetting Gym environments by episode seed does not seed network initialization, replay sampling and all action-selection RNG. The notebook also stores no execution outputs (`train_colab.ipynb:14-16,61-63,112-114`, and later cells similarly have null execution counts/empty outputs). The saved results are checkable, but the exact run state is not fully reconstructible from the notebook file alone.

## 3. Robot environment, observations, actions, and task definition

`RobotArmEnv` loads a fixed-base KUKA iiwa URDF and plane (`src/environment.py:140-151`). PyBullet's model has seven physical joints; the final wrist-roll joint is locked and six are controlled (`src/environment.py:148-151,162-167`). The 23 observations are six positions, six velocities, end-effector XYZ, target XYZ, phase, normalized buffer, normalized last grid draw, distance, and step fraction (`src/environment.py:9-18,120-137,315-339`).

There is an apparent documentation difference, but not a runtime contradiction: the Gymnasium environment exposes a continuous six-value `Box` (`src/environment.py:135-137`), while `DQNAgent` maps 89 discrete choices to six-component velocity vectors (`src/dqn_teacher.py:44-63`). PyBullet receives those vectors after clipping/scaling (`src/environment.py:197-220`). Describe the agent's 89 discrete policy actions and the environment's continuous control interface separately.

Targets are XYZ points sampled from two coordinate ranges; at step 50 the target changes from the light to heavy target (`src/environment.py:169-206`). Success is only reaching the heavy target within 0.15 m before the 200-step limit (`src/environment.py:67-70,281-294`). This is **end-effector target reaching**, not an object manipulation task.

## 4. Reward and energy calculation

The environment uses `ALPHA=0.002`, `PEAK_PEN=0.30`, `BUFFER_REW=0.10`, `BUFFER_USE=0.20`, `BETA=20`, and `COMPLETION_BONUS=100` (`src/environment.py:76-85`). Its source computes per-joint work as applied joint torque times joint velocity times the 1/240-second step, sums positive work as raw draw, and sums negative work as regenerative energy (`src/environment.py:230-238`). Buffer charging is capped at 50 Ws and applies 80% charging efficiency; in the heavy phase stored energy is subtracted from raw work without a discharge-efficiency loss (`src/environment.py:240-264`).

**Excerpt from the actual reward implementation** (`src/environment.py:266-291`):

```python
peak_spike = max(0.0, grid_draw - self.PEAK_THRESHOLD)
peak_penalty = float(grid_draw > self.PEAK_THRESHOLD)
reward = (
    - self.ALPHA * grid_draw
    - self.PEAK_PEN * peak_penalty
    + self.BUFFER_REW * buffer_saved
    + self.BUFFER_USE * buffer_deployed
    + self.BETA * (distance_before - distance)
)
if task_done:
    reward += self.COMPLETION_BONUS
```

Criticisms:

- The module's reward comment says penalize `peak_spike` (`src/environment.py:24-32`), but the actual reward penalizes a **boolean threshold crossing** (`peak_penalty`) with a fixed 0.30, regardless of how far above the threshold the draw is (`src/environment.py:266-288`). `peak_spike` is emitted for logging but not used in the reward. This is a code/documentation mismatch and weak shaping for peak shaving.
- The `3.0` threshold and “peak” are per-step energy quantities, not calibrated instantaneous electrical power. Each step is 1/240 s (`src/environment.py:231`); dividing per-step joules by dt would be needed for watts. The report generally labels the metric Ws. Avoid saying measured peak power or utility-grid peak demand.
- The model infers mechanical work from simulated torque/velocity and assumes all negative mechanical work is recoverable after an 80% charge loss. It does not model a specific motor drive, DC bus, inverter, controller losses, speed-dependent regeneration limits, battery state-of-charge constraints, thermal effects, or measured hardware. The README itself calls the model simplified and unvalidated (`README.md:77`; `src/environment.py:1-7`).
- No electricity tariff, time-of-use schedule, Ugandan shilling cost, demand charge, or cost objective appears in the environment. It optimizes a reward with simulated energy proxies, not observed energy **cost**. “Energy-cost reduction” is currently an intended application, not a result.
- Charging efficiency is applied while charging, but buffer energy is deployed one-for-one (`src/environment.py:250-264`). This makes storage round-trip loss incomplete and can overstate delivered energy.
- Reward weights mix energy, distance progress, buffer incentives and completion without a documented scale analysis or ablation. With a weak energy coefficient and a large progress/completion objective, task completion may dominate, but the files do not prove this caused 3/500 success.

## 5. IK baseline and comparison fairness

The baseline asks PyBullet for inverse-kinematics joint angles on every step, then sends the sign of each position error as a joint-velocity command (`src/training.py:51-63,191-216`). The action is therefore approximately fixed-magnitude joint velocity in each nonzero joint; `MAX_JOINT_VELOCITY=2 rad/s` is applied by the shared environment (`src/environment.py:67-68,213-220`). In Colab, baseline and teacher are evaluated on the same reset seed, and both use the same environment energy-buffer setting (`train_colab.ipynb:192-220`). Final evaluation likewise uses shared seeds and only marks energy comparison valid when both succeed (`src/training.py:219-273`).

This is a **reasonable and fairly strong operational baseline** for this narrow target-reaching task: same seeds, same simulator, same success threshold and energy bookkeeping. It is also the expert that trains the teacher, so the comparison measures whether the learned approximation can match an IK controller, not whether DQN discovers a policy independent of IK. The baseline has direct access to PyBullet IK at every step, while the DQN chooses among a coarse action set. That is acceptable if stated as the research question, but it is not an apples-to-apples algorithm comparison of equally informed controllers.

Energy is correctly restricted to the 44 matched successes for the paired headline. The summary's mean energy among all teacher successes (274.0852 Ws) and all baseline successes (250.5545 Ws) is **not paired** because those success sets differ. Do not compute the headline energy claim by subtracting those two summary means. The paired CSV formula is baseline-minus-teacher over baseline (`src/training.py:242-273`).

## 6. Training methodology and checkpoint selection

The current local runner seeds Python, NumPy and Torch with 42 (`src/training.py:415-417`), collects 100 IK demonstrations (`src/training.py:103-142,475-480`), runs 3,000 behavior-cloning updates (`src/training.py:145-165,489-500`), and then runs DQN episodes with epsilon `max(0.01, 0.10 * 0.995**(episode-1)` and target-network copying every five episodes (`src/training.py:518,638-639`). The Colab notebook uses the same demonstration/fine-tuning structure and 500 online episodes (`train_colab.ipynb:117-169,179-185,243`). Epsilon starts at 0.10 and bottoms at 0.01; it is not 0.99.

Checkpoint selection compares the warm-start and fine-tuned policy on 20 seeds starting at 70,000, prioritizing success count, then mean energy among successful episodes (`src/training.py:168-188,660-690`; notebook selection at lines 251-274). Final local evaluation uses seeds starting at 80,000 (`src/training.py:18-20,219-230`); the submitted Colab evaluation uses seeds 1000-1049 (`train_colab.ipynb:374-435`). Thus the observed final Colab set is distinct from the visible 70,000 selection set and training seeds 1-500. The model file hash confirms the root `models/teacher.pt` is identical to the Colab result teacher (`SHA-256 4611ddca3a580f7e8d8c9777c0447a47783965d08831fe0540248ffa40046dd2`).

Checkpoint selection risks: only 20 validation seeds, selection can overfit that set, and energy is considered only for successful trajectories. A final independent test set reduces direct leakage, but one training run and one selected checkpoint still provide little run-level evidence. The Colab notebook only saves weights with `teacher.save('models/teacher.pt')` (`train_colab.ipynb:276-281`); the ZIP's enumerated contents do not include optimizer/replay checkpoint files (`train_colab.ipynb:616-625`). Full optimizer/replay/RNG resume is implemented in the **local** code (`src/dqn_teacher.py:211-280`, `src/training.py:447-480,638-703`) and has a test definition (`tests/test_dqn_teacher.py:48-76`), but it is not integrated into the submitted Colab run. So checkpointing is **partially implemented at project level**, not verified for the Colab experiment.

There are multiple weight files. The current `models/teacher.pt` and result teacher have 89 output actions and 94,809 parameters. Older `models/teacher_colab.pt`, `models/teachers.pt`, and `models/teacher_pre_ik_assisted.pt` have 25 output actions and 78,361 parameters (independently inspected tensor shapes). The backend upload validator constructs the 89-action architecture and loads weights strictly (`src/backend.py:337-352`), so those old checkpoints are not compatible with the current action table. Keep them clearly marked as historical; do not upload them as current teacher models.

## 7. Training failure: what is known and plausible causes

**Known from data:** the saved Colab training history has 500 rows and exactly 3 `task_done=True` rows. Completion during online training is 0.6%. This is independently verified from `results/colab_run_2026-11-24/training_history.csv`. The separate held-out teacher result is 44/50 after IK warm-start and checkpoint selection (`results/colab_run_2026-11-24/README.md:23-30`; the generated verified summary).

Potential causes supported by the implementation, but **not proven causal by an ablation**:

1. Low exploration for an 89-action problem: epsilon begins at 0.10, falls toward 0.01 (`src/training.py:518`; notebook cell at line 179).
2. Coarse, limited multi-joint action combinations (`src/dqn_teacher.py:44-63`).
3. Strong competing progress/completion terms relative to the small energy coefficient, plus a binary rather than magnitude-sensitive peak penalty (`src/environment.py:77-82,266-291`).
4. Bootstrapping through time-limit truncations (`src/training.py:524-532`; `src/dqn_teacher.py:171-178`).
5. Demonstration quality: behavior cloning stores every visited IK state/action, including episodes that time out; the collection code does not discard failed episodes (`src/training.py:108-142`).
6. Basic DQN instability and mixed-scale, unnormalized observations (`src/dqn_teacher.py:167-200`; `src/environment.py:315-339`).
7. Only 500 episodes with a hard 200-step limit and a changing target mid-episode (`src/environment.py:67-85,200-206`).

To establish causes, run controlled ablations over multiple training seeds and hold all evaluation seeds fixed. Do not state any of these plausible explanations as a measured fact.

## 8. Evaluation validity and independently verified results

The result CSVs agree on the main numbers. `verified_evaluation_summary.csv` reports DQN 44/50, 88%, mean energy over its successful episodes 274.0852 Ws; IK 45/50, 90%, 250.5545 Ws; student 6/50, 12%, 242.4473 Ws among only six successes. `paired_evaluation.csv` has 50 rows, 44 valid rows and the same 44/5/1/0 success contingency. Recalculation from its 44 rounded per-seed savings gives mean -10.8227%, median -8.955%, sample SD 12.5406 percentage points, 35 negative values and 9 positive values. This reproduces -10.82%, mean peak reduction 0.22%, and median peak reduction 3.24% in the README (`results/colab_run_2026-11-24/README.md:7-21`).

The original `evaluation_summary.csv` in the Colab folder has only teacher and student rows; it omits the IK baseline. The newer `verified_evaluation_summary.csv` adds the baseline, and `paired_evaluation.csv` supplies the seed-level DQN/IK comparison. Cite the verified file for the final baseline comparison, not the original summary by itself.

### Statistical limitations

- Only 50 paired seeds and one selected teacher checkpoint are represented. Seeds are simulator scenarios, not independent training runs.
- The completion difference is one discordant seed in favor of IK and none in favor of DQN. Exact paired McNemar p=1.0; this is no evidence of a completion advantage, but it is not proof of equivalence.
- The 44 paired energy percentages are rounded to two decimals in the file; the mean recalculated from the saved values is -10.8227%. The reported value is arithmetically consistent.
- The result files do not report a confidence interval, bootstrap, pre-registered analysis plan, or repeated-seed training variance. A more complete report should include a paired bootstrap interval for energy percentage change and repeated training seeds.
- Student mean energy is conditional on only six successes; it cannot support a claim that the student is more efficient than the teacher.
- The training learning curve is online exploratory behavior and is not the held-out selected-checkpoint score. The project's own README correctly warns against merging them (`README.md:59-62`).
- Notebook cells show `execution_count: null` and `outputs: []` throughout (`train_colab.ipynb:14-16,61-63,112-114,285-287,364-366,442-444,482-484,532-534,606-608`). The CSV and model artifacts can be checked, but the notebook itself does not preserve a Colab execution transcript. The teacher model checksum does establish that the root teacher copy and dated result copy are byte-identical.

## 9. Distillation and student model

The notebook defines a 23-input, 64/64-hidden-layer student with 89 outputs (`train_colab.ipynb:293-314`). It distills teacher softmax outputs at temperature 3 and combines KL loss with a cross-entropy hard-action loss using alpha 0.7 (`train_colab.ipynb:316-355`). It selects one random minibatch of at most 256 replay states per loop iteration and runs 50 iterations labelled `EPOCHS=50` (`train_colab.ipynb:324-355`); these are not 50 full passes over the replay dataset. Hard CE labels are the actions actually stored in replay, which can include exploratory actions, rather than only expert or successful teacher actions.

The log shows loss falling from 23.9062 to 12.3625 over those 50 updates (`results/colab_run_2026-11-24/distillation_losses.csv`). That establishes that the scalar training objective decreased, **not** that the student preserved behavior. The student completed 6/50 held-out targets (12%) compared with the teacher's 44/50 (88%). The Colab report is right to say it is not a capable current controller (`docs/PROJECT_HANDOFF.md:99-101`; `models/README.md:7-12`). No student-vs-teacher paired success retention or validated student energy advantage is demonstrated.

## 10. Parameter counts, latency, ONNX, and edge readiness

Saved tensor shapes independently confirm 94,809 teacher parameters (23-256-256-89) and 11,481 student parameters (23-64-64-89), matching `model_comparison.csv`. This is an 8.2579x reduction in parameter count (about 87.9%). It does not by itself prove a speed, memory, energy, or task-performance improvement.

As an audit-only, in-memory CPU forward-pass timing on this Windows Intel host (one Torch CPU thread, batch size 1, 100 warm-ups, 1,000 timed inferences), the current teacher measured median 0.137 ms, p95 0.236 ms, p99 0.340 ms; the student measured median 0.099 ms, p95 0.167 ms, p99 0.187 ms. These are **not Raspberry Pi measurements**, do not include environment/sensor/actuator costs, and should not be compared to the proposal's 100 ms target as an edge result. They are a local desktop neural-network forward-pass check only.

ONNX is partially implemented: the Colab notebook exports opset 17, calls `onnx.checker`, then does one dummy CPU ONNX Runtime inference and prints output shape (`train_colab.ipynb:447-477`). The file exists at `results/colab_run_2026-11-24/models/student.onnx` and `models/student_experimental.onnx`, both 1,873 bytes. The notebook's empty execution outputs do not establish that those checks passed. ONNX Runtime/onnx are not installed in the current `.conda-training` environment used for this audit (`import onnx` failed); I therefore did not independently load/check the ONNX graph. A 1,873-byte graph is also unexpectedly small compared with the 11,481-parameter FP32 PyTorch student (about 45.9 KB of raw parameter values). This discrepancy is a validation warning, not proof that the file is corrupt. The ONNX graph should be opened with the declared ONNX checker, inference compared numerically with the PyTorch student on multiple real observations, and its initializer/external-data contents explained before claiming a valid deployment artifact.

**Edge deployment readiness: not ready.** No Raspberry Pi model/OS/runtime benchmark, memory profile, sustained thermal/power test, real-time control integration, sensor/action interface, hardware safety validation, or student behavioral pass criterion is documented. The student's 12% completion alone is disqualifying for robot control. A small file/parameter count and an ONNX extension are not deployment evidence.

## 11. Energy buffer assumptions

The buffer starts empty, has a 50 Ws capacity and 80% charge efficiency (`src/environment.py:71-74,183-191`). Positive work counts as grid draw; negative work can be stored in the light phase; in the heavy phase it offsets the next positive draw (`src/environment.py:230-264`). This is a **simulated bookkeeping mechanism**, not a measured battery or regenerative drive. It omits round-trip discharge losses and electrical conversion; it does not use real current, voltage, battery state-of-charge, power limits, or grid price. The peak metric is a per-step energy sample, not a measured grid-demand meter. These limitations are acknowledged in `src/environment.py:1-7` and `README.md:77`.

## 12. Research originality, limits, and unmet objectives

The auditable repository supports a student prototype/evaluation contribution, not a novel DQN algorithm claim. IK behavior cloning and DQN fine-tuning are established method families; the project should state its contribution as applying/evaluating an IK-assisted learning and simulated energy-buffer pipeline for this setup. The two-page status report contains no bibliography (`docs/Energy_Aware_Robot_DQN_Report.pdf`, pages 1-2). README claims a proposal and systematic-review draft screened 195 records and included six studies (`README.md:54-55`), but those documents are not present in the audited project tree, so the review, gap and originality cannot be independently judged here. The poster includes a small selected bibliography, which improves the artifact but is not a substitute for the missing review evidence.

The industrial steel CSV is not robot telemetry and does not drive the training scenario (`README.md:44-48,77,184-188`). It is not necessary to find a new dataset to validate this simulator, but it must not be presented as robot energy data. The experiment creates target and joint-state/action data in PyBullet.

### Status by feature/objective

| Feature/objective | Status | Evidence and limit |
|---|---|---|
| PyBullet KUKA simulation and Gymnasium API | **Implemented** | `src/environment.py:57-62,87-137,140-151`; environment tests are defined at `tests/test_environment.py:12-47`. Simulation only. |
| 23 state values and 89 DQN commands | **Implemented** | `src/environment.py:120-137`; `src/dqn_teacher.py:44-80`. The Gym API itself exposes continuous Box actions. |
| IK baseline and demonstrations | **Implemented** | `src/training.py:33-98,103-165,191-216`; strong baseline, reused to warm-start DQN. |
| Online training and per-episode CSV | **Implemented, poor learning outcome** | 500 recorded episodes, 3 successes, in dated CSV. |
| Held-out same-seed evaluation | **Implemented** | 50 seeds and paired CSVs. One checkpoint, 50 scenarios; no replicated training-run distribution. |
| O1: reproducible training environment/setup | **Partially implemented** | Windows Conda and Colab paths exist, but dependency versions are ranges and notebook execution/Colab environment metadata are absent. See `requirements.txt:1-19`, `environment-training.yml:1-10`, notebook null outputs. |
| O2: optimizer/replay checkpoint and resume | **Partially implemented** | Local `.pt` checkpoint saves/restores optimizer, replay and RNG; test definition exists. Colab only exports model weights and does not package the full resume checkpoint. |
| O3: one >=50-episode run and per-episode metrics | **Implemented as a run/log deliverable** | 500 rows with reward, task status, energy, etc. The run had only 3 successes; fulfilling the logging objective is not evidence of learning success. |
| O4: baseline comparison/report | **Implemented with important caveats** | 2-page PDF and paired CSV; the valid energy result is negative. No formal uncertainty analysis in the report. |
| Energy saving/cost objective | **Not achieved / cost objective not modeled** | Matched mean saving -10.82%; no tariff/cost calculation. |
| Distillation | **Partially implemented** | Student and loss CSV exist; 6/50 targets. Loss convergence did not produce useful behavior. |
| ONNX export | **Artifact present; validation unverified in saved run** | Notebook contains checker and one dummy inference code but no saved cell outputs; current audit runtime lacks ONNX package. |
| Raspberry Pi/edge deployment | **Missing** | No Pi benchmark or deployed control loop. |
| Physical KUKA or measured electrical energy | **Missing** | No hardware/power evidence. |
| Actual pick-and-place | **Missing** | No gripper, object, grasp/release state, or placement condition in the task code. |
| Research literature review/originality evidence | **Claimed in README, files missing here** | README `54-55`; review manuscript/proposal not in the audited tree. |

## 13. Documentation/code/result mismatches and audit flags

1. **Peak reward mismatch:** comment describes `peak_spike`; reward uses boolean `peak_penalty` (`src/environment.py:24-32,266-288`).
2. **Continuous-vs-discrete wording:** environment declares six continuous values; teacher selects among a custom discrete 89-entry table (`src/environment.py:135-137`; `src/dqn_teacher.py:44-63`). Explain the two layers clearly.
3. **Resumability scope:** local code checkpoints optimizer/replay/RNG; Colab notebook ZIP list saves weights/results only (`src/dqn_teacher.py:211-280`; `train_colab.ipynb:616-625`). Do not claim the submitted Colab run can resume full optimizer/replay state.
4. **Old model compatibility:** three older root teacher files have 25 outputs versus current 89; current backend strict-load expects 89 (`src/backend.py:343-352`).
5. **ONNX evidence:** export/check code is present, but notebook cells have no stored execution trace; model file is only 1,873 bytes, with no local checker available. Treat ONNX readiness as unverified.
6. **Report differences:** original Colab `evaluation_summary.csv` omits the IK baseline; verified summary and paired CSV contain the usable baseline comparison. cite `verified_evaluation_summary.csv` and `paired_evaluation.csv`.
7. **Average energy interpretation:** per-controller completed-episode means are based on different success sets. Use paired per-seed percent results for the energy headline.
8. **Training outcome and evaluation outcome:** the 3/500 learning-history result and 44/50 selected-checkpoint test are not contradictory; they are different phases. Label each precisely.
9. **Research claims:** README says systematic review and proposal exist, but their documents are not included in the current file tree (`README.md:54-55`). Missing from checkout means unverified, not necessarily nonexistent elsewhere.
10. **Training-test evidence:** `docs/PROJECT_HANDOFF.md:58-75` records a previous 28-test pass. This audit did not rerun the suite; passing software tests would not prove energy savings in any event.

## 14. Is this pick-and-place?

**No. It is target reaching.** The environment creates two XYZ target locations (`src/environment.py:169-206`), measures end-effector distance (`src/environment.py:208-211,276-282`), and declares task success when the end effector is close to the heavy target (`src/environment.py:281-294`). The scene loads only a plane and robot arm (`src/environment.py:140-151`). There is no manipulated object, gripper actuation, grasp detection, collision-aware object handling, transport phase, placement region, or release verification in this task. “Pick-and-place” should not appear as a completed capability.

## 15. Provisional DSC3114 rubric assessment

This is an artifact-based supervisor estimate against the supplied 10-mark categories, not an official examiner grade. The oral response category cannot be scored from files because no live defence or recording was observed.

| Criterion | Evidence-based readiness | Strict assessment |
|---|---:|---|
| Research grounding | **5/10** | The project problem and two research references appear in the current poster; the repository's two-page result report has no citations, and the claimed proposal/systematic-review files are absent from this checkout. Originality/gap cannot be independently verified. |
| Project implementation | **7/10** | Real simulator, DQN, IK baseline, logging, matched evaluation, checkpointing, dashboard and student artifact exist. The energy objective is negative; the student is weak; physical robot, cost model and edge validation are missing. |
| Articulation | **7/10 provisional** | The poster states method, sample, headline result, negative finding and limitations in accessible language. Keep the live explanation focused; avoid “energy saving,” “pick-and-place,” “real KUKA test,” or “deployment.” |
| Response to questions | **Not scorable from files** | A written question bank can prepare you, but only an observed oral answer demonstrates clarity, correctness and context. No mark is invented here. |

The defensible written-artifact subtotal is **19/30**. It is not converted into a 40-mark total because the response-to-questions performance has not been observed.

## 16. Priority improvements before making stronger claims

1. **Lock the experiment.** Record exact commit/source ZIP hash, package versions, Python/PyTorch/PyBullet versions, device, random seeds, selected checkpoint identity, and notebook execution outputs.
2. **Preserve the time-limit flag correctly.** Store `terminated or truncated` as done where a 200-step episode is terminal, or document and test a different finite-horizon convention.
3. **Separate training, checkpoint selection, and final test.** Use multiple training seeds, a validation set for selection, and a held-out test set untouched by tuning. Report run-level mean/interval.
4. **Add paired uncertainty.** Provide raw paired seed values and bootstrap confidence intervals for energy saving; report discordant completion counts and paired test. Retain the negative estimate.
5. **Audit reward scales.** Log each reward component, use magnitude-aware peak penalties if that is the intended objective, and run ablations. Do not infer causality without these experiments.
6. **Calibrate the energy model.** Define units explicitly, report watts vs joules per simulation step, model round-trip losses, and compare with a measured drive or a validated simulator if making physical claims.
7. **Keep costs distinct from energy.** If the thesis claims cost optimisation, introduce a documented tariff/time profile and calculate currency cost; otherwise limit claims to simulated energy.
8. **Fix student evaluation before deployment.** Train on curated expert/successful behavior or an explicitly justified teacher objective, evaluate over several seeds, and specify a retention criterion before deploying.
9. **Validate ONNX.** Run checker, confirm inputs/outputs and 89 actions, compare PyTorch/ONNX logits/actions on a batch of real observations, and record Pi latency/memory/thermal results. Reconcile the 1,873-byte file.
10. **Name the task accurately.** If pick-and-place is a goal, add object/gripper/grasp/placement success metrics and test them. Until then call this target reaching.
11. **Resolve model artifacts.** Mark 25-action weights as legacy and validate/upload only the 89-action checkpoint. Keep source and result copies with hashes.
12. **Add the actual literature review and proposal** to the repository or cite exact archival locations so panel claims of research gap and originality are auditable.

## 17. Challenging examination questions with accurate answers

1. **What is the research problem?**  
   The current controller must be evaluated on target completion and simulated energy use. The experiment does not yet show energy reduction. Cite `README.md:32-40`; `src/environment.py:1-7`.

2. **What is your actual research question?**  
   Whether an IK-assisted DQN can reach targets while lowering simulated grid energy against a fixed-speed IK baseline on the same scenarios. The measured answer here is: it reached targets, but did not lower matched-success energy. Cite `README.md:36-39`; `results/colab_run_2026-11-24/README.md:7-21`.

3. **What is the contribution if the energy target failed?**  
   The contribution is a working simulation/training/evaluation pipeline and an honest negative result for this selected checkpoint. It is not evidence of energy savings or a novel RL algorithm. Cite `src/environment.py:57-62`; `src/training.py:219-369`.

4. **Did the DQN learn entirely by itself?**  
   No. It was warm-started by 100 IK demonstration episodes and 3,000 behavior-cloning updates, then fine-tuned with DQN and an imitation term. Cite `train_colab.ipynb:122-167,147-157,184-185`.

5. **What is the robot doing?**  
   It moves a simulated end effector to two XYZ targets. It is not grasping or placing an object. Cite `src/environment.py:140-151,169-206,281-294`.

6. **Why use an IK baseline?**  
   IK converts the desired Cartesian point to joint angles and provides a clear target-reaching reference. It is strong and is also the demonstrator, so the comparison is about approximating/augmenting IK, not learning without privileged IK. Cite `src/training.py:51-63,103-142,191-216`.

7. **How many physical joints does the model have?**  
   The PyBullet iiwa model has seven; the environment locks the final wrist-roll joint and controls six. Cite `src/environment.py:44-47,148-151,162-167`.

8. **What does the agent observe?**  
   23 float values: six positions, six velocities, end-effector XYZ, target XYZ, phase, normalized buffer, normalized recent grid draw, target distance and episode fraction. Cite `src/environment.py:9-18,120-137,315-339`.

9. **How do you get 89 actions?**  
   One stop action plus 24 single-joint commands (six joints times four nonzero levels) plus 64 coordinated sign combinations, total 89. Cite `src/dqn_teacher.py:38-63`.

10. **Why is the Gym action space continuous if this is DQN?**  
    The environment accepts a six-component continuous velocity vector; the DQN chooses a discrete index, and `action_table` maps that index to a six-component vector. Cite `src/environment.py:135-137,197-220`; `src/dqn_teacher.py:44-63,128-138`.

11. **Describe the teacher network.**  
    A 23-256-256-89 MLP with ReLU between layers. The output is one Q-value per discrete command. Cite `src/dqn_teacher.py:66-80`.

12. **What stabilizes DQN training?**  
    Uniform replay and a target network; the target network is copied every five episodes in the project runner. It is still basic DQN and can overestimate values. Cite `src/dqn_teacher.py:83-104,120-147,171-178`; `src/training.py:638-639`.

13. **What hyperparameters matter?**  
    Adam defaults to learning rate 1e-3, gamma 0.99, replay capacity 10,000; training samples batches of 64 and epsilon begins at 0.10 and decays toward 0.01. Cite `src/dqn_teacher.py:113-126`; `src/training.py:518,533-539`.

14. **Why are there both 3/500 and 44/50 results?**  
    3/500 is online exploratory fine-tuning success. 44/50 is greedy evaluation of a selected, IK-warm-started checkpoint on a different held-out set. They are different phases and policies. Cite `results/colab_run_2026-11-24/README.md:23-30`; `README.md:59-62`.

15. **Does 44/50 mean 88% training success?**  
    No. It means the selected checkpoint succeeded on 44 of 50 held-out evaluation episodes. The online training history succeeded 3 of 500. Cite both sources in answer 14.

16. **How did you select the checkpoint?**  
    The runner scores warm-start and fine-tuned checkpoints on 20 validation seeds, prioritizes success count and then successful-episode energy. Selection is susceptible to the small validation set; the final evaluation should remain separate. Cite `src/training.py:168-188,660-690`; `train_colab.ipynb:251-274`.

17. **Is the baseline comparison fair?**  
    Both controllers see the same seeded targets and simulator limits, and paired energy is restricted to common successes. But IK has direct access to PyBullet IK each step while DQN chooses from a coarse action table; state that asymmetry. Cite `src/training.py:191-216,219-273`; `src/dqn_teacher.py:44-63`.

18. **How is success defined?**  
    End-effector distance below 0.15 m from the heavy target before the 200-step horizon. Cite `src/environment.py:67-70,281-294`.

19. **What does -10.82% mean?**  
    Saving is `(baseline energy - DQN energy)/baseline energy * 100`; a negative mean means DQN used more energy. It is the mean of 44 per-seed percentage changes where both completed. Cite `src/training.py:242-273` and paired CSV.

20. **Why not compare 274.09 Ws and 250.55 Ws directly?**  
    Those averages are computed over different successful seed sets (44 DQN successes, 45 IK successes). Use the 44 matched paired changes for the energy headline. Cite `verified_evaluation_summary.csv`; `paired_evaluation.csv`.

21. **Is 88% better than 90%?**  
    No. It is one fewer success. The paired table has zero DQN-only wins and one IK-only win; exact McNemar p=1.0. Fifty seeds do not establish equivalence or superiority. Cite `paired_evaluation.csv`.

22. **How was the result independently checked?**  
    The saved paired CSV has 50 rows and 44 valid paired-success rows. Recomputing the 44 rounded percentages gives mean -10.8227%, median -8.955%; the separate training history has 500 rows and 3 true task flags. Files are under `results/colab_run_2026-11-24/`.

23. **What is the confidence interval?**  
    The files do not report one for paired energy. Individual Wilson intervals for completion are about 76.2-94.4% and 78.6-95.7%, but a paired analysis is preferable. I would add a paired bootstrap interval over the 44 success pairs and multiple training runs. No interval should be invented as a reported project result.

24. **How is energy estimated?**  
    Each controlled joint uses applied torque times angular velocity times dt; positive work is summed as raw draw, negative as regenerative work. This is simulated mechanical-work accounting, not electrical-meter data. Cite `src/environment.py:230-238`.

25. **What does 50 Ws mean?**  
    It is a 50 joule-second-equivalent energy capacity (watt-second = joule) in the simulator, not 50 watts of power. The code charges at 80%; it currently subtracts deployed buffer energy one-for-one. Cite `src/environment.py:71-74,250-264`.

26. **Was grid peak power actually measured?**  
    No. The code records the maximum per-step grid-energy sample. It does not convert it to watts or use a physical demand meter. Cite `src/environment.py:231,266-273`.

27. **Does reward penalize how far above the peak threshold the system goes?**  
    Not currently. Although `peak_spike` is calculated, reward subtracts a fixed penalty when grid draw exceeds the threshold. Cite `src/environment.py:266-288`.

28. **Could the time-limit handling affect learning?**  
    Yes. The episode loop ends on truncation, but replay stores only `terminated`; DQN therefore bootstraps at a time limit. This is a likely value-target inconsistency if the 200-step horizon is terminal. Cite `src/training.py:520-532`; `src/dqn_teacher.py:171-178`; `src/environment.py:293-294`.

29. **What dataset trains the robot?**  
    The DQN uses simulator-generated states, targets and IK demonstrations. The steel-industry CSV is contextual and is not robot telemetry or a policy input. Cite `README.md:44-48,77,184-188`.

30. **What is the student model?**  
    A 23-64-64-89 MLP trained with temperature-scaled KL matching plus cross-entropy on replay actions for 50 minibatch updates. Cite `train_colab.ipynb:293-355`.

31. **Did distillation succeed?**  
    The loss decreased from about 23.91 to 12.36, but held-out task completion was only 6/50 versus teacher 44/50. Loss convergence is not behavior retention. Cite `distillation_losses.csv`; `verified_evaluation_summary.csv`.

32. **How much smaller is the student?**  
    Parameter counts are 94,809 teacher and 11,481 student, a ratio of about 8.26x. That is a size reduction, not proof of usable control. Cite `model_comparison.csv`.

33. **Is the ONNX export validated?**  
    The notebook contains checker and one dummy CPU inference code, but stored notebook cells have no outputs, and the current local audit environment lacked the ONNX package. I cannot claim independent ONNX validation from these files. Cite `train_colab.ipynb:447-477` and its null output metadata.

34. **Did you meet the 100 ms Raspberry Pi latency target?**  
    There is no Pi measurement. This audit measured only a batch-one PyTorch model forward pass on a desktop CPU (student median 0.099 ms); it excludes simulation and I/O and says nothing about Pi latency. Do not present it as the proposal's edge result.

35. **Is the student ready for edge deployment?**  
    No. It reaches 12% of held-out targets, and no Pi, memory, power, thermal, control-loop or safety evaluation exists. The ONNX file still needs integrity/parity checks. Cite `README.md:47,60`; `docs/PROJECT_HANDOFF.md:99-101`.

36. **What is missing for actual pick-and-place?**  
    A gripper, object model, grasp/attach condition, transport target, release and placement-success condition. The current task only tests end-effector distance to a target. Cite `src/environment.py:140-151,169-206,281-294`.

37. **Can you claim industrial energy-cost savings?**  
    No. The energy model is simulated and no tariff/cost or factory measurement is used. State “simulated joint-work proxy” and the negative paired result. Cite `src/environment.py:1-7,230-264`; `README.md:77`.

38. **Why does the training outcome fail?**  
    The observed fact is 3/500 successes. Low exploration, action discretization, reward scale, demo quality and time-limit bootstrapping are plausible causes, but the project has not isolated them experimentally. Cite the relevant implementation in `src/training.py:108-165,518,524-539`; `src/dqn_teacher.py:44-63,171-200`; `src/environment.py:77-82,266-294`.

39. **Can you resume the exact Colab training?**  
    Not from the submitted model weights alone. Local code has a full optimizer/replay/RNG checkpoint, but the Colab artifact list includes teacher/student weights and CSVs rather than that full checkpoint. Cite `src/dqn_teacher.py:211-280`; `train_colab.ipynb:616-625`.

40. **What is your next experiment?**  
    Fix and document finite-horizon handling, run multiple independent training seeds, keep checkpoint-selection and test seeds separate, report completion and paired energy with confidence intervals, then improve student behavior before measuring Pi. Do not change the old result; publish new results as a new run.

## 18. Sources and file evidence

Primary research sources relevant to framing (not evidence that this implementation achieved their results):

- Li, X., Liu, H., and Dong, M. (2022). “A General Framework of Motion Planning for Redundant Robot Manipulator Based on Deep Reinforcement Learning.” *IEEE Transactions on Industrial Informatics*, 18(8), 5253-5263. [doi:10.1109/TII.2021.3125447](https://doi.org/10.1109/TII.2021.3125447).
- Peri, S., Perincherry, A., Pandit, B., and Lee, S. (2025). “Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control.” *Proceedings of the 9th Conference on Robot Learning*, PMLR 305, 221-237. [PMLR paper](https://proceedings.mlr.press/v305/peri25a.html).

No implementation or experimental result was edited while preparing this audit. No project test suite was run in this audit; a prior dated handoff reports 28 passed tests (`docs/PROJECT_HANDOFF.md:58-66`), which remains a historical record rather than a fresh run here.
