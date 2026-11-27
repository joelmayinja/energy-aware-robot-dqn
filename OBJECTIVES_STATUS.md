# Research objectives and evidence status

Status recorded 26 November 2026. This file distinguishes the preserved historical experiment from V1.1 implementation work. V1.1 is not a claim of improved research performance.

| Objective / claim | Status | Evidence |
|---|---|---|
| Simulate KUKA iiwa target-reaching with PyBullet and Gymnasium | Implemented in V1.0; preserved in V1.1 | `src/environment.py`; isolated wrapper `experiments/v1_1/environment_v1_1.py` |
| 23-value observation and 89-action DQN teacher | Implemented | `src/dqn_teacher.py`; checkpoint shape verified by `tests/test_v1_1_pipeline.py` |
| Task completion for selected teacher | Historically demonstrated: 44/50 (88%) | `results/colab_run_2026-11-24/verified_evaluation_summary.csv` |
| IK baseline performance | Historically demonstrated: 45/50 (90%) | Same verified summary; paired evaluation CSV |
| Positive energy saving | Not achieved historically | Mean paired saving -10.82%, i.e. teacher consumed more energy on the 44 shared successful scenarios |
| Successful online DQN learning | Weak historical result: 3/500 episodes (0.6%) | `results/colab_run_2026-11-24/training_history.csv` |
| Student behavior retention | Not achieved historically: 6/50 (12%) versus teacher 44/50 | Verified evaluation summary; original student checkpoint remains preserved |
| V1.1 energy ledger and magnitude-sensitive peak term | Implemented in isolated experimental code; local unit tests passed | `experiments/v1_1/environment_v1_1.py`; `tests/test_v1_1_pipeline.py` |
| V1.1 Double DQN, Huber loss, clipping, timeout terminal mask | Implemented in isolated trainer; local update test passed | `experiments/v1_1/pipeline.py`; unit test |
| V1.1 student distillation | Smoke-tested only locally (3 demo trajectories, 2 epochs); research result pending | `experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/` |
| Self-contained Colab notebook | Generated, JSON/syntax validation pending final delivery check | `FINAL_RESEARCH_TRAINING.ipynb` |
| New robot datasets | Metadata inspected; neither raw archive retrieved in this local session | `DATASET_MANIFEST.json`, `DATASET_PROVENANCE.md` |
| ONNX parity and edge deployment | Pending. Local training environment lacks ONNX packages; notebook includes checks | Do not claim deployment readiness or Raspberry Pi results |
| Pick-and-place | Missing / out of scope of implementation | Current task is end-effector target reaching; there are no objects, gripper, grasp, or placement success states |

No current report should state that the DQN has proven energy-cost savings. The project demonstrates a target-reaching prototype and an experimental measurement pipeline; improved results remain to be established by the new fresh-seed evaluation.
