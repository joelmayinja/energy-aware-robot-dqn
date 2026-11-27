# Research methodology

## Task and simulator

PyBullet DIRECT in training and PyBullet GUI in the local demonstration use the existing KUKA iiwa URDF. Six joints are controlled and the wrist joint remains locked. The observation has 23 values and the DQN action table has 89 discrete joint-velocity choices. Episodes have two target phases and a 200-step limit. Success is reaching the second-phase target within the configured tolerance. There is no object, gripper operation, grasp, or placement.

## Energy model

V1.1 estimates joint mechanical work as the sum of joint torque multiplied by angular velocity and the actual simulation timestep. Positive work is charged to the simulated grid ledger. Negative work may charge a finite buffer in the light phase; buffer charge and discharge efficiencies are assumed as 0.80 and 0.90. The result is a simulation proxy. It omits measured motor electrical losses, drive electronics, mains input, and tariff structure.

## Training and selection

The selected teacher began from the original 94,809-parameter checkpoint. Greedy teacher demonstrations were collected on a training seed group; 89 successful trajectories out of 100 were retained for student distillation. The 11,481-parameter 23-64-64-89 student was trained for 100 epochs with temperature-scaled KL, teacher-action cross-entropy, Q-value regression, trajectory-disjoint validation, and periodic closed-loop validation. Its selected checkpoint is `student_best_final.pt`. Its state tensors match `student_best_from_original_teacher.pt`, confirming the selected teacher source.

Online teacher fine-tuning was limited to 30 episodes and failed to produce a useful policy. The original teacher was retained based on a separate 20-scenario validation group (19/20 vs 0/20 for fine-tuned). Baseline validation used another separate seed group. Three alternative speed controllers had 0/20 validation completion; therefore fixed-speed IK is the only usable conventional comparator in this run.

## Held-out evaluation and statistics

Nine policies were evaluated once each on the same 50 fresh seeds, 140000–140049. The CSV contains 450 policy/seed rows, no duplicate pairs, and identical seed sets for all policies. Completion intervals are Wilson score 95% intervals. Paired energy saving is calculated only for shared successful episodes as `(fixed IK grid energy − policy grid energy) / fixed IK grid energy × 100`. Percentile bootstrap intervals use 20,000 resamples and seed 20261126. Completion discordance uses a two-sided exact McNemar test.

Failed-attempt energy is reported separately and included in all-attempt average energy and aggregate attempt energy per completed task. If a policy completes no tasks, aggregate energy per completed task and success-conditional energy saving are unavailable, not zero.

## Scope of inference

The independent unit is the scenario seed, but there is one training run per policy. The intervals quantify scenario variation conditional on the trained checkpoints; they do not capture variation across independent training seeds. The 50 scenarios and simplified physics are insufficient to generalise to industrial electricity bills or physical robot deployment.
