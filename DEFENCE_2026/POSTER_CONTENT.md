# Poster content draft — V1.1 internal progress

## Title
Optimising Energy Costs of Robots Using a Lightweight Deep Q-Network

## Aim
Investigate target-reaching performance and simulated energy use for a DQN teacher and a compressed student controlling a PyBullet KUKA iiwa.

## Method
Six controlled joints; 23-value state; 89 discrete actions; two-phase target-reaching episodes; 100 teacher demonstration trajectories; student trained for 100 epochs; 50 shared held-out scenarios (seeds 140000–140049). Energy is estimated from joint mechanical work with assumed buffer efficiencies.

## Results
- Fixed-speed IK: 44/50 targets (88%).
- Selected DQN teacher: 46/50 (92%).
- Selected student: 45/50 (90%); original student: 7/50 (14%).
- Student: 11,481 parameters vs 94,809 for teacher (87.89% reduction).
- Paired energy saving vs IK on 44 shared successes: teacher −180.58%; student −157.77%.
- Fine-tuned teacher: 0/50; original teacher retained.

## Conclusion
Distillation substantially improved the compact student’s simulated target-reaching performance. The energy objective was not achieved: both learned policies used more estimated energy than fixed IK. This is a preliminary simulation result, not measured electricity savings.

## Required caveat on poster
Target reaching only; no grasping or placement. One training run per policy. No physical robot, real robot electrical dataset, Raspberry Pi, or end-to-end deployment validation. ONNX parity was checked on this Windows x86_64 environment only.
