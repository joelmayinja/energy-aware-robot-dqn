# Limitations and unresolved issues

1. **Energy validity:** energy is a simplified mechanical-work proxy from PyBullet joint torque and velocity. It is not mains electricity or tariff cost. Buffer efficiencies are assumptions.
2. **Negative energy result:** successful learned policies use considerably more estimated energy than fixed-speed IK. Energy optimization remains unsolved.
3. **Failure-cost interpretation:** policies with zero completions can have low total energy because they stop making progress. Such values must not be described as savings. Success-conditioned metrics are N/A with zero successful pairs.
4. **Training replication:** there is one recovered training run per policy. No across-training-seed uncertainty is available.
5. **Task scope:** the robot reaches targets in a two-phase task. There is no object, grasping, payload interaction, or placement.
6. **Baseline quality:** fixed-speed IK is the usable comparator. The three newly configured speed variants scored 0/20 on validation, so this experiment does not establish superiority over well-tuned adaptive controllers.
7. **Student generalisation:** 45/50 is promising but based on one trained student and 50 simulator seeds. Teacher-to-student completion ratio is descriptive, not an equivalence guarantee.
8. **Teacher fine-tuning:** 30 episodes produced 0/50 final completion. The run documents failure but does not identify a unique cause.
9. **External robot data:** no SDU or Mendeley robot measurement rows were used. SDU download requires account access; only Mendeley documentation was obtained.
10. **Deployment:** teacher/student ONNX parity passed locally at batch sizes 1, 2 and 8, but no Raspberry Pi test, physical robot trial, or end-to-end latency measurement is available.
11. **Steel dataset:** industrial load data are descriptive context and cannot validate robot energy behavior.
12. **Peak power:** reported values derive from per-step simulated grid-energy divided by timestep; they are not meter readings or grid-demand measurements.
