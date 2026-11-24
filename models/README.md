# Prototype checkpoints

- `teacher.pt` is the selected IK-assisted DQN checkpoint from the Colab run on
  24 November 2026. It reached 44/50 held-out targets (88%) in the submitted
  evaluation. Use it for the simulated prototype demonstration.
- `teacher_pre_ik_assisted.pt` preserves the previous local teacher checkpoint.
- `student_experimental.pt` and `student_experimental.onnx` are retained for
  reference only. The student reached 6/50 targets (12%) and is not ready to
  control the prototype.

The teacher result demonstrates target reaching. It did not reduce energy
against the matched IK baseline in the latest paired evaluation.
