# Objective status at internal defence

| Objective | Status | Evidence / boundary |
|---|---|---|
| Reproducible PyBullet target-reaching environment | Implemented and run | V1.1 config and recovered Colab episode metrics; target reaching only. |
| Corrected student distillation | Completed for this run | 100 epochs; 89 successful teacher trajectories selected; trajectory split, loss history and selected checkpoint preserved. |
| Improve student task completion | Achieved in this simulation | 45/50 vs original student 7/50; final evaluation seeds 140000–140049. |
| Retain teacher behavior while compressing | Achieved on observed completion count | Teacher 46/50; student 45/50; 97.83% ratio; student 87.89% fewer parameters. |
| Reduce simulated energy versus fixed IK | Not achieved | Teacher −180.58%; student −157.77% mean paired saving on 44 jointly successful seeds. |
| Improve teacher via fine-tuning | Failed experiment | Fine-tuned checkpoint 0/50 held-out; original checkpoint retained. |
| Strong alternative IK baselines | Not achieved | Slow/proportional/adaptive settings each had 0/20 validation and 0/50 final successes; fixed-speed IK is the only working comparator here. |
| Validate with real robot electrical measurements | Not completed | SDU data access required login; Mendeley README inspected, bulk measurements not downloaded. |
| ONNX export and parity | Completed for teacher and student | ONNX checker passed; batch sizes 1, 2 and 8 matched PyTorch within absolute tolerance 5e-5 and greedy actions agreed 100%. |
| Raspberry Pi deployment / ≤100 ms end-to-end control | Not tested | Only Colab x86_64 neural-network timing was recorded. |
| Physical grasp-and-place | Out of current implementation | Simulated two-phase end-effector target reaching only. |

Report failed objectives plainly. Do not describe reduced energy, physical deployment, or pick-and-place as achieved.
