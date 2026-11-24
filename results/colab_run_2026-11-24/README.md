# Colab prototype run - 24 November 2026

This directory preserves the exact files from `prototype_deliverables (9).zip`.
The original CSVs and plot are retained unchanged.

## Reported held-out results

The teacher, IK baseline, and student were re-evaluated on the same 50 seeds
(1000-1049) using the project simulator and the submitted model weights.

| Controller | Completed | Completion | Mean energy on completed episodes |
|---|---:|---:|---:|
| IK-assisted DQN | 44/50 | 88% | 274.09 Ws |
| IK baseline | 45/50 | 90% | 250.55 Ws |
| Student | 6/50 | 12% | 242.45 Ws (six successes) |

Among the 44 matched successful teacher/IK episodes, mean per-seed energy
savings was -10.82% (the teacher used more energy). Mean peak reduction was
0.22%; median peak reduction was 3.24%. This run demonstrates target reaching,
not energy savings.

`matched_seed_evaluation.csv` contains all 150 controller/seed rows.
`paired_evaluation.csv` contains teacher-vs-IK comparisons and leaves invalid
comparisons blank unless both policies completed the same seed.
`verified_evaluation_summary.csv` contains the same-seed summary. The original
`evaluation_summary.csv` is preserved as delivered by Colab.

The original training history records 3/500 online successes. The 44/50 result
is a separate greedy evaluation of the selected warm-start checkpoint, so the
two rates describe different policies and phases of the pipeline.
