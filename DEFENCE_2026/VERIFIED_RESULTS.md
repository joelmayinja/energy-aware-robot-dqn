# Verified results from the recovered outer deadline run

These figures were recomputed from `results/internal_defence_v1_1_2026-11-26/outer_deadline/final_episode_metrics.csv`. The nested partial deadline output is not mixed into these calculations. There are 9 policies × 50 shared seeds = 450 rows, with no duplicate policy/seed pairs.

## Held-out policy results

| Policy | Completion | Wilson 95% CI | Mean grid J / all attempts | Mean grid J / successes | Mean failed-attempt J | Aggregate J / completed task | Mean peak power W |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-speed IK | 44/50 (88%) | 76.2–94.4% | 101.86 | 99.21 | 121.35 | 115.76 | 700.50 |
| Original teacher | 46/50 (92%) | 81.2–96.8% | 267.10 | 261.42 | 332.44 | 290.33 | 917.29 |
| Selected teacher | 46/50 (92%) | 81.2–96.8% | 267.10 | 261.42 | 332.44 | 290.33 | 917.29 |
| Fine-tuned teacher | 0/50 (0%) | 0–7.1% | 18.39 | N/A | 18.39 | N/A | 112.97 |
| Original student | 7/50 (14%) | 7.0–26.2% | 278.21 | 244.37 | 283.72 | 1,987.20 | 976.32 |
| Selected student | 45/50 (90%) | 78.6–95.7% | 251.51 | 241.94 | 337.58 | 279.45 | 878.85 |

The other fixed-slow, proportional-speed and adaptive-speed IK controllers each completed 0/50 final scenarios. Their success-conditional energy comparison is unavailable. Their failed-attempt energy is in `corrected_policy_summary.csv`.

“Aggregate J / completed task” is total energy across all attempts divided by completed tasks. It therefore includes energy spent on failures. It is not a per-success-only measure. N/A denotes an undefined quantity when there are no successes.

## Paired comparison against fixed-speed IK

| Policy | Shared successful seeds | Mean saving | Median saving | Bootstrap 95% CI | Completion pairs (both / IK-only / policy-only / neither) | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Selected teacher | 44 | −180.58% | −180.15% | −207.35% to −154.39% | 44 / 0 / 2 / 4 | 0.50 |
| Selected student | 44 | −157.77% | −152.34% | −180.56% to −135.70% | 44 / 0 / 1 / 5 | 1.00 |
| Original student | 7 | −232.82% | −211.71% | −291.07% to −175.48% | 7 / 37 / 0 / 6 | 1.46×10⁻¹¹ |
| Fine-tuned teacher | 0 | N/A | N/A | N/A | 0 / 44 / 0 / 6 | 1.14×10⁻¹³ |

Negative energy saving means more energy than fixed IK. The teacher/student intervals use only the 44 scenarios where both policy and baseline succeeded; failed tasks are separately included in the table above. The fine-tuned teacher had no shared successful scenarios, so a success-conditional energy statistic cannot be calculated. Its low all-attempt energy is a failure artefact, not an energy achievement.

The student completion-retention calculation is `45 / 46 × 100 = 97.83%`. It is a ratio of observed completion counts, not a claim of statistical equivalence. The paired McNemar test did not show a significant completion difference between selected student and IK on this sample (p=1.00).

## Models and run details

- Teacher: 94,809 parameters, 379,236 state bytes.
- Student: 11,481 parameters, 45,924 state bytes; parameter reduction 87.89%.
- Distillation: 100 recorded epochs; 89/100 successful teacher trajectories selected.
- Fine-tuning: 30 recorded episodes; 0/50 final held-out completion, so original teacher retained.
- Inference benchmark: 1,000 batch-1 neural-network forward passes on a Colab x86_64 CUDA host. Median teacher 0.130 ms; median student 0.137 ms. This is not whole-controller latency or Raspberry Pi performance.

See `corrected_policy_summary.csv`, `corrected_paired_comparisons.csv`, `verified_model_benchmark.csv`, `independent_verification.json`, and `figures/` for all recomputed values and charts.
