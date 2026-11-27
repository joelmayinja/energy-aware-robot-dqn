# Poster-ready results (verified historical run)

## Main result

On the preserved 50-scenario evaluation, the selected DQN teacher reached 44 targets (88%) and fixed-speed IK reached 45 (90%). On 44 shared successful scenarios, the teacher's mean paired simulated-energy saving was **−10.82%**; it therefore used more simulated energy than IK on average. The median paired saving was −8.96%. Paired completion counts were 44 both successful, five both failed, one IK-only success, and zero DQN-only successes (exact McNemar p=1.0).

## Student and training

The original student reached 6/50 targets (12%) versus 44/50 for the teacher. Online exploratory training completed 3/500 episodes (0.6%). Do not describe the student as retaining teacher behavior or as ready for edge deployment.

## Safe presentation wording

“The prototype performs simulated target reaching with an IK-assisted DQN and a simplified mechanical-work energy estimate. In the preserved matched evaluation the teacher reached 88%, compared with 90% for fixed-speed IK. The teacher used 10.82% more simulated energy on average among shared successful cases, so the energy-saving objective is not achieved yet. The original student completed 12%; improving distillation and validating the energy model are ongoing.”

## Scope and uncertainty

These are target-reaching simulation results, not a pick-and-place demonstration or electrical-meter reading. The steel dataset is industrial context, not robot telemetry. V1.1 smoke checks are implementation checks and must not be mixed with the historical 50-seed results. Exact source rows and calculations are in `results/colab_run_2026-11-24/` and `PROJECT_AUDIT.md`.
