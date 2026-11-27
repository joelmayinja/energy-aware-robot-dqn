# Research roadmap to April 2027 defence

## Core milestones

| Window | Work | Evidence to retain | Gate |
|---|---|---|---|
| 26 Nov–2 Dec 2026 | Freeze V1.0 evidence; run bounded V1.1 student distillation and paired evaluation; document energy-model assumptions. | Versioned configs, seeds, checkpoint hashes, raw episode CSVs, failures, environment versions. | Do not present a new score until the run completes and is reproducible.
| Dec 2026 | Review and fix V1.1 findings; acquire and inspect robot energy data if licenses/access permit. | Dataset manifests, code review, repeated runs. | Separate observed results from simulated estimates.
| Jan 2027 | Improve calibrated energy model and speed-sensitive controller; compare against fixed, slow, proportional and adaptive IK. | Held-out trajectory/controller tests, uncertainty intervals, ablations. | Keep test scenarios frozen before controller tuning.
| Feb 2027 | Run multiple training seeds and final confirmatory paired scenarios; quantify failures and per-success energy. | Analysis notebook, paired test CSV, bootstrap intervals, seed registry. | No tuning on confirmatory outcomes.
| Mar 2027 | Port student inference to candidate edge board only if hardware is available; measure batch-1 inference and full control-loop latency. | Device ID, clock/power settings, software versions, repeatable benchmark. | No Raspberry Pi readiness claim without actual Pi measurement.
| Apr 2027 | Freeze artifacts, reproduce figures, write limitations and defend scope. | Final report, model/config hashes, demo video or live simulation, backup plan. | No physical industrial deployment claim without access and safety review.

## Core versus optional

Core: valid energy accounting, fair baselines, reproducible multi-seed evaluation, compact student with measured closed-loop behavior, transparent uncertainty and limitations. Optional: IK-guided adaptive-speed DQN, physical robot validation, physical Raspberry Pi demonstration, funding proposal. A Pi driving the simulator can demonstrate edge inference but does not validate real robot energy savings.

## Risks and contingency

- Restricted third-party datasets: retain the documented inability to access; seek author-approved public alternatives and record license.
- Student underperforms: report the failure, compare architecture and distribution-shift diagnostics, do not substitute teacher outputs.
- Energy model lacks electrical calibration: narrow claims to simulated joint mechanical work and label the buffer as an assumed model.
- Hardware unavailable: report desktop/Colab batch-1 neural-network latency only and leave physical deployment as future work.
