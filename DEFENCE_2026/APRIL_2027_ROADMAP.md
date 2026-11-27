# April 2027 research roadmap — separate V2 extension

V1.1 is a completed experimental milestone with a successful compact student for simulated target reaching and a negative energy result. Preserve it and its source data as the baseline. The following are future objectives, not achieved outcomes.

## V2 research question
Can an IK-guided DQN that selects motion-speed multipliers reduce validated energy per completed task while maintaining completion and accuracy against strong conventional controllers?

## Milestones

1. **Validate measurement model:** obtain openly licensed robot electrical measurements or instrument available hardware; synchronize voltage/current and motion timestamps; compare electrical energy with simulated mechanical work.
2. **Repair comparative baseline:** tune fixed and adaptive IK on validation seeds until alternatives complete reliably; pre-register tolerances and freeze settings before test evaluation.
3. **Energy-aware speed policy:** keep IK responsible for motion direction and let a separate policy select a bounded speed multiplier. Treat this as V2, not a replacement for the 89-action V1.1 model.
4. **Reward and stability study:** report reward-component scales, failure penalties, energy per completed task, and sensitivity to buffer efficiency; use multiple independent training seeds.
5. **Confirmatory evaluation:** use matched fresh scenarios, at least 50 seeds per checkpoint and multiple independent training runs; report paired intervals and task failures.
6. **Lightweight deployment:** export dynamic-batch ONNX, validate checker/runtime parity, then benchmark batch-1 and end-to-end control on Raspberry Pi 4. A ≤100 ms decision target is proposed for the deployment test.
7. **Tangible demonstration:** if hardware access is unavailable, investigate a Pi running the policy while controlling the simulator. State that this does not prove physical industrial energy savings.
8. **Economic case:** estimate total energy per completed task and convert to cost only after obtaining a defensible tariff and measured electrical energy.
9. **Funding proposal:** define hardware access, sensor needs, permissions, budget, milestones and reproducibility plan.

The target of at least 20% energy reduction against conventional IK is an April research objective, not a V1.1 result or guarantee.
