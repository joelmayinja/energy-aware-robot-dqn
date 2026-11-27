# Internal defence questions and concise answers

1. **What is the research problem?** I am investigating whether a lightweight DQN can control a simulated robot while reducing its estimated energy use and still reaching targets.
2. **What does the robot do?** The end effector reaches a close target and then a farther target. It does not grasp or place an object.
3. **Why call it energy-aware?** The observation and reward include a simplified grid-energy estimate, peak penalty, and an energy buffer. The present results show that the learned policies did not save energy.
4. **What is the robot?** A KUKA iiwa model in PyBullet. Six joints are controlled; the seventh wrist joint is locked.
5. **What is the observation size?** 23 values: joint positions and velocities, end-effector and target coordinates, phase, buffer, recent grid draw, distance, and step fraction.
6. **What is the action size?** 89 discrete actions mapped to six joint-velocity commands; this keeps the DQN action space finite.
7. **How many teacher parameters?** 94,809 in a 23-256-256-89 network.
8. **How many student parameters?** 11,481 in a 23-64-64-89 network, 87.89% fewer than the teacher.
9. **How was the student trained?** It distilled teacher Q-values and greedy actions using temperature-scaled KL divergence, cross-entropy, and a small Q-regression term over 100 epochs.
10. **What data trained the student?** Greedy teacher trajectories. The selection record reports 89 successful trajectories of 100; complete trajectories were split for training and validation.
11. **Why filter to successful expert trajectories?** To teach target-reaching behavior instead of making failures part of the expert label set. This changes the training-data distribution and is documented.
12. **How did the new student perform?** It completed 45/50 held-out scenarios versus 7/50 for the original student.
13. **How did it compare with its source teacher?** The retained original teacher completed 46/50; the student completed 45/50, or 97.83% of that observed rate.
14. **Does this prove student and teacher are equivalent?** No. It is a descriptive ratio on 50 scenarios. The paired completion test did not find a difference, but this is not an equivalence study.
15. **What was the fixed IK result?** 44/50, or 88%, on the same held-out seeds.
16. **Did the DQN reduce energy?** No. Mean matched-success saving was −180.58% for the teacher and −157.77% for the student; negative means they used more simulated energy.
17. **What does −157.77% mean?** On average in the 44 shared successful scenarios, `(IK energy − student energy)/IK energy ×100` was −157.77%; it indicates higher student grid-energy estimate, not a 157% physical electricity measurement.
18. **Why only 44 paired successes?** The comparison conditions on scenarios where both fixed IK and the policy completed. Failed attempts are reported separately and included in all-attempt and aggregate energy measures.
19. **What is the student’s paired-energy interval?** Bootstrap percentile 95% CI: −180.56% to −135.70%, from 20,000 resamples of those 44 paired outcomes.
20. **What is the completion interval?** Wilson score 95% interval; for the student it is 78.6%–95.7%.
21. **What did McNemar’s test show for student vs IK?** Paired outcomes were 44 both succeed, 0 IK-only, 1 student-only, 5 both fail; exact two-sided p=1.00. This does not establish a completion advantage.
22. **Why is fine-tuned teacher energy not a positive result?** It completed 0/50. Its low energy is associated with failure; success-conditional energy saving is unavailable.
23. **Why was the original teacher retained?** It scored 19/20 on teacher validation while the fine-tuned policy scored 0/20, and the independent final test also showed 46/50 versus 0/50.
24. **Why did fine-tuning fail?** The data show poor completion after 30 episodes. Likely causes include short training budget, reward/action exploration, and distribution/optimization instability; the run does not isolate one cause.
25. **How is energy calculated?** Simulated joint torque × angular velocity × timestep is accumulated as mechanical work, with assumed buffer conversion efficiencies.
26. **Is that electricity use?** No. It is not measured wall power. It omits motor-drive electrical losses and needs validation against real electrical sensors or robot data.
27. **What does the buffer represent?** A simplified store with a capacity and assumed 80% charging and 90% discharging efficiency. It is a model assumption, not a physical battery test.
28. **Why is fixed IK the comparator?** It was the established conventional controller and completed 44/50. The alternative speed controllers tested here had 0/20 validation completion, so they were not successful comparators.
29. **Are all policies evaluated fairly?** The final controllers share the same 50 scenario seeds, task, environment configuration, and energy ledger. Training and validation seeds are separate from the final seed range.
30. **What is the main statistical limitation?** There is one trained checkpoint per policy; intervals reflect scenario variation, not variation across independent training runs.
31. **What role does the steel dataset play?** It describes industrial electricity demand patterns only. It is not robot telemetry, power supply, or training labels.
32. **Were robot energy datasets used?** No. The SDU measurement archive required login; the Mendeley README was inspected but the multi-gigabyte measurements were not downloaded.
33. **Was ONNX validated?** Yes. Teacher and student passed ONNX checker and runtime comparisons for batch sizes 1, 2 and 8. Maximum absolute difference was below 5e-5 and greedy actions agreed on every tested observation.
34. **Was Raspberry Pi tested?** No. Colab recorded only neural-network batch-1 timing on an x86_64 CUDA host; it is not end-to-end controller latency.
35. **What is the strongest contribution so far?** A reproducible simulation pipeline and a much stronger compact student for target completion, alongside an honest finding that energy optimization remains unsolved.
36. **What would you do next?** Validate energy accounting against electrical measurements, stabilize reward/training, compare against working adaptive IK, run multiple training seeds, and test a separate IK-guided speed-control extension.
37. **What can you claim today?** In this one simulated run, the selected student reached 45/50 targets with 87.89% fewer parameters than the teacher, but it consumed more estimated energy than fixed IK.
38. **What can you not claim?** Real industrial electricity savings, physical robot performance, pick-and-place, Pi deployment, end-to-end latency, or success across independent training seeds.
