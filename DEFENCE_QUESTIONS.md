# Technical defence questions and accurate answers

These answers describe the implementation and preserved evidence. Update the V1.1 answers only after the full notebook run has produced actual outputs.

1. **What problem does the prototype address?**  
   It tests whether a DQN can reach simulated end-effector targets while lowering a modeled energy measure. A tariff or actual monetary cost is not currently calculated.

2. **Does it perform pick-and-place?**  
   No. It performs two-phase target reaching. There are no objects, grasp state, gripper actuation, or placement test.

3. **Which robot is simulated?**  
   PyBullet's KUKA iiwa URDF. Seven physical joints are present; six are controlled and the seventh is held fixed.

4. **What does the 23-value observation contain?**  
   Six joint positions, six joint velocities, three end-effector coordinates, three target coordinates, task phase, normalized buffer level, normalized last grid draw, target distance, and episode fraction.

5. **What does one DQN action mean?**  
   It selects one row from an 89-entry table of joint-velocity commands. It does not command torque directly.

6. **How many teacher parameters are there?**  
   94,809 in the verified 23–256–256–89 network.

7. **How many original student parameters are there?**  
   11,481 in the verified 23–64–64–89 network, about 87.9% fewer than the teacher. Fewer parameters do not prove faster full control or retained task performance.

8. **How does experience replay help?**  
   It samples past transitions in mixed batches, reducing the correlation of sequential observations during Q-network updates.

9. **Why use a target network?**  
   It makes TD targets change more slowly than if the same rapidly updated network supplied both current estimates and targets.

10. **What changes in the V1.1 learner?**  
    It selects the next action with the online Q-network and evaluates it with the target network (Double DQN), uses Huber loss, clips gradient norm, and marks finite-horizon truncation as terminal for the target.

11. **Why treat timeout as terminal here?**  
    The episode has a fixed horizon represented in the observation. A time-limit end is therefore treated as the end of this finite-horizon task. The choice is documented and differs from continuing-task bootstrapping.

12. **What is the energy measure in V1.0?**  
    Per joint, it estimates signed mechanical work from torque multiplied by angular velocity and a timestep. Positive work is counted as draw; negative work is treated as potentially recoverable.

13. **Is that electrical energy from the wall?**  
    No. It is a mechanical-work estimate from the simulator. It omits motor/controller losses and has not been calibrated against a KUKA electricity meter.

14. **Why report both joules and watts in V1.1?**  
    Energy is per interval; power is interval energy divided by the actual PyBullet fixed timestep. A maximum per-step energy should not be mislabeled as power.

15. **What does the buffer represent?**  
    A simplified store that accepts a fraction of simulated braking work in the light phase and supplies some later load in the heavy phase. It is not a physical battery model.

16. **Which buffer efficiencies are used in V1.1?**  
    Charging efficiency remains 0.80 from the prior prototype. Discharge efficiency is set to 0.90 as an explicit assumption. Neither value has been measured; sensitivity analysis is needed.

17. **How does the V1.1 ledger avoid creating energy?**  
    It limits stored energy by capacity, reports overflow as spilled recovery, debits the buffer by delivered energy divided by discharge efficiency, and asserts nonnegative grid draw. Unit tests check those balances.

18. **Why remove buffer bonuses from the experimental reward?**  
    Rewarding storage or discharge directly can encourage cycling for reward even if it does not help the task. V1.1 keeps the buffer's effect in grid draw and does not award buffer movement by itself.

19. **What historical completion results are verified?**  
    Fixed IK completed 45/50 (90%); the selected teacher completed 44/50 (88%); the original student completed 6/50 (12%).

20. **Why are there also 3 successes out of 500?**  
    That is the online exploratory teacher training history. The 44/50 is a separate held-out evaluation of a selected, IK-warm-started teacher checkpoint. They are different phases and must be labeled separately.

21. **Did the historical teacher save energy?**  
    No. Mean paired saving was −10.82% on the 44 scenarios where both teacher and IK succeeded, which means the teacher consumed more of the modeled energy on average.

22. **Why compare on matched seeds?**  
    Each controller sees the same reset and target scenario, so paired differences are less confounded by scenario difficulty.

23. **What do 44 both-success, 5 both-fail, 1 IK-only, 0 DQN-only mean?**  
    They are paired outcomes for teacher versus IK across 50 scenarios. There was one discordant case favoring IK and none favoring DQN; exact McNemar p=1.0, so the sample does not establish a completion advantage.

24. **Why is the student's lower successful-energy mean not evidence of efficiency?**  
    It is conditioned on only six completed runs. A policy that often fails can appear to use less energy among its few successes. Failed-attempt energy and completion must be reported together.

25. **What does teacher-to-student distillation teach?**  
    It stores teacher Q-values and greedy actions on visited observations, then trains the student with temperature-scaled KL divergence, action cross-entropy, and optional Q regression.

26. **Why split the distillation data by trajectory?**  
    Randomly splitting adjacent states can place near-duplicate states from one episode in both sets. Grouping by full trajectory reduces this leakage.

27. **Why can behavior cloning fail after good validation action agreement?**  
    The closed-loop student can make small errors, enter unseen states, and compound those errors. That is why V1.1 evaluates the policy independently in PyBullet and can collect bounded teacher labels on student-visited states.

28. **What is the steel dataset used for?**  
    It describes industrial energy demand patterns. It has no robot joint actions and is not a DQN training-label source or a measure of available power to the robot.

29. **What evidence supports real robot electrical-energy claims?**  
    None currently. The SDU and Mendeley robot datasets were not downloaded in the local session; their raw files must be obtained and inspected before modeling. Cross-platform data would still require a transfer argument before claiming KUKA validation.

30. **Can the project claim edge readiness or Raspberry Pi performance?**  
    Not yet. Parameter count and desktop/Colab neural-network timing are not full control-loop or Pi measurements. ONNX must pass checker/runtime parity on real observations, and a Pi must be benchmarked directly before such a claim.
