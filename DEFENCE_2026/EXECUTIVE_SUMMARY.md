# Executive summary

The project studies whether a lightweight DQN can control a simulated KUKA iiwa arm while reducing estimated energy use. V1.1 preserves the 23-value observation, 89-action teacher, six controlled joints, two-phase task and V1.0 historical artifacts. The simulated task is reaching a target in the heavy phase; it does not grasp or place an object.

The recovered deadline run completed 100 student-distillation epochs using 89 successful teacher demonstrations. The retained original teacher was selected after the fine-tuned teacher scored 0% on final evaluation. On 50 fresh shared seeds (140000–140049), fixed-speed IK completed 44 tasks (88%), the selected teacher 46 (92%), and the selected student 45 (90%). The selected student improved substantially over the old student (7/50 to 45/50) and retained 97.83% of teacher completion performance.

The energy objective was **not** achieved. On 44 shared successful scenarios, mean paired simulated energy saving was −180.58% for the teacher and −157.77% for the student. Negative savings mean greater estimated grid energy than fixed IK. The 95% bootstrap intervals were also entirely below zero. The student is smaller by 87.89%, but the recorded median batch-1 neural-network inference time on Colab's x86_64 CUDA host was 0.137 ms versus 0.130 ms for the teacher. This is neural-network-only timing; it is not end-to-end control timing and not a Raspberry Pi result.

The defensible conclusion is that corrected distillation produced a compact student with strong simulated task completion, while its energy policy remains poor. The result supports continued research; it does not support a claim of energy-cost optimisation success.
