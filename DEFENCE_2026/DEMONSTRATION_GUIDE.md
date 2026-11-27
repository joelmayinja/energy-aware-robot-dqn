# Local PyBullet dashboard and video demonstration

The dashboard reuses the V1.1 KUKA environment and recovered checkpoints. The 50-scenario evaluation remains separate. The documented selected scenario is seed `140000`; IK, original teacher and improved student reach on this seed, while the original student does not. That failure is shown honestly.

## Prepare

From the repository root, run:

```powershell
& .\.conda-training\python.exe -m experiments.v1_1.export_defence_models
```

This creates separate model copies in `DEFENCE_2026\models`. It does not overwrite `models\teacher.pt` or historical result files. It also rebuilds dynamic-batch ONNX files and runs checker/runtime parity tests. Current outcomes are recorded in `DEFENCE_2026\models\model_provenance.json`.

## Launch the live dashboard and 3D simulation

```powershell
& .\.conda-training\python.exe .\defence_demo.py --dashboard --seed 140000
```

This opens the Tk control panel and a separate PyBullet 3D window. Select one of four controllers, change the integer seed, then use **Start / Resume**, **Pause**, and **Reset scenario**. Enable **Record this run to MP4** before starting to save the active run. Cyan marks the light target; orange marks the heavy target. The dashboard reports phase, success, distance, simulated time, energy estimate, peak power, model parameters and measured per-action decision latency. The latency is local per-step timing, not a Raspberry Pi or end-to-end latency benchmark.

Use the same seed to compare controllers fairly. The terminal commands use the internal controller IDs:

```powershell
& .\.conda-training\python.exe .\defence_demo.py --controller ik --seed 140000
& .\.conda-training\python.exe .\defence_demo.py --controller teacher_original --seed 140000
& .\.conda-training\python.exe .\defence_demo.py --controller student_original --seed 140000
& .\.conda-training\python.exe .\defence_demo.py --controller student_improved --seed 140000
```

Choose any integer seed with `--seed`. Runs append actual measurements to `DEFENCE_2026\demo_visual_runs.csv`.

## Export individual MP4s

From PowerShell at the repository root:

```powershell
& .\.conda-training\python.exe .\defence_demo.py --controller ik --seed 140000 --headless --record-video
& .\.conda-training\python.exe .\defence_demo.py --controller teacher_original --seed 140000 --headless --record-video
& .\.conda-training\python.exe .\defence_demo.py --controller student_original --seed 140000 --headless --record-video
& .\.conda-training\python.exe .\defence_demo.py --controller student_improved --seed 140000 --headless --record-video
```

Videos are written under `DEFENCE_2026\videos\`. They use actual PyBullet camera renders at 960×540, 30 fps. One real simulator pose is captured every four physics steps and repeated to slow playback; no motion or metric values are synthesized. Overlays include the controller, seed, phase, result, distance, simulated time, energy, peak power, decision latency and parameter count. All four videos have been generated for seed 140000.

## Create the 2×2 comparison video

```powershell
& .\.conda-training\python.exe .\defence_demo.py --compare --seed 140000 --max-steps 200 --fps 30
```

This records four sequential PyBullet episodes, reinitializing the same seed and task for each, then composes them into a 1920×1080 comparison MP4. The panels are synchronized by episode frame index; they are not simultaneous robot instances. A JSON manifest stores each run's measurements and video paths. The comparison video has been generated.

## Headless simulation without recording

```powershell
& .\.conda-training\python.exe .\defence_demo.py --controller student_improved --seed 140000 --headless --max-steps 20
```

If needed, install dependencies into the project environment with `& .\.conda-training\python.exe -m pip install -r .\DEFENCE_2026\requirements-defence.txt`.

The demonstration reports simulated mechanical-work energy, not measured electrical consumption. Do not present a single run's outcome as the overall evaluation result.

## Verification record

The live GUI path was run for the selected student on seed 140000. It completed that one scenario in 169 steps (0.704 simulated seconds), with 195.34 J simulated grid energy and 874.18 W peak simulated power. These values are in `demo_runs.csv`; they are one demonstration episode and are not the 50-scenario result. The four-controller headless comparison is in `demo_headless_verification.csv`.
