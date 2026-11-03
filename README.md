# Energy-Aware Robot Control: Lightweight DQN + Knowledge Distillation

> Optimizing Energy Costs of Industrial Machines (Robots) through Lightweight Deep Learning
> Final Year BSc. Computer Science Research Project at Uganda Christian University

**Status:**  In progress (Phase 1: Teacher model development)

## Overview

Industrial robots can waste electrical energy through poorly scheduled movement and
peak grid demand. Energy efficiency is the primary objective; electricity cost is secondary.
This project builds:

1. **A DQN Teacher** trained in PyBullet to control six joints of a KUKA iiwa 7-joint arm,
   with the final wrist joint locked.
2. **A distilled Student model** trained to mimic the Teacher, then exported to ONNX for
   planned Raspberry Pi 4 inference.
3. **Energy evaluation** comparing grid draw and task completion against a fixed-speed
   baseline, with the on-board buffer model charged during light tasks and used during
   heavy tasks.

The 400-step, two-phase simulation has a 23-value observation and a 25-action DQN. The
simulation currently uses discrete joint-velocity commands; these are not torque actions.
Steel-industry load labels are intended for later calibration of the light/heavy phases;
the current simulator uses seeded synthetic task targets. Results in the dashboard are
simulation results, not measurements from a physical robot or Raspberry Pi.

## Project Structure

```
energy-aware-robot-dqn/
├── src/                # Core source code
├── web/                # Local training dashboard
├── data/
│   ├── raw/            # Original datasets (e.g. Steel Industry Energy dataset, Kaggle)
│   └── processed/      # Cleaned/derived data
├── notebooks/          # Jupyter notebooks for exploration & plots
├── models/             # Saved model weights (Teacher / Student)
├── docs/               # Proposal, diagrams, writeups
├── tinkercad/          # Physical demo notes / Arduino sketch
└── tests/              # Small sanity-check scripts
```

## Roadmap

- [x] Repo setup
- [x] Dataset exploration (Steel Industry Energy Consumption, Kaggle)
- [x] Simulated pick-and-place environment (Gym/PyBullet)
- [ ] Run the implemented Teacher DQN training pipeline (500 research episodes)
- [ ] Knowledge distillation → Student model
- [ ] Grid-instability evaluation
- [ ] Tinkercad physical demo
- [ ] Final write-up & poster

## Setup

The project specification targets Python 3.10. On Windows, install Python 3.10 and
Visual Studio Build Tools with **Desktop development with C++** (needed to compile
PyBullet, which does not publish Windows wheels), then create a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run commands with the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Flask training dashboard

From the repository root, start the Flask server:

```powershell
python -m src.dashboard
```

Open **http://127.0.0.1:5000** in your browser. The page is served by the Python
application; opening `web/index.html` directly as a file will not connect to training.
Choose 1–3 episodes for a first run, then click **Start teacher training**. The project
plan is 500 episodes; use that after the short run works. For each episode the server
first measures a constant maximum-speed baseline, then trains the DQN on the same seeded
KUKA task. Pause, resume, stop, and playback-speed controls are available on the page.
The terminal shows episode summaries. Teacher weights, the replay states/actions used
for planned distillation, and an append-only `training_history.csv` are saved locally;
generated model/data outputs are excluded from Git.

The live chart compares actual per-step grid draw for the baseline and Teacher. Episode
results compare peak grid draw, total grid energy, and task completion. Reward is the
agent's learning signal, epsilon is its random-action probability, and loss is its latest
Q-value prediction error. The arm image is a schematic driven by the simulated joint
angles, not a PyBullet camera render. A short run verifies the workflow, but does not
establish that the agent has learned a good policy.

## Personalize the dashboard

Open the project folder in VS Code, press **Ctrl+P**, and enter `web/index.html`.
The visible navigation labels and dashboard text are HTML in that file; the colors,
spacing, and responsive layout are in its `<style>` block. To use your own photo or
diagram, put the image in `web/assets/` and reference it from the HTML with a path such
as `/assets/robot-demo.jpg`. The Flask dashboard serves files from this assets folder.
For a static preview, restart `python -m http.server 8000 --directory web` and use
`http://127.0.0.1:8000`.

If training reports a missing module, finish installing `requirements.txt` inside the
same `.venv` and restart the server. If PyBullet reports a compiler error, verify Visual
Studio Build Tools' C++ workload is installed, reopen PowerShell, and retry installation.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Tests exercise the PyBullet environment, DQN action selection, and a replay-buffer
learning update. The first training episodes may not show a loss until the replay buffer
has collected enough experience.

## Author

Mayinja Joel  Reg No: S24B23/047  BSc. Computer Science, Uganda Christian University
