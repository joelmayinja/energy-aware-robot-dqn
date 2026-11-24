<div align="center">
  <img src="docs/project-banner.svg" alt="Energy-aware robot control project banner" width="100%">

  <h1>IK-Assisted Energy-Aware Robot Control with Deep Q-Learning</h1>

  <p>
    <strong>A research prototype exploring energy-aware control of a simulated industrial robot.</strong><br>
    Mayinja Joel · BSc Computer Science · Uganda Christian University
  </p>

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10"></a>
    <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Environment-Gymnasium-2E8B57" alt="Gymnasium"></a>
    <a href="https://pybullet.org/"><img src="https://img.shields.io/badge/Physics-PyBullet-087F82" alt="PyBullet"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <img src="https://img.shields.io/badge/Status-Research%20prototype-ED8455" alt="Research prototype">
  </p>

  <p>
    <a href="#project-overview">Overview</a> ·
    <a href="#current-progress">Progress</a> ·
    <a href="#quick-start">Quick start</a> ·
    <a href="#dashboard">Dashboard</a> ·
    <a href="#research-and-presentation">Research guide</a>
  </p>
</div>

---

## Project overview

Industrial robots draw energy during repeated movement. This project uses inverse-kinematics (IK) demonstrations to warm-start a Deep Q-Network (DQN), then fine-tunes the policy in a simulated KUKA iiwa environment. The current prototype demonstrates target reaching; energy reduction remains an open result. Model distillation and edge deployment are exploratory stages.

The work is motivated by the energy and computing constraints that can affect manufacturing SMEs in Uganda. It is an academic research prototype—not a deployed factory controller.

### Research questions

1. Can an IK-assisted DQN preserve target completion while reducing energy against a fixed-speed IK controller?
2. How much teacher performance can a smaller distilled student retain, and can it run on a Raspberry Pi 4?
3. How does the controller perform under simulated brownouts, outages, and solar surplus?

| Stage | Purpose | Status |
|---|---|---|
| Industrial energy data | Explore steel-industry load patterns and motivate task scenarios. | Dataset and exploratory notebook are present; the CSV is not connected to DQN training. |
| Robot simulation | Model a KUKA iiwa arm and compare energy-related behavior. | PyBullet/Gymnasium simulation is operational. |
| IK-assisted DQN | Warm-start from IK demonstrations, fine-tune with DQN, and compare against IK. | Latest held-out evaluation: teacher 44/50 completions; IK baseline 45/50. No energy savings demonstrated. |
| Student and edge deployment | Distill the teacher, export ONNX, and evaluate on Raspberry Pi 4. | Student artifact exports, but latest evaluation reached 6/50 targets (12%); not ready for deployment. |
| Physical-facing demonstration | Communicate the energy-buffer concept with an Arduino/Tinkercad mock-up. | Circuit illustration and simple threshold-based sketch are present; this is not the DQN or industrial robot. |

> **Research integrity:** the proposal's 20% energy-saving, 85% student-retention, and 100 ms Raspberry Pi latency values are targets to test—not achieved results. Dashboard values come from simulation, not physical robot measurements.

## Current progress

- Research proposal (Chapters 1–3) and a systematic literature review draft establish the motivation, research gap, and planned evaluation.
- The literature review screened 195 unique records and included six primary studies; it identifies a gap at the intersection of energy-aware DRL, policy distillation, and low-cost deployment.
- The repository includes a steel-industry CSV with **35,040 quarter-hourly observations from 2018**, plus a notebook for load-pattern exploration.
- The simulation uses a 23-value observation, a 200-step episode, light/heavy phases, regenerative buffer charging, and an IK-guided DQN policy warm-start.
- The dashboard provides Overview, Live monitoring, Data detection, and Model management pages.
- **Latest Colab prototype result:** on the same 50 held-out seeds, the selected IK-assisted DQN reached 44/50 targets (88%) and the fixed-speed IK baseline reached 45/50 (90%). On the 44 matched successful cases, the teacher used 10.8% more energy on average; peak reduction was approximately zero. Target reaching is demonstrated, energy reduction is not.
- **Distillation limitation:** the student reached 6/50 targets (12%). The student is not suitable for the prototype demonstration or edge deployment yet.
- **Training-history caveat:** online exploratory training completed 3/500 targets, while held-out greedy evaluation of the selected warm-start checkpoint completed 44/50. The online learning curve and selected-checkpoint test measure different policies and must be labeled separately.
- **Dashboard comparisons are gated:** energy savings are reported only when both policies complete the same seeded target. Use `evaluation_summary.csv` and `paired_evaluation.csv` for final comparisons, not energy totals from failed episodes.
- The submitted Colab ZIP and a matched-seed re-evaluation are preserved under [`results/colab_run_2026-11-24/`](results/colab_run_2026-11-24/). The selected teacher checkpoint is installed at `models/teacher.pt`; the prior local checkpoint is retained as `models/teacher_pre_ik_assisted.pt`.

## Technical snapshot

| Area | Current prototype |
|---|---|
| Robot | PyBullet KUKA iiwa model: seven physical joints, six controlled, final wrist joint locked |
| Observation | 23 values: joint state, end-effector and target position, phase, buffer, grid draw, distance, and episode progress |
| Actions | 89 discrete choices: zero movement, single-joint velocity commands, and 64 coordinated six-joint directions |
| Teacher network | PyTorch Q-network with two 256-unit hidden layers, IK demonstration warm-start, and DQN fine-tuning |
| Simulation | 200-step episodes with seeded light/heavy targets and a simplified 50 Ws regenerative buffer |
| Current comparison | IK-assisted DQN versus fixed-speed IK on matched held-out seeds |
| Dashboard | Flask API and responsive HTML dashboard |

The action values are **joint-velocity commands**, not torque commands. The simulator estimates positive joint work as energy use and negative work during braking as recoverable energy, with an 80% buffer charging efficiency. This simplified model is not a physical battery or motor validation. The steel-industry CSV is exploratory context; it contains no robot joint telemetry and is not used by training.

> **Method alignment:** some proposal details—such as the richer motor-loss model, additional baselines, grid-state scenarios, student architecture, and Raspberry Pi evaluation—are not yet present in the current runner. See the [progress guide](docs/Progress_and_Presentation_Guide.pdf) for a detailed implementation-versus-plan checklist.

## Repository structure

```text
.
├── data/
│   ├── processed/                 # Generated/derived data (not committed)
│   └── raw/                       # Raw data location
├── docs/
│   ├── Progress_and_Presentation_Guide.pdf
│   └── project-banner.svg
├── models/                        # Generated model weights (not committed)
├── notebooks/
│   ├── dataset_prototype.ipynb
│   └── steel_data.csv
├── src/
│   ├── backend.py                 # Flask API and training controller
│   ├── dashboard.py               # Compatibility entry point for Flask/Codespaces
│   ├── launcher.py                # Starts the complete dashboard application
│   ├── dqn_teacher.py             # DQN agent and replay buffer
│   ├── environment.py             # Gymnasium/PyBullet robot environment
│   └── training.py                # Baseline and teacher training loop
├── tests/
├── tinkercad/                     # Arduino sketch and circuit concept
└── frontend/
    ├── assets/                    # Frontend images
    └── web.html                   # Dashboard frontend and theme toggle
```

## Quick start (Windows)

1. Install **Python 3.10** and make sure the Python launcher (`py`) is available.
2. From Git Bash, enter the project folder and run the Windows launcher:

   ```bash
   cd /c/Users/JOEL/Desktop/DQN/Energy_consumption_DQN
   cmd.exe /c start_dashboard.bat
   ```

   You can also double-click `start_dashboard.bat` in File Explorer.
3. On first run, the launcher creates a small `.venv310` environment and installs Flask only. It starts the dashboard and opens **http://127.0.0.1:5000**.
4. Keep the terminal open while using the dashboard. Press **Ctrl+C** there to stop the server.

The Flask application serves the frontend, assets, training API, and model controls together at `127.0.0.1:5000`. This is one project and one running web application; the frontend and backend remain in separate files so each has a clear responsibility. The dashboard opens in the Codespaces-inspired dark theme and includes a light/dark toggle. Neither Codespaces nor Visual Studio Build Tools is needed to view it.

### Start in Codespaces

From the project root, run:

```bash
.venv/bin/python -m flask --app src.dashboard:app run --host 0.0.0.0 --port 5000 --no-reload
```

Open port 5000 through the VS Code **Ports** tab. The `src.dashboard:app`
compatibility import serves the same Flask app as `src.backend:app`; it is not a
second backend.

The dashboard offers:

- **Overview** — progress cards and episode-energy charts.
- **Live monitoring** — training controls, live grid draw, energy buffer, and arm schematic.
- **Data detection** — simulation signals, threshold alerts, and completed episode results.
- **Model management** — upload a compatible PyTorch teacher `.pt`/`.pth` state dictionary and optionally initialize training from it, along with teacher settings and the student/edge research roadmap.

### Training setup without Visual Studio Build Tools

The launcher intentionally does **not** install the full training stack. The dashboard can open without PyBullet; training controls report clearly when the simulator is unavailable. The official PyPI PyBullet release does not provide a Windows wheel, so installing it with pip on Windows can require a C++ compiler.

For local Windows training without the Visual Studio C++ workload, install [Miniforge](https://github.com/conda-forge/miniforge) and create the optional prebuilt conda-forge environment from the project folder:

```powershell
conda env create --prefix .conda-training --file environment-training.yml
```

The launcher automatically uses `.conda-training` when it exists. Then start it the same way as above. The Conda environment is ignored by Git and can be removed later with `conda env remove --prefix .conda-training`.

If you prefer the existing Python 3.10 environment, installing the full `requirements.txt` with pip on Windows may require C++ Build Tools to build PyBullet. On Linux (including Codespaces), PyBullet's published wheel avoids that Windows compiler requirement.

## Dataset notes

The exploratory notebook is [notebooks/dataset_prototype.ipynb](notebooks/dataset_prototype.ipynb), and its CSV is [notebooks/steel_data.csv](notebooks/steel_data.csv). The supplied file has 35,040 rows, 11 columns, 15-minute observations across 2018, and light/medium/maximum load labels.

This is **industrial energy-consumption data, not robot telemetry**. It supports initial load-pattern exploration; a defensible method is still needed before using it to calibrate simulated task phases or grid scenarios.

## Tests

For the full test suite, first create the optional training environment described above. Then run from the project folder:

```bash
.conda-training\python.exe -m unittest discover -s tests -v
```

Tests cover dashboard endpoints, DQN action selection, environment behavior, and training updates. A passing test suite does not establish that the agent has learned an effective energy-saving policy.

## Customize the dashboard

1. Open the repository folder in VS Code.
2. Press **Ctrl+P** and open `frontend/web.html`.
3. Edit the HTML text and navigation, or adjust the CSS in its `<style>` block.
4. Put your own JPG or PNG in `frontend/assets/`, then reference it as `/assets/your-image.jpg`.
5. Refresh the browser. Flask serves the frontend and API together on port 5000.

The frontend files are in [frontend/](frontend/); the backend API is in [src/backend.py](src/backend.py). The launcher is [src/launcher.py](src/launcher.py).

## Research and presentation

- [Project progress and presentation guide](docs/Progress_and_Presentation_Guide.pdf) — dataset rationale, proposal-to-code status, prototype explanation, talk track, and likely questions.
- [Tinkercad and Arduino notes](tinkercad/README.md) — physical-facing demonstration scope and limitations.
- [Arduino demo sketch](tinkercad/Energy_aware_code_demo.ino)

## Safety and scope

The Arduino/Tinkercad circuit is a low-voltage educational mock-up. It is separate from the Python teacher model. The project is simulation-based; **do not connect this prototype to mains power or an industrial robot**. Hardware deployment requires independent engineering, electrical, and safety validation.

## Author

**Mayinja Joel**<br>
BSc Computer Science · Uganda Christian University<br>
Registration No: S24B23/047
