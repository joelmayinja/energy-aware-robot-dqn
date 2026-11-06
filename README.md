<div align="center">
  <img src="docs/project-banner.svg" alt="Energy-aware robot control project banner" width="100%">

  <h1>Energy-Aware Robot Control with Lightweight Deep Q-Learning</h1>

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

Industrial robots can draw substantial energy during repeated movement. This project investigates whether a Deep Q-Network (DQN) can learn to control a simulated robot arm with energy use in mind, while maintaining task performance. The longer-term research plan also studies model distillation for low-cost edge hardware and operation under simulated grid instability.

The work is motivated by the energy and computing constraints that can affect manufacturing SMEs in Uganda. It is an academic research prototype—not a deployed factory controller.

### Research questions

1. How much energy per completed task can a DQN teacher save against a constant-speed controller in simulation?
2. How much teacher performance can a smaller distilled student retain, and can it run on a Raspberry Pi 4?
3. How does the controller perform under simulated brownouts, outages, and solar surplus?

| Stage | Purpose | Status |
|---|---|---|
| Industrial energy data | Explore steel-industry load patterns and motivate task scenarios. | Dataset and exploratory notebook are present; the CSV is not connected to DQN training. |
| Robot simulation | Model a KUKA iiwa arm and compare energy-related behavior. | PyBullet/Gymnasium environment code is present; live execution still needs verification. |
| Teacher DQN | Learn a discrete joint-velocity policy and compare it with a fixed-speed baseline. | Training pipeline is implemented; verified training results are not yet reported. |
| Student and edge deployment | Distill the teacher, export ONNX, and evaluate on Raspberry Pi 4. | Planned research work; not implemented or measured yet. |
| Physical-facing demonstration | Communicate the energy-buffer concept with an Arduino/Tinkercad mock-up. | Circuit illustration and simple threshold-based sketch are present; this is not the DQN or industrial robot. |

> **Research integrity:** the proposal's 20% energy-saving, 85% student-retention, and 100 ms Raspberry Pi latency values are targets to test—not achieved results. Dashboard values come from simulation, not physical robot measurements.

## Current progress

- Research proposal (Chapters 1–3) and a systematic literature review draft establish the motivation, research gap, and planned evaluation.
- The literature review screened 195 unique records and included six primary studies; it identifies a gap at the intersection of energy-aware DRL, policy distillation, and low-cost deployment.
- The repository includes a steel-industry CSV with **35,040 quarter-hourly observations from 2018**, plus a notebook for load-pattern exploration.
- The simulation source defines a 23-value observation, a 400-step episode, light/heavy phases, and an energy-buffer model.
- The dashboard provides Overview, Live monitoring, Data detection, and Model management pages.
- **Environment setup is still a blocker on Windows:** the latest reported PyBullet install stopped because Microsoft C++ Build Tools were missing. The live Flask training flow and model performance therefore remain unverified.

## Technical snapshot

| Area | Current prototype |
|---|---|
| Robot | PyBullet KUKA iiwa model: seven physical joints, six controlled, final wrist joint locked |
| Observation | 23 values: joint state, end-effector and target position, phase, buffer, grid draw, distance, and episode progress |
| Actions | 25 discrete choices: zero movement plus four non-zero velocity levels for each of six joints |
| Teacher network | PyTorch Q-network with two 256-unit hidden layers and a replay buffer |
| Simulation | 400-step episodes with seeded light/heavy targets and a simplified 50 Ws energy buffer |
| Current comparison | Fixed-speed inverse-kinematics baseline versus teacher policy |
| Dashboard | Flask API and responsive HTML dashboard |

The action values are **joint-velocity commands**, not torque commands. The current simulated energy estimate sums `|torque × angular velocity|` over the controlled joints. The steel-industry CSV is exploratory context; it contains no robot joint telemetry and is not currently used by the training loop.

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
│   ├── dashboard.py               # Flask API and local dashboard server
│   ├── dqn_teacher.py             # DQN agent and replay buffer
│   ├── environment.py             # Gymnasium/PyBullet robot environment
│   └── training.py                # Baseline and teacher training loop
├── tests/
├── tinkercad/                     # Arduino sketch and circuit concept
└── web/
    ├── assets/                    # Custom dashboard images
    └── index.html                 # Dashboard interface
```

## Quick start (Windows)

1. Install **Python 3.10** and make sure the Python launcher (`py`) is available. On Windows, PyBullet installation requires **Visual Studio Build Tools** with the **Desktop development with C++** workload (MSVC and a Windows SDK).
2. Open the `Energy_consumption_DQN` folder and double-click **`start_dashboard.bat`**.
3. On first run, the launcher creates `.venv310` and installs the project requirements. It opens the dashboard at **http://127.0.0.1:5000** when ready.
4. Choose **1–3 episodes** for an initial training check. Keep the launcher window open; press **Ctrl+C** there to stop the server.

The dashboard always uses `127.0.0.1:5000`. Starting the launcher again opens the existing project dashboard instead of starting another server. If another application owns that port, startup reports the conflict; it does not switch ports. If the launcher reports missing C++ Build Tools, install the workload above and run it again. If Python 3.10 is missing, install Python 3.10 and retry.

The dashboard offers:

- **Overview** — progress cards and episode-energy charts.
- **Live monitoring** — training controls, live grid draw, energy buffer, and arm schematic.
- **Data detection** — simulation signals, threshold alerts, and completed episode results.
- **Model management** — upload a compatible PyTorch teacher `.pt`/`.pth` state dictionary and optionally initialize training from it, along with teacher settings and the student/edge research roadmap.

> Opening `web/index.html` or running Python's static server shows only a page preview. It does **not** provide the Flask API or run training. Training requires PyBullet; the launcher installs it in the project's Python 3.10 environment.

### Static design preview only (no training)

If dependencies are not installed yet, you can preview the HTML layout without training:

```bash
python -m http.server 8000 --directory web
```

Open **http://127.0.0.1:8000**. The training controls remain disconnected in this mode.

## Dataset notes

The exploratory notebook is [notebooks/dataset_prototype.ipynb](notebooks/dataset_prototype.ipynb), and its CSV is [notebooks/steel_data.csv](notebooks/steel_data.csv). The supplied file has 35,040 rows, 11 columns, 15-minute observations across 2018, and light/medium/maximum load labels.

This is **industrial energy-consumption data, not robot telemetry**. It supports initial load-pattern exploration; a defensible method is still needed before using it to calibrate simulated task phases or grid scenarios.

## Tests

From the project folder:

```bash
.venv310\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover dashboard endpoints, DQN action selection, environment behavior, and training updates. A passing test suite does not establish that the agent has learned an effective energy-saving policy.

## Customize the dashboard

1. Open the repository folder in VS Code.
2. Press **Ctrl+P** and open `web/index.html`.
3. Edit the HTML text and navigation, or adjust the CSS in its `<style>` block.
4. Put your own JPG or PNG in `web/assets/`, then reference it as `/assets/your-image.jpg`.
5. Refresh the browser. Use the Flask server for live assets/API or the static preview for layout only.

The Flask app serves custom files from `web/assets/`. The [dashboard source](web/index.html) and [editing notes](README.md#customize-the-dashboard) are in this repository.

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
