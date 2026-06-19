# Energy-Aware Robot Control: Lightweight DQN + Knowledge Distillation

> Optimizing Energy Costs of Industrial Machines (Robots) through Lightweight Deep Learning
> Second Year BSc. Computer Science Research Project at Uganda Christian University

**Status:** 🚧 In progress (Phase 1: Teacher model development)

## Overview

Industrial robots are typically "energy-blind";they run at constant speed regardless of
how much power is actually available or how much it costs. This project builds:

1. **A DQN "Teacher"** that learns to control a simulated 6-DoF (Deggress of Freedom) robot arm in a way that
   minimizes energy use (watt-seconds per task) instead of just minimizing time or distance.
2. **A distilled "Student" model**  a much smaller network (<500K parameters) trained to
   mimic the Teacher's decisions, small enough to run in real time on a Raspberry Pi 4.
3. **Grid-instability evaluation**  testing the Student controller against simulated
   Ugandan grid conditions (blackouts, brownouts, solar surplus).

This repo is the code companion to the research proposal in `docs/`.

## Project Structure

```
energy-aware-robot-dqn/
├── src/                # Core source code
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
- [ ] Dataset exploration (Steel Industry Energy Consumption, Kaggle)
- [ ] Simulated pick-and-place environment (Gym/PyBullet)
- [ ] DQN Teacher training
- [ ] Knowledge distillation → Student model
- [ ] Grid-instability evaluation
- [ ] Tinkercad physical demo
- [ ] Final write-up & poster

## Setup

```bash
git clone https://github.com/<joelmayinja>/energy-aware-robot-dqn.git
cd energy-aware-robot-dqn
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Author

Mayinja Joel  Reg No: S24B23/047  BSc. Computer Science, Uganda Christian University


