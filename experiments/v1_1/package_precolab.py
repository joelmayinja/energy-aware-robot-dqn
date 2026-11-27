"""Package completed historic/local smoke evidence without implying a full run."""
from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "FINAL_RESEARCH_RESULTS.zip"
files = [
    "FINAL_RESEARCH_TRAINING.ipynb", "PROJECT_AUDIT.md", "FINAL_RESEARCH_REPORT.md",
    "OBJECTIVES_STATUS.md", "POSTER_RESULTS.md", "DEFENCE_QUESTIONS.md",
    "DATASET_MANIFEST.json", "DATASET_ANALYSIS.md", "DATASET_PROVENANCE.md",
    "LITERATURE_IMPLEMENTATION_MATRIX.md", "APRIL_RESEARCH_ROADMAP.md", "CHANGELOG_V1_1.md",
    "experiments/v1_1/environment_v1_1.py", "experiments/v1_1/pipeline.py",
    "experiments/v1_1/analyze_steel.py", "experiments/v1_1/download_robot_datasets.py",
    "experiments/v1_1/colab_source_v1_1.zip", "experiments/v1_1/colab_source_v1_1.sha256",
    "experiments/v1_1/runs/2026-11-26/dataset_analysis/steel_analysis_summary.json",
    "experiments/v1_1/runs/2026-11-26/dataset_analysis/steel_hourly_power_summary.csv",
    "experiments/v1_1/runs/2026-11-26/dataset_analysis/steel_top_1pct_intervals.csv",
    "experiments/v1_1/runs/2026-11-26/dataset_analysis/steel_demand_analysis.png",
    "experiments/v1_1/runs/2026-11-26/dataset_download_attempt.json",
    "data/raw/high_payload_robot/readme.txt",
    "experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/smoke_demonstrations.npz",
    "experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/smoke_evaluation.csv",
    "experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/student_training_history.csv",
    "experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/student_best.pt",
    "experiments/v1_1/runs/2026-11-26/smoke_local_v1_1/checkpoint_roundtrip.pt",
    "results/colab_run_2026-11-24/README.md",
    "results/colab_run_2026-11-24/verified_evaluation_summary.csv",
    "results/colab_run_2026-11-24/paired_evaluation.csv",
    "results/colab_run_2026-11-24/matched_seed_evaluation.csv",
    "results/colab_run_2026-11-24/training_history.csv",
    "results/colab_run_2026-11-24/distillation_losses.csv",
    "results/colab_run_2026-11-24/models/teacher.pt",
    "results/colab_run_2026-11-24/models/student.pt",
]
status = """# Package status\n\nThis archive contains verified historical results and local V1.1 smoke evidence as of 2026-11-26. It is a preparation package, **not** the completed V1.1 research run. The notebook still needs to run in Colab in deadline or full mode before fresh performance claims can be added. The local smoke output is deliberately labelled smoke and must not be presented as research results.\n"""
with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.writestr("RUN_STATUS.md", status)
    missing = []
    for rel in files:
        path = ROOT / rel
        if path.is_file():
            zf.write(path, rel)
        else:
            missing.append(rel)
    if missing:
        zf.writestr("PACKAGE_MISSING_FILES.txt", "\n".join(missing) + "\n")
print(f"Wrote {DEST} ({DEST.stat().st_size} bytes)")
if missing:
    print("Missing expected files:", *missing, sep="\n- ")
