"""Package only the essential internal-defence materials and run evidence."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "INTERNAL_DEFENCE_V1_1.zip"
RESULTS = ROOT / "results" / "internal_defence_v1_1_2026-11-26"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> Path:
    include: dict[str, Path] = {}
    for path in sorted((ROOT / "DEFENCE_2026").rglob("*")):
        if path.is_file():
            include[path.relative_to(ROOT).as_posix()] = path
    explicit = [
        "defence_demo.py",
        "FINAL_RESEARCH_TRAINING.ipynb",
        "CHANGELOG_V1_1.md",
        "experiments/v1_1/environment_v1_1.py",
        "experiments/v1_1/pipeline.py",
        "experiments/v1_1/export_defence_models.py",
        "experiments/v1_1/reconstruct_defence_results.py",
        "experiments/v1_1/integrate_recovered_results.py",
        "experiments/v1_1/defence_visualizer.py",
        "experiments/v1_1/package_internal_defence.py",
        "src/environment.py", "src/dqn_teacher.py",
        "RECOVERED_RESEARCH_RESULTS.zip",
        "results/internal_defence_v1_1_2026-11-26/RECOVERED_RESULTS_MANIFEST.json",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/experiment_config.json",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/seed_manifest.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/final_episode_metrics.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/final_policy_summary.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/paired_energy_saving.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/model_comparison.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/inference_benchmark.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/stages/student_training_history.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/stages/teacher_training_history.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/stages/teacher_validation_results.csv",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/stages/teacher_selection.json",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/stages/demonstration_selection.json",
        "results/internal_defence_v1_1_2026-11-26/outer_deadline/dataset_download_attempt.json",
    ]
    for rel in explicit:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required defence package item missing: {rel}")
        include[rel] = path
    manifest = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
                for name, path in sorted(include.items())]
    with zipfile.ZipFile(DEST, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("PACKAGE_CONTENTS.md", "# Internal defence package contents\n\n"
            "This archive contains internal-defence documentation, corrected tables and figures, "
            "the PyBullet demonstration and its model exports, the unchanged recovered Colab "
            "archive, and selected source evidence. It excludes virtual environments, caches, "
            "and raw industrial datasets.\n")
        for name, path in sorted(include.items()):
            archive.write(path, name)
        archive.writestr("PACKAGE_SHA256_MANIFEST.json", json.dumps({
            "generated_date": "2026-11-26", "files": manifest}, indent=2))
    with zipfile.ZipFile(DEST) as archive:
        bad = archive.testzip()
        if bad:
            raise zipfile.BadZipFile(f"Package CRC failure: {bad}")
    print(f"Created {DEST} ({DEST.stat().st_size} bytes; {len(manifest)} files; CRC passed)")
    return DEST


if __name__ == "__main__":
    main()
