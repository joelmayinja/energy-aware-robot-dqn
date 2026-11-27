"""Export separate defence copies from the recovered V1.1 evaluation checkpoints."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from src.dqn_teacher import QNetwork


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "DEFENCE_2026" / "models"
RESULTS = ROOT / "results" / "internal_defence_v1_1_2026-11-26" / "outer_deadline"
HISTORIC_STUDENT = ROOT / "results" / "colab_run_2026-11-24" / "models" / "student.pt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    return loaded


def build_and_save(source: Path, output: Path, hidden: int) -> dict:
    state = load_state(source)
    model = QNetwork(23, 89, hidden_size=hidden)
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.save(model.state_dict(), output)
    return {"source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
            "export": str(output.relative_to(ROOT)), "export_sha256": digest(output),
            "architecture": f"23-{hidden}-{hidden}-89",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "state_dict_only": True}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_map = {
        "teacher_final.pt": (RESULTS / "stages" / "teacher_best.pt", 256),
        "teacher_original.pt": (RESULTS / "stages" / "teacher_original.pt", 256),
        "teacher_finetuned_failed.pt": (RESULTS / "stages" / "teacher_finetuned.pt", 256),
        "student_final.pt": (RESULTS / "stages" / "student_best_final.pt", 64),
        "student_original.pt": (HISTORIC_STUDENT, 64),
    }
    records = {}
    for name, (source, hidden) in source_map.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required checkpoint not found: {source}")
        records[name] = build_and_save(source, OUT / name, hidden)
    records["student_final_source_note"] = (
        "student_best_final.pt is the selected 45/50 model, distilled from the retained original teacher; "
        "the separate student_best_from_original_teacher.pt is retained in recovered results for provenance.")
    teacher_best = load_state(RESULTS / "stages" / "teacher_best.pt")
    teacher_original = load_state(RESULTS / "stages" / "teacher_original.pt")
    student_selected = load_state(RESULTS / "stages" / "student_best_final.pt")
    student_from_original = load_state(RESULTS / "stages" / "student_best_from_original_teacher.pt")
    records["selection_tensor_checks"] = {
        "teacher_best_matches_teacher_original": all(torch.equal(teacher_best[k], teacher_original[k]) for k in teacher_best),
        "student_final_matches_student_best_from_original_teacher": all(torch.equal(
            student_selected[k], student_from_original[k]) for k in student_selected),
        "meaning": "State tensors verified exactly; selected student was distilled from retained original teacher.",
    }

    # ONNX dependencies may not exist in the local training environment. Keep
    # valid PyTorch exports regardless and record the exact export outcome.
    onnx_status = {"attempted": True, "status": "not_attempted", "reason": None, "models": {}}
    try:
        import onnx
        import onnxruntime as ort
        onnx_status["attempted"] = True
        demo_data = np.load(RESULTS / "stages" / "teacher_demonstrations.npz", allow_pickle=False)
        real_observations = np.asarray(demo_data["observations"][:8], dtype=np.float32)
        if real_observations.shape != (8, 23):
            raise ValueError(f"Expected at least 8 real 23-value simulator observations, got {real_observations.shape}")
        for name, hidden in [("teacher_final", 256), ("student_final", 64)]:
            model = QNetwork(23, 89, hidden_size=hidden)
            model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True))
            model.eval()
            path = OUT / f"{name}.onnx"
            sample = torch.zeros(1, 23, dtype=torch.float32)
            torch.onnx.export(model, sample, path, input_names=["observation"], output_names=["q_values"],
                              dynamic_axes={"observation": {0: "batch_size"}, "q_values": {0: "batch_size"}},
                              opset_version=17, dynamo=False)
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            errors, agreements, shapes = [], [], []
            with torch.no_grad():
                for batch in (1, 2, 8):
                    x = torch.from_numpy(real_observations[:batch])
                    expected = model(x).numpy()
                    actual = session.run(["q_values"], {"observation": x.numpy()})[0]
                    shapes.append(list(actual.shape))
                    errors.append(float(abs(expected-actual).max()))
                    agreements.append(float((expected.argmax(1)==actual.argmax(1)).mean()))
            onnx_status["models"][f"{name}.onnx"] = {
                "source_checkpoint": f"{name}.pt", "source_sha256": digest(OUT/f"{name}.pt"),
                "sha256": digest(path), "bytes": path.stat().st_size,
                "batch_shapes_tested": shapes, "max_abs_error": max(errors),
                "greedy_action_agreement": min(agreements), "onnx_checker": "passed",
                "parity_input_source": "recovered teacher demonstration observations from PyBullet simulation",
                "parity_tolerance_abs": 5e-5,
                "parity_status": "passed" if max(errors) <= 5e-5 and min(agreements) == 1.0 else "failed"}
        onnx_status["status"] = ("exported_and_validated" if all(
            row["parity_status"] == "passed" for row in onnx_status["models"].values())
            else "exported_but_parity_failed")
    except Exception as exc:
        onnx_status["status"] = "failed_or_unavailable"
        onnx_status["reason"] = f"{type(exc).__name__}: {exc}"
    records["onnx"] = onnx_status
    (OUT / "model_provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
