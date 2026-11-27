"""Flask dashboard for monitoring the KUKA energy-aware DQN project."""

from importlib.util import find_spec
import csv
import io
import json
import os
import pickle
from pathlib import Path, PureWindowsPath
import tempfile
from threading import Event, Lock, Thread

from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from src.config import HOST, PORT
from src.training import train


FRONTEND_ROOT = Path(__file__).resolve().parent.parent / "frontend"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "colab_run_2026-11-24"
DEFENCE_ROOT = FRONTEND_ROOT.parent / "DEFENCE_2026"
RECOVERED_ROOT = RESULTS_ROOT.parent / "internal_defence_v1_1_2026-11-26"
RECORDING_ID = "seed140000_20261127_002702"
RECORDING_MANIFEST = f"comparison_2x2_{RECORDING_ID}.json"
RECORDING_FILES = {
    f"{controller}_{RECORDING_ID}.mp4"
    for controller in ("ik", "teacher_original", "student_original", "student_improved", "comparison_2x2")
}
EVIDENCE_FILES = {
    name: DEFENCE_ROOT / name for name in (
        "corrected_policy_summary.csv", "corrected_paired_comparisons.csv",
        "verified_model_benchmark.csv", "demo_visual_runs.csv", "independent_verification.json",
        "student_training_history_source_copy.csv", "teacher_finetuning_history_source_copy.csv",
        "VERIFIED_RESULTS.md", "RESEARCH_METHODOLOGY.md",
    )
}
EVIDENCE_FILES.update({
    "final_episode_metrics.csv": RECOVERED_ROOT / "outer_deadline" / "final_episode_metrics.csv",
    "RECOVERED_RESULTS_MANIFEST.json": RECOVERED_ROOT / "RECOVERED_RESULTS_MANIFEST.json",
    RECORDING_MANIFEST: DEFENCE_ROOT / "videos" / RECORDING_MANIFEST,
})
EVIDENCE_FIGURES = {
    "completion_rates": "Completion rates and Wilson 95% intervals",
    "energy_success_and_failure": "Simulated grid energy by task outcome",
    "aggregate_energy_per_completion": "Total simulated grid energy per completion",
    "peak_power": "Mean episode peak simulated power",
    "paired_energy_saving": "Paired energy saving and bootstrap 95% intervals",
    "paired_energy_saving_distribution": "Distribution of energy saving on shared successes",
    "model_size_and_latency": "Model size and neural-network forward-pass latency",
    "parameter_reduction": "Student parameter reduction",
    "student_distillation_curves": "Student distillation and held-out action agreement",
    "teacher_finetuning": "Teacher fine-tuning history",
}
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "teacher.pt"
TRAINING_CHECKPOINT_PATH = MODEL_PATH.with_name("teacher_training_checkpoint.pt")
MAX_EPISODES_PER_RUN = 500
MAX_MODEL_UPLOAD_BYTES = 16 * 1024 * 1024


def pybullet_available():
    return find_spec("pybullet") is not None


class TrainingController:
    """Runs the DQN in a background thread and shares metrics with Flask."""

    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.pause_event = Event()
        self.pause_event.set()
        self.speed = 3
        self.thread = None
        self.state = self._empty_state()

    @staticmethod
    def _empty_state():
        return {
            "status": "idle",
            "stage": "waiting",
            "episodes_requested": 0,
            "episode": 0,
            "episode_step": 0,
            "max_steps": 400,
            "reward": 0.0,
            "agent_grid_draw": 0.0,
            "baseline_grid_draw": 0.0,
            "total_grid_draw": 0.0,
            "baseline_total_grid_draw": 0.0,
            "buffer_level": 0.0,
            "buffer_used": 0.0,
            "distance": 0.0,
            "phase": 0,
            "epsilon": 0.1,
            "loss": None,
            "joint_positions": [0.0] * 6,
            "series": [],
            "series_episode": 0,
            "history": [],
            "evaluation_summary": [],
            "run_id": 0,
            "speed": 3,
            "error": None,
        }

    def snapshot(self):
        with self.lock:
            return {
                **self.state,
                "series": [dict(point) for point in self.state["series"]],
                "history": [dict(item) for item in self.state["history"]],
            }

    def start(self, episodes, resume_model=False):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            run_id = self.state["run_id"] + 1
            self.stop_event.clear()
            self.pause_event.set()
            self.state = self._empty_state()
            self.state.update(
                status="running",
                stage="warmup",
                episodes_requested=episodes,
                speed=self.speed,
                run_id=run_id,
            )
            self.thread = Thread(
                target=self._run, args=(episodes, resume_model), daemon=True
            )
            self.thread.start()
            return True

    def stop(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                return False
            self.stop_event.set()
            self.pause_event.set()
            return True

    def pause(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                return False
            self.pause_event.clear()
            self.state["status"] = "paused"
            return True

    def resume(self):
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                return False
            self.pause_event.set()
            self.state["status"] = "running"
            return True

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed
            self.state["speed"] = speed

    def reset(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            self.state = self._empty_state()
            self.state["speed"] = self.speed
            return True

    def _update_step(self, metrics):
        with self.lock:
            self.state.update(metrics)
            self.state["status"] = "paused" if not self.pause_event.is_set() else "running"
            if metrics.get("episode") != self.state["series_episode"]:
                self.state["series"] = []
                self.state["series_episode"] = metrics.get("episode", 0)
            step_index = metrics["episode_step"] - 1
            while len(self.state["series"]) <= step_index:
                self.state["series"].append(
                    {"step": len(self.state["series"]) + 1, "agent": None, "baseline": None}
                )
            point = self.state["series"][step_index]
            point["baseline"] = metrics["baseline_grid_draw"]
            if metrics.get("agent_grid_draw") is not None:
                point["agent"] = metrics["agent_grid_draw"]

    def _record_episode(self, result):
        with self.lock:
            self.state["history"].append(result)
            self.state["history"] = self.state["history"][-500:]

    def _record_evaluation(self, summary):
        with self.lock:
            self.state["evaluation_summary"] = [dict(row) for row in summary]

    def _run(self, episodes, resume_model=False):
        try:
            train(
                episodes,
                self.stop_event,
                self.pause_event,
                lambda: self.speed,
                self._update_step,
                self._record_episode,
                resume_model=resume_model,
                on_evaluation=self._record_evaluation,
            )
            with self.lock:
                self.state["status"] = "stopped" if self.stop_event.is_set() else "completed"
        except Exception as error:
            with self.lock:
                self.state["status"] = "error"
                self.state["error"] = f"{type(error).__name__}: {error}"
            print(f"Training failed: {type(error).__name__}: {error}")


controller = TrainingController()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_UPLOAD_BYTES


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(error="Model uploads must be 16 MB or smaller."), 413


@app.get("/")
@app.get("/login")
def frontend():
    return send_from_directory(FRONTEND_ROOT, "login.html")


@app.get("/app")
@app.get("/dashboard")
def dashboard_frontend():
    return send_from_directory(FRONTEND_ROOT, "web.html")


@app.get("/assets/<path:filename>")
def frontend_asset(filename):
    return send_from_directory(FRONTEND_ROOT / "assets", filename)


@app.get("/api/simulation-evidence")
def simulation_evidence():
    """Read the defence evidence from the original local artifacts."""
    manifest_path = EVIDENCE_FILES[RECORDING_MANIFEST]
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        tables = {}
        for key, filename in (
            ("policies", "corrected_policy_summary.csv"),
            ("paired", "corrected_paired_comparisons.csv"),
            ("benchmark", "verified_model_benchmark.csv"),
        ):
            with EVIDENCE_FILES[filename].open(encoding="utf-8-sig", newline="") as source:
                tables[key] = list(csv.DictReader(source))
    except FileNotFoundError:
        return jsonify(error="The local defence evidence files are unavailable."), 404

    return jsonify(
        **tables,
        seed=manifest["seed"],
        max_steps=manifest["max_steps"],
        recordings=[
            {**measurements, "file": PureWindowsPath(path).name}
            for path, measurements in zip(manifest["individual_videos"], manifest["results"], strict=True)
        ],
        comparison=PureWindowsPath(manifest["comparison_video"]).name,
        figures=[{"file": f"{name}.png", "label": label} for name, label in EVIDENCE_FIGURES.items()],
        downloads=list(EVIDENCE_FILES),
    )


@app.get("/evidence/videos/<filename>")
def evidence_video(filename):
    if filename not in RECORDING_FILES:
        abort(404)
    return send_from_directory(DEFENCE_ROOT / "videos", filename, mimetype="video/mp4", conditional=True)


@app.get("/evidence/figures/<filename>")
def evidence_figure(filename):
    if not filename.endswith(".png") or filename[:-4] not in EVIDENCE_FIGURES:
        abort(404)
    return send_from_directory(DEFENCE_ROOT / "figures", filename)


@app.get("/evidence/downloads/<filename>")
def evidence_download(filename):
    if filename not in EVIDENCE_FILES:
        abort(404)
    path = EVIDENCE_FILES[filename]
    return send_from_directory(path.parent, path.name, as_attachment=True)


@app.get("/api/status")
def training_status():
    return jsonify(
        **controller.snapshot(),
        pybullet_available=pybullet_available(),
        model_available=MODEL_PATH.is_file(),
        checkpoint_available=TRAINING_CHECKPOINT_PATH.is_file(),
    )


@app.get("/api/research-results")
def research_results():
    """Return the checked-in matched-seed evaluation for the research page."""
    summary_path = RESULTS_ROOT / "verified_evaluation_summary.csv"
    paired_path = RESULTS_ROOT / "paired_evaluation.csv"
    if not summary_path.is_file() or not paired_path.is_file():
        return jsonify(error="The verified evaluation files are not available."), 404

    with summary_path.open(newline="", encoding="utf-8") as file:
        summary = list(csv.DictReader(file))
    with paired_path.open(newline="", encoding="utf-8") as file:
        paired = list(csv.DictReader(file))

    def as_bool(value):
        return str(value).strip().lower() in {"true", "1", "yes"}

    def as_number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    summary_fields = (
        "episodes",
        "completed",
        "completion_rate_pct",
        "mean_energy_all_ws",
        "mean_energy_completed_ws",
        "mean_peak_completed_ws",
        "mean_final_distance_m",
    )
    for row in summary:
        for field in summary_fields:
            number = as_number(row.get(field))
            row[field] = int(number) if field in {"episodes", "completed"} and number is not None else number

    for row in paired:
        row["seed"] = int(row["seed"])
        for field in ("ik_task_done", "dqn_task_done", "comparison_valid"):
            row[field] = as_bool(row[field])
        for field in (
            "ik_energy_ws",
            "dqn_energy_ws",
            "energy_savings_percent",
            "ik_peak_ws",
            "dqn_peak_ws",
            "peak_reduction_percent",
        ):
            row[field] = as_number(row.get(field))

    comparable = [row for row in paired if row["comparison_valid"]]

    def mean_field(field):
        values = [row[field] for row in comparable if row[field] is not None]
        return round(sum(values) / len(values), 2) if values else None

    return jsonify(
        evaluation_date="2026-11-24",
        seeds_tested=len(paired),
        matched_successes=len(comparable),
        mean_energy_savings_pct=mean_field("energy_savings_percent"),
        mean_peak_reduction_pct=mean_field("peak_reduction_percent"),
        summary=summary,
        paired=paired,
    )


@app.get("/api/health")
def dashboard_health():
    return jsonify(application="energy-consumption-dqn")


@app.post("/api/train")
def start_training():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400
    episodes = payload.get("episodes", 3)
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        return jsonify(error="Episodes must be a whole number."), 400
    if not 1 <= episodes <= MAX_EPISODES_PER_RUN:
        return jsonify(error=f"Choose between 1 and {MAX_EPISODES_PER_RUN} episodes."), 400
    resume_model = payload.get("resume_model", False)
    if not isinstance(resume_model, bool):
        return jsonify(error="resume_model must be true or false."), 400
    if resume_model and not MODEL_PATH.is_file() and not TRAINING_CHECKPOINT_PATH.is_file():
        return jsonify(error="No saved teacher model is available to continue from."), 409
    if not pybullet_available():
        return jsonify(error="PyBullet is unavailable. Follow the optional training setup in README.md, then restart the dashboard."), 503
    if not controller.start(episodes, resume_model=resume_model):
        return jsonify(error="A training run is already active."), 409
    return jsonify(
        status="running",
        episodes_requested=episodes,
        resumed_model=resume_model,
    ), 202


@app.post("/api/model/upload")
def upload_model():
    if controller.thread is not None and controller.thread.is_alive():
        return jsonify(error="Stop training before uploading a model."), 409
    uploaded_file = request.files.get("model")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Choose a .pt or .pth model file to upload."), 400
    if Path(uploaded_file.filename).suffix.lower() not in {".pt", ".pth"}:
        return jsonify(error="Only PyTorch .pt or .pth teacher weights are supported."), 400

    model_bytes = uploaded_file.read(MAX_MODEL_UPLOAD_BYTES + 1)
    if len(model_bytes) > MAX_MODEL_UPLOAD_BYTES:
        return jsonify(error="Model uploads must be 16 MB or smaller."), 413
    if not model_bytes:
        return jsonify(error="The selected model file is empty."), 400

    try:
        import torch
        from src.dqn_teacher import DQNAgent
    except ImportError as error:
        return jsonify(error=f"PyTorch is unavailable in this dashboard environment: {error}"), 503

    try:
        state_dict = torch.load(
            io.BytesIO(model_bytes), map_location="cpu", weights_only=True
        )
        if not isinstance(state_dict, dict):
            return jsonify(error="The file must contain a teacher state dictionary."), 400
        agent = DQNAgent(state_dim=23, num_joints=6, device="cpu")
        agent.q_network.load_state_dict(state_dict, strict=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        return jsonify(error=f"Invalid teacher weights: {error}"), 400

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent, prefix=".teacher-", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(model_bytes)
        os.replace(temporary_path, MODEL_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return jsonify(model_available=True, message="Teacher weights uploaded and validated.")


@app.post("/api/stop")
def stop_training():
    if not controller.stop():
        return jsonify(error="There is no active training run."), 409
    return jsonify(status="stopping")


@app.post("/api/pause")
def pause_training():
    if not controller.pause():
        return jsonify(error="There is no active training run."), 409
    return jsonify(status="paused")


@app.post("/api/resume")
def resume_training():
    if not controller.resume():
        return jsonify(error="There is no paused training run."), 409
    return jsonify(status="running")


@app.post("/api/speed")
def set_training_speed():
    payload = request.get_json(silent=True)
    speed = payload.get("speed") if isinstance(payload, dict) else None
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 5:
        return jsonify(error="Speed must be a whole number from 1 to 5."), 400
    controller.set_speed(speed)
    return jsonify(speed=speed)


@app.post("/api/reset")
def reset_dashboard():
    if not controller.reset():
        return jsonify(error="Stop training before resetting the dashboard."), 409
    return jsonify(controller.snapshot())


def main():
    print(f"KUKA energy-aware DQN dashboard: http://{HOST}:{PORT}")
    print("Stop this server with Ctrl+C.")
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
