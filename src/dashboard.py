"""Flask dashboard for monitoring the KUKA energy-aware DQN project."""

from pathlib import Path
from threading import Event, Lock, Thread

from flask import Flask, jsonify, request, send_from_directory

from src.training import train


WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MAX_EPISODES_PER_RUN = 500


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
            "epsilon": 1.0,
            "loss": None,
            "joint_positions": [0.0] * 6,
            "series": [],
            "series_episode": 0,
            "history": [],
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

    def start(self, episodes):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            run_id = self.state["run_id"] + 1
            self.stop_event.clear()
            self.pause_event.set()
            self.state = self._empty_state()
            self.state.update(
                status="running",
                stage="baseline",
                episodes_requested=episodes,
                speed=self.speed,
                run_id=run_id,
            )
            self.thread = Thread(target=self._run, args=(episodes,), daemon=True)
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

    def _run(self, episodes):
        try:
            train(
                episodes,
                self.stop_event,
                self.pause_event,
                lambda: self.speed,
                self._update_step,
                self._record_episode,
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


@app.get("/")
def dashboard_page():
    return send_from_directory(WEB_ROOT, "index.html")


@app.get("/api/status")
def training_status():
    return jsonify(controller.snapshot())


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
    if not controller.start(episodes):
        return jsonify(error="A training run is already active."), 409
    return jsonify(status="running", episodes_requested=episodes), 202


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
    print("KUKA energy-aware DQN dashboard: http://127.0.0.1:5000")
    print("Stop this server with Ctrl+C.")
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
