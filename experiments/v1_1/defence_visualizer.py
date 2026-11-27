"""Tkinter control dashboard, PyBullet camera recording, and matched-seed MP4s."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from experiments.v1_1.environment_v1_1 import RobotArmEnvV11
from experiments.v1_1.pipeline import env_action_from_index
from src.dqn_teacher import QNetwork

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "DEFENCE_2026" / "models"
VIDEO_DIR = ROOT / "DEFENCE_2026" / "videos"
CONTROLLERS = ("ik", "teacher_original", "student_original", "student_improved")
LABELS = {
    "ik": "Conventional fixed-speed IK",
    "teacher_original": "Original teacher DQN",
    "student_original": "Original student DQN",
    "student_improved": "Improved lightweight student",
}
MODEL_FILES = {
    "teacher_original": ("teacher_original.pt", 256, 94809),
    "student_original": ("student_original.pt", 64, 11481),
    "student_improved": ("student_final.pt", 64, 11481),
}
WIDTH, HEIGHT = 960, 540
VIDEO_FPS = 30
CAPTURE_INTERVAL = 4  # capture actual simulator state every 4 physics steps; repeat at video FPS


def video_libs():
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        return imageio, Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Video recording needs ImageIO, imageio-ffmpeg and Pillow. Install with: "
                           r".\.conda-training\python.exe -m pip install imageio imageio-ffmpeg pillow") from exc


def load_controller(name: str, device: torch.device = torch.device("cpu")):
    if name == "ik":
        return None, 0
    filename, hidden, expected_parameters = MODEL_FILES[name]
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Recovered checkpoint not found: {path}")
    model = QNetwork(23, 89, hidden_size=hidden).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected_parameters:
        raise ValueError(f"Checkpoint architecture mismatch for {name}: {count} parameters; expected {expected_parameters}")
    return model, count


def overlay_frame(rgb: np.ndarray, metrics: dict, *, compact: bool = False) -> np.ndarray:
    """Draw factual live values onto actual PyBullet camera pixels."""
    _, Image, ImageDraw, ImageFont = video_libs()
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    scale = max(1.0, image.width / WIDTH)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    try:
        font = ImageFont.truetype(str(font_path), int((14 if not compact else 12) * scale)) if font_path.exists() else ImageFont.load_default()
        small = ImageFont.truetype(str(font_path), int(12 * scale)) if font_path.exists() else ImageFont.load_default()
    except OSError:
        font = small = ImageFont.load_default()
    pad = int(14 * scale)
    panel_h = int((138 if not compact else 115) * scale)
    draw.rounded_rectangle((pad, pad, image.width-pad, panel_h), radius=int(12*scale), fill=(8, 15, 27))
    title = f"{metrics['label']}   |   seed {metrics['seed']}   |   SELECTED DEMONSTRATION"
    draw.text((pad*2, pad*1.3), title, fill=(255,255,255), font=font)
    phase_name = "LIGHT TARGET" if int(metrics["phase"]) == 0 else "HEAVY TARGET"
    state = "REACHED" if metrics["success"] else ("RUNNING" if metrics["running"] else "TIME LIMIT / NOT REACHED")
    c1 = (f"Phase: {phase_name}    Status: {state}    Distance: {metrics['distance']:.3f} m    "
          f"Sim time: {metrics['elapsed_s']:.3f} s")
    c2 = (f"Grid energy estimate: {metrics['energy_j']:.2f} J    Peak simulated power: {metrics['peak_w']:.2f} W    "
          f"Decision latency: {metrics['latency_ms']:.3f} ms")
    c3 = f"Model parameters: {metrics['parameters']:,}" if metrics["parameters"] else "Model parameters: N/A (IK controller)"
    draw.text((pad*2, int(pad*3.6)), c1, fill=(200,220,238), font=small)
    draw.text((pad*2, int(pad*5.0)), c2, fill=(200,220,238), font=small)
    draw.text((pad*2, int(pad*6.4)), c3, fill=(200,220,238), font=small)
    footer = "PyBullet V1.1 | simulated joint mechanical-work proxy, not measured electricity | 50-seed evaluation is reported separately"
    draw.rectangle((0, image.height-int(27*scale), image.width, image.height), fill=(8,15,27))
    draw.text((pad, image.height-int(23*scale)), footer, fill=(255,205,125), font=small)
    return np.asarray(image)


class VideoRecorder:
    def __init__(self, path: Path, fps: int = VIDEO_FPS):
        imageio, _, _, _ = video_libs()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=None, pixelformat="yuv420p")
        self.frames = 0

    def append(self, frame: np.ndarray, repeats: int = 1):
        encoded = np.ascontiguousarray(frame, dtype=np.uint8)
        for _ in range(max(1, int(repeats))):
            self.writer.append_data(encoded)
            self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def camera_frame(env: RobotArmEnvV11, metrics: dict) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.28, 0.0, 0.43], distance=1.23, yaw=48,
        pitch=-27, roll=0, upAxisIndex=2)
    projection = p.computeProjectionMatrixFOV(fov=48, aspect=WIDTH/HEIGHT,
                                               nearVal=0.04, farVal=4.0)
    renderer = p.ER_BULLET_HARDWARE_OPENGL if env.render_mode == "human" else p.ER_TINY_RENDERER
    frame = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=projection,
                             renderer=renderer, physicsClientId=env.physics_client)
    rgba = np.asarray(frame[2], dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
    return overlay_frame(rgba[:, :, :3], metrics)


def add_target_markers(env: RobotArmEnvV11):
    client = env.physics_client
    definitions = [
        (env.light_target, [0.15, 0.72, 1.0, 0.9], "LIGHT TARGET"),
        (env.heavy_target, [1.0, 0.39, 0.14, 0.95], "HEAVY TARGET"),
    ]
    for target, rgba, label in definitions:
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.027, rgbaColor=rgba,
                                     physicsClientId=client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=visual, basePosition=target.tolist(),
                          physicsClientId=client)
        if env.render_mode == "human":
            pos = (np.asarray(target, dtype=float) + np.array([0, 0, 0.05])).tolist()
            p.addUserDebugText(label, pos, textColorRGB=rgba[:3], textSize=1.1,
                               lifeTime=0, physicsClientId=client)


def set_camera(env: RobotArmEnvV11):
    if env.render_mode == "human":
        p.resetDebugVisualizerCamera(1.25, 48, -27, [0.28, 0.0, 0.43],
                                     physicsClientId=env.physics_client)


class SimulationSession:
    """One physical simulation world; student inference uses only that student."""
    def __init__(self, controller: str, seed: int, render_mode: str | None,
                 max_steps: int = 200):
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.device = torch.device("cpu")
        self.controller = controller
        self.model, self.parameters = load_controller(controller, self.device)
        self.env = RobotArmEnvV11(render_mode=render_mode)
        self.seed = int(seed)
        self.recorder: VideoRecorder | None = None
        self.reset(self.seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        self.obs, _ = self.env.reset(seed=self.seed)
        add_target_markers(self.env)
        set_camera(self.env)
        self.steps = 0
        self.elapsed_s = 0.0
        self.energy_j = 0.0
        self.mechanical_j = 0.0
        self.peak_w = 0.0
        self.latency_ms = 0.0
        self.success = False
        self.done = False
        self.info = {"task_phase": 0, "distance": float("nan"), "timestep_s": 1/240,
                     "grid_draw": 0.0, "energy_raw": 0.0, "grid_power_w": 0.0}

    def set_controller(self, name: str):
        if name not in CONTROLLERS:
            raise ValueError(f"Unsupported controller {name}")
        self.controller = name
        self.model, self.parameters = load_controller(name, self.device)

    def current_action(self):
        start = time.perf_counter_ns()
        if self.controller == "ik":
            targets = p.calculateInverseKinematics(
                self.env.robot_id, self.env.end_effector_link, self.env.target_position,
                maxNumIterations=100, residualThreshold=1e-4,
                physicsClientId=self.env.physics_client)
            joint_positions = np.asarray([
                p.getJointState(self.env.robot_id, j, physicsClientId=self.env.physics_client)[0]
                for j in self.env.controlled_joints])
            delta = np.asarray(targets[:len(self.env.controlled_joints)]) - joint_positions
            desired = np.where(np.abs(delta) > 1e-3, np.sign(delta)*2.0, 0.0)
            action = np.clip(desired/self.env.MAX_JOINT_VELOCITY, -1, 1).astype(np.float32)
        else:
            with torch.no_grad():
                tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_index = int(self.model(tensor).argmax(1).item())
            action = env_action_from_index(self.env, action_index)
        self.latency_ms = (time.perf_counter_ns()-start)/1e6
        return action

    def step(self):
        if self.done:
            return self.metrics(running=False)
        action = self.current_action()
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        self.steps += 1
        self.elapsed_s += float(self.info["timestep_s"])
        self.energy_j += float(self.info["grid_draw"])
        self.mechanical_j += float(self.info["energy_raw"])
        self.peak_w = max(self.peak_w, float(self.info["grid_power_w"]))
        self.success = self.success or bool(terminated)
        self.done = bool(terminated or truncated or self.steps >= self.max_steps)
        metrics = self.metrics(running=not self.done)
        if self.recorder and (self.steps % CAPTURE_INTERVAL == 0 or self.done):
            remainder = self.steps % CAPTURE_INTERVAL
            repeats = remainder if self.done and remainder else CAPTURE_INTERVAL
            self.recorder.append(camera_frame(self.env, metrics), repeats)
        return metrics

    def metrics(self, running: bool):
        return {"label": LABELS[self.controller], "controller": self.controller,
                "seed": self.seed, "phase": int(self.info.get("task_phase", 0)),
                "success": bool(self.success), "running": bool(running),
                "distance": float(self.info.get("distance", float("nan"))),
                "elapsed_s": self.elapsed_s, "energy_j": self.energy_j,
                "mechanical_j": self.mechanical_j, "peak_w": self.peak_w,
                "latency_ms": self.latency_ms, "parameters": self.parameters,
                "steps": self.steps}

    def start_recording(self, path: Path, fps: int = VIDEO_FPS):
        if self.recorder:
            self.stop_recording()
        self.recorder = VideoRecorder(path, fps)
        self.recorder.append(camera_frame(self.env, self.metrics(running=True)), CAPTURE_INTERVAL)

    def stop_recording(self):
        if self.recorder:
            self.recorder.close()
            self.recorder = None

    def close(self):
        self.stop_recording()
        self.env.close()


def metrics_row(metrics: dict) -> dict:
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "controller": metrics["controller"], "seed": metrics["seed"],
            "demonstration_episodes": 1, "is_full_evaluation": False,
            "success": metrics["success"], "steps": metrics["steps"],
            "completion_time_s": metrics["elapsed_s"],
            "final_position_error_m": metrics["distance"],
            "grid_energy_j": metrics["energy_j"],
            "mechanical_energy_estimate_j": metrics["mechanical_j"],
            "peak_simulated_power_w": metrics["peak_w"],
            "decision_latency_ms_last_step": metrics["latency_ms"],
            "parameter_count": metrics["parameters"],
            "task": "two-phase target reaching; no grasp/place",
            "energy_note": "V1.1 simulated joint mechanical-work ledger; not measured electrical consumption"}


def append_run_log(metrics: dict, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = metrics_row(metrics)
    # Keep older defence_demo.csv intact; use a separate schema with latency.
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_headless_episode(controller: str, seed: int, max_steps: int, video_path: Path,
                         fps: int = VIDEO_FPS, progress=None) -> dict:
    session = SimulationSession(controller, seed, render_mode=None, max_steps=max_steps)
    session.start_recording(video_path, fps)
    try:
        while not session.done:
            metrics = session.step()
            if progress and (metrics["steps"] % 20 == 0 or session.done):
                progress(metrics)
        session.stop_recording()
        result = session.metrics(running=False)
        result["video"] = str(video_path)
        return result
    finally:
        session.close()


def compose_comparison(video_paths: list[Path], output_path: Path, fps: int = VIDEO_FPS):
    """Compose four actual recorded episodes. Runs are sequential, same-seed panels."""
    imageio, Image, ImageDraw, ImageFont = video_libs()
    readers = [imageio.get_reader(str(path)) for path in video_paths]
    last = [None]*len(readers)
    exhausted = [False]*len(readers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, pixelformat="yuv420p")
    try:
        while not all(exhausted):
            cells = []
            for idx, reader in enumerate(readers):
                if not exhausted[idx]:
                    try:
                        last[idx] = reader.get_next_data()
                    except (StopIteration, IndexError):
                        exhausted[idx] = True
                if last[idx] is None:
                    continue
                image = Image.fromarray(last[idx]).resize((WIDTH, HEIGHT))
                cells.append(image)
            if len(cells) != 4:
                break
            canvas = Image.new("RGB", (WIDTH*2, HEIGHT*2), (5,10,18))
            positions = [(0,0),(WIDTH,0),(0,HEIGHT),(WIDTH,HEIGHT)]
            for image, position in zip(cells, positions):
                canvas.paste(image, position)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        for reader in readers:
            reader.close()


def run_comparison(seed: int = 140000, max_steps: int = 200, fps: int = VIDEO_FPS) -> list[dict]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results, paths = [], []
    for name in CONTROLLERS:
        path = VIDEO_DIR / f"{name}_seed{seed}_{stamp}.mp4"
        print(f"Recording {LABELS[name]} on common seed {seed} -> {path}", flush=True)
        def progress(row):
            print(f"  step={row['steps']:3d} phase={row['phase']} distance={row['distance']:.3f}m "
                  f"done={row['success']} energy={row['energy_j']:.2f}J peak={row['peak_w']:.1f}W "
                  f"latency={row['latency_ms']:.3f}ms", flush=True)
        result = run_headless_episode(name, seed, max_steps, path, fps, progress)
        append_run_log(result, ROOT/"DEFENCE_2026"/"demo_visual_runs.csv")
        results.append(result); paths.append(path)
    comparison_path = VIDEO_DIR / f"comparison_2x2_seed{seed}_{stamp}.mp4"
    compose_comparison(paths, comparison_path, fps)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "seed": seed, "max_steps": max_steps, "run_count": len(results),
               "runs_are_sequential_same_seed": True,
               "note": "Each panel is a separately initialized PyBullet run on the same seed. No teacher controls either student.",
               "individual_videos": [str(path.relative_to(ROOT)) for path in paths],
               "comparison_video": str(comparison_path.relative_to(ROOT)),
               "results": [{k:v for k,v in row.items() if k not in {"video"}} for row in results]}
    manifest_path = VIDEO_DIR / f"comparison_2x2_seed{seed}_{stamp}.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Comparison video:", comparison_path)
    print("Run manifest:", manifest_path)
    return results


class DefenceDashboard:
    """A responsive Tk control surface plus the actual PyBullet GUI window."""
    def __init__(self, controller: str, seed: int, max_steps: int,
                 record_on_start: bool = False, fps: int = VIDEO_FPS):
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("Energy-Aware Robot DQN | Defence Demonstration")
        self.root.geometry("720x570")
        self.root.minsize(660, 520)
        self.seed = int(seed); self.max_steps = int(max_steps); self.fps = int(fps)
        self.running = False; self.finished = False; self.session = None
        self.record_var = tk.BooleanVar(value=record_on_start)
        self.controller_var = tk.StringVar(value=LABELS[controller])
        self.seed_var = tk.StringVar(value=str(seed))
        self.status_var = tk.StringVar(value="Ready — selected scenario, not the 50-seed evaluation")
        self.values = {}
        self._style()
        self._build()
        self.session = SimulationSession(controller, seed, render_mode="human", max_steps=max_steps)
        self._update_metrics(self.session.metrics(running=False))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(33, self._tick)

    def _style(self):
        style = self.ttk.Style(self.root)
        try: style.theme_use("clam")
        except self.tk.TclError: pass
        style.configure("TFrame", background="#0e1726")
        style.configure("TLabel", background="#0e1726", foreground="#e6edf6", font=("Segoe UI",10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold",17), foreground="#ffffff")
        style.configure("Card.TLabel", font=("Segoe UI Semibold",11), foreground="#79d9df")
        style.configure("Value.TLabel", font=("Segoe UI Semibold",15), foreground="#ffffff")
        style.configure("Hint.TLabel", foreground="#a6b4c6", font=("Segoe UI",9))
        style.configure("TButton", font=("Segoe UI Semibold",10), padding=(12,7))
        style.configure("Accent.TButton", background="#087f8c", foreground="white")
        style.map("Accent.TButton", background=[("active","#0b9aa8")])

    def _build(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=18); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="ENERGY-AWARE ROBOT CONTROL", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Research simulation • KUKA iiwa • PyBullet V1.1", style="Hint.TLabel").pack(anchor="w", pady=(2,14))
        controls = ttk.Frame(outer); controls.pack(fill="x", pady=(0,12))
        ttk.Label(controls,text="Controller").grid(row=0,column=0,sticky="w")
        self.combo = ttk.Combobox(controls,textvariable=self.controller_var,
                                  values=[LABELS[name] for name in CONTROLLERS],state="readonly",width=34)
        self.combo.grid(row=1,column=0,sticky="ew",padx=(0,12)); self.combo.bind("<<ComboboxSelected>>",self._controller_changed)
        ttk.Label(controls,text="Scenario seed").grid(row=0,column=1,sticky="w")
        seed_entry=ttk.Entry(controls,textvariable=self.seed_var,width=14); seed_entry.grid(row=1,column=1,sticky="ew",padx=(0,12))
        ttk.Label(controls,text="Video FPS").grid(row=0,column=2,sticky="w")
        self.fps_var=tk.StringVar(value=str(self.fps)); ttk.Entry(controls,textvariable=self.fps_var,width=8).grid(row=1,column=2,sticky="ew")
        controls.columnconfigure(0,weight=3); controls.columnconfigure(1,weight=2)
        ttk.Checkbutton(outer,text="Record this run to MP4 (real PyBullet camera frames)",variable=self.record_var).pack(anchor="w",pady=(0,12))
        buttons=ttk.Frame(outer);buttons.pack(fill="x",pady=(0,14))
        ttk.Button(buttons,text="Start / Resume",style="Accent.TButton",command=self.start).pack(side="left",padx=(0,8))
        ttk.Button(buttons,text="Pause",command=self.pause).pack(side="left",padx=(0,8))
        ttk.Button(buttons,text="Reset scenario",command=self.reset).pack(side="left")
        ttk.Label(outer,textvariable=self.status_var,wraplength=660,style="Hint.TLabel").pack(anchor="w",pady=(0,12))
        grid=ttk.Frame(outer);grid.pack(fill="both",expand=True)
        definitions=[("phase","PHASE"),("status","TASK STATUS"),("distance","DISTANCE"),
                     ("time","SIMULATED TIME"),("energy","GRID ENERGY ESTIMATE"),
                     ("peak","PEAK SIMULATED POWER"),("parameters","MODEL PARAMETERS"),
                     ("latency","DECISION LATENCY")]
        for idx,(key,title) in enumerate(definitions):
            r,c=divmod(idx,2)
            card=ttk.Frame(grid,padding=12);card.grid(row=r,column=c,sticky="nsew",padx=(0,8) if c==0 else (8,0),pady=6)
            ttk.Label(card,text=title,style="Card.TLabel").pack(anchor="w")
            var=tk.StringVar(value="—");self.values[key]=var
            ttk.Label(card,textvariable=var,style="Value.TLabel").pack(anchor="w",pady=(4,0))
        grid.columnconfigure(0,weight=1);grid.columnconfigure(1,weight=1)
        grid.rowconfigure(0,weight=1);grid.rowconfigure(1,weight=1);grid.rowconfigure(2,weight=1);grid.rowconfigure(3,weight=1)
        ttk.Label(outer,text="Energy is a simulated joint mechanical-work estimate, not measured electricity. Students act independently.",style="Hint.TLabel",wraplength=670).pack(anchor="w",pady=(10,0))

    def _update_metrics(self, m):
        self.values["phase"].set("Light target" if m["phase"]==0 else "Heavy target")
        self.values["status"].set("RUNNING" if m["running"] else ("COMPLETED" if m["success"] else ("NOT REACHED" if self.finished else "READY")))
        self.values["distance"].set(f"{m['distance']:.3f} m" if np.isfinite(m["distance"]) else "—")
        self.values["time"].set(f"{m['elapsed_s']:.3f} s")
        self.values["energy"].set(f"{m['energy_j']:.2f} J")
        self.values["peak"].set(f"{m['peak_w']:.1f} W")
        self.values["parameters"].set(f"{m['parameters']:,}" if m["parameters"] else "N/A (IK)")
        self.values["latency"].set(f"{m['latency_ms']:.3f} ms/action")

    def _controller_changed(self, _event=None):
        self.pause()
        self.session.stop_recording()
        try:
            seed=int(self.seed_var.get())
            name=next(key for key,value in LABELS.items() if value==self.controller_var.get())
        except (ValueError,StopIteration):
            self.status_var.set("Choose a listed controller and enter an integer scenario seed."); return
        self.session.set_controller(name)
        self.session.reset(seed)
        self.finished=False
        self.status_var.set("Controller changed and same-seed scenario reset. Press Start.")
        self._update_metrics(self.session.metrics(running=False))

    def start(self):
        try:
            selected_seed=int(self.seed_var.get())
            self.fps=max(1,int(self.fps_var.get()))
        except ValueError:
            self.status_var.set("Seed and video FPS must be whole numbers."); return
        if selected_seed != self.session.seed:
            self.running=False; self.finished=False
            self.session.stop_recording(); self.session.reset(selected_seed)
        if self.finished:
            self.reset()
        if self.record_var.get() and self.session.recorder is None:
            VIDEO_DIR.mkdir(parents=True,exist_ok=True)
            stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
            path=VIDEO_DIR/f"{self.session.controller}_seed{self.session.seed}_{stamp}.mp4"
            self.session.start_recording(path,self.fps)
            self.status_var.set(f"Recording actual PyBullet frames: {path.name}")
        self.running=True
        if not self.record_var.get(): self.status_var.set("Running selected demo scenario (not the 50-seed evaluation).")

    def pause(self):
        self.running=False
        self.status_var.set("Paused. Select Start / Resume to continue, or Reset scenario to restart the same seed.")

    def reset(self):
        self.running=False; self.finished=False
        self.session.stop_recording()
        try: seed=int(self.seed_var.get())
        except ValueError:
            self.status_var.set("Seed must be an integer."); return
        self.session.reset(seed)
        try: self.fps=max(1,int(self.fps_var.get()))
        except ValueError: self.fps=VIDEO_FPS
        self.status_var.set("Reset to the selected seed. Press Start to run.")
        self._update_metrics(self.session.metrics(running=False))

    def _tick(self):
        if self.running:
            try:
                m=self.session.step(); self._update_metrics(m)
                if self.session.done:
                    self.running=False; self.finished=True; self.session.stop_recording()
                    append_run_log(m,ROOT/"DEFENCE_2026"/"demo_visual_runs.csv")
                    self.status_var.set(f"Episode ended: {'target reached' if m['success'] else 'target not reached'}. Logged one demonstration episode.")
            except Exception as exc:
                self.running=False; self.session.stop_recording()
                self.status_var.set(f"Simulation error: {exc}")
        self.root.after(33,self._tick)

    def close(self):
        self.running=False
        if self.session:
            self.session.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser=argparse.ArgumentParser(description="Research-grade PyBullet live demo and video recorder; does not train models.")
    parser.add_argument("--dashboard",action="store_true",help="Open the Tk dashboard and PyBullet GUI.")
    parser.add_argument("--controller",choices=CONTROLLERS,default=None)
    parser.add_argument("--seed",type=int,default=140000,help="Scenario seed; use the same seed for all four policies.")
    parser.add_argument("--headless",action="store_true",help="Use PyBullet DIRECT; useful with --record-video.")
    parser.add_argument("--record-video",action="store_true",help="Record actual simulation camera frames to MP4.")
    parser.add_argument("--compare",action="store_true",help="Record all four controllers on the same seed and compose a 2x2 comparison MP4.")
    parser.add_argument("--max-steps",type=int,default=200)
    parser.add_argument("--fps",type=int,default=VIDEO_FPS)
    parser.add_argument("--slowdown",type=float,default=5.0,help="GUI wall-clock slowdown; does not change simulation physics.")
    return parser.parse_args()


def run_cli(args):
    VIDEO_DIR.mkdir(parents=True,exist_ok=True)
    controller=args.controller or "student_improved"
    if args.max_steps<1 or args.fps<1: raise ValueError("--max-steps and --fps must be positive")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    record=args.record_video
    video_path=VIDEO_DIR/f"{controller}_seed{args.seed}_{stamp}.mp4" if record else None
    session=SimulationSession(controller,args.seed,render_mode=None if args.headless else "human",max_steps=args.max_steps)
    if record: session.start_recording(video_path,args.fps)
    print(f"SELECTED DEMONSTRATION — {LABELS[controller]}, seed={args.seed}; not the 50-scenario evaluation")
    print("Targets: light=cyan, heavy=orange. Energy is simulated mechanical-work estimate.")
    try:
        while not session.done:
            m=session.step()
            if not args.headless and args.slowdown>0:
                time.sleep(min(float(session.info["timestep_s"])*args.slowdown,0.05))
            if m["steps"]%20==0 or session.done:
                print(f"step={m['steps']:3d} phase={m['phase']} distance={m['distance']:.3f}m "
                      f"status={'REACHED' if m['success'] else ('RUNNING' if not session.done else 'NOT REACHED')} "
                      f"sim_t={m['elapsed_s']:.3f}s energy={m['energy_j']:.2f}J peak={m['peak_w']:.1f}W "
                      f"decision={m['latency_ms']:.3f}ms params={m['parameters'] or 'N/A'}")
        result=session.metrics(running=False)
        append_run_log(result,ROOT/"DEFENCE_2026"/"demo_visual_runs.csv")
        if record:
            session.stop_recording()
            result["video"] = str(video_path.relative_to(ROOT))
        print("DEMONSTRATION RESULT:",json.dumps(result,indent=2))
        if record: print("MP4:",video_path)
        return result
    finally:
        session.close()


def main():
    args=parse_args()
    if args.compare:
        run_comparison(args.seed,args.max_steps,args.fps)
    elif args.dashboard or (args.controller is None and not args.headless and not args.record_video):
        app=DefenceDashboard("student_improved",args.seed,args.max_steps,fps=args.fps)
        app.run()
    else:
        run_cli(args)


if __name__=="__main__":
    main()
