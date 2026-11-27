"""Build the self-contained Colab notebook and source bundle from this repo."""
from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v1_1"
RUN = EXP / "runs" / "2026-11-26"
RUN.mkdir(parents=True, exist_ok=True)
bundle_path = EXP / "colab_source_v1_1.zip"
included = [
    (ROOT / "src" / "environment.py", "src/environment.py"),
    (ROOT / "src" / "dqn_teacher.py", "src/dqn_teacher.py"),
    (EXP / "__init__.py", "experiments/v1_1/__init__.py"),
    (EXP / "environment_v1_1.py", "experiments/v1_1/environment_v1_1.py"),
    (EXP / "pipeline.py", "experiments/v1_1/pipeline.py"),
    (EXP / "analyze_steel.py", "experiments/v1_1/analyze_steel.py"),
    (EXP / "download_robot_datasets.py", "experiments/v1_1/download_robot_datasets.py"),
    (ROOT / "models" / "teacher.pt", "models/teacher.pt"),
    (ROOT / "results" / "colab_run_2026-11-24" / "models" / "student.pt", "models/student_original.pt"),
    (ROOT / "notebooks" / "steel_data.csv", "notebooks/steel_data.csv"),
    (ROOT / "DATASET_MANIFEST.json", "DATASET_MANIFEST.json"),
    (ROOT / "DATASET_ANALYSIS.md", "DATASET_ANALYSIS.md"),
    (ROOT / "DATASET_PROVENANCE.md", "DATASET_PROVENANCE.md"),
    (ROOT / "LITERATURE_IMPLEMENTATION_MATRIX.md", "LITERATURE_IMPLEMENTATION_MATRIX.md"),
    (ROOT / "APRIL_RESEARCH_ROADMAP.md", "APRIL_RESEARCH_ROADMAP.md"),
    (ROOT / "results" / "colab_run_2026-11-24" / "verified_evaluation_summary.csv", "historical/verified_evaluation_summary.csv"),
    (ROOT / "results" / "colab_run_2026-11-24" / "paired_evaluation.csv", "historical/paired_evaluation.csv"),
    (ROOT / "results" / "colab_run_2026-11-24" / "README.md", "historical/README.md"),
]
with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as zf:
    for path, archive_name in included:
        zf.write(path, archive_name)
bundle_bytes = bundle_path.read_bytes()
bundle_b64 = base64.b64encode(bundle_bytes).decode("ascii")
(EXP / "colab_source_v1_1.sha256").write_text(hashlib.sha256(bundle_bytes).hexdigest() + "  colab_source_v1_1.zip\n", encoding="utf-8")

setup = r'''# Install runtime-only dependencies. PyTorch is supplied by the selected Colab runtime.
import os, sys, subprocess, zipfile, base64, hashlib, json, platform, time
from pathlib import Path
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "pybullet>=3.2", "gymnasium>=0.29", "pandas>=2", "matplotlib>=3.7", "onnx>=1.15", "onnxruntime>=1.16", "onnxscript", "tabulate", "requests"])
import torch, numpy as np, pandas as pd
from google.colab import drive

MODE = "deadline"          # smoke | deadline | full | resume
RESUME_FROM_MODE = "deadline"  # resume a saved deadline or full run
USE_GOOGLE_DRIVE = True    # set False to run in temporary Colab storage
RESUME = True
if USE_GOOGLE_DRIVE:
    drive.mount("/content/drive")
    BASE_OUTPUT = Path("/content/drive/MyDrive/EnergyDQN_V1_1/2026-11-26")
else:
    BASE_OUTPUT = Path("/content/FINAL_RESEARCH_RESULTS")
OUTPUT = BASE_OUTPUT
OUTPUT.mkdir(parents=True, exist_ok=True)

EMBEDDED_BUNDLE_B64 = "__BUNDLE_B64__"
bundle_bytes = base64.b64decode(EMBEDDED_BUNDLE_B64)
BUNDLE = Path("/content/v1_1_source")
BUNDLE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(__import__("io").BytesIO(bundle_bytes)) as zf:
    zf.extractall(BUNDLE)
sys.path.insert(0, str(BUNDLE))
ROOT = BUNDLE
print("Device:", "cuda" if torch.cuda.is_available() else "cpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor())
print("Python", platform.python_version(), "torch", torch.__version__, "numpy", np.__version__)
print("Bundle SHA256:", hashlib.sha256(bundle_bytes).hexdigest())
'''.replace("__BUNDLE_B64__", bundle_b64)

run = r'''import shutil, csv, copy, statistics, zipfile, subprocess, importlib.metadata
from src.dqn_teacher import build_discrete_action_table
from dataclasses import asdict
from experiments.v1_1.environment_v1_1 import RobotArmEnvV11
from experiments.v1_1.pipeline import (
    RunConfig, seed_everything, build_agent, collect_teacher_trajectories,
    collect_corrective_trajectories,
    distill_student, evaluate_policy, evaluate_student_validation,
    successful_trajectory_subset, distillation_data_signature,
    new_student, DQNAgentV11, train_teacher, run_ik_baseline,
)

RUN_MODE = RESUME_FROM_MODE if MODE == "resume" else MODE
if RUN_MODE not in {"smoke", "deadline", "full"}:
    raise ValueError("MODE must be smoke, deadline, full, or resume; resume uses RESUME_FROM_MODE")
if MODE == "resume" and RESUME_FROM_MODE not in {"deadline", "full"}:
    raise ValueError("Resume mode can resume a deadline or full run")
if MODE == "resume": RESUME = True
if RUN_MODE == "smoke":
    cfg = RunConfig(mode=RUN_MODE, seed=20261126, trajectories=3, epochs=2, batch_size=32,
                    patience=2, evaluation_episodes=2, max_steps=12, device="cpu")
    teacher_episodes = 0
elif RUN_MODE == "deadline":
    cfg = RunConfig(mode=RUN_MODE, trajectories=100, epochs=100, batch_size=256,
                    patience=12, evaluation_episodes=50,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    teacher_episodes = 30
else:
    cfg = RunConfig(mode=RUN_MODE, trajectories=400, epochs=500, batch_size=256,
                    patience=30, evaluation_episodes=50,
                    device="cuda" if torch.cuda.is_available() else "cpu")
    teacher_episodes = 500

RobotArmEnvV11.MAX_STEPS = cfg.max_steps
seed_everything(cfg.seed)
OUTPUT = BASE_OUTPUT / RUN_MODE
OUTPUT.mkdir(parents=True, exist_ok=True)
STAGE = OUTPUT / "stages"
STAGE.mkdir(parents=True, exist_ok=True)
config_payload = {
    "config": asdict(cfg), "teacher_finetune_episodes": teacher_episodes,
    "final_test_seed_start": 140000, "student_validation_seeds": list(range(160000, 160005)),
    "training_seed_start": 150000, "energy_model": "joint mechanical work; assumed charge_efficiency=0.80, discharge_efficiency=0.90",
    "task": "two-phase target reaching; no grasp or placement", "torch": torch.__version__,
    "numpy": np.__version__, "device": str(cfg.device), "hardware": platform.processor(),
    "package_versions": {name:importlib.metadata.version(name) for name in ["pybullet","gymnasium","pandas","matplotlib","onnx","onnxruntime"]},
    "observation_dim":23,"action_count":89,
    "action_table":np.asarray(build_discrete_action_table(6),dtype=np.float32).tolist(),
    "teacher_architecture":"23-256-256-89","student_architecture":"23-64-64-89",
    "source_bundle_sha256":hashlib.sha256(bundle_bytes).hexdigest(),
    "teacher_source_sha256":hashlib.sha256((ROOT/"models"/"teacher.pt").read_bytes()).hexdigest(),
    "student_source_sha256":hashlib.sha256((ROOT/"models"/"student_original.pt").read_bytes()).hexdigest(),
    "observation_normalization":{"applied":False,"reason":"preserve the trained teacher input convention and teacher/student observation comparability"}
}
config_path = OUTPUT / "experiment_config.json"
if RESUME and config_path.exists():
    previous = json.loads(config_path.read_text(encoding="utf-8"))
    old_cfg, new_cfg = dict(previous.get("config", {})), dict(config_payload["config"])
    old_cfg.pop("device", None); new_cfg.pop("device", None)
    if (old_cfg != new_cfg or previous.get("teacher_finetune_episodes") != teacher_episodes
            or previous.get("source_bundle_sha256") != config_payload["source_bundle_sha256"]
            or previous.get("teacher_source_sha256") != config_payload["teacher_source_sha256"]
            or previous.get("student_source_sha256") != config_payload["student_source_sha256"]):
        raise RuntimeError("Resume configuration differs from the saved run. Set RESUME=False and use a new output folder.")
elif not RESUME and config_path.exists():
    raise RuntimeError("This output folder already has an experiment config. Enable resume for the exact same run or choose a new date/output folder; existing results will not be overwritten.")
config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
seed_rows=[]
for role,start,count in [
    ("teacher_demonstration_training",150000,cfg.trajectories),
    ("student_validation",160000,5),
    ("teacher_checkpoint_validation",161000,2 if RUN_MODE=="smoke" else 20),
    ("baseline_selection_validation",163000,2 if RUN_MODE=="smoke" else 20),
    ("teacher_online_finetuning",170000,teacher_episodes),
    ("final_heldout_test",180000 if RUN_MODE=="smoke" else 140000,2 if RUN_MODE=="smoke" else 50),
    ("student_corrective_training",190000,1 if RUN_MODE=="smoke" else (20 if RUN_MODE=="deadline" else 50)),
    ("improved_teacher_student_training",200000,min(100,cfg.trajectories) if RUN_MODE!="smoke" else 0),
]:
    seed_rows.extend({"role":role,"seed":start+i,"split":"test" if role=="final_heldout_test" else ("validation" if "validation" in role or "selection" in role else "training"),"status":"configured_not_yet_confirmed"} for i in range(count))
pd.DataFrame(seed_rows).to_csv(OUTPUT/"seed_manifest.csv",index=False)

teacher = build_agent(device=cfg.device)
teacher_weights = torch.load(ROOT / "models" / "teacher.pt", map_location=teacher.device, weights_only=True)
teacher.q_network.load_state_dict(teacher_weights)
teacher.target_network.load_state_dict(teacher_weights)
teacher.save(STAGE / "teacher_original.pt")
print("Verified source teacher: 23 inputs,", teacher.num_actions, "actions;", sum(p.numel() for p in teacher.q_network.parameters()), "parameters")

# P2: Freeze the historical competent teacher and collect greedy demonstrations.
demo_path = STAGE / "teacher_demonstrations.npz"
if not (RESUME and demo_path.exists()):
    data = collect_teacher_trajectories(teacher, cfg.trajectories, 150000, cfg.max_steps,
        progress=lambda i,n,s,e: print(f"teacher trajectories {i}/{n}; success={s}; grid_j={e:.2f}"))
    np.savez_compressed(demo_path, **data)
else:
    data = dict(np.load(demo_path))
data, selection = successful_trajectory_subset(data)
(STAGE/"demonstration_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
print("Distillation transitions:", len(data["actions"]), "selected trajectories:", len(np.unique(data["trajectory_ids"])), selection)

# P2/P3: complete-epoch student distillation with trajectory-disjoint validation.
student = new_student()
student_path = STAGE / "student_best.pt"
student_complete = STAGE / "student_training_complete.json"
if RESUME and student_complete.exists():
    complete_record=json.loads(student_complete.read_text(encoding="utf-8"))
    saved_student_cfg=dict(complete_record.get("config",{})); current_student_cfg=asdict(cfg)
    saved_student_cfg.pop("device",None); current_student_cfg.pop("device",None)
    if (not student_path.exists() or complete_record.get("data_signature")!=distillation_data_signature(data)
            or saved_student_cfg!=current_student_cfg):
        raise RuntimeError("Completed student artifacts do not match current teacher data/config. Use a fresh output folder; stale models are not evaluated.")
def student_val(model, epoch):
    score = evaluate_student_validation(model, list(range(160000,160005)), cfg.max_steps, cfg.device)
    print(f"student closed-loop validation epoch={epoch} success={score:.1%}")
    return score
if not (RESUME and student_complete.exists()):
    distill_student(data, student, cfg, STAGE,
        progress=lambda r: print("student", r), closed_loop_check=student_val,
        evaluate_every=1 if RUN_MODE == "smoke" else 10, resume=RESUME)
student_ckpt = torch.load(student_path, map_location=cfg.device, weights_only=False)
student.load_state_dict(student_ckpt["state_dict"])
student_validation_seeds=list(range(160000,160005))
teacher_validation=float(np.mean([r["success"] for r in evaluate_policy(teacher,len(student_validation_seeds),student_validation_seeds[0],cfg.max_steps,"teacher_validation")]))
student_validation=evaluate_student_validation(student,student_validation_seeds,cfg.max_steps,cfg.device)
corrective_triggered=student_validation < 0.85 * teacher_validation
selection_marker=STAGE/"corrective_data_selection.json"
final_student_path=STAGE/"student_best_final.pt"
if not (RESUME and selection_marker.exists() and final_student_path.exists()):
    chosen="teacher_trajectory_distillation"
    if corrective_triggered:
        correction_count=1 if RUN_MODE=="smoke" else (20 if RUN_MODE=="deadline" else 50)
        corrective=collect_corrective_trajectories(teacher,student,correction_count,190000,cfg.max_steps,cfg.device)
        aggregate={}
        for key in ["observations","q_values","actions","trajectory_ids"]:
            aggregate[key]=np.concatenate([data[key],corrective[key]],axis=0)
        aggregate["trajectory_success"]=np.concatenate([data["trajectory_success"],corrective["trajectory_success"]])
        aggregate["trajectory_energy_j"]=np.concatenate([data["trajectory_energy_j"],corrective["trajectory_energy_j"]])
        np.savez_compressed(STAGE/"teacher_student_aggregated_demonstrations.npz",**aggregate)
        corrective_dir=STAGE/"corrective_student"
        candidate=new_student()
        corrective_cfg=copy.deepcopy(cfg); corrective_cfg.epochs=min(cfg.epochs,50); corrective_cfg.patience=min(cfg.patience,8)
        if not (RESUME and (corrective_dir/"student_training_complete.json").exists()):
            distill_student(aggregate,candidate,corrective_cfg,corrective_dir,
                progress=lambda r: print("corrective student",r),
                closed_loop_check=lambda model,epoch:evaluate_student_validation(model,student_validation_seeds,cfg.max_steps,cfg.device),
                evaluate_every=1 if RUN_MODE=="smoke" else 10)
        candidate_ckpt=torch.load(corrective_dir/"student_best.pt",map_location=cfg.device,weights_only=False)
        candidate.load_state_dict(candidate_ckpt["state_dict"])
        candidate_rate=evaluate_student_validation(candidate,student_validation_seeds,cfg.max_steps,cfg.device)
        if candidate_rate > student_validation:
            student=candidate; student_validation=candidate_rate; chosen="student_guided_corrective_distillation"
    torch.save({"state_dict":student.state_dict(),"selection":chosen,"validation_success_rate":student_validation},final_student_path)
    selection_marker.write_text(json.dumps({"teacher_validation_success_rate":teacher_validation,
        "student_validation_success_rate":student_validation,"selected":chosen,
        "corrective_triggered":corrective_triggered},indent=2),encoding="utf-8")
else:
    final_ckpt=torch.load(final_student_path,map_location=cfg.device,weights_only=False)
    student.load_state_dict(final_ckpt["state_dict"])
student_agent = build_agent(device=cfg.device)
student_agent.q_network = student.to(student_agent.device)
student_agent.target_network = copy.deepcopy(student_agent.q_network)
student_original = new_student()
student_original.load_state_dict(torch.load(ROOT / "models" / "student_original.pt", map_location=cfg.device, weights_only=True))
student_original_agent = build_agent(device=cfg.device)
student_original_agent.q_network = student_original.to(student_original_agent.device)
student_original_agent.target_network = copy.deepcopy(student_original_agent.q_network)

# P4: optional bounded fine-tuning, initialized from the preserved teacher.
finetuned_path = STAGE / "teacher_finetuned.pt"
if teacher_episodes and not (RESUME and finetuned_path.exists()):
    fine = DQNAgentV11(23, 6, device=cfg.device)
    fine.q_network.load_state_dict(teacher_weights)
    fine.target_network.load_state_dict(teacher_weights)
    train_teacher(fine, teacher_episodes, 170000, cfg.max_steps, STAGE,
                  progress=lambda row: print("teacher fine-tune", row), resume=RESUME)
    fine.save(finetuned_path)

# Select teacher_best using a validation seed group only; final test seeds stay untouched.
teacher_best_agent=build_agent(device=cfg.device)
teacher_best_agent.q_network.load_state_dict(teacher_weights)
teacher_best_agent.target_network.load_state_dict(teacher_weights)
validation_count=2 if RUN_MODE=="smoke" else 20
teacher_validation_rows=evaluate_policy(teacher,validation_count,161000,cfg.max_steps,"teacher_original_validation")
validation_candidates={"teacher_original":{"completion_rate":float(np.mean([r["success"] for r in teacher_validation_rows])),
    "mean_energy_success_j":float(np.mean([r["grid_energy_j"] for r in teacher_validation_rows if r["success"]])) if any(r["success"] for r in teacher_validation_rows) else float("inf")}}
if finetuned_path.exists():
    fine_for_selection=build_agent(device=cfg.device)
    fine_for_selection.q_network.load_state_dict(torch.load(finetuned_path,map_location=cfg.device,weights_only=True))
    fine_for_selection.target_network.load_state_dict(fine_for_selection.q_network.state_dict())
    fine_validation_rows=evaluate_policy(fine_for_selection,validation_count,161000,cfg.max_steps,"teacher_finetuned_validation")
    validation_candidates["teacher_finetuned"]={"completion_rate":float(np.mean([r["success"] for r in fine_validation_rows])),
        "mean_energy_success_j":float(np.mean([r["grid_energy_j"] for r in fine_validation_rows if r["success"]])) if any(r["success"] for r in fine_validation_rows) else float("inf")}
    pd.DataFrame(teacher_validation_rows+fine_validation_rows).to_csv(STAGE/"teacher_validation_results.csv",index=False)
selected_teacher=max(validation_candidates,key=lambda k:(validation_candidates[k]["completion_rate"],-validation_candidates[k]["mean_energy_success_j"]))
if selected_teacher=="teacher_finetuned":
    teacher_best_agent.q_network.load_state_dict(torch.load(finetuned_path,map_location=cfg.device,weights_only=True))
    teacher_best_agent.target_network.load_state_dict(teacher_best_agent.q_network.state_dict())
teacher_best_agent.save(STAGE/"teacher_best.pt")
(STAGE/"teacher_selection.json").write_text(json.dumps({"validation_seed_start":161000,
    "validation_episodes_per_candidate":validation_count,"candidates":validation_candidates,
    "selected":selected_teacher,"selection_rule":"higher validation completion, then lower mean energy among validation successes"},indent=2),encoding="utf-8")

# If fine-tuning truly wins validation, distil it as a distinct follow-up student.
student_original_teacher_path=STAGE/"student_best_from_original_teacher.pt"
if not student_original_teacher_path.exists(): torch.save({"state_dict":student.state_dict()},student_original_teacher_path)
if selected_teacher=="teacher_finetuned":
    student2_dir=STAGE/"student_from_finetuned_teacher"
    student2_dir.mkdir(parents=True,exist_ok=True)
    student2_path=student2_dir/"student_best.pt"
    student2_marker=student2_dir/"student_training_complete.json"
    if not (RESUME and student2_marker.exists()):
        demo2_count=min(100,cfg.trajectories)
        demo2=collect_teacher_trajectories(teacher_best_agent,demo2_count,200000,cfg.max_steps)
        demo2,selection2=successful_trajectory_subset(demo2)
        np.savez_compressed(student2_dir/"teacher_demonstrations.npz",**demo2)
        (student2_dir/"demonstration_selection.json").write_text(json.dumps(selection2,indent=2),encoding="utf-8")
        candidate2=new_student()
        cfg2=copy.deepcopy(cfg); cfg2.epochs=min(cfg.epochs,100); cfg2.patience=min(cfg.patience,12)
        student2_dir.mkdir(parents=True,exist_ok=True)
        distill_student(demo2,candidate2,cfg2,student2_dir,
            progress=lambda r: print("improved-teacher student",r),
            closed_loop_check=lambda model,epoch:evaluate_student_validation(model,list(range(162000,162005)),cfg.max_steps,cfg.device),
            evaluate_every=1 if RUN_MODE=="smoke" else 10)
    candidate2=new_student(); candidate2.load_state_dict(torch.load(student2_path,map_location=cfg.device,weights_only=False)["state_dict"])
    original_candidate=new_student(); original_candidate.load_state_dict(torch.load(student_original_teacher_path,map_location=cfg.device,weights_only=False)["state_dict"])
    original_rate=evaluate_student_validation(original_candidate,list(range(162000,162005)),cfg.max_steps,cfg.device)
    new_teacher_rate=evaluate_student_validation(candidate2,list(range(162000,162005)),cfg.max_steps,cfg.device)
    (STAGE/"student_teacher_source_comparison.json").write_text(json.dumps({
        "validation_seeds":list(range(162000,162005)),
        "student_from_original_teacher_success_rate":original_rate,
        "student_from_finetuned_teacher_success_rate":new_teacher_rate,
        "selected":"finetuned_teacher" if new_teacher_rate>original_rate else "original_teacher"
    },indent=2),encoding="utf-8")
    if new_teacher_rate>original_rate:
        student=candidate2
        torch.save({"state_dict":student.state_dict(),"teacher_source":"teacher_finetuned",
                    "validation_success_rate":new_teacher_rate},STAGE/"student_best_final.pt")

# Optional steel analysis is independent of robot training.
from experiments.v1_1.analyze_steel import analyze
steel_out = OUTPUT / "dataset_analysis"
steel_out.mkdir(exist_ok=True)
analyze(ROOT / "notebooks" / "steel_data.csv", steel_out)

# Select the strongest conventional controller on a third, disjoint validation seed group.
baseline_val_count=2 if RUN_MODE=="smoke" else 20
baseline_val_rows=[]
for name in ["fixed_original","fixed_slow","proportional","adaptive"]:
    cache=STAGE/f"validation_baseline_{name}.csv"
    if RESUME and cache.exists() and len(pd.read_csv(cache))==baseline_val_count:
        baseline_val_rows.extend(pd.read_csv(cache).to_dict("records"))
    else:
        vals=run_ik_baseline(name,baseline_val_count,163000,cfg.max_steps)
        pd.DataFrame(vals).to_csv(cache,index=False); baseline_val_rows.extend(vals)
base_val=pd.DataFrame(baseline_val_rows)
base_score={}
for name,grp in base_val.groupby("policy"):
    successful=grp.loc[grp.success==1,"grid_energy_j"]
    base_score[name]={"completion_rate":float(grp.success.mean()),
                      "mean_energy_success_j":float(successful.mean()) if len(successful) else float("inf")}
strongest_baseline=max(base_score,key=lambda k:(base_score[k]["completion_rate"],-base_score[k]["mean_energy_success_j"]))
(OUTPUT/"baseline_validation_selection.json").write_text(json.dumps({"seed_start":163000,
    "episodes_per_controller":baseline_val_count,"scores":base_score,"selected":strongest_baseline},indent=2),encoding="utf-8")

# Final fresh held-out evaluation. All policies use the same seeds and environment.
if RUN_MODE == "smoke":
    n_test, seed_test = 2, 180000
else:
    n_test, seed_test = 50, 140000
rows = []
for policy, fn in [
    ("IK_fixed_original", lambda: run_ik_baseline("fixed_original", n_test, seed_test, cfg.max_steps)),
    ("IK_fixed_slow", lambda: run_ik_baseline("fixed_slow", n_test, seed_test, cfg.max_steps)),
    ("IK_proportional", lambda: run_ik_baseline("proportional", n_test, seed_test, cfg.max_steps)),
    ("IK_adaptive", lambda: run_ik_baseline("adaptive", n_test, seed_test, cfg.max_steps)),
    ("teacher_original", lambda: evaluate_policy(teacher, n_test, seed_test, cfg.max_steps, "teacher_original")),
    ("teacher_best", lambda: evaluate_policy(teacher_best_agent, n_test, seed_test, cfg.max_steps, "teacher_best")),
    ("student_original", lambda: evaluate_policy(student_original_agent, n_test, seed_test, cfg.max_steps, "student_original")),
    ("student_best", lambda: evaluate_policy(student_agent, n_test, seed_test, cfg.max_steps, "student_best")),
]:
    print("Evaluating", policy)
    policy_file=STAGE/f"evaluation_{policy}.csv"
    if policy_file.exists():
        cached=pd.read_csv(policy_file)
        cache_valid=(len(cached)==n_test and cached.seed.nunique()==n_test
                     and set(cached.seed.astype(int))==set(range(seed_test,seed_test+n_test))
                     and set(cached.policy.astype(str))=={policy})
        if RESUME and cache_valid:
            rows.extend(cached.to_dict("records")); continue
        if not cache_valid:
            raise RuntimeError(f"Stale or mismatched evaluation cache: {policy_file}. Use a fresh output folder; cached metrics are not reused or silently overwritten.")
        if not RESUME:
            raise RuntimeError(f"Evaluation file already exists: {policy_file}. Use resume=True or a fresh output folder.")
    policy_rows=fn()
    pd.DataFrame(policy_rows).to_csv(policy_file,index=False)
    rows.extend(policy_rows)
if finetuned_path.exists():
    fine_eval = build_agent(device=cfg.device)
    fine_eval.q_network.load_state_dict(torch.load(finetuned_path, map_location=cfg.device, weights_only=True))
    fine_eval.target_network.load_state_dict(fine_eval.q_network.state_dict())
    policy_file=STAGE/"evaluation_teacher_finetuned.csv"
    if policy_file.exists():
        cached=pd.read_csv(policy_file)
        cache_valid=(len(cached)==n_test and cached.seed.nunique()==n_test
                     and set(cached.seed.astype(int))==set(range(seed_test,seed_test+n_test))
                     and set(cached.policy.astype(str))=={"teacher_finetuned"})
        if RESUME and cache_valid:
            rows.extend(cached.to_dict("records"))
        elif not cache_valid:
            raise RuntimeError(f"Stale or mismatched evaluation cache: {policy_file}. Use a fresh output folder.")
        else:
            raise RuntimeError(f"Evaluation file already exists: {policy_file}. Use resume=True or a fresh output folder.")
    else:
        policy_rows=evaluate_policy(fine_eval,n_test,seed_test,cfg.max_steps,"teacher_finetuned")
        pd.DataFrame(policy_rows).to_csv(policy_file,index=False); rows.extend(policy_rows)

# Dataset access is independent and non-blocking: attempt only official public endpoints.
from experiments.v1_1.download_robot_datasets import main as attempt_public_robot_dataset_downloads
try:
    attempt_public_robot_dataset_downloads()
except Exception as exc:
    print("Optional dataset retrieval did not complete; core training/evaluation continues:", repr(exc))
download_record = BUNDLE/"experiments"/"v1_1"/"runs"/"2026-11-26"/"dataset_download_attempt.json"
if download_record.exists(): shutil.copy2(download_record, OUTPUT/"dataset_download_attempt.json")
for doc_path in [BUNDLE/"data"/"raw"/"high_payload_robot"/"readme.txt",
                 BUNDLE/"data"/"raw"/"collaborative_robot"/"README.md"]:
    if doc_path.exists():
        docs_out=OUTPUT/"dataset_documentation"; docs_out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(doc_path,docs_out/doc_path.name)
file_inventory=[]
import hashlib
for base in [BUNDLE/"data"/"raw"/"collaborative_robot", BUNDLE/"data"/"raw"/"high_payload_robot"]:
    if base.exists():
        for fp in base.rglob("*"):
            if not fp.is_file(): continue
            digest=hashlib.sha256(fp.read_bytes()).hexdigest()
            item={"path":str(fp.relative_to(BUNDLE)),"bytes":fp.stat().st_size,"sha256":digest,"format":fp.suffix.lower()}
            if fp.suffix.lower()==".csv":
                try:
                    frame=pd.read_csv(fp,low_memory=False)
                    item.update({"rows":len(frame),"columns":frame.columns.astype(str).tolist(),
                                 "missing_cells":int(frame.isna().sum().sum()),"duplicate_rows":int(frame.duplicated().sum())})
                except Exception as exc: item["inspection_error"]=repr(exc)
            elif fp.suffix.lower()==".zip":
                with zipfile.ZipFile(fp) as zf: item["archive_members"]=zf.namelist()[:100]
            file_inventory.append(item)
(OUTPUT/"robot_dataset_file_inventory.json").write_text(json.dumps(file_inventory,indent=2),encoding="utf-8")

results = pd.DataFrame(rows)
results.to_csv(OUTPUT / "final_episode_metrics.csv", index=False)
summary = results.groupby("policy").agg(episodes=("seed","count"), completions=("success","sum"),
    completion_rate=("success","mean"), mean_energy_all_attempts=("grid_energy_j","mean"),
    mean_energy_success_only=("energy_per_success_j","mean"), mean_energy_failed_attempts=("failure_energy_j","mean"),
    aggregate_energy_per_completed_task=("grid_energy_j",lambda x: float(x.sum()/int(results.loc[x.index,"success"].sum())) if int(results.loc[x.index,"success"].sum())>0 else np.nan),
    mean_mechanical_energy_j=("mechanical_energy_j","mean"), mean_peak_power_w=("peak_grid_power_w","mean"),
    mean_position_error_m=("position_error_m","mean"), mean_completion_time_s=("completion_time_s","mean"),
    mean_buffer_capacity_fraction=("buffer_capacity_fraction","mean"), mean_buffer_discharged_j=("buffer_discharged_j","mean"),
    mean_steps=("steps","mean")).reset_index()
def wilson(k,n,z=1.959963984540054):
    if n<=0: return (float("nan"),float("nan"))
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return center-half,center+half
wilson_bounds=[wilson(int(r.completions),int(r.episodes)) for r in summary.itertuples()]
summary["completion_wilson95_low"]= [x[0] for x in wilson_bounds]
summary["completion_wilson95_high"]= [x[1] for x in wilson_bounds]
summary.to_csv(OUTPUT / "final_policy_summary.csv", index=False)
baseline = results[results.policy == "fixed_original"].set_index("seed")
if baseline.empty or baseline.index.duplicated().any():
    raise ValueError("Expected exactly one fixed_original baseline row per held-out seed.")
paired = []
bootstrap_rng = np.random.default_rng(20261126)
for name, group in results.groupby("policy"):
    other = group.set_index("seed")
    common = baseline.index.intersection(other.index)
    if len(common) != len(baseline) or other.index.duplicated().any():
        raise ValueError(f"Policy {name} does not have one matched observation for every fixed_original seed.")
    both = [s for s in common if baseline.loc[s,"success"] == 1 and other.loc[s,"success"] == 1]
    savings = [(baseline.loc[s,"grid_energy_j"]-other.loc[s,"grid_energy_j"])/baseline.loc[s,"grid_energy_j"]*100
               for s in both if baseline.loc[s,"grid_energy_j"] > 0]
    completion_pairs = [(int(baseline.loc[s,"success"]), int(other.loc[s,"success"])) for s in common]
    paired_counts = {"both_success":sum(a==1 and b==1 for a,b in completion_pairs),
                     "baseline_only_success":sum(a==1 and b==0 for a,b in completion_pairs),
                     "policy_only_success":sum(a==0 and b==1 for a,b in completion_pairs),
                     "both_fail":sum(a==0 and b==0 for a,b in completion_pairs)}
    discordant=paired_counts["baseline_only_success"]+paired_counts["policy_only_success"]
    smaller=min(paired_counts["baseline_only_success"],paired_counts["policy_only_success"])
    from math import comb
    exact_mcnemar=min(1.0,2.0*sum(comb(discordant,i) for i in range(smaller+1))/(2**discordant)) if discordant else 1.0
    boot = [float(np.mean(bootstrap_rng.choice(savings, size=len(savings), replace=True)))
            for _ in range(2000)] if savings else []
    paired.append({"policy":name,"paired_successes_with_fixed_IK":len(both),
                   "mean_paired_saving_pct":float(np.mean(savings)) if savings else np.nan,
                   "median_paired_saving_pct":float(np.median(savings)) if savings else np.nan,
                   "sd_paired_saving_pct":float(np.std(savings,ddof=1)) if len(savings)>1 else np.nan,
                   "bootstrap_95_low_pct":float(np.percentile(boot,2.5)) if boot else np.nan,
                   "bootstrap_95_high_pct":float(np.percentile(boot,97.5)) if boot else np.nan,
                   "exact_mcnemar_p":exact_mcnemar,
                   **paired_counts})
pd.DataFrame(paired).to_csv(OUTPUT / "paired_energy_saving.csv", index=False)

# Parameter counts and measured batch-1 PyTorch forward latency (NN-only).
import time, platform
bench=[]
for name, model in [("teacher",teacher.q_network),("student",student_agent.q_network)]:
    model.eval(); x=torch.zeros((1,23),device=cfg.device)
    with torch.no_grad():
        for _ in range(100): model(x)
        values=[]
        for _ in range(1000):
            start=time.perf_counter(); model(x)
            if cfg.device.startswith("cuda"): torch.cuda.synchronize()
            values.append((time.perf_counter()-start)*1000)
    bench.append({"model":name,"parameters":sum(p.numel() for p in model.parameters()),
        "size_bytes":sum(t.numel()*t.element_size() for t in model.state_dict().values()),
        "device":cfg.device,"hardware":platform.processor(),"passes":len(values),
        "mean_ms":float(np.mean(values)),"median_ms":float(np.median(values)),
        "p95_ms":float(np.percentile(values,95)),"p99_ms":float(np.percentile(values,99)),
        "scope":"neural network forward only; excludes simulator and controller"})
pd.DataFrame(bench).to_csv(OUTPUT / "inference_benchmark.csv",index=False)

comparison=pd.DataFrame(bench)
comparison["parameters_reduction_vs_teacher_pct"]=(1-comparison.parameters/comparison.loc[comparison.model=="teacher","parameters"].iloc[0])*100
comparison.to_csv(OUTPUT/"model_comparison.csv",index=False)

# ONNX export and runtime numerical/action parity check on real environment observations.
export_dir=OUTPUT/"models"; export_dir.mkdir(exist_ok=True)
for name,model in [("teacher",teacher.q_network),("teacher_best",teacher_best_agent.q_network),("student",student_agent.q_network)]:
    model.eval(); sample=torch.as_tensor(data["observations"][:1],dtype=torch.float32,device=cfg.device)
    path=export_dir/f"{name}.onnx"
    torch.onnx.export(model,sample,path,input_names=["observation"],output_names=["q_values"],
                      dynamic_axes={"observation":{0:"batch_size"},"q_values":{0:"batch_size"}},
                      opset_version=17,dynamo=False)
    import onnx, onnxruntime as ort
    exported=onnx.load(str(path)); onnx.checker.check_model(exported)
    input_shape=[d.dim_value for d in exported.graph.input[0].type.tensor_type.shape.dim]
    output_shape=[d.dim_value for d in exported.graph.output[0].type.tensor_type.shape.dim]
    if input_shape[-1]!=23 or output_shape[-1]!=89:
        raise ValueError(f"Unexpected ONNX dimensions for {name}: {input_shape} -> {output_shape}")
    session=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
    real=np.asarray(data["observations"][:min(32,len(data["observations"]))],dtype=np.float32)
    with torch.no_grad(): expected=model(torch.as_tensor(real,device=cfg.device)).cpu().numpy()
    actual=session.run(["q_values"],{"observation":real})[0]
    parity=float(np.max(np.abs(expected-actual)))
    agreement=float(np.mean(expected.argmax(1)==actual.argmax(1)))
    (export_dir/f"{name}_onnx_validation.json").write_text(json.dumps({"input_shape":input_shape,
        "output_shape":output_shape,"max_abs_error":parity,"greedy_action_agreement":agreement,
        "samples":len(real),"onnx_file_bytes":path.stat().st_size},indent=2))

# Research figures and transparent, run-specific report artifacts.
import matplotlib.pyplot as plt
fig, axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
summary.set_index("policy").completion_rate.mul(100).plot.bar(ax=axes[0],color="#167d88")
axes[0].set_ylabel("Completion (%)"); axes[0].set_title("Held-out target-reaching completion")
summary.set_index("policy").mean_energy_success_only.plot.bar(ax=axes[1],color="#e97942")
axes[1].set_ylabel("J per completed episode"); axes[1].set_title("Energy on successful attempts")
fig.savefig(OUTPUT/"completion_energy_comparison.png",dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));
valid=results[(results.policy!="IK_fixed_original") & (results.success==1)]
for name,g in valid.groupby("policy"):
    paired_vals=[]
    for _,r in g.iterrows():
        if int(r.seed) in baseline.index and int(baseline.loc[int(r.seed),"success"])==1 and baseline.loc[int(r.seed),"grid_energy_j"]>0:
            paired_vals.append((baseline.loc[int(r.seed),"grid_energy_j"]-r.grid_energy_j)/baseline.loc[int(r.seed),"grid_energy_j"]*100)
    if paired_vals: ax.hist(paired_vals,bins=15,alpha=.45,label=name)
ax.axvline(0,color="black",linewidth=1); ax.set_xlabel("Paired energy saving vs fixed IK (%)"); ax.set_ylabel("Count"); ax.legend()
fig.tight_layout(); fig.savefig(OUTPUT/"paired_energy_saving_distribution.png",dpi=180); plt.close(fig)

student_curve=STAGE/"student_training_history.csv"
if student_curve.exists():
    curve=pd.read_csv(student_curve)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(curve.epoch,curve.train_loss,label="train")
    axes[0].plot(curve.epoch,curve.validation_loss,label="trajectory validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Distillation loss"); axes[0].legend()
    if "closed_loop_validation_success" in curve:
        valid_curve=curve.dropna(subset=["closed_loop_validation_success"])
        axes[1].plot(valid_curve.epoch,valid_curve.closed_loop_validation_success*100,marker="o")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Closed-loop validation completion (%)")
    fig.savefig(OUTPUT/"student_training_curves.png",dpi=180); plt.close(fig)
teacher_curve=STAGE/"teacher_training_history.csv"
if teacher_curve.exists():
    curve=pd.read_csv(teacher_curve)
    fig,axes=plt.subplots(2,1,figsize=(8,6),constrained_layout=True)
    axes[0].plot(curve.episode,curve.reward); axes[0].set_ylabel("Episode reward")
    axes[1].plot(curve.episode,curve.success.rolling(10,min_periods=1).mean()*100)
    axes[1].set_ylabel("10-episode completion (%)"); axes[1].set_xlabel("Episode")
    fig.savefig(OUTPUT/"teacher_learning_curves.png",dpi=180); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
comparison.set_index("model").parameters.plot.bar(ax=axes[0],color="#167d88")
axes[0].set_ylabel("Trainable parameters"); axes[0].set_title("Model size")
comparison.set_index("model").median_ms.plot.bar(ax=axes[1],color="#e97942")
axes[1].set_ylabel("Median forward pass (ms)"); axes[1].set_title("Batch-1 neural-network timing")
fig.savefig(OUTPUT/"model_size_latency.png",dpi=180); plt.close(fig)

report = ["# V1.1 final research run report", "", f"Mode: {RUN_MODE}",
          f"Episodes in fresh final evaluation: {n_test} per policy; seed range {seed_test}–{seed_test+n_test-1}.",
          f"Strongest conventional baseline selected on separate validation data: {strongest_baseline}.",
          f"Teacher checkpoint selected using validation data: {selected_teacher}.",
          "Historical benchmark (separate examined seeds, not this fresh test): IK 45/50 (90%), teacher 44/50 (88%), student 6/50 (12%), and online teacher 3/500 (0.6%); historical paired mean energy saving -10.82%. See `historical/` for preserved source files.",
          "", "These are simulated target-reaching results using signed joint mechanical work. Buffer efficiencies (charge 0.80, discharge 0.90) are assumptions, not electrical measurements. Failed attempts are included in all-attempt energy; successful-only comparisons use shared successful seeds.",
          "", "## Per-policy summary", "", summary.to_markdown(index=False), "", "## Paired energy results", "", pd.DataFrame(paired).to_markdown(index=False),
          "", "## Limitations", "", "The KUKA iiwa task does not include object grasping or placement. The steel dataset is industrial demand context only. This run does not establish real robot energy or tariff savings. Student suitability requires the displayed closed-loop completion and paired results; do not infer improvement from distillation loss alone."]
(OUTPUT/"FINAL_RESEARCH_REPORT.md").write_text("\n".join(report),encoding="utf-8")
(OUTPUT/"OBJECTIVES_STATUS.md").write_text("# V1.1 objectives status\n\nGenerated from this run. Completion and energy values in `final_policy_summary.csv` are actual only for the number of episodes listed. No unexecuted target is marked complete.\n\n- Environment and source bundle: executed in this run; see config and notebook logs.\n- Teacher checkpoint: original checkpoint retained; fine-tune episodes are in `teacher_training_history.csv` when enabled.\n- Student: actual collected teacher trajectories, epochs, validation results and closed-loop score are in the corresponding NPZ/CSV and episode metrics.\n- Paired evaluation: see `final_episode_metrics.csv`, `paired_energy_saving.csv`, and report.\n- External robot datasets: not downloaded in the original local preparation; retry access separately.\n",encoding="utf-8")
(OUTPUT/"POSTER_RESULTS.md").write_text("# Poster results from this execution\n\nUse `final_policy_summary.csv` and `paired_energy_saving.csv` only after this notebook finishes. State the exact evaluated sample size and whether results come from smoke, deadline, or full mode. Distinguish simulation estimates from electrical measurements. If this is smoke mode, do not present it as confirmatory research.\n",encoding="utf-8")
(OUTPUT/"DEFENCE_QUESTIONS.md").write_text("# Defence preparation prompts\n\n1. What exactly is the task? Target reaching, not pick-and-place.\n2. What physical quantity is estimated as energy? Signed joint torque times angular velocity integrated over simulator time.\n3. What is the main limitation of that quantity? It is mechanical work, not measured wall electricity or motor losses.\n4. Why use matched seeds? To compare controllers on identical starts and targets.\n5. How is failed-attempt energy handled? Reported separately and also included in all-attempt energy.\n6. Why split teacher data by trajectory? To prevent states from the same trajectory leaking across train and validation.\n7. Why can teacher action cloning fail in closed loop? The student changes the state distribution and errors can compound.\n8. What does the energy buffer represent? An assumed simplified store with explicit charge/discharge losses.\n9. What evidence is required for edge deployment? Real-device batch-1 and end-to-end control latency, memory and reliability.\n10. What is needed to claim energy savings? Positive paired savings on successful shared scenarios with uncertainty, without hidden task failures.\n",encoding="utf-8")

# Final seed manifest is written only after actual stage counts are known.
seed_rows=[]
def add_seed_range(role,start,count,split):
    seed_rows.extend({"role":role,"seed":int(start+i),"split":split,"status":"executed"} for i in range(int(count)))
raw_demos=np.load(demo_path)
add_seed_range("teacher_demonstration_trajectories",150000,len(raw_demos["trajectory_success"]),"training")
add_seed_range("student_closed_loop_validation",160000,5,"validation")
add_seed_range("teacher_checkpoint_validation",161000,validation_count,"validation")
add_seed_range("baseline_selection_validation",163000,baseline_val_count,"validation")
add_seed_range("final_heldout_test",seed_test,n_test,"test")
add_seed_range("model_initialization_rng",cfg.seed,1,"training")
teacher_history_path=STAGE/"teacher_training_history.csv"
if teacher_history_path.exists(): add_seed_range("teacher_online_finetuning",170000,len(pd.read_csv(teacher_history_path)),"training")
aggregated_path=STAGE/"teacher_student_aggregated_demonstrations.npz"
if aggregated_path.exists():
    agg=np.load(aggregated_path); ids=np.unique(agg["trajectory_ids"]); corrected=ids[ids>=1190000]
    add_seed_range("student_guided_corrective_trajectories",190000,len(corrected),"training")
improved_demo=STAGE/"student_from_finetuned_teacher"/"teacher_demonstrations.npz"
if improved_demo.exists(): add_seed_range("finetuned_teacher_student_trajectories",200000,len(np.load(improved_demo)["trajectory_success"]),"training")
pd.DataFrame(seed_rows).to_csv(OUTPUT/"seed_manifest.csv",index=False)

zip_path=OUTPUT/"FINAL_RESEARCH_RESULTS.zip"
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
    for p in OUTPUT.rglob("*"):
        if p.is_file() and p != zip_path and "raw_dataset" not in p.parts:
            zf.write(p,p.relative_to(OUTPUT).as_posix())
    zf.write(ROOT/"experiments"/"v1_1"/"pipeline.py","source_snapshot/pipeline.py")
    zf.write(ROOT/"experiments"/"v1_1"/"environment_v1_1.py","source_snapshot/environment_v1_1.py")
    zf.write(ROOT/"experiments"/"v1_1"/"analyze_steel.py","source_snapshot/analyze_steel.py")
    zf.write(ROOT/"src"/"environment.py","source_snapshot/src_environment.py")
    zf.write(ROOT/"src"/"dqn_teacher.py","source_snapshot/src_dqn_teacher.py")
    for rel in ["DATASET_MANIFEST.json","DATASET_ANALYSIS.md","DATASET_PROVENANCE.md",
                "LITERATURE_IMPLEMENTATION_MATRIX.md","APRIL_RESEARCH_ROADMAP.md"]:
        if (ROOT/rel).exists(): zf.write(ROOT/rel,"project_docs/"+rel)
print("Results ZIP:",zip_path,"bytes:",zip_path.stat().st_size)

print(summary.to_string(index=False))
print("Energy buffer efficiencies are assumptions; these are simulation estimates, not measured electrical or tariff costs.")
'''

md = """# Final research training — Version 1.1\n\nThis notebook embeds the source modules, current historical teacher and student weights, and existing steel CSV. Upload this notebook alone to Colab. It preserves the V1.0 code and historical results.\n\n**Recommended:** GPU runtime (T4 if available), `MODE = \"deadline\"`, Drive enabled, then Run all. Smoke mode verifies plumbing only; full mode uses a large training budget. Choose `MODE = \"resume\"` and `RESUME_FROM_MODE = \"deadline\"` or `\"full\"` to continue the matching Drive run. The standard task is target reaching, not pick-and-place. Energy is simplified simulated joint mechanical work with assumed buffer efficiencies, not measured electrical cost.\n\nDrive keeps source demonstrations, student optimizer checkpoints, teacher replay/optimizer checkpoints, and completed evaluation stages. Keep the run mode, configuration and seeds unchanged when resuming.\n"""

def mcell(source):
    return {"cell_type":"markdown","metadata":{},"source":source.splitlines(keepends=True)}

def ccell(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}

notebook = {"cells":[mcell(md), ccell(setup), mcell("## Configure and run\n\nDeadline mode uses the preserved teacher to distil the student first, periodically checks closed-loop validation, then performs bounded teacher fine-tuning and fresh paired evaluation."), ccell(run)],
           "metadata":{"colab":{"name":"FINAL_RESEARCH_TRAINING.ipynb","provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},"nbformat":4,"nbformat_minor":5}
out_path = ROOT / "FINAL_RESEARCH_TRAINING.ipynb"
out_path.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",",":")), encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes); source bundle {len(bundle_bytes)} bytes")
