"""Independently reconstruct matched-seed statistics and publication figures."""
from __future__ import annotations

from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "internal_defence_v1_1_2026-11-26" / "outer_deadline"
OUT = ROOT / "DEFENCE_2026"
FIG = OUT / "figures"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261126


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*np.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return centre-half, centre+half


def exact_mcnemar(baseline_only: int, policy_only: int) -> float:
    discordant = baseline_only + policy_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, policy_only)
    return min(1.0, 2.0*sum(comb(discordant, i) for i in range(tail+1))/(2**discordant))


def reconstruct() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(DATA / "final_episode_metrics.csv")
    required = {"policy", "seed", "success", "grid_energy_j", "mechanical_energy_j",
                "peak_grid_power_w", "position_error_m", "completion_time_s",
                "failure_energy_j", "energy_per_success_j"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"Missing required episode columns: {sorted(missing)}")
    if episodes.duplicated(["policy", "seed"]).any():
        raise ValueError("Duplicate policy/seed rows in recovered episode CSV.")
    policy_names = set(episodes.policy.astype(str))
    expected = {"fixed_original", "fixed_slow", "proportional", "adaptive",
                "teacher_original", "teacher_best", "teacher_finetuned",
                "student_original", "student_best"}
    if policy_names != expected:
        raise ValueError(f"Unexpected policy identifiers: {sorted(policy_names ^ expected)}")
    seed_sets = {name: set(group.seed.astype(int)) for name, group in episodes.groupby("policy")}
    common_seeds = seed_sets["fixed_original"]
    if len(common_seeds) != 50 or any(seeds != common_seeds for seeds in seed_sets.values()):
        raise ValueError("Expected the same 50 held-out scenario seeds for every policy.")
    if common_seeds != set(range(min(common_seeds), max(common_seeds)+1)):
        raise ValueError("Held-out seed sequence contains gaps.")

    rows = []
    for name, group in episodes.groupby("policy", sort=True):
        n = len(group)
        k = int(group.success.astype(int).sum())
        lo, hi = wilson(k, n)
        success_energy = group.loc[group.success.astype(bool), "grid_energy_j"]
        failed_energy = group.loc[~group.success.astype(bool), "grid_energy_j"]
        rows.append({
            "policy": name, "episodes": n, "completions": k,
            "completion_rate": k/n, "completion_wilson95_low": lo,
            "completion_wilson95_high": hi,
            "mean_grid_energy_all_attempts_j": float(group.grid_energy_j.mean()),
            "mean_grid_energy_success_j": float(success_energy.mean()) if len(success_energy) else np.nan,
            "mean_grid_energy_failed_j": float(failed_energy.mean()) if len(failed_energy) else np.nan,
            "aggregate_energy_per_completed_task_j": float(group.grid_energy_j.sum()/k) if k else np.nan,
            "mean_mechanical_energy_j": float(group.mechanical_energy_j.mean()),
            "mean_peak_power_w": float(group.peak_grid_power_w.mean()),
            "max_peak_power_w": float(group.peak_grid_power_w.max()),
            "mean_completion_time_s_success": float(group.loc[group.success.astype(bool), "completion_time_s"].mean()) if k else np.nan,
            "mean_position_error_m": float(group.position_error_m.mean()),
            "mean_buffer_discharged_j": float(group.buffer_discharged_j.mean()) if "buffer_discharged_j" in group else np.nan,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "corrected_policy_summary.csv", index=False)

    baseline = episodes[episodes.policy.eq("fixed_original")].set_index("seed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired_rows = []
    savings_by_policy: dict[str, np.ndarray] = {}
    for name in sorted(policy_names):
        candidate = episodes[episodes.policy.eq(name)].set_index("seed")
        matched = baseline.join(candidate, how="inner", lsuffix="_baseline", rsuffix="_policy")
        if len(matched) != len(common_seeds):
            raise ValueError(f"Seed mismatch for {name}.")
        bsuccess = matched.success_baseline.astype(int).to_numpy()
        psuccess = matched.success_policy.astype(int).to_numpy()
        both_success = (bsuccess == 1) & (psuccess == 1)
        base_e = matched.grid_energy_j_baseline.to_numpy(dtype=float)
        policy_e = matched.grid_energy_j_policy.to_numpy(dtype=float)
        valid = both_success & (base_e > 0)
        savings = (base_e[valid] - policy_e[valid]) / base_e[valid] * 100.0
        savings_by_policy[name] = savings
        if len(savings):
            draws = rng.choice(savings, size=(BOOTSTRAP_REPLICATES, len(savings)), replace=True).mean(axis=1)
            ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan
        both = int(((bsuccess == 1) & (psuccess == 1)).sum())
        base_only = int(((bsuccess == 1) & (psuccess == 0)).sum())
        policy_only = int(((bsuccess == 0) & (psuccess == 1)).sum())
        neither = int(((bsuccess == 0) & (psuccess == 0)).sum())
        paired_rows.append({
            "policy": name, "scenarios_matched": len(matched),
            "shared_success_scenarios": int(valid.sum()),
            "mean_paired_energy_saving_pct": float(np.mean(savings)) if len(savings) else np.nan,
            "median_paired_energy_saving_pct": float(np.median(savings)) if len(savings) else np.nan,
            "sd_paired_energy_saving_pct": float(np.std(savings, ddof=1)) if len(savings) > 1 else np.nan,
            "bootstrap_ci95_low_pct": float(ci_low), "bootstrap_ci95_high_pct": float(ci_high),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES if len(savings) else 0,
            "both_success": both, "baseline_only_success": base_only,
            "policy_only_success": policy_only, "both_fail": neither,
            "exact_mcnemar_two_sided_p": exact_mcnemar(base_only, policy_only),
        })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "corrected_paired_comparisons.csv", index=False)

    # Student action agreement reported in the recovered trajectory-disjoint validation history.
    history = pd.read_csv(DATA / "stages" / "student_training_history.csv")
    history.to_csv(OUT / "student_training_history_source_copy.csv", index=False)
    benchmark = pd.read_csv(DATA / "model_comparison.csv")
    benchmark.to_csv(OUT / "verified_model_benchmark.csv", index=False)
    teacher_history = pd.read_csv(DATA / "stages" / "teacher_training_history.csv")
    teacher_history.to_csv(OUT / "teacher_finetuning_history_source_copy.csv", index=False)

    colors = {"fixed_original":"#596273", "teacher_original":"#087f8c", "teacher_best":"#10a5a2",
              "teacher_finetuned":"#8c2d45", "student_original":"#e49b36", "student_best":"#d95f32",
              "fixed_slow":"#a2a9b3", "proportional":"#7d91ab", "adaptive":"#b794cc"}
    # Completion with explicitly visible Wilson intervals.
    fig, ax = plt.subplots(figsize=(11, 5))
    s = summary.sort_values("completion_rate", ascending=False)
    yerr = np.maximum(0.0, np.vstack([s.completion_rate-s.completion_wilson95_low,
                                     s.completion_wilson95_high-s.completion_rate]) * 100)
    ax.bar(s.policy, s.completion_rate*100, color=[colors.get(x,"#777") for x in s.policy])
    ax.errorbar(np.arange(len(s)), s.completion_rate*100, yerr=yerr, fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("Completed episodes (%)"); ax.set_ylim(0, 105); ax.tick_params(axis="x", rotation=35)
    ax.set_title("Held-out target-reaching completion (50 shared seeds; Wilson 95% CI)")
    fig.tight_layout(); fig.savefig(FIG/"completion_rates.png", dpi=200); plt.close(fig)

    # Energy plot separates successful and failed attempts; it never hides failed-task cost.
    x = np.arange(len(summary)); width = .38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-width/2, summary.mean_grid_energy_success_j, width, label="Successful attempts", color="#087f8c")
    ax.bar(x+width/2, summary.mean_grid_energy_failed_j, width, label="Failed attempts", color="#e49b36")
    ax.set_xticks(x, summary.policy, rotation=35, ha="right"); ax.set_ylabel("Mean simulated grid energy (J)")
    ax.set_title("Energy per attempt, split by task outcome (NaN means no such episodes)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/"energy_success_and_failure.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary.policy, summary.mean_peak_power_w, color=[colors.get(x,"#777") for x in summary.policy])
    ax.set_ylabel("Mean episode peak simulated power (W)"); ax.tick_params(axis="x", rotation=35)
    ax.set_title("Peak simulated power (all attempts)")
    fig.tight_layout(); fig.savefig(FIG/"peak_power.png", dpi=200); plt.close(fig)

    plot_names = ["teacher_original", "teacher_best", "student_original", "student_best"]
    means = paired.set_index("policy").loc[plot_names, "mean_paired_energy_saving_pct"]
    lows = paired.set_index("policy").loc[plot_names, "bootstrap_ci95_low_pct"]
    highs = paired.set_index("policy").loc[plot_names, "bootstrap_ci95_high_pct"]
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = means.to_numpy(dtype=float); lower = vals-lows.to_numpy(dtype=float); upper=highs.to_numpy(dtype=float)-vals
    ax.errorbar(plot_names, vals, yerr=np.vstack([lower,upper]), fmt="o", capsize=5, color="#d95f32")
    ax.axhline(0,color="black",linewidth=1); ax.set_ylabel("Paired energy saving (%)")
    ax.set_title("Energy saving on shared successful scenarios (bootstrap 95% CI)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIG/"paired_energy_saving.png", dpi=200); plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,5))
    distribution_names=["teacher_best","student_best","student_original"]
    distribution=[savings_by_policy[name] for name in distribution_names if len(savings_by_policy[name])]
    labels=[name for name in distribution_names if len(savings_by_policy[name])]
    ax.boxplot(distribution,tick_labels=labels,showmeans=True,showfliers=True)
    ax.axhline(0,color="black",linewidth=1)
    ax.set_ylabel("Paired energy saving (%)")
    ax.set_title("Scenario-level savings on shared successes only; negative means higher policy energy")
    fig.tight_layout(); fig.savefig(FIG/"paired_energy_saving_distribution.png",dpi=200); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axes[0].plot(history.epoch,history.train_loss,label="Training loss")
    axes[0].plot(history.epoch,history.validation_loss,label="Trajectory validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Distillation objective"); axes[0].legend()
    axes[1].plot(history.epoch,history.validation_action_agreement*100,color="#087f8c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Teacher action agreement (%)")
    axes[1].set_title("Held-out trajectories")
    fig.savefig(FIG/"student_distillation_curves.png",dpi=200); plt.close(fig)

    fig, ax=plt.subplots(figsize=(8,4))
    checks=teacher_history.dropna(subset=["success"])
    ax.plot(checks.episode,checks.success.rolling(5,min_periods=1).mean()*100,marker="o")
    ax.set_xlabel("Fine-tuning episode"); ax.set_ylabel("Rolling 5-episode completion (%)")
    ax.set_title("Fine-tuned teacher (training episodes; final held-out score was 0/50)")
    fig.tight_layout(); fig.savefig(FIG/"teacher_finetuning.png",dpi=200); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    axes[0].bar(benchmark.model,benchmark.parameters,color=["#087f8c","#d95f32"])
    axes[0].set_ylabel("Parameters"); axes[0].set_title("Model parameter count")
    axes[1].bar(benchmark.model,benchmark.median_ms,color=["#087f8c","#d95f32"])
    axes[1].set_ylabel("Median forward pass (ms)"); axes[1].set_title("Batch-1 NN only; recorded Colab hardware")
    fig.savefig(FIG/"model_size_and_latency.png",dpi=200); plt.close(fig)

    fig, ax=plt.subplots(figsize=(6,4))
    parameter_labels=["Teacher","Student"]
    parameter_values=benchmark.set_index("model").loc[["teacher","student"],"parameters"].to_numpy()
    bars=ax.bar(parameter_labels,parameter_values,color=["#087f8c","#d95f32"])
    ax.set_ylabel("Trainable parameters")
    ax.set_title("Student parameter reduction")
    reduction=(1-parameter_values[1]/parameter_values[0])*100
    ax.text(0.5,max(parameter_values)*0.72,f"{reduction:.2f}% fewer",ha="center",fontsize=12)
    for bar,value in zip(bars,parameter_values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{int(value):,}",ha="center",va="bottom")
    fig.tight_layout(); fig.savefig(FIG/"parameter_reduction.png",dpi=200); plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,4))
    ax.bar(summary.policy,summary.aggregate_energy_per_completed_task_j,color=[colors.get(x,"#777") for x in summary.policy])
    ax.set_ylabel("Total attempt energy / completed tasks (J/task)")
    ax.set_title("Aggregate energy per successful completion; undefined for 0-completion policies")
    ax.tick_params(axis="x",rotation=35)
    fig.tight_layout(); fig.savefig(FIG/"aggregate_energy_per_completion.png",dpi=200); plt.close(fig)

    # Numeric integrity checks against headline values, never hard-code outputs.
    indexed = summary.set_index("policy")
    result = {
        "outer_deadline_rows": int(len(episodes)), "policy_count": int(episodes.policy.nunique()),
        "heldout_seed_count": len(common_seeds), "heldout_seed_min": min(common_seeds),
        "heldout_seed_max": max(common_seeds), "duplicate_policy_seed_rows": 0,
        "fixed_original_completions": int(indexed.loc["fixed_original","completions"]),
        "teacher_original_completions": int(indexed.loc["teacher_original","completions"]),
        "teacher_best_completions": int(indexed.loc["teacher_best","completions"]),
        "teacher_finetuned_completions": int(indexed.loc["teacher_finetuned","completions"]),
        "student_original_completions": int(indexed.loc["student_original","completions"]),
        "student_best_completions": int(indexed.loc["student_best","completions"]),
        "student_teacher_completion_retention_pct": float(indexed.loc["student_best","completion_rate"] / indexed.loc["teacher_best","completion_rate"] * 100),
        "teacher_best_mean_paired_saving_pct": float(paired.set_index("policy").loc["teacher_best","mean_paired_energy_saving_pct"]),
        "student_best_mean_paired_saving_pct": float(paired.set_index("policy").loc["student_best","mean_paired_energy_saving_pct"]),
        "fine_tuned_teacher_mean_energy_saving_pct": None,
        "note": "Fine-tuned teacher has no shared successful episodes, so success-conditional energy saving is unavailable; its failed-attempt energy remains in summary.",
        "paired_energy_denominator": "shared successful scenario seeds only; (fixed IK grid J - policy grid J) / fixed IK grid J * 100",
        "completion_interval": "Wilson score 95% interval",
        "energy_interval": f"percentile bootstrap 95% CI, {BOOTSTRAP_REPLICATES} resamples, seed {BOOTSTRAP_SEED}",
        "source": "Recovered outer deadline final_episode_metrics.csv; separate nested checkpoint files are not mixed into these results.",
    }
    import json
    (OUT/"independent_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__ == "__main__":
    print(reconstruct())
