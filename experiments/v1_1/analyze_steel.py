"""Analyze the repository's existing UCI steel dataset without training on it."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze(csv_path: Path, output_dir: Path) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["date"], dayfirst=True)
    gaps = df["date"].sort_values().diff().dropna()
    spacing = gaps.mode().iloc[0]
    rows = {
        "source_local_path": str(csv_path), "rows": int(len(df)), "columns": list(df.columns),
        "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
        "start": df.date.min().isoformat(), "end": df.date.max().isoformat(),
        "modal_interval_seconds": float(spacing.total_seconds()),
        "irregular_intervals": int((gaps != spacing).sum()),
        "usage_kwh_summary": {k: float(v) for k, v in df.Usage_kWh.describe().items()},
        "load_counts": {str(k): int(v) for k, v in df.Load_Type.value_counts().items()},
        "load_usage_kwh": {str(k): {name: float(val) for name, val in vals.items()}
                            for k, vals in df.groupby("Load_Type").Usage_kWh.agg(
                                count="count", mean="mean", median="median", max="max").to_dict("index").items()},
    }
    interval_hours = spacing.total_seconds() / 3600.0
    df["average_power_kw"] = df.Usage_kWh / interval_hours
    df["hour"] = df.date.dt.hour
    df["date_only"] = df.date.dt.date
    df["weekday"] = df.date.dt.day_name()
    demand = df.groupby("hour").average_power_kw.agg(["mean", "median", "max"]).reset_index()
    demand.to_csv(output_dir / "steel_hourly_power_summary.csv", index=False)
    peaks = df.nlargest(max(1, int(round(len(df) * 0.01))), "average_power_kw")
    peaks[["date", "Usage_kWh", "average_power_kw", "Load_Type"]].to_csv(output_dir / "steel_top_1pct_intervals.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    df.groupby("Load_Type", observed=False).average_power_kw.mean().sort_values().plot.bar(ax=axes[0, 0], color="#167d88")
    axes[0, 0].set_title("Mean interval power by recorded load category"); axes[0, 0].set_ylabel("kW")
    demand.plot(x="hour", y="mean", ax=axes[0, 1], color="#e97942", legend=False)
    axes[0, 1].set_title("Mean power by hour of day"); axes[0, 1].set_ylabel("kW"); axes[0, 1].set_xlabel("Hour")
    axes[1, 0].hist(df.average_power_kw, bins=50, color="#317b99", alpha=.9)
    axes[1, 0].set_title("Interval power distribution"); axes[1, 0].set_xlabel("kW")
    axes[1, 1].plot(df.date, df.average_power_kw, linewidth=.35, alpha=.7)
    axes[1, 1].axhline(df.average_power_kw.quantile(.99), linestyle="--", color="#c9473f", label="99th percentile")
    axes[1, 1].set_title("Recorded demand and upper-tail threshold"); axes[1, 1].set_ylabel("kW"); axes[1, 1].legend()
    fig.suptitle("UCI Steel Industry Energy Consumption — descriptive context")
    fig.savefig(output_dir / "steel_demand_analysis.png", dpi=180)
    plt.close(fig)
    (output_dir / "steel_analysis_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "experiments" / "v1_1" / "runs" / "2026-11-26" / "dataset_analysis"
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(analyze(root / "notebooks" / "steel_data.csv", out), indent=2))
