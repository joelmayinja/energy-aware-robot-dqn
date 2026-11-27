"""Check notebook JSON/code syntax and integrity of its embedded source bundle."""
from __future__ import annotations

import ast
import base64
import io
import json
import re
import zipfile
from pathlib import Path


root = Path(__file__).resolve().parents[2]
notebook_path = root / "FINAL_RESEARCH_TRAINING.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
for code in codes:
    ast.parse(code)
run_source = "\n".join(codes)
for required_fix in (
    'OUTPUT = BASE_OUTPUT / RUN_MODE',
    'np.asarray(build_discrete_action_table(6),dtype=np.float32).tolist()',
    'baseline = results[results.policy == "fixed_original"]',
    'dynamic_axes={"observation":{0:"batch_size"}',
    'aggregate_energy_per_completed_task=("grid_energy_j",lambda x: float(x.sum()/int(results.loc[x.index,"success"].sum())) if int(results.loc[x.index,"success"].sum())>0 else np.nan)',
):
    assert required_fix in run_source, f"Expected V1.1 regression fix absent from notebook: {required_fix}"
match = re.search(r'EMBEDDED_BUNDLE_B64 = "([A-Za-z0-9+/=]+)"', codes[0])
assert match, "Embedded source archive variable not found"
bundle = base64.b64decode(match.group(1), validate=True)
with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
    assert zf.testzip() is None
    names = set(zf.namelist())
    required = {"src/environment.py", "src/dqn_teacher.py", "experiments/v1_1/pipeline.py",
                "experiments/v1_1/environment_v1_1.py", "models/teacher.pt",
                "models/student_original.pt", "notebooks/steel_data.csv"}
    assert required <= names, f"Missing embedded files: {required - names}"
print({"notebook": str(notebook_path), "cells": len(notebook["cells"]),
       "code_cells_parse": len(codes), "embedded_bundle_bytes": len(bundle),
       "embedded_entries": len(names), "zip_crc": "passed"})
