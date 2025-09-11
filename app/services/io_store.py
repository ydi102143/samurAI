# app/services/io_store.py
from __future__ import annotations
from pathlib import Path
import json, os, time
from typing import Any, Dict, Optional

ROOT = Path("storage")
DATASETS = ROOT / "datasets"
META = DATASETS / "_meta" / "problem_meta.json"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Optional[dict]:
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def write_json(p: Path, obj: Any):
    _ensure_dir(p.parent)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)

# ---- タスク/県メタの取得 ----
def get_problem_info(problem: str) -> dict:
    meta = read_json(META) or {}
    return meta.get(problem, {})

def set_problem_info(problem: str, info: dict):
    meta = read_json(META) or {}
    meta[problem] = info
    write_json(META, meta)

# ---- レベル操作 ----
def get_level(problem: str) -> int:
    return int(get_problem_info(problem).get("level", 1))

def set_level(problem: str, level: int) -> int:
    info = get_problem_info(problem)
    info["level"] = int(level)
    set_problem_info(problem, info)
    return int(level)

def bump_level(problem: str, step: int = 1) -> int:
    lv = get_level(problem)
    return set_level(problem, lv + step)

# ---- processed/pipeline/feature パス ----
def problem_region(problem: str) -> str:
    info = get_problem_info(problem)
    region = info.get("region")
    if not region:
        # region-less fallback: first region that has <problem>.csv
        for r in (DATASETS.iterdir() if DATASETS.exists() else []):
            if (r / f"{problem}.csv").exists():
                return r.name
        raise FileNotFoundError(f"Region for {problem} not found")
    return region

def raw_csv(problem: str) -> Path:
    r = problem_region(problem)
    p = DATASETS / r / f"{problem}.csv"
    if not p.exists():
        raise FileNotFoundError(f"raw csv not found: {p}")
    return p

def proc_dir(problem: str) -> Path:
    return raw_csv(problem).parent / "_processed" / problem

def proc_csv(problem: str) -> Path:
    return proc_dir(problem) / "latest.csv"

def pipe_json(problem: str) -> Path:
    return proc_dir(problem) / "pipeline.json"

def feat_json(problem: str) -> Path:
    return proc_dir(problem) / "feature_selection.json"

# ---- モデルカード置き場 ----
def model_dir(problem: str) -> Path:
    r = problem_region(problem)
    return ROOT / "models" / r / problem

def cards_json(problem: str) -> Path:
    return model_dir(problem) / "cards.json"
