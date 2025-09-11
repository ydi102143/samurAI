from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

from .io_store import pipeline_json, proc_csv, raw_csv, read_json, write_json

# ---- pipeline I/O ----
def load_pipeline(problem: str) -> list[dict]:
    return read_json(pipeline_json(problem), [])

def save_pipeline(problem: str, steps: list[dict]):
    write_json(pipeline_json(problem), steps)

def has_pipeline(problem: str) -> bool:
    return len(load_pipeline(problem)) > 0

# ---- transforms ----
def _cols_or_infer(df: pd.DataFrame, cols: list[str] | None, want: str) -> list[str]:
    if cols:
        return [c for c in cols if c in df.columns]
    if want == "number":
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if want == "category":
        return [c for c in df.columns if df[c].dtype == "object"]
    return list(df.columns)

def _apply_missing(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    rules = step.get("rules")
    out = df.copy()
    if rules:
        for r in rules:
            col = r.get("col")
            if not col or col not in out.columns: 
                continue
            stg = (r.get("strategy") or "auto").lower()
            s = out[col]
            is_num = pd.api.types.is_numeric_dtype(s)
            if stg == "constant":
                out[col] = s.fillna(r.get("value"))
            elif stg == "mean" and is_num:
                out[col] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
            elif stg == "median" and is_num:
                out[col] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).median())
            elif stg == "mode":
                m = s.mode(dropna=True)
                val = m.iloc[0] if len(m) else (0.0 if is_num else "")
                out[col] = s.fillna(val)
            elif stg == "zero" and is_num:
                out[col] = s.fillna(0.0)
            elif stg == "ffill":
                out[col] = s.ffill()
            elif stg == "bfill":
                out[col] = s.bfill()
            else:
                if is_num:
                    out[col] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
                else:
                    m = s.mode(dropna=True)
                    val = m.iloc[0] if len(m) else ""
                    out[col] = s.fillna(val).replace("", val)
        return out

    # bulk
    stg = (step.get("strategy") or "auto").lower()
    cols = _cols_or_infer(df, step.get("columns"), "any")
    val = step.get("value", None)
    for c in cols:
        if c not in out.columns: continue
        s = out[c]; is_num = pd.api.types.is_numeric_dtype(s)
        if stg == "constant": out[c] = s.fillna(val)
        elif stg == "mean" and is_num:
            out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
        elif stg == "median" and is_num:
            out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).median())
        elif stg == "mode":
            m = s.mode(dropna=True); v = m.iloc[0] if len(m) else (0.0 if is_num else "")
            out[c] = s.fillna(v)
        elif stg == "zero" and is_num:
            out[c] = s.fillna(0.0)
        elif stg == "ffill": out[c] = s.ffill()
        elif stg == "bfill": out[c] = s.bfill()
        else:
            if is_num:
                out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
            else:
                m = s.mode(dropna=True); v = m.iloc[0] if len(m) else ""
                out[c] = s.fillna(v).replace("", v)
    return out

def _apply_onehot(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    cols = _cols_or_infer(df, step.get("columns"), "category")
    if not cols: return df
    base = df.copy()
    for c in cols:
        if c in base.columns:
            base[c] = base[c].astype(str).replace({"": "__NA__"}).fillna("__NA__")
    dummies = pd.get_dummies(base[cols], prefix=cols, dummy_na=False)
    return pd.concat([base.drop(columns=[c for c in cols if c in base.columns]), dummies], axis=1)

def _apply_scale(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    kind = step.get("kind")  # "standard" | "minmax"
    cols = _cols_or_infer(df, step.get("columns"), "number")
    out = df.copy()
    for c in cols:
        if c not in out.columns: continue
        x = pd.to_numeric(out[c], errors="coerce")
        if kind == "standard":
            mu = float(np.nanmean(x)); sd = float(np.nanstd(x)) or 1.0
            out[c] = (x - mu) / sd
        else:
            lo = float(np.nanmin(x)); hi = float(np.nanmax(x)); rng = (hi - lo) or 1.0
            out[c] = (x - lo) / rng
    return out

# ---- materialize processed ----
def rebuild_processed(problem: str) -> Path:
    df = pd.read_csv(raw_csv(problem), encoding="utf-8")
    for s in load_pipeline(problem):
        t = s.get("type")
        if t == "missing": df = _apply_missing(df, s)
        elif t == "onehot": df = _apply_onehot(df, s)
        elif t == "scale":  df = _apply_scale(df, s)
    out = proc_csv(problem)
    # date を安全に文字列化
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df2.to_csv(out.with_suffix(".tmp"), index=False, encoding="utf-8")
    out.with_suffix(".tmp").replace(out)
    return out

def ensure_processed(problem: str) -> Path:
    p = proc_csv(problem)
    return p if p.exists() else rebuild_processed(problem)