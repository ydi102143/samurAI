# app/routers/offline.py — オフライン（ページ + API）
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

# ===== ログ =====
logger = logging.getLogger(__name__)

# ===== パス定義 =====
DATA_ROOT = Path("storage/datasets")
MODELS_ROOT = Path("storage/models")
META_PREF_PATH = DATA_ROOT / "_meta" / "pref_meta.json"       # 既存互換
META_PROBLEM_PATH = DATA_ROOT / "_meta" / "problem_meta.json"

# ===== Jinja (ページ用) =====
page_router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["time"] = time

# ===== ページ: /offline/play/{problem} =====
@page_router.get("/offline/play/{problem}", response_class=HTMLResponse)
def offline_play_page(request: Request, problem: str):
  region = "any"
  ts = int(time.time())
  return templates.TemplateResponse(
      "offline.html",
      {"request": request, "problem": problem, "pref": problem, "region": region, "ts": ts},
  )

# ===== API ルータ（prefix=/v1/datasets） =====
router = APIRouter(prefix="/v1/datasets", tags=["datasets"])

# 透明1x1 PNG（プレースホルダー用）
_transparent_png = base64.b64decode(
  b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

# ===== メタ情報 =====
_meta_cache: Dict[str, Any] = {}
_meta_mtime: Optional[float] = None

def _meta_path() -> Path:
  # problem_meta.json 優先、なければ pref_meta.json
  return META_PROBLEM_PATH if META_PROBLEM_PATH.exists() else META_PREF_PATH

def _load_meta() -> Dict[str, Any]:
  global _meta_cache, _meta_mtime
  mpath = _meta_path()
  if not mpath.exists():
    return {}
  mt = mpath.stat().st_mtime
  if _meta_mtime != mt:
    _meta_cache = json.loads(mpath.read_text(encoding="utf-8"))
    _meta_mtime = mt
  return _meta_cache

def _filter_models_by_task(models: List[str], task_type: str) -> List[str]:
  if task_type == "regression":
    keep = ("Regressor", "LinearRegression", "SVR", "KNNRegressor")
  else:
    keep = ("Classifier", "LogisticRegression", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier")
  out = [m for m in (models or []) if any(k in m for k in keep)]
  if not out:
    out = ["LinearRegression"] if task_type == "regression" else ["LogisticRegression"]
  return out

# ===== 互換: /v1/region/task（prefixなしで提供） =====
@page_router.get("/v1/region/task")
def region_task(problem: str | None = Query(None), pref: str | None = Query(None)):
  """
  UI 互換エンドポイント。problem_meta.json / pref_meta.json から
  指定 problem の task/metric/allowed を返す。
  """
  pb = (problem or pref or "").strip().lower()
  if not pb:
    raise HTTPException(400, "problem is required")

  meta = _load_meta().get(pb)
  if not meta:
    raise HTTPException(404, f"problem meta not found for '{pb}'")

  tgt = meta.get("target", {"name": "y", "type": "classification"})
  task_type = tgt.get("type", "classification")
  allowed_models = _filter_models_by_task(meta.get("allowed_models", []), task_type)

  return {
    "problem": pb,
    "level": meta.get("level", 1),
    "metric": meta.get("metric", "accuracy" if task_type == "classification" else "r2"),
    "target": tgt,  # {"name":"y","type":"classification"|"regression"}
    "allowed_features": meta.get("allowed_features", []),
    "allowed_models": allowed_models,
  }

# ===== ヘルパ =====
def _region_of(problem: str) -> str:
  meta = _load_meta()
  if problem in meta:
    return meta[problem].get("region", "any")
  # fallback: ディレクトリ探索（古い構成互換）
  for rdir in DATA_ROOT.iterdir():
    if (rdir / f"{problem}.csv").exists():
      return rdir.name
  raise HTTPException(404, f"region not found for '{problem}'")

def _raw_csv(problem: str) -> Path:
  r = _region_of(problem)
  p = DATA_ROOT / r / f"{problem}.csv"
  if not p.exists():
    raise HTTPException(404, f"raw csv not found: {p}")
  return p

def _proc_dir(problem: str) -> Path:
  r = _region_of(problem)
  return DATA_ROOT / r / "_processed" / problem

def _proc_csv(problem: str) -> Path:
  return _proc_dir(problem) / "latest.csv"

def _pipe_json(problem: str) -> Path:
  return _proc_dir(problem) / "pipeline.json"

def _feat_json(problem: str) -> Path:
  return _proc_dir(problem) / "feature_selection.json"

def _cards_json(problem: str) -> Path:
  r = _region_of(problem)
  return MODELS_ROOT / r / problem / "cards.json"

def _model_path(problem: str, name: str) -> Path:
  r = _region_of(problem)
  return MODELS_ROOT / r / problem / f"model__{name}.pkl"

def _safe_write_csv(df: pd.DataFrame, out: Path):
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_suffix(".tmp")
  df2 = df.copy()
  if "date" in df2.columns:
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
  df2.to_csv(tmp, index=False, encoding="utf-8")
  os.replace(tmp, out)

def _load_pipeline(problem: str) -> List[Dict[str, Any]]:
  p = _pipe_json(problem)
  if p.exists():
    try:
      return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
      return []
  return []

def _save_pipeline(problem: str, steps: List[Dict[str, Any]]):
  pj = _pipe_json(problem)
  pj.parent.mkdir(parents=True, exist_ok=True)
  pj.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")

def _cols_or_infer(df: pd.DataFrame, cols: Optional[List[str]], want: str) -> List[str]:
  if cols:
    return [c for c in cols if c in df.columns]
  if want == "number":
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
  if want == "category":
    return [c for c in df.columns if df[c].dtype == "object"]
  return list(df.columns)

def _apply_missing(df: pd.DataFrame, step: dict) -> pd.DataFrame:
  out = df.copy()
  rules = step.get("rules")
  if rules:
    for r in rules:
      col = r.get("col")
      stg = (r.get("strategy") or "").lower()
      if not col or col not in out.columns:
        continue
      s = out[col]
      if stg == "mean":
        out[col] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
      elif stg == "median":
        out[col] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).median())
      elif stg == "mode":
        m = s.mode(dropna=True)
        out[col] = s.fillna(m.iloc[0] if len(m) else (0.0 if pd.api.types.is_numeric_dtype(s) else ""))
      elif stg == "constant":
        out[col] = s.fillna(r.get("value", 0))
      elif stg == "ffill":
        out[col] = s.ffill()
      elif stg == "bfill":
        out[col] = s.bfill()
    return out

  stg = (step.get("strategy") or "auto").lower()
  cols = _cols_or_infer(df, step.get("columns"), "any")
  val = step.get("value", None)
  for c in cols:
    if c not in out.columns:
      continue
    s = out[c]
    is_num = pd.api.types.is_numeric_dtype(s)
    if stg == "constant":
      out[c] = s.fillna(val)
    elif stg == "mean" and is_num:
      out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
    elif stg == "median" and is_num:
      out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).median())
    elif stg == "mode":
      m = s.mode(dropna=True)
      out[c] = s.fillna(m.iloc[0] if len(m) else (0.0 if is_num else ""))
    elif stg == "ffill":
      out[c] = s.ffill()
    elif stg == "bfill":
      out[c] = s.bfill()
    else:
      if is_num:
        out[c] = s.fillna(pd.to_numeric(s, errors="coerce").astype(float).mean())
      else:
        m = s.mode(dropna=True)
        fillv = m.iloc[0] if len(m) else ""
        out[c] = s.fillna(fillv).replace("", fillv)
  return out

def _apply_onehot(df: pd.DataFrame, step: dict) -> pd.DataFrame:
  cols = _cols_or_infer(df, step.get("columns"), "category")
  if not cols:
    return df
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
    if c not in out.columns:
      continue
    x = pd.to_numeric(out[c], errors="coerce")
    if kind == "standard":
      mu = float(np.nanmean(x))
      sd = float(np.nanstd(x)) or 1.0
      out[c] = (x - mu) / sd
    else:
      lo = float(np.nanmin(x))
      hi = float(np.nanmax(x))
      rng = (hi - lo) or 1.0
      out[c] = (x - lo) / rng
  return out

def _rebuild_processed(problem: str) -> Path:
  df = pd.read_csv(_raw_csv(problem), encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  for s in _load_pipeline(problem):
    t = s.get("type")
    if t == "missing":
      df = _apply_missing(df, s)
    elif t == "onehot":
      df = _apply_onehot(df, s)
    elif t == "scale":
      df = _apply_scale(df, s)
  _safe_write_csv(df, _proc_csv(problem))
  return _proc_csv(problem)

def _ensure_processed(problem: str) -> Path:
  pcsv = _proc_csv(problem)
  return pcsv if pcsv.exists() else _rebuild_processed(problem)

def _build_dataframe_for_train(problem: str) -> pd.DataFrame:
  p = _ensure_processed(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  # feature_selection.json を読む
  fj = _feat_json(problem)
  num_cols, cat_cols = [], []
  if fj.exists():
    try:
      js = json.loads(fj.read_text(encoding="utf-8"))
      num_cols = list(js.get("num_cols") or [])
      cat_cols = list(js.get("cat_cols") or [])
    except Exception:
      num_cols, cat_cols = [], []

  if num_cols or cat_cols:
    feats = [c for c in (num_cols + cat_cols) if c in df.columns]
    keep = (["date"] if "date" in df.columns else []) + feats + (["y"] if "y" in df.columns else [])
    if keep:
      df = df[keep].copy()
  return df

def _xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
  if "y" not in df.columns:
    raise HTTPException(400, "y not found")
  X = df.drop(columns=[c for c in ["y", "date"] if c in df.columns])
  y = df["y"]
  if pd.api.types.is_numeric_dtype(y):
    y = pd.to_numeric(y, errors="coerce")
  return X, y

def _task_of(problem: str) -> Tuple[str, str]:
  meta = _load_meta().get(problem, {})
  target = meta.get("target") or {}
  task = target.get("type", "classification")
  metric = meta.get("metric", "accuracy" if task == "classification" else "r2")
  return task, metric

def _png_response(png_bytes: bytes) -> Response:
  return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})

# ====== Matplotlib（サーバ用バックエンド）======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  mean_absolute_error,
  mean_squared_error,
  r2_score,
  roc_auc_score,
  roc_curve,
)

# ---------- 欠損率バー（前処理タブ左上に表示） ----------
def _plot_missing_bar(df: pd.DataFrame) -> StreamingResponse:
  na_ratio = (df.isna().mean() * 100.0).fillna(0.0).sort_values(ascending=False)
  fig = plt.figure(figsize=(max(6, 0.4 * max(1, len(na_ratio))), 4))
  ax = fig.add_subplot(111)
  na_ratio.plot(kind="bar", ax=ax, edgecolor="black")
  ax.set_ylabel("欠損率 (%)")
  ax.set_title("欠損値の割合（列ごと）")
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return StreamingResponse(buf, media_type="image/png")

@router.get("/{region}/eda/plot/missing")
def eda_plot_missing(region: str, problem: str = Query(None), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  """欠損率バーを返す（problem 未指定/ファイル無は透明PNG）"""
  if not problem:
    return _png_response(_transparent_png)
  try:
    p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
    df = pd.read_csv(p, encoding="utf-8")
    if "date" in df.columns:
      df = df.drop(columns=["date"])
    return _plot_missing_bar(df)
  except Exception as e:
    logger.exception(f"[eda_plot_missing] region='{region}', problem='{problem}', source='{source}': {e}")
    return _png_response(_transparent_png)

# ===== EDA: スキーマ/記述統計/相関/単変量/散布図 =====
def _summarize(df: pd.DataFrame) -> List[dict]:
  out: List[dict] = []
  for c in df.columns:
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
      dtype = "number"
    elif s.dtype == "object":
      dtype = "category"
    else:
      dtype = str(s.dtype)
    out.append({"name": c, "dtype": dtype, "na_count": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))})
  return out

@router.get("/{region}/eda/schema")
def eda_schema(region: str, problem: str = Query(None), pref: str = Query(None), preview_n: int = 20, source: str = Query("auto", regex="^(raw|processed|auto)$")):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  pv_rows = df.head(preview_n).fillna("").to_dict(orient="records")
  return {"path": str(p), "rows": len(df), "preview": {"columns": list(df.columns), "rows": pv_rows}}

@router.get("/{region}/eda/describe")
def eda_describe(region: str, problem: str = Query(None), pref: str = Query(None), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  return {"summary": _summarize(df), "rows": len(df), "path": str(p), "source": source}

@router.get("/{region}/eda/plot/corr")
def eda_plot_corr(region: str, problem: str = Query(None), pref: str = Query(None), method: str = Query("pearson", regex="^(pearson|spearman|kendall)$"), max_cols: int = Query(40, ge=2, le=120), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  num_df = df.select_dtypes(include=[np.number]).copy()
  if num_df.shape[1] < 2:
    raise HTTPException(400, "numeric columns < 2")
  if num_df.shape[1] > max_cols:
    variances = num_df.var(numeric_only=True).sort_values(ascending=False)
    keep_cols = list(variances.index[:max_cols])
    num_df = num_df[keep_cols]
  corr = num_df.corr(method=method, numeric_only=True)
  n = corr.shape[0]
  fig_w = max(8, min(20, n * 0.6))
  fig_h = max(6, min(20, n * 0.6))
  fig = plt.figure(figsize=(fig_w, fig_h))
  ax = fig.add_subplot(111)
  im = ax.imshow(corr.values, vmin=-1, vmax=1)
  ax.set_xticks(range(n)); ax.set_yticks(range(n))
  ax.set_xticklabels(corr.columns, rotation=90)
  ax.set_yticklabels(corr.index)
  ax.set_title(f"Correlation heatmap ({method})")
  cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  cbar.ax.set_ylabel("corr", rotation=90, va="center")
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

@router.get("/{region}/eda/plot/univar")
def eda_plot_univariate(region: str, problem: str = Query(None), pref: str = Query(None), col: str = Query(...), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  if col not in df.columns:
    raise HTTPException(404, f"column not found: {col}")
  fig = plt.figure(figsize=(6, 3.6))
  ax = fig.add_subplot(111)
  s = df[col]
  if pd.api.types.is_numeric_dtype(s):
    s = pd.to_numeric(s, errors="coerce")
    s.plot(kind="hist", bins=30, ax=ax)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col); ax.set_ylabel("count")
  else:
    s.value_counts(dropna=False).head(20).plot(kind="bar", ax=ax)
    ax.set_title(f"Top categories: {col}")
    ax.set_xlabel(col); ax.set_ylabel("count")
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

@router.get("/{region}/eda/plot/scatter")
def eda_plot_scatter(region: str, problem: str = Query(None), pref: str = Query(None), x: str = Query(...), y: str = Query(...), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  if x not in df.columns or y not in df.columns:
    raise HTTPException(404, "x or y not found")
  xv = pd.to_numeric(df[x], errors="coerce")
  yv = pd.to_numeric(df[y], errors="coerce")
  fig = plt.figure(figsize=(6, 4))
  ax = fig.add_subplot(111)
  ax.scatter(xv, yv, s=10, alpha=0.7)
  ax.set_xlabel(x); ax.set_ylabel(y)
  ax.set_title(f"Scatter: {x} vs {y}")
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

# =========================================================
# Preprocess API
# =========================================================
@router.get("/{region}/preproc/columns/candidates")
def preproc_candidates(region: str, problem: str = Query(None), pref: str = Query(None), source: str = Query("auto", regex="^(raw|processed|auto)$")):
  """
  数値判定を堅牢化：
    - dtype が numeric なら数値
    - そうでなくても to_numeric で欠損を無視して 90%以上が数値化できれば数値扱い
    - それ以外はカテゴリ扱い
    - y は除外
  """
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")

  p = _ensure_processed(problem) if (source in ("processed", "auto")) else _raw_csv(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])

  num_cols: List[str] = []
  cat_cols: List[str] = []
  for c in df.columns:
    if c == "y":
      continue
    s = df[c]
    if pd.api.types.is_numeric_dtype(s):
      num_cols.append(c)
      continue
    sv = pd.to_numeric(s, errors="coerce")
    valid_ratio = float(sv.notna().mean()) if len(sv) else 0.0
    if valid_ratio >= 0.90:  # 90%以上数値にできるなら数値とみなす
      num_cols.append(c)
    else:
      cat_cols.append(c)

  all_cols = [c for c in df.columns if c != "y"]  # y を除外
  return {
    "num": num_cols,
    "cat": cat_cols,
    "all": all_cols,
    # 互換キー（古いフロント/ツール対策）
    "numeric": num_cols,
    "categorical": cat_cols,
  }

@router.post("/{region}/preproc/reset")
def preproc_reset(region: str, problem: str):
  raw = _raw_csv(problem)
  out = _proc_csv(problem)
  out.parent.mkdir(parents=True, exist_ok=True)
  df = pd.read_csv(raw, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  _safe_write_csv(df, out)

  # 履歴クリア
  pjson = _pipe_json(problem)
  fjson = _feat_json(problem)
  if pjson.exists():
    pjson.unlink()
  if fjson.exists():
    fjson.unlink()

  return {"ok": True, "reset_to": str(raw), "processed": str(out)}

@router.post("/{region}/preproc/missing/save")
async def preproc_missing_save(region: str, request: Request, problem: str = Query(None), pref: str = Query(None), strategy: str | None = Query(None), columns: str | None = None, value: float | None = None):
  """
  (A) BODY: {"rules":[{"col":"a","strategy":"median"}, {"col":"b","strategy":"constant","value":0}]}
  (B) 一括: ?strategy=mean&columns=col1,col2 （constant は &value=0）
  """
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")

  rules = None
  try:
    body = await request.json()
    rules = (body or {}).get("rules")
  except Exception:
    pass

  steps = [s for s in _load_pipeline(problem) if s.get("type") != "missing"]
  if rules:
    steps.append({"type": "missing", "rules": rules})
  else:
    cols = [c for c in (columns.split(",") if columns else []) if c]
    s = {"type": "missing", "strategy": (strategy or "auto"), "columns": cols}
    if (strategy or "").lower() == "constant":
      s["value"] = value
    steps.append(s)

  _save_pipeline(problem, steps)
  out = _rebuild_processed(problem)
  return {"ok": True, "path": str(out), "steps": steps}

@router.post("/{region}/preproc/encode/save")
def preproc_encode_save(region: str, problem: str = Query(None), pref: str = Query(None), columns: str | None = None):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  cols = [c for c in (columns.split(",") if columns else []) if c]
  steps = [s for s in _load_pipeline(problem) if s.get("type") != "onehot"]
  if cols:
    steps.append({"type": "onehot", "columns": cols})
  _save_pipeline(problem, steps)
  out = _rebuild_processed(problem)
  return {"ok": True, "path": str(out), "steps": steps}

@router.post("/{region}/preproc/scale/save")
def preproc_scale_save(region: str, kind: str = Query(..., regex="^(standard|minmax)$"), problem: str = Query(None), pref: str = Query(None), columns: str | None = None):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  cols = [c for c in (columns.split(",") if columns else []) if c]
  steps = [s for s in _load_pipeline(problem) if not (s.get("type") == "scale" and s.get("kind") == kind)]
  if cols:
    steps.append({"type": "scale", "kind": kind, "columns": cols})
  _save_pipeline(problem, steps)
  out = _rebuild_processed(problem)
  return {"ok": True, "path": str(out), "steps": steps}

# 互換: /preproc/scale（BODYに columns を持つ派生実装を吸収）
@router.post("/{region}/preproc/scale")
def preproc_scale_compat(region: str, kind: str = Query(..., regex="^(standard|minmax)$"), problem: str = Query(None), pref: str = Query(None), body: dict = Body(...)):
  problem = problem or pref
  cols = ",".join([c for c in (body.get("columns") or []) if c])
  return preproc_scale_save(region=region, kind=kind, problem=problem, pref=None, columns=cols)

@router.post("/{region}/preproc/feature/save")
def preproc_feature_save(region: str, problem: str = Query(None), pref: str = Query(None), body: dict = {}):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  num_cols = list(body.get("num_cols") or [])
  cat_cols = list(body.get("cat_cols") or [])
  fj = _feat_json(problem)
  fj.parent.mkdir(parents=True, exist_ok=True)
  fj.write_text(json.dumps({"num_cols": num_cols, "cat_cols": cat_cols}, ensure_ascii=False, indent=2), encoding="utf-8")
  return {"ok": True, "saved": {"num_cols": num_cols, "cat_cols": cat_cols}}

# =========================================================
# モデル作成・評価（UI短名エイリアス対応版）
# =========================================================
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  RandomForestClassifier,
  RandomForestRegressor,
  VotingClassifier,
  VotingRegressor,
  StackingClassifier,
  StackingRegressor,
)

def _lib_ok(name: str) -> bool:
  try:
    __import__(name)
    return True
  except Exception:
    return False

# ---- UI短名 → 実装クラス名のエイリアス（タスク別に解決） ----
_ALIAS: Dict[str, Dict[str, str]] = {
  "classification": {
    "LogisticRegression": "LogisticRegression",
    "DecisionTree": "DecisionTreeClassifier",
    "KNN": "KNeighborsClassifier",
    "SVM": "SVC",
    "RandomForest": "RandomForestClassifier",
    "GradientBoosting": "GradientBoostingClassifier",
    "XGBoost": "XGBoostClassifier",
    "LightGBM": "LightGBMClassifier",
    "CatBoost": "CatBoostClassifier",
  },
  "regression": {
    "LinearRegression": "LinearRegression",
    "DecisionTree": "DecisionTreeRegressor",
    "KNN": "KNeighborsRegressor",
    "SVM": "SVR",
    "RandomForest": "RandomForestRegressor",
    "GradientBoosting": "GradientBoostingRegressor",
    "XGBoost": "XGBoostRegressor",
    "LightGBM": "LightGBMRegressor",
    "CatBoost": "CatBoostRegressor",
  },
}

def _resolve_impl(name: str, task: str) -> str:
  """UI短名/実装名のどちらでも受け、実装名に正規化する"""
  if name in (_ALIAS.get(task) or {}):
    return _ALIAS[task][name]
  return name  # すでに実装名ならそのまま

# ---- カタログ（UIは短名を期待） ----
_CLS_BASIC = [
  dict(name="LogisticRegression", label="ロジスティック回帰"),
  dict(name="DecisionTree",      label="決定木"),
  dict(name="KNN",               label="K近傍法"),
  dict(name="SVM",               label="SVM"),
]
_REG_BASIC = [
  dict(name="LinearRegression",  label="線形回帰"),
  dict(name="DecisionTree",      label="決定木"),
  dict(name="KNN",               label="K近傍法"),
  dict(name="SVM",               label="SVM回帰"),
]
_BAG = [ dict(name="RandomForest", label="ランダムフォレスト") ]
_BOOST: List[Dict[str, str]] = [ dict(name="GradientBoosting", label="勾配ブースティング") ]
if _lib_ok("xgboost"):
  _BOOST += [dict(name="XGBoost", label="XGBoost")]
if _lib_ok("lightgbm"):
  _BOOST += [dict(name="LightGBM", label="LightGBM")]
if _lib_ok("catboost"):
  _BOOST += [dict(name="CatBoost", label="CatBoost")]

_STACK = [
  dict(name="StackingClassifier", label="スタッキング（分類）"),
  dict(name="StackingRegressor",  label="スタッキング（回帰）"),
]

# ---- パラメータスキーマ（短名も同居） ----
_PARAM_DB = {
  "LogisticRegression": {"schema": {"C": {"type": "float", "default": 1.0, "range": [0.001, 100.0], "step": 0.001},
                                    "penalty": {"type": "select", "choices": ["l2"], "default": "l2"}}},
  "DecisionTreeClassifier": {"schema": {"max_depth": {"type": "int", "default": 6, "range": [1, 20]},
                                        "min_samples_split": {"type": "int", "default": 2, "range": [2, 50]}}},
  "KNeighborsClassifier": {"schema": {"n_neighbors": {"type": "int", "default": 5, "range": [1, 50]}}},
  "SVC": {"schema": {"C": {"type": "float", "default": 1.0, "range": [0.001, 100.0], "step": 0.001},
                     "kernel": {"type": "select", "choices": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"}}},
  "LinearRegression": {"schema": {}},
  "DecisionTreeRegressor": {"schema": {"max_depth": {"type": "int", "default": 6, "range": [1, 20]}}},
  "KNeighborsRegressor": {"schema": {"n_neighbors": {"type": "int", "default": 5, "range": [1, 50]}}},
  "SVR": {"schema": {"C": {"type": "float", "default": 1.0, "range": [0.001, 100.0], "step": 0.001},
                     "kernel": {"type": "select", "choices": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf"}}},
  "RandomForestClassifier": {"schema": {"n_estimators": {"type": "int", "default": 200, "range": [10, 1000]},
                                        "max_depth": {"type": "int", "default": 8, "range": [1, 50]}}},
  "RandomForestRegressor": {"schema": {"n_estimators": {"type": "int", "default": 200, "range": [10, 1000]},
                                       "max_depth": {"type": "int", "default": 8, "range": [1, 50]}}},
  "GradientBoostingClassifier": {"schema": {"n_estimators": {"type": "int", "default": 150, "range": [10, 1000]},
                                            "learning_rate": {"type": "float", "default": 0.1, "range": [0.001, 1.0], "step": 0.001}}},
  "GradientBoostingRegressor": {"schema": {"n_estimators": {"type": "int", "default": 150, "range": [10, 1000]},
                                           "learning_rate": {"type": "float", "default": 0.1, "range": [0.001, 1.0], "step": 0.001}}},
  "XGBoostClassifier": {"schema": {"n_estimators": {"type": "int", "default": 300, "range": [50, 2000]},
                                   "max_depth": {"type": "int", "default": 6, "range": [1, 16]}}},
  "XGBoostRegressor": {"schema": {"n_estimators": {"type": "int", "default": 300, "range": [50, 2000]},
                                  "max_depth": {"type": "int", "default": 6, "range": [1, 16]}}},
  "LightGBMClassifier": {"schema": {"n_estimators": {"type": "int", "default": 300, "range": [50, 2000]},
                                    "num_leaves": {"type": "int", "default": 31, "range": [7, 255]}}},
  "LightGBMRegressor": {"schema": {"n_estimators": {"type": "int", "default": 300, "range": [50, 2000]},
                                   "num_leaves": {"type": "int", "default": 31, "range": [7, 255]}}},
  "CatBoostClassifier": {"schema": {"iterations": {"type": "int", "default": 300, "range": [50, 2000]},
                                    "depth": {"type": "int", "default": 6, "range": [1, 10]}}},
  "CatBoostRegressor": {"schema": {"iterations": {"type": "int", "default": 300, "range": [50, 2000]},
                                   "depth": {"type": "int", "default": 6, "range": [1, 10]}}},
  "StackingClassifier": {"schema": {}},
  "StackingRegressor": {"schema": {}},
}
_PARAM_DB.update({
  "DecisionTree":     _PARAM_DB["DecisionTreeClassifier"],
  "KNN":              _PARAM_DB["KNeighborsClassifier"],
  "SVM":              _PARAM_DB["SVC"],
  "RandomForest":     _PARAM_DB["RandomForestClassifier"],
  "GradientBoosting": _PARAM_DB["GradientBoostingClassifier"],
  "XGBoost":          _PARAM_DB.get("XGBoostClassifier", {"schema": {}}),
  "LightGBM":         _PARAM_DB.get("LightGBMClassifier", {"schema": {}}),
  "CatBoost":         _PARAM_DB.get("CatBoostClassifier", {"schema": {}}),
})

@router.get("/{region}/models/params")
def model_params(region: str, model: str):
  """UIの短名/実装名どちらでも受け、スキーマを返す"""
  return _PARAM_DB.get(model, {"schema": {}})

@router.get("/{region}/models/catalog")
def models_catalog(region: str, problem: str):
  """UIは短名を期待するので短名で返す。解決はサーバ側で行う。"""
  meta = region_task(problem=problem)  # ← 既存の region_task を利用
  is_cls = (meta["target"]["type"] == "classification")
  cat = {
    "basic":   _CLS_BASIC if is_cls else _REG_BASIC,
    "bagging": _BAG,
    "boosting": _BOOST,
    "stacking": _STACK,
  }
  lv = meta.get("level", 1)
  if lv < 3: cat["bagging"] = []
  if lv < 6: cat["boosting"] = []
  if lv < 9: cat["stacking"] = []
  return cat

def _make_model(name: str, task: str, params: Optional[dict] = None):
  """UI短名/実装名どちらでも受け、task に応じて解決してから作成"""
  p = dict(params or {})
  impl = _resolve_impl(name, task)

  # 回帰
  if task == "regression":
    if impl == "LinearRegression":          return LinearRegression(**{k: v for k, v in p.items() if k in {"fit_intercept", "n_jobs", "positive"}})
    if impl == "DecisionTreeRegressor":     return DecisionTreeRegressor(**p)
    if impl == "KNeighborsRegressor":       return KNeighborsRegressor(**p)
    if impl == "SVR":                       return SVR(**p)
    if impl == "RandomForestRegressor":     return RandomForestRegressor(**p)
    if impl == "GradientBoostingRegressor": return GradientBoostingRegressor(**p)
    if impl == "XGBoostRegressor" and _lib_ok("xgboost"):
      from xgboost import XGBRegressor;   return XGBRegressor(**p)
    if impl == "LightGBMRegressor" and _lib_ok("lightgbm"):
      from lightgbm import LGBMRegressor; return LGBMRegressor(**p)
    if impl == "CatBoostRegressor" and _lib_ok("catboost"):
      from catboost import CatBoostRegressor; return CatBoostRegressor(**p)
    if impl == "StackingRegressor":
      raise HTTPException(400, "StackingRegressor は後続ステップで実装予定")
    return LinearRegression()

  # 分類
  if impl == "LogisticRegression":            return LogisticRegression(max_iter=1000, **p)
  if impl == "DecisionTreeClassifier":        return DecisionTreeClassifier(**p)
  if impl == "KNeighborsClassifier":          return KNeighborsClassifier(**p)
  if impl == "SVC":                           return SVC(**p)
  if impl == "RandomForestClassifier":        return RandomForestClassifier(**p)
  if impl == "GradientBoostingClassifier":    return GradientBoostingClassifier(**p)
  if impl == "XGBoostClassifier" and _lib_ok("xgboost"):
    from xgboost import XGBClassifier;     return XGBClassifier(**p)
  if impl == "LightGBMClassifier" and _lib_ok("lightgbm"):
    from lightgbm import LGBMClassifier;   return LGBMClassifier(**p)
  if impl == "CatBoostClassifier" and _lib_ok("catboost"):
    from catboost import CatBoostClassifier; return CatBoostClassifier(**p)
  if impl == "StackingClassifier":
    raise HTTPException(400, "StackingClassifier は後続ステップで実装予定")

  return LogisticRegression(max_iter=1000)

def _proba_or_score(model, X: pd.DataFrame) -> np.ndarray:
  if hasattr(model, "predict_proba"):
    p = model.predict_proba(X)
    if p.ndim == 2 and p.shape[1] == 2:
      return p[:, 1]
    return p.max(axis=1)
  if hasattr(model, "decision_function"):
    s = model.decision_function(X)
    if np.ndim(s) == 1:
      return 1 / (1 + np.exp(-s))
    smax = (s - s.max(axis=1, keepdims=True))
    return np.exp(smax) / np.exp(smax).sum(axis=1, keepdims=True).max(axis=1)
  pred = model.predict(X)
  return (pred == 1).astype(float) if set(np.unique(pred)).issubset({0, 1}) else np.zeros(len(X))

@router.post("/{region}/train")
async def train(region: str, problem: str, request: Request):
  """
  JSON: {"model":"RandomForest","params":{"n_estimators":300,"max_depth":8}}
  互換: クエリ ?model=... でも可（短名/実装名どちらでも）
  """
  try:
    body = await request.json()
  except Exception:
    body = {}
  model_name = (body.get("model") or request.query_params.get("model") or "LogisticRegression")
  params = body.get("params") or {}

  p = _ensure_processed(problem)
  df = pd.read_csv(p, encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  X, y = _xy(df)

  task, _ = _task_of(problem)
  mdl = _make_model(model_name, task, params)
  mdl.fit(X, y)

  # 保存名はフロントと整合する“受け取った名前”のまま（短名でOK）
  mp = _model_path(problem, model_name)
  mp.parent.mkdir(parents=True, exist_ok=True)
  with open(mp, "wb") as f:
    import pickle
    pickle.dump({"model": mdl, "columns": list(X.columns), "task": task, "params": params}, f)

  return {"ok": True, "model": model_name, "params": params}

@router.get("/{region}/evaluate")
def evaluate(region: str, problem: str, model: str):
  import pickle

  mp = _model_path(problem, model)
  if not mp.exists():
    raise HTTPException(404, "model not trained")
  obj = pickle.loads(mp.read_bytes())
  mdl, cols, task = obj["model"], obj["columns"], obj["task"]

  df = pd.read_csv(_ensure_processed(problem), encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  n = len(df)
  n_pub = max(1, int(n * 0.2))
  public = df.tail(n_pub)
  X_pub, y_pub = _xy(public)

  for c in cols:
    if c not in X_pub.columns:
      X_pub[c] = 0.0
  X_pub = X_pub[cols]

  # 回帰
  if task == "regression":
    pred = mdl.predict(X_pub)
    r2 = float(r2_score(y_pub, pred))
    mae = float(mean_absolute_error(y_pub, pred))
    rmse = float(mean_squared_error(y_pub, pred, squared=False))
    passed = r2 >= 0.5
    try:
      from app.services import model_card as mc
      mc.save_card(
        region, problem, model, task,
        score=r2, metric="r2", passed=passed,
        extra={"mae": mae, "rmse": rmse, "public_rows": int(n_pub)},
        params=getattr(mdl, "get_params", lambda: {})(),
        features=cols, pipeline=_load_pipeline(problem),
      )
    except Exception:
      pass
    return {"score": r2, "metric": "r2", "mae": mae, "rmse": rmse, "passed": passed}

  # 分類
  if hasattr(mdl, "predict_proba"):
    proba = mdl.predict_proba(X_pub)[:, 1]
  elif hasattr(mdl, "decision_function"):
    z = mdl.decision_function(X_pub)
    proba = 1 / (1 + np.exp(-z))
  else:
    proba = (mdl.predict(X_pub) == 1).astype(float)

  fpr, tpr, thresh = roc_curve(y_pub, proba)
  j = tpr - fpr
  j_idx = int(np.argmax(j))
  best_th = float(thresh[j_idx])
  best_tpr, best_fpr = float(tpr[j_idx]), float(fpr[j_idx])
  auc_ = float(roc_auc_score(y_pub, proba))

  y_hat = (proba >= best_th).astype(int)
  acc = float((y_hat == y_pub).mean())
  cm = confusion_matrix(y_pub, y_hat).tolist()
  passed = acc >= 0.7

  try:
    from app.services import model_card as mc
    mc.save_card(
      region, problem, model, task,
      score=acc, metric="accuracy", passed=passed,
      extra={
        "auc": auc_, "best_threshold": best_th,
        "best_tpr": best_tpr, "best_fpr": best_fpr,
        "confusion_matrix": cm, "public_rows": int(n_pub),
      },
      params=getattr(mdl, "get_params", lambda: {})(),
      features=cols, pipeline=_load_pipeline(problem),
    )
  except Exception:
    pass

  return {"score": acc, "metric": "accuracy", "passed": passed,
          "auc": auc_, "best_threshold": best_th,
          "tpr": best_tpr, "fpr": best_fpr, "confusion_matrix": cm}

@router.post("/{region}/submit")
def submit(region: str, problem: str = Query(...), model: str = Query(None)):
  """
  モック提出:
    - model が明示されていればそれを、無ければ直近学習モデル名推定はせず 400 を返す
    - evaluate と同じロジックで即時評価し、既存の passed 基準で合否を決める
  フロントは body 付きでも来るが、ここでは使わない
  """
  import pickle

  if not model:
    raise HTTPException(400, "model is required")

  mp = _model_path(problem, model)
  if not mp.exists():
    raise HTTPException(404, "model not trained")
  obj = pickle.loads(mp.read_bytes())
  mdl, cols, task = obj["model"], obj["columns"], obj["task"]

  df = pd.read_csv(_ensure_processed(problem), encoding="utf-8")
  if "date" in df.columns:
    df = df.drop(columns=["date"])
  n = len(df)
  n_pub = max(1, int(n * 0.2))
  public = df.tail(n_pub)
  X_pub, y_pub = _xy(public)
  for c in cols:
    if c not in X_pub.columns:
      X_pub[c] = 0.0
  X_pub = X_pub[cols]

  if task == "regression":
    pred = mdl.predict(X_pub)
    r2 = float(r2_score(y_pub, pred))
    mae = float(mean_absolute_error(y_pub, pred))
    rmse = float(mean_squared_error(y_pub, pred, squared=False))
    passed = r2 >= 0.5
    return {"submitted": passed, "score": r2, "metric": "r2", "mae": mae, "rmse": rmse, "passed": passed}
  else:
    proba = _proba_or_score(mdl, X_pub)
    fpr, tpr, thresh = roc_curve(y_pub, proba)
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    best_th = float(thresh[j_idx])
    y_hat = (proba >= best_th).astype(int)
    acc = float((y_hat == y_pub).mean())
    passed = acc >= 0.7
    return {"submitted": passed, "score": acc, "metric": "accuracy", "best_threshold": best_th, "passed": passed}

# =========================================================
# 可視化（PNG）
# =========================================================
@router.get("/{region}/eval/roc.png")
def eval_roc(region: str, model: str, problem: str = Query(None), pref: str = Query(None)):
  import pickle
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  mp = _model_path(problem, model)
  if not mp.exists():
    raise HTTPException(404, "model not trained")
  obj = pickle.loads(mp.read_bytes())
  mdl, cols, task = obj["model"], obj["columns"], obj["task"]
  if task != "classification":
    raise HTTPException(400, "only for classification")

  df = _build_dataframe_for_train(problem)
  n = len(df)
  n_pub = max(1, int(n * 0.2))
  public = df.tail(n_pub)
  X_pub, y_pub = _xy(public)
  for c in cols:
    if c not in X_pub.columns:
      X_pub[c] = 0.0
  X_pub = X_pub[cols]
  proba = _proba_or_score(mdl, X_pub)
  fpr, tpr, _ = roc_curve(y_pub, proba)
  auc = roc_auc_score(y_pub, proba)

  fig = plt.figure(figsize=(4, 4))
  ax = fig.add_subplot(111)
  ax.plot(fpr, tpr, label=f"ROC AUC={auc:.3f}")
  ax.plot([0, 1], [0, 1], "--", alpha=0.5)
  ax.set_xlabel("FPR")
  ax.set_ylabel("TPR")
  ax.legend()
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

@router.get("/{region}/eval/cm.png")
def eval_cm(region: str, model: str, problem: str = Query(None), pref: str = Query(None), threshold: float = 0.5):
  import pickle
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  mp = _model_path(problem, model)
  if not mp.exists():
    raise HTTPException(404, "model not trained")
  obj = pickle.loads(mp.read_bytes())
  mdl, cols, task = obj["model"], obj["columns"], obj["task"]
  if task != "classification":
    raise HTTPException(400, "only for classification")

  df = _build_dataframe_for_train(problem)
  n = len(df)
  n_pub = max(1, int(n * 0.2))
  public = df.tail(n_pub)
  X_pub, y_pub = _xy(public)
  for c in cols:
    if c not in X_pub.columns:
      X_pub[c] = 0.0
  X_pub = X_pub[cols]
  proba = _proba_or_score(mdl, X_pub)
  pred = (proba >= float(threshold)).astype(int)
  cm = confusion_matrix(y_pub, pred)

  fig = plt.figure(figsize=(3.6, 3.1))
  ax = fig.add_subplot(111)
  im = ax.imshow(cm, cmap="Blues")
  for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center")
  ax.set_xlabel("Pred")
  ax.set_ylabel("True")
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

@router.get("/{region}/eval/reg_scatter.png")
def eval_reg_scatter(region: str, model: str, problem: str = Query(None), pref: str = Query(None)):
  import pickle
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  mp = _model_path(problem, model)
  if not mp.exists():
    raise HTTPException(404, "model not trained")
  obj = pickle.loads(mp.read_bytes())
  mdl, cols, task = obj["model"], obj["columns"], obj["task"]
  if task != "regression":
    raise HTTPException(400, "only for regression")

  df = _build_dataframe_for_train(problem)
  n = len(df)
  n_pub = max(1, int(n * 0.2))
  public = df.tail(n_pub)
  X_pub, y_pub = _xy(public)
  for c in cols:
    if c not in X_pub.columns:
      X_pub[c] = 0.0
  X_pub = X_pub[cols]
  pred = mdl.predict(X_pub)

  fig = plt.figure(figsize=(4, 4))
  ax = fig.add_subplot(111)
  ax.scatter(y_pub, pred, s=10, alpha=0.6)
  lo = float(min(y_pub.min(), pred.min()))
  hi = float(max(y_pub.max(), pred.max()))
  ax.plot([lo, hi], [lo, hi], "--", alpha=0.5)
  ax.set_xlabel("y_true")
  ax.set_ylabel("y_pred")
  fig.tight_layout()
  buf = io.BytesIO()
  fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
  plt.close(fig)
  buf.seek(0)
  return Response(content=buf.read(), media_type="image/png")

# =========================================================
# モデルカード / 再学習
# =========================================================
def _append_card(problem: str, card: dict):
  cj = _cards_json(problem)
  cj.parent.mkdir(parents=True, exist_ok=True)
  arr = []
  if cj.exists():
    try:
      arr = json.loads(cj.read_text(encoding="utf-8"))
    except Exception:
      arr = []
  arr.append(card)
  cj.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

@router.get("/{region}/cards/list")
def cards_list(region: str, problem: str = Query(None), pref: str = Query(None)):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  cj = _cards_json(problem)
  if not cj.exists():
    return {"cards": []}
  return {"cards": json.loads(cj.read_text(encoding="utf-8"))}

# 互換: /{region}/cards → /cards/list のエイリアス
@router.get("/{region}/cards")
def cards_list_alias(region: str, problem: str = Query(None), pref: str = Query(None)):
  return cards_list(region, problem, pref)

# CSV エクスポート: /{region}/cards.csv
@router.get("/{region}/cards.csv")
def cards_csv(region: str, problem: str = Query(None), pref: str = Query(None)):
  import csv
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  arr = cards_list(region, problem)["cards"]
  buf = io.StringIO()
  if not arr:
    buf.write("problem,model,task,metric,score,passed\n")
  else:
    rows = []
    for c in arr:
      rows.append({
        "problem": problem,
        "model": c.get("model"),
        "task": c.get("task"),
        "metric": c.get("metric"),
        "score": c.get("score"),
        "passed": c.get("passed"),
        "extra": json.dumps(c.get("extra") or {}, ensure_ascii=False),
        "params": json.dumps(c.get("params") or {}, ensure_ascii=False),
        "features": json.dumps(c.get("features") or [], ensure_ascii=False),
      })
    fieldnames = list(rows[0].keys())
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow(r)
  return Response(
    buf.getvalue(),
    media_type="text/csv",
    headers={
      "Content-Disposition": f'attachment; filename="model_cards_{problem}.csv"',
      "Cache-Control": "no-store",
    },
  )

@router.post("/{region}/retrain_from_card")
def retrain_from_card(region: str, problem: str = Query(None), pref: str = Query(None), card_index: int = Query(..., ge=0)):
  import pickle
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  cards = cards_list(region, problem)["cards"]
  if not (0 <= card_index < len(cards)):
    raise HTTPException(404, "card not found")
  card = cards[card_index]
  model = card["model"]
  task = card.get("task", "classification")

  # 1) 前処理を復元
  steps = card.get("pipeline") or []
  _save_pipeline(problem, steps)
  _rebuild_processed(problem)

  # 2) 学習
  df = _build_dataframe_for_train(problem)
  X, y = _xy(df)
  mdl = _make_model(model, task)
  mdl.fit(X, y)
  mp = _model_path(problem, model)
  mp.parent.mkdir(parents=True, exist_ok=True)
  with open(mp, "wb") as f:
    pickle.dump({"model": mdl, "columns": list(X.columns), "task": task}, f)
  return {"ok": True, "model": model}

# =========================================================
# かんたんアンサンブル
# =========================================================
@router.post("/{region}/ensemble/train")
def ensemble_train(region: str, problem: str = Query(None), pref: str = Query(None), kind: str = Query(..., regex="^(voting|stacking)$"), base_models: str = Query(..., description="カンマ区切り（例: LogisticRegression,RandomForestClassifier）"), task_force: Optional[str] = Query(None)):
  problem = problem or pref
  if not problem:
    raise HTTPException(400, "problem is required")
  task, _ = _task_of(problem)
  if task_force in ("classification", "regression"):
    task = task_force

  df = _build_dataframe_for_train(problem)
  X, y = _xy(df)

  names = [s.strip() for s in base_models.split(",") if s.strip()]
  if not names:
    raise HTTPException(400, "base_models is empty")
  ests = [(n, _make_model(n, task)) for n in names]

  if kind == "voting":
    mdl = VotingClassifier(estimators=ests, voting="soft") if task == "classification" else VotingRegressor(estimators=ests)
  else:
    if task == "classification":
      mdl = StackingClassifier(estimators=ests, final_estimator=LogisticRegression(max_iter=1000), passthrough=False)
    else:
      mdl = StackingRegressor(estimators=ests, final_estimator=LinearRegression(), passthrough=False)

  mdl.fit(X, y)
  name = f"{'Voting' if kind=='voting' else 'Stacking'}({'-'.join(names)})"
  mp = _model_path(problem, name)
  mp.parent.mkdir(parents=True, exist_ok=True)
  import pickle
  with open(mp, "wb") as f:
    pickle.dump({"model": mdl, "columns": list(X.columns), "task": task}, f)
  return {"ok": True, "model": name}

# =========================================================
# データセットの簡易メタ
# =========================================================
URBAN_REG = {"tokyo", "kanagawa", "osaka", "aichi", "saitama", "fukuoka", "miyagi", "kyoto", "nara"}
SNOW_CLS = {"hokkaido", "niigata", "toyama", "nagano", "iwate"}
BEACH_CLS = {"okinawa"}
BEACH_REG = {"kanagawa", "miyazaki", "wakayama", "chiba", "shizuoka"}
TOUR_REG = {"ishikawa", "ehime", "oita", "yamanashi", "gifu", "tochigi", "mie", "shimane", "tottori", "kagawa"}

def _target_desc(pref: str) -> dict:
  if pref in URBAN_REG:
    return {"task": "regression", "label": "都市の人流・輸送需要を予測（0〜1の需要指数）"}
  if pref in SNOW_CLS:
    return {"task": "classification", "label": "冬期の雪関連リスクを判定（0/1）"}
  if pref in BEACH_CLS:
    return {"task": "classification", "label": "ビーチ・海況のリスクを判定（0/1）"}
  if pref in BEACH_REG:
    return {"task": "regression", "label": "浜の賑わい・魅力度を予測（0〜1）"}
  if pref in TOUR_REG:
    return {"task": "regression", "label": "観光活況度を予測（0〜1）"}
  return {"task": "classification", "label": "交通の乱れリスクを判定（0/1）"}

@router.get("/{region}/meta")
def dataset_meta(region: str, problem: str):
  meta_path = _meta_path()
  meta = json.loads(meta_path.read_text(encoding="utf-8")).get(problem, {}) if meta_path.exists() else {}
  td = _target_desc(problem)
  return {"level": meta.get("level"), "metric": meta.get("metric"), "task": td["task"], "target_desc": td["label"], "problem": problem, "region": meta.get("region")}
