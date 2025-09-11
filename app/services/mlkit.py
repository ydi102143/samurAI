# app/services/mlkit.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from pathlib import Path
import os, json, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

# =========================================================
# パス関連（datasets 配下のどの region にあるかを自動検出）
# =========================================================
DATA_ROOT = Path("storage/datasets")
MODEL_ROOT = Path("storage/models")
META_JSON = DATA_ROOT / "_meta" / "problem_meta.json"

def _problem_region(problem: str) -> str:
    # 明示メタがあればそこから
    try:
        if META_JSON.exists():
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            reg = (meta.get(problem) or {}).get("region")
            if reg:
                return reg
    except Exception:
        pass
    # スキャンして推定
    for d in DATA_ROOT.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / f"{problem}.csv").exists():
            return d.name
    raise FileNotFoundError(f"region not found for problem='{problem}' under {DATA_ROOT}")

def _raw_csv(problem: str) -> Path:
    region = _problem_region(problem)
    p = DATA_ROOT / region / f"{problem}.csv"
    if not p.exists():
        raise FileNotFoundError(f"raw csv not found: {p}")
    return p

def _proc_dir(problem: str) -> Path:
    region = _problem_region(problem)
    return DATA_ROOT / region / "_processed" / problem

def _proc_csv(problem: str) -> Path:
    return _proc_dir(problem) / "latest.csv"

def _pipe_json(problem: str) -> Path:
    return _proc_dir(problem) / "pipeline.json"

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# データ読み込みと前処理済データの選択
#   - 既存コードと整合： processed があれば優先、無ければ raw
# =========================================================
def _choose_csv(problem: str) -> Path:
    pcsv = _proc_csv(problem)
    return pcsv if pcsv.exists() else _raw_csv(problem)

def load_dataframe(problem: str) -> pd.DataFrame:
    p = _choose_csv(problem)
    return pd.read_csv(p, encoding="utf-8")

def xy_from_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "y" not in df.columns:
        raise ValueError("target 'y' not found")
    X = df.drop(columns=[c for c in ["y", "date"] if c in df.columns])
    y = pd.to_numeric(df["y"], errors="coerce")
    return X, y

# =========================================================
# タスクとメトリクス
# =========================================================
def task_of(problem: str) -> str:
    """'classification' or 'regression'（メタ無ければ classification デフォルト）"""
    try:
        if META_JSON.exists():
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            t = (meta.get(problem) or {}).get("target", {}).get("type")
            if t in ("classification", "regression"):
                return t
    except Exception:
        pass
    return "classification"

def default_metric(problem: str) -> str:
    """推奨メトリクス（メタ無ければ accuracy / r2）"""
    try:
        if META_JSON.exists():
            meta = json.loads(META_JSON.read_text(encoding="utf-8"))
            m = (meta.get(problem) or {}).get("metric")
            if m:
                return m
    except Exception:
        pass
    return "r2" if task_of(problem) == "regression" else "accuracy"

# =========================================================
# モデル辞書
# =========================================================
CLASSIFIERS: Dict[str, BaseEstimator] = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

REGRESSORS: Dict[str, BaseEstimator] = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=10000),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=6, random_state=42),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "KNNRegressor": KNeighborsRegressor(n_neighbors=7),
    "SVR": SVR(),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
}

def get_model(name: str, task: str) -> BaseEstimator:
    if task == "regression":
        return clone(REGRESSORS.get(name, LinearRegression()))
    return clone(CLASSIFIERS.get(name, LogisticRegression(max_iter=1000)))

# =========================================================
# モデル保存/読み込み（storage/models/<region>/<problem>/<model>）
# =========================================================
def model_dir(problem: str, model_name: str) -> Path:
    region = _problem_region(problem)
    return MODEL_ROOT / region / problem / model_name

def save_model(problem: str, model_name: str, estimator: BaseEstimator, columns: List[str], task: str):
    d = model_dir(problem, model_name)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as f:
        pickle.dump({"model": estimator, "columns": list(columns), "task": task}, f)
    (d / "columns.json").write_text(json.dumps({"columns": list(columns)}, ensure_ascii=False, indent=2), encoding="utf-8")

def load_model(problem: str, model_name: str) -> Dict[str, Any]:
    d = model_dir(problem, model_name)
    obj = pickle.loads((d / "model.pkl").read_bytes())
    return obj

# =========================================================
# 学習・評価（単体）
# =========================================================
def train_single(problem: str, model_name: str) -> Dict[str, Any]:
    df = load_dataframe(problem)
    X, y = xy_from_df(df)
    task = task_of(problem)
    model = get_model(model_name, task)
    model.fit(X, y)
    save_model(problem, model_name, model, list(X.columns), task)
    return {"ok": True, "model": model_name, "task": task, "n_train": len(X)}

def _public_split(df: pd.DataFrame, ratio_tail: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_pub = max(1, int(n * ratio_tail))
    return df.iloc[:-n_pub, :], df.iloc[-n_pub:, :]

def evaluate_single(problem: str, model_name: str, metric: Optional[str] = None) -> Dict[str, Any]:
    obj = load_model(problem, model_name)
    model: BaseEstimator = obj["model"]
    cols: List[str] = obj.get("columns", [])
    task: str = obj.get("task", task_of(problem))

    df = load_dataframe(problem)
    _, public = _public_split(df)
    X_pub, y_pub = xy_from_df(public)

    # 列合わせ（不足は0埋め）
    for c in cols:
        if c not in X_pub.columns:
            X_pub[c] = 0.0
    X_pub = X_pub[cols]

    metric = metric or default_metric(problem)

    if task == "regression":
        pred = model.predict(X_pub)
        score = float(r2_score(y_pub, pred) if metric == "r2" else r2_score(y_pub, pred))
        passed = score >= 0.5
        return {"task": task, "metric": metric, "score": score, "passed": passed, "n_public": len(X_pub)}

    # classification
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_pub)
        p1 = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    else:
        # decision functionのときはsigmoid近似
        if hasattr(model, "decision_function"):
            z = model.decision_function(X_pub)
            p1 = 1 / (1 + np.exp(-z))
        else:
            # fallback: 予測を{0,1}で返し、そのまま確率扱い
            p1 = model.predict(X_pub)
    pred = (p1 >= 0.5).astype(int)

    if metric.lower() in ("accuracy", "acc"):
        score = float(accuracy_score(y_pub, pred))
    elif metric.lower() in ("f1", "f1_score"):
        score = float(f1_score(y_pub, pred))
    elif metric.lower() in ("roc_auc", "auc"):
        try:
            score = float(roc_auc_score(y_pub, p1))
        except Exception:
            score = float(accuracy_score(y_pub, pred))
    else:
        score = float(accuracy_score(y_pub, pred))

    passed = score >= 0.7 if metric.lower() != "roc_auc" else score >= 0.75
    return {"task": task, "metric": metric, "score": score, "passed": passed, "n_public": len(X_pub)}

# =========================================================
# しきい値探索（二値分類）
# =========================================================
def threshold_search(y_true: np.ndarray, p1: np.ndarray, metric: str = "f1", beta: float = 1.0, steps: int = 101) -> Dict[str, Any]:
    ths = np.linspace(0.0, 1.0, steps)
    best_th, best_score = 0.5, -1.0
    for th in ths:
        pred = (p1 >= th).astype(int)
        if metric == "accuracy":
            s = accuracy_score(y_true, pred)
        elif metric == "f1":
            s = f1_score(y_true, pred)
        elif metric.startswith("f") and metric[1:].replace(".","",1).isdigit():
            # Fβ (e.g., f0.5, f2)
            b = float(metric[1:])
            p, r, f, _ = precision_recall_fscore_support(y_true, pred, beta=b, average="binary", zero_division=0)
            s = f
        else:
            s = f1_score(y_true, pred)
        if s > best_score:
            best_score, best_th = s, th
    return {"best_threshold": float(best_th), "best_score": float(best_score), "metric": metric}

# =========================================================
# アンサンブル（Lv7）: Classifier = soft/hard vote, Regressor = 平均
# =========================================================
@dataclass
class EnsembleSpec:
    base_models: List[str]             # ["LogisticRegression","RandomForest",...]
    method: str = "soft"               # "soft" | "hard" | "avg"（regressionは常に "avg"）
    name: str = "Ensemble"

def train_ensemble(problem: str, spec: EnsembleSpec) -> Dict[str, Any]:
    task = task_of(problem)
    df = load_dataframe(problem)
    X, y = xy_from_df(df)
    models = []
    for m in spec.base_models:
        models.append((m, get_model(m, task)))

    # 学習
    for _, est in models:
        est.fit(X, y)

    # 推論器をラップして保存（シンプル実装）
    ens = {"task": task, "models": [(n, est) for n, est in models], "method": spec.method}

    # 予測関数
    def _predict(Xt: pd.DataFrame):
        if task == "regression":
            preds = np.column_stack([est.predict(Xt) for _, est in ens["models"]])
            return preds.mean(axis=1)
        # classification
        if spec.method == "hard":
            preds = np.column_stack([est.predict(Xt) for _, est in ens["models"]])
            # 多数決
            return np.round(np.mean(preds, axis=1)).astype(int)
        else:
            # soft-vote: proba 平均
            probas = []
            for _, est in ens["models"]:
                if hasattr(est, "predict_proba"):
                    p = est.predict_proba(Xt)[:, 1]
                else:
                    if hasattr(est, "decision_function"):
                        z = est.decision_function(Xt); p = 1/(1+np.exp(-z))
                    else:
                        p = est.predict(Xt)
                probas.append(p)
            return np.mean(np.column_stack(probas), axis=1)

    ens_dir = model_dir(problem, spec.name)
    ens_dir.mkdir(parents=True, exist_ok=True)
    with open(ens_dir / "model.pkl", "wb") as f:
        pickle.dump({"ensemble": ens, "columns": list(X.columns), "task": task}, f)
    (ens_dir / "columns.json").write_text(json.dumps({"columns": list(X.columns)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "model": spec.name, "task": task}

def evaluate_ensemble(problem: str, name: str = "Ensemble", metric: Optional[str] = None) -> Dict[str, Any]:
    d = model_dir(problem, name)
    obj = pickle.loads((d / "model.pkl").read_bytes())
    ens = obj["ensemble"]; cols = obj["columns"]; task = obj["task"]
    df = load_dataframe(problem); _, public = _public_split(df)
    X_pub, y_pub = xy_from_df(public)
    for c in cols:
        if c not in X_pub.columns: X_pub[c] = 0.0
    X_pub = X_pub[cols]
    metric = metric or default_metric(problem)

    # 推論
    if task == "regression":
        preds = []
        for _, est in ens["models"]:
            preds.append(est.predict(X_pub))
        pred = np.mean(np.column_stack(preds), axis=1)
        score = float(r2_score(y_pub, pred))
        return {"task": task, "metric": "r2", "score": score, "passed": score >= 0.5, "n_public": len(X_pub)}

    # classification
    method = ens.get("method", "soft")
    if method == "hard":
        preds = np.column_stack([est.predict(X_pub) for _, est in ens["models"]])
        pred = np.round(np.mean(preds, axis=1)).astype(int)
        score = float(accuracy_score(y_pub, pred)) if metric == "accuracy" else float(f1_score(y_pub, pred))
        return {"task": task, "metric": metric, "score": score, "passed": score >= 0.7, "n_public": len(X_pub)}
    else:
        probas = []
        for _, est in ens["models"]:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_pub)[:, 1]
            elif hasattr(est, "decision_function"):
                z = est.decision_function(X_pub); p = 1/(1+np.exp(-z))
            else:
                p = est.predict(X_pub)
            probas.append(p)
        p1 = np.mean(np.column_stack(probas), axis=1)
        pred = (p1 >= 0.5).astype(int)
        if metric.lower() == "roc_auc":
            try:
                score = float(roc_auc_score(y_pub, p1))
            except Exception:
                score = float(accuracy_score(y_pub, pred))
        elif metric.lower() == "f1":
            score = float(f1_score(y_pub, pred))
        else:
            score = float(accuracy_score(y_pub, pred))
        return {"task": task, "metric": metric, "score": score, "passed": (score >= 0.7 if metric!='roc_auc' else score>=0.75), "n_public": len(X_pub)}

# =========================================================
# スタッキング（Lv8）
# =========================================================
@dataclass
class StackingSpec:
    base_models: List[str]
    final_model: str                  # 例: "LogisticRegression" or "LinearRegression"
    name: str = "Stacking"

def train_stacking(problem: str, spec: StackingSpec) -> Dict[str, Any]:
    task = task_of(problem)
    df = load_dataframe(problem); X, y = xy_from_df(df)
    estimators = [(m, get_model(m, task)) for m in spec.base_models]

    if task == "regression":
        final_est = get_model(spec.final_model, "regression")
        stk = StackingRegressor(estimators=estimators, final_estimator=final_est, passthrough=False, n_jobs=None)
    else:
        final_est = get_model(spec.final_model, "classification")
        stk = StackingClassifier(estimators=estimators, final_estimator=final_est, passthrough=False)

    stk.fit(X, y)
    d = model_dir(problem, spec.name); d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as f:
        pickle.dump({"model": stk, "columns": list(X.columns), "task": task}, f)
    (d / "columns.json").write_text(json.dumps({"columns": list(X.columns)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "model": spec.name, "task": task}

def evaluate_stacking(problem: str, name: str = "Stacking", metric: Optional[str] = None) -> Dict[str, Any]:
    d = model_dir(problem, name)
    obj = pickle.loads((d / "model.pkl").read_bytes())
    mdl: BaseEstimator = obj["model"]; cols = obj["columns"]; task = obj["task"]
    df = load_dataframe(problem); _, public = _public_split(df)
    X_pub, y_pub = xy_from_df(public)
    for c in cols:
        if c not in X_pub.columns: X_pub[c] = 0.0
    X_pub = X_pub[cols]
    metric = metric or default_metric(problem)
    if task == "regression":
        pred = mdl.predict(X_pub)
        score = float(r2_score(y_pub, pred))
        return {"task": task, "metric": "r2", "score": score, "passed": score >= 0.5, "n_public": len(X_pub)}
    else:
        if hasattr(mdl, "predict_proba"):
            p1 = mdl.predict_proba(X_pub)[:, 1]
            pred = (p1 >= 0.5).astype(int)
        else:
            pred = mdl.predict(X_pub)
            p1 = pred
        if metric.lower() == "roc_auc":
            try:
                score = float(roc_auc_score(y_pub, p1))
            except Exception:
                score = float(accuracy_score(y_pub, pred))
        elif metric.lower() == "f1":
            score = float(f1_score(y_pub, pred))
        else:
            score = float(accuracy_score(y_pub, pred))
        return {"task": task, "metric": metric, "score": score, "passed": (score >= 0.7 if metric!='roc_auc' else score>=0.75), "n_public": len(X_pub)}

# =========================================================
# Cross Validation（簡易）
# =========================================================
def cross_validate(problem: str, model_name: str, n_splits: int = 5, shuffle: bool = True, random_state: int = 42, metric: Optional[str] = None) -> Dict[str, Any]:
    task = task_of(problem); metric = metric or default_metric(problem)
    df = load_dataframe(problem); X, y = xy_from_df(df)
    scores: List[float] = []

    if task == "regression":
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for tr, va in kf.split(X):
            mdl = get_model(model_name, task)
            mdl.fit(X.iloc[tr], y.iloc[tr])
            pred = mdl.predict(X.iloc[va])
            s = r2_score(y.iloc[va], pred)
            scores.append(float(s))
        return {"task": task, "metric": "r2", "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}

    # classification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for tr, va in skf.split(X, y):
        mdl = get_model(model_name, task)
        mdl.fit(X.iloc[tr], y.iloc[tr])
        if metric.lower() == "roc_auc":
            if hasattr(mdl, "predict_proba"):
                p1 = mdl.predict_proba(X.iloc[va])[:, 1]
            elif hasattr(mdl, "decision_function"):
                z = mdl.decision_function(X.iloc[va]); p1 = 1/(1+np.exp(-z))
            else:
                p1 = mdl.predict(X.iloc[va])
            try: s = roc_auc_score(y.iloc[va], p1)
            except Exception: s = accuracy_score(y.iloc[va], (p1>=0.5).astype(int))
        elif metric.lower() == "f1":
            pred = mdl.predict(X.iloc[va]); s = f1_score(y.iloc[va], pred)
        else:
            pred = mdl.predict(X.iloc[va]); s = accuracy_score(y.iloc[va], pred)
        scores.append(float(s))
    return {"task": task, "metric": metric, "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
