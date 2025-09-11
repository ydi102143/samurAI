from __future__ import annotations
"""
services/tutorial.py — samurAI チュートリアルエンジン（API付き）

目的:
- 県ごとの level / task / metric に応じて、段階的なチュートリアル手順を提供
- UI からのイベント(例: open_tab:preprocess / click:train / impute_done)に応じて進捗を更新
- 進捗は storage/state/tutorial/{uid}/{pref}.json に保存（匿名=anonymous）

使い方:
- main.py などで:  `from services.tutorial import tutorial_router` / `app.include_router(tutorial_router, prefix="/v1")`
- UI 側:
  - GET  /v1/tutorial?problem=pref&uid=UID → 現在のフローとステップを取得
  - POST /v1/tutorial/progress {problem, uid, event} → イベント送信で該当ステップが完了扱い
  - POST /v1/tutorial/reset    {problem, uid}       → 進捗リセット
  - POST /v1/tutorial/skip     {problem, uid}       → 現在ステップをスキップ

設計メモ:
- pref_meta.json から level と target.type を取得してフローを動的生成
- アクションはUIが発火しやすいよう文字列(enum風)で定義
- 条件評価は最小限（アクション一致で完了）。厳密な条件は将来 Version 2 で導入可
"""

from pathlib import Path
from typing import List, Dict, Optional
import json

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

# =====================================================
# 設定
# =====================================================
PREF_META_PATH = Path("storage/meta/pref_meta.json")
STATE_ROOT = Path("storage/state/tutorial")
STATE_ROOT.mkdir(parents=True, exist_ok=True)

# UI 側で使う アクションの定数（例）
# - タブ操作:   open_tab:overview / open_tab:preprocess / open_tab:eda / open_tab:model / open_tab:evaluate
# - クリック系: click:eda_preview / click:eda_schema / click:impute_apply / click:train / click:evaluate / click:submit
# - 前処理系:   impute_done / onehot_done / scale_done
# - 可視化系:   viz_corr_done / viz_hist_done / viz_scatter_done
# - 学習系:     select_model:<ModelName> / cv_set / tune_random_run / tune_bayes_run
# - 分類Lv8+:   threshold_sweep_youden_done
# - Lv9:        cards_export_done / stacking_fit_done

# =====================================================
# モデル
# =====================================================
class TutorialStep(BaseModel):
    id: str
    title: str
    body: str
    action: str = Field(
        ..., description="UI が発火するイベント名。例: 'click:train', 'open_tab:preprocess'"
    )
    hint: Optional[str] = None
    skippable: bool = True


class TutorialFlow(BaseModel):
    flow_id: str
    level: int
    pref: str
    target_type: str  # "classification" or "regression"
    metric: str
    steps: List[TutorialStep]
    current_index: int = 0

    @property
    def completed(self) -> bool:
        return self.current_index >= len(self.steps)

    def current_step(self) -> Optional[TutorialStep]:
        if self.completed:
            return None
        return self.steps[self.current_index]

    def to_public_dict(self, done_ids: Optional[List[str]] = None) -> Dict:
        done_ids = done_ids or []
        return {
            "flow_id": self.flow_id,
            "level": self.level,
            "pref": self.pref,
            "target_type": self.target_type,
            "metric": self.metric,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "body": s.body,
                    "action": s.action,
                    "hint": s.hint,
                    "skippable": s.skippable,
                    "status": "done" if s.id in done_ids else (
                        "current" if i == self.current_index else "pending"
                    ),
                }
                for i, s in enumerate(self.steps)
            ],
            "current_index": self.current_index,
            "completed": self.completed,
        }


# =====================================================
# 進捗保存
# =====================================================

def _state_path(uid: str, pref: str) -> Path:
    safe_uid = uid or "anonymous"
    d = STATE_ROOT / safe_uid
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{pref}.json"


def load_state(uid: str, pref: str) -> Dict:
    p = _state_path(uid, pref)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"current_index": 0, "done_ids": []}


def save_state(uid: str, pref: str, state: Dict) -> None:
    p = _state_path(uid, pref)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# =====================================================
# メタ読み込み
# =====================================================

def load_pref_meta() -> Dict:
    if not PREF_META_PATH.exists():
        raise FileNotFoundError(f"pref_meta.json not found: {PREF_META_PATH}")
    return json.loads(PREF_META_PATH.read_text(encoding="utf-8"))


def get_pref_record(pref: str) -> Dict:
    meta = load_pref_meta()
    if pref not in meta:
        raise KeyError(f"Unknown pref: {pref}")
    return meta[pref]


# =====================================================
# フロー定義（レベル別ビルダー）
# =====================================================

def build_flow(pref: str) -> TutorialFlow:
    r = get_pref_record(pref)
    level = int(r.get("level", 1))
    target_type = r.get("target", {}).get("type", "classification")
    metric = r.get("metric", "accuracy")

    # レベル×タスクでフローを出し分け
    if level == 1:
        if target_type == "classification":
            steps = _steps_lv1_classification(metric)
            flow_id = "lv1_cls"
        else:
            steps = _steps_lv1_regression(metric)
            flow_id = "lv1_reg"
    elif level == 2:
        steps = _steps_lv2_common(metric)
        flow_id = "lv2_common"
    elif level == 3:
        steps = _steps_lv3_common(metric)
        flow_id = "lv3_common"
    elif level == 4:
        steps = _steps_lv4_cv_scaling(metric)
        flow_id = "lv4_cv_scaling"
    elif level == 5:
        steps = _steps_lv5_tuning_intro(metric)
        flow_id = "lv5_tune_intro"
    elif level == 6:
        steps = _steps_lv6_gbm_intro(metric)
        flow_id = "lv6_gbm_intro"
    elif level == 7:
        steps = _steps_lv7_xgb(metric)
        flow_id = "lv7_xgb"
    elif level == 8:
        steps = _steps_lv8_threshold(metric, target_type)
        flow_id = "lv8_threshold"
    else:  # 9+
        steps = _steps_lv9_stacking(metric)
        flow_id = "lv9_stacking"

    return TutorialFlow(
        flow_id=flow_id,
        level=level,
        pref=pref,
        target_type=target_type,
        metric=metric,
        steps=steps,
    )


# --------------------------
# Lv1 基礎（学習→評価 最短動線）
# --------------------------

def _steps_lv1_classification(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="read_task",
            title="課題を確認しよう",
            body="課題説明タブで、目的変数と評価指標(accuracy)を確認します。",
            action="open_tab:overview",
            hint="まずは課題のゴールを把握。",
        ),
        TutorialStep(
            id="open_preprocess",
            title="前処理タブを開こう",
            body="欠損率グラフを見て、必要なら欠損補完を実行します。",
            action="open_tab:preprocess",
        ),
        TutorialStep(
            id="impute",
            title="欠損補完を実行",
            body="列指定または一括で mean/median/mode/constant を選び、適用をクリック。",
            action="impute_done",
            hint="impute ボタン押下時に event を送ってください。",
        ),
        TutorialStep(
            id="select_model",
            title="モデルを選ぼう（ロジスティック回帰）",
            body="Model タブで LogisticRegression を選択します。",
            action="select_model:LogisticRegression",
        ),
        TutorialStep(
            id="train",
            title="学習を実行",
            body="Train をクリックして学習。",
            action="click:train",
        ),
        TutorialStep(
            id="evaluate",
            title="評価を確認",
            body=f"Evaluate を押して {metric.upper()} を表示。",
            action="click:evaluate",
        ),
        TutorialStep(
            id="submit",
            title="提出してみよう",
            body="Submit を押してスコアを記録（任意）。",
            action="click:submit",
            skippable=True,
        ),
    ]


def _steps_lv1_regression(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="read_task",
            title="課題を確認しよう",
            body="課題説明タブで、目的変数と評価指標(R²)を確認します。",
            action="open_tab:overview",
        ),
        TutorialStep(
            id="open_preprocess",
            title="前処理タブを開こう",
            body="欠損率グラフを見て、必要なら欠損補完を実行します。",
            action="open_tab:preprocess",
        ),
        TutorialStep(
            id="impute",
            title="欠損補完を実行",
            body="列指定または一括で mean/median/mode/constant を選び、適用をクリック。",
            action="impute_done",
        ),
        TutorialStep(
            id="select_model",
            title="モデルを選ぼう（線形回帰）",
            body="Model タブで LinearRegression を選択します。",
            action="select_model:LinearRegression",
        ),
        TutorialStep(
            id="train",
            title="学習を実行",
            body="Train をクリックして学習。",
            action="click:train",
        ),
        TutorialStep(
            id="evaluate",
            title="評価を確認",
            body=f"Evaluate を押して {metric.upper()} を表示。",
            action="click:evaluate",
        ),
        TutorialStep(
            id="submit",
            title="提出してみよう",
            body="Submit を押してスコアを記録（任意）。",
            action="click:submit",
            skippable=True,
        ),
    ]


# --------------------------
# Lv2: EDA 20行プレビュー / schema
# --------------------------

def _steps_lv2_common(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="eda_preview",
            title="データを覗いてみよう",
            body="EDA の 20 行プレビューを表示。列名・値の雰囲気を掴もう。",
            action="click:eda_preview",
        ),
        TutorialStep(
            id="eda_schema",
            title="列の型ヒントを確認",
            body="schema を表示して、数値/カテゴリ、欠損率を把握。",
            action="click:eda_schema",
        ),
        TutorialStep(
            id="train",
            title="学習して評価しよう",
            body=f"モデル選択→学習→評価({metric.upper()}) まで実行。",
            action="click:evaluate",
        ),
    ]


# --------------------------
# Lv3: 可視化 / One-Hot
# --------------------------

def _steps_lv3_common(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="viz_corr",
            title="相関ヒートマップを確認",
            body="数値列の関連性を俯瞰する。高相関の列には要注意。",
            action="viz_corr_done",
        ),
        TutorialStep(
            id="viz_hist",
            title="単変量ヒストグラム",
            body="選択列の分布を確認。歪度や外れ値の気配を掴む。",
            action="viz_hist_done",
        ),
        TutorialStep(
            id="viz_scatter",
            title="散布図で関係をみる",
            body="X-Y の関係から単純な分離や線形性をチェック。",
            action="viz_scatter_done",
        ),
        TutorialStep(
            id="onehot",
            title="カテゴリ列を One-Hot",
            body="カテゴリ列をワンクリックでダミー化。",
            action="onehot_done",
        ),
        TutorialStep(
            id="train_eval",
            title="学習→評価",
            body=f"モデルを選び、学習→評価({metric.upper()}) を実行。",
            action="click:evaluate",
        ),
    ]


# --------------------------
# Lv4: スケーリング / CV
# --------------------------

def _steps_lv4_cv_scaling(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="scale",
            title="スケーリングを適用",
            body="Standard または MinMax スケーリングを適用。",
            action="scale_done",
        ),
        TutorialStep(
            id="cv",
            title="交差検証を設定",
            body="cv=K を設定（未指定なら通常学習）。",
            action="cv_set",
        ),
        TutorialStep(
            id="train_eval",
            title="CV で評価",
            body=f"学習→評価({metric.upper()})。CV により汎化性能を確認。",
            action="click:evaluate",
        ),
    ]


# --------------------------
# Lv5: 重要度/ランダムサーチ入門
# --------------------------

def _steps_lv5_tuning_intro(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="feature_importance",
            title="特徴量重要度を見よう",
            body="木系モデルで重要度を可視化。",
            action="feature_importance_done",
            skippable=True,
        ),
        TutorialStep(
            id="tune_random",
            title="ランダムサーチを走らせる",
            body="試行回数 N を指定して探索。",
            action="tune_random_run",
        ),
        TutorialStep(
            id="best_save",
            title="ベストモデル保存",
            body="ベストモデルを保存し、再利用できるように。",
            action="best_save_done",
        ),
    ]


# --------------------------
# Lv6: GBM 入門（GB/Adaboost、重要度・ログ）
# --------------------------

def _steps_lv6_gbm_intro(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="gbm_select",
            title="ブースティング系を選ぶ",
            body="GradientBoosting または AdaBoost を選択。",
            action="select_model:GradientBoosting|AdaBoost",
        ),
        TutorialStep(
            id="train",
            title="学習を実行",
            body="Train をクリックして学習。",
            action="click:train",
        ),
        TutorialStep(
            id="evaluate",
            title="評価を確認",
            body=f"Evaluate を押して {metric.upper()} を確認。",
            action="click:evaluate",
        ),
        TutorialStep(
            id="best_load",
            title="ベストモデルを復元",
            body="保存済みのベストモデルを読み込んで比較。",
            action="best_load_done",
            skippable=True,
        ),
    ]


# --------------------------
# Lv7: XGBoost
# --------------------------

def _steps_lv7_xgb(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="xgb_select",
            title="XGBoost を選ぶ",
            body="XGBoost をモデルとして選択。",
            action="select_model:XGBoost",
        ),
        TutorialStep(
            id="tune_random",
            title="ハイパーパラメータ探索",
            body="ランダムまたは少数の手動チューニングで効果を体験。",
            action="tune_random_run",
            skippable=True,
        ),
        TutorialStep(
            id="evaluate",
            title="評価を確認",
            body=f"Evaluate → {metric.upper()} を比較。",
            action="click:evaluate",
        ),
    ]


# --------------------------
# Lv8: しきい値スイープ（分類） / PR, ROC
# --------------------------

def _steps_lv8_threshold(metric: str, target_type: str) -> List[TutorialStep]:
    steps = [
        TutorialStep(
            id="roc_pr",
            title="ROC/PR 曲線を描こう",
            body="モデルの確信度を理解するために ROC/PR を表示。",
            action="roc_pr_done",
        )
    ]
    if target_type == "classification":
        steps.append(
            TutorialStep(
                id="threshold_youden",
                title="Youden's J で最適しきい値",
                body="ROC から Youden's J を用いて最適なしきい値を提案・適用。",
                action="threshold_sweep_youden_done",
            )
        )
    steps.append(
        TutorialStep(
            id="evaluate",
            title="評価を確認",
            body=f"しきい値反映後に Evaluate ({metric.upper()}) を確認。",
            action="click:evaluate",
        )
    )
    return steps


# --------------------------
# Lv9: モデルカード比較 / スタッキング
# --------------------------

def _steps_lv9_stacking(metric: str) -> List[TutorialStep]:
    return [
        TutorialStep(
            id="stacking_fit",
            title="スタッキングを構築",
            body="ベース学習器を複数選び、メタ学習器(Logistic/Linear)で統合。",
            action="stacking_fit_done",
        ),
        TutorialStep(
            id="cards_export",
            title="モデルカードをCSV出力",
            body="複数カードを比較し、CSVにエクスポート。",
            action="cards_export_done",
        ),
    ]


# =====================================================
# API 層
# =====================================================
class TutorialGetResponse(BaseModel):
    flow: Dict


class ProgressPayload(BaseModel):
    problem: str
    event: str
    uid: Optional[str] = "anonymous"


class PrefUIDPayload(BaseModel):
    problem: str
    uid: Optional[str] = "anonymous"


tutorial_router = APIRouter(tags=["tutorial"])


@tutorial_router.get("/tutorial", response_model=TutorialGetResponse)
def get_tutorial(problem: str = Query(..., description="pref code e.g. okinawa"),
                 uid: str = Query("anonymous")):
    flow = build_flow(problem)
    state = load_state(uid, problem)

    # current_index/done_ids を反映
    flow.current_index = min(state.get("current_index", 0), len(flow.steps))
    return {"flow": flow.to_public_dict(done_ids=state.get("done_ids", []))}


@tutorial_router.post("/tutorial/progress", response_model=TutorialGetResponse)
def post_progress(payload: ProgressPayload):
    flow = build_flow(payload.problem)
    state = load_state(payload.uid, payload.problem)
    cur_idx = min(state.get("current_index", 0), len(flow.steps))

    # 既に完了済みならそのまま返す
    if cur_idx >= len(flow.steps):
        return {"flow": flow.to_public_dict(done_ids=state.get("done_ids", []))}

    current_step = flow.steps[cur_idx]

    # アクション一致で完了とする
    def _matches(expected: str, event: str) -> bool:
        # 'select_model:GradientBoosting|AdaBoost' のように OR 指定を許容
        if expected.startswith("select_model:") and event.startswith("select_model:"):
            ex = expected.split(":", 1)[1]
            ev = event.split(":", 1)[1]
            return ev in ex.split("|")
        return expected == event

    if _matches(current_step.action, payload.event):
        done_ids = set(state.get("done_ids", []))
        done_ids.add(current_step.id)
        state["done_ids"] = sorted(done_ids)
        state["current_index"] = cur_idx + 1
        save_state(payload.uid, payload.problem, state)
    else:
        # 期待と異なるイベント → 進捗は保持、ヒントを返す
        pass

    # 反映して返却
    flow.current_index = min(state.get("current_index", 0), len(flow.steps))
    return {"flow": flow.to_public_dict(done_ids=state.get("done_ids", []))}


@tutorial_router.post("/tutorial/reset", response_model=TutorialGetResponse)
def post_reset(payload: PrefUIDPayload):
    # リセット
    save_state(payload.uid, payload.problem, {"current_index": 0, "done_ids": []})
    flow = build_flow(payload.problem)
    return {"flow": flow.to_public_dict(done_ids=[])}


@tutorial_router.post("/tutorial/skip", response_model=TutorialGetResponse)
def post_skip(payload: PrefUIDPayload):
    flow = build_flow(payload.problem)
    state = load_state(payload.uid, payload.problem)
    cur_idx = min(state.get("current_index", 0), len(flow.steps))

    if cur_idx < len(flow.steps):
        # 現在ステップをスキップ（done扱いにして次へ）
        step = flow.steps[cur_idx]
        done_ids = set(state.get("done_ids", []))
        done_ids.add(step.id)
        state["done_ids"] = sorted(done_ids)
        state["current_index"] = cur_idx + 1
        save_state(payload.uid, payload.problem, state)

    flow.current_index = min(state.get("current_index", 0), len(flow.steps))
    return {"flow": flow.to_public_dict(done_ids=state.get("done_ids", []))}


# =====================================================
# 便利: Flow を直接確認する CLI 的関数（任意）
# =====================================================
if __name__ == "__main__":
    # 簡易テスト: python services/tutorial.py で現在の okinawa のフローを表示
    pref = "okinawa"
    try:
        f = build_flow(pref)
        print(json.dumps(f.to_public_dict(), ensure_ascii=False, indent=2))
    except Exception as e:
        print("Error:", e)
