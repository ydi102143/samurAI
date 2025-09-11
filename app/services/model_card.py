# app/services/model_card.py
from __future__ import annotations
from pathlib import Path
import json, time, uuid
from typing import Any, Dict, List, Optional

ROOT = Path("storage/models")

def _cards_path(region: str, problem: str) -> Path:
    p = ROOT / region / problem / "cards.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _read_cards(region: str, problem: str) -> List[dict]:
    p = _cards_path(region, problem)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def _write_cards(region: str, problem: str, cards: List[dict]) -> None:
    p = _cards_path(region, problem)
    p.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

def save_card(
    region: str,
    problem: str,
    model: str,
    task: str,
    score: float,
    metric: str,
    passed: bool,
    params: Dict[str, Any] | None = None,
    features: List[str] | None = None,
    pipeline: List[dict] | None = None,
    extras: Dict[str, Any] | None = None,
) -> dict:
    """学習/評価の結果を1件のカードとして追記保存。返り値は保存したカード。"""
    cards = _read_cards(region, problem)
    card = {
        "id": str(uuid.uuid4())[:8],
        "ts": int(time.time()),
        "region": region,
        "problem": problem,
        "task": task,
        "model": model,
        "score": float(score),
        "metric": metric,
        "passed": bool(passed),
        "params": params or {},
        "features": features or [],
        "pipeline": pipeline or [],
        "extras": extras or {},
    }
    cards.append(card)
    # 新しい順に
    cards = sorted(cards, key=lambda x: x["ts"], reverse=True)
    _write_cards(region, problem, cards)
    return card

def list_cards(region: str, problem: str) -> List[dict]:
    return _read_cards(region, problem)

def get_card(region: str, problem: str, card_id: str) -> Optional[dict]:
    for c in _read_cards(region, problem):
        if c.get("id") == card_id:
            return c
    return None

def compare_cards(cards: List[dict], keys: List[str] | None = None) -> List[dict]:
    """カードを同一キーセットで比較用に並べ替える（薄いヘルパ）。"""
    keys = keys or ["metric", "score", "model", "task", "passed", "ts"]
    # 表形式にしやすい配列へ
    out = []
    for c in cards:
        row = {k: c.get(k) for k in keys}
        row["id"] = c.get("id")
        row["problem"] = c.get("problem")
        out.append(row)
    return out
