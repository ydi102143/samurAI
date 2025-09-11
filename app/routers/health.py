# app/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/version")
def version():
    return {"app": "ml-battle", "mode": "offline-quick", "ok": True}
