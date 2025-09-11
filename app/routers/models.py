# app/routers/models.py
from fastapi import APIRouter

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/ping")
def ping_models():
    return {"ok": True}
