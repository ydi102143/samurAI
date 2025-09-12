# app/main.py
from __future__ import annotations

import base64
import re
from fastapi import FastAPI, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import Response

# 透明1x1 PNG（プレースホルダー）
_TRANSPARENT_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

def create_app() -> FastAPI:
    app = FastAPI(title="samurAI", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 静的ファイル
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # ===== ミドルウェア：?problem= 空の EDA画像要求は 200 PNG で吸収 =====
    @app.middleware("http")
    async def eda_empty_problem_guard(request: Request, call_next):
        path = request.url.path
        if re.search(r"/v1/datasets/[^/]+/eda/plot/", path):
            q = dict(request.query_params)
            problem = q.get("problem")
            if not problem:  # None or ""
                return Response(
                    content=_TRANSPARENT_PNG,
                    media_type="image/png",
                    headers={"Cache-Control": "no-store"},
                )
        return await call_next(request)

    # ---- ルータ登録 ----
    from app.routers.offline import page_router, router as datasets_router

    # ページ: /offline/play/{problem} と /v1/region/task（page_router内で定義）
    app.include_router(page_router)

    # API: /v1/datasets/*  （※ 追加prefixは絶対に付けない）
    app.include_router(datasets_router)

    # ヘルス
    health = APIRouter(tags=["health"])
    @health.get("/health")
    def _health():
        return {"ok": True}
    @health.get("/version")
    def _version():
        return {"version": app.version}
    app.include_router(health)

    return app

# これだけ残す
app = create_app()
if __name__ == "__main__":
    import os, uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
