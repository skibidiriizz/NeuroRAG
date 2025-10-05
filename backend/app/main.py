from fastapi import FastAPI, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from . import db, auth, cache

app = FastAPI(title="NeuroRAG Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await db.init_db()
    await cache.init_redis()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/auth/login")
async def login(payload=Depends(auth.login)):
    return payload


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            # echo back for now
            await ws.send_text(f"echo: {data}")
    except Exception:
        await ws.close()
