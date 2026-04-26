import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import backend.api as api_module
from backend.api import router
from backend.db import init_db
from backend.ws_manager import WSManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler replacing the deprecated @app.on_event."""
    init_db()
    sim_service = api_module.ensure_sim()
    if sim_service.llm.is_enabled():
        print("--- ATLAS AI Mode Active: LLM CEO is taking charge! ---")
    else:
        print("--- ATLAS Manual/Random Mode: No API keys found. ---")
    yield
    # Cleanup (if needed) goes here after yield


app = FastAPI(title="ATLAS Backend", lifespan=lifespan)
ws_manager = WSManager()
stream_task = None
stream_lock = asyncio.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep existing frontend contract under `/api/*`.
app.include_router(router, prefix="/api")
# Also expose OpenEnv-style endpoints at the root (`/reset`, `/step`, `/state`),
# so OpenEnv clients/judges can interact without needing the `/api` prefix.
app.include_router(router)


async def simulation_stream_loop():
    global stream_task
    last_done_episode_id = None
    try:
        while ws_manager.connections:
            if api_module.sim_paused:
                await asyncio.sleep(0.5)
                continue

            sim = api_module.ensure_sim()
            if sim.done:
                if sim.episode_id != last_done_episode_id:
                    # Use state_snapshot() to avoid direct internal attribute access.
                    await ws_manager.broadcast("episode_done", {"final_state": sim.env.state_snapshot()})
                    last_done_episode_id = sim.episode_id
                await asyncio.sleep(1.0)
                continue

            frame = sim.step()
            await ws_manager.broadcast("state_update", frame)
            if frame["event"]:
                await ws_manager.broadcast("market_event", {"event": frame["event"]})
            await ws_manager.broadcast("reward_update", {"reward": frame["reward"]})
            if frame["done"]:
                await ws_manager.broadcast("episode_done", {"final_state": frame["state"]})
                last_done_episode_id = frame["episode_id"]
            await asyncio.sleep(1.0 / api_module.sim_speed)
    finally:
        stream_task = None


async def ensure_stream_started():
    global stream_task
    async with stream_lock:
        if stream_task is None or stream_task.done():
            stream_task = asyncio.create_task(simulation_stream_loop())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    await ensure_stream_started()
    try:
        while True:
            # Keep socket alive; server pushes updates through broadcast.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)

# Mount frontend static files if they exist (for production deployment)
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"msg": "Frontend not built. Run 'npm run build' in frontend directory."}
