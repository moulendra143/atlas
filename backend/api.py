import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.db import EpisodeLog, SessionLocal, StepLog
from backend.schemas import ResetRequest, StepRequest
from backend.services.report import generate_investor_report
from backend.services.simulator import SimulationService
from env.startup_env import PHASES

router = APIRouter()
sim = None
sim_paused = False
sim_speed = 1.0


def ensure_sim() -> SimulationService:
    global sim
    if sim is None:
        sim = SimulationService()
    return sim


@router.post("/reset")
def reset(req: ResetRequest):
    global sim
    sim = SimulationService(preset=req.preset, mandate=req.mandate)
    # Use state_snapshot() — public read-only view, avoids direct internal access.
    return {"ok": True, "state": sim.env.state_snapshot(), "episode_id": sim.episode_id}


@router.post("/step")
def step(req: StepRequest):
    return ensure_sim().step(req.action_idx)


@router.get("/state")
def state():
    """Returns current environment state. Matches AtlasObservation schema for OpenEnv clients."""
    current_sim = ensure_sim()
    return {
        "state": current_sim.env.state_snapshot(),  # public read-only view
        "reward": 0.0,          # AtlasObservation compatibility field
        "done": current_sim.done,
        "info": {
            "log_size": len(current_sim.decision_log),
            "episode_id": current_sim.episode_id,
            "day": current_sim.env.day,
            "phase": PHASES[current_sim.env.phase_idx],
        },
    }


@router.post("/pause")
def pause():
    global sim_paused
    sim_paused = True
    return {"paused": True}


@router.post("/resume")
def resume():
    global sim_paused
    sim_paused = False
    return {"paused": False}


@router.post("/speed")
def speed(val: float = 1.0):
    global sim_speed
    sim_speed = max(0.1, min(val, 5.0))
    return {"speed": sim_speed}


@router.get("/leaderboard")
def leaderboard(limit: int = 20):
    db = SessionLocal()
    try:
        rows = db.query(EpisodeLog).order_by(EpisodeLog.total_reward.desc()).limit(limit).all()
        return [
            {
                "id": row.id,
                "mode": row.mode,
                "policy_name": row.policy_name,
                "total_reward": row.total_reward,
                "steps": row.steps,
                "created_at": row.created_at.isoformat() + "Z",
                "final_cash": row.final_cash,
                "final_revenue": row.final_revenue,
            }
            for row in rows
        ]
    finally:
        db.close()


@router.get("/replay/{episode_id}")
def replay_episode(episode_id: int):
    db = SessionLocal()
    try:
        steps = (
            db.query(StepLog)
            .filter(StepLog.episode_id == episode_id)
            .order_by(StepLog.id.asc())
            .all()
        )
        if not steps:
            raise HTTPException(status_code=404, detail="Episode not found")
        return [
            {
                "day": step.day,
                "phase": step.phase,
                "action": step.action,
                "reward": step.reward,
                "event": step.event,
                "state": step.state,
            }
            for step in steps
        ]
    finally:
        db.close()


@router.get("/investor-report/{episode_id}")
def investor_report(episode_id: int):
    db = SessionLocal()
    try:
        ep = db.query(EpisodeLog).filter(EpisodeLog.id == episode_id).first()
        if not ep:
            raise HTTPException(status_code=404, detail="Episode not found")
    finally:
        db.close()

    os.makedirs("data", exist_ok=True)
    path = f"data/investor_report_{episode_id}.pdf"
    generate_investor_report(
        path,
        {
            "episode_id": ep.id,
            "mode": ep.mode,
            "policy": ep.policy_name,
            "total_reward": round(ep.total_reward, 2),
            "final_cash": round(ep.final_cash, 2),
            "final_revenue": round(ep.final_revenue, 2),
            "steps": ep.steps,
        },
    )
    return FileResponse(path=path, filename=os.path.basename(path), media_type="application/pdf")
