import concurrent.futures
import random
from typing import Dict, Optional

from agents.employee import EmployeeAgent
from agents.personalities import PERSONALITIES
from backend.db import EpisodeLog, SessionLocal, StepLog
from env.startup_env import ACTIONS, AtlasStartupEnv
from backend.services.llm_service import LLMService


class SimulationService:
    def __init__(self, preset: str = "startup", policy_name: str = "random",
                 mandate: Optional[str] = None):
        self.preset = preset
        self.policy_name = policy_name
        self.env = AtlasStartupEnv(preset=preset)
        # Pass mandate as an option so env.reset() uses it; if None, env picks randomly.
        self.obs, self.info = self.env.reset(options={"mandate": mandate} if mandate else None)
        self.done = False
        self.decision_log = []
        self.total_reward = 0.0
        self.episode_id = None
        self.llm = LLMService()
        self.employee_agents = [
            EmployeeAgent("engineering_manager", PERSONALITIES["engineering_manager"]),
            EmployeeAgent("sales_lead", PERSONALITIES["sales_lead"]),
            EmployeeAgent("hr_recruiter", PERSONALITIES["hr_recruiter"]),
            EmployeeAgent("finance_officer", PERSONALITIES["finance_officer"]),
            EmployeeAgent("customer_success", PERSONALITIES["customer_success"]),
        ]
        self._start_episode()

    def _start_episode(self):
        db = SessionLocal()
        try:
            row = EpisodeLog(mode=self.preset, policy_name=self.policy_name)
            db.add(row)
            db.commit()
            db.refresh(row)
            self.episode_id = row.id
        finally:
            db.close()

    def step(self, action_idx=None) -> Dict:
        if action_idx is None:
            if self.llm.is_enabled():
                # Use state_snapshot() — public read-only view.
                state_plus = self.env.state_snapshot()
                state_plus["mandate"] = getattr(self.env, "mandate", "None")
                # Guide §10: Wall-clock timeout so LLM CEO never hangs the WebSocket loop.
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        future = ex.submit(self.llm.get_action, state_plus)
                        action_idx = future.result(timeout=8.0)
                except concurrent.futures.TimeoutError:
                    print("LLM CEO timed out (>8s); falling back to random action.")
                    action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                action_idx = random.randint(0, len(ACTIONS) - 1)
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        self.obs = obs
        self.done = terminated or truncated
        self.total_reward += reward
        # Always trust env-level action_name so invalid indices cannot crash logging.
        action_name = str(info.get("action_name", "invalid_action"))

        # Use state_snapshot() — avoids exposing mutable internal dict to agent layer.
        current_state = self.env.state_snapshot()
        reactions = [agent.react(action_name, current_state) for agent in self.employee_agents]
        frame = {
            "state": current_state,
            "day": info["day"],
            "phase": info["phase"],
            "mandate": getattr(self.env, "mandate", "None"),
            "action": action_name,
            "reward": reward,
            "event": {"name": info.get("event"), "severity": "critical" if info.get("event") in ["server_outage", "market_crash", "key_employee_resigns", "customer_complaints_spike"] else "info"} if info.get("event") else None,
            "reactions": reactions,
            "done": self.done,
            "episode_id": self.episode_id,
        }
        self.decision_log.append(frame)
        self._persist_step(frame)
        if self.done:
            self._finalize_episode()
        return frame

    def _persist_step(self, frame: Dict) -> None:
        db = SessionLocal()
        try:
            db.add(
                StepLog(
                    episode_id=self.episode_id,
                    day=frame["day"],
                    phase=frame["phase"],
                    action=frame["action"],
                    reward=float(frame["reward"]),
                    event=frame["event"],
                    state=frame["state"],
                )
            )
            db.commit()
        finally:
            db.close()

    def _finalize_episode(self) -> None:
        db = SessionLocal()
        try:
            ep = db.query(EpisodeLog).filter(EpisodeLog.id == self.episode_id).first()
            if ep:
                current_state = self.env.state_snapshot()
                ep.total_reward = float(self.total_reward)
                ep.steps = len(self.decision_log)
                ep.final_cash = float(current_state["cash_balance"])
                ep.final_revenue = float(current_state["revenue"])
                ep.summary = {
                    "morale": current_state["employee_morale"],
                    "customer_satisfaction": current_state["customer_satisfaction"],
                    "investor_trust": current_state["investor_trust"],
                }
                db.commit()
        finally:
            db.close()
