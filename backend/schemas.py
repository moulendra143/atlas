from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    preset: str = "startup"
    mandate: Optional[str] = None  # Optional board mandate override (see MANDATES in startup_env.py)



class StepRequest(BaseModel):
    action_idx: int = Field(..., ge=0, description="Discrete action index (0-12).")


class ReplayStepOut(BaseModel):
    day: int
    phase: str
    action: str
    reward: float
    event: Optional[Dict[str, Any]]
    state: Dict[str, Any]
