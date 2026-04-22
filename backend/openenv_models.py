from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AtlasAction(BaseModel):
    """
    Minimal action model for OpenEnv manifest compatibility.

    Note: ATLAS also supports the existing `/api/step` contract via `backend.schemas.StepRequest`.
    This model describes the same payload in a more explicit form for OpenEnv-style tooling.
    """

    action_idx: int = Field(..., ge=0, description="Discrete action index to execute.")


class AtlasObservation(BaseModel):
    """
    Minimal observation model for OpenEnv manifest compatibility.

    The running backend returns a richer JSON payload from `/step`; this schema captures the
    common fields used for training/eval loops and judging demos.
    """

    state: Dict[str, Any] = Field(default_factory=dict, description="Current simulator state.")
    reward: float = Field(0.0, description="Reward for the last transition.")
    done: bool = Field(False, description="Episode termination flag.")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Additional step metadata.")

