from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    task_id: str
    step: int
    message: str


class Action(BaseModel):
    action_type: str  # "classify", "revise", "flag_uncertain"
    content: Optional[str] = None


class Reward(BaseModel):
    value: float