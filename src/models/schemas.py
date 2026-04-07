from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class Observation(BaseModel):
    task_id: str
    instruction: str
    retrieved_memories: List[MemoryEntry]
    history: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0

class Action(BaseModel):
    response: str
    decision_type: str
    memory_update: Optional[str] = None

class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    correctness: float
    memory_utility: float
    improvement_bonus: float
    explanation: str

class EnvState(BaseModel):
    is_done: bool
    current_task_idx: int
    memory_store: List[MemoryEntry] = Field(default_factory=list)
    performance_history: List[float] = Field(default_factory=list)
