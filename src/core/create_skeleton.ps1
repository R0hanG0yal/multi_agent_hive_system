$root = 'C:\Users\pc\OneDrive\Desktop\multi_agent_hive_system'

$dirs = @(
  "$root",
  "$root\src",
  "$root\src\core",
  "$root\src\agents",
  "$root\src\decision",
  "$root\src\memory",
  "$root\src\reward",
  "$root\agents_config",
  "$root\tests",
  "$root\scripts",
  "$root\notebooks",
  "$root\docker"
)
foreach ($d in $dirs) { New-Item -Path $d -ItemType Directory -Force | Out-Null }

function Write-Here($path, $content) {
  $dir = Split-Path $path -Parent
  if (-not (Test-Path $dir)) { New-Item -Path $dir -ItemType Directory -Force | Out-Null }
  $content | Set-Content -Path $path -Encoding UTF8
}

# Core
Write-Here "$root\src\__init__.py" "'multi_agent_hive_system package'"
Write-Here "$root\src\core\__init__.py" "'core components'"

Write-Here "$root\src\core\vector_encoder.py" @'
"""vector_encoder.py
Simple encoder interface (stub).
"""
from typing import List

def encode(text: str) -> List[float]:
    """Return a dummy embedding for text. Replace with provider implementation."""
    return [0.0]
'@

Write-Here "$root\src\core\shared_knowledge_space.py" @'
"""shared_knowledge_space.py
In-memory vector store and metadata (dev stub).
"""
from typing import Any, Dict, List

class SharedKnowledgeSpace:
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def upsert(self, id: str, vector: List[float], meta: Dict[str, Any]):
        self.vectors[id] = vector
        self.metadata[id] = meta

    def query(self, vector: List[float], top_k: int = 5):
        # TODO: implement similarity search
        return []
'@

Write-Here "$root\src\core\router_planner.py" @'
"""router_planner.py
Simple rule-based router to map requests to agent nodes.
"""
from typing import List

def route(intention: str) -> List[str]:
    mapping = {
        "food": ["food_agent"],
        "business": ["business_agent"],
        "code": ["coding_agent"],
        "research": ["research_agent"],
    }
    return mapping.get(intention, ["research_agent","coding_agent"])
'@

# Base agent + utils (combined)
Write-Here "$root\src\base_agent.py" @'
"""
Base agent, models, and utils (starter).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
from pydantic import BaseModel, Field

class AgentConfig(BaseModel):
    agent_id: str
    agent_type: str
    domain: str
    model: str = "gpt-4"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(500, gt=0)
    enabled: bool = True

class Decision(BaseModel):
    agent_id: str
    decision_type: str
    content: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class Logger:
    @staticmethod
    def setup(name: str, level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        if not logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            h.setFormatter(fmt)
            logger.addHandler(h)
        return logger

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str, config: Optional[AgentConfig] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.logger = Logger.setup(f"Agent.{agent_id}")
    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any]) -> Decision:
        pass
    @abstractmethod
    def extract_domain_knowledge(self, query: str) -> Dict[str, Any]:
        pass
'@

# Agents
Write-Here "$root\src\agents\__init__.py" "'agents package'"

Write-Here "$root\src\agents\food_agent.py" @'
from src.base_agent import BaseAgent, Decision
class FoodAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="answer", content="FoodAgent stub", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"food"}
'@

Write-Here "$root\src\agents\business_agent.py" @'
from src.base_agent import BaseAgent, Decision
class BusinessAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="answer", content="BusinessAgent stub", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"business"}
'@

Write-Here "$root\src\agents\coding_agent.py" @'
from src.base_agent import BaseAgent, Decision
class CodingAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="code", content="CodingAgent stub", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"coding"}
'@

Write-Here "$root\src\agents\research_agent.py" @'
from src.base_agent import BaseAgent, Decision
class ResearchAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="research", content="ResearchAgent stub", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"research"}
'@

Write-Here "$root\src\agents\memory_agent.py" @'
from src.base_agent import BaseAgent, Decision
class MemoryAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="memory", content="MemoryAgent stub", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"memory"}
'@

# Decision
Write-Here "$root\src\decision\conflict_resolver.py" @'
def resolve(responses):
    if not responses: return None
    # choose highest confidence
    return max(responses, key=lambda r: getattr(r, "confidence", r.get("confidence", 0)))
'@

Write-Here "$root\src\decision\fusion_engine.py" @'
def fuse(responses):
    if not responses: return None
    total = 0.0; wsum = 0.0
    for r in responses:
        s = r.get("score", getattr(r, "confidence", 0.0))
        w = r.get("weight", 1.0)
        total += s * w; wsum += w
    return {"fused_score": total / max(wsum, 1e-9)}
'@

# Memory
Write-Here "$root\src\memory\update_manager.py" @'
def update(memory_store, new_record):
    memory_store.append(new_record)
    return True
'@

Write-Here "$root\src\memory\decay_mechanism.py" @'
def apply_decay(score, seconds):
    return score * (0.999 ** (seconds/3600))
'@

Write-Here "$root\src\memory\probability_adjuster.py" @'
def calibrate(p):
    return max(0.0, min(1.0, p))
'@

# Reward
Write-Here "$root\src\reward\evaluator.py" @'
def evaluate(response, ground_truth=None):
    return {"reward": 0.0}
'@

# Configs
Write-Here "$root\agents_config\food_agent.yaml" "name: food_agent`npriority: 1"
Write-Here "$root\agents_config\business_agent.yaml" "name: business_agent`npriority: 1"
Write-Here "$root\agents_config\coding_agent.yaml" "name: coding_agent`npriority: 1"
Write-Here "$root\agents_config\research_agent.yaml" "name: research_agent`npriority: 1"
Write-Here "$root\agents_config\memory_config.yaml" "memory_retention_days: 30"

# Scripts, tests, notebook, docker, project files
Write-Here "$root\scripts\run_multi_agent.py" @'
from src.base_agent import Logger
if __name__ == "__main__":
    logger = Logger.setup("demo")
    logger.info("Run multi-agent demo (stub)")
'@

Write-Here "$root\scripts\demo_hive.py" "print('Demo hive')"
Write-Here "$root\scripts\visualize_agents.py" "print('visualize agents (stub)')"
Write-Here "$root\tests\test_agents.py" "def test_placeholder():`n    assert True"
Write-Here "$root\notebooks\hive_tutorial.ipynb" "{}"
Write-Here "$root\docker\Dockerfile" @'
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt || true
CMD ["python","scripts/run_multi_agent.py"]
'@
Write-Here "$root\pyproject.toml" @'
[project]
name = "multi_agent_hive_system"
version = "0.0.1"
'@
Write-Here "$root\requirements.txt" "pydantic`npytest"
Write-Here "$root\README.md" "# Multi Agent Hive System`nSkeleton created by assistant."
Write-Here "$root\LICENSE" "MIT"

Write-Host "Skeleton created at $root"