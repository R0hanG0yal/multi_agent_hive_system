from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryScenario:
    query: str
    expected_domain: str
    expected_keywords: tuple[str, ...]
    requires_memory: bool = False
    description: str = ""


@dataclass
class MemoryRecord:
    key: str
    domain: str
    query: str
    summary: str
    keywords: tuple[str, ...]
    weight: float = 0.5
    last_step: int = 0
    accesses: int = 0
    vector: tuple[float, ...] = field(default_factory=tuple)
    updated_at: float = 0.0
    is_user_provided: bool = False


@dataclass(frozen=True)
class AgentProposal:
    agent_name: str
    domain: str
    confidence: float
    keywords_hit: tuple[str, ...]
    base_response: str
    memory_response: str | None = None
    rationale: str = ""

    def render(self, use_memory: bool) -> str:
        if use_memory and self.memory_response:
            return self.memory_response
        return self.base_response


@dataclass(frozen=True)
class AgentAction:
    selected_agent: str
    use_memory: bool
    confidence_threshold: float

    @property
    def key(self) -> str:
        memory_flag = "mem" if self.use_memory else "fresh"
        return f"{self.selected_agent}|{memory_flag}|{self.confidence_threshold:.2f}"


@dataclass
class FusionResult:
    agent_scores: dict[str, float]
    ranked_agents: tuple[str, ...]
    candidate_actions: tuple[AgentAction, ...]
    consensus_agent: str
    top_score_margin: float
    query_vector: dict[str, float]


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    accuracy: float
    keyword_coverage: float
    memory_usage: float
    confidence_alignment: float
    task_difficulty: str = "medium"
    difficulty_bonus: float = 0.0


@dataclass(frozen=True)
class FeedbackReward:
    total: float
    user_feedback: float
    memory_signal: float
    confidence_alignment: float
    task_difficulty: str = "medium"
    difficulty_bonus: float = 0.0


@dataclass
class StepResult:
    scenario: QueryScenario
    state_key: str
    action: AgentAction
    action_mode: str
    final_agent: str
    answer: str
    reward: RewardBreakdown
    recalled_memory_keys: tuple[str, ...]
    agent_scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    trace: dict[str, object] = field(default_factory=dict)


@dataclass
class EpisodeSummary:
    episode: int
    total_reward: float
    average_reward: float
    accuracy: float
    memory_items: int


@dataclass
class InferenceResult:
    query: str
    state_key: str
    action: AgentAction
    final_agent: str
    answer: str
    recalled_memory_keys: tuple[str, ...]
    agent_scores: dict[str, float]
    inferred_keywords: tuple[str, ...]
    explanation: str = ""
    trace: dict[str, object] = field(default_factory=dict)
