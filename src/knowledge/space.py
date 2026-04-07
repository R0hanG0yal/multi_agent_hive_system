"""
Shared Knowledge Space + Conflict Resolution Node
-------------------------------------------------
Flowchart steps:
  • Agents write results into the Shared Knowledge Space (async read/write)
  • Conflict Resolution Node detects contradictions between agent outputs
  • Decision Engine checks confidence scores, weights outputs, combines into best response

Data model: AgentOutput list → ConflictReport → DecisionOutput
"""
from dataclasses import dataclass
from typing import List, Optional
from src.agents.base_node import AgentOutput


@dataclass
class ConflictReport:
    has_conflict: bool
    conflicting_agents: List[str]
    conflict_description: str
    resolution_strategy: str  # 'highest_confidence' | 'merge' | 'escalate'


@dataclass
class DecisionOutput:
    chosen_response: str
    primary_agent: str
    confidence: float
    all_agent_scores: dict
    conflict_report: ConflictReport
    memory_context: str


class SharedKnowledgeSpace:
    """
    In-process blackboard where agent outputs are aggregated
    and made available for conflict resolution and the decision engine.
    """
    def __init__(self):
        self._entries: List[AgentOutput] = []

    def write(self, output: AgentOutput):
        self._entries.append(output)

    def read_all(self) -> List[AgentOutput]:
        return list(self._entries)

    def clear(self):
        self._entries = []


class ConflictResolutionNode:
    """
    Scans all agent outputs for factual contradictions.
    Contradictions are detected via simple divergence in key numeric/factual tokens.
    """

    def resolve(self, outputs: List[AgentOutput]) -> ConflictReport:
        if len(outputs) <= 1:
            return ConflictReport(
                has_conflict=False, conflicting_agents=[],
                conflict_description="Single agent — no conflict possible.",
                resolution_strategy="highest_confidence"
            )

        # Detect numeric discrepancies (e.g., two agents cite different temperatures)
        import re
        agent_numbers: dict[str, set] = {}
        for out in outputs:
            nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', out.response))
            if nums:
                agent_numbers[out.agent_name] = nums

        conflicting_agents = []
        if len(agent_numbers) >= 2:
            names = list(agent_numbers.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    overlap = agent_numbers[names[i]] & agent_numbers[names[j]]
                    exclusive_i = agent_numbers[names[i]] - agent_numbers[names[j]]
                    exclusive_j = agent_numbers[names[j]] - agent_numbers[names[i]]
                    if exclusive_i and exclusive_j:
                        conflicting_agents.extend([names[i], names[j]])

        conflicting_agents = list(set(conflicting_agents))
        has_conflict = len(conflicting_agents) > 0

        return ConflictReport(
            has_conflict=has_conflict,
            conflicting_agents=conflicting_agents,
            conflict_description=(
                f"Numeric discrepancy detected between {conflicting_agents}."
                if has_conflict else "No conflicts detected."
            ),
            resolution_strategy="highest_confidence" if has_conflict else "merge"
        )


class DecisionEngine:
    """
    Flowchart: Decision Engine checks confidence scores of each agent,
    weights them, combines into a new source and selects best option.

    Strategy:
      1. Filter out MemoryAgentNode (it provides context, not final answer).
      2. Apply Q-learning weights from the RL controller to boost proven agents.
      3. Select highest weighted-confidence agent as primary.
      4. If top-2 agents are within 0.1 confidence, merge their responses.
    """

    def decide(
        self,
        outputs: List[AgentOutput],
        conflict_report: ConflictReport,
        memory_context: str,
        q_weights: dict,  # {agent_name: learned_weight}
        groq_client
    ) -> DecisionOutput:

        # Exclude memory node from final answer selection
        specialist_outputs = [o for o in outputs if o.agent_name != "MemoryAgentNode" and o.response.strip()]

        if not specialist_outputs:
            return DecisionOutput(
                chosen_response="I was unable to generate a response. Please try again.",
                primary_agent="None",
                confidence=0.0,
                all_agent_scores={},
                conflict_report=conflict_report,
                memory_context=memory_context
            )

        # Apply Q-learning weights to raw confidence
        weighted_scores = {}
        for out in specialist_outputs:
            rl_weight = q_weights.get(out.agent_name, 1.0)  # Default weight=1.0
            weighted_score = out.confidence * rl_weight
            weighted_scores[out.agent_name] = {
                "raw_confidence": out.confidence,
                "rl_weight": rl_weight,
                "weighted_score": weighted_score
            }

        # Sort by weighted score
        ranked = sorted(specialist_outputs, key=lambda o: weighted_scores[o.agent_name]["weighted_score"], reverse=True)
        primary = ranked[0]

        # Merge if top-2 are very close (within 0.12 weighted score)
        chosen_response = primary.response
        if len(ranked) >= 2:
            score_gap = (weighted_scores[ranked[0].agent_name]["weighted_score"] -
                         weighted_scores[ranked[1].agent_name]["weighted_score"])
            if score_gap < 0.12:
                chosen_response = self._merge_responses(ranked[0], ranked[1], groq_client, memory_context)

        return DecisionOutput(
            chosen_response=chosen_response,
            primary_agent=primary.agent_name,
            confidence=weighted_scores[primary.agent_name]["weighted_score"],
            all_agent_scores=weighted_scores,
            conflict_report=conflict_report,
            memory_context=memory_context
        )

    def _merge_responses(self, a: AgentOutput, b: AgentOutput, groq_client, memory_context: str) -> str:
        """LLM-assisted response merging when two agents are equally confident."""
        try:
            merge_prompt = (
                f"You are a master synthesis agent. Merge these two expert responses into "
                f"a single, superior, non-redundant answer.\n\n"
                f"[{a.agent_name}]:\n{a.response}\n\n"
                f"[{b.agent_name}]:\n{b.response}\n\n"
                f"{'Personal context: ' + memory_context if memory_context else ''}\n\n"
                f"Synthesized answer:"
            )
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": merge_prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=600,
                temperature=0.4
            )
            return completion.choices[0].message.content
        except Exception:
            return a.response  # Fallback to highest-confidence agent
