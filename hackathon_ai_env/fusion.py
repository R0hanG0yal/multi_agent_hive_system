from __future__ import annotations

from .agents import encode_query_vector
from .models import AgentAction, AgentProposal, FusionResult, MemoryRecord
from .utils import clamp


class DecisionFusionEngine:
    def __init__(self, thresholds: tuple[float, ...] = (0.45, 0.65, 0.8)) -> None:
        self.thresholds = thresholds

    def combine(
        self,
        query: str,
        proposals: dict[str, AgentProposal],
        recalled_memory: list[MemoryRecord],
        prob_matrix: dict[str, float] = None,
        explicit_shares: tuple[tuple[str, str], ...] = (),
    ) -> FusionResult:
        query_vector = encode_query_vector(query)
        agent_scores: dict[str, float] = {}
        share_targets = {target for _, target in explicit_shares}
        for agent_name, proposal in proposals.items():
            score = proposal.confidence + (0.06 if proposal.memory_response else 0.0)
            if proposal.domain != "memory":
                score += 0.08 * query_vector.get(proposal.domain, 0.0)
                if proposal.domain in share_targets:
                    score += 0.1
            else:
                score += 0.08 if recalled_memory else -0.04
            agent_scores[agent_name] = round(clamp(score), 4)

        ranked_agents = tuple(
            sorted(agent_scores, key=lambda name: (-agent_scores[name], name))
        )
        top_score = agent_scores[ranked_agents[0]]
        second_score = agent_scores[ranked_agents[1]] if len(ranked_agents) > 1 else 0.0
        margin = round(top_score - second_score, 4)

        if prob_matrix is None:
            prob_matrix = {}
            
        scored_actions: list[tuple[float, AgentAction]] = []
        for agent_name in ranked_agents:
            proposal = proposals[agent_name]
            weight = prob_matrix.get(agent_name, 1.0)
            
            for use_memory in (False, True):
                for threshold in self.thresholds:
                    # Implement diagram equation: sum(agent_Score * confidence * weight)
                    utility = agent_scores[agent_name] * proposal.confidence * weight
                    
                    # Apply tiny memory utility if appropriate to break symmetrical ties
                    if use_memory and proposal.memory_response:
                        utility += 0.08
                    elif use_memory:
                        utility -= 0.03
                        
                    scored_actions.append(
                        (
                            round(utility, 4),
                            AgentAction(
                                selected_agent=agent_name,
                                use_memory=use_memory,
                                confidence_threshold=threshold,
                            ),
                        )
                    )

        scored_actions.sort(key=lambda item: (-item[0], item[1].key))
        candidate_actions = tuple(action for _, action in scored_actions[:12])

        return FusionResult(
            agent_scores=agent_scores,
            ranked_agents=ranked_agents,
            candidate_actions=candidate_actions,
            consensus_agent=ranked_agents[0],
            top_score_margin=margin,
            query_vector=query_vector,
        )
