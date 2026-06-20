from __future__ import annotations

from dataclasses import asdict

from .agents import build_default_agents
from .fusion import DecisionFusionEngine
from .live_knowledge import KnowledgeRetriever
from .memory import SharedKnowledgeSpace
from .models import AgentAction, EpisodeSummary, FeedbackReward, InferenceResult, QueryScenario, StepResult
from .q_learning import QLearningController
from .reward import DeterministicRewardNode, UserFeedbackRewardNode
from .state import StateEncoder
from .utils import parse_memory_sharing_preferences, tokenize
from .conflict import ConflictResolutionNode
from .llm_client import llm_enhance


class HackathonAIEnvironment:
    def __init__(
        self,
        seed: int = 7,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.agents = build_default_agents(knowledge_retriever=knowledge_retriever)
        self.memory = SharedKnowledgeSpace()
        self.fusion_engine = DecisionFusionEngine()
        self.state_encoder = StateEncoder()
        self.reward_node = DeterministicRewardNode()
        self.feedback_reward_node = UserFeedbackRewardNode()
        self.conflict_resolver = ConflictResolutionNode()
        self.q_controller = QLearningController(seed=seed)
        self.audit_log: list[dict[str, object]] = []

    def reset_episode(self) -> None:
        self.memory.reset()

    def _build_context(
        self,
        query: str,
        allow_live: bool = False,
    ):
        explicit_shares = parse_memory_sharing_preferences(query)
        proposal_memory: dict[str, list] = {}
        proposals = {}
        for name, agent in self.agents.items():
            target_domain = None if getattr(agent, "domain", None) == "memory" else agent.domain
            recalled = self.memory.recall(
                query,
                target_domain=target_domain,
                explicit_shares=explicit_shares,
            )
            proposal_memory[name] = recalled
            proposals[name] = agent.propose(query, recalled, allow_live=allow_live)
        recalled = self._merge_recalled_memory(proposal_memory)
        proposals = self.conflict_resolver.resolve(proposals)
        prob_matrix = self.memory.probability_of_use_matrix()
        fusion = self.fusion_engine.combine(
            query,
            proposals,
            recalled,
            prob_matrix,
            explicit_shares=explicit_shares,
        )
        state_key = self.state_encoder.encode(fusion, recalled, query=query)
        return recalled, proposal_memory, proposals, fusion, state_key, explicit_shares

    def _merge_recalled_memory(self, proposal_memory: dict[str, list]) -> list:
        merged: list = []
        seen: set[str] = set()
        for records in proposal_memory.values():
            for record in records:
                if record.key in seen:
                    continue
                merged.append(record)
                seen.add(record.key)
        return merged

    def _trace_payload(
        self,
        query: str,
        proposal_memory: dict[str, list],
        proposals,
        fusion,
        state_key: str,
        action=None,
        final_agent: str | None = None,
        final_recalled: list | None = None,
        reward=None,
        updated_memory_key: str | None = None,
        explicit_shares: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, object]:
        selected_agent = action.selected_agent if action is not None else None
        payload: dict[str, object] = {
            "query": query,
            "query_vector": fusion.query_vector,
            "state_key": state_key,
            "consensus_agent": fusion.consensus_agent,
            "top_score_margin": fusion.top_score_margin,
            "candidate_actions": [candidate.key for candidate in fusion.candidate_actions[:6]],
            "agent_scores": fusion.agent_scores,
            "proposal_confidences": {name: proposal.confidence for name, proposal in proposals.items()},
            "proposal_rationales": {name: proposal.rationale for name, proposal in proposals.items()},
            "memory_banks_used": {
                name: [record.key for record in records]
                for name, records in proposal_memory.items()
            },
            "shared_domains": [f"{source}->{target}" for source, target in explicit_shares],
        }
        if action is not None:
            payload["selected_action"] = {
                "agent": action.selected_agent,
                "use_memory": action.use_memory,
                "threshold": action.confidence_threshold,
            }
            payload["q_learning"] = self.q_controller.last_update
        if final_agent is not None:
            payload["final_agent"] = final_agent
        if final_recalled is not None:
            payload["final_memory_keys"] = [record.key for record in final_recalled]
        if reward is not None:
            payload["reward"] = asdict(reward)
        if updated_memory_key is not None:
            payload["updated_memory_key"] = updated_memory_key
        return payload

    def _build_explanation(
        self,
        fusion,
        action,
        final_agent: str,
        final_recalled: list,
    ) -> str:
        ranked = sorted(fusion.query_vector.items(), key=lambda item: (-item[1], item[0]))
        route_summary = ", ".join(f"{domain}={score:.3f}" for domain, score in ranked[:3])
        memory_keys = ", ".join(record.key for record in final_recalled[:3]) or "none"
        return (
            f"Vector router: {route_summary}. "
            f"Fusion consensus={fusion.consensus_agent} with margin {fusion.top_score_margin:.3f}. "
            f"Q-policy selected {action.selected_agent} "
            f"({'memory' if action.use_memory else 'fresh'}) at threshold {action.confidence_threshold:.2f}. "
            f"Final agent={final_agent}. Memory bank used: {memory_keys}."
        )

    def _record_log(self, event: str, payload: dict[str, object]) -> None:
        entry = {"event": event, **payload}
        self.audit_log.append(entry)
        if len(self.audit_log) > 120:
            self.audit_log = self.audit_log[-120:]

    def _resolve_final_agent(self, action, fusion, proposals):
        selected = proposals[action.selected_agent]
        if selected.confidence >= action.confidence_threshold:
            return action.selected_agent, selected
        fallback = fusion.consensus_agent
        return fallback, proposals[fallback]

    def _find_memory_variant(self, selected_agent: str, threshold: float, candidate_actions: tuple[AgentAction, ...]) -> AgentAction | None:
        exact = next(
            (
                candidate
                for candidate in candidate_actions
                if candidate.selected_agent == selected_agent
                and candidate.use_memory
                and candidate.confidence_threshold == threshold
            ),
            None,
        )
        if exact is not None:
            return exact
        return next(
            (
                candidate
                for candidate in candidate_actions
                if candidate.selected_agent == selected_agent and candidate.use_memory
            ),
            None,
        )

    def _prefer_memory_for_user_query(self, action, fusion, proposals, recalled):
        memory_priority_prefixes = ("preference:", "health:", "profile:", "kitchen:", "ingredient:", "feedback:")
        if not recalled or not any(record.key.startswith(memory_priority_prefixes) for record in recalled):
            return action

        selected = proposals[action.selected_agent]
        if selected.memory_response:
            preferred = self._find_memory_variant(
                selected_agent=action.selected_agent,
                threshold=action.confidence_threshold,
                candidate_actions=fusion.candidate_actions,
            )
            if preferred is not None:
                return preferred
            return AgentAction(
                selected_agent=action.selected_agent,
                use_memory=True,
                confidence_threshold=action.confidence_threshold,
            )

        fallback_agent = fusion.consensus_agent
        fallback = proposals[fallback_agent]
        if fallback.memory_response:
            preferred = self._find_memory_variant(
                selected_agent=fallback_agent,
                threshold=action.confidence_threshold,
                candidate_actions=fusion.candidate_actions,
            )
            if preferred is not None:
                return preferred
            return AgentAction(
                selected_agent=fallback_agent,
                use_memory=True,
                confidence_threshold=action.confidence_threshold,
            )
        return action

    def _peek_next_state(self, scenario: QueryScenario) -> tuple[str, tuple[str, ...]]:
        recalled, proposal_memory, proposals, fusion, state_key, explicit_shares = self._build_context(scenario.query, allow_live=False)
        _ = recalled
        _ = proposal_memory
        _ = explicit_shares
        return state_key, tuple(action.key for action in fusion.candidate_actions)

    def run_training_step(
        self,
        scenario: QueryScenario,
        next_scenario: QueryScenario | None = None,
    ) -> StepResult:
        recalled, proposal_memory, proposals, fusion, state_key, explicit_shares = self._build_context(scenario.query, allow_live=False)
        action, action_mode = self.q_controller.choose_action(state_key, fusion.candidate_actions)
        final_agent, final_proposal = self._resolve_final_agent(action, fusion, proposals)
        final_recalled = proposal_memory.get(final_agent, recalled)

        if action.use_memory and final_recalled:
            self.memory.mark_access(final_recalled)

        answer = final_proposal.render(action.use_memory)
        reward = self.reward_node.evaluate(
            scenario=scenario,
            action=action,
            final_proposal=final_proposal,
            answer=answer,
            recalled_count=len(final_recalled),
        )
        updated_record = self.memory.integrate(
            scenario=scenario,
            answer=answer,
            final_domain=final_agent,
            reward=reward.total,
        )

        next_state_key = None
        next_action_keys: tuple[str, ...] = ()
        if next_scenario is not None:
            next_state_key, next_action_keys = self._peek_next_state(next_scenario)

        self.q_controller.update(
            state_key=state_key,
            action_key=action.key,
            reward=reward.total,
            next_state_key=next_state_key,
            next_action_keys=next_action_keys,
        )
        self.q_controller.decay()

        trace = self._trace_payload(
            query=scenario.query,
            proposal_memory=proposal_memory,
            proposals=proposals,
            fusion=fusion,
            state_key=state_key,
            action=action,
            final_agent=final_agent,
            final_recalled=final_recalled,
            reward=reward,
            updated_memory_key=updated_record.key,
            explicit_shares=explicit_shares,
        )
        explanation = self._build_explanation(fusion, action, final_agent, final_recalled)
        self._record_log("train_step", trace)

        return StepResult(
            scenario=scenario,
            state_key=state_key,
            action=action,
            action_mode=action_mode,
            final_agent=final_agent,
            answer=answer,
            reward=reward,
            recalled_memory_keys=tuple(record.key for record in final_recalled),
            agent_scores=fusion.agent_scores,
            explanation=explanation,
            trace=trace,
        )

    def train(
        self,
        scenarios: list[QueryScenario],
        episodes: int = 30,
    ) -> list[EpisodeSummary]:
        summaries: list[EpisodeSummary] = []
        for episode in range(1, episodes + 1):
            self.reset_episode()
            results: list[StepResult] = []
            for index, scenario in enumerate(scenarios):
                next_scenario = scenarios[index + 1] if index + 1 < len(scenarios) else None
                results.append(self.run_training_step(scenario, next_scenario))

            total_reward = sum(result.reward.total for result in results)
            accuracy = sum(
                1 for result in results if result.final_agent == result.scenario.expected_domain
            ) / len(results)
            summaries.append(
                EpisodeSummary(
                    episode=episode,
                    total_reward=round(total_reward, 4),
                    average_reward=round(total_reward / len(results), 4),
                    accuracy=round(accuracy, 4),
                    memory_items=len(self.memory.records),
                )
            )
        return summaries

    def evaluate(self, scenarios: list[QueryScenario]) -> list[StepResult]:
        self.reset_episode()
        results: list[StepResult] = []
        for scenario in scenarios:
            recalled, proposal_memory, proposals, fusion, state_key, explicit_shares = self._build_context(scenario.query, allow_live=False)
            action = self.q_controller.best_action(state_key, fusion.candidate_actions)
            final_agent, final_proposal = self._resolve_final_agent(action, fusion, proposals)
            final_recalled = proposal_memory.get(final_agent, recalled)
            answer = final_proposal.render(action.use_memory)
            reward = self.reward_node.evaluate(
                scenario=scenario,
                action=action,
                final_proposal=final_proposal,
                answer=answer,
                recalled_count=len(final_recalled),
            )
            if action.use_memory and final_recalled:
                self.memory.mark_access(final_recalled)
            updated_record = self.memory.integrate(
                scenario=scenario,
                answer=answer,
                final_domain=final_agent,
                reward=reward.total,
            )
            trace = self._trace_payload(
                query=scenario.query,
                proposal_memory=proposal_memory,
                proposals=proposals,
                fusion=fusion,
                state_key=state_key,
                action=action,
                final_agent=final_agent,
                final_recalled=final_recalled,
                reward=reward,
                updated_memory_key=updated_record.key,
                explicit_shares=explicit_shares,
            )
            explanation = self._build_explanation(fusion, action, final_agent, final_recalled)
            self._record_log("evaluation_step", trace)
            results.append(
                StepResult(
                    scenario=scenario,
                    state_key=state_key,
                    action=action,
                    action_mode="exploit",
                    final_agent=final_agent,
                    answer=answer,
                    reward=reward,
                    recalled_memory_keys=tuple(record.key for record in final_recalled),
                    agent_scores=fusion.agent_scores,
                    explanation=explanation,
                    trace=trace,
                )
            )
        return results

    def prime_memory(self, scenarios: list[QueryScenario]) -> None:
        self.evaluate(scenarios)

    def answer_query(self, query: str) -> InferenceResult:
        recalled, proposal_memory, proposals, fusion, state_key, explicit_shares = self._build_context(query, allow_live=True)
        action = self.q_controller.best_action(state_key, fusion.candidate_actions)
        action = self._prefer_memory_for_user_query(action, fusion, proposals, recalled)
        final_agent, final_proposal = self._resolve_final_agent(action, fusion, proposals)
        final_recalled = proposal_memory.get(final_agent, recalled)
        answer = final_proposal.render(action.use_memory)
        # Enhance via LLM proxy (uses API_BASE_URL / API_KEY env vars)
        answer = llm_enhance(query, answer, final_agent)
        if action.use_memory and final_recalled:
            self.memory.mark_access(final_recalled)

        self.memory.remember_memory_sharing(query)
        self.memory.remember_identity(query)
        inferred_keywords = tuple(tokenize(query)[:4]) or ("general",)
        self.memory.remember_cooking_constraints(query)
        self.memory.remember_ingredient_preferences(query)
        self.memory.remember_preference(query)
        self.memory.remember_health_profile(query)
        trace = self._trace_payload(
            query=query,
            proposal_memory=proposal_memory,
            proposals=proposals,
            fusion=fusion,
            state_key=state_key,
            action=action,
            final_agent=final_agent,
            final_recalled=final_recalled,
            explicit_shares=explicit_shares,
        )
        explanation = self._build_explanation(fusion, action, final_agent, final_recalled)
        self._record_log("inference", trace)
        return InferenceResult(
            query=query,
            state_key=state_key,
            action=action,
            final_agent=final_agent,
            answer=answer,
            recalled_memory_keys=tuple(record.key for record in final_recalled),
            agent_scores=fusion.agent_scores,
            inferred_keywords=inferred_keywords,
            explanation=explanation,
            trace=trace,
        )

    def apply_feedback(
        self,
        inference: InferenceResult,
        rating: int,
        notes: str = "",
    ) -> FeedbackReward:
        reward = self.feedback_reward_node.evaluate(inference, rating)
        self.q_controller.update(
            state_key=inference.state_key,
            action_key=inference.action.key,
            reward=reward.total,
            next_state_key=None,
            next_action_keys=(),
        )
        updated_record = self.memory.integrate(
            scenario=QueryScenario(
                query=inference.query,
                expected_domain=inference.final_agent,
                expected_keywords=inference.inferred_keywords,
                requires_memory=bool(inference.recalled_memory_keys),
            ),
            answer=inference.answer,
            final_domain=inference.final_agent,
            reward=reward.total,
            is_user_provided=True,
        )
        if notes.strip():
            self.memory.remember_memory_sharing(notes)
            self.memory.remember_feedback(
                query=inference.query,
                notes=notes,
                domain=inference.final_agent,
            )
            self.memory.remember_identity(notes)
            self.memory.remember_cooking_constraints(notes)
            self.memory.remember_ingredient_preferences(notes)
            self.memory.remember_health_profile(notes)
            self.memory.remember_preference(notes)
        self._record_log(
            "feedback",
            {
                "query": inference.query,
                "state_key": inference.state_key,
                "action_key": inference.action.key,
                "final_agent": inference.final_agent,
                "reward": asdict(reward),
                "q_learning": self.q_controller.last_update,
                "updated_memory_key": updated_record.key,
                "notes": notes.strip(),
            },
        )
        return reward
