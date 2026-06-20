from __future__ import annotations

from .models import (
    AgentAction,
    AgentProposal,
    FeedbackReward,
    InferenceResult,
    QueryScenario,
    RewardBreakdown,
)
from .utils import classify_task_difficulty, clamp, token_set


class DeterministicRewardNode:
    def evaluate(
        self,
        scenario: QueryScenario,
        action: AgentAction,
        final_proposal: AgentProposal,
        answer: str,
        recalled_count: int,
    ) -> RewardBreakdown:
        accuracy = 1.0 if final_proposal.domain == scenario.expected_domain else 0.0
        expected = set(scenario.expected_keywords)
        answer_tokens = token_set(answer)
        keyword_coverage = (
            len(expected & answer_tokens) / len(expected)
            if expected
            else 0.0
        )

        if scenario.requires_memory:
            if action.use_memory and recalled_count > 0:
                memory_usage = 1.0
            elif action.use_memory:
                memory_usage = 0.2
            else:
                memory_usage = 0.0
        else:
            if action.use_memory and recalled_count > 0:
                memory_usage = 0.8
            elif action.use_memory:
                memory_usage = 0.3
            else:
                memory_usage = 0.6

        confidence_target = 0.85 if accuracy else 0.2
        confidence_alignment = clamp(1.0 - abs(final_proposal.confidence - confidence_target))
        difficulty = classify_task_difficulty(scenario.query)
        difficulty_weight = {"easy": 0.04, "medium": 0.06, "hard": 0.08}[difficulty]
        difficulty_signal = round((accuracy + keyword_coverage) / 2, 4)
        difficulty_bonus = difficulty_weight * difficulty_signal
        total = round(
            clamp(
                (0.62 * accuracy)
                + (0.12 * keyword_coverage)
                + (0.12 * memory_usage)
                + (0.06 * confidence_alignment)
                + difficulty_bonus
            ),
            4,
        )
        return RewardBreakdown(
            total=total,
            accuracy=round(accuracy, 4),
            keyword_coverage=round(keyword_coverage, 4),
            memory_usage=round(memory_usage, 4),
            confidence_alignment=round(confidence_alignment, 4),
            task_difficulty=difficulty,
            difficulty_bonus=round(difficulty_bonus, 4),
        )


class UserFeedbackRewardNode:
    def evaluate(
        self,
        inference: InferenceResult,
        rating: int,
    ) -> FeedbackReward:
        if rating < 1 or rating > 5:
            raise ValueError("rating must be between 1 and 5")

        user_feedback = round((rating - 1) / 4, 4)
        if inference.action.use_memory and inference.recalled_memory_keys:
            memory_signal = 0.85
        elif inference.action.use_memory:
            memory_signal = 0.35
        else:
            memory_signal = 0.55

        confidence = inference.agent_scores.get(inference.final_agent, 0.5)
        confidence_alignment = clamp(1.0 - abs(confidence - user_feedback))
        difficulty = classify_task_difficulty(inference.query)
        difficulty_weight = {"easy": 0.05, "medium": 0.08, "hard": 0.12}[difficulty]
        difficulty_signal = (user_feedback + confidence_alignment) / 2
        difficulty_bonus = difficulty_weight * difficulty_signal
        total = round(
            clamp(
                (0.62 * user_feedback)
                + (0.13 * memory_signal)
                + (0.13 * confidence_alignment)
                + difficulty_bonus
            ),
            4,
        )
        return FeedbackReward(
            total=total,
            user_feedback=user_feedback,
            memory_signal=round(memory_signal, 4),
            confidence_alignment=round(confidence_alignment, 4),
            task_difficulty=difficulty,
            difficulty_bonus=round(difficulty_bonus, 4),
        )
