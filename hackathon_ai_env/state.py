from __future__ import annotations

from .models import FusionResult, MemoryRecord
from .utils import classify_task_difficulty, parse_memory_sharing_preferences
from .vector import VectorEncoder


class StateEncoder:
    def __init__(self) -> None:
        self._encoder = VectorEncoder()

    def encode(
        self,
        fusion: FusionResult,
        recalled_memory: list[MemoryRecord],
        query: str = "",
    ) -> str:
        hint = max(fusion.query_vector, key=fusion.query_vector.get)
        leader = fusion.ranked_agents[0]
        top_confidence = fusion.agent_scores[leader]
        if top_confidence < 0.4:
            confidence_band = "low"
        elif top_confidence < 0.7:
            confidence_band = "medium"
        else:
            confidence_band = "high"

        if fusion.top_score_margin < 0.1:
            margin_band = "contested"
        elif fusion.top_score_margin < 0.25:
            margin_band = "narrow"
        else:
            margin_band = "clear"

        memory_strength = sum(record.weight for record in recalled_memory)
        if memory_strength <= 0.0:
            memory_band = "none"
        elif memory_strength < 1.0:
            memory_band = "light"
        else:
            memory_band = "strong"

        # Compute vector similarity between query and recalled memory.
        # This lets the Q-table distinguish "high vector match" states
        # (feedback/live knowledge are close in coordinate space) from
        # "low vector match" states (keyword overlap but vector mismatch).
        vec_sim_band = "low"
        records_with_vectors = [r for r in recalled_memory if r.vector]
        if query and records_with_vectors:
            query_vector = self._encoder.encode(query)
            best_sim = max(
                VectorEncoder.cosine_similarity(query_vector, r.vector)
                for r in records_with_vectors
            )
            if best_sim >= 0.4:
                vec_sim_band = "high"

        task_diff = classify_task_difficulty(query) if query else "easy"
        explicit_shares = parse_memory_sharing_preferences(query) if query else ()
        share_band = (
            ",".join(f"{source}->{target}" for source, target in explicit_shares)
            if explicit_shares
            else "none"
        )

        recall_band = str(min(len(recalled_memory), 3))
        return (
            f"hint={hint}|leader={leader}|confidence={confidence_band}|"
            f"margin={margin_band}|memory={memory_band}|recall={recall_band}|vec_sim={vec_sim_band}|diff={task_diff}|share={share_band}"
        )
