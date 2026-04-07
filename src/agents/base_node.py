"""
Base Agent Node
All specialized nodes (Food, Business, Coding, Research, Memory) inherit from this.
Each node is responsible for processing a query from its domain perspective
and returning a structured AgentOutput with a confidence score.
"""
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class AgentOutput:
    agent_name: str       # e.g. "CodingNode"
    domain: str           # e.g. "coding"
    response: str         # The agent's generated answer
    confidence: float     # 0.0 - 1.0 self-assessed confidence
    reasoning: str        # Why this agent produced this output
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

class BaseAgentNode:
    """
    Abstract base for all specialized Hive agent nodes.
    Each subclass must implement `process(query, context, groq_client)`.
    """
    name: str = "BaseNode"
    domain: str = "general"
    # Keywords that signal this domain is relevant
    domain_keywords: list = []

    def relevance_score(self, query: str) -> float:
        """
        Compute 0-1 relevance of this node to the given query.
        Based on keyword overlap. Router Planner uses this.
        """
        query_lower = query.lower()
        if not self.domain_keywords:
            return 0.1
        hits = sum(1 for kw in self.domain_keywords if kw in query_lower)
        return min(1.0, hits / max(1, len(self.domain_keywords) * 0.3))

    def process(self, query: str, context: str, groq_client, history: list = []) -> AgentOutput:
        raise NotImplementedError

    def _call_llm(self, system_prompt: str, user_message: str,
                  groq_client, history: list, model: str = "llama-3.1-8b-instant") -> str:
        """Shared LLM call wrapper used by all agent nodes."""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        t0 = time.time()
        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=512,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"[{self.name} ERROR]: {str(e)}"
