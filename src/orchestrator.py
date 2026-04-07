"""
Hive Orchestrator — The Central Brain
--------------------------------------
This is the main entry point that wires together the complete flowchart pipeline:

  User Query
    ↓
  Vector Encoding (via MemoryManager)
    ↓
  Router Planner → selects active Agent Nodes
    ↓
  [FoodNode | BusinessNode | CodingNode | ResearchNode] run in parallel
                    ↕ (bidirectional read/write)
           Shared Knowledge Space
                    ↓
  Conflict Resolution Node
                    ↓
  Decision Engine (Q-weighted confidence scoring → merge or select best)
                    ↓
  Final Output + User Feedback
       ↕
  Q-Learning Controller (Reward → Bellman Update → Q-Table)
                    ↓
  Reward Node → Memory Update → Next State
"""
import concurrent.futures
from typing import List, Optional

from src.router.planner import route, describe_routing
from src.knowledge.space import SharedKnowledgeSpace, ConflictResolutionNode, DecisionEngine, DecisionOutput
from src.agents.base_node import AgentOutput
from src.rl.controller import QLearningController
from src.memory.manager import MemoryManager


class HiveOrchestrator:
    """
    Singleton orchestrator — initialised once on server startup.
    """
    def __init__(self, groq_client, db_path: str = "hive_data.db"):
        self.groq_client = groq_client
        self.memory = MemoryManager()
        self.conflict_resolver = ConflictResolutionNode()
        self.decision_engine = DecisionEngine()
        self.rl_controller = QLearningController(db_path=db_path)
        # Tracks the last routing state for RL feedback
        self._last_state: str = "general"
        self._last_primary_agent: str = "ResearchNode"
        self._last_total_tokens: int = 0

    # ─── Main Pipeline ────────────────────────────────────────────────────────

    def process(
        self,
        query: str,
        chat_history: list,
        search_results: str = "",
        user_name: str = ""
    ) -> dict:
        """Full multi-agent pipeline (non-streaming). Returns complete response dict."""
        routing_meta = self.prepare_stream(query, chat_history, search_results, user_name)
        try:
            completion = self.groq_client.chat.completions.create(
                messages=routing_meta["messages"],
                model="llama-3.1-8b-instant",
                temperature=0.72,
                max_tokens=700
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            response_text = f"I encountered an issue generating a response: {e}"

        self.finalize_stream(query, response_text, routing_meta)
        return {"response": response_text, "hive_meta": routing_meta["hive_meta"]}

    def prepare_stream(
        self,
        query: str,
        chat_history: list,
        search_results: str = "",
        user_name: str = ""
    ) -> dict:
        """
        Runs all non-LLM pipeline steps and returns the message list
        ready for Groq streaming, plus hive_meta for the response.
        """
        # ── 1. Memory retrieval ───────────────────────────────────────────────
        memories = self.memory.retrieve(query, top_k=3)
        memory_context = "\n".join([f"- {m.content}" for m in memories])

        # ── 2. Router Planner ─────────────────────────────────────────────────
        has_search = bool(search_results.strip())
        specialist_nodes, _ = route(query, has_search_data=has_search)
        routing_state = specialist_nodes[0].domain if specialist_nodes else "general"
        routing_debug = describe_routing(query)

        # ── 3. Q-Learning weights ─────────────────────────────────────────────
        q_weights = self.rl_controller.get_weights(routing_state)

        # ── 4. Pick best agent (highest Q-weighted relevance) ─────────────────
        best_node = specialist_nodes[0]
        best_score = 0.0
        for node in specialist_nodes:
            score = node.relevance_score(query) * q_weights.get(node.name, 1.0)
            if score > best_score:
                best_score = score
                best_node = node

        # ── 5. Build name-aware system prompt ────────────────────────────────
        name_line = f"The user's name is {user_name}. Always address them by name naturally in conversation. " if user_name else ""
        memory_line = f"\nRelevant personal context from memory:\n{memory_context}" if memory_context else ""
        search_line = f"\n\n[LIVE SEARCH DATA — USE THIS]:\n{search_results}" if search_results else ""

        # Select node-specific personality prompt
        node_prompts = {
            "FoodNode":     "You are MAHA, a warm and knowledgeable food & nutrition expert. Speak like a brilliant friend — engaging, practical, and enthusiastic.",
            "BusinessNode": "You are MAHA, a sharp business mentor. Be direct, insightful, and end with an actionable tip. Avoid corporate jargon.",
            "CodingNode":   "You are MAHA, a skilled software engineer and patient mentor. Use markdown code blocks. Empathise, then fix. Offer to explore edge cases.",
            "ResearchNode": "You are MAHA, a curious research expert. Share findings conversationally as if excited. Never say 'As an AI' or 'I cannot'.",
        }
        base_prompt = node_prompts.get(best_node.name, "You are MAHA, a helpful and warm AI assistant.")

        system_prompt = (
            f"{base_prompt} "
            f"{name_line}"
            f"Respond naturally — vary sentence length, use contractions, show warmth. "
            f"Be concise but never terse."
            f"{memory_line}{search_line}"
        )

        # ── 6. Build messages list for Groq ───────────────────────────────────
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-8:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        if search_results:
            messages.append({"role": "user", "content": f"[SEARCH DATA]:\n{search_results}\n\nUSER: {query}"})
        else:
            messages.append({"role": "user", "content": query})

        # ── 7. Track routing state for RL ─────────────────────────────────────
        self._last_state = routing_state
        self._last_primary_agent = best_node.name
        approx_tokens = int(len(query) / 4)
        self._last_total_tokens = approx_tokens

        hive_meta = {
            "primary_agent": best_node.name,
            "confidence": round(best_score, 3),
            "routing": routing_debug,
            "active_nodes": [n.name for n in specialist_nodes],
            "has_conflict": False,
            "conflict_agents": [],
            "agent_scores": {},
            "rl_epsilon": round(self.rl_controller.epsilon, 4),
            "q_weights": {k: round(v, 3) for k, v in q_weights.items()},
        }

        return {"messages": messages, "hive_meta": hive_meta, "routing_state": routing_state}

    def finalize_stream(self, query: str, response_text: str, routing_meta: dict):
        """Save to memory and update token count after streaming completes."""
        self.memory.add_memory(f"User asked: {query}")
        self.memory.add_memory(f"MAHA responded: {response_text[:120]}...")
        approx_tokens = int((len(query) + len(response_text)) / 4)
        self._last_total_tokens = approx_tokens

    # ─── RL Feedback Loop ─────────────────────────────────────────────────────

    def apply_feedback(self, feedback: str) -> dict:
        """
        Called when user clicks 👍 or 👎.
        Reward Node → Bellman Update → Q-Table → Memory Update
        feedback: 'positive' | 'negative'
        """
        reward = self.rl_controller.compute_reward(feedback)
        update_result = self.rl_controller.update(
            state=self._last_state,
            agent_name=self._last_primary_agent,
            reward=reward,
            next_state=self._last_state,
            total_tokens=self._last_total_tokens
        )
        return {
            "status": "Q-table updated",
            "reward": reward,
            "rl_update": update_result,
            "stats": self.rl_controller.get_stats()
        }
