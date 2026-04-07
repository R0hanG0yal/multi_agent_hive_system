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
        search_results: str = ""
    ) -> dict:
        """
        Full multi-agent pipeline. Returns a dict suitable for the API response.
        """
        # ── 1. Retrieve relevant memories (Shared Knowledge Space seeding) ──
        memories = self.memory.retrieve(query, top_k=3)
        memory_context = "\n".join([f"- {m.content}" for m in memories])

        # ── 2. Router Planner ────────────────────────────────────────────────
        has_search = bool(search_results.strip())
        specialist_nodes, memory_node = route(query, has_search_data=has_search)

        # Determine primary routing state for RL
        routing_state = specialist_nodes[0].domain if specialist_nodes else "general"
        routing_debug = describe_routing(query)

        # ── 3. Q-Learning weights for this state ─────────────────────────────
        q_weights = self.rl_controller.get_weights(routing_state)

        # ── 4. Run Memory Agent (always, fast) ───────────────────────────────
        memory_output: AgentOutput = memory_node.process(
            query, memory_context, self.groq_client, chat_history
        )

        # ── 5. Run Specialist agents in parallel (ThreadPoolExecutor) ────────
        knowledge_space = SharedKnowledgeSpace()
        knowledge_space.write(memory_output)

        def run_node(node):
            if hasattr(node, 'process'):
                # ResearchNode gets search_results injected
                if node.name == "ResearchNode" and has_search:
                    return node.process(query, memory_context, self.groq_client,
                                        chat_history, search_results=search_results)
                return node.process(query, memory_context, self.groq_client, chat_history)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(specialist_nodes)) as executor:
            futures = {executor.submit(run_node, node): node for node in specialist_nodes}
            for future in concurrent.futures.as_completed(futures):
                try:
                    output = future.result(timeout=15)
                    if output:
                        knowledge_space.write(output)
                except Exception as e:
                    print(f"[Orchestrator] Node failed: {e}")

        all_outputs: List[AgentOutput] = knowledge_space.read_all()

        # ── 6. Conflict Resolution ────────────────────────────────────────────
        conflict_report = self.conflict_resolver.resolve(
            [o for o in all_outputs if o.agent_name != "MemoryAgentNode"]
        )

        # ── 7. Decision Engine ────────────────────────────────────────────────
        decision: DecisionOutput = self.decision_engine.decide(
            outputs=all_outputs,
            conflict_report=conflict_report,
            memory_context=memory_output.response,
            q_weights=q_weights,
            groq_client=self.groq_client
        )

        # ── 8. Save this interaction to long-term memory ──────────────────────
        self.memory.add_memory(f"User asked: {query}")
        self.memory.add_memory(f"Hive responded via {decision.primary_agent}: {decision.chosen_response[:120]}...")

        # ── 9. Track state for upcoming RL feedback update ────────────────────
        self._last_state = routing_state
        self._last_primary_agent = decision.primary_agent
        # Estimate token usage (Groq sometimes returns usage, fallback to char-based estimate)
        approx_tokens = int((len(query) + len(decision.chosen_response)) / 4)
        self._last_total_tokens = approx_tokens

        return {
            "response": decision.chosen_response,
            "hive_meta": {
                "primary_agent": decision.primary_agent,
                "confidence": round(decision.confidence, 3),
                "routing": routing_debug,
                "active_nodes": [n.name for n in specialist_nodes],
                "has_conflict": conflict_report.has_conflict,
                "conflict_agents": conflict_report.conflicting_agents,
                "agent_scores": decision.all_agent_scores,
                "rl_epsilon": round(self.rl_controller.epsilon, 4),
                "q_weights": {k: round(v, 3) for k, v in q_weights.items()},
                "approx_tokens": approx_tokens,
                "task_difficulty": self.rl_controller.get_reward_history(1)[0]["task_difficulty"]
                    if self.rl_controller.interaction_count > 0 else "easy"
            }
        }

    # ─── RL Feedback Loop ─────────────────────────────────────────────────────

    def apply_feedback(self, feedback: str) -> dict:
        """
        Called when user clicks 👍 or 👎.
        Reward Node → Bellman Update → Q-Table → Memory Update
        feedback: 'positive' | 'negative'
        """
        reward = self.rl_controller.compute_reward(feedback)

        # Next state = same (single-turn feedback) — can be improved with episode tracking
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
