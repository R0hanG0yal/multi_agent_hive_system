"""
Q-Learning Controller
---------------------
Flowchart: Q-Learning Controller [Action → Env → Reward → Q-table update]

Implements:
  • Bellman Equation: Q(s,a) = Immediate Reward + γ(Max Future Reward at Next Step)
  • TD Error: Target - Old Estimate → Q function update
  • Q-Table: Agent confidence weights that improve over time
  • Epsilon-Greedy (ε): Exploration vs Exploitation toggle
    - ε → 1: More exploration (try different agents)
    - ε → 0: More exploitation (always pick best known agent)
  • Reward Node: Calculates reward signal from user feedback
  • Memory Update: Stores Q-values persistently via SQLite

States = domain categories detected by the Router
Actions = agent node selections
Rewards = user feedback (👍 = +1, 👎 = -1, no feedback = 0)
"""
import json
import random
import sqlite3
from typing import Dict

GAMMA = 0.9          # Discount factor — balances immediate vs future reward
ALPHA = 0.1          # Learning rate — how fast Q-values update
EPSILON_START = 0.3  # Initial exploration rate
EPSILON_MIN = 0.05   # Minimum exploration (always do some exploration)
EPSILON_DECAY = 0.995  # Multiply ε by this after each interaction

# Agent names (actions)
AGENTS = ["FoodNode", "BusinessNode", "CodingNode", "ResearchNode"]

# State space = domain categories
STATES = ["food", "business", "coding", "research", "general"]


class QLearningController:
    """
    Tabular Q-Learning controller as specified in the flowchart.

    Q-Table: dict[state][agent_name] → Q-value (0.5 default)
    Higher Q-value = higher weight given to that agent's response for this domain.
    """

    def __init__(self, db_path: str = "hive_data.db"):
        self.db_path = db_path
        self.epsilon = EPSILON_START
        self.q_table: Dict[str, Dict[str, float]] = self._load_or_init_qtable()
        self.interaction_count = self._load_interaction_count()

    # ─── Q-Table Persistence ──────────────────────────────────────────────────

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _load_or_init_qtable(self) -> Dict[str, Dict[str, float]]:
        """Load Q-table from SQLite; initialise with 0.5 if not found."""
        conn = self._get_conn()
        conn.execute('''CREATE TABLE IF NOT EXISTS q_table (
            state TEXT,
            agent TEXT,
            q_value REAL,
            PRIMARY KEY (state, agent)
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS rl_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )''')
        conn.commit()

        cursor = conn.execute("SELECT state, agent, q_value FROM q_table")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            # Initialise Q-table to 0.5 (neutral) for all state-action pairs
            table = {}
            for state in STATES:
                table[state] = {agent: 0.5 for agent in AGENTS}
            self._save_qtable(table)
            return table

        table = {}
        for state, agent, q_value in rows:
            if state not in table:
                table[state] = {}
            table[state][agent] = q_value
        return table

    def _save_qtable(self, table: Dict[str, Dict[str, float]]):
        conn = self._get_conn()
        for state, actions in table.items():
            for agent, q_val in actions.items():
                conn.execute(
                    "INSERT OR REPLACE INTO q_table (state, agent, q_value) VALUES (?,?,?)",
                    (state, agent, q_val)
                )
        conn.commit()
        conn.close()

    def _load_interaction_count(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("SELECT value FROM rl_meta WHERE key='interaction_count'")
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row else 0

    def _save_meta(self):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO rl_meta (key, value) VALUES ('interaction_count', ?)",
            (str(self.interaction_count),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO rl_meta (key, value) VALUES ('epsilon', ?)",
            (str(self.epsilon),)
        )
        conn.commit()
        conn.close()

    # ─── Core Q-Learning Logic ────────────────────────────────────────────────

    def get_weights(self, state: str) -> Dict[str, float]:
        """
        Returns Q-values (weights) for all agents in a given state.
        These weights multiply agent confidence scores in the Decision Engine.
        Implements Epsilon-Greedy: with probability ε, return uniform weights (exploration).
        """
        # Epsilon-Greedy Exploration vs Exploitation
        if random.random() < self.epsilon:
            # Exploration: return uniform weights
            return {agent: 1.0 for agent in AGENTS}

        # Exploitation: return learned Q-values normalised to [0.5, 1.5]
        state_values = self.q_table.get(state, {agent: 0.5 for agent in AGENTS})
        max_q = max(state_values.values()) or 1.0
        min_q = min(state_values.values()) or 0.0
        denom = (max_q - min_q) or 1.0

        weights = {}
        for agent, q_val in state_values.items():
            # Normalise to [0.5, 1.5] range
            weights[agent] = 0.5 + ((q_val - min_q) / denom)
        return weights

    def compute_reward(self, feedback: str) -> float:
        """
        Reward Node — translates user feedback into RL reward signal.
        👍 = +1.0 (correct)     👎 = -0.5 (incorrect, still partial)
        None = 0.0 (no signal)
        """
        if feedback == "positive":
            return 1.0
        elif feedback == "negative":
            return -0.5
        return 0.0

    def update(self, state: str, agent_name: str, reward: float, next_state: str):
        """
        Bellman Equation Q-update:
        Q(s, a) ← Q(s, a) + α [R + γ·max_a'(Q(s', a')) − Q(s, a)]

        TD Error = R + γ·max_a'(Q(s', a')) − Q(s, a)
        Q(s, a) ← Q(s, a) + α · TD_Error
        """
        if state not in self.q_table:
            self.q_table[state] = {agent: 0.5 for agent in AGENTS}
        if agent_name not in self.q_table[state]:
            self.q_table[state][agent_name] = 0.5

        current_q = self.q_table[state][agent_name]

        # Max Q-value for the next state (future reward)
        next_state_values = self.q_table.get(next_state, {agent: 0.5 for agent in AGENTS})
        max_future_q = max(next_state_values.values()) if next_state_values else 0.5

        # Bellman TD Target
        td_target = reward + GAMMA * max_future_q
        # TD Error
        td_error = td_target - current_q
        # Q Update
        new_q = current_q + ALPHA * td_error

        # Clamp Q-value to [0.0, 1.0]
        self.q_table[state][agent_name] = max(0.0, min(1.0, new_q))

        # Epsilon Decay (towards exploitation over time)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.interaction_count += 1

        # Persist to DB
        self._save_qtable(self.q_table)
        self._save_meta()

        return {
            "td_error": round(td_error, 4),
            "old_q": round(current_q, 4),
            "new_q": round(self.q_table[state][agent_name], 4),
            "epsilon": round(self.epsilon, 4)
        }

    def get_stats(self) -> dict:
        """Returns RL stats for monitoring/UI display."""
        return {
            "epsilon": round(self.epsilon, 4),
            "interaction_count": self.interaction_count,
            "q_table": {
                state: {k: round(v, 3) for k, v in agents.items()}
                for state, agents in self.q_table.items()
            }
        }
