"""
Q-Learning Controller
---------------------
Flowchart: Q-Learning Controller [Action → Env → Reward → Q-table update]

Implements:
  • Bellman Equation: Q(s,a) = Immediate Reward + γ(Max Future Reward at Next Step)
  • TD Error: Target - Old Estimate → Q function update
  • Q-Table: Agent confidence weights that improve over time
  • Epsilon-Greedy (ε): Exploration vs Exploitation toggle
  • Reward Node: Calculates reward signal from user feedback
  • Memory Update: Stores Q-values + reward history persistently via SQLite
  • Task Classifier: Easy / Medium / Hard based on token usage
"""
import json
import random
import sqlite3
import datetime
from typing import Dict, List

GAMMA = 0.9
ALPHA = 0.1
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

AGENTS = ["FoodNode", "BusinessNode", "CodingNode", "ResearchNode"]
STATES = ["food", "business", "coding", "research", "general"]

# Task difficulty thresholds (by total tokens used: prompt + completion)
DIFFICULTY_EASY   = 300   # ≤ 300 tokens  → Easy
DIFFICULTY_MEDIUM = 700   # ≤ 700 tokens  → Medium
# > 700 tokens            → Hard


def classify_task(total_tokens: int) -> str:
    """
    Classifies a task as Easy / Medium / Hard based on token consumption.
    Token usage is a proxy for query and response complexity.
    """
    if total_tokens <= DIFFICULTY_EASY:
        return "easy"
    elif total_tokens <= DIFFICULTY_MEDIUM:
        return "medium"
    else:
        return "hard"


class QLearningController:
    """
    Tabular Q-Learning controller as specified in the flowchart.
    Q-Table: dict[state][agent_name] → Q-value (0.5 default)
    """

    def __init__(self, db_path: str = "hive_data.db"):
        self.db_path = db_path
        self.epsilon = EPSILON_START
        self.q_table: Dict[str, Dict[str, float]] = self._load_or_init_qtable()
        self.interaction_count = self._load_interaction_count()

    # ─── DB Helpers ───────────────────────────────────────────────────────────

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _load_or_init_qtable(self) -> Dict[str, Dict[str, float]]:
        conn = self._get_conn()
        conn.execute('''CREATE TABLE IF NOT EXISTS q_table (
            state TEXT, agent TEXT, q_value REAL, PRIMARY KEY (state, agent))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS rl_meta (
            key TEXT PRIMARY KEY, value TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS reward_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            interaction INTEGER,
            state TEXT,
            agent TEXT,
            reward REAL,
            td_error REAL,
            old_q REAL,
            new_q REAL,
            epsilon REAL,
            task_difficulty TEXT,
            total_tokens INTEGER
        )''')
        conn.commit()

        cursor = conn.execute("SELECT state, agent, q_value FROM q_table")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            table = {state: {agent: 0.5 for agent in AGENTS} for state in STATES}
            self._save_qtable(table)
            return table

        table = {}
        for state, agent, q_value in rows:
            if state not in table:
                table[state] = {}
            table[state][agent] = q_value
        return table

    def _save_qtable(self, table):
        conn = self._get_conn()
        for state, actions in table.items():
            for agent, q_val in actions.items():
                conn.execute(
                    "INSERT OR REPLACE INTO q_table (state, agent, q_value) VALUES (?,?,?)",
                    (state, agent, q_val))
        conn.commit(); conn.close()

    def _load_interaction_count(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("SELECT value FROM rl_meta WHERE key='interaction_count'")
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row else 0

    def _save_meta(self):
        conn = self._get_conn()
        conn.execute("INSERT OR REPLACE INTO rl_meta (key, value) VALUES ('interaction_count', ?)",
                     (str(self.interaction_count),))
        conn.execute("INSERT OR REPLACE INTO rl_meta (key, value) VALUES ('epsilon', ?)",
                     (str(self.epsilon),))
        conn.commit(); conn.close()

    def _save_reward_event(self, state, agent, reward, td_error, old_q, new_q, difficulty, tokens):
        conn = self._get_conn()
        conn.execute('''INSERT INTO reward_history
            (timestamp, interaction, state, agent, reward, td_error, old_q, new_q, epsilon, task_difficulty, total_tokens)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (
            datetime.datetime.utcnow().isoformat(),
            self.interaction_count,
            state, agent, reward, td_error, old_q, new_q,
            round(self.epsilon, 4), difficulty, tokens
        ))
        conn.commit(); conn.close()

    # ─── Core Q-Learning Logic ─────────────────────────────────────────────────

    def get_weights(self, state: str) -> Dict[str, float]:
        if random.random() < self.epsilon:
            return {agent: 1.0 for agent in AGENTS}
        state_values = self.q_table.get(state, {agent: 0.5 for agent in AGENTS})
        max_q = max(state_values.values()) or 1.0
        min_q = min(state_values.values()) or 0.0
        denom = (max_q - min_q) or 1.0
        return {agent: 0.5 + ((q - min_q) / denom) for agent, q in state_values.items()}

    def compute_reward(self, feedback: str) -> float:
        """Reward Node: 👍=+1.0  👎=-0.5  none=0.0"""
        if feedback == "positive":   return 1.0
        elif feedback == "negative": return -0.5
        return 0.0

    def update(self, state: str, agent_name: str, reward: float, next_state: str,
               total_tokens: int = 0):
        """
        Bellman Q-update + persist reward event with task difficulty.
        Q(s,a) ← Q(s,a) + α[R + γ·maxQ(s',a') − Q(s,a)]
        """
        if state not in self.q_table:
            self.q_table[state] = {agent: 0.5 for agent in AGENTS}
        if agent_name not in self.q_table[state]:
            self.q_table[state][agent_name] = 0.5

        current_q = self.q_table[state][agent_name]
        next_vals  = self.q_table.get(next_state, {a: 0.5 for a in AGENTS})
        max_future = max(next_vals.values()) if next_vals else 0.5

        td_target = reward + GAMMA * max_future
        td_error  = td_target - current_q
        new_q     = max(0.0, min(1.0, current_q + ALPHA * td_error))

        self.q_table[state][agent_name] = new_q
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        self.interaction_count += 1

        difficulty = classify_task(total_tokens)

        self._save_qtable(self.q_table)
        self._save_meta()
        self._save_reward_event(state, agent_name, reward, td_error,
                                current_q, new_q, difficulty, total_tokens)

        return {
            "td_error": round(td_error, 4),
            "old_q":    round(current_q, 4),
            "new_q":    round(new_q, 4),
            "epsilon":  round(self.epsilon, 4),
            "difficulty": difficulty,
            "total_tokens": total_tokens
        }

    def get_reward_history(self, limit: int = 50) -> List[dict]:
        """Returns last N reward events for charting."""
        conn = self._get_conn()
        cursor = conn.execute('''
            SELECT interaction, reward, td_error, new_q, epsilon,
                   task_difficulty, total_tokens, agent, state, timestamp
            FROM reward_history
            ORDER BY id DESC LIMIT ?''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        cols = ["interaction","reward","td_error","new_q","epsilon",
                "task_difficulty","total_tokens","agent","state","timestamp"]
        return [dict(zip(cols, r)) for r in reversed(rows)]

    def get_stats(self) -> dict:
        return {
            "epsilon": round(self.epsilon, 4),
            "interaction_count": self.interaction_count,
            "q_table": {
                state: {k: round(v, 3) for k, v in agents.items()}
                for state, agents in self.q_table.items()
            },
            "reward_history": self.get_reward_history(50)
        }
