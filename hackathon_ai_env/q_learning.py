from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .models import AgentAction


class QLearningController:
    def __init__(
        self,
        alpha: float = 0.35,
        gamma: float = 0.85,
        epsilon: float = 0.28,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 0.985,
        seed: int = 7,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.q_table: dict[str, dict[str, float]] = defaultdict(dict)
        self.last_update: dict[str, float | str] | None = None

    def q_value(self, state_key: str, action_key: str) -> float:
        return self.q_table.get(state_key, {}).get(action_key, 0.0)

    def choose_action(
        self,
        state_key: str,
        candidate_actions: tuple[AgentAction, ...],
    ) -> tuple[AgentAction, str]:
        if not candidate_actions:
            raise ValueError("candidate_actions cannot be empty")
        if self.random.random() < self.epsilon:
            return self.random.choice(candidate_actions), "explore"
        return self.best_action(state_key, candidate_actions), "exploit"

    def best_action(
        self,
        state_key: str,
        candidate_actions: tuple[AgentAction, ...],
    ) -> AgentAction:
        indexed_actions = list(enumerate(candidate_actions))
        action_count = max(len(indexed_actions), 1)
        _, action = max(
            indexed_actions,
            key=lambda item: (
                self.q_value(state_key, item[1].key) + (0.02 * ((action_count - item[0]) / action_count)),
                -item[0],
            ),
        )
        return action

    def update(
        self,
        state_key: str,
        action_key: str,
        reward: float,
        next_state_key: str | None,
        next_action_keys: tuple[str, ...] = (),
    ) -> None:
        current = self.q_value(state_key, action_key)
        next_best = 0.0
        if next_state_key and next_action_keys:
            next_best = max(self.q_value(next_state_key, candidate) for candidate in next_action_keys)
        target = reward + (self.gamma * next_best)
        td_error = target - current
        self.q_table.setdefault(state_key, {})
        updated = current + self.alpha * td_error
        self.q_table[state_key][action_key] = updated
        self.last_update = {
            "state_key": state_key,
            "action_key": action_key,
            "reward": round(reward, 4),
            "target": round(target, 4),
            "td_error": round(td_error, 4),
            "updated_q": round(updated, 4),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": round(self.epsilon, 4),
        }

    def decay(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "q_table": {
                state: {action: round(value, 6) for action, value in actions.items()}
                for state, actions in self.q_table.items()
            },
            "last_update": self.last_update,
        }

    def load_dict(self, payload: dict[str, object]) -> None:
        self.alpha = float(payload.get("alpha", self.alpha))
        self.gamma = float(payload.get("gamma", self.gamma))
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        self.min_epsilon = float(payload.get("min_epsilon", self.min_epsilon))
        self.epsilon_decay = float(payload.get("epsilon_decay", self.epsilon_decay))

        restored: defaultdict[str, dict[str, float]] = defaultdict(dict)
        raw_table = payload.get("q_table", {})
        if isinstance(raw_table, dict):
            for state_key, actions in raw_table.items():
                if not isinstance(state_key, str) or not isinstance(actions, dict):
                    continue
                restored[state_key] = {
                    str(action_key): float(value)
                    for action_key, value in actions.items()
                }
        self.q_table = restored

        raw_update = payload.get("last_update")
        self.last_update = raw_update if isinstance(raw_update, dict) else None

    def load(self, source: str | Path) -> Path:
        path = Path(source)
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Q-learning state file must contain a JSON object.")
        self.load_dict(payload)
        return path
