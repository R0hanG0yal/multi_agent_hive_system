from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hackathon_ai_env.models import AgentAction
from hackathon_ai_env.q_learning import QLearningController


class QLearningControllerTest(unittest.TestCase):
    def test_best_action_prefers_highest_q_value(self) -> None:
        controller = QLearningController(epsilon=0.0)
        state_key = "demo-state"
        actions = (
            AgentAction("food", False, 0.45),
            AgentAction("business", True, 0.65),
            AgentAction("research", False, 0.8),
        )
        controller.q_table[state_key] = {
            actions[0].key: 0.2,
            actions[1].key: 0.9,
            actions[2].key: 0.6,
        }

        chosen = controller.best_action(state_key, actions)
        self.assertEqual(chosen.key, actions[1].key)

    def test_update_moves_q_value_toward_target(self) -> None:
        controller = QLearningController(alpha=0.5, gamma=0.0, epsilon=0.0)
        controller.update(
            state_key="s",
            action_key="a",
            reward=1.0,
            next_state_key=None,
            next_action_keys=(),
        )
        self.assertAlmostEqual(controller.q_value("s", "a"), 0.5)

    def test_save_and_load_round_trip_preserves_q_table(self) -> None:
        controller = QLearningController(epsilon=0.12)
        controller.q_table["demo-state"] = {"food|mem|0.45": 0.81}
        controller.last_update = {
            "state_key": "demo-state",
            "action_key": "food|mem|0.45",
            "reward": 0.9,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "q_table.json"
            controller.save(path)

            restored = QLearningController()
            restored.load(path)

        self.assertAlmostEqual(restored.q_value("demo-state", "food|mem|0.45"), 0.81)
        self.assertAlmostEqual(restored.epsilon, 0.12)
        self.assertEqual(restored.last_update["action_key"], "food|mem|0.45")


if __name__ == "__main__":
    unittest.main()
