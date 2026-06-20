from __future__ import annotations

from copy import deepcopy
import json
import os
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from .environment import HackathonAIEnvironment
from .live_knowledge import InternetKnowledgeRetriever
from .models import EpisodeSummary, FeedbackReward, InferenceResult, StepResult, QueryScenario
from .scenarios import default_scenarios

ASSETS_DIR = Path(__file__).with_name("web_assets")
CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


def _serialize_episode(summary: EpisodeSummary) -> dict[str, float | int]:
    return {
        "episode": summary.episode,
        "total_reward": summary.total_reward,
        "average_reward": summary.average_reward,
        "accuracy": summary.accuracy,
        "memory_items": summary.memory_items,
    }


def _serialize_step(result: StepResult) -> dict[str, object]:
    return {
        "query": result.scenario.query,
        "description": result.scenario.description,
        "expected_domain": result.scenario.expected_domain,
        "requires_memory": result.scenario.requires_memory,
        "state_key": result.state_key,
        "action_mode": result.action_mode,
        "selected_agent": result.action.selected_agent,
        "final_agent": result.final_agent,
        "use_memory": result.action.use_memory,
        "threshold": result.action.confidence_threshold,
        "reward": asdict(result.reward),
        "recalled_memory_keys": list(result.recalled_memory_keys),
        "agent_scores": result.agent_scores,
        "answer": result.answer,
        "explanation": result.explanation,
        "trace": result.trace,
    }


def _serialize_inference(result: InferenceResult) -> dict[str, object]:
    return {
        "query": result.query,
        "state_key": result.state_key,
        "selected_agent": result.action.selected_agent,
        "final_agent": result.final_agent,
        "use_memory": result.action.use_memory,
        "threshold": result.action.confidence_threshold,
        "recalled_memory_keys": list(result.recalled_memory_keys),
        "agent_scores": result.agent_scores,
        "answer": result.answer,
        "inferred_keywords": list(result.inferred_keywords),
        "explanation": result.explanation,
        "trace": result.trace,
    }


def _serialize_feedback_reward(reward: FeedbackReward) -> dict[str, float]:
    return {
        "total": reward.total,
        "user_feedback": reward.user_feedback,
        "memory_signal": reward.memory_signal,
        "confidence_alignment": reward.confidence_alignment,
        "difficulty_bonus": reward.difficulty_bonus,
        "task_difficulty": reward.task_difficulty,
    }


def _serialize_memory_record(record) -> dict[str, object]:
    return {
        "key": record.key,
        "domain": record.domain,
        "summary": record.summary,
        "keywords": list(record.keywords),
        "weight": round(record.weight, 4),
        "accesses": record.accesses,
        "last_step": record.last_step,
        "updated_at": round(record.updated_at, 3),
    }


def _serialize_q_state(state_key: str, actions: dict[str, float]) -> dict[str, object]:
    ranked_actions = sorted(actions.items(), key=lambda item: (-item[1], item[0]))
    return {
        "state_key": state_key,
        "best_value": round(ranked_actions[0][1], 4) if ranked_actions else 0.0,
        "actions": [
            {"action_key": action_key, "value": round(value, 4)}
            for action_key, value in ranked_actions[:4]
        ],
    }


def _serialize_training_timeline(
    summaries: list[EpisodeSummary],
    limit: int = 12,
) -> list[dict[str, float | int]]:
    return [
        _serialize_episode(summary)
        for summary in summaries[-limit:]
    ]


@dataclass
class DashboardState:
    default_episodes: int = 8
    seed: int = 7
    scenarios: list = field(default_factory=default_scenarios)
    storage_path: Path | None = None

    def __post_init__(self) -> None:
        self.lock = Lock()
        self.storage_path = self._resolve_storage_path(self.storage_path)
        self._persistent_user_records = {}
        self._persistent_sharing_rules = {}
        self._reset_unlocked(clear_user_context=True)
        self._load_persisted_state_unlocked()

    def _resolve_storage_path(self, raw_path: Path | str | None) -> Path | None:
        if raw_path is not None:
            return Path(raw_path)
        configured = os.environ.get("APP_STATE_PATH", "").strip()
        if configured:
            return Path(configured)
        data_dir = Path("/data")
        if data_dir.exists() and data_dir.is_dir():
            return data_dir / "hackathon_ai_env_state.json"
        return None

    def _user_memory_prefixes(self) -> tuple[str, ...]:
        return ("preference:", "health:", "profile:", "kitchen:", "ingredient:", "feedback:")

    def _capture_user_context_unlocked(self) -> None:
        prefixes = self._user_memory_prefixes()
        self._persistent_user_records = {
            key: deepcopy(record)
            for key, record in self.env.memory.records.items()
            if key.startswith(prefixes) or record.is_user_provided
        }
        self._persistent_sharing_rules = {
            source: set(targets)
            for source, targets in self.env.memory.sharing_rules.items()
        }

    def _restore_user_context_unlocked(self) -> None:
        for key, record in deepcopy(self._persistent_user_records).items():
            self.env.memory.records[key] = record
        self.env.memory.sharing_rules = {
            source: set(targets)
            for source, targets in deepcopy(self._persistent_sharing_rules).items()
        }

    def _persistence_payload_unlocked(self) -> dict[str, object]:
        return {
            "version": 1,
            "trained_episodes": self.trained_episodes,
            "memory_primed": self.memory_primed,
            "training_summaries": [asdict(summary) for summary in self.training_summaries],
            "last_feedback": self.last_feedback,
            "q_learning": self.env.q_controller.to_dict(),
            "memory": self.env.memory.to_dict(),
            "audit_log": self.env.audit_log[-120:],
            "scenarios": [asdict(s) for s in self.scenarios],
        }

    def _write_persisted_state_unlocked(self) -> None:
        if self.storage_path is None:
            return
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps(self._persistence_payload_unlocked(), indent=2, sort_keys=True)
            )
        except (OSError, TypeError, ValueError):
            return

    def _load_persisted_state_unlocked(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return

        raw_scenarios = payload.get("scenarios")
        if isinstance(raw_scenarios, list):
            self.scenarios = []
            for s in raw_scenarios:
                if isinstance(s, dict):
                    self.scenarios.append(QueryScenario(
                        query=s.get("query", ""),
                        expected_domain=s.get("expected_domain", ""),
                        expected_keywords=tuple(s.get("expected_keywords", [])),
                        requires_memory=bool(s.get("requires_memory", False)),
                        description=s.get("description", "")
                    ))

        raw_q_learning = payload.get("q_learning", {})
        if isinstance(raw_q_learning, dict):
            self.env.q_controller.load_dict(raw_q_learning)

        raw_memory = payload.get("memory", {})
        if isinstance(raw_memory, dict):
            self.env.memory.load_dict(raw_memory)

        raw_training = payload.get("training_summaries", [])
        if isinstance(raw_training, list):
            self.training_summaries = [
                EpisodeSummary(**item)
                for item in raw_training
                if isinstance(item, dict)
            ]

        raw_feedback = payload.get("last_feedback")
        self.last_feedback = raw_feedback if isinstance(raw_feedback, dict) else None
        self.trained_episodes = int(payload.get("trained_episodes", 0))
        self.memory_primed = bool(payload.get("memory_primed", bool(self.env.memory.records)))

        raw_audit_log = payload.get("audit_log", [])
        self.env.audit_log = raw_audit_log[-120:] if isinstance(raw_audit_log, list) else []
        self.feedback_pending = False
        self.last_inference = None
        self.last_evaluation = []
        self._capture_user_context_unlocked()

    def _clear_persisted_state_unlocked(self) -> None:
        if self.storage_path is None:
            return
        try:
            if self.storage_path.exists():
                self.storage_path.unlink()
        except OSError:
            return

    def _reset_unlocked(self, clear_user_context: bool = False) -> None:
        if clear_user_context:
            self._persistent_user_records = {}
            self._persistent_sharing_rules = {}
        self.env = HackathonAIEnvironment(
            seed=self.seed,
            knowledge_retriever=InternetKnowledgeRetriever(),
        )
        if not clear_user_context:
            self._restore_user_context_unlocked()
        self.training_summaries: list[EpisodeSummary] = []
        self.last_evaluation: list[StepResult] = []
        self.last_inference: InferenceResult | None = None
        self.last_feedback: dict[str, object] | None = None
        self.feedback_pending = False
        self.trained_episodes = 0
        self.memory_primed = False
        self.openenv_episode_id = getattr(self, "openenv_episode_id", 0)
        self.openenv_step_count = 0
        self.openenv_done = False
        self.openenv_last_reward = 0.0

    def _status_unlocked(self) -> dict[str, object]:
        return {
            "trained": self.trained_episodes > 0,
            "trained_episodes": self.trained_episodes,
            "memory_primed": self.memory_primed,
            "scenario_count": len(self.scenarios),
            "q_state_count": len(self.env.q_controller.q_table),
            "epsilon": round(self.env.q_controller.epsilon, 4),
            "memory_snapshot": self.env.memory.snapshot(),
            "policy": self._policy_snapshot_unlocked(),
            "feedback_pending": self.feedback_pending,
            "last_feedback": self.last_feedback,
            "persistence": {
                "enabled": self.storage_path is not None,
                "path": str(self.storage_path) if self.storage_path is not None else None,
            },
            "audit_log": self.env.audit_log[-15:],
            "last_inference": (
                _serialize_inference(self.last_inference)
                if self.last_inference is not None
                else None
            ),
            "last_train": (
                _serialize_episode(self.training_summaries[-1])
                if self.training_summaries
                else None
            ),
            "training_timeline": _serialize_training_timeline(self.training_summaries),
        }

    def _openenv_state_snapshot_unlocked(self) -> dict[str, object]:
        return {
            "episode_id": self.openenv_episode_id,
            "step_count": self.openenv_step_count,
            "done": self.openenv_done,
            "trained": self.trained_episodes > 0,
            "trained_episodes": self.trained_episodes,
            "feedback_pending": self.feedback_pending,
            "q_state_count": len(self.env.q_controller.q_table),
            "memory_count": len(self.env.memory.records),
            "last_reward": round(self.openenv_last_reward, 4),
            "last_query": self.last_inference.query if self.last_inference is not None else None,
            "last_agent": self.last_inference.final_agent if self.last_inference is not None else None,
            "last_state_key": self.last_inference.state_key if self.last_inference is not None else None,
        }

    def openenv_metadata(self) -> dict[str, object]:
        with self.lock:
            return {
                "name": "memory_augmented_ai_environment",
                "version": "0.1.0",
                "description": (
                    "Multi-agent reinforcement-learning environment with routing, "
                    "memory recall, feedback-driven rewards, and a browser dashboard."
                ),
                "routes": {
                    "reset": ["/reset", "/openenv/reset"],
                    "step": ["/step", "/openenv/step"],
                    "state": ["/state", "/openenv/state"],
                    "health": ["/health", "/openenv/health"],
                    "metadata": ["/metadata", "/openenv/metadata"],
                },
                "action_schema": {
                    "query": "string",
                    "warm_memory": "boolean, optional",
                    "rating": "integer 1-5, optional",
                    "notes": "string, optional",
                    "episodes": "positive integer, optional",
                },
                "observation_schema": {
                    "answer": "string",
                    "final_agent": "string",
                    "selected_agent": "string",
                    "state_key": "string",
                    "explanation": "string",
                    "recalled_memory_keys": "string[]",
                },
            }

    def openenv_health(self) -> dict[str, object]:
        with self.lock:
            return {
                "status": "ok",
                "service": "memory_augmented_ai_environment",
                "trained": self.trained_episodes > 0,
                "persistence_enabled": self.storage_path is not None,
            }

    def openenv_state(self) -> dict[str, object]:
        with self.lock:
            snapshot = self._openenv_state_snapshot_unlocked()
            return {
                "state": snapshot,
                "observation": snapshot,
                "info": {
                    "feedback_pending": self.feedback_pending,
                    "last_feedback": self.last_feedback,
                },
            }

    def openenv_reset(self, episodes: int | None = None) -> dict[str, object]:
        with self.lock:
            # Only train if already trained and episode count changed; never block reset
            # on a cold-start training run (which would time out the checker).
            if self.trained_episodes > 0:
                target = episodes or self.default_episodes
                if self.trained_episodes != target:
                    self._reset_unlocked(clear_user_context=False)
                    self.training_summaries = self.env.train(self.scenarios, episodes=target)
                    self._restore_user_context_unlocked()
                    self.trained_episodes = target
                    self.memory_primed = False
            self._capture_user_context_unlocked()
            self.env.reset_episode()
            self._restore_user_context_unlocked()
            self.last_inference = None
            self.last_feedback = None
            self.feedback_pending = False
            self.last_evaluation = []
            self.memory_primed = False
            self.openenv_episode_id += 1
            self.openenv_step_count = 0
            self.openenv_done = False
            self.openenv_last_reward = 0.0
            self._write_persisted_state_unlocked()
            state = self._openenv_state_snapshot_unlocked()
            return {
                "observation": {
                    "message": "Environment reset and ready for a new episode.",
                    "episode_id": self.openenv_episode_id,
                },
                "reward": 0.0,
                "done": False,
                "info": {
                    "trained_episodes": self.trained_episodes,
                    "memory_preserved": bool(self._persistent_user_records),
                },
                "state": state,
            }

    def openenv_step(self, payload: dict[str, object]) -> dict[str, object]:
        raw_action = payload.get("action")
        action = raw_action if isinstance(raw_action, dict) else payload
        query = _first_text(
            action,
            ("query", "prompt", "message", "input", "task"),
        )
        if not query:
            raise ValueError("step action must include query, prompt, message, input, or task")

        warm_memory = _coerce_bool(_first_value(action, ("warm_memory", "warmMemory", "use_memory"), True))
        raw_rating = _first_value(action, ("rating", "score", "feedback_rating"), None)
        notes = _first_text(action, ("notes", "feedback", "comment"), "")
        max_steps = _coerce_positive_int(_first_value(action, ("max_steps",), 4), 4)
        episodes = _coerce_positive_int(_first_value(action, ("episodes",), self.default_episodes), self.default_episodes)

        with self.lock:
            self._ensure_trained_unlocked(episodes)
            if self.openenv_episode_id == 0:
                self.openenv_episode_id = 1
            if warm_memory and not self.memory_primed:
                self._capture_user_context_unlocked()
                self.env.prime_memory(self.scenarios)
                self._restore_user_context_unlocked()
                self.memory_primed = True

            self.last_inference = self.env.answer_query(query)
            self._capture_user_context_unlocked()
            self.feedback_pending = True
            self.last_feedback = None

            reward_total = 0.0
            reward_payload = None
            feedback_applied = False
            if raw_rating not in (None, ""):
                rating = _coerce_rating(raw_rating)
                reward = self.env.apply_feedback(self.last_inference, rating, notes=notes)
                self.feedback_pending = False
                reward_total = reward.total
                reward_payload = _serialize_feedback_reward(reward)
                self.last_feedback = {
                    "query": self.last_inference.query,
                    "rating": rating,
                    "notes": notes.strip(),
                    "reward": reward_payload,
                }
                feedback_applied = True

            self.openenv_step_count += 1
            self.openenv_last_reward = reward_total
            self.openenv_done = self.openenv_step_count >= max_steps or feedback_applied
            self._write_persisted_state_unlocked()

            observation = {
                "query": self.last_inference.query,
                "answer": self.last_inference.answer,
                "selected_agent": self.last_inference.action.selected_agent,
                "final_agent": self.last_inference.final_agent,
                "state_key": self.last_inference.state_key,
                "explanation": self.last_inference.explanation,
                "recalled_memory_keys": list(self.last_inference.recalled_memory_keys),
                "inferred_keywords": list(self.last_inference.inferred_keywords),
            }
            return {
                "observation": observation,
                "reward": round(reward_total, 4),
                "done": self.openenv_done,
                "info": {
                    "feedback_applied": feedback_applied,
                    "feedback": reward_payload,
                    "trace": self.last_inference.trace,
                },
                "state": self._openenv_state_snapshot_unlocked(),
            }

    def _policy_snapshot_unlocked(self) -> dict[str, object]:
        q_table = [
            _serialize_q_state(state_key, actions)
            for state_key, actions in self.env.q_controller.q_table.items()
        ]
        q_table.sort(key=lambda entry: (-entry["best_value"], entry["state_key"]))

        memory_bank = [
            _serialize_memory_record(record)
            for record in self.env.memory.records.values()
        ]
        memory_bank.sort(key=lambda entry: (-entry["weight"], entry["key"]))

        return {
            "q_table": q_table[:10],
            "memory_bank": memory_bank[:10],
            "sharing_rules": {
                source: sorted(targets)
                for source, targets in sorted(self.env.memory.sharing_rules.items())
            },
        }

    def reset(self) -> dict[str, object]:
        with self.lock:
            self._reset_unlocked(clear_user_context=True)
            self._clear_persisted_state_unlocked()
            return {
                "message": "Environment reset. Train again to rebuild the policy.",
                "status": self._status_unlocked(),
            }

    def _ensure_trained_unlocked(self, episodes: int | None) -> int:
        target_episodes = episodes or self.default_episodes
        if self.trained_episodes != target_episodes:
            self._reset_unlocked(clear_user_context=False)
            self.training_summaries = self.env.train(self.scenarios, episodes=target_episodes)
            self._restore_user_context_unlocked()
            self.trained_episodes = target_episodes
            self.memory_primed = False
        return target_episodes

    def status(self) -> dict[str, object]:
        with self.lock:
            return {"status": self._status_unlocked()}

    def train(self, episodes: int | None) -> dict[str, object]:
        with self.lock:
            target_episodes = episodes or self.default_episodes
            self._reset_unlocked(clear_user_context=False)
            self.training_summaries = self.env.train(self.scenarios, episodes=target_episodes)
            self._restore_user_context_unlocked()
            self.trained_episodes = target_episodes
            self.memory_primed = False
            last = self.training_summaries[-1]
            self._write_persisted_state_unlocked()
            return {
                "message": f"Training completed for {target_episodes} episodes.",
                "status": self._status_unlocked(),
                "summary": _serialize_episode(last),
                "timeline": _serialize_training_timeline(self.training_summaries),
            }

    def evaluate(self, episodes: int | None) -> dict[str, object]:
        with self.lock:
            target_episodes = self._ensure_trained_unlocked(episodes)
            self._capture_user_context_unlocked()
            self.last_evaluation = self.env.evaluate(self.scenarios)
            self._restore_user_context_unlocked()
            self.memory_primed = True
            self._write_persisted_state_unlocked()
            accuracy = sum(
                1
                for result in self.last_evaluation
                if result.final_agent == result.scenario.expected_domain
            ) / len(self.last_evaluation)
            average_reward = sum(
                result.reward.total for result in self.last_evaluation
            ) / len(self.last_evaluation)
            return {
                "message": "Evaluation complete across the built-in benchmark scenarios.",
                "status": self._status_unlocked(),
                "summary": {
                    "episodes": target_episodes,
                    "accuracy": round(accuracy, 4),
                    "average_reward": round(average_reward, 4),
                    "results": len(self.last_evaluation),
                },
                "results": [_serialize_step(result) for result in self.last_evaluation],
            }

    def ask(
        self,
        query: str,
        episodes: int | None,
        warm_memory: bool,
    ) -> dict[str, object]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        with self.lock:
            target_episodes = self._ensure_trained_unlocked(episodes)
            if warm_memory and not self.memory_primed:
                self._capture_user_context_unlocked()
                self.env.prime_memory(self.scenarios)
                self._restore_user_context_unlocked()
                self.memory_primed = True
            self.last_inference = self.env.answer_query(query)
            self._capture_user_context_unlocked()
            self.last_feedback = None
            self.feedback_pending = True
            self._write_persisted_state_unlocked()
            return {
                "message": (
                    f"Answered query using the policy trained for {target_episodes} episodes. "
                    "Now rate the answer so the reward can be calculated from your feedback."
                ),
                "status": self._status_unlocked(),
                "result": _serialize_inference(self.last_inference),
            }

    def submit_feedback(
        self,
        rating: int,
        notes: str,
    ) -> dict[str, object]:
        with self.lock:
            if self.last_inference is None or not self.feedback_pending:
                raise ValueError("Ask a query first, then submit feedback for that answer.")
            reward = self.env.apply_feedback(self.last_inference, rating, notes=notes)
            
            new_scenario = QueryScenario(
                query=self.last_inference.query,
                expected_domain=self.last_inference.final_agent,
                expected_keywords=self.last_inference.inferred_keywords,
                requires_memory=bool(self.last_inference.recalled_memory_keys),
                description=f"User feedback note: {notes.strip()}"
            )
            existing_idx = next((i for i, s in enumerate(self.scenarios) if s.query == new_scenario.query), None)
            if existing_idx is not None:
                self.scenarios[existing_idx] = new_scenario
            else:
                self.scenarios.append(new_scenario)
                
            saved_inference = self.last_inference
            feedback_block = {
                "query": saved_inference.query,
                "rating": rating,
                "notes": notes.strip(),
                "reward": _serialize_feedback_reward(reward),
            }
            self._capture_user_context_unlocked()
            
            # Automatically train immediately upon feedback
            self._reset_unlocked(clear_user_context=False)
            self.training_summaries = self.env.train(self.scenarios, episodes=self.default_episodes)
            self._restore_user_context_unlocked()
            self.trained_episodes = self.default_episodes
            self.memory_primed = False

            self.last_inference = saved_inference
            self.last_feedback = feedback_block
            self.feedback_pending = False
            
            self._write_persisted_state_unlocked()
            return {
                "message": "Feedback recorded and environment automatically trained from it.",
                "status": self._status_unlocked(),
                "feedback": self.last_feedback,
            }


class DashboardHandler(BaseHTTPRequestHandler):
    state = DashboardState()

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/status":
            self._send_json(HTTPStatus.OK, self.state.status())
            return
        if route in {"/health", "/openenv/health"}:
            self._send_json(HTTPStatus.OK, self.state.openenv_health())
            return
        if route in {"/metadata", "/openenv/metadata"}:
            self._send_json(HTTPStatus.OK, self.state.openenv_metadata())
            return
        if route in {"/state", "/openenv/state"}:
            self._send_json(HTTPStatus.OK, self.state.openenv_state())
            return
        if route in {"/", "/index.html"}:
            self._send_asset("index.html")
            return
        if route in {"/styles.css", "/app.js"}:
            self._send_asset(route.lstrip("/"))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route in {"/reset", "/openenv/reset"}:
                self._send_json(
                    HTTPStatus.OK,
                    self.state.openenv_reset(
                        _coerce_positive_int(payload.get("episodes"), self.state.default_episodes)
                    ),
                )
                return
            if route in {"/step", "/openenv/step"}:
                self._send_json(HTTPStatus.OK, self.state.openenv_step(payload))
                return
            if route in {"/state", "/openenv/state"}:
                self._send_json(HTTPStatus.OK, self.state.openenv_state())
                return
            if route == "/api/reset":
                self._send_json(HTTPStatus.OK, self.state.reset())
                return
            if route == "/api/train":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.train(_coerce_positive_int(payload.get("episodes"), self.state.default_episodes)),
                )
                return
            if route == "/api/eval":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.evaluate(_coerce_positive_int(payload.get("episodes"), self.state.default_episodes)),
                )
                return
            if route == "/api/ask":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.ask(
                        query=str(payload.get("query", "")),
                        episodes=_coerce_positive_int(payload.get("episodes"), self.state.default_episodes),
                        warm_memory=bool(payload.get("warm_memory", True)),
                    ),
                )
                return
            if route == "/api/feedback":
                self._send_json(
                    HTTPStatus.OK,
                    self.state.submit_feedback(
                        rating=_coerce_rating(payload.get("rating")),
                        notes=str(payload.get("notes", "")),
                    ),
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request body must be valid JSON."})

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        text = raw.decode("utf-8").strip()
        if not text:
            return {}
        return json.loads(text)

    def _send_asset(self, name: str) -> None:
        asset_path = ASSETS_DIR / name
        if not asset_path.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Missing asset: {name}"})
            return
        content = asset_path.read_bytes()
        content_type = CONTENT_TYPES.get(asset_path.suffix, "application/octet-stream")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(content)


def _coerce_positive_int(raw_value: object, fallback: int) -> int:
    if raw_value in (None, ""):
        return fallback
    value = int(raw_value)
    if value <= 0:
        raise ValueError("episodes must be a positive integer")
    return value


def _coerce_rating(raw_value: object) -> int:
    value = int(raw_value)
    if value < 1 or value > 5:
        raise ValueError("rating must be between 1 and 5")
    return value


def _coerce_bool(raw_value: object, default: bool = False) -> bool:
    if raw_value in (None, ""):
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _first_value(payload: dict[str, object], keys: tuple[str, ...], default: object = None) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _first_text(payload: dict[str, object], keys: tuple[str, ...], default: str = "") -> str:
    value = _first_value(payload, keys, default)
    if value is None:
        return default
    return str(value).strip()


def serve_dashboard(host: str = "127.0.0.1", port: int = 8000, default_episodes: int = 40) -> None:
    DashboardHandler.state = DashboardState(default_episodes=default_episodes)
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
    finally:
        server.server_close()
