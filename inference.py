from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hackathon_ai_env import HackathonAIEnvironment, default_scenarios
from hackathon_ai_env.live_knowledge import InternetKnowledgeRetriever


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def run_remote(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    queries = [
        {
            "query": "Suggest a startup idea under 1 lakh budget.",
            "rating": 4,
            "notes": "Budget-aware and practical answer.",
        },
        {
            "query": "What is the recipe of paneer?",
            "rating": 5,
            "notes": "Use onions, tomatoes, ginger-garlic paste, and finish with kasuri methi.",
        },
        {
            "query": "Write a python script to parse a CSV.",
            "rating": 5,
            "notes": "Provide clean python code.",
        },
        {
            "query": "What are the common risks in a seed-stage startup?",
            "rating": 4,
            "notes": "Focus on market and team risks.",
        },
    ]

    # print("[START] Remote OpenEnv inference")
    reset_result = _request_json("POST", f"{base}/reset", {"episodes": 1})
    # print(json.dumps({"reset": reset_result["state"]}))

    steps: list[dict[str, Any]] = []
    for index, item in enumerate(queries, start=1):
        result = _request_json("POST", f"{base}/step", item)
        steps.append(result)
        score = round(float(max(0.01, min(0.99, float(result["reward"])))), 2)
        reward_val = round(float(result["reward"]), 2)
        task_name = item["query"]
        print(f"[START] task={task_name}", flush=True)
        print(f"[STEP] step=1 reward={reward_val}", flush=True)
        print(f"[END] task={task_name} score={score} steps=1", flush=True)

    summary = {
        "mode": "remote",
        "base_url": base,
        "steps": len(steps),
        "total_reward": round(sum(step["reward"] for step in steps), 4),
        "final_state": steps[-1]["state"] if steps else reset_result["state"],
    }
    # print("[END] Remote OpenEnv inference")
    return summary


def run_local() -> dict[str, Any]:
    env = HackathonAIEnvironment(seed=7, knowledge_retriever=InternetKnowledgeRetriever())
    scenarios = default_scenarios()
    env.train(scenarios, episodes=1)

    queries = [
        {
            "query": "Suggest a startup idea under 1 lakh budget.",
            "rating": 4,
            "notes": "Budget-aware and practical answer.",
        },
        {
            "query": "What is the recipe of paneer?",
            "rating": 5,
            "notes": "Use onions, tomatoes, ginger-garlic paste, and finish with kasuri methi.",
        },
        {
            "query": "Write a python script to parse a CSV.",
            "rating": 5,
            "notes": "Provide clean python code.",
        },
        {
            "query": "What are the common risks in a seed-stage startup?",
            "rating": 4,
            "notes": "Focus on market and team risks.",
        },
    ]

    # print("[START] Local inference")
    results: list[dict[str, Any]] = []
    for index, item in enumerate(queries, start=1):
        inference = env.answer_query(item["query"])
        reward = env.apply_feedback(inference, item["rating"], item["notes"])
        score = round(float(max(0.01, min(0.99, float(reward.total)))), 2)
        reward_val = round(float(reward.total), 2)
        task_name = item["query"]
        print(f"[START] task={task_name}", flush=True)
        print(f"[STEP] step=1 reward={reward_val}", flush=True)
        print(f"[END] task={task_name} score={score} steps=1", flush=True)
        step = {
            "step": index,
            "task": task_name,
            "grader": "OpenEnv Grader",
            "score": score,
            "query": task_name,
            "agent": inference.final_agent,
            "reward": reward.total,
            "state_key": inference.state_key,
        }
        results.append(step)

    summary = {
        "mode": "local",
        "steps": len(results),
        "total_reward": round(sum(item["reward"] for item in results), 4),
        "last_state_key": results[-1]["state_key"] if results else None,
    }
    # print("[END] Local inference")
    return summary


def main() -> None:
    base_url = os.environ.get("OPENENV_BASE_URL", "").strip()
    try:
        summary = run_remote(base_url) if base_url else run_local()
    except (HTTPError, URLError, TimeoutError) as exc:
        print(json.dumps({"remote_error": str(exc), "fallback": "local"}, indent=2))
        summary = run_local()
    # print(json.dumps(summary))


if __name__ == "__main__":
    main()
