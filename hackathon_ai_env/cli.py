from __future__ import annotations

import argparse
from pathlib import Path

from .environment import HackathonAIEnvironment
from .live_knowledge import InternetKnowledgeRetriever
from .scenarios import default_scenarios
from .web import serve_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hackathon multi-agent environment with Q-learning control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the environment on sample scenarios.")
    train_parser.add_argument("--episodes", type=int, default=40)
    train_parser.add_argument(
        "--save-q-table",
        type=Path,
        default=Path("artifacts/q_table.json"),
    )

    eval_parser = subparsers.add_parser("eval", help="Train then evaluate on the sample scenarios.")
    eval_parser.add_argument("--episodes", type=int, default=40)

    ask_parser = subparsers.add_parser("ask", help="Train, warm memory, and answer a custom query.")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--episodes", type=int, default=40)

    web_parser = subparsers.add_parser("web", help="Launch a local dashboard in the browser.")
    web_parser.add_argument("--host", type=str, default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)
    web_parser.add_argument("--episodes", type=int, default=40)

    return parser


def command_train(episodes: int, save_q_table: Path) -> None:
    env = HackathonAIEnvironment()
    scenarios = default_scenarios()
    summaries = env.train(scenarios, episodes=episodes)
    saved = env.q_controller.save(save_q_table)
    last = summaries[-1]
    print(f"trained_episodes={episodes}")
    print(f"last_accuracy={last.accuracy:.3f}")
    print(f"last_average_reward={last.average_reward:.3f}")
    print(f"memory_items={last.memory_items}")
    print(f"q_table={saved}")


def command_eval(episodes: int) -> None:
    env = HackathonAIEnvironment()
    scenarios = default_scenarios()
    env.train(scenarios, episodes=episodes)
    results = env.evaluate(scenarios)
    total_reward = sum(result.reward.total for result in results)
    accuracy = sum(1 for result in results if result.final_agent == result.scenario.expected_domain) / len(results)
    print(f"accuracy={accuracy:.3f}")
    print(f"average_reward={total_reward / len(results):.3f}")
    print("details:")
    for result in results:
        print(
            f"- expected={result.scenario.expected_domain:<8} "
            f"chosen={result.final_agent:<8} "
            f"reward={result.reward.total:.3f} "
            f"state={result.state_key}"
        )


def command_ask(query: str, episodes: int) -> None:
    env = HackathonAIEnvironment(knowledge_retriever=InternetKnowledgeRetriever())
    scenarios = default_scenarios()
    env.train(scenarios, episodes=episodes)
    env.prime_memory(scenarios)
    result = env.answer_query(query)
    print(f"state={result.state_key}")
    print(f"selected_agent={result.action.selected_agent}")
    print(f"final_agent={result.final_agent}")
    print(f"use_memory={result.action.use_memory}")
    print(f"threshold={result.action.confidence_threshold:.2f}")
    print(f"recalled={', '.join(result.recalled_memory_keys) or 'none'}")
    print("explanation:")
    print(result.explanation)
    print("answer:")
    print(result.answer)


def command_web(host: str, port: int, episodes: int) -> None:
    serve_dashboard(host=host, port=port, default_episodes=episodes)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        command_train(args.episodes, args.save_q_table)
        return
    if args.command == "eval":
        command_eval(args.episodes)
        return
    if args.command == "ask":
        command_ask(args.query, args.episodes)
        return
    if args.command == "web":
        command_web(args.host, args.port, args.episodes)
        return


if __name__ == "__main__":
    main()
