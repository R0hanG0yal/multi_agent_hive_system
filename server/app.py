"""server/app.py - OpenEnv entry point."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hackathon_ai_env.web import serve_dashboard

def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    episodes = int(os.environ.get("EPISODES", "8"))
    serve_dashboard(host=host, port=port, default_episodes=episodes)

if __name__ == "__main__":
    main()
