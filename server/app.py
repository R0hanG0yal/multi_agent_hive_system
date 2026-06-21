"""server/app.py - OpenEnv entry point."""
from __future__ import annotations
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from hackathon_ai_env.web import serve_dashboard

def keep_alive() -> None:
    """Pings the Render external URL every 14 minutes to keep the server awake."""
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        time.sleep(14 * 60)  # 14 minutes
        try:
            req = urllib.request.Request(f"{url}/health", headers={'User-Agent': 'KeepAlivePing/1.0'})
            urllib.request.urlopen(req)
        except Exception as e:
            print(f"Keep-alive ping failed: {e}")

def main() -> None:
    # Start the keep-alive background thread
    threading.Thread(target=keep_alive, daemon=True).start()
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    episodes = int(os.environ.get("EPISODES", "8"))
    serve_dashboard(host=host, port=port, default_episodes=episodes)

if __name__ == "__main__":
    main()
