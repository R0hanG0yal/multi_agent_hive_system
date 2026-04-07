"""
Hive System — Web Server
Wires the complete Multi-Agent pipeline into FastAPI endpoints.
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import sys

sys.path.append(os.getcwd())

from src.orchestrator import HiveOrchestrator

app = FastAPI(title="Hive Multi-Agent System")

# ── Initialize Groq Client ────────────────────────────────────────────────────
try:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=api_key)
except ImportError:
    groq_client = None

# ── Initialize Hive Orchestrator (singleton, holds all agents + RL controller) ──
orchestrator = HiveOrchestrator(groq_client=groq_client, db_path="hive_data.db")

# ── Web Search (live internet using ddgs) ─────────────────────────────────────
try:
    from ddgs import DDGS
    ddgs_available = True
except ImportError:
    ddgs_available = False

SEARCH_KEYWORDS = [
    "weather", "today", "current", "news", "price", "stock",
    "latest", "now", "who is", "what is", "how much", "score"
]


def get_search_results(query: str) -> str:
    """Runs DuckDuckGo search and returns formatted results string."""
    if not ddgs_available:
        return ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='wt-wt', max_results=3))
        if results:
            print(f"[Search] Found {len(results)} results for: {query}")
            return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        print(f"[Search] Failed: {e}")
    return ""


# ─── Request/Response Schemas ─────────────────────────────────────────────────
from typing import List, Dict

class ChatRequest(BaseModel):
    text: str
    history: List[Dict[str, str]] = []

class FeedbackRequest(BaseModel):
    feedback: str  # 'positive' | 'negative'


# ─── Chat Endpoint (Main Pipeline) ────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not groq_client:
        return {"error": "Groq client not initialized. Set GROQ_API_KEY environment variable."}

    # Check if query needs live internet data
    needs_search = any(kw in request.text.lower() for kw in SEARCH_KEYWORDS)
    search_results = get_search_results(request.text) if needs_search else ""

    try:
        result = orchestrator.process(
            query=request.text,
            chat_history=request.history,
            search_results=search_results
        )
        return result
    except Exception as e:
        print(f"[Orchestrator ERROR]: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


# ─── Feedback Endpoint (RL Reward Node) ───────────────────────────────────────
@app.post("/api/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """
    Called when user clicks 👍 or 👎.
    Triggers the Q-Learning Bellman update.
    """
    result = orchestrator.apply_feedback(request.feedback)
    return result


# ─── RL Stats Endpoint ────────────────────────────────────────────────────────
@app.get("/api/rl_stats")
async def rl_stats():
    return orchestrator.rl_controller.get_stats()


# ─── Legacy Environment Endpoints (kept for compatibility) ────────────────────
from src.env.assistant_env import MemoryAugmentedAssistantEnv
from src.models.schemas import Action

env = MemoryAugmentedAssistantEnv()

@app.get("/api/reset")
async def reset_env():
    obs = env.reset()
    return obs.dict()

@app.post("/api/step")
async def step_env(action_data: dict):
    action = Action(**action_data)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "state": info["state"]
    }


# ─── HTML Page Routes ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_chat():
    with open("public/chat.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/login", response_class=HTMLResponse)
async def read_login():
    with open("public/login.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/signup", response_class=HTMLResponse)
async def read_signup():
    with open("public/signup.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/support", response_class=HTMLResponse)
async def read_support():
    with open("public/support.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard():
    with open("public/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/dev", response_class=HTMLResponse)
async def read_dev():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ─── Authentication System ────────────────────────────────────────────────────
import sqlite3
import bcrypt
import jwt
import datetime

db_conn = sqlite3.connect("hive_data.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
db_conn.commit()

JWT_SECRET = "hive_production_jwt_secret_key"

class AuthRequest(BaseModel):
    email: str
    password: str

@app.post("/api/register")
async def register_user(request: AuthRequest):
    try:
        db_cursor.execute("SELECT email FROM users WHERE email=?", (request.email,))
        if db_cursor.fetchone():
            return {"error": "Email is already registered"}
        hashed = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db_cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (request.email, hashed))
        db_conn.commit()
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/login_user")
async def auth_login(request: AuthRequest):
    try:
        db_cursor.execute("SELECT password_hash FROM users WHERE email=?", (request.email,))
        row = db_cursor.fetchone()
        if not row or not bcrypt.checkpw(request.password.encode('utf-8'), row[0].encode('utf-8')):
            return {"error": "Invalid email or password"}
        expiration = datetime.datetime.utcnow() + datetime.timedelta(days=7)
        token = jwt.encode({"email": request.email, "exp": expiration}, JWT_SECRET, algorithm="HS256")
        return {"token": token, "email": request.email}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
