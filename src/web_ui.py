from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add current path to sys.path
sys.path.append(os.getcwd())

from src.env.assistant_env import MemoryAugmentedAssistantEnv
from src.models.schemas import Action

app = FastAPI()
env = MemoryAugmentedAssistantEnv()

# Initialize Groq Client
try:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=api_key)
except ImportError:
    groq_client = None

from typing import List, Dict

class QueryRequest(BaseModel):
    text: str
    history: List[Dict[str, str]] = []

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

from ddgs import DDGS

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    if not groq_client:
        return {"error": "Groq library not installed properly or client failing."}
    
    # Retrieve relevant past memories using Vector Search
    memories = env.memory.retrieve(request.text)
    mem_text = "\n".join([f"- {m.content}" for m in memories])
    
    system_prompt = "You are MAHA, the Memory-Augmented Hive Assistant. Be helpful, concise, and clear. You have real-time internet access."
    
    # Basic router: check if query needs internet
    search_keywords = ["weather", "today", "current", "news", "price", "stock", "time", "latest", "now", "who is", "what is"]
    needs_search = any(k in request.text.lower() for k in search_keywords)
    
    if needs_search:
        try:
            with DDGS() as ddgs:
                # Use region wt-wt to avoid localized blocks
                results = list(ddgs.text(request.text, region='wt-wt', max_results=3))
                if results:
                    search_results_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
                    print("DDG SEARCH SUCCESS! Found:", len(results), "results.")
                    system_prompt += f"\n\n[CRITICAL INSTRUCTION: YOU HAVE BEEN PROVIDED LIVE INTERNET SEARCH RESULTS BELOW.]\n[DO NOT say you lack real-time access. USE this data to confidently answer the user:]\n\nSEARCH RESULTS FOR '{request.text}':\n{search_results_text}"
                else:
                    print("DDG SEARCH RETURNED EMPTY LIST!")
        except Exception as e:
            print(f"--- DDG SEARCH FAILED! ERROR: {e} ---")

    if mem_text:
        system_prompt += f"\n\n[LONG-TERM MEMORY CONTEXT]:\n{mem_text}"
        
    messages = [{"role": "system", "content": system_prompt}]
    
    # Append recent rolling chat history
    for msg in request.history[-10:]:  # Keep last 10 messages for short-term context
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Append current user query
    messages.append({"role": "user", "content": request.text})
        
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
        )
        response_text = chat_completion.choices[0].message.content
        
        # Save this interaction to Vector Memory
        env.memory.add_memory(f"User profile/fact: {request.text}")
        
        return {"response": response_text}
    except Exception as e:
        return {"error": str(e)}

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

@app.get("/dev", response_class=HTMLResponse)
async def read_dev():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return f.read()

# --- AUTHENTICATION SYSTEM ---
import sqlite3
import bcrypt
import jwt
import datetime

# Initialize SQLite DB
db_conn = sqlite3.connect("hive_data.db", check_same_thread=False)
db_cursor = db_conn.cursor()
db_cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')
db_conn.commit()

JWT_SECRET = "production_ready_secret_key_hive"

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
