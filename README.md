---
title: Hackathon AI Dashboard
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Hackathon AI Environment

This project turns the shared design into a runnable prototype:

```text
User Query
   -> Vector Encoding
   -> Multi-Agent Layer
   -> Shared Knowledge Space
   -> Decision Fusion Engine
   -> Q-Learning Controller
   -> Deterministic Reward Node
   -> Memory Update with Decay
   -> Next State
```

It includes four agents:

- `food`
- `business`
- `research`
- `memory`

The environment is designed for hackathon demos where you want:

- multi-agent routing
- deterministic, judge-friendly rewards
- a real Q-table update loop
- memory recall across sequential queries
- a clean CLI for training and evaluation

## Project Layout

- `hackathon_ai_env/agents.py`: specialist agents and query-domain encoding
- `hackathon_ai_env/fusion.py`: decision fusion and action generation
- `hackathon_ai_env/q_learning.py`: epsilon-greedy action selection and Q updates
- `hackathon_ai_env/reward.py`: deterministic reward scoring
- `hackathon_ai_env/memory.py`: shared memory with decay and reinforcement
- `hackathon_ai_env/environment.py`: end-to-end environment loop
- `hackathon_ai_env/scenarios.py`: sample benchmark tasks
- `main.py`: CLI entrypoint

## Run It

Train on the built-in scenarios:

```bash
python3 main.py train --episodes 40
```

Train and evaluate the learned policy:

```bash
python3 main.py eval --episodes 40
```

Train, warm shared memory, and ask a custom question:

```bash
python3 main.py ask "What pricing model should we use for a student startup?" --episodes 40
```

Launch the browser dashboard:

```bash
python3 main.py web --port 8000 --episodes 40
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Deploy On Hugging Face

This repo is now set up for a Hugging Face Docker Space.

1. Create a new Space and choose the `Docker` SDK.
2. Push this entire repository to that Space.
3. Hugging Face will build the included `Dockerfile` and start the dashboard on port `7860`.

The container starts the app with:

```bash
python3 main.py web --host 0.0.0.0 --port ${PORT:-7860} --episodes ${EPISODES:-40}
```

Optional runtime variables you can set in the Space settings:

- `EPISODES`: default training episodes for the dashboard, defaults to `40`
- `PORT`: app port, defaults to `7860`
- `HOST`: bind address, defaults to `0.0.0.0`
- `APP_STATE_PATH`: persisted dashboard state file, defaults to `/data/hackathon_ai_env_state.json` in the Docker image

Notes for deployment:

- Hugging Face Docker Spaces expect a single exposed app port; this repo uses `7860`.
- The dashboard now autosaves the Q-table, memory bank, training summaries, and latest feedback to `APP_STATE_PATH`, and reloads them on startup.
- To keep that state across Hugging Face restarts, enable persistent storage for the Space so `/data` survives restarts.
- Local browser URLs such as `127.0.0.1` are only for local development; Hugging Face will provide the public Space URL after deployment.

## What The Q-Learning Action Means

Each action is a compact policy choice:

- which agent to trust
- whether to use shared memory
- what confidence threshold must be met before accepting that agent

If the selected agent does not clear the threshold, the environment falls back to the fusion winner.

## Reward Design

The reward node is deterministic and reproducible. It scores:

- domain accuracy
- keyword coverage
- memory usage quality
- confidence alignment

This keeps the system hackathon-friendly because the grader does not depend on a stochastic LLM reward.

## Tests

Run the lightweight test suite with:

```bash
python3 -m unittest discover -s tests
```
