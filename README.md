# Multi Agent Hive System
This repository is a prototype multi-agent hive system.

Providers supported for embeddings:
- Local sentence-transformers (recommended, no API key)
- OpenAI Embeddings (requires OPENAI_API_KEY)
- Hugging Face Inference API (requires HUGGINGFACE_API_KEY or HF_TOKEN)

Quick setup (PowerShell):

# Create virtual env and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# To use OpenAI: set env var OPENAI_API_KEY
# To use Hugging Face: set env var HUGGINGFACE_API_KEY or HF_TOKEN

# Run demo
python scripts\run_multi_agent.py
