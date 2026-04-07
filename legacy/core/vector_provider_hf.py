"""Hugging Face Inference API embedding provider wrapper.

Usage:
    encoder = HFEncoder()
    emb = encoder.encode("some text")

Requires HUGGINGFACE_API_KEY (or HF_TOKEN) env var and the `requests` package.
"""
from typing import List, Iterable, Union
import os
import requests

HF_API_URL = os.getenv('HUGGINGFACE_API_URL', 'https://api-inference.huggingface.co/embeddings')
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')

class HFEncoder:
    def __init__(self, model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = model
        if not HF_TOKEN:
            raise RuntimeError('HUGGINGFACE_API_KEY or HF_TOKEN not set')
        self.headers = { 'Authorization': f'Bearer {HF_TOKEN}' }

    def encode(self, text: Union[str, Iterable[str]]) -> List[List[float]]:
        if isinstance(text, str):
            payload = {"model": self.model, "inputs": text}
        else:
            payload = {"model": self.model, "inputs": list(text)}

        resp = requests.post(f"https://api-inference.huggingface.co/models/{self.model}/embeddings", headers=self.headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"HF API error: {resp.status_code} - {resp.text}")
        data = resp.json()
        # expected shape: {'embeddings': [...] } or list of embeddings
        if isinstance(data, dict) and 'embeddings' in data:
            embeddings = data['embeddings']
        else:
            embeddings = data
        return embeddings if isinstance(embeddings, list) and len(embeddings) > 1 else embeddings[0]
