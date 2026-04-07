"""encoder.py
Unified encoder interface. Chooses OpenAI provider when OPENAI_API_KEY is set,
otherwise uses local VectorEncoder.
"""
import os
from typing import List, Union

from src.core.vector_encoder import VectorEncoder

try:
    from src.core.vector_provider_openai import OpenAIEncoder
except Exception:
    OpenAIEncoder = None

try:
    from src.core.vector_provider_hf import HFEncoder
except Exception:
    HFEncoder = None


def get_encoder():
    # Priority: local VectorEncoder (if sentence-transformers installed) -> OpenAI -> HuggingFace -> local VectorEncoder
    # We try to detect sentence-transformers by importing the package
    try:
        import sentence_transformers  # type: ignore
        return VectorEncoder()
    except Exception:
        pass

    if os.getenv("OPENAI_API_KEY") and OpenAIEncoder is not None:
        return OpenAIEncoder()

    if (os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HF_TOKEN')) and HFEncoder is not None:
        return HFEncoder()

    return VectorEncoder()


def encode(text: Union[str, List[str]]):
    encoder = get_encoder()
    if isinstance(text, str):
        return encoder.encode(text)
    else:
        # support batch encoding
        if hasattr(encoder, 'encode_batch'):
            return encoder.encode_batch(text)
        return [encoder.encode(t) for t in text]
