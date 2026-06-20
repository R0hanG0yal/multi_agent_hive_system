from __future__ import annotations
import math

class VectorEncoder:
    """
    Lightweight Text-to-Numbers Encoder.
    Uses character bigrams to generate a deterministic 64-dimensional sequence.
    This replaces abstract terms with actual Coordinate Space search points.
    """
    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def encode(self, text: str) -> tuple[float, ...]:
        if not text:
            return tuple([0.0] * self.dimensions)
            
        vector = [0.0] * self.dimensions
        normalized = text.lower().strip()
        if len(normalized) < 2:
            normalized += " "
            
        # Distribute string semantic features across a fixed-size vector space
        # using a lightweight rolling hash of bigrams to capture term features.
        for i in range(len(normalized) - 1):
            bigram = normalized[i:i+2]
            idx = (ord(bigram[0]) * 31 + ord(bigram[1])) % self.dimensions
            vector[idx] += 1.0
            
        # L2 Normalize
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [round(v / magnitude, 4) for v in vector]
            
        return tuple(vector)

    @staticmethod
    def cosine_similarity(v1: tuple[float, ...], v2: tuple[float, ...]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return round(dot_product / (mag1 * mag2), 4)
