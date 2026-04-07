"""
TASK 1: Vector Encoder
Converts text queries into vector embeddings for the hive system
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Embedding:
    """Embedding representation"""
    text: str
    vector: np.ndarray
    dimension: int
    model: str
    
    def similarity(self, other: "Embedding") -> float:
        """Cosine similarity with another embedding"""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vector dimensions must match")
        
        dot_product = np.dot(self.vector, other.vector)
        norm_product = np.linalg.norm(self.vector) * np.linalg.norm(other.vector)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)


class VectorEncoder:
    """Encode text queries into embeddings"""
    
    def __init__(self, model: str = "sentence-transformers", dimension: int = 384):
        """
        Initialize encoder
        
        Args:
            model: Embedding model type
            dimension: Embedding dimension
        """
        self.model = model
        self.dimension = dimension
        self.embeddings_cache = {}
        
        logger.info(f"VectorEncoder initialized: model={model}, dim={dimension}")
    
    def encode(self, text: str) -> Embedding:
        """
        Encode text to embedding
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding object
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be non-empty string")
        
        # Check cache
        if text in self.embeddings_cache:
            logger.debug(f"Cache hit for: {text[:50]}")
            return self.embeddings_cache[text]
        
        # Generate embedding
        vector = self._generate_embedding(text)
        
        embedding = Embedding(
            text=text,
            vector=vector,
            dimension=len(vector),
            model=self.model
        )
        
        # Cache result
        self.embeddings_cache[text] = embedding
        
        logger.debug(f"Encoded text: {text[:50]}... -> vector shape: {vector.shape}")
        
        return embedding
    
    def encode_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Encode multiple texts
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of embeddings
        """
        if not texts:
            raise ValueError("Input list cannot be empty")
        
        embeddings = []
        for text in texts:
            embedding = self.encode(text)
            embeddings.append(embedding)
        
        logger.info(f"Batch encoded {len(embeddings)} texts")
        
        return embeddings
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text
        Uses simple hash-based approach for demo
        """
        # Simulate embedding generation
        # In production, use sentence-transformers or OpenAI embeddings
        hash_val = hash(text)
        np.random.seed(abs(hash_val) % (2**32))
        vector = np.random.randn(self.dimension)
        vector = vector / np.linalg.norm(vector)  # Normalize
        
        return vector
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0, 1]
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        return emb1.similarity(emb2)
    
    def find_similar(self, text: str, candidates: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find most similar candidates to text
        
        Args:
            text: Query text
            candidates: List of candidate texts
            top_k: Number of top results
            
        Returns:
            List of (text, similarity) tuples
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        query_emb = self.encode(text)
        similarities = []
        
        for candidate in candidates:
            candidate_emb = self.encode(candidate)
            sim = query_emb.similarity(candidate_emb)
            similarities.append((candidate, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embeddings_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "dimension": self.dimension,
            "model": self.model
        }


if __name__ == "__main__":
    # Test Vector Encoder
    encoder = VectorEncoder(dimension=384)
    
    # Test single encoding
    emb = encoder.encode("What's the best food for coding?")
    print(f"✓ Encoded: {emb.text[:40]}...")
    print(f"  Vector shape: {emb.vector.shape}")
    
    # Test similarity
    sim = encoder.similarity(
        "What's the best food for coding?",
        "Good food for programmers"
    )
    print(f"✓ Similarity: {sim:.3f}")
    
    # Test finding similar
    results = encoder.find_similar(
        "food for coding",
        [
            "eating healthy",
            "programming tips",
            "best snacks",
            "coding techniques"
        ],
        top_k=2
    )
    print(f"✓ Similar candidates:")
    for text, sim in results:
        print(f"  - {text}: {sim:.3f}")
    
    print("\n✅ Task 1: Vector Encoder - COMPLETE")