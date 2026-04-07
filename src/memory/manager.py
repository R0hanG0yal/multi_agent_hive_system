import uuid
import numpy as np
from typing import List
from src.models.schemas import MemoryEntry

try:
    from sentence_transformers import SentenceTransformer
    # Load a tiny, super-fast embedding model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    encoder = None

class MemoryManager:
    def __init__(self):
        self.store: List[MemoryEntry] = []
        self.embeddings = []
    
    def add_memory(self, content: str, metadata: dict = None):
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or {}
        )
        self.store.append(entry)
        
        # Create Vector Embedding if model is available
        if encoder:
            vec = encoder.encode(content)
            self.embeddings.append(vec)
            
        return entry
    
    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        if not self.store:
            return []
            
        # 1. Semantic Vector Search (True Meaning Recognition)
        if encoder and len(self.embeddings) > 0:
            query_vec = encoder.encode(query)
            
            # Calculate Cosine Similarity
            similarities = []
            for idx, vec in enumerate(self.embeddings):
                cos_sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
                similarities.append((cos_sim, self.store[idx]))
            
            # Sort by highest semantic meaning, threshold at 0.3 for relevance
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [entry for score, entry in similarities[:top_k] if score > 0.3]
            
        # 2. Fallback: Keyword Search
        query_words = set(query.lower().split())
        scored = []
        for entry in self.store:
            content_words = set(entry.content.lower().split())
            score = len(query_words.intersection(content_words))
            scored.append((score, entry))
        
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [entry for score, entry in scored[:top_k] if score > 0]

    def get_all(self):
        return self.store
