"""
TASK 2: Shared Knowledge Space
Central hub where all agents read/write signals and knowledge
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Signal from an agent to shared knowledge space"""
    agent_id: str
    signal_type: str  # "prediction", "confidence", "suggestion"
    content: Any
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be in [0, 1]")


@dataclass
class AgentState:
    """State of each agent"""
    agent_id: str
    agent_type: str
    is_active: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    accuracy: float = 0.5
    total_decisions: int = 0
    correct_decisions: int = 0


class SharedKnowledgeSpace:
    def __init__(self, dimension: int = 384, max_signals: int = 1000):
        self.dimension = dimension
        self.max_signals = max_signals
        self.signals: Dict[str, List[Signal]] = {}
        self.vector_store: Dict[str, np.ndarray] = {}
        self.confidence_matrix: Dict[str, Dict[str, float]] = {}
        self.probability_distributions: Dict[str, Dict] = {}
        self.agents: Dict[str, AgentState] = {}
        self.query: str = ""
        self.domain: str = ""
        self.timestamp: datetime = datetime.now()
        logger.info(f"SharedKnowledgeSpace initialized: dim={dimension}, max_signals={max_signals}")

    def register_agent(self, agent_id: str, agent_type: str):
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered")
            return
        self.agents[agent_id] = AgentState(agent_id=agent_id, agent_type=agent_type)
        self.signals[agent_id] = []
        self.confidence_matrix[agent_id] = {}
        logger.info(f"Agent registered: {agent_id} ({agent_type})")

    def publish_signal(self, agent_id: str, signal_type: str, content: Any, confidence: float):
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        signal = Signal(agent_id=agent_id, signal_type=signal_type, content=content, confidence=confidence)
        self.signals[agent_id].append(signal)
        if len(self.signals[agent_id]) > self.max_signals:
            self.signals[agent_id] = self.signals[agent_id][-self.max_signals:]
        logger.debug(f"Signal published: {agent_id} -> {signal_type} (conf={confidence:.2f})")

    def get_agent_signals(self, agent_id: str, signal_type: Optional[str] = None) -> List[Signal]:
        if agent_id not in self.signals:
            return []
        signals = self.signals[agent_id]
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return signals

    def get_all_signals(self, signal_type: Optional[str] = None) -> List[Signal]:
        all_signals: List[Signal] = []
        for agent_id in self.signals:
            all_signals.extend(self.get_agent_signals(agent_id, signal_type))
        return all_signals

    def store_vector(self, key: str, vector: Any):
        vec = np.asarray(vector, dtype=float)
        if vec.ndim != 1:
            raise ValueError("Vector must be one-dimensional")
        # Allow dynamic dimension: set on first store
        if not self.vector_store and vec.size > 0 and self.dimension != vec.size:
            # adapt dimension to first vector
            self.dimension = vec.size
        if vec.size != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {vec.size} != {self.dimension}")
        self.vector_store[key] = vec
        logger.debug(f"Vector stored: {key}")

    def get_vector(self, key: str) -> Optional[np.ndarray]:
        return self.vector_store.get(key)

    def query(self, vector_or_key_or_text: Any, top_k: int = 5, encoder=None) -> List[Tuple[str, float]]:
        """Query the vector store.
        Accepts:
         - numpy array vector
         - stored key (str) to look up its vector
         - raw text (str) if encoder provided
        Returns list of (key, score) sorted descending by cosine similarity
        """
        if isinstance(vector_or_key_or_text, str) and encoder is not None and vector_or_key_or_text not in self.vector_store:
            # treat as raw text
            emb = encoder.encode(vector_or_key_or_text)
            # encoder may return a dataclass with .vector or raw list
            if hasattr(emb, 'vector'):
                query_vec = np.asarray(emb.vector, dtype=float)
            else:
                query_vec = np.asarray(emb, dtype=float)
        elif isinstance(vector_or_key_or_text, str) and vector_or_key_or_text in self.vector_store:
            query_vec = self.vector_store[vector_or_key_or_text]
        else:
            query_vec = np.asarray(vector_or_key_or_text, dtype=float)

        if query_vec.ndim != 1:
            raise ValueError('Query vector must be a 1-D array')

        results: List[Tuple[str, float]] = []
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            return []

        for key, vec in self.vector_store.items():
            # compute cosine similarity
            denom = (qnorm * np.linalg.norm(vec))
            if denom == 0:
                score = 0.0
            else:
                score = float(np.dot(query_vec, vec) / denom)
            results.append((key, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def set_confidence(self, agent_id: str, item: str, confidence: float):
        if agent_id not in self.confidence_matrix:
            self.confidence_matrix[agent_id] = {}
        self.confidence_matrix[agent_id][item] = confidence
        logger.debug(f"Confidence set: {agent_id}[{item}] = {confidence:.2f}")

    def get_confidence(self, agent_id: str, item: str) -> Optional[float]:
        return self.confidence_matrix.get(agent_id, {}).get(item)

    def set_probability(self, state_name: str, agent_id: str, value: float):
        if state_name not in self.probability_distributions:
            self.probability_distributions[state_name] = {}
        self.probability_distributions[state_name][agent_id] = value
        logger.debug(f"Probability set: {state_name}[{agent_id}] = {value:.2f}")

    def update_agent_accuracy(self, agent_id: str, correct: bool):
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        agent = self.agents[agent_id]
        agent.total_decisions += 1
        if correct:
            agent.correct_decisions += 1
        agent.accuracy = agent.correct_decisions / agent.total_decisions if agent.total_decisions > 0 else 0.5
        agent.last_updated = datetime.now()
        logger.debug(f"Accuracy updated: {agent_id} = {agent.accuracy:.2f}")

    def get_agent_ranking(self) -> List[tuple]:
        rankings = []
        for agent_id, agent in self.agents.items():
            rankings.append((agent_id, agent.accuracy, agent.total_decisions))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def reset_for_query(self, query: str, domain: str = "general"):
        self.query = query
        self.domain = domain
        self.timestamp = datetime.now()
        for agent_id in self.signals:
            self.signals[agent_id] = []
        self.confidence_matrix = {agent_id: {} for agent_id in self.agents}
        self.probability_distributions = {}
        logger.info(f"Knowledge space reset for query: {query[:50]}... (domain={domain})")

    def get_state_snapshot(self) -> Dict:
        return {
            "query": self.query,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "active_agents": len([a for a in self.agents.values() if a.is_active]),
            "total_signals": sum(len(s) for s in self.signals.values()),
            "vector_count": len(self.vector_store),
            "agent_rankings": self.get_agent_ranking(),
            "probability_states": list(self.probability_distributions.keys())
        }

    def clear(self):
        self.signals = {agent_id: [] for agent_id in self.agents}
        self.vector_store = {}
        self.confidence_matrix = {agent_id: {} for agent_id in self.agents}
        self.probability_distributions = {}
        logger.info("Knowledge space cleared")
    
    def get_agent_signals(self, agent_id: str, signal_type: Optional[str] = None) -> List[Signal]:
        """
        Get signals from specific agent
        
        Args:
            agent_id: Agent ID
            signal_type: Optional filter by type
            
        Returns:
            List of signals
        """
        if agent_id not in self.signals:
            return []
        
        signals = self.signals[agent_id]
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        return signals
    
    def get_all_signals(self, signal_type: Optional[str] = None) -> List[Signal]:
        """Get all signals from all agents"""
        all_signals = []
        for agent_id in self.signals:
            signals = self.get_agent_signals(agent_id, signal_type)
            all_signals.extend(signals)
        
        return all_signals
    
    def store_vector(self, key: str, vector: np.ndarray):
        """Store embedding vector in knowledge space"""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {len(vector)} != {self.dimension}")
        
        self.vector_store[key] = vector
        logger.debug(f"Vector stored: {key}")
    
    def get_vector(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding vector"""
        return self.vector_store.get(key)
    
    def set_confidence(self, agent_id: str, item: str, confidence: float):
        """
        Set confidence score for agent on item
        
        Args:
            agent_id: Agent ID
            item: Item being evaluated
            confidence: Confidence score
        """
        if agent_id not in self.confidence_matrix:
            self.confidence_matrix[agent_id] = {}
        
        self.confidence_matrix[agent_id][item] = confidence
        logger.debug(f"Confidence set: {agent_id}[{item}] = {confidence:.2f}")
    
    def get_confidence(self, agent_id: str, item: str) -> Optional[float]:
        """Get confidence score"""
        return self.confidence_matrix.get(agent_id, {}).get(item)
    
    def get_confidence_matrix(self, item: str) -> Dict[str, float]:
        """Get confidence scores from all agents for an item"""
        confidences = {}
        for agent_id in self.confidence_matrix:
            conf = self.confidence_matrix[agent_id].get(item)
            if conf is not None:
                confidences[agent_id] = conf
        
        return confidences
    
    def set_probability(self, state_name: str, agent_id: str, value: float):
        """
        Set probability in distribution
        
        Args:
            state_name: Name of probability state
            agent_id: Agent providing probability
            value: Probability value
        """
        if state_name not in self.probability_distributions:
            self.probability_distributions[state_name] = {}
        
        self.probability_distributions[state_name][agent_id] = value
        logger.debug(f"Probability set: {state_name}[{agent_id}] = {value:.2f}")
    
    def get_probabilities(self, state_name: str) -> Dict[str, float]:
        """Get all probabilities for a state"""
        return self.probability_distributions.get(state_name, {})
    
    def update_agent_accuracy(self, agent_id: str, correct: bool):
        """Update agent's accuracy metric"""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        agent = self.agents[agent_id]
        agent.total_decisions += 1
        if correct:
            agent.correct_decisions += 1
        
        agent.accuracy = agent.correct_decisions / agent.total_decisions if agent.total_decisions > 0 else 0.5
        agent.last_updated = datetime.now()
        
        logger.debug(f"Accuracy updated: {agent_id} = {agent.accuracy:.2f}")
    
    def get_agent_accuracy(self, agent_id: str) -> float:
        """Get agent's accuracy"""
        if agent_id not in self.agents:
            return 0.0
        
        return self.agents[agent_id].accuracy
    
    def get_agent_ranking(self) -> List[tuple]:
        """Rank agents by accuracy"""
        rankings = []
        for agent_id, agent in self.agents.items():
            rankings.append((agent_id, agent.accuracy, agent.total_decisions))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def reset_for_query(self, query: str, domain: str = "general"):
        """Reset space for new query"""
        self.query = query
        self.domain = domain
        self.timestamp = datetime.now()
        
        # Clear signals for fresh start
        for agent_id in self.signals:
            self.signals[agent_id] = []
        
        self.confidence_matrix = {agent_id: {} for agent_id in self.agents}
        self.probability_distributions = {}
        
        logger.info(f"Knowledge space reset for query: {query[:50]}... (domain={domain})")
    
    def get_state_snapshot(self) -> Dict:
        """Get snapshot of current knowledge space state"""
        return {
            "query": self.query,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "active_agents": len([a for a in self.agents.values() if a.is_active]),
            "total_signals": sum(len(s) for s in self.signals.values()),
            "vector_count": len(self.vector_store),
            "agent_rankings": self.get_agent_ranking(),
            "probability_states": list(self.probability_distributions.keys())
        }
    
    def clear(self):
        """Clear all knowledge space"""
        self.signals = {agent_id: [] for agent_id in self.agents}
        self.vector_store = {}
        self.confidence_matrix = {agent_id: {} for agent_id in self.agents}
        self.probability_distributions = {}
        logger.info("Knowledge space cleared")


if __name__ == "__main__":
    # Test Shared Knowledge Space
    space = SharedKnowledgeSpace(dimension=384)
    
    # Register agents
    space.register_agent("food_agent", "food")
    space.register_agent("business_agent", "business")
    space.register_agent("coding_agent", "coding")
    
    # Reset for query
    space.reset_for_query("Healthy food for coding marathon", "food+coding")
    
    # Publish signals
    space.publish_signal("food_agent", "prediction", "salad", 0.8)
    space.publish_signal("coding_agent", "prediction", "energy drink", 0.6)
    space.publish_signal("business_agent", "suggestion", "snack box", 0.7)
    
    # Set confidences
    space.set_confidence("food_agent", "salad", 0.8)
    space.set_confidence("coding_agent", "energy drink", 0.6)
    space.set_confidence("business_agent", "snack box", 0.7)
    
    # Update accuracy
    space.update_agent_accuracy("food_agent", True)
    space.update_agent_accuracy("coding_agent", False)
    space.update_agent_accuracy("business_agent", True)
    
    # Get state
    state = space.get_state_snapshot()
    print(f"✓ Knowledge space state:")
    print(f"  Query: {state['query']}")
    print(f"  Active agents: {state['active_agents']}")
    print(f"  Total signals: {state['total_signals']}")
    print(f"  Rankings: {state['agent_rankings']}")
    
    print("\n✅ Task 2: Shared Knowledge Space - COMPLETE")