"""
TASKS 4-5-6: Base Agent Interface, Pydantic Models, Utils & Helpers
Core foundations for the multi-agent system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path

from pydantic import BaseModel, Field, validator


# ============================================================================
# TASK 5: PYDANTIC MODELS
# ============================================================================

class AgentConfig(BaseModel):
    """Agent configuration"""
    agent_id: str
    agent_type: str
    domain: str
    model: str = "gpt-4"
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(500, gt=0)
    enabled: bool = True


class Decision(BaseModel):
    """Decision made by agent"""
    agent_id: str
    decision_type: str
    content: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryUpdate(BaseModel):
    """Memory update message"""
    agent_id: str
    action: str  # "store", "retrieve", "decay", "consolidate"
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class RewardInfo(BaseModel):
    """Reward information"""
    agent_id: str
    score: float = Field(0.5, ge=0.0, le=1.0)
    correctness: float = Field(0.5, ge=0.0, le=1.0)
    speed: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    feedback: str = ""


class SystemConfig(BaseModel):
    """System-wide configuration"""
    project_name: str = "MultiAgentHive"
    version: str = "1.0.0"
    log_level: str = "INFO"
    max_agents: int = 10
    embedding_dim: int = 384
    vector_store_size: int = 1000
    memory_capacity: int = 5000


# ============================================================================
# TASK 6: UTILS & HELPERS
# ============================================================================

class Logger:
    """Centralized logging setup"""
    
    @staticmethod
    def setup(name: str, level: str = "INFO") -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger


class ConfigManager:
    """Configuration management"""
    
    def __init__(self, config_dir: str = "agents_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML config as JSON for simplicity"""
        import yaml
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config not found: {filepath}")
        
        with open(filepath) as f:
            return yaml.safe_load(f)
    
    def save_config(self, filename: str, data: Dict[str, Any]):
        """Save config"""
        import yaml
        filepath = self.config_dir / filename
        with open(filepath, 'w') as f:
            yaml.dump(data, f)


class Validator:
    """Input validation"""
    
    @staticmethod
    def validate_confidence(value: float) -> bool:
        """Validate confidence score"""
        return 0.0 <= value <= 1.0
    
    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate query"""
        return isinstance(query, str) and len(query.strip()) > 0
    
    @staticmethod
    def validate_agent_id(agent_id: str) -> bool:
        """Validate agent ID"""
        return isinstance(agent_id, str) and len(agent_id) > 0


class ErrorHandler:
    """Error handling and recovery"""
    
    @staticmethod
    def handle_agent_error(agent_id: str, error: Exception, logger: logging.Logger):
        """Handle agent error gracefully"""
        logger.error(f"Agent {agent_id} error: {str(error)}")
        # Return fallback decision
        return Decision(
            agent_id=agent_id,
            decision_type="fallback",
            content="unable_to_process",
            confidence=0.0,
            reasoning=f"Error: {str(error)}"
        )
    
    @staticmethod
    def handle_knowledge_space_error(error: Exception, logger: logging.Logger):
        """Handle knowledge space error"""
        logger.error(f"Knowledge space error: {str(error)}")
        # System continues with degraded functionality


# ============================================================================
# TASK 4: BASE AGENT INTERFACE
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: str, config: Optional[AgentConfig] = None):
        """
        Initialize agent
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (food, business, coding, research, memory)
            config: Agent configuration
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            domain=agent_type
        )
        
        self.logger = Logger.setup(f"Agent.{agent_id}")
        self.decision_history: List[Decision] = []
        self.accuracy = 0.5
        self.total_decisions = 0
        self.correct_decisions = 0
        
        self.logger.info(f"Agent initialized: {agent_id} ({agent_type})")
    
    @abstractmethod
    def process_query(self, query: str, context: Dict[str, Any]) -> Decision:
        """
        Process query and return decision
        
        Args:
            query: Input query
            context: Additional context from knowledge space
            
        Returns:
            Decision object
        """
        pass
    
    @abstractmethod
    def extract_domain_knowledge(self, query: str) -> Dict[str, Any]:
        """Extract domain-specific knowledge from query"""
        pass
    
    def make_decision(self, query: str, context: Dict[str, Any]) -> Decision:
        """
        Main decision-making method
        
        Args:
            query: Input query
            context: Knowledge space context
            
        Returns:
            Decision
        """
        if not Validator.validate_query(query):
            return Decision(
                agent_id=self.agent_id,
                decision_type="error",
                content="invalid_query",
                confidence=0.0
            )
        
        try:
            decision = self.process_query(query, context)
            self.decision_history.append(decision)
            self.total_decisions += 1
            
            self.logger.debug(f"Decision made: {decision.content} (conf={decision.confidence:.2f})")
            
            return decision
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return ErrorHandler.handle_agent_error(self.agent_id, e, self.logger)
    
    def update_accuracy(self, correct: bool):
        """Update accuracy metric"""
        if correct:
            self.correct_decisions += 1
        
        self.accuracy = self.correct_decisions / max(1, self.total_decisions)
        self.logger.debug(f"Accuracy updated: {self.accuracy:.2f}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "accuracy": self.accuracy,
            "total_decisions": self.total_decisions,
            "correct_decisions": self.correct_decisions,
            "decision_history_size": len(self.decision_history)
        }
    
    def publish_signal(self, knowledge_space, signal_type: str, content: Any, confidence: float):
        """Publish signal to knowledge space"""
        knowledge_space.publish_signal(
            agent_id=self.agent_id,
            signal_type=signal_type,
            content=content,
            confidence=confidence
        )
    
    def reset(self):
        """Reset agent state"""
        self.decision_history.clear()
        self.accuracy = 0.5
        self.total_decisions = 0
        self.correct_decisions = 0
        self.logger.info(f"Agent reset: {self.agent_id}")


if __name__ == "__main__":
    # Test models
    config = AgentConfig(
        agent_id="test_agent",
        agent_type="food",
        domain="food"
    )
    print(f"✓ Config: {config.agent_id}")
    
    decision = Decision(
        agent_id="test_agent",
        decision_type="classification",
        content="salad",
        confidence=0.85,
        reasoning="Detected health keywords"
    )
    print(f"✓ Decision: {decision.content} (conf={decision.confidence})")
    
    logger = Logger.setup("test")
    print(f"✓ Logger setup")
    
    validator = Validator()
    print(f"✓ Query valid: {validator.validate_query('test query')}")
    
    print("\n✅ Tasks 4-5-6: Base Agent, Models, Utils - COMPLETE")