"""
TASK 3: Router/Planner
Intelligent routing of queries to appropriate agents
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Task domains"""
    FOOD = "food"
    BUSINESS = "business"
    CODING = "coding"
    RESEARCH = "research"
    MEMORY = "memory"
    GENERAL = "general"


@dataclass
class RoutingDecision:
    """Routing decision result"""
    primary_agent: str
    secondary_agents: List[str]
    domain: Domain
    confidence: float
    reasoning: str


class RouterPlanner:
    """Route queries to appropriate agents"""
    
    def __init__(self):
        """Initialize router"""
        self.agent_domains = {
            "food_agent": [Domain.FOOD],
            "business_agent": [Domain.BUSINESS],
            "coding_agent": [Domain.CODING],
            "research_agent": [Domain.RESEARCH],
            "memory_agent": [Domain.MEMORY]
        }
        
        self.keywords = {
            Domain.FOOD: ["food", "eat", "meal", "recipe", "nutrition", "diet", "snack", "drink"],
            Domain.BUSINESS: ["business", "company", "market", "sales", "strategy", "startup", "profit"],
            Domain.CODING: ["code", "program", "algorithm", "debug", "function", "python", "javascript"],
            Domain.RESEARCH: ["research", "study", "analyze", "investigate", "paper", "data", "findings"],
            Domain.MEMORY: ["remember", "recall", "history", "past", "previous", "stored"]
        }
        
        logger.info("RouterPlanner initialized")
    
    def detect_domain(self, query: str) -> Tuple[Domain, float]:
        """
        Detect domain from query
        
        Args:
            query: Input query
            
        Returns:
            (Domain, confidence)
        """
        query_lower = query.lower()
        domain_scores = {}
        
        # Score each domain
        for domain, keywords in self.keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[domain] = score
        
        # Find best match
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Calculate confidence
        if max_score == 0:
            confidence = 0.5
            best_domain = Domain.GENERAL
        else:
            # More keywords = higher confidence
            confidence = min(1.0, 0.5 + (max_score * 0.1))
        
        logger.debug(f"Domain detected: {best_domain.value} (conf={confidence:.2f})")
        
        return best_domain, confidence
    
    def route_query(self, query: str) -> RoutingDecision:
        """
        Route query to appropriate agents
        
        Args:
            query: Input query
            
        Returns:
            RoutingDecision with primary and secondary agents
        """
        # Detect domain
        domain, domain_confidence = self.detect_domain(query)
        
        # Find agents for this domain
        primary_agent = None
        secondary_agents = []
        
        for agent_id, domains in self.agent_domains.items():
            if domain in domains:
                if primary_agent is None:
                    primary_agent = agent_id
                else:
                    secondary_agents.append(agent_id)
        
        # Handle case where no specific agent found
        if primary_agent is None:
            primary_agent = "memory_agent"  # Default fallback
            secondary_agents = list(self.agent_domains.keys())[:2]
        
        # Determine confidence
        if domain == Domain.GENERAL:
            confidence = 0.5
        else:
            confidence = domain_confidence
        
        reasoning = f"Query contains {domain.value} keywords. Primary agent: {primary_agent}"
        
        decision = RoutingDecision(
            primary_agent=primary_agent,
            secondary_agents=secondary_agents,
            domain=domain,
            confidence=confidence,
            reasoning=reasoning
        )
        
        logger.info(f"Routing decision: {primary_agent} (domain={domain.value}, conf={confidence:.2f})")
        
        return decision
    
    def route_multi_domain(self, query: str) -> List[RoutingDecision]:
        """
        Handle multi-domain queries
        
        Args:
            query: Input query
            
        Returns:
            List of routing decisions for multiple domains
        """
        query_lower = query.lower()
        detected_domains = []
        
        # Find all relevant domains
        for domain, keywords in self.keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domains.append(domain)
        
        # If multiple domains, create routing for each
        if len(detected_domains) > 1:
            logger.info(f"Multi-domain query detected: {[d.value for d in detected_domains]}")
            
            decisions = []
            for domain in detected_domains:
                # Find agent for this domain
                agents_for_domain = [
                    agent_id for agent_id, domains in self.agent_domains.items()
                    if domain in domains
                ]
                
                if agents_for_domain:
                    decision = RoutingDecision(
                        primary_agent=agents_for_domain[0],
                        secondary_agents=agents_for_domain[1:],
                        domain=domain,
                        confidence=0.8,
                        reasoning=f"Multi-domain: handling {domain.value} aspect"
                    )
                    decisions.append(decision)
            
            return decisions
        else:
            # Single domain - use regular routing
            return [self.route_query(query)]
    
    def get_load_balanced_agent(self, domain: Domain, agent_loads: Dict[str, int]) -> str:
        """
        Get least loaded agent for domain
        
        Args:
            domain: Task domain
            agent_loads: Dict of agent_id -> current_load
            
        Returns:
            Agent ID with lowest load
        """
        domain_agents = [
            agent_id for agent_id, domains in self.agent_domains.items()
            if domain in domains
        ]
        
        if not domain_agents:
            return list(self.agent_domains.keys())[0]
        
        # Return agent with lowest load
        best_agent = min(domain_agents, key=lambda a: agent_loads.get(a, 0))
        
        logger.debug(f"Load-balanced agent: {best_agent} (load={agent_loads.get(best_agent, 0)})")
        
        return best_agent
    
    def explain_routing(self, decision: RoutingDecision) -> str:
        """Explain routing decision to user"""
        explanation = f"""
Routing Decision:
- Primary Agent: {decision.primary_agent}
- Secondary Agents: {', '.join(decision.secondary_agents) if decision.secondary_agents else 'None'}
- Domain: {decision.domain.value}
- Confidence: {decision.confidence:.1%}
- Reasoning: {decision.reasoning}
        """
        return explanation.strip()
    
    def add_agent(self, agent_id: str, domains: List[Domain]):
        """Add new agent"""
        self.agent_domains[agent_id] = domains
        logger.info(f"Agent added: {agent_id} -> {[d.value for d in domains]}")
    
    def remove_agent(self, agent_id: str):
        """Remove agent"""
        if agent_id in self.agent_domains:
            del self.agent_domains[agent_id]
            logger.info(f"Agent removed: {agent_id}")
    
    def get_agents_for_domain(self, domain: Domain) -> List[str]:
        """Get all agents for a domain"""
        return [
            agent_id for agent_id, domains in self.agent_domains.items()
            if domain in domains
        ]
    
    def get_all_agents(self) -> List[str]:
        """Get all registered agents"""
        return list(self.agent_domains.keys())



def route(intention: str) -> List[str]:
    """Base router function to match environment expectations"""
    mapping = {
        "food": ["food_agent"],
        "business": ["business_agent"],
        "code": ["coding_agent"],
        "research": ["research_agent"],
    }
    return mapping.get(intention, ["research_agent", "coding_agent"])


if __name__ == "__main__":

    # Test Router/Planner
    router = RouterPlanner()
    
    # Test single domain query
    decision = router.route_query("What's the best food for a coding marathon?")
    print(f"✓ Routed: {decision.primary_agent}")
    print(f"  Domain: {decision.domain.value}")
    print(f"  Confidence: {decision.confidence:.2f}")
    
    # Test multi-domain query
    decisions = router.route_multi_domain("Healthy food for my tech startup")
    print(f"\n✓ Multi-domain routing: {len(decisions)} agents")
    for d in decisions:
        print(f"  - {d.primary_agent} ({d.domain.value})")
    
    # Test domain detection
    domain, conf = router.detect_domain("Debug my Python code")
    print(f"\n✓ Domain detected: {domain.value} (conf={conf:.2f})")
    
    print("\n✅ Task 3: Router/Planner - COMPLETE")