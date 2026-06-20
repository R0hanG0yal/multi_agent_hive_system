from __future__ import annotations

from .models import AgentProposal

class ConflictResolutionNode:
    """
    Sits between Shared Knowledge Space and the Decision Engine, resolving
    tie-breaks or conflicts in agent proposals before final fusion scoring.
    """
    
    def resolve(self, proposals: dict[str, AgentProposal]) -> dict[str, AgentProposal]:
        resolved: dict[str, AgentProposal] = {}
        
        # Track confidences to handle exact ties
        confidence_map: dict[float, list[str]] = {}
        for name, proposal in proposals.items():
            confidence_map.setdefault(proposal.confidence, []).append(name)
            resolved[name] = proposal
            
        for confidence, tied_agents in confidence_map.items():
            if len(tied_agents) > 1:
                # Perform conflict resolution by penalizing identical confidences slightly 
                # so the Decision Engine has a clear frontrunner.
                tied_agents.sort()  # deterministic alphabetic tie-break
                for idx, agent_name in enumerate(tied_agents):
                    if idx > 0:
                        original = resolved[agent_name]
                        # Apply tiny penalty to break exact ties and resolve conflict
                        new_conf = round(original.confidence - (0.0001 * idx), 4)
                        resolved[agent_name] = AgentProposal(
                            agent_name=original.agent_name,
                            domain=original.domain,
                            confidence=new_conf,
                            keywords_hit=original.keywords_hit,
                            base_response=original.base_response,
                            memory_response=original.memory_response,
                            rationale=original.rationale + f" (resolved tie-break at {confidence})"
                        )
                        
        return resolved
