from src.base_agent import BaseAgent, Decision
class CodingAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="code", content="CodingAgent stub.", confidence=0.5)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"coding"}
