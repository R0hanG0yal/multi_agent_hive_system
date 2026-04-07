from src.base_agent import BaseAgent, Decision
class FoodAgent(BaseAgent):
    def process_query(self, query: str, context: dict) -> Decision:
        return Decision(agent_id=self.agent_id, decision_type="answer", content="Healthy snacks for coding include nuts, berries, or yogurt. Avoid heavy or sugary snacks to maintain focus.", confidence=0.9)
    def extract_domain_knowledge(self, query: str) -> dict:
        return {"domain":"food"}
