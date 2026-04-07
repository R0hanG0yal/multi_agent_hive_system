"""
Specialized Agent Nodes
-----------------------
Implements all 5 domain nodes from the flowchart:
  - FoodNode
  - BusinessNode
  - CodingNode
  - ResearchNode
  - MemoryAgentNode

Each node processes the user query from its expert perspective
and returns an AgentOutput with a confidence score.
"""
import time
from src.agents.base_node import BaseAgentNode, AgentOutput


class FoodNode(BaseAgentNode):
    name = "FoodNode"
    domain = "food"
    domain_keywords = [
        "food", "recipe", "cook", "eat", "restaurant", "meal", "diet",
        "nutrition", "ingredient", "kitchen", "bake", "grill", "calories",
        "vegan", "vegetarian", "cuisine", "breakfast", "lunch", "dinner"
    ]

    def process(self, query: str, context: str, groq_client, history: list = []) -> AgentOutput:
        t0 = time.time()
        system = (
            "You are the Hive Food Expert Agent. You specialise in nutrition, recipes, "
            "cooking techniques, restaurant recommendations, and dietary advice. "
            "Answer only from your food/nutrition expertise. Be specific and practical. "
            f"{('Context from memory: ' + context) if context else ''}"
        )
        response = self._call_llm(system, query, groq_client, history)
        latency = (time.time() - t0) * 1000

        # Self-assessed confidence: high if many domain keywords hit
        confidence = min(0.95, 0.5 + self.relevance_score(query))
        return AgentOutput(
            agent_name=self.name, domain=self.domain,
            response=response, confidence=confidence,
            reasoning=f"Query matched food domain. Relevance: {self.relevance_score(query):.2f}",
            latency_ms=latency
        )


class BusinessNode(BaseAgentNode):
    name = "BusinessNode"
    domain = "business"
    domain_keywords = [
        "business", "startup", "finance", "investment", "market", "revenue",
        "profit", "company", "entrepreneur", "strategy", "marketing", "sales",
        "product", "customer", "budget", "economy", "stock", "money", "cost"
    ]

    def process(self, query: str, context: str, groq_client, history: list = []) -> AgentOutput:
        t0 = time.time()
        system = (
            "You are the Hive Business & Finance Expert Agent. You specialise in business "
            "strategy, entrepreneurship, market analysis, financial planning, and economics. "
            "Provide sharp, actionable business insights. "
            f"{('Context from memory: ' + context) if context else ''}"
        )
        response = self._call_llm(system, query, groq_client, history)
        latency = (time.time() - t0) * 1000
        confidence = min(0.95, 0.5 + self.relevance_score(query))
        return AgentOutput(
            agent_name=self.name, domain=self.domain,
            response=response, confidence=confidence,
            reasoning=f"Query matched business domain. Relevance: {self.relevance_score(query):.2f}",
            latency_ms=latency
        )


class CodingNode(BaseAgentNode):
    name = "CodingNode"
    domain = "coding"
    domain_keywords = [
        "code", "coding", "programming", "python", "javascript", "java", "c++",
        "algorithm", "bug", "debug", "function", "class", "api", "database",
        "sql", "html", "css", "react", "flask", "fastapi", "git", "docker",
        "error", "exception", "loop", "array", "string", "variable", "compile"
    ]

    def process(self, query: str, context: str, groq_client, history: list = []) -> AgentOutput:
        t0 = time.time()
        system = (
            "You are the Hive Software Engineering Expert Agent. You specialise in "
            "programming, debugging, system design, algorithms, and software best practices. "
            "Provide accurate, clean code samples when relevant. Use markdown code blocks. "
            f"{('Context from memory: ' + context) if context else ''}"
        )
        response = self._call_llm(system, query, groq_client, history)
        latency = (time.time() - t0) * 1000
        confidence = min(0.95, 0.5 + self.relevance_score(query))
        return AgentOutput(
            agent_name=self.name, domain=self.domain,
            response=response, confidence=confidence,
            reasoning=f"Query matched coding domain. Relevance: {self.relevance_score(query):.2f}",
            latency_ms=latency
        )


class ResearchNode(BaseAgentNode):
    name = "ResearchNode"
    domain = "research"
    domain_keywords = [
        "research", "science", "study", "history", "explain", "what is", "who is",
        "how does", "why does", "theory", "fact", "data", "analysis", "academic",
        "experiment", "discovery", "physics", "chemistry", "biology", "mathematics",
        "weather", "news", "current", "today", "latest", "world", "country"
    ]

    def process(self, query: str, context: str, groq_client, history: list = [],
                search_results: str = "") -> AgentOutput:
        t0 = time.time()
        system = (
            "You are the Hive Research & Knowledge Expert Agent. You specialise in science, "
            "history, current events, factual research, and real-world knowledge. "
            "IMPORTANT: If real-time search data is provided below, USE it to answer accurately. "
            "Do NOT say you lack real-time access if search results are provided. "
            f"{('Context from memory: ' + context) if context else ''}"
        )
        user_message = query
        if search_results:
            user_message = (
                f"[LIVE SEARCH DATA — USE THIS TO ANSWER ACCURATELY]:\n{search_results}\n\n"
                f"USER QUESTION: {query}"
            )
        response = self._call_llm(system, user_message, groq_client, history)
        latency = (time.time() - t0) * 1000
        confidence = min(0.95, 0.5 + self.relevance_score(query))
        if search_results:
            confidence = min(0.98, confidence + 0.1)  # Boost confidence when backed by real data
        return AgentOutput(
            agent_name=self.name, domain=self.domain,
            response=response, confidence=confidence,
            reasoning=f"Query matched research domain. Live search: {bool(search_results)}",
            latency_ms=latency,
            metadata={"has_search_data": bool(search_results)}
        )


class MemoryAgentNode(BaseAgentNode):
    """
    Special node: synthesises the user's personal history
    from vector memory to personalise any response.
    """
    name = "MemoryAgentNode"
    domain = "memory"
    domain_keywords = [
        "remember", "recall", "forgot", "earlier", "before", "previous",
        "last time", "you said", "my name", "i told", "i said", "preference"
    ]

    def process(self, query: str, context: str, groq_client, history: list = []) -> AgentOutput:
        t0 = time.time()
        if not context:
            return AgentOutput(
                agent_name=self.name, domain=self.domain,
                response="", confidence=0.0,
                reasoning="No relevant memories found for this query.",
                latency_ms=0.0
            )
        system = (
            "You are the Hive Memory Agent. Your ONLY job is to synthesise the user's "
            "personal history, preferences, and past interactions stored in memory "
            "to add a personalised, contextual layer to the response. "
            "Do NOT answer the question directly — provide the relevant personal context "
            "that other agents should consider.\n\n"
            f"MEMORY STORE:\n{context}"
        )
        response = self._call_llm(system, f"What personal context is relevant to: {query}", groq_client, [])
        latency = (time.time() - t0) * 1000
        confidence = 0.6 if context else 0.0
        return AgentOutput(
            agent_name=self.name, domain=self.domain,
            response=response, confidence=confidence,
            reasoning="Memory context synthesised from vector store.",
            latency_ms=latency
        )
