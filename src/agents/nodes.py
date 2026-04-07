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
            "You are MAHA, an advanced AI assistant built on the Hive multi-agent system. "
            "Respond in a warm, conversational, and engaging tone — like a knowledgeable friend, not a textbook. "
            "Use natural language: vary your sentence lengths, occasionally use contractions (you're, I'd, it's), "
            "and show genuine curiosity or enthusiasm where appropriate. "
            "Be concise but never terse. End responses with a follow-up question or offer to dig deeper when relevant. "
            "You specialise in food and nutrition. Answer only from your food/nutrition expertise and be specific and practical. "
            f"{('Here is some personal context about this user from memory: ' + context) if context else ''}"
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
            "You are MAHA, an advanced AI assistant with deep business and finance expertise. "
            "Talk like a smart, experienced mentor — warm, direct, and genuinely helpful. "
            "Use plain language, not corporate jargon. Vary your tone: be sharp when precision matters, "
            "conversational when explaining concepts. Use contractions naturally. "
            "Always end with an actionable tip or a clarifying question. "
            f"{('Personal context from memory: ' + context) if context else ''}"
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
            "You are MAHA, an expert software engineer who loves helping people with code. "
            "Explain things clearly — like a senior dev pair-programming with a junior. "
            "Be encouraging, precise, and practical. When showing code, always use markdown code blocks. "
            "If there's a bug, empathise first ('Ah, I see what's happening here...'), then fix it. "
            "Offer to explain further or explore edge cases at the end. "
            f"{('Personal context from memory: ' + context) if context else ''}"
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
            "You are MAHA, a curious and knowledgeable research assistant. "
            "Communicate findings conversationally — as if you're excitedly sharing something fascinating. "
            "Break down complex topics clearly with natural transitions. Never say 'As an AI...' or 'I cannot'. "
            "If real-time search data is provided below, use it confidently and cite the key facts naturally. "
            "Close with a relevant follow-up insight or question if appropriate. "
            f"{('Personal context from memory: ' + context) if context else ''}"
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
