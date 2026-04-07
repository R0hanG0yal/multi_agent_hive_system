"""environment.py
Simplified pipeline runner: encode -> register agents -> run agents -> fuse -> return final
"""
from typing import Dict, Any, List
from src.core.encoder import get_encoder
from src.core.shared_knowledge_space import SharedKnowledgeSpace
from src.core.router_planner import route
from src.decision.fusion_engine import fuse
from src.agents.food_agent import FoodAgent
from src.agents.business_agent import BusinessAgent
from src.agents.coding_agent import CodingAgent
from src.agents.research_agent import ResearchAgent
from src.agents.memory_agent import MemoryAgent


def run_query(text: str) -> Dict[str, Any]:
    encoder = get_encoder()
    sks = SharedKnowledgeSpace()

    # instantiate agents and register
    agents = [
        FoodAgent('food_agent', 'food'),
        BusinessAgent('business_agent', 'business'),
        CodingAgent('coding_agent', 'coding'),
        ResearchAgent('research_agent', 'research'),
        MemoryAgent('memory_agent', 'memory')
    ]

    for a in agents:
        sks.register_agent(a.agent_id, a.agent_type)

    sks.reset_for_query(text)

    # get embedding
    emb = encoder.encode(text)
    if hasattr(emb, 'vector'):
        vec = emb.vector
    else:
        vec = emb

    sks.store_vector('query_0', vec)

    # simple intent detection by keywords
    intent = 'general'
    low = text.lower()
    if any(w in low for w in ['food', 'eat', 'snack']):
        intent = 'food'
    elif any(w in low for w in ['business', 'market', 'strategy']):
        intent = 'business'
    elif any(w in low for w in ['code', 'bug', 'implement']):
        intent = 'code'
    elif any(w in low for w in ['research', 'study', 'paper']):
        intent = 'research'

    # route
    agent_ids = route(intent)

    # run selected agents (if route returned names; otherwise run all)
    responses = []
    for a in agents:
        if agent_ids and a.agent_id not in agent_ids and agent_ids != ['research_agent','coding_agent']:
            continue
        decision = a.process_query(text, {})
        # decision may be pydantic model
        try:
            score = float(getattr(decision, 'confidence', decision.get('confidence', 0.0)))
            content = getattr(decision, 'content', decision.get('content'))
        except Exception:
            score = 0.0
            content = str(decision)
        sks.publish_signal(a.agent_id, 'prediction', content, score)
        responses.append({'agent_id': a.agent_id, 'content': content, 'score': score})

    fused = fuse(responses)

    return {
        'query': text,
        'responses': responses,
        'fused': fused,
        'sks_snapshot': sks.get_state_snapshot()
    }
