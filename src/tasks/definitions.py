from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Task:
    id: str
    instruction: str
    ground_truth: str
    difficulty: str
    required_memories: List[str] = field(default_factory=list)

TASKS = [
    Task(
        id="EASY_01",
        instruction="USER PREFERENCE: 'I love double-shot espresso.' RESPONSE: What is the user's favorite coffee?",
        ground_truth="double-shot espresso",
        difficulty="easy"
    ),
    Task(
        id="MEDIUM_01",
        instruction="RECALL: 'Project deadline is October 15th.' CONTEXT: Today is October 10th. QUESTION: How many days left until the project deadline?",
        ground_truth="5 days",
        difficulty="medium"
    ),
    Task(
        id="HARD_01",
        instruction="WORKFLOW: 'User prefers Marriott hotels.' ACTION: Plan a travel itinerary for a trip next Tuesday to a Marriott-affiliated location. What hotel chain are we using?",
        ground_truth="Marriott",
        difficulty="hard"
    )
]
