# This class defines the agent state that is shared between nodes
from typing import Any, Dict, List, TypedDict

class AgentState(TypedDict, total=False):
    conversation_history: List[Dict[str, str]]
    conversation_summary: str

    client_message: str
    advisor_message: str

    known_facts: Dict[str, Any]

    enough_info: bool
    analyst_task: str
    analyst_response: str

    review_decision: str
    review_notes: str

    client_satisfied: bool
    final_answer: str

    analyst_rounds: int
    max_analyst_rounds: int