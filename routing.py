from langgraph.graph import END
from agent_state import AgentState


def route_after_intake(state: AgentState) -> str:
    if state.get("enough_info", False):
        return "advisor_task"
    return "client_answer"


def route_after_review(state: AgentState) -> str:
    if state.get("review_decision") == "ready":
        return "advisor_present"

    if state.get("analyst_rounds", 0) >= state.get("max_analyst_rounds", 3):
        return "advisor_present"

    return "advisor_task"


def route_after_feedback(state: AgentState) -> str:
    if state.get("client_satisfied", False):
        return END

    state["analyst_response"] = ""
    state["review_notes"] = state.get("client_message", "")
    return "advisor_intake"