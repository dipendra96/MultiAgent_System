from typing import Dict, Any
from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from routing import route_after_intake, route_after_review, route_after_feedback
from nodes.client_nodes import (
    make_client_start_node,
    make_client_answer_node,
    make_client_feedback_node,
)
from nodes.advisor_nodes import (
    advisor_intake_node,
    advisor_task_node,
    analyst_node,
    advisor_review_node,
    advisor_present_node,
)


def build_graph(
    hidden_profile: Dict[str, Any],
    llm,
    intake_llm,
    review_llm,
    client_answer_llm,
    client_feedback_llm,
    analyst_agent,
):
    graph = StateGraph(AgentState)

    graph.add_node("client_start", make_client_start_node(hidden_profile, llm))
    graph.add_node("client_answer", make_client_answer_node(hidden_profile, client_answer_llm))
    graph.add_node("client_feedback", make_client_feedback_node(hidden_profile, client_feedback_llm))

    graph.add_node("advisor_intake", lambda state: advisor_intake_node(state, llm, intake_llm))
    graph.add_node("advisor_task", lambda state: advisor_task_node(state, llm))
    graph.add_node("analyst", lambda state: analyst_node(state, analyst_agent))
    graph.add_node("advisor_review", lambda state: advisor_review_node(state, llm, review_llm))
    graph.add_node("advisor_present", lambda state: advisor_present_node(state, llm))

    graph.add_edge(START, "client_start")
    graph.add_edge("client_start", "advisor_intake")

    graph.add_conditional_edges(
        "advisor_intake",
        route_after_intake,
        {
            "client_answer": "client_answer",
            "advisor_task": "advisor_task",
        },
    )

    graph.add_edge("client_answer", "advisor_intake")
    graph.add_edge("advisor_task", "analyst")
    graph.add_edge("analyst", "advisor_review")

    graph.add_conditional_edges(
        "advisor_review",
        route_after_review,
        {
            "advisor_task": "advisor_task",
            "advisor_present": "advisor_present",
        },
    )

    graph.add_edge("advisor_present", "client_feedback")

    graph.add_conditional_edges(
        "client_feedback",
        route_after_feedback,
        {
            "advisor_intake": "advisor_intake",
            END: END,
        },
    )

    return graph.compile()