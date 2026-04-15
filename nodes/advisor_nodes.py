import json

from langchain_core.messages import HumanMessage

from agent_state import AgentState
from helpers import add_msg, maybe_update_summary, build_context


def advisor_intake_node(state: AgentState, llm, intake_llm) -> AgentState:
    maybe_update_summary(state, llm)
    context = build_context(state, recent_n=4)

    prompt = f"""
                You are the advisor. You are friendly and professional in dealing with clients.

                From the available context, do two things:
                1. update the known client facts
                2. decide whether you have enough information to send a task to the analyst

                You need enough information about:
                - goals
                - risk tolerance
                - constraints or liquidity needs
                - useful financial or portfolio context

                If enough_info is false, advisor_message should ask the next best question(s).
                If enough_info is true, advisor_message should be a short internal summary for task creation.

                {context}
                """.strip()

    try:
        result = intake_llm.invoke(prompt)
        state["known_facts"] = result.known_facts.model_dump()
        state["enough_info"] = result.enough_info
        state["advisor_message"] = result.advisor_message
    except Exception:
        state["enough_info"] = False
        state["advisor_message"] = (
            "Could you tell me more about your financial goals, time horizon, "
            "and comfort with risk?"
        )

    add_msg(state, "advisor", state["advisor_message"])
    return state


def advisor_task_node(state: AgentState, llm) -> AgentState:
    maybe_update_summary(state, llm)
    known_facts = state.get("known_facts", {})
    review_notes = state.get("review_notes", "")
    summary = state.get("conversation_summary", "")

    prompt = f"""
                You are the advisor.

                Create one clear research task for the analyst based on these known client facts.
                The analyst should search the knowledge store first.
                If that is not enough, the analyst should search the web.

                If review notes exist, use them to improve the task.

                Return only the task text.

                Known facts:
                {json.dumps(known_facts, indent=2)}

                Conversation summary:
                {summary if summary else "No summary yet."}

                Review notes:
                {review_notes}
                """.strip()

    task = llm.invoke(prompt).content.strip()
    state["analyst_task"] = task
    add_msg(state, "advisor", f"Task for analyst: {task}")
    return state


def analyst_node(state: AgentState, analyst_agent) -> AgentState:
    task = state.get("analyst_task", "")

    prompt = f"""
                You are the analyst.

                You must follow this order:
                1. Use the knowledge store first
                2. If the knowledge store is not enough, use web search
                3. Write a practical analysis for the advisor

                Available knowledge ids:
                - risk_profile_rules
                - asset_allocation_templates
                - goal_bucket_rules
                - diversification_rules
                - suitability_rules

                Task:
                {task}
                """.strip()

    result = analyst_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    response = result["messages"][-1].content

    state["analyst_response"] = response
    state["analyst_rounds"] = state.get("analyst_rounds", 0) + 1
    add_msg(state, "analyst", response)
    return state


def advisor_review_node(state: AgentState, llm, review_llm) -> AgentState:
    maybe_update_summary(state, llm)
    known_facts = state.get("known_facts", {})
    task = state.get("analyst_task", "")
    analysis = state.get("analyst_response", "")
    summary = state.get("conversation_summary", "")

    prompt = f"""
                You are the advisor. You are friendly and professional in dealing with clients.

                Review the analyst response and decide whether it is ready for the client.
                Make sure the analyst response is accurate, practical, and addresses the client's needs based on the known facts.

                Known facts:
                {json.dumps(known_facts, indent=2)}

                Conversation summary:
                {summary if summary else "No summary yet."}

                Task:
                {task}

                Analyst response:
                {analysis}
                """.strip()

    try:
        result = review_llm.invoke(prompt)
        state["review_decision"] = result.decision
        state["review_notes"] = result.notes
    except Exception:
        state["review_decision"] = "revise"
        state["review_notes"] = "The analyst response is not reliable enough yet."

    add_msg(state, "advisor", f"Review: {state['review_decision']}. {state['review_notes']}")
    return state


def advisor_present_node(state: AgentState, llm) -> AgentState:
    maybe_update_summary(state, llm)
    known_facts = state.get("known_facts", {})
    analyst_response = state.get("analyst_response", "")
    summary = state.get("conversation_summary", "")

    prompt = f"""
                You are the advisor. You are friendly and professional in dealing with clients.

                Use the analyst response to give clear, practical, concise, easy-to-understand client-facing advice.
                Do not mention the analyst or tools.
                Do not give any advice which is not based on the analyst response.

                Known facts:
                {json.dumps(known_facts, indent=2)}

                Conversation summary:
                {summary if summary else "No summary yet."}

                Analyst response:
                {analyst_response}
                """.strip()

    advisor_msg = llm.invoke(prompt).content.strip()
    state["advisor_message"] = advisor_msg
    state["final_answer"] = advisor_msg
    add_msg(state, "advisor", advisor_msg)
    return state