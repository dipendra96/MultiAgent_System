import json
from typing import Dict, Any

from agent_state import AgentState
from helpers import add_msg


def make_client_start_node(profile: Dict[str, Any], llm):
    def client_start(state: AgentState) -> AgentState:
        prompt = f"""
                    You are a financial client.

                    Start the conversation naturally by asking for investment help.
                    Your conversation style should be based on your profile.
                    Do not reveal your whole profile at once.

                    Profile:
                    {json.dumps(profile, indent=2)}
                    """.strip()

        msg = llm.invoke(prompt).content.strip()
        state["client_message"] = msg
        add_msg(state, "client", msg)
        return state

    return client_start


def make_client_answer_node(profile: Dict[str, Any], client_answer_llm):
    def client_answer(state: AgentState) -> AgentState:
        advisor_message = state.get("advisor_message", "")

        prompt = f"""
                    You are a financial client. You came to the advisor for help with your investments.
                    The advisor has just asked you a question to better understand your situation.

                    Reply naturally to the advisor using your profile.
                    Only share what is relevant to the advisor's latest question.
                    Your conversation style should be based on your profile.

                    Profile:
                    {json.dumps(profile, indent=2)}

                    Advisor message:
                    {advisor_message}
                    """.strip()

        result = client_answer_llm.invoke(prompt)
        state["client_message"] = result.client_message
        add_msg(state, "client", result.client_message)
        return state

    return client_answer


def make_client_feedback_node(profile: Dict[str, Any], client_feedback_llm):
    def client_feedback(state: AgentState) -> AgentState:
        advisor_message = state.get("advisor_message", "")

        prompt = f"""
                    You are a financial client.

                    The advisor has now given you investment advice.
                    Reply naturally using your profile.
                    Evaluate that the advice addresses your needs and concerns based on your profile.
                    Only be satisifed if you don't have follow-up questions or concerns based on the advice and your profile.

                    Say:
                    - whether the advice helps
                    - whether you are satisfied
                    - if not satisfied, what is still missing or unclear

                    Profile:
                    {json.dumps(profile, indent=2)}

                    Advisor advice:
                    {advisor_message}
                    """.strip()

        result = client_feedback_llm.invoke(prompt)
        state["client_message"] = result.client_message
        state["client_satisfied"] = result.satisfied
        add_msg(state, "client", result.client_message)
        return state

    return client_feedback