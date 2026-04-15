import os
import json
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI


from model import (
    AdvisorIntakeResult,
    AdvisorReviewResult,
    ClientAnswerResult,
    ClientFeedbackResult,
)
from agent_state import AgentState
from tools import search_web, search_knowledge
from agent_state import AgentState
from graph_builder import build_graph

# =========================================================
# Setup
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ["github_pat"],
    base_url="https://models.inference.ai.azure.com",
)

# Wrap LLM with structured output for different nodes
intake_llm = llm.with_structured_output(AdvisorIntakeResult)
review_llm = llm.with_structured_output(AdvisorReviewResult)
client_answer_llm = llm.with_structured_output(ClientAnswerResult)
client_feedback_llm = llm.with_structured_output(ClientFeedbackResult)


# Analyst agent with access to tools
analyst_agent = create_react_agent(llm, [search_knowledge, search_web])


if __name__ == "__main__":
    with open("data/client_profile.json", "r") as f:
        profiles = json.load(f)

    hidden_profile = profiles[0]

    app = build_graph(
        hidden_profile=hidden_profile,
        llm=llm,
        intake_llm=intake_llm,
        review_llm=review_llm,
        client_answer_llm=client_answer_llm,
        client_feedback_llm=client_feedback_llm,
        analyst_agent=analyst_agent,
    )

    initial_state: AgentState = {
        "conversation_history": [],
        "conversation_summary": "",
        "client_message": "",
        "advisor_message": "",
        "known_facts": {},
        "enough_info": False,
        "analyst_task": "",
        "analyst_response": "",
        "review_decision": "revise",
        "review_notes": "",
        "client_satisfied": False,
        "final_answer": "",
        "analyst_rounds": 0,
        "max_analyst_rounds": 3,
    }

    result = app.invoke(initial_state)

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("final_answer", ""))

    print("\n=== CLIENT SATISFIED ===\n")
    print(result.get("client_satisfied", False))

    print("\n=== KNOWN FACTS ===\n")
    print(json.dumps(result.get("known_facts", {}), indent=2))

    print("\n=== CONVERSATION SUMMARY ===\n")
    print(result.get("conversation_summary", ""))

    print("\n=== CONVERSATION HISTORY ===\n")
    for msg in result.get("conversation_history", []):
        print(f"{msg['role'].upper()}: {msg['content']}\n")