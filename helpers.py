from typing import List, Dict
from agent_state import AgentState
import json
from langchain_openai import ChatOpenAI


def add_msg(state: AgentState, role: str, content: str):
    history = state.get("conversation_history", [])
    history.append({"role": role, "content": content})
    state["conversation_history"] = history


def get_recent_history(state: AgentState, n: int = 4) -> List[Dict[str, str]]:
    return state.get("conversation_history", [])[-n:]

def maybe_update_summary(state: AgentState, llm: ChatOpenAI, trigger_len: int = 8, recent_n: int = 6) -> None:
    """
    Update rolling summary only when conversation gets long enough.
    Keeps full conversation_history intact for debugging/inspection.
    """
    history = state.get("conversation_history", [])
    if len(history) < trigger_len:
        return

    old_summary = state.get("conversation_summary", "")
    recent = history[-recent_n:]

    prompt = f"""
                You are maintaining compact memory for a financial advisory workflow.

                Update the running summary of the conversation.
                Keep only important information:
                - client goals
                - risk tolerance
                - constraints
                - liquidity needs
                - portfolio / financial context
                - unresolved questions
                - what advice has already been given
                - what the client still wants clarified

                Existing summary:
                {old_summary if old_summary else "No summary yet."}

                Recent conversation:
                {json.dumps(recent, indent=2)}

                Return only the updated summary.
                """.strip()

    try:
        state["conversation_summary"] = llm.invoke(prompt).content.strip()
    except Exception:
        # keep old summary on failure
        state["conversation_summary"] = old_summary


def build_context(state: AgentState, recent_n: int = 4) -> str:
    """
    Compact context passed to advisor nodes instead of full conversation history.
    """
    summary = state.get("conversation_summary", "")
    known_facts = state.get("known_facts", {})
    recent = get_recent_history(state, recent_n)

    return f"""
                Known facts:
                {json.dumps(known_facts, indent=2)}

                Conversation summary:
                {summary if summary else "No summary yet."}

                Recent conversation:
                {json.dumps(recent, indent=2)}
                """.strip()