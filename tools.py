from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import json

@tool
def search_web(query: str) -> str:
    """Search the web for relevant financial planning information."""
    try:
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def search_knowledge(knowledge_id: str) -> str:
    """Get one knowledge entry from knowledge_store.json by knowledge_id."""
    try:
        with open("data/knowledge_store.json", "r") as f:
            data = json.load(f)

        for item in data:
            if item.get("knowledge_id") == knowledge_id:
                return json.dumps(item, indent=2)

        return f"No knowledge found for id: {knowledge_id}"
    except Exception as e:
        return f"Knowledge store error: {e}"