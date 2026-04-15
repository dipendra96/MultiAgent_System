# Multi-Agent Financial Investment Advisor System

This project implements a multi-agent system using LangGraph for providing personalized financial investment advice. The system simulates a client-advisor-analyst interaction where agents collaborate to deliver tailored recommendations based on client profiles, internal knowledge, and web research.

## Overview

The system consists of three main agents:
- **Client Agent**: Simulates a financial client with hidden profile data, generating queries and responding to advisor messages.
- **Advisor Agent**: Manages the conversation, extracts client information, defines research tasks for the analyst, and provides final advice.
- **Analyst Agent**: Researches using internal knowledge store and web search to provide data-driven insights.

The workflow uses LangGraph to orchestrate agent interactions with conditional routing, structured outputs, and conversation memory.

## Workflow Diagram

```
START
└── client_start: Client agent generates initial query based on hidden profile
    └── advisor_intake: Advisor extracts known facts and assesses if enough info for research
        ├── Not enough info? → client_answer: Client responds to advisor questions
        │   └── Loop back to advisor_intake (for clarification)
        └── Enough info? → advisor_task: Advisor creates specific research task for analyst
            └── analyst: Analyst uses knowledge store and web search to gather insights
                └── advisor_review: Advisor evaluates analyst response
                    ├── Needs revision AND < max_rounds? → Loop back to advisor_task
                    └── Ready OR max_rounds reached? → advisor_present: Advisor delivers final advice
                        └── client_feedback: Client evaluates advice
                            ├── Not satisfied? → Loop back to advisor_intake (with feedback notes)
                            └── Satisfied? → END
```

## Setup Instructions

### Prerequisites
- Python 3.10+

### 1. Install Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Alternatively, if using conda:
```bash
conda install -c conda-forge langchain langchain-openai langchain-community langgraph pydantic
```

### 2. Set Environment Variables
github gives access to free LLM endpoints but anyone can use their own endpoints.
This configuration is only for LLM endpoint from [github marketplace](https://github.com/marketplace/models/)

Create a `.env` file or set environment variables directly:

```bash
export github_pat="your-github-pat-key-here"
```

### 3. Prepare Data Files
Ensure the following files exist in the `data/` directory:
- `client_profile.json`: Contains client profiles (simulated client persona).
- `knowledge_store.json`: Internal knowledge base with financial planning rules.

### 5. Run the System
Execute the main script:

```bash
python agent_system.py
```

The system will:
- Load the first client profile from `client_profile.json`.
- Simulate a conversation.
- Output the final advice, client satisfaction, known facts, and conversation history.

## Module Explanations

### Core Files
- **`agent_system.py`**: Main entry point. Initializes LLMs with structured outputs, creates the analyst agent, builds the graph, and runs the simulation.
- **`graph_builder.py`**: Constructs the LangGraph StateGraph. Defines nodes, edges, and conditional routing logic.
- **`agent_state.py`**: Defines the `AgentState` TypedDict, which holds shared state across nodes (conversation history, known facts, messages, etc.).
- **`model.py`**: Contains Pydantic models for structured LLM outputs (e.g., `AdvisorIntakeResult`, `ClientFeedbackResult`).
- **`routing.py`**: Functions for conditional edges in the graph (e.g., route based on whether enough info is gathered).
- **`helpers.py`**: Utility functions for conversation management, such as adding messages, updating summaries, and building context.
- **`tools.py`**: Defines tools for the analyst agent: `search_web` (Google Serper API) and `search_knowledge` (retrieves from `knowledge_store.json` by ID).

### Node Modules
Located in the `nodes/` folder:
- **`advisor_nodes.py`**: Implements advisor-related nodes (intake, task creation, review, presentation).
- **`client_nodes.py`**: Implements client-related nodes (start conversation, answer questions, provide feedback).

### Data Files
- **`data/client_profile.json`**: JSON array of client profiles with demographics, finances, goals, etc.
- **`data/knowledge_store.json`**: JSON array of knowledge entries (rules, heuristics) for financial planning.
- **`requirements.txt`**: List of Python dependencies with versions.

## Workflow Explanation

The system follows a conversational, iterative workflow using LangGraph:

1. **Client Start** (`client_start`): Client agent generates an initial query based on hidden profile.

2. **Advisor Intake** (`advisor_intake`): Advisor extracts known facts from conversation, decides if more info is needed.
   - If not enough info: Routes to `client_answer` for clarification.
   - If enough: Routes to `advisor_task`.

3. **Client Answer** (`client_answer`): Client responds to advisor questions.

4. **Advisor Task** (`advisor_task`): Advisor creates a research task for the analyst based on known facts.

5. **Analyst** (`analyst`): Analyst uses tools (knowledge store, web search) to research and provide insights.

6. **Advisor Review** (`advisor_review`): Advisor reviews analyst response.
   - If ready: Routes to `advisor_present`.
   - If needs revision and under max rounds: Loops back to `advisor_task`.
   - If max rounds reached: Proceeds to present.

7. **Advisor Present** (`advisor_present`): Advisor provides final client-facing advice.

8. **Client Feedback** (`client_feedback`): Client evaluates advice.
   - If satisfied: End.
   - If not: Routes back to `advisor_intake` with feedback as notes.

### Key Features
- **Structured Outputs**: Uses Pydantic models for reliable LLM responses (e.g., parsing decisions, facts).
- **Memory Management**: Maintains full conversation history and a rolling summary to handle long interactions.
- **Tool Integration**: Analyst accesses curated knowledge and web search for up-to-date info.
- **Iterative Refinement**: Allows multiple analyst rounds if initial research is insufficient.
- **Simulation**: Client is AI-simulated, enabling automated testing with different profiles.

### Data Sources
- **Client Profiles**: Pre-defined JSON with realistic financial data.
- **Knowledge Store**: Internal rules for risk assessment, asset allocation, etc.
- **Web Search**: Real-time information via Google Serper API.

### Customization
- Change client profile in `agent_system.py` (e.g., `profiles[1]` for different client).
- Adjust max analyst rounds in initial state.
- Modify prompts in node functions for different behavior.
- Add new tools or knowledge entries as needed.

## Evaluation

To validate the effectiveness of the multi-agent system, a lightweight evaluation approach was used focusing on workflow correctness, recommendation quality, and system termination.

### 1. Test Setup
The system was tested on 5 simulated client profiles with varying characteristics:
- Different risk tolerances (conservative, moderate, aggressive)
- Diverse financial goals (retirement, home purchase, emergency fund)
- Constraints (liquidity needs, no leverage, low-cost preference)

### 2. Evaluation Criteria

#### a. Workflow Correctness
- Client initiates the conversation
- Advisor gathers sufficient information before providing advice
- Advisor creates a structured task for the Analyst
- Analyst uses knowledge store first, then web search if needed
- Advisor reviews Analyst output before responding to client

#### b. Recommendation Quality
- Advice aligns with client goals and time horizon
- Risk level matches client risk tolerance
- Constraints (e.g., liquidity, no leverage) are respected
- Output is clear, practical, and actionable

#### c. Agent Interaction Quality
- Relevant context is passed between Advisor and Analyst
- Analyst retrieves and applies appropriate knowledge rules
- Advisor requests refinement when analysis is insufficient

#### d. Termination Behavior
- System concludes only when the client expresses satisfaction
- Follow-up questions trigger additional reasoning loops when needed

### 3. Results Summary
Across tested scenarios:
- The system successfully followed the intended multi-agent workflow
- Recommendations were generally aligned with client profiles
- Iterative refinement between Advisor and Analyst improved output quality
- Conversations terminated correctly when client satisfaction was reached

### 4. Limitations
- Evaluation is based on a small set of simulated profiles
- No quantitative scoring or large-scale benchmarking was performed
- Real-world financial advice would require stricter validation and compliance checks