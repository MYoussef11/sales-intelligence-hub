import os
import sys
import logging
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings
from ml_services.rag_agent import InternalSalesAgent
from ml_services.sql_agent import SecureSQLAgent

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize singletons
rag_agent = InternalSalesAgent()
sql_agent = SecureSQLAgent()

class AgentState(TypedDict):
    messages: list
    next_step: str
    final_answer: str

def router_node(state: AgentState):
    """
    Decides whether to route to SQL or RAG based on the user's question.
    """
    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content if hasattr(last_message, "content") else str(last_message)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    system_prompt = (
        "You are a routing assistant. "
        "Your task is to classify the user's question into one of two categories:\n"
        "1. 'sql' -> For questions about data, numbers, sales, dealers, inventory, revenue, or 'how many'.\n"
        "2. 'rag' -> For questions about policies, text documents, rules, incentives, compliance, or warranty.\n"
        "Return ONLY the keyword 'sql' or 'rag'."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])
    
    choice = response.content.strip().lower()
    
    # Fallback/Safety
    if "sql" in choice:
        return {"next_step": "sql"}
    else:
        return {"next_step": "rag"}

def sql_node(state: AgentState):
    messages = state["messages"]
    question = messages[-1].content
    
    logger.info(f"Routing to SQL Agent: {question}")
    try:
        response = sql_agent.run_query(question)
    except Exception as e:
        response = f"SQL Agent Error: {str(e)}"
        
    return {"final_answer": response}

def rag_node(state: AgentState):
    messages = state["messages"]
    question = messages[-1].content
    
    logger.info(f"Routing to RAG Agent: {question}")
    try:
        response = rag_agent.query(question)
    except Exception as e:
        response = f"RAG Agent Error: {str(e)}"
        
    return {"final_answer": response}

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("rag_agent", rag_node)

workflow.set_entry_point("router")

def route_decision(state):
    return state["next_step"] + "_agent"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "sql_agent": "sql_agent",
        "rag_agent": "rag_agent"
    }
)

workflow.add_edge("sql_agent", END)
workflow.add_edge("rag_agent", END)

app_graph = workflow.compile()

def run_chat(user_input: str) -> str:
    """
    Main entry point for the API.
    """
    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        result = app_graph.invoke(inputs)
        return result.get("final_answer", "No answer generated.")
    except Exception as e:
        logger.error(f"Orchestrator Error: {e}", exc_info=True)
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    print(run_chat("What is the return policy?"))
    print(run_chat("How many dealers do we have?"))
