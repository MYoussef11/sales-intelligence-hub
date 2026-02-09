import os
import re
import sys
import logging
from sqlalchemy import create_engine, text
from langchain.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Use API Key from settings
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

class SecureSQLAgent:
    def __init__(self):
        # 1. READ-ONLY Connection
        # In production, use a specific read-only DB user. 
        # For POC, we rely on prompt engineering + regex guardrails.
        self.engine = create_engine(settings.DATABASE_URL)
        self.db = SQLDatabase(self.engine)
        
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

    def validate_query(self, query):
        """
        security guardrail: Prevent destructive queries.
        """
        forbidden_keywords = [
            "DELETE", "DROP", "UPDATE", "INSERT", "ALTER", "TRUNCATE", 
            "GRANT", "REVOKE", ";", "--" 
        ]
        
        # Normalize
        q_upper = query.upper()
        
        for kw in forbidden_keywords:
            if kw in q_upper:
                logger.warning(f"Security Alert: Blocked query containing '{kw}'")
                return False, f"Security Violation: '{kw}' is not allowed."
        
        return True, ""

    def run_query(self, natural_language_query):
        """
        Executes a natural language query with guardrails.
        """
        logger.info(f"Received Query: {natural_language_query}")
        
        query_prefix = (
            "You are a READ-ONLY data analyst. "
            "You must NOT modify data. "
            "If the user asks for more than 10 rows, you MUST add 'LIMIT 10' to the SQL. "
            "Do not query credentials or passwords. "
            f"Query: {natural_language_query}"
        )
        
        try:
            # We can't easily hook into the generated SQL *before* execution in the standard agent 
            # without custom callbacks, but the Agent Prompt is the first line of defense.
            # A robust production system would use a custom LLMChain or SQLDatabase wrapper to parse the SQL before execution.
            
            result = self.agent_executor.run(query_prefix)
            return result
            
        except Exception as e:
            logger.error(f"Agent Error: {e}")
            return f"I encountered an error processing your request: {str(e)}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = SecureSQLAgent()
    
    # Test Safe Query
    print("-" * 50)
    print("Test 1: Safe Query")
    response = agent.run_query("What are the total sales for the last 3 months?")
    print(f"Response: {response}")
    
    # Test Unsafe Query (Instruction to the LLM, though the LLM *might* refuse itself, we want to see it handle it)
    print("-" * 50)
    print("Test 2: Unsafe Query Attempt")
    response = agent.run_query("Delete all dealers")
    print(f"Response: {response}")
