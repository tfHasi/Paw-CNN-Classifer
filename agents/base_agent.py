from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
import streamlit as st
from config import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE

class PawAgent:
    def __init__(self, api_key=None, model_name=None, temperature=None):
        self.api_key = api_key or st.secrets.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key must be provided or set in st.secrets")
            
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=model_name or DEFAULT_MODEL_NAME,
            temperature=temperature or DEFAULT_TEMPERATURE
        )
        
        self.tools = []
        self.agent_executor = None
    
    def _create_agent(self, tools, prompt_template):
        agent = create_react_agent(self.llm, tools, prompt_template)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        if not self.agent_executor:
            raise ValueError("Agent executor not initialized. Make sure to call setup_agent() in the child class.")
        try:
            response = self.agent_executor.invoke({"input": query})
            if "output" in response:
                final_answer_match = re.search(r"Final Answer:(.*?)$", response["output"], re.DOTALL)
                if final_answer_match:
                    return final_answer_match.group(1).strip()
                return response["output"]
            return "No response generated."
        except Exception as e:
            return f"Error during execution: {str(e)}"