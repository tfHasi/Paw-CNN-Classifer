from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
import re

class PawAgent:
    def __init__(self, groq_api_key=None):
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif "GROQ_API_KEY" not in os.environ:
            raise ValueError("Groq API key must be provided or set as environment variable")
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2
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