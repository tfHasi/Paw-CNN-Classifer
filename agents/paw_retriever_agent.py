from langchain.agents import Tool
from .base_agent import PawAgent
from tools.paw_retriever_tool import PawRetrieverTool
from prompts import get_retriever_prompt

class PawRetrieverAgent(PawAgent):
    def __init__(self, groq_api_key=None):
        super().__init__(groq_api_key)
        self.retriever_tool = PawRetrieverTool()
        self.tools = [
            Tool(
                name="PawRetriever",
                func=self._retrieve_breed_info,
                description="Retrieves detailed information about a specific dog breed. Input should be the name of the dog breed."
            )
        ]
        prompt = get_retriever_prompt()
        self.agent_executor = self._create_agent(self.tools, prompt)
    
    def _retrieve_breed_info(self, breed_name: str) -> str:
        if not breed_name or len(breed_name) < 2:
            return "Error: Please provide a valid dog breed name"
        
        result = self.retriever_tool.scrape_breed_info(breed_name)
        if not result["success"] or len(result["content"]) == 0:
            return f"Error retrieving information for {breed_name}. {result.get('error', '')}"
        return self._format_breed_info(result["content"])
    
    def _format_breed_info(self, content_dict):
        formatted_text = ""

        for source, data in content_dict.items():
            formatted_text += f"===== SOURCE: {source.upper()} =====\n\n"
            if "general_info" in data and data["general_info"]:
                formatted_text += "GENERAL INFO:\n"
                for key, value in data["general_info"].items():
                    formatted_text += f"- {key}: {value}\n"
                formatted_text += "\n"
            for section in ["temperament", "health", "history", "care"]:
                if section in data and data[section]:
                    formatted_text += f"{section.upper()}:\n{data[section]}\n\n"
        
        return formatted_text