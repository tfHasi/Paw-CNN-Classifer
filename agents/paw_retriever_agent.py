# agents/paw_retriever_agent.py
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from .base_agent import PawAgent
from tools.paw_retriever_tool import PawRetrieverTool

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
        prompt = self._create_prompt()
        self.agent_executor = self._create_agent(self.tools, prompt)
    
    def _retrieve_breed_info(self, breed_name: str) -> str:
        if not breed_name or len(breed_name) < 2:
            return "Error: Please provide a valid dog breed name"
        
        result = self.retriever_tool.scrape_breed_info(breed_name)
        if not result["success"] or len(result["content"]) == 0:
            return f"Error retrieving information for {breed_name}. {result.get('error', '')}"
        
        return str(result["content"])

    def _create_prompt(self) -> PromptTemplate:
        template = """You are a dog breed information specialist.
Your task is to provide detailed, well-organized information about dog breeds based on web data.

When a user asks about a specific dog breed, follow these steps:
1. Use the DogBreedInfoRetriever tool to gather information about the breed
2. Summarize and organize the information into these sections:
   - General Characteristics
   - Temperament and Personality
   - Care Requirements
   - Health Considerations
   - History and Background

Make your response conversational, informative, and helpful for potential dog owners.
Present the information clearly, and mention when sources disagree about particular characteristics.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""   
        return PromptTemplate.from_template(template)