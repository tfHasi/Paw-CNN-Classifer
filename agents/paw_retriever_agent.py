from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from tools.paw_retriever_tool import PawRetrieverTool

class PawRetrieverAgent:
    def __init__(self, groq_api_key=None):
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif "GROQ_API_KEY" not in os.environ:
            raise ValueError("Groq API key must be provided or set as environment variable")
        self.retriever_tool = PawRetrieverTool()
        self.tools = [
            Tool(
                name="PawRetriever",
                func=self._retrieve_breed_info,
                description="Retrieves detailed information about a specific dog breed. Input should be the name of the dog breed."
            )
        ]
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        prompt = self._create_prompt()
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
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
    
    def run(self, query: str) -> str:
        return self.agent_executor.invoke({"input": query})["output"]