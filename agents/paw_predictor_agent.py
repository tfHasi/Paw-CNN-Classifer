from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from typing import List
import os
from tools.paw_predictor_tool import PawPredictorTool

class PawPredictorAgent:
    def __init__(self, model_path="models/Paw_Detector_Final_Model.keras", 
                 labels_path="Dataset/labels.csv",
                 groq_api_key=None):
        # Set up Groq API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif "GROQ_API_KEY" not in os.environ:
            raise ValueError("Groq API key must be provided or set as environment variable")
        
        # Initialize the breed predictor tool
        self.predictor_tool = PawPredictorTool(model_path, labels_path)
        
        # Create the LangChain tool
        self.tools = [
            Tool(
                name="PawPredictor",
                func=self._predict_breed,
                description="Predicts the breed of a dog from an image file path. Input should be a valid path to an image file."
            )
        ]
        
        # Initialize the LLM
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2
        )
        
        # Create the agent
        prompt = self._create_prompt()
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _predict_breed(self, img_path: str) -> str:
        if not os.path.exists(img_path):
            return f"Error: Image file not found at {img_path}"
        
        result = self.predictor_tool.predict_breed(img_path)
        if "error" in result:
            return f"Error during prediction: {result['error']}"
        response = f"Breed: {result['breed']}\nConfidence: {result['confidence']:.2%}\n"
        if not result['is_reliable']:
            response += "Note: This prediction has low confidence. Alternative possibilities include:\n"
            for alt in result['alternatives']:
                response += f"- {alt['breed']} ({alt['confidence']:.2%})\n"
        
        return response
    
    def _create_prompt(self) -> PromptTemplate:
        template = """You are an AI assistant specialized in dog breed identification.
Your primary task is to help users identify dog breeds from images they provide.

When a user provides an image path, use the PawPredictor tool to analyze the image and identify the dog breed.
Present the results clearly, including the confidence level and any alternative possibilities if the prediction has low confidence.

If the user asks questions about a specific dog breed after identification, inform them that you'll need to use web searching capabilities to provide detailed information.

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
