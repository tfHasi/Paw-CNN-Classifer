from langchain.agents import Tool
from langchain.prompts import PromptTemplate
import os
from .base_agent import PawAgent
from tools.paw_predictor_tool import PawPredictorTool

class PawPredictorAgent(PawAgent):
    def __init__(self, model_path="models/Paw_Detector_Final_Model.keras", 
                 labels_path="Dataset/labels.csv",
                 groq_api_key=None):
        super().__init__(groq_api_key)
        self.predictor_tool = PawPredictorTool(model_path, labels_path)
        self.tools = [
            Tool(
                name="PawPredictor",
                func=self._predict_breed,
                description="Predicts the breed of a dog from an image file path. Input should be a valid path to an image file."
            )
        ]
        prompt = self._create_prompt()
        self.agent_executor = self._create_agent(self.tools, prompt)
    
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