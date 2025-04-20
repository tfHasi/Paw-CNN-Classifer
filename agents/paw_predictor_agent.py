from langchain.agents import Tool
import os
from .base_agent import PawAgent
from tools.paw_predictor_tool import PawPredictorTool
from prompts import get_predictor_prompt

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
        prompt = get_predictor_prompt()
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
            for i, alt in enumerate(result['alternatives'], 1):
                response += f"Alternative {i}: {alt['breed']} ({alt['confidence']:.2%})\n"
        return response