import os
import re
from typing import Dict, Any, List, Optional, Tuple

from agents.paw_predictor_agent import PawPredictorAgent
from agents.paw_retriever_agent import PawRetrieverAgent

class DogBreedChatbot:
    def __init__(self, 
                 model_path: str = "Models/Paw Detector Final Model.keras",
                 labels_path: str = "Dataset/labels.csv",
                 groq_api_key: Optional[str] = None):
        if groq_api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
        self.predictor_agent = PawPredictorAgent(
            model_path=model_path,
            labels_path=labels_path,
            groq_api_key=groq_api_key
        )
        self.retriever_agent = PawRetrieverAgent(groq_api_key=groq_api_key)
        self.context = {
            "current_breed": None,
            "current_image": None,
            "history": []
        }
    
    def process_message(self, message: str, image_path: Optional[str] = None) -> str:
        self.context["history"].append({"role": "user", "content": message})
        try:
            if image_path and os.path.exists(image_path):
                self.context["current_image"] = image_path
                return self._process_image(image_path, message)
            if self.context["current_breed"] and self._is_breed_inquiry(message):
                return self._get_breed_info(self.context["current_breed"])
            if self._is_help_request(message):
                return self._get_help_message()
            return self._get_default_response()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _process_image(self, image_path: str, message: str) -> str:
        prediction_result = self.predictor_agent.run(f"Identify the dog breed in: {image_path}")
        breed_name = self._extract_breed_from_prediction(prediction_result)
        confidence_str = self._extract_confidence_from_prediction(prediction_result)
        
        if breed_name:
            self.context["current_breed"] = breed_name
            response = f"🐾 Breed Identification Results\n\n"
            response += f"I've identified this cutie as a **{breed_name}**{confidence_str}!\n\n"
            if self._is_breed_inquiry(message):
                breed_info = self._get_breed_info(breed_name)
                return f"{response}\n\n{breed_info}"
            else:
                return f"{response}Would you like to learn more about {breed_name}s? Just ask!"
        else:
            return "I couldn't identify a dog breed in this image. Please try another image with a clearer view of the dog."
    
    def _extract_breed_from_prediction(self, prediction_result: str) -> Optional[str]:
        breed_match = re.search(r"Breed: ([A-Za-z ]+)", prediction_result)
        if not breed_match:
            breed_match = re.search(r"is a ([A-Za-z ]+)( with| and|\.)", prediction_result)
        
        return breed_match.group(1).strip() if breed_match else None
    
    def _extract_confidence_from_prediction(self, prediction_result: str) -> str:
        confidence_match = re.search(r"Confidence: ([0-9.]+%)", prediction_result)
        return f" ({confidence_match.group(1)})" if confidence_match else ""
    
    def _get_breed_info(self, breed_name: str) -> str:
        breed_info = self.retriever_agent.run(f"Tell me about {breed_name}")
        return f" 🦮 About the {breed_name}\n\n{breed_info}"
    
    def _is_breed_inquiry(self, message: str) -> bool:
        inquiry_phrases = [
            "tell me more", "more info", "learn more", "yes", 
            "tell me about", "information", "details", "characteristics",
            "what can you tell me", "breed info", "about this breed"
        ]
        message = message.lower()
        return any(phrase in message for phrase in inquiry_phrases)
    
    def _is_help_request(self, message: str) -> bool:
        help_phrases = ["help", "can you", "what", "how to", "guide", "instructions"]
        message = message.lower()
        return any(phrase in message for phrase in help_phrases)
    
    def _get_help_message(self) -> str:
        return (
            "## 🐾 Paw Detector Help\n\n"
            "Here's how you can use the Paw Detector:\n\n"
            "1. Upload your cute doggie's image, and I'll identify the breed\n"
            "2. Ask for more information about the identified breed\n"
            "3. You can upload a new image at any time\n\n"
            "Example questions I can answer about identified breeds:\n"
            "- Tell me more about this breed\n"
            "- What are the care requirements?\n"
            "- Is this breed good with children?\n"
            "- What is the history of this breed?"
        )
    
    def _get_default_response(self) -> str:
        return "Woof! Paw Detector at your service! Provide me with your cute doggie's picture and I'll identify their breed for you."