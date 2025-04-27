import os
import re
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from agents.paw_predictor_agent import PawPredictorAgent
from agents.paw_retriever_agent import PawRetrieverAgent
from config import PAW_DETECTOR_MODEL, LABELS_PATH

class DogBreedChatbot:
    def __init__(self, api_key=None, model_name=None, temperature=None,
                 model_path=None, labels_path=None):
        self.api_key = api_key
        
        self.predictor_agent = PawPredictorAgent(
            api_key=self.api_key,
            model_name=model_name,
            temperature=temperature,
            model_path=model_path or PAW_DETECTOR_MODEL,
            labels_path=labels_path or LABELS_PATH
        )
        
        self.retriever_agent = PawRetrieverAgent(
            api_key=self.api_key,
            model_name=model_name,
            temperature=temperature
        )
        
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
                formatted_breed = self._format_breed_name(self.context["current_breed"])
                return self._get_breed_info(formatted_breed)
            if self._is_help_request(message):
                return self._get_help_message()
            return self._get_default_response()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _format_breed_name(self, breed_name: str) -> str:
        return " ".join(word.capitalize() for word in breed_name.split("_"))

    def _process_image(self, image_path: str, message: str) -> str:
        prediction_result = self.predictor_agent.run(f"Identify the dog breed in: {image_path}")
        breed_name = self._extract_breed_from_prediction(prediction_result)
        confidence_str = self._extract_confidence_from_prediction(prediction_result)
        confidence_value = self._extract_confidence_value(prediction_result)
        alternatives = []
        alt_matches = re.findall(r"Alternative \d+: ([A-Za-z_]+) \(([0-9.]+)%\)", prediction_result)
        for alt_breed, alt_conf in alt_matches:
            if float(alt_conf) > 20:  # Only include alternatives with >20% confidence
                alternatives.append((alt_breed, float(alt_conf)))
        
        if breed_name:
            self.context["current_breed"] = breed_name
            formatted_breed = self._format_breed_name(breed_name)
            response = f"🐾 Breed Identification Results\n\n"
            response += f"I've identified this cutie as a **{formatted_breed}**{confidence_str}!\n\n"
            
            # Add alternatives if confidence is low
            if confidence_value < 70 and alternatives:
                response += "Other possible breeds:\n"
                for alt_breed, alt_conf in alternatives:
                    formatted_alt = self._format_breed_name(alt_breed)
                    response += f"- {formatted_alt} ({alt_conf:.2f}%)\n"
                response += "\n"
                if len(alternatives) > 0:
                    response += "Learn more about the primary prediction or one of the alternatives? Just say which breed you're interested in.\n\n"
            
            if self._is_breed_inquiry(message):
                breed_info = self._get_breed_info(formatted_breed)
                return f"{response}\n\n{breed_info}"
            else:
                return f"{response}Would you like to learn more about {formatted_breed}s? Just ask!"
        else:
            return "I couldn't identify a dog breed in this image. Please try another image with a clearer view of the dog."

    def _extract_confidence_value(self, prediction_result: str) -> float:
        confidence_match = re.search(r"Confidence: ([0-9.]+)%", prediction_result)
        if confidence_match:
            return float(confidence_match.group(1))
        return 0.0

    def _extract_breed_from_prediction(self, prediction_result: str) -> Optional[str]:
        breed_match = re.search(r"Breed: ([A-Za-z_]+)", prediction_result)
        if breed_match:
            return breed_match.group(1).strip()
        breed_match = re.search(r"is a ([A-Za-z_]+)( with| and|\.)", prediction_result)
        if breed_match:
            return breed_match.group(1).strip()
        return None
    
    def _extract_confidence_from_prediction(self, prediction_result: str) -> str:
        confidence_match = re.search(r"Confidence: ([0-9.]+)%", prediction_result)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            if confidence < 70:
                return f" (low confidence: {confidence:.2f}%)"
            return f" ({confidence:.2f}%)"
        return ""
    
    def _get_breed_info(self, breed_name: str) -> str:
        api_breed_name = breed_name.lower().replace(" ", "_")
        breed_info = self.retriever_agent.run(f"Tell me about {api_breed_name}")
        final_answer_match = re.search(r"Final Answer:(.*?)$", breed_info, re.DOTALL)
        if final_answer_match:
            breed_info = final_answer_match.group(1).strip()
        breed_info = re.sub(r"Thought:.*?Action:", "", breed_info, flags=re.DOTALL)
        breed_info = re.sub(r"Action Input:.*?Observation:", "", breed_info, flags=re.DOTALL)
        
        return f"🦮 About {breed_info}"
    
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