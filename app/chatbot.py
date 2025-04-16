import os
import sys
import re
from typing import Dict, Any, List, Optional, Tuple

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.paw_predictor_agent import PawPredictorAgent
from agents.paw_retriever_agent import PawRetrieverAgent
from tempfile import NamedTemporaryFile

class DogBreedChatbot:
    def __init__(self, 
                 model_path: str = "Models/Paw Detector Final Model.keras",
                 labels_path: str = "Dataset/labels.csv",
                 groq_api_key: Optional[str] = None):
        if groq_api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
        # Initialize agents
        self.predictor = PawPredictorAgent(
            model_path=model_path,
            labels_path=labels_path,
            groq_api_key=groq_api_key
        )
        self.retriever = PawRetrieverAgent(groq_api_key=groq_api_key)
        # Chat context to maintain conversation state
        self.context = {
            "current_breed": None,
            "current_image": None,
            "history": []
        }
    
    def process_message(self, message: str, image_path: Optional[str] = None) -> str:
        """Process user messages and image uploads"""
        # Add to history
        self.context["history"].append({"role": "user", "content": message})
        
        # Handle image upload
        if image_path and os.path.exists(image_path):
            self.context["current_image"] = image_path
            return self._process_image(image_path, message)
            
        # If asking about the detected breed
        if self.context["current_breed"] and self._is_breed_inquiry(message):
            return self._get_breed_info(self.context["current_breed"])
            
        # General chat about dogs
        if "help" in message.lower() or "can you" in message.lower() or "what" in message.lower():
            return self._get_help_message()
            
        # Default response
        return "Woof! Paw Detector at your service! Provide me with your cute doggie's picture and I'll identify their breed for you."
    
    def _process_image(self, image_path: str, message: str) -> str:
        """Process image using the PawPredictorAgent"""
        try:
            # Call the predictor agent with the image path
            prediction_result = self.predictor.run(f"Identify the dog breed in: {image_path}")
            
            # Extract breed from prediction result using regex pattern matching
            breed_match = re.search(r"Breed: ([A-Za-z ]+)", prediction_result)
            confidence_match = re.search(r"Confidence: ([0-9.]+%)", prediction_result)
            
            # If no explicit breed found, try alternative pattern matching
            if not breed_match:
                breed_match = re.search(r"is a ([A-Za-z ]+)( with| and|\.)", prediction_result)
            
            if breed_match:
                breed_name = breed_match.group(1).strip()
                self.context["current_breed"] = breed_name
                
                confidence_str = ""
                if confidence_match:
                    confidence_str = f" ({confidence_match.group(1)})"
                
                # Format the response for user
                response = f"## 🐾 Breed Identification Results\n\n"
                response += f"I've identified this cutie as a **{breed_name}**{confidence_str}!\n\n"
                
                # Check if user wants breed info right away
                if self._is_breed_inquiry(message):
                    breed_info = self._get_breed_info(breed_name)
                    return f"{response}\n\n{breed_info}"
                else:
                    return f"{response}Would you like to learn more about {breed_name}s? Just ask!"
            else:
                return "I couldn't identify a dog breed in this image. Please try another image with a clearer view of the dog."
        except Exception as e:
            return f"An error occurred while processing the image: {str(e)}"
    
    def _get_breed_info(self, breed_name: str) -> str:
        """Get detailed information about a specific breed"""
        try:
            breed_info = self.retriever.run(f"Tell me about {breed_name}")
            return f"## 🦮 About the {breed_name}\n\n{breed_info}"
        except Exception as e:
            return f"Sorry, I couldn't retrieve information about {breed_name}: {str(e)}"
    
    def _is_breed_inquiry(self, message: str) -> bool:
        """Check if message is asking about breed information"""
        inquiry_phrases = [
            "tell me more", "more info", "learn more", "yes", 
            "tell me about", "information", "details", "characteristics",
            "what can you tell me", "breed info", "about this breed"
        ]
        message = message.lower()
        return any(phrase in message for phrase in inquiry_phrases)
    
    def _get_help_message(self) -> str:
        """Provide help information to the user"""
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