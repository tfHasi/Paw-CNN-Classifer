import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dotenv import load_dotenv
from agents.paw_predictor_agent import PawPredictorAgent

def main():
    load_dotenv()
    agent = PawPredictorAgent(
        model_path="Models/Paw Detector Final Model.keras",
        labels_path="Dataset/labels.csv",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )   
    test_image_path = "Dataset/golden_retriever.jpg"
    response = agent.run(f"Can you identify the dog breed in this image: {test_image_path}")
    print(response)
if __name__ == "__main__":
    main()