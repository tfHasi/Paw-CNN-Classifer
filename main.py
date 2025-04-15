import os
from dotenv import load_dotenv
from agents.paw_predictor_agent import PawPredictorAgent

load_dotenv()

# Initialize the agent
agent = PawPredictorAgent(
    model_path="Models/Paw Detector Final Model.keras",
    labels_path="Dataset/labels.csv",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

test_image_path = "Dataset/golden_retriever.jpg"  

# Run the agent with a user query 
response = agent.run(f"Can you identify the dog breed in this image: {test_image_path}")
print(response)