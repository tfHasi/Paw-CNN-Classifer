import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from dotenv import load_dotenv
from agents.paw_predictor_agent import PawPredictorAgent
from agents.paw_retriever_agent import PawRetrieverAgent

def main():
    load_dotenv()
    # Initialize agents
    predictor = PawPredictorAgent(
        model_path="Models/Paw Detector Final Model.keras",
        labels_path="Dataset/labels.csv",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    retriever = PawRetrieverAgent(groq_api_key=os.getenv("GROQ_API_KEY"))
    
    image_path = "Dataset/golden_retriever.jpg"
    prediction = predictor.run(f"Identify the dog breed in: {image_path}")
    breed = prediction.split("is a ")[1].split(" with")[0].strip() if "is a " in prediction else None
    print(prediction)
    if breed:
        print("\nBREED INFORMATION:")
        print(retriever.run(f"Tell me about {breed}"))

if __name__ == "__main__":
    main()