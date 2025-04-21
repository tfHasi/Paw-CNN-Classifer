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

if __name__ == "__main__":
    main()