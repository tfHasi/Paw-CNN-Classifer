import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = os.path.join(BASE_DIR, "Models")

PAW_DETECTOR_MODEL = os.path.join(MODELS_DIR, "Paw Detector Final Model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")

DEFAULT_MODEL_NAME = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.2