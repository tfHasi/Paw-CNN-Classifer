import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import os

class PawPredictorTool:
    def __init__(self, model_path, labels_path):
        # Load model and labels
        self.model = tf.keras.models.load_model(model_path)
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
            
        self.labels = pd.read_csv(labels_path)
        self._create_class_mapping()
        self.IMG_SIZE = (224, 224)
    
    def _create_class_mapping(self):
        unique_breeds = self.labels['breed'].unique()
        self.class_indices = {breed: i for i, breed in enumerate(sorted(unique_breeds))}
        self.inv_class_indices = {v: k for k, v in self.class_indices.items()}
    
    def predict_breed(self, img_path, confidence_threshold=0.7):
        try:
            # Preprocess image
            img = image.load_img(img_path, target_size=self.IMG_SIZE)
            img_array = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
            
            # Get predictions
            preds = self.model.predict(img_array)[0]
            top_indices = np.argsort(preds)[-3:][::-1]  # Top 3 predictions
            
            # Format results
            top_breed = self.inv_class_indices[top_indices[0]]
            top_confidence = float(preds[top_indices[0]])
            is_reliable = top_confidence >= confidence_threshold
            
            # Get alternatives if confidence is low
            alternatives = [
                {"breed": self.inv_class_indices[i], "confidence": float(preds[i])}
                for i in top_indices[1:] if preds[i] > 0.2
            ]
            
            return {
                "breed": top_breed,
                "confidence": top_confidence,
                "is_reliable": is_reliable,
                "alternatives": alternatives
            }
            
        except Exception as e:
            return {"error": str(e)}