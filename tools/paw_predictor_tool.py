import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import os

class PawPredictorTool:
    def __init__(self, model_path, labels_path):
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        # Load labels and create class mapping
        if os.path.exists(labels_path):
            self.labels = pd.read_csv(labels_path)
            self._create_class_mapping()
        else:
            raise FileNotFoundError(f"Labels file not found at {labels_path}")
        self.IMG_SIZE = (224, 224)
    
    def _create_class_mapping(self):
        unique_breeds = self.labels['breed'].unique()
        self.class_indices = {breed: i for i, breed in enumerate(sorted(unique_breeds))}
        self.inv_class_indices = {v: k for k, v in self.class_indices.items()}
    
    def predict_breed(self, img_path, confidence_threshold=0.7):
        try:
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=self.IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = preprocess_input(np.expand_dims(img_array, axis=0))
            
            # Make prediction
            preds = self.model.predict(img_array)[0]
            
            # Get the top prediction
            top_index = np.argmax(preds)
            top_breed = self.inv_class_indices[top_index]
            top_confidence = float(preds[top_index])
            
            # Get alternative predictions (if confidence is low)
            is_reliable = top_confidence >= confidence_threshold
            alternatives = []
            
            if not is_reliable:
                # Get 2nd and 3rd most likely breeds
                top_indices = np.argsort(preds)[-3:][::-1]
                alternatives = [
                    {"breed": self.inv_class_indices[i], 
                     "confidence": float(preds[i])}
                    for i in top_indices[1:3]  # Skip the first one (top prediction)
                    if preds[i] > 0.1  # Only include if confidence > 10%
                ]
            
            return {
                "breed": top_breed,
                "confidence": top_confidence,
                "is_reliable": is_reliable,
                "alternatives": alternatives
            }
            
        except Exception as e:
            return {"error": str(e)}