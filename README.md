# 🐶 Paw Detective

A simple multi-agent LangChain application that identifies dog breeds from images and retrieves detailed information about the detected breed.

## How It Works

1. **Image Inference Agent**  
   - Uses a CNN model to predict the dog breed from an uploaded image.

2. **Information Retrieval Agent**  
   - Based on the predicted breed, retrieves relevant breed-specific information from curated sources.

3. **LLM Output**  
   - Combines the retrieved content to generate a natural language description of the breed.

## Example Flow

1. Upload a dog photo.
2. Model predicts: `Golden Retriever`
3. System responds:
   > "This is likely a Golden Retriever — a friendly, intelligent breed known for its loyalty and ease of training."

## 🔧 Tech Stack

- CNN (TensorFlow) MobileNetV2 architecture for breed classification
- LangChain for multi-agent orchestration
- LLM (llama) for final response generation

## Features

- Lightweight and easy to extend
- RAG-style architecture using multi-agent design
- Plug-and-play image classification + retrieval

---

Feel free to clone, modify, or integrate into your own AI assistant projects!
