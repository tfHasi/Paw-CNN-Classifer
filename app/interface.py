import os
import sys
import streamlit as st
from PIL import Image
import tempfile
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chatbot import DogBreedChatbot
MODEL_PATH = "Models/Paw Detector Final Model.keras"
LABELS_PATH = "Dataset/labels.csv"

st.set_page_config(
    page_title="Paw Detector!",
    page_icon="🐾",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chatbot" not in st.session_state:
    from dotenv import load_dotenv
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    try:
        st.session_state.chatbot = DogBreedChatbot(
            model_path=MODEL_PATH,
            labels_path=LABELS_PATH,
            groq_api_key=groq_api_key
        )
        st.session_state.initialization_error = None
    except Exception as e:
        st.session_state.initialization_error = str(e)

# App header
st.title("🐾 Paw Detective at your service!🐾")
st.markdown("""
Upload your cute doggie's image and our AI agents will help you identify the breed and provide information!
""")

if "initialization_error" in st.session_state and st.session_state.initialization_error:
    st.error(f"Error initializing the application: {st.session_state.initialization_error}")
    st.stop()

# Sidebar for image upload
with st.sidebar:
    with st.container():
        uploaded_file = st.file_uploader("Upload Your Dog's Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            # Save image to temporary file
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, f"dog_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            image.save(image_path)
            if st.button("Identify Breed", key="identify_button"):
                st.session_state.messages.append({"role": "user", "content": "What breed is this dog?", "image": image})
                with st.spinner("Identifying the curious paw..."):
                    try:
                        # Process the image using our chatbot
                        response = st.session_state.chatbot.process_message("What breed is this dog?", image_path)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error processing image: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.rerun()

# Main chat interface
st.header("Chat with Paw Detector")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "image" in message:
                st.image(message["image"], width=300)
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the identified breed or upload a new image..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.chatbot.process_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    st.rerun()

# App footer
st.markdown("---")
st.markdown("""
### How to use the Paw Detector:
1. **Upload** your dog's picture using the sidebar
2. Click **Identify Breed** to analyze the image
3. **Ask questions** about the identified breed
4. Upload a **new image** anytime to identify another breed

### About Paw Detector
Paw Detector uses a specialized deep learning model (MobileNetV2) to identify dog breeds and provides detailed information about each breed's characteristics, temperament, care requirements, and more!
""")