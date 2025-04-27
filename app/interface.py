import os
import sys
import streamlit as st
from PIL import Image
import tempfile
import datetime
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from app.chatbot import DogBreedChatbot
from config import PAW_DETECTOR_MODEL, LABELS_PATH

TEMP_DIR = tempfile.gettempdir()

def initialize_app():
    st.set_page_config(
        page_title="Paw Detector",
        page_icon="🐾",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
        <style>
            .main .block-container {
                max-width: 100%;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            .stApp {
                max-width: 100%;
            }
            .stChatMessage {
                max-width: 100%;
            }
            .stImage {
                max-width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []   
    if "chatbot" not in st.session_state:
        initialize_chatbot()
    
    if "initialization_error" in st.session_state and st.session_state.initialization_error:
        st.error(f"Error initializing the application: {st.session_state.initialization_error}")
        st.stop()

def initialize_chatbot():
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("GROQ_API_KEY")
        st.session_state.chatbot = DogBreedChatbot(
            api_key=api_key,
            model_path=PAW_DETECTOR_MODEL,
            labels_path=LABELS_PATH
        )
        st.session_state.initialization_error = None
    except Exception as e:
        st.session_state.initialization_error = str(e)

def create_sidebar():
    with st.sidebar:
        st.title("🐾 Paw Detective")
        with st.expander("ℹ️ Help & Information", expanded=True):
            st.markdown("""
            ### How to use Paw Detector:
            1. **Upload** your dog's picture using the form below
            2. Click **Identify Breed** to analyze the image
            3. **Ask questions** about the identified breed
            4. Upload a **new image** anytime to identify another breed
            
            ### About Paw Detector
            Paw Detector uses a specialized deep learning model named MobileNetV2 to identify dog breeds and provides detailed information about each breed's characteristics, temperament, care requirements, and more!
            """)

def process_image(image, image_path):
    st.session_state.messages.append({
        "role": "user",
        "content": "What breed is this dog?",
        "image": image
    })
    
    with st.spinner("Identifying the curious paw..."):
        try:
            response = st.session_state.chatbot.process_message("What breed is this dog?", image_path)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

def create_chat_interface():
    # Create a container for the header
    header_container = st.container()
    with header_container:
        st.markdown("<h1 style='text-align: center;'><b>🐾 Paw Detective at your service 🐾</b></h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Upload your cute doggie's image and our AI agents will identify the breed for you!</p>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Upload Your Dog's Image</h3>", unsafe_allow_html=True)
    
    upload_container = st.container()
    with upload_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                image_path = os.path.join(TEMP_DIR, f"dog_image_{timestamp}.jpg")
                image.save(image_path)
                if st.button("Identify Breed", key="identify_button", use_container_width=True):
                    process_image(image, image_path)
    
    chat_container = st.container()
    with chat_container:
        message_area = st.container()
        with message_area:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if "image" in message:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(message["image"], width=400)
                    st.markdown(message["content"])
    
    input_container = st.container()
    with input_container:
        if prompt := st.chat_input("Woof! Woof! Woof!"):
            process_chat_message(prompt)

def process_chat_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.chatbot.process_message(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    st.rerun()

def main():
    initialize_app()
    create_sidebar()
    col1, col2 = st.columns([5, 1])
    with col1:
        create_chat_interface()
    with col2:
        pass

if __name__ == "__main__":
    main()