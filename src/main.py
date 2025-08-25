import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image


# --- Load the trained model ---
@st.cache_resource
def load_plant_model():
    model = load_model("inceptionv3_plant_disease_model.keras")
    return model


model = load_plant_model()

IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# --- Page Config ---
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered",
)

# --- Custom CSS inspired by infographic ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* Apply font + background to entire app */
    html, body, [class*="stApp"] {
        font-family: 'Poppins', sans-serif !important;
        background-color: #E6F0D5 !important;  /* soft pastel green */
        color: #2E5339 !important;  /* deep earthy green */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        color: #2E5339 !important;
        font-weight: 600 !important;
    }

    /* Text */
    p, div, span {
        font-family: 'Poppins', sans-serif !important;
        color: #2E5339 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #5C8A57 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6em 1.2em !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        background-color: #4E7348 !important;
        color: #fff !important;
    }

    /* Cards/containers */
    .box {
        background-color: #FFFFFF !important;
        padding: 1.5em !important;
        border-radius: 18px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
        margin-bottom: 1.5em !important;
    }

    /* Expander header styling - Fixed overlapping issue */
    .streamlit-expanderHeader {
    }

    /* Expander container */
    .streamlit-expander {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
        margin-top: 1em !important;
        margin-bottom: 1em !important;
        padding: 0.8em !important;
    }

    /* File uploader label */
    .stFileUploader label {
        font-weight: 800 !important;
        text-align: center !important;
        display: block !important;
        color: #2E5339 !important;
        font-size: 1.1rem !important;
    }

    /* Section divider */
    .divider {
        height: 1px !important;
        background-color: #B8D4A8 !important;
        margin: 2rem 0 !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header ---
st.title("🌱 Plant Disease Detection")
st.write("Identify and monitor plant health using AI-powered image classification.")

# --- Project Info Section ---
st.subheader("📖 About this Project")
st.markdown(
    """
    This application helps farmers, gardeners, and researchers quickly **identify plant diseases** 
    from leaf images using a deep learning model based on **InceptionV3**.  

    ### Why this matters:
    - **Early detection** reduces crop loss  
    - **Accurate identification** avoids misdiagnosis  
    - **AI-driven solution** saves time and resources  

    ### How it works:
    1. Upload a photo of a plant leaf.  
    2. The model predicts the most likely disease.  
    3. Confidence scores help guide trust in the prediction.  
    """
)

# --- Divider ---
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# --- Upload + Prediction ---
uploaded_file = st.file_uploader(
    "**📤 Upload a leaf image...**", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    # Add divider after image
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Preprocess the image
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    # Show result
    st.subheader("🔍 Prediction Result")
    st.write(
        f"**Disease:** <span class='highlight'>{predicted_class_name}</span>",
        unsafe_allow_html=True,
    )
    st.write(f"**Confidence:** {confidence:.2f} %")
