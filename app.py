import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- Page Setup ---
st.set_page_config(
    page_title="Latent TB Research",
    page_icon="🧬",
    layout="centered"
)

# --- 🎨 MODERN "DARK MEDICAL" THEME (CSS) ---
st.markdown("""
    <style>
    /* 1. Force a Dark Professional Background */
    .stApp {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    /* 2. Trendy "Glassmorphism" Card for Title & Author */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
    }

    /* 3. Text Styling inside the Glass Card */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #ffffff !important; /* Force White */
        margin-bottom: 10px;
    }
    
    .author-text {
        font-size: 18px;
        color: #e0e0e0 !important; /* Light Grey */
        margin-top: 10px;
    }

    .highlight-text {
        color: #4facfe; /* Neon Blue for Name */
        font-weight: bold;
    }

    /* 4. Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px; /* Rounded Pill Shape */
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('jitsa_tb_model.keras')
    return model

# --- HEADER SECTION (Glass Card) ---
st.markdown("""
    <div class="glass-card">
        <div class="main-title">Detection and Diagnosis of Latent Tuberculosis (TB) in Patients using Machine Learning and AI</div>
        <div style="font-size: 16px; color: #b0c4de;">Using Deep Learning & AI CNN Architecture</div>
        <hr style="border-color: rgba(255,255,255,0.2);">
        <div class="author-text">
            Developed By <span class="highlight-text">Jitendra Prajapat</span><br>
            <span style="font-size: 14px; opacity: 0.8;">Suresh Gyan Vihar University</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1️⃣ Upload X-Ray")
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file)
        st.image(image, caption='Patient Scan', use_container_width=True)

with col2:
    st.markdown("### 2️⃣ AI Diagnosis")
    
    if file is not None:
        if st.button("Initialize Detection Protocol"):
            
            with st.spinner('Processing Neural Network Layers...'):
                # Load & Preprocess
                model = load_model()
                size = (180, 180)
                image_ops = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(image_ops).astype(np.float32)
                
                if len(img_array.shape) == 2:
                     img_array = np.stack((img_array,)*3, axis=-1)
                
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                score = tf.nn.softmax(prediction[0])
                
                class_names = ['Normal', 'Tuberculosis']
                result = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

            # --- RESULT CARD (Modern Popup) ---
            if result == "Tuberculosis":
                st.markdown(f"""
                <div style="background-color: rgba(255, 75, 75, 0.2); border: 1px solid #ff4b4b; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #ff4b4b; margin:0;">⚠️ POSITIVE DETECTED</h2>
                    <p style="color: white;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(0, 200, 83, 0.2); border: 1px solid #00c853; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: #00c853; margin:0;">✅ NEGATIVE (HEALTHY)</h2>
                    <p style="color: white;">Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            
    else:
        st.info("Waiting for X-Ray input...")
        st.markdown("""
        <div style="font-size: 12px; color: grey; margin-top: 20px;">
        System Status: <span style="color: #00ff00;">● Online</span><br>
        Model Version: Jitsa_v1.0 (Research Build)
        </div>
        """, unsafe_allow_html=True)