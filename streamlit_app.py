import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import cv2
import os
import gdown

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Page configuration
st.set_page_config(
    page_title="Bird vs Drone Classifier",
    page_icon="🦅",
    layout="wide"
)

# Title and description
st.title("🦅 Bird vs Drone Classification & Detection")
st.markdown("""### AI-Powered Aerial Object Recognition System
**Upload an image to classify whether it contains a Bird or Drone!**
---
""")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Custom CNN", "ResNet50", "MobileNet", "YOLOv8 Detection"]
)

# ---------------- Model loading functions ---------------- #

@st.cache_re_
