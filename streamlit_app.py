import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from PIL import Image
import gdown

# Try YOLO import (optional detection part)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # Will be None on Streamlit Cloud (ultralytics too heavy)

# ---------------- Paths & constants ---------------- #

BASE_DIR = Path(".")
CNN_MODEL_PATH = BASE_DIR / "custom_cnn_model.h5"   # Custom CNN weights
RESNET_WEIGHTS = BASE_DIR / "resnet_weights.h5"     # Optional fine-tuned weights
MOBILENET_WEIGHTS = BASE_DIR / "mobilenet_weights.h5"
YOLO_WEIGHTS = BASE_DIR / "yolov8n.pt"              # YOLOv8n weights (auto-download)

# Google Drive ID for custom_cnn_model.h5
CNN_DRIVE_ID = "1q7CkuixuGhfauILzP3QBHJvSmf5w2TGa"

IMG_SIZE = (224, 224)

# ---------------- Streamlit page config ---------------- #

st.set_page_config(
    page_title="Bird vs Drone Classifier & Detector",
    page_icon="🦅",
    layout="wide",
)

st.title("🦅 Aerial Object Classification & Detection")
st.markdown(
    """
### AI-Powered Bird vs Drone Recognition System

This app demonstrates:

- ✅ Custom CNN classification model  
- ✅ Transfer Learning (ResNet50, MobileNetV2)  
- ✅ YOLOv8 object detection  

Upload an image and choose your mode on the left.
---
"""
)

# ---------------- Sidebar ---------------- #

st.sidebar.header("⚙️ Mode & Model")

# If YOLO failed to import (e.g., Streamlit Cloud), only show classification
if YOLO is None:
    mode = st.sidebar.radio(
        "Select Task",
        ["Classification"],
    )
    st.sidebar.warning(
        "YOLOv8 detection is available in the **local version** of this project.\n\n"
        "The Streamlit Cloud environment cannot install the heavy 'ultralytics' "
        "library (PyTorch + CUDA)."
    )
else:
    mode = st.sidebar.radio(
        "Select Task",
        ["Classification", "Object Detection (YOLOv8)"],
    )

if mode == "Classification":
    model_choice = st.sidebar.selectbox(
        "Select Classification Model",
        ["Custom CNN", "ResNet50", "MobileNetV2"],
    )
else:
    model_choice = "YOLOv8 Detection"

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """This project implements:

- Custom CNN classification
- Transfer learning (ResNet50, MobileNetV2)
- YOLOv8 object detection
- Streamlit-based web deployment

Use case: Aerial surveillance, wildlife monitoring, security & defense.
"""
)

# ---------------- Model loading functions ---------------- #

@st.cache_resource
def load_cnn_model():
    """Load custom CNN model, downloading from Google Drive on first use."""
    try:
        if not CNN_MODEL_PATH.exists():
            st.info("Downloading Custom CNN model from Google Drive...")
            url = f"https://drive.google.com/uc?export=download&id={CNN_DRIVE_ID}"
            gdown.download(url, str(CNN_MODEL_PATH), quiet=False)

        return keras.models.load_model(str(CNN_MODEL_PATH))
    except Exception as e:
        st.warning(f"Custom CNN model not available: {e}")
        return None


@st.cache_resource
def build_resnet_model():
    """Build ResNet50-based binary classifier (optionally load fine-tuned weights)."""
    try:
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        if RESNET_WEIGHTS.exists():
            model.load_weights(str(RESNET_WEIGHTS))

        return model
    except Exception as e:
        st.warning(f"ResNet50 model error: {e}")
        return None


@st.cache_resource
def build_mobilenet_model():
    """Build MobileNetV2-based binary classifier (optionally load fine-tuned weights)."""
    try:
        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        if MOBILENET_WEIGHTS.exists():
            model.load_weights(str(MOBILENET_WEIGHTS))

        return model
    except Exception as e:
        st.warning(f"MobileNetV2 model error: {e}")
        return None


@st.cache_resource
def load_yolov8_model():
    """
    Load YOLOv8 model.

    - On Streamlit Cloud: ultralytics usually cannot be installed (too heavy),
      so YOLO is None and we show a clear message.
    - On local machine (where ultralytics is installed): YOLO will work.
    """
    if YOLO is None:
        st.error(
            "YOLOv8 model could not be loaded.\n\n"
            "The current environment does not have the 'ultralytics' package "
            "installed (or it failed to install). This is expected on Streamlit Cloud "
            "because YOLOv8 depends on large GPU libraries.\n\n"
            "✅ To use YOLOv8 object detection, run this project locally on your "
            "own computer and install:\n\n"
            "`pip install ultralytics`\n"
        )
        return None

    try:
        return YOLO(str(YOLO_WEIGHTS))
    except Exception as e:
        st.error(
            "YOLOv8 model could not be loaded.\n\n"
            "This usually happens in restricted environments. "
            "To run YOLOv8, execute this project locally with 'ultralytics' installed.\n\n"
            f"Technical details: {e}"
        )
        return None


# ---------------- Utility functions ---------------- #

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize and normalize image for classification models."""
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(model, image: Image.Image, model_name: str):
    """Run classification and return predicted label and confidence.

    We assume the model output is P(Drone). We then compute:
      - P(Drone) = p_drone
      - P(Bird)  = 1 - p_drone

    To reduce false Drone predictions on birds, we use a slightly
    higher threshold for Drone (0.7). Below that, we call it Bird.
    """
    processed_image = preprocess_image(image, target_size=IMG_SIZE)

    with st.spinner(f"Analyzing image with {model_name}..."):
        prediction = model.predict(processed_image, verbose=0)
        p_drone = float(prediction[0][0])
        p_bird = 1.0 - p_drone

    DRONE_THRESHOLD = 0.7  # tune if needed

    if p_drone >= DRONE_THRESHOLD:
        class_name = "Drone"
        class_emoji = "🚁"
        confidence = p_drone
    else:
        class_name = "Bird"
        class_emoji = "🦅"
        confidence = p_bird

    return class_name, class_emoji, confidence


# ---------------- File uploader ---------------- #

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an aerial image containing a bird or drone",
)

# ---------------- Main logic ---------------- #

if uploaded_file is not None:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📷 Input Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=500)

    with col2:
        st.subheader("🎯 Prediction / Detection Results")

        # ---------- Classification mode ---------- #
        if mode == "Classification":
            if model_choice == "Custom CNN":
                model = load_cnn_model()
                model_name = "Custom CNN"
            elif model_choice == "ResNet50":
                model = build_resnet_model()
                model_name = "ResNet50 (Transfer Learning)"
            else:
                model = build_mobilenet_model()
                model_name = "MobileNetV2 (Transfer Learning)"

            if model is None:
                st.error(f"{model_choice} model could not be loaded. Check model file / requirements.")
            else:
                class_name, class_emoji, confidence = classify_image(
                    model, image, model_name
                )

                st.success(f"## {class_emoji} {class_name}")
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.progress(confidence)

                st.info(
                    f"""**Model Used:** {model_name}
**Prediction:** {class_name}
**Confidence:** {confidence * 100:.2f}%"""
                )

        # ---------- YOLOv8 Detection mode ---------- #
        else:
            yolo_model = load_yolov8_model()
            if yolo_model is not None:
                with st.spinner("Running YOLOv8 detection..."):
                    # YOLOv8 accepts numpy RGB images directly
                    img_np = np.array(image)  # RGB
                    results = yolo_model(img_np)

                if len(results) > 0 and len(results[0].boxes) > 0:
                    st.success(f"✅ Objects Detected: {len(results[0].boxes)}")
                    annotated_bgr = results[0].plot()         # BGR array
                    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
                    st.image(
                        annotated_rgb,
                        width=500,
                        caption="YOLOv8 Detection Output",
                    )
                else:
                    st.info("No objects detected with high confidence.")

# ---------------- Static performance section ---------------- #

st.markdown("---")
st.markdown("### 📊 Model Performance Summary (from training experiments)")

c1, c2, c3 = st.columns(3)
c1.metric("Custom CNN Test Accuracy", "94.5%")
c2.metric("ResNet50 Test Accuracy", "97.2%")
c3.metric("MobileNetV2 Test Accuracy", "95.8%")

st.markdown(
    """
> *Note:* Performance metrics are derived from offline training on the Bird vs Drone dataset
(train/val/test splits as described in the project document).
"""
)

# ---------------- Footer ---------------- #

st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using TensorFlow/Keras, YOLOv8 and Streamlit</p>
        <p>🎓 Aerial Object Classification & Detection | Bird vs Drone Project</p>
    </div>
    """,
    unsafe_allow_html=True,
)
