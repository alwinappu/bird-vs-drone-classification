import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from PIL import Image, ImageDraw
import gdown
import onnxruntime as ort

# ---------------- Paths & constants ---------------- #

BASE_DIR = Path(".")
CNN_MODEL_PATH = BASE_DIR / "custom_cnn_model.h5"     # Custom CNN weights
RESNET_WEIGHTS = BASE_DIR / "resnet_weights.h5"       # Optional fine-tuned weights
MOBILENET_WEIGHTS = BASE_DIR / "mobilenet_weights.h5" # Optional fine-tuned weights
YOLO_ONNX_PATH = BASE_DIR / "yolov8n.onnx"            # YOLOv8 ONNX model

# Google Drive ID for custom_cnn_model.h5
CNN_DRIVE_ID = "1q7CkuixuGhfauILzP3QBHJvSmf5w2TGa"

IMG_SIZE = (224, 224)  # for classification models

# ---------------- Streamlit page config ---------------- #

st.set_page_config(
    page_title="Aerial Object Classification & Detection",
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
- ✅ YOLOv8 object detection (ONNX, CPU)

Upload an image and choose your mode on the left.
---
"""
)

# ---------------- Sidebar ---------------- #

st.sidebar.header("⚙️ Mode & Model")

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
- YOLOv8 object detection via ONNX (CPU)
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
def load_yolov8_onnx():
    """Load YOLOv8 ONNX model using onnxruntime (CPU)."""
    if not YOLO_ONNX_PATH.exists():
        st.error("YOLOv8 ONNX model 'yolov8n.onnx' not found in app directory.")
        return None

    try:
        session = ort.InferenceSession(
            str(YOLO_ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        return session
    except Exception as e:
        st.error(f"Failed to load YOLOv8 ONNX model: {e}")
        return None


# ---------------- Utility functions ---------------- #

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize and normalize image for classification models."""
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
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


def run_yolo_onnx(
    session: ort.InferenceSession,
    image: Image.Image,
    img_size: int = 640,
    conf_thres: float = 0.10,   # lower threshold to get more detections
):
    """
    Run YOLOv8 ONNX inference on a PIL image.
    Returns (annotated_image, num_detections).

    NOTE: Ultralytics YOLOv8 ONNX output for detection is:
      (1, 84, 8400)  ->  84 = 4 box coords + 80 class scores.
    There is NO separate objectness column here.
    """

    orig_w, orig_h = image.size

    # 1) Preprocess: resize to square, normalize, NCHW
    img_resized = image.resize((img_size, img_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0  # (H,W,3) RGB
    img_np = np.transpose(img_np, (2, 0, 1))  # (3,H,W)
    img_np = np.expand_dims(img_np, axis=0)   # (1,3,H,W)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_np})
    preds = outputs[0]  # e.g. (1, 84, 8400) or (1, 8400, 84)

    # 2) Ensure shape is (num_anchors, num_attrs)
    if preds.ndim == 3:
        # If shape is (1,84,8400) -> transpose to (1,8400,84)
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0]  # (num_anchors, num_attrs) = (8400, 84)

    # YOLOv8 format here: [x, y, w, h, cls0, cls1, ... cls79]
    boxes_xywh = preds[:, 0:4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    # Filter by confidence
    keep = scores > conf_thres
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    num_det = 0

    for (cx, cy, w, h), score, cls in zip(boxes_xywh, scores, class_ids):
        # cx,cy,w,h are in pixels relative to img_size
        x1 = (cx - w / 2) * (orig_w / img_size)
        y1 = (cy - h / 2) * (orig_h / img_size)
        x2 = (cx + w / 2) * (orig_w / img_size)
        y2 = (cy + h / 2) * (orig_h / img_size)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Clip to image bounds
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw rectangle (red)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # Simple label (class id + score)
        label = f"ID {int(cls)} ({score:.2f})"
        text_bg_w = draw.textlength(label) + 6
        draw.rectangle([x1, y1 - 16, x1 + text_bg_w, y1], fill=(255, 0, 0))
        draw.text((x1 + 3, y1 - 14), label, fill=(255, 255, 255))

        num_det += 1

    return annotated, num_det


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
            session = load_yolov8_onnx()
            if session is not None:
                with st.spinner("Running YOLOv8 (ONNX) detection..."):
                    annotated_img, num_det = run_yolo_onnx(session, image)

                if num_det > 0:
                    st.success(f"✅ Objects detected: {num_det}")
                    st.image(
                        annotated_img,
                        width=500,
                        caption="YOLOv8 (ONNX) Detection Output",
                    )
                else:
                    st.info("No objects detected above confidence threshold.")

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
        <p>Built with ❤️ using TensorFlow/Keras, YOLOv8 (ONNX) and Streamlit</p>
        <p>🎓 Aerial Object Classification & Detection | Bird vs Drone Project</p>
    </div>
    """,
    unsafe_allow_html=True,
)
