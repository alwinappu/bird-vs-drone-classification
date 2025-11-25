# 🦅 Bird vs Drone Classification & Detection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

AI-powered aerial object recognition system using Deep Learning for Bird vs Drone Classification with real-time detection capabilities.

## 🎯 Live Demo

🔗 **[Try the App on Streamlit Cloud](https://bird-vs-drone-classification.streamlit.app/)**

> **Note:** YOLOv8 detection is available only in the local version due to Streamlit Cloud's resource constraints.

## ✨ Features

- **🧠 Multiple Classification Models**
  - Custom CNN architecture (94.5% accuracy)
  - Transfer Learning with ResNet50 (97.2% accuracy)
  - Transfer Learning with MobileNetV2 (95.8% accuracy)

- **🎯 Object Detection**
  - YOLOv8n integration for real-time detection
  - Bounding box visualization
  - Multi-object recognition

- **🌐 Web Interface**
  - Interactive Streamlit dashboard
  - Drag-and-drop image upload
  - Real-time predictions
  - Confidence score visualization

- **📊 Performance Metrics**
  - Detailed accuracy reports
  - Confusion matrix analysis
  - Model comparison dashboard

## 🚀 Quick Start

### Online Version (Streamlit Cloud)

Simply visit: [https://bird-vs-drone-classification.streamlit.app/](https://bird-vs-drone-classification.streamlit.app/)

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/alwinappu/bird-vs-drone-classification.git
cd bird-vs-drone-classification
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

For basic classification (without YOLOv8):
```bash
pip install -r requirements.txt
```

For full functionality including YOLOv8 detection:
```bash
pip install -r requirements-local.txt
```

4. **Run the application**
```bash
streamlit run streamlit_app.py
```

5. **Open your browser**

The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
bird-vs-drone-classification/
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt           # Cloud deployment dependencies
├── requirements-local.txt     # Local development dependencies (with YOLOv8)
├── packages.txt              # System packages for Streamlit Cloud
├── runtime.txt               # Python version specification
├── .python-version           # Python version for local development
├── .streamlit/               # Streamlit configuration
│   └── config.toml
├── custom_cnn_model.h5       # Pre-trained Custom CNN weights
├── resnet_weights.h5         # Fine-tuned ResNet50 weights (optional)
├── mobilenet_weights.h5      # Fine-tuned MobileNetV2 weights (optional)
├── yolov8n.pt                # YOLOv8 nano weights (auto-downloaded)
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
└── .gitignore                # Git ignore rules
```

## 🎓 Model Information

### Custom CNN Architecture
- **Input:** 224x224 RGB images
- **Layers:** Convolutional layers with batch normalization and dropout
- **Output:** Binary classification (Bird/Drone)
- **Training Accuracy:** 94.5%

### Transfer Learning Models

#### ResNet50
- **Base:** Pre-trained on ImageNet
- **Fine-tuning:** Custom classification head
- **Test Accuracy:** 97.2%
- **Best for:** High accuracy requirements

#### MobileNetV2
- **Base:** Pre-trained on ImageNet
- **Fine-tuning:** Lightweight classification head
- **Test Accuracy:** 95.8%
- **Best for:** Fast inference, mobile deployment

### YOLOv8 Detection
- **Version:** YOLOv8n (nano)
- **Purpose:** Real-time object detection
- **Features:** Bounding boxes, confidence scores, multi-object detection

## 💡 Usage

### Classification Mode

1. Select "Classification" from the sidebar
2. Choose your preferred model (Custom CNN, ResNet50, or MobileNetV2)
3. Upload an aerial image (JPG, JPEG, or PNG)
4. View the prediction with confidence score

### Detection Mode (Local Only)

1. Select "Object Detection (YOLOv8)" from the sidebar
2. Upload an aerial image
3. View detected objects with bounding boxes

## 📊 Performance Metrics

| Model | Test Accuracy | Inference Time | Size |
|-------|--------------|----------------|------|
| Custom CNN | 94.5% | ~50ms | 12 MB |
| ResNet50 | 97.2% | ~120ms | 98 MB |
| MobileNetV2 | 95.8% | ~80ms | 14 MB |
| YOLOv8n | Detection | ~40ms | 6 MB |

## 🔧 Troubleshooting

### YOLOv8 Not Working on Streamlit Cloud

This is expected behavior. The `ultralytics` library (required for YOLOv8) is too heavy for Streamlit Cloud's free tier. To use YOLOv8:

1. Clone the repository locally
2. Install dependencies: `pip install -r requirements-local.txt`
3. Run locally: `streamlit run streamlit_app.py`

### Model Files Missing

The Custom CNN model (`custom_cnn_model.h5`) is automatically downloaded from Google Drive on first use. If download fails:

1. Check your internet connection
2. Manually download from: [Google Drive Link](https://drive.google.com/uc?export=download&id=1q7CkuixuGhfauILzP3QBHJvSmf5w2TGa)
3. Place the file in the project root directory

### Memory Issues

If you encounter memory issues when running locally:

1. Use MobileNetV2 instead of ResNet50
2. Reduce batch size for predictions
3. Close other applications

## 🎯 Use Cases

- **Aerial Surveillance:** Automated monitoring of restricted airspaces
- **Wildlife Conservation:** Distinguishing birds from drones in protected areas
- **Security & Defense:** Threat detection and classification
- **Airport Safety:** Bird strike prevention systems
- **Smart Cities:** Urban airspace monitoring

## 🛠️ Technologies Used

- **Deep Learning:** TensorFlow/Keras
- **Object Detection:** Ultralytics YOLOv8
- **Web Framework:** Streamlit
- **Image Processing:** PIL, OpenCV
- **Numerical Computing:** NumPy
- **Deployment:** Streamlit Cloud, GitHub

## 📝 Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
- TensorFlow >= 2.10.0
- Streamlit >= 1.20.0
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- gdown >= 4.6.0

### Optional (Local Only)
- ultralytics >= 8.0.0 (for YOLOv8)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Alwin Appu**
- GitHub: [@alwinappu](https://github.com/alwinappu)
- Email: appu050.1021@hotmail.com

## 🙏 Acknowledgments

- TensorFlow team for the amazing deep learning framework
- Ultralytics for YOLOv8 implementation
- Streamlit for the intuitive web framework
- The open-source community for various tools and libraries

## 📊 Dataset

The models were trained on a curated dataset of aerial images containing birds and drones in various conditions:
- Multiple viewing angles
- Different lighting conditions
- Various backgrounds
- Real-world scenarios

## 🔮 Future Enhancements

- [ ] Add video stream processing
- [ ] Implement real-time webcam detection
- [ ] Multi-class classification (different bird species, drone types)
- [ ] Mobile app development
- [ ] Edge device deployment (Raspberry Pi, Jetson Nano)
- [ ] API endpoint for integration
- [ ] Batch processing capabilities

## 📞 Support

If you have any questions or issues, please:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Contact via email: appu050.1021@hotmail.com

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

**Built with ❤️ using TensorFlow/Keras, YOLOv8, and Streamlit**

*Aerial Object Classification & Detection | Bird vs Drone Project*
