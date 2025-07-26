# ASL Translator and Emotion Communicator

A desktop application that translates American Sign Language (ASL) gestures and facial emotions in real-time using a webcam. This application is designed for people who are deaf or mute to help them communicate more effectively.

## Features

- Real-time hand gesture detection using MediaPipe Hands
- Custom gesture classification for 80+ ASL words using TensorFlow/Keras
- Facial emotion recognition using MediaPipe FaceMesh
- Completely offline functionality - no internet connection required
- Beautiful GUI with PyQt5 (Dark mode, Live camera feed, translation panel)
- Translation history saved locally
- Optional text-to-speech output

## Tech Stack

- Python 3.x
- OpenCV for webcam input
- MediaPipe (Hands, FaceMesh)
- TensorFlow/Keras or TFLite for gesture and emotion classification
- PyQt5 for user interface
- NumPy, Pandas for data processing

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

## Project Structure

```
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── assets/                # Images, icons, and other static assets
├── data/                  # Data for training models
├── models/                # Pre-trained models
│   ├── asl_model/         # ASL gesture recognition model
│   └── emotion_model/     # Facial emotion recognition model
└── src/                   # Source code
    ├── gui/               # PyQt5 GUI components
    ├── asl/               # ASL detection and classification
    ├── emotion/           # Emotion detection and classification
    ├── utils/             # Utility functions
    └── history/           # Translation history management
```

## License

MIT