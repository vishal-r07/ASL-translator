#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASL Translator and Emotion Communicator

This script runs the ASL Translator and Emotion Communicator application.
It serves as the entry point for users.
"""

import os
import sys
import argparse
import logging

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from src.utils.config import get_config
from src.utils.common import get_models_dir


def setup_logging():
    """
    Set up logging configuration.
    """
    config = get_config()
    log_level = getattr(logging, config.get("log_level", "INFO"))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log"))
        ]
    )


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise.
    """
    try:
        # Check core dependencies
        import cv2
        import mediapipe
        import tensorflow as tf
        import numpy as np
        from PyQt5.QtWidgets import QApplication
        
        # Check optional dependencies
        try:
            import qdarktheme
        except ImportError:
            logging.warning("qdarktheme not found. Dark mode may not work properly.")
        
        try:
            import pyttsx3
        except ImportError:
            logging.warning("pyttsx3 not found. Text-to-speech functionality will be disabled.")
        
        return True
    
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        print(f"Error: Missing dependency: {e}")
        print("Please install all required dependencies by running:")
        print("pip install -r requirements.txt")
        return False


def check_models():
    """
    Check if required models are available.
    
    Returns:
        bool: True if models are available, False otherwise.
    """
    models_dir = get_models_dir()
    
    # Check ASL model
    asl_model_path = os.path.join(models_dir, "asl_model", "asl_model.h5")
    asl_tflite_path = os.path.join(models_dir, "asl_model", "asl_model.tflite")
    
    # Check emotion model
    emotion_model_path = os.path.join(models_dir, "emotion_model", "emotion_model.h5")
    emotion_tflite_path = os.path.join(models_dir, "emotion_model", "emotion_model.tflite")
    
    # Check if at least one version of each model exists
    asl_model_exists = os.path.exists(asl_model_path) or os.path.exists(asl_tflite_path)
    emotion_model_exists = os.path.exists(emotion_model_path) or os.path.exists(emotion_tflite_path)
    
    if not asl_model_exists:
        logging.warning("ASL model not found. Please train or download the model.")
        print("Warning: ASL model not found. Please train or download the model.")
        print("You can train the model by running:")
        print("python -m src.asl.train_model --collect --train")
        print("Or download pre-trained models by running:")
        print("python -m src.utils.download_models --model asl")
    
    if not emotion_model_exists:
        logging.warning("Emotion model not found. Please train or download the model.")
        print("Warning: Emotion model not found. Please train or download the model.")
        print("You can train the model by running:")
        print("python -m src.emotion.train_model --collect --train")
        print("Or download pre-trained models by running:")
        print("python -m src.utils.download_models --model emotion")
    
    return asl_model_exists and emotion_model_exists


def main():
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description="ASL Translator and Emotion Communicator")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index to use (default: use value from config)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check models
    models_available = check_models()
    if not models_available:
        print("\nContinuing without models. Some functionality may be limited.")
    
    # Configure GPU usage
    config = get_config()
    if args.debug:
        config.set("debug_mode", True)
    
    if args.no_gpu or not config.get("enable_gpu", True):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logging.info("GPU acceleration disabled")
    else:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logging.error(f"Error configuring GPU: {e}")
        else:
            logging.info("No GPU found. Using CPU.")
    
    # Override camera index if specified
    if args.camera is not None:
        config.set("camera_index", args.camera)
    
    # Import main application module here to avoid circular imports
    from main import run_app
    
    # Run the application
    return run_app()


if __name__ == "__main__":
    sys.exit(main())