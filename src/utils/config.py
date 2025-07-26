#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Application Configuration

This module provides configuration settings for the ASL Translator and Emotion Communicator application.
It includes default settings and functions to load/save user configurations.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_user_data_dir

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    "app_name": "ASL Translator and Emotion Communicator",
    "app_version": "1.0.0",
    "dark_mode": True,
    "save_history": True,
    "history_max_entries": 1000,
    
    # Camera settings
    "camera_index": 0,  # Default camera (usually the built-in webcam)
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    
    # ASL detection settings
    "asl_detection_enabled": True,
    "asl_confidence_threshold": 0.7,  # Minimum confidence to accept a gesture classification
    "asl_temporal_window": 5,  # Number of frames to average for temporal smoothing
    "asl_min_detection_confidence": 0.5,  # MediaPipe Hands detection confidence
    "asl_min_tracking_confidence": 0.5,  # MediaPipe Hands tracking confidence
    "asl_max_num_hands": 2,  # Maximum number of hands to detect
    
    # Emotion detection settings
    "emotion_detection_enabled": True,
    "emotion_confidence_threshold": 0.6,  # Minimum confidence to accept an emotion classification
    "emotion_temporal_window": 10,  # Number of frames to average for temporal smoothing
    "emotion_min_detection_confidence": 0.5,  # MediaPipe FaceMesh detection confidence
    "emotion_min_tracking_confidence": 0.5,  # MediaPipe FaceMesh tracking confidence
    
    # Display settings
    "show_landmarks": True,  # Show hand and face landmarks
    "show_fps": True,  # Show frames per second
    "text_size": 1.0,  # Text size multiplier
    "animation_speed": 1.0,  # Animation speed multiplier
    
    # Text-to-speech settings
    "tts_enabled": True,  # Enable text-to-speech
    "tts_rate": 150,  # Speech rate (words per minute)
    "tts_volume": 1.0,  # Speech volume (0.0 to 1.0)
    "tts_voice_index": 0,  # Voice index (depends on available voices)
    
    # Advanced settings
    "use_tflite": True,  # Use TensorFlow Lite models for better performance
    "enable_gpu": True,  # Use GPU acceleration if available
    "debug_mode": False,  # Enable debug mode
    "log_level": "INFO",  # Logging level
}


class Config:
    """
    Configuration class for the ASL Translator and Emotion Communicator application.
    """
    
    def __init__(self):
        """
        Initialize the configuration with default settings.
        """
        self._config = DEFAULT_CONFIG.copy()
        self._config_file = os.path.join(get_user_data_dir(), "config.json")
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from file if it exists.
        """
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Update configuration with user settings
                self._config.update(user_config)
                logger.info(f"Configuration loaded from {self._config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self) -> bool:
        """
        Save the current configuration to file.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
            
            # Save configuration to file
            with open(self._config_file, 'w') as f:
                json.dump(self._config, f, indent=4)
            
            logger.info(f"Configuration saved to {self._config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """
        Reset configuration to default settings.
        """
        self._config = DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key (str): The configuration key.
            default (Any, optional): Default value if key doesn't exist.
            
        Returns:
            Any: The configuration value.
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key (str): The configuration key.
            value (Any): The configuration value.
        """
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration settings.
        
        Returns:
            Dict[str, Any]: All configuration settings.
        """
        return self._config.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update multiple configuration settings.
        
        Args:
            config_dict (Dict[str, Any]): Dictionary of configuration settings to update.
        """
        self._config.update(config_dict)
    
    @property
    def config_file(self) -> str:
        """
        Get the path to the configuration file.
        
        Returns:
            str: Path to the configuration file.
        """
        return self._config_file


# Create a singleton instance
config = Config()


def get_config() -> Config:
    """
    Get the configuration instance.
    
    Returns:
        Config: The configuration instance.
    """
    return config


def set_config_value(key: str, value: Any) -> None:
    """
    Set a configuration value and save the configuration.
    
    Args:
        key (str): The configuration key.
        value (Any): The configuration value.
    """
    config.set(key, value)
    config.save_config()


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value.
    
    Args:
        key (str): The configuration key.
        default (Any, optional): Default value if key doesn't exist.
        
    Returns:
        Any: The configuration value.
    """
    return config.get(key, default)


def reset_config() -> None:
    """
    Reset configuration to default settings and save.
    """
    config.reset_to_defaults()
    config.save_config()