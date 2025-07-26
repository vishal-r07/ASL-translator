#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Help Dialog

This module provides a dialog for displaying help information about the application.
"""

import os
import sys
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QTextBrowser, QDialogButtonBox, QScrollArea
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions and configuration
from src.utils.config import get_config
from src.utils.common import get_assets_dir


class HelpDialog(QDialog):
    """
    Dialog for displaying help information about the application.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the help dialog.
        
        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.setWindowTitle("Help")
        self.setMinimumSize(700, 500)
        
        # Get configuration
        self.config = get_config()
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Getting Started tab
        getting_started_tab = QWidget()
        getting_started_layout = QVBoxLayout(getting_started_tab)
        
        getting_started_text = QTextBrowser()
        getting_started_text.setOpenExternalLinks(True)
        getting_started_text.setHtml("""
        <h2>Getting Started</h2>
        
        <p>Welcome to the ASL Translator and Emotion Communicator! This application helps translate American Sign Language (ASL) gestures and facial emotions in real-time using your webcam.</p>
        
        <h3>Basic Setup</h3>
        
        <ol>
            <li><strong>Camera Setup:</strong> Make sure your webcam is connected and working properly. The application will automatically detect and use the default camera.</li>
            <li><strong>Lighting:</strong> Ensure you are in a well-lit environment for better detection accuracy.</li>
            <li><strong>Positioning:</strong> Position yourself so that your upper body, particularly your hands and face, are clearly visible in the camera frame.</li>
            <li><strong>Background:</strong> A plain, uncluttered background works best for optimal detection.</li>
        </ol>
        
        <h3>Main Interface</h3>
        
        <p>The main interface consists of:</p>
        
        <ul>
            <li><strong>Camera Feed:</strong> Shows the live video from your webcam.</li>
            <li><strong>Translation Panel:</strong> Displays the detected ASL gestures and emotions.</li>
            <li><strong>Control Buttons:</strong> Allows you to start/stop detection, clear history, and access settings.</li>
        </ul>
        
        <h3>Quick Start</h3>
        
        <ol>
            <li>Launch the application.</li>
            <li>Grant camera permissions if prompted.</li>
            <li>Position yourself in front of the camera.</li>
            <li>Start signing! The application will detect and translate your gestures and emotions in real-time.</li>
        </ol>
        """)
        
        getting_started_layout.addWidget(getting_started_text)
        tab_widget.addTab(getting_started_tab, "Getting Started")
        
        # ASL Detection tab
        asl_tab = QWidget()
        asl_layout = QVBoxLayout(asl_tab)
        
        asl_text = QTextBrowser()
        asl_text.setOpenExternalLinks(True)
        asl_text.setHtml("""
        <h2>ASL Detection</h2>
        
        <p>The ASL Translator can detect and translate a wide range of American Sign Language gestures.</p>
        
        <h3>Supported Gestures</h3>
        
        <p>The application can detect over 80 common ASL words and phrases, including:</p>
        
        <ul>
            <li>Alphabets (A-Z)</li>
            <li>Numbers (0-10)</li>
            <li>Common words (hello, thank you, please, etc.)</li>
            <li>Basic phrases (how are you, my name is, etc.)</li>
        </ul>
        
        <h3>Tips for Better Detection</h3>
        
        <ul>
            <li><strong>Clear Gestures:</strong> Make your hand gestures clear and deliberate.</li>
            <li><strong>Hand Position:</strong> Keep your hands within the camera frame and avoid rapid movements.</li>
            <li><strong>Distance:</strong> Maintain an appropriate distance from the camera (about 2-3 feet).</li>
            <li><strong>Lighting:</strong> Ensure your hands are well-lit and not in shadow.</li>
        </ul>
        
        <h3>Customization</h3>
        
        <p>You can customize the ASL detection settings in the Settings dialog:</p>
        
        <ul>
            <li><strong>Detection Confidence:</strong> Adjust the minimum confidence threshold for gesture detection.</li>
            <li><strong>Temporal Filtering:</strong> Adjust how long a gesture needs to be detected before being recognized.</li>
            <li><strong>Model Selection:</strong> Choose between different ASL detection models (if available).</li>
        </ul>
        """)
        
        asl_layout.addWidget(asl_text)
        tab_widget.addTab(asl_tab, "ASL Detection")
        
        # Emotion Detection tab
        emotion_tab = QWidget()
        emotion_layout = QVBoxLayout(emotion_tab)
        
        emotion_text = QTextBrowser()
        emotion_text.setOpenExternalLinks(True)
        emotion_text.setHtml("""
        <h2>Emotion Detection</h2>
        
        <p>The Emotion Communicator can detect and display facial emotions in real-time.</p>
        
        <h3>Supported Emotions</h3>
        
        <p>The application can detect the following emotions:</p>
        
        <ul>
            <li>Happy</li>
            <li>Sad</li>
            <li>Angry</li>
            <li>Surprised</li>
            <li>Fearful</li>
            <li>Disgusted</li>
            <li>Neutral</li>
        </ul>
        
        <h3>Tips for Better Detection</h3>
        
        <ul>
            <li><strong>Face Position:</strong> Ensure your face is clearly visible and centered in the camera frame.</li>
            <li><strong>Lighting:</strong> Good lighting on your face improves detection accuracy.</li>
            <li><strong>Expressions:</strong> Make clear facial expressions for better emotion recognition.</li>
            <li><strong>Glasses and Accessories:</strong> Glasses, hats, or other accessories may affect detection accuracy.</li>
        </ul>
        
        <h3>Customization</h3>
        
        <p>You can customize the emotion detection settings in the Settings dialog:</p>
        
        <ul>
            <li><strong>Detection Confidence:</strong> Adjust the minimum confidence threshold for emotion detection.</li>
            <li><strong>Temporal Filtering:</strong> Adjust how long an emotion needs to be detected before being recognized.</li>
            <li><strong>Model Selection:</strong> Choose between different emotion detection models (if available).</li>
        </ul>
        """)
        
        emotion_layout.addWidget(emotion_text)
        tab_widget.addTab(emotion_tab, "Emotion Detection")
        
        # Features tab
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        
        features_text = QTextBrowser()
        features_text.setOpenExternalLinks(True)
        features_text.setHtml("""
        <h2>Features</h2>
        
        <h3>Text-to-Speech</h3>
        
        <p>The application can convert detected ASL gestures and emotions to spoken words using text-to-speech technology.</p>
        
        <p>To use this feature:</p>
        
        <ol>
            <li>Enable text-to-speech in the Settings dialog.</li>
            <li>Adjust volume, rate, and voice settings as needed.</li>
            <li>The application will speak the detected gestures and emotions as they are recognized.</li>
        </ol>
        
        <h3>Translation History</h3>
        
        <p>The application keeps a history of detected ASL gestures and emotions.</p>
        
        <p>History features include:</p>
        
        <ul>
            <li><strong>View History:</strong> Access the history dialog to view past translations.</li>
            <li><strong>Search:</strong> Search for specific words or emotions in the history.</li>
            <li><strong>Filter:</strong> Filter history by type (ASL or emotion) and date range.</li>
            <li><strong>Export:</strong> Export history to CSV file for external use.</li>
            <li><strong>Clear:</strong> Clear history as needed.</li>
        </ul>
        
        <h3>Customization</h3>
        
        <p>The application offers various customization options:</p>
        
        <ul>
            <li><strong>Theme:</strong> Dark mode for comfortable viewing.</li>
            <li><strong>Display Settings:</strong> Customize how translations are displayed on screen.</li>
            <li><strong>Camera Settings:</strong> Select camera, adjust resolution, and more.</li>
            <li><strong>Detection Settings:</strong> Fine-tune ASL and emotion detection parameters.</li>
            <li><strong>Text-to-Speech Settings:</strong> Customize speech output.</li>
        </ul>
        """)
        
        features_layout.addWidget(features_text)
        tab_widget.addTab(features_tab, "Features")
        
        # Troubleshooting tab
        troubleshooting_tab = QWidget()
        troubleshooting_layout = QVBoxLayout(troubleshooting_tab)
        
        troubleshooting_text = QTextBrowser()
        troubleshooting_text.setOpenExternalLinks(True)
        troubleshooting_text.setHtml("""
        <h2>Troubleshooting</h2>
        
        <h3>Camera Issues</h3>
        
        <p><strong>Problem:</strong> Camera not detected or not working.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
            <li>Ensure your webcam is properly connected.</li>
            <li>Check if other applications are using the camera and close them.</li>
            <li>Verify camera permissions in your operating system settings.</li>
            <li>Try selecting a different camera in the Settings dialog if multiple cameras are available.</li>
            <li>Restart the application.</li>
        </ul>
        
        <h3>Detection Accuracy</h3>
        
        <p><strong>Problem:</strong> Poor ASL or emotion detection accuracy.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
            <li>Ensure you are in a well-lit environment.</li>
            <li>Position yourself properly in the camera frame.</li>
            <li>Make clear, deliberate gestures and expressions.</li>
            <li>Adjust detection confidence thresholds in the Settings dialog.</li>
            <li>Try different models if available.</li>
        </ul>
        
        <h3>Performance Issues</h3>
        
        <p><strong>Problem:</strong> Application running slowly or lagging.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
            <li>Close other resource-intensive applications.</li>
            <li>Lower the camera resolution in the Settings dialog.</li>
            <li>Reduce the detection frequency in the Advanced settings.</li>
            <li>Disable features you don't need (e.g., text-to-speech).</li>
            <li>Ensure your computer meets the minimum system requirements.</li>
        </ul>
        
        <h3>Text-to-Speech Issues</h3>
        
        <p><strong>Problem:</strong> Text-to-speech not working or sounds incorrect.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
            <li>Ensure text-to-speech is enabled in the Settings dialog.</li>
            <li>Check your system's audio settings and volume.</li>
            <li>Try different voices or speech rates in the Settings dialog.</li>
            <li>Restart the application.</li>
        </ul>
        
        <h3>Application Crashes</h3>
        
        <p><strong>Problem:</strong> Application crashes or freezes.</p>
        <p><strong>Solutions:</strong></p>
        <ul>
            <li>Ensure your operating system and drivers are up to date.</li>
            <li>Check for application updates.</li>
            <li>Try resetting the application settings to default.</li>
            <li>Restart your computer.</li>
            <li>If the issue persists, check the application logs for error messages.</li>
        </ul>
        """)
        
        troubleshooting_layout.addWidget(troubleshooting_text)
        tab_widget.addTab(troubleshooting_tab, "Troubleshooting")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Create button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        
        # Add button box to main layout
        main_layout.addWidget(button_box)