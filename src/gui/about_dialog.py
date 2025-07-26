#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
About Dialog

This module provides a dialog for displaying information about the application.
"""

import os
import sys
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QTextBrowser, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions and configuration
from src.utils.config import get_config
from src.utils.common import get_assets_dir


class AboutDialog(QDialog):
    """
    Dialog for displaying information about the application.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the about dialog.
        
        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.setWindowTitle("About")
        self.setMinimumSize(600, 400)
        
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
        
        # Header layout
        header_layout = QHBoxLayout()
        
        # Logo (placeholder - you would need to create an actual logo)
        logo_path = os.path.join(get_assets_dir(), "logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaled(QSize(64, 64), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header_layout.addWidget(logo_label)
        
        # Title and version
        title_layout = QVBoxLayout()
        
        app_name = self.config.get("app_name", "ASL Translator and Emotion Communicator")
        app_version = self.config.get("app_version", "1.0.0")
        
        title_label = QLabel(app_name)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        
        version_label = QLabel(f"Version {app_version}")
        title_layout.addWidget(version_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch(1)
        
        # Add header layout to main layout
        main_layout.addLayout(header_layout)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # About tab
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextBrowser()
        about_text.setOpenExternalLinks(True)
        about_text.setHtml("""
        <p>ASL Translator and Emotion Communicator is a desktop application designed to help people who are deaf or mute communicate more effectively.</p>
        
        <p>The application uses computer vision and machine learning to detect and translate American Sign Language (ASL) gestures and facial emotions in real-time.</p>
        
        <p>Key features:</p>
        <ul>
            <li>Real-time ASL gesture detection and translation</li>
            <li>Facial emotion recognition</li>
            <li>Fully offline operation - no internet connection required</li>
            <li>Text-to-speech output</li>
            <li>Translation history management</li>
        </ul>
        
        <p>This application was developed using Python, OpenCV, MediaPipe, TensorFlow, and PyQt5.</p>
        """)
        
        about_layout.addWidget(about_text)
        tab_widget.addTab(about_tab, "About")
        
        # Credits tab
        credits_tab = QWidget()
        credits_layout = QVBoxLayout(credits_tab)
        
        credits_text = QTextBrowser()
        credits_text.setOpenExternalLinks(True)
        credits_text.setHtml("""
        <h3>Credits</h3>
        
        <p>This application uses the following open-source libraries and frameworks:</p>
        
        <ul>
            <li><a href="https://www.python.org/">Python</a> - Programming language</li>
            <li><a href="https://opencv.org/">OpenCV</a> - Computer vision library</li>
            <li><a href="https://mediapipe.dev/">MediaPipe</a> - Machine learning framework for multimodal applied ML pipelines</li>
            <li><a href="https://www.tensorflow.org/">TensorFlow</a> - Machine learning framework</li>
            <li><a href="https://www.riverbankcomputing.com/software/pyqt/">PyQt5</a> - GUI framework</li>
            <li><a href="https://github.com/5yutan5/PyQtDarkTheme">QDarkTheme</a> - Dark theme for PyQt applications</li>
            <li><a href="https://github.com/nateshmbhat/pyttsx3">pyttsx3</a> - Text-to-speech library</li>
        </ul>
        
        <p>Special thanks to:</p>
        <ul>
            <li>The MediaPipe team for their excellent hand and face tracking solutions</li>
            <li>The TensorFlow team for their machine learning framework</li>
            <li>The PyQt team for their GUI framework</li>
            <li>All the open-source contributors who made this project possible</li>
        </ul>
        """)
        
        credits_layout.addWidget(credits_text)
        tab_widget.addTab(credits_tab, "Credits")
        
        # License tab
        license_tab = QWidget()
        license_layout = QVBoxLayout(license_tab)
        
        license_text = QTextBrowser()
        license_text.setOpenExternalLinks(True)
        license_text.setHtml("""
        <h3>MIT License</h3>
        
        <p>Copyright (c) 2023 ASL Translator and Emotion Communicator</p>
        
        <p>Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:</p>
        
        <p>The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.</p>
        
        <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.</p>
        """)
        
        license_layout.addWidget(license_text)
        tab_widget.addTab(license_tab, "License")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        # Create button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        
        # Add button box to main layout
        main_layout.addWidget(button_box)