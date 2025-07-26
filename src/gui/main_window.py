#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Window for ASL Translator and Emotion Communicator

This module contains the MainWindow class which is the primary GUI component
for the application.
"""

import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QComboBox,
    QAction, QMenu, QMenuBar, QStatusBar, QTabWidget,
    QSplitter, QFrame, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
import cv2
import numpy as np

# Import our modules
from src.asl.detector_optimized import OptimizedASLDetector
from src.emotion.detector import EmotionDetector
from src.utils.camera import Camera
from src.history.manager import HistoryManager
from src.gui.about_dialog import AboutDialog
from src.gui.help_dialog import HelpDialog
from src.utils.text_to_speech import TextToSpeech
from src.utils.prediction_enhancer import PredictionEnhancer


class MainWindow(QMainWindow):
    """Main window for the ASL Translator and Emotion Communicator application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.camera = Camera()
        self.asl_detector = OptimizedASLDetector()
        self.emotion_detector = EmotionDetector()
        self.history_manager = HistoryManager()
        self.text_to_speech = TextToSpeech()
        
        # Initialize prediction enhancer with a default mapping file path
        mapping_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "data", "word_emotion_mapping.json")
        self.prediction_enhancer = PredictionEnhancer(mapping_file)
        
        # Setup UI
        self.init_ui()
        
        # Setup timer for camera updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (approx. 33 fps)
        
        # Initialize variables
        self.current_asl_text = ""
        self.current_emotion = ""
        self.is_recording = False
        self.text_to_speech_enabled = False
        self.last_spoken_text = ""  # Track last spoken text to avoid repetition
    
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle("ASL Translator and Emotion Communicator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Camera feed and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Camera feed display
        self.camera_label = QLabel("Camera feed will appear here")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #333; color: #ddd; border-radius: 10px;")
        left_layout.addWidget(self.camera_label)
        
        # Camera controls
        camera_controls_layout = QHBoxLayout()
        
        # Start/Stop button
        self.camera_button = QPushButton("Stop Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        camera_controls_layout.addWidget(self.camera_button)
        
        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        camera_controls_layout.addWidget(self.record_button)
        
        # Text-to-speech toggle
        self.tts_button = QPushButton("Enable Text-to-Speech")
        self.tts_button.clicked.connect(self.toggle_text_to_speech)
        camera_controls_layout.addWidget(self.tts_button)
        
        left_layout.addLayout(camera_controls_layout)
        
        # Right panel - Translation display and history
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Tabs for different views
        tabs = QTabWidget()
        
        # Translation tab
        translation_tab = QWidget()
        translation_layout = QVBoxLayout(translation_tab)
        
        # Current ASL translation
        translation_layout.addWidget(QLabel("ASL Translation:"))
        self.asl_text = QTextEdit()
        self.asl_text.setReadOnly(True)
        self.asl_text.setMinimumHeight(100)
        self.asl_text.setStyleSheet("font-size: 18pt; font-weight: bold;")
        translation_layout.addWidget(self.asl_text)
        
        # Current emotion
        translation_layout.addWidget(QLabel("Detected Emotion:"))
        self.emotion_text = QTextEdit()
        self.emotion_text.setReadOnly(True)
        self.emotion_text.setMaximumHeight(60)
        self.emotion_text.setStyleSheet("font-size: 16pt;")
        translation_layout.addWidget(self.emotion_text)
        
        # Add translation tab to tabs
        tabs.addTab(translation_tab, "Live Translation")
        
        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        # History display
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        history_layout.addWidget(self.history_text)
        
        # History controls
        history_controls_layout = QHBoxLayout()
        
        # Clear history button
        clear_history_button = QPushButton("Clear History")
        clear_history_button.clicked.connect(self.clear_history)
        history_controls_layout.addWidget(clear_history_button)
        
        # Save history button
        save_history_button = QPushButton("Save History")
        save_history_button.clicked.connect(self.save_history)
        history_controls_layout.addWidget(save_history_button)
        
        history_layout.addLayout(history_controls_layout)
        
        # Add history tab to tabs
        tabs.addTab(history_tab, "History")
        
        # Add tabs to right panel
        right_layout.addWidget(tabs)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # Set initial sizes for splitter
        splitter.setSizes([600, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_menu_bar(self):
        """Create the menu bar with actions."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # Save history action
        save_action = QAction("&Save History", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_history)
        file_menu.addAction(save_action)
        
        # Exit action
        exit_action = QAction("&Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = self.menuBar().addMenu("&Settings")
        
        # Camera settings action
        camera_action = QAction("&Camera Settings", self)
        camera_action.triggered.connect(self.open_camera_settings)
        settings_menu.addAction(camera_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # Help action
        help_action = QAction("&Help", self)
        help_action.setShortcut("F1")
        help_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(help_action)
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    

    
    def update_frame(self):
        """Update the camera frame and process for ASL and emotion detection."""
        # Get frame from camera
        ret, frame = self.camera.get_frame()
        
        if ret:
            # Process frame for emotion detection first
            emotion_result = self.emotion_detector.process_frame(frame, self.current_asl_text, self.prediction_enhancer)
            if emotion_result:
                self.current_emotion = emotion_result
                self.emotion_text.setText(self.current_emotion)
            
            # Process frame for ASL detection
            asl_result, annotated_frame = self.asl_detector.process_frame(frame)
            if asl_result:
                self.current_asl_text = asl_result
                self.asl_text.setText(self.current_asl_text)
                
                # Add to history if recording
                if self.is_recording:
                    self.history_manager.add_entry(self.current_asl_text, self.current_emotion)
                    self.update_history_display()
                
                # Text-to-speech if enabled
                if self.text_to_speech_enabled:
                    self.speak_text(self.current_asl_text)
            
            # Convert the (potentially annotated) frame to QImage for display
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale the image to fit the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.width(), 
                self.camera_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
    
    def toggle_camera(self):
        """Toggle the camera on/off."""
        if self.timer.isActive():
            self.timer.stop()
            self.camera_button.setText("Start Camera")
            self.camera_label.setText("Camera Stopped")
            self.camera_label.setPixmap(QPixmap())  # Clear the pixmap
        else:
            self.timer.start(30)
            self.camera_button.setText("Stop Camera")
    
    def toggle_recording(self):
        """Toggle recording of translations."""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_button.setText("Stop Recording")
            self.statusBar().showMessage("Recording started")
        else:
            self.record_button.setText("Start Recording")
            self.statusBar().showMessage("Recording stopped")
    
    def toggle_text_to_speech(self):
        """Toggle text-to-speech functionality."""
        self.text_to_speech_enabled = not self.text_to_speech_enabled
        if self.text_to_speech_enabled:
            self.tts_button.setText("Disable Text-to-Speech")
            self.statusBar().showMessage("Text-to-speech enabled")
        else:
            self.tts_button.setText("Enable Text-to-Speech")
            self.statusBar().showMessage("Text-to-speech disabled")
    
    def speak_text(self, text):
        """Speak the given text using text-to-speech."""
        # Avoid repeating the same text in quick succession
        if text != self.last_spoken_text:
            self.text_to_speech.speak(text)
            self.last_spoken_text = text
    
    def update_history_display(self):
        """Update the history display with the latest entries."""
        history_entries = self.history_manager.get_entries()
        self.history_text.clear()
        
        for entry in history_entries:
            timestamp = entry["timestamp"]
            asl_text = entry["asl_text"]
            emotion = entry["emotion"]
            
            # Format the entry
            formatted_entry = f"[{timestamp}] ASL: {asl_text} | Emotion: {emotion}\n"
            self.history_text.append(formatted_entry)
    
    def clear_history(self):
        """Clear the translation history."""
        self.history_manager.clear_entries()
        self.history_text.clear()
        self.statusBar().showMessage("History cleared")
    
    def save_history(self):
        """Save the translation history to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save History", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            success = self.history_manager.save_to_file(file_path)
            if success:
                self.statusBar().showMessage(f"History saved to {file_path}")
            else:
                QMessageBox.warning(
                    self, "Save Error", "Failed to save history to file."
                )
    
    def open_camera_settings(self):
        """Open camera settings dialog."""
        # This will be implemented later
        QMessageBox.information(
            self, "Camera Settings", "Camera settings dialog not implemented yet."
        )
    
    def show_about_dialog(self):
        """Show the about dialog."""
        about_dialog = AboutDialog(self)
        about_dialog.exec_()
    
    def show_help_dialog(self):
        """Show the help dialog."""
        help_dialog = HelpDialog(self)
        help_dialog.exec_()
    
    def closeEvent(self, event):
        """Handle the window close event."""
        # Stop the camera and timer
        if self.timer.isActive():
            self.timer.stop()
        
        # Release the camera
        self.camera.release()
        
        # Accept the event to close the window
        event.accept()