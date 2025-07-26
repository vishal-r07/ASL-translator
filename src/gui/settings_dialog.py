#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Settings Dialog

This module provides a dialog for configuring application settings.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel,
    QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QSlider, QGroupBox, QFormLayout, QDialogButtonBox, QFileDialog,
    QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions and configuration
from src.utils.config import get_config, set_config_value, reset_config
from src.utils.common import get_assets_dir


class SettingsDialog(QDialog):
    """
    Dialog for configuring application settings.
    """
    
    # Signal emitted when settings are applied
    settings_applied = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the settings dialog.
        
        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(600, 500)
        
        # Get current configuration
        self.config = get_config()
        self.current_settings = self.config.get_all()
        self.modified_settings = {}
        
        # Initialize UI
        self._init_ui()
    
    def _init_ui(self):
        """
        Initialize the user interface.
        """
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self._create_general_tab()
        self._create_camera_tab()
        self._create_asl_tab()
        self._create_emotion_tab()
        self._create_display_tab()
        self._create_tts_tab()
        self._create_advanced_tab()
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
        # Create button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply | QDialogButtonBox.Reset)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        button_box.button(QDialogButtonBox.Reset).clicked.connect(self._reset_settings)
        
        # Add button box to main layout
        main_layout.addWidget(button_box)
    
    def _create_general_tab(self):
        """
        Create the general settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # General settings group
        group = QGroupBox("General Settings")
        form_layout = QFormLayout(group)
        
        # Dark mode
        self.dark_mode_checkbox = QCheckBox()
        self.dark_mode_checkbox.setChecked(self.current_settings.get("dark_mode", True))
        form_layout.addRow("Dark Mode:", self.dark_mode_checkbox)
        
        # Save history
        self.save_history_checkbox = QCheckBox()
        self.save_history_checkbox.setChecked(self.current_settings.get("save_history", True))
        form_layout.addRow("Save Translation History:", self.save_history_checkbox)
        
        # History max entries
        self.history_max_entries_spinbox = QSpinBox()
        self.history_max_entries_spinbox.setRange(10, 10000)
        self.history_max_entries_spinbox.setValue(self.current_settings.get("history_max_entries", 1000))
        form_layout.addRow("Maximum History Entries:", self.history_max_entries_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "General")
    
    def _create_camera_tab(self):
        """
        Create the camera settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Camera settings group
        group = QGroupBox("Camera Settings")
        form_layout = QFormLayout(group)
        
        # Camera index
        self.camera_index_spinbox = QSpinBox()
        self.camera_index_spinbox.setRange(0, 10)
        self.camera_index_spinbox.setValue(self.current_settings.get("camera_index", 0))
        form_layout.addRow("Camera Index:", self.camera_index_spinbox)
        
        # Camera width
        self.camera_width_spinbox = QSpinBox()
        self.camera_width_spinbox.setRange(320, 1920)
        self.camera_width_spinbox.setSingleStep(80)
        self.camera_width_spinbox.setValue(self.current_settings.get("camera_width", 640))
        form_layout.addRow("Camera Width:", self.camera_width_spinbox)
        
        # Camera height
        self.camera_height_spinbox = QSpinBox()
        self.camera_height_spinbox.setRange(240, 1080)
        self.camera_height_spinbox.setSingleStep(60)
        self.camera_height_spinbox.setValue(self.current_settings.get("camera_height", 480))
        form_layout.addRow("Camera Height:", self.camera_height_spinbox)
        
        # Camera FPS
        self.camera_fps_spinbox = QSpinBox()
        self.camera_fps_spinbox.setRange(10, 60)
        self.camera_fps_spinbox.setValue(self.current_settings.get("camera_fps", 30))
        form_layout.addRow("Camera FPS:", self.camera_fps_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "Camera")
    
    def _create_asl_tab(self):
        """
        Create the ASL detection settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # ASL detection settings group
        group = QGroupBox("ASL Detection Settings")
        form_layout = QFormLayout(group)
        
        # ASL detection enabled
        self.asl_detection_enabled_checkbox = QCheckBox()
        self.asl_detection_enabled_checkbox.setChecked(self.current_settings.get("asl_detection_enabled", True))
        form_layout.addRow("Enable ASL Detection:", self.asl_detection_enabled_checkbox)
        
        # ASL confidence threshold
        self.asl_confidence_threshold_spinbox = QDoubleSpinBox()
        self.asl_confidence_threshold_spinbox.setRange(0.1, 1.0)
        self.asl_confidence_threshold_spinbox.setSingleStep(0.05)
        self.asl_confidence_threshold_spinbox.setDecimals(2)
        self.asl_confidence_threshold_spinbox.setValue(self.current_settings.get("asl_confidence_threshold", 0.7))
        form_layout.addRow("Confidence Threshold:", self.asl_confidence_threshold_spinbox)
        
        # ASL temporal window
        self.asl_temporal_window_spinbox = QSpinBox()
        self.asl_temporal_window_spinbox.setRange(1, 20)
        self.asl_temporal_window_spinbox.setValue(self.current_settings.get("asl_temporal_window", 5))
        form_layout.addRow("Temporal Window (frames):", self.asl_temporal_window_spinbox)
        
        # ASL min detection confidence
        self.asl_min_detection_confidence_spinbox = QDoubleSpinBox()
        self.asl_min_detection_confidence_spinbox.setRange(0.1, 1.0)
        self.asl_min_detection_confidence_spinbox.setSingleStep(0.05)
        self.asl_min_detection_confidence_spinbox.setDecimals(2)
        self.asl_min_detection_confidence_spinbox.setValue(self.current_settings.get("asl_min_detection_confidence", 0.5))
        form_layout.addRow("Min Detection Confidence:", self.asl_min_detection_confidence_spinbox)
        
        # ASL min tracking confidence
        self.asl_min_tracking_confidence_spinbox = QDoubleSpinBox()
        self.asl_min_tracking_confidence_spinbox.setRange(0.1, 1.0)
        self.asl_min_tracking_confidence_spinbox.setSingleStep(0.05)
        self.asl_min_tracking_confidence_spinbox.setDecimals(2)
        self.asl_min_tracking_confidence_spinbox.setValue(self.current_settings.get("asl_min_tracking_confidence", 0.5))
        form_layout.addRow("Min Tracking Confidence:", self.asl_min_tracking_confidence_spinbox)
        
        # ASL max num hands
        self.asl_max_num_hands_spinbox = QSpinBox()
        self.asl_max_num_hands_spinbox.setRange(1, 4)
        self.asl_max_num_hands_spinbox.setValue(self.current_settings.get("asl_max_num_hands", 2))
        form_layout.addRow("Max Number of Hands:", self.asl_max_num_hands_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "ASL Detection")
    
    def _create_emotion_tab(self):
        """
        Create the emotion detection settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Emotion detection settings group
        group = QGroupBox("Emotion Detection Settings")
        form_layout = QFormLayout(group)
        
        # Emotion detection enabled
        self.emotion_detection_enabled_checkbox = QCheckBox()
        self.emotion_detection_enabled_checkbox.setChecked(self.current_settings.get("emotion_detection_enabled", True))
        form_layout.addRow("Enable Emotion Detection:", self.emotion_detection_enabled_checkbox)
        
        # Emotion confidence threshold
        self.emotion_confidence_threshold_spinbox = QDoubleSpinBox()
        self.emotion_confidence_threshold_spinbox.setRange(0.1, 1.0)
        self.emotion_confidence_threshold_spinbox.setSingleStep(0.05)
        self.emotion_confidence_threshold_spinbox.setDecimals(2)
        self.emotion_confidence_threshold_spinbox.setValue(self.current_settings.get("emotion_confidence_threshold", 0.6))
        form_layout.addRow("Confidence Threshold:", self.emotion_confidence_threshold_spinbox)
        
        # Emotion temporal window
        self.emotion_temporal_window_spinbox = QSpinBox()
        self.emotion_temporal_window_spinbox.setRange(1, 30)
        self.emotion_temporal_window_spinbox.setValue(self.current_settings.get("emotion_temporal_window", 10))
        form_layout.addRow("Temporal Window (frames):", self.emotion_temporal_window_spinbox)
        
        # Emotion min detection confidence
        self.emotion_min_detection_confidence_spinbox = QDoubleSpinBox()
        self.emotion_min_detection_confidence_spinbox.setRange(0.1, 1.0)
        self.emotion_min_detection_confidence_spinbox.setSingleStep(0.05)
        self.emotion_min_detection_confidence_spinbox.setDecimals(2)
        self.emotion_min_detection_confidence_spinbox.setValue(self.current_settings.get("emotion_min_detection_confidence", 0.5))
        form_layout.addRow("Min Detection Confidence:", self.emotion_min_detection_confidence_spinbox)
        
        # Emotion min tracking confidence
        self.emotion_min_tracking_confidence_spinbox = QDoubleSpinBox()
        self.emotion_min_tracking_confidence_spinbox.setRange(0.1, 1.0)
        self.emotion_min_tracking_confidence_spinbox.setSingleStep(0.05)
        self.emotion_min_tracking_confidence_spinbox.setDecimals(2)
        self.emotion_min_tracking_confidence_spinbox.setValue(self.current_settings.get("emotion_min_tracking_confidence", 0.5))
        form_layout.addRow("Min Tracking Confidence:", self.emotion_min_tracking_confidence_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "Emotion Detection")
    
    def _create_display_tab(self):
        """
        Create the display settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Display settings group
        group = QGroupBox("Display Settings")
        form_layout = QFormLayout(group)
        
        # Show landmarks
        self.show_landmarks_checkbox = QCheckBox()
        self.show_landmarks_checkbox.setChecked(self.current_settings.get("show_landmarks", True))
        form_layout.addRow("Show Landmarks:", self.show_landmarks_checkbox)
        
        # Show FPS
        self.show_fps_checkbox = QCheckBox()
        self.show_fps_checkbox.setChecked(self.current_settings.get("show_fps", True))
        form_layout.addRow("Show FPS:", self.show_fps_checkbox)
        
        # Text size
        self.text_size_spinbox = QDoubleSpinBox()
        self.text_size_spinbox.setRange(0.5, 2.0)
        self.text_size_spinbox.setSingleStep(0.1)
        self.text_size_spinbox.setDecimals(1)
        self.text_size_spinbox.setValue(self.current_settings.get("text_size", 1.0))
        form_layout.addRow("Text Size Multiplier:", self.text_size_spinbox)
        
        # Animation speed
        self.animation_speed_spinbox = QDoubleSpinBox()
        self.animation_speed_spinbox.setRange(0.5, 2.0)
        self.animation_speed_spinbox.setSingleStep(0.1)
        self.animation_speed_spinbox.setDecimals(1)
        self.animation_speed_spinbox.setValue(self.current_settings.get("animation_speed", 1.0))
        form_layout.addRow("Animation Speed Multiplier:", self.animation_speed_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "Display")
    
    def _create_tts_tab(self):
        """
        Create the text-to-speech settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Text-to-speech settings group
        group = QGroupBox("Text-to-Speech Settings")
        form_layout = QFormLayout(group)
        
        # TTS enabled
        self.tts_enabled_checkbox = QCheckBox()
        self.tts_enabled_checkbox.setChecked(self.current_settings.get("tts_enabled", True))
        form_layout.addRow("Enable Text-to-Speech:", self.tts_enabled_checkbox)
        
        # TTS rate
        self.tts_rate_spinbox = QSpinBox()
        self.tts_rate_spinbox.setRange(50, 300)
        self.tts_rate_spinbox.setValue(self.current_settings.get("tts_rate", 150))
        form_layout.addRow("Speech Rate (words per minute):", self.tts_rate_spinbox)
        
        # TTS volume
        self.tts_volume_spinbox = QDoubleSpinBox()
        self.tts_volume_spinbox.setRange(0.0, 1.0)
        self.tts_volume_spinbox.setSingleStep(0.1)
        self.tts_volume_spinbox.setDecimals(1)
        self.tts_volume_spinbox.setValue(self.current_settings.get("tts_volume", 1.0))
        form_layout.addRow("Speech Volume:", self.tts_volume_spinbox)
        
        # TTS voice index
        self.tts_voice_index_spinbox = QSpinBox()
        self.tts_voice_index_spinbox.setRange(0, 10)
        self.tts_voice_index_spinbox.setValue(self.current_settings.get("tts_voice_index", 0))
        form_layout.addRow("Voice Index:", self.tts_voice_index_spinbox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "Text-to-Speech")
    
    def _create_advanced_tab(self):
        """
        Create the advanced settings tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Advanced settings group
        group = QGroupBox("Advanced Settings")
        form_layout = QFormLayout(group)
        
        # Use TFLite
        self.use_tflite_checkbox = QCheckBox()
        self.use_tflite_checkbox.setChecked(self.current_settings.get("use_tflite", True))
        form_layout.addRow("Use TensorFlow Lite:", self.use_tflite_checkbox)
        
        # Enable GPU
        self.enable_gpu_checkbox = QCheckBox()
        self.enable_gpu_checkbox.setChecked(self.current_settings.get("enable_gpu", True))
        form_layout.addRow("Enable GPU Acceleration:", self.enable_gpu_checkbox)
        
        # Debug mode
        self.debug_mode_checkbox = QCheckBox()
        self.debug_mode_checkbox.setChecked(self.current_settings.get("debug_mode", False))
        form_layout.addRow("Debug Mode:", self.debug_mode_checkbox)
        
        # Log level
        self.log_level_combobox = QComboBox()
        self.log_level_combobox.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        log_level = self.current_settings.get("log_level", "INFO")
        index = self.log_level_combobox.findText(log_level)
        if index >= 0:
            self.log_level_combobox.setCurrentIndex(index)
        form_layout.addRow("Log Level:", self.log_level_combobox)
        
        # Add group to layout
        layout.addWidget(group)
        
        # Add stretch to push widgets to the top
        layout.addStretch(1)
        
        # Add tab to tab widget
        self.tab_widget.addTab(tab, "Advanced")
    
    def _get_modified_settings(self) -> Dict[str, Any]:
        """
        Get the modified settings.
        
        Returns:
            Dict[str, Any]: Dictionary of modified settings.
        """
        modified = {}
        
        # General settings
        modified["dark_mode"] = self.dark_mode_checkbox.isChecked()
        modified["save_history"] = self.save_history_checkbox.isChecked()
        modified["history_max_entries"] = self.history_max_entries_spinbox.value()
        
        # Camera settings
        modified["camera_index"] = self.camera_index_spinbox.value()
        modified["camera_width"] = self.camera_width_spinbox.value()
        modified["camera_height"] = self.camera_height_spinbox.value()
        modified["camera_fps"] = self.camera_fps_spinbox.value()
        
        # ASL detection settings
        modified["asl_detection_enabled"] = self.asl_detection_enabled_checkbox.isChecked()
        modified["asl_confidence_threshold"] = self.asl_confidence_threshold_spinbox.value()
        modified["asl_temporal_window"] = self.asl_temporal_window_spinbox.value()
        modified["asl_min_detection_confidence"] = self.asl_min_detection_confidence_spinbox.value()
        modified["asl_min_tracking_confidence"] = self.asl_min_tracking_confidence_spinbox.value()
        modified["asl_max_num_hands"] = self.asl_max_num_hands_spinbox.value()
        
        # Emotion detection settings
        modified["emotion_detection_enabled"] = self.emotion_detection_enabled_checkbox.isChecked()
        modified["emotion_confidence_threshold"] = self.emotion_confidence_threshold_spinbox.value()
        modified["emotion_temporal_window"] = self.emotion_temporal_window_spinbox.value()
        modified["emotion_min_detection_confidence"] = self.emotion_min_detection_confidence_spinbox.value()
        modified["emotion_min_tracking_confidence"] = self.emotion_min_tracking_confidence_spinbox.value()
        
        # Display settings
        modified["show_landmarks"] = self.show_landmarks_checkbox.isChecked()
        modified["show_fps"] = self.show_fps_checkbox.isChecked()
        modified["text_size"] = self.text_size_spinbox.value()
        modified["animation_speed"] = self.animation_speed_spinbox.value()
        
        # Text-to-speech settings
        modified["tts_enabled"] = self.tts_enabled_checkbox.isChecked()
        modified["tts_rate"] = self.tts_rate_spinbox.value()
        modified["tts_volume"] = self.tts_volume_spinbox.value()
        modified["tts_voice_index"] = self.tts_voice_index_spinbox.value()
        
        # Advanced settings
        modified["use_tflite"] = self.use_tflite_checkbox.isChecked()
        modified["enable_gpu"] = self.enable_gpu_checkbox.isChecked()
        modified["debug_mode"] = self.debug_mode_checkbox.isChecked()
        modified["log_level"] = self.log_level_combobox.currentText()
        
        return modified
    
    def _apply_settings(self):
        """
        Apply the modified settings.
        """
        # Get modified settings
        modified = self._get_modified_settings()
        
        # Update configuration
        self.config.update(modified)
        self.config.save_config()
        
        # Emit signal
        self.settings_applied.emit(modified)
        
        # Show confirmation message
        QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully.")
    
    def _reset_settings(self):
        """
        Reset settings to defaults.
        """
        # Confirm reset
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset configuration
            reset_config()
            
            # Reload settings
            self.current_settings = self.config.get_all()
            
            # Update UI
            self._update_ui_from_settings()
            
            # Show confirmation message
            QMessageBox.information(self, "Settings Reset", "Settings have been reset to defaults.")
    
    def _update_ui_from_settings(self):
        """
        Update UI elements from current settings.
        """
        # General settings
        self.dark_mode_checkbox.setChecked(self.current_settings.get("dark_mode", True))
        self.save_history_checkbox.setChecked(self.current_settings.get("save_history", True))
        self.history_max_entries_spinbox.setValue(self.current_settings.get("history_max_entries", 1000))
        
        # Camera settings
        self.camera_index_spinbox.setValue(self.current_settings.get("camera_index", 0))
        self.camera_width_spinbox.setValue(self.current_settings.get("camera_width", 640))
        self.camera_height_spinbox.setValue(self.current_settings.get("camera_height", 480))
        self.camera_fps_spinbox.setValue(self.current_settings.get("camera_fps", 30))
        
        # ASL detection settings
        self.asl_detection_enabled_checkbox.setChecked(self.current_settings.get("asl_detection_enabled", True))
        self.asl_confidence_threshold_spinbox.setValue(self.current_settings.get("asl_confidence_threshold", 0.7))
        self.asl_temporal_window_spinbox.setValue(self.current_settings.get("asl_temporal_window", 5))
        self.asl_min_detection_confidence_spinbox.setValue(self.current_settings.get("asl_min_detection_confidence", 0.5))
        self.asl_min_tracking_confidence_spinbox.setValue(self.current_settings.get("asl_min_tracking_confidence", 0.5))
        self.asl_max_num_hands_spinbox.setValue(self.current_settings.get("asl_max_num_hands", 2))
        
        # Emotion detection settings
        self.emotion_detection_enabled_checkbox.setChecked(self.current_settings.get("emotion_detection_enabled", True))
        self.emotion_confidence_threshold_spinbox.setValue(self.current_settings.get("emotion_confidence_threshold", 0.6))
        self.emotion_temporal_window_spinbox.setValue(self.current_settings.get("emotion_temporal_window", 10))
        self.emotion_min_detection_confidence_spinbox.setValue(self.current_settings.get("emotion_min_detection_confidence", 0.5))
        self.emotion_min_tracking_confidence_spinbox.setValue(self.current_settings.get("emotion_min_tracking_confidence", 0.5))
        
        # Display settings
        self.show_landmarks_checkbox.setChecked(self.current_settings.get("show_landmarks", True))
        self.show_fps_checkbox.setChecked(self.current_settings.get("show_fps", True))
        self.text_size_spinbox.setValue(self.current_settings.get("text_size", 1.0))
        self.animation_speed_spinbox.setValue(self.current_settings.get("animation_speed", 1.0))
        
        # Text-to-speech settings
        self.tts_enabled_checkbox.setChecked(self.current_settings.get("tts_enabled", True))
        self.tts_rate_spinbox.setValue(self.current_settings.get("tts_rate", 150))
        self.tts_volume_spinbox.setValue(self.current_settings.get("tts_volume", 1.0))
        self.tts_voice_index_spinbox.setValue(self.current_settings.get("tts_voice_index", 0))
        
        # Advanced settings
        self.use_tflite_checkbox.setChecked(self.current_settings.get("use_tflite", True))
        self.enable_gpu_checkbox.setChecked(self.current_settings.get("enable_gpu", True))
        self.debug_mode_checkbox.setChecked(self.current_settings.get("debug_mode", False))
        log_level = self.current_settings.get("log_level", "INFO")
        index = self.log_level_combobox.findText(log_level)
        if index >= 0:
            self.log_level_combobox.setCurrentIndex(index)
    
    def accept(self):
        """
        Handle dialog acceptance.
        """
        # Apply settings
        self._apply_settings()
        
        # Close dialog
        super().accept()