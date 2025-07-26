#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Splash Screen

This module provides a splash screen to display while the application is loading.
"""

import os
import sys
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QLabel, QWidget
from PyQt5.QtCore import Qt, QSize, QTimer, QRect, QRectF
from PyQt5.QtGui import QPixmap, QFont, QPainter, QColor
from PyQt5.QtSvg import QSvgRenderer

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions and configuration
from src.utils.config import get_config
from src.utils.common import get_assets_dir


class SplashScreen(QSplashScreen):
    """
    Splash screen to display while the application is loading.
    """
    
    def __init__(self):
        """
        Initialize the splash screen.
        """
        # Get configuration
        self.config = get_config()
        
        # Create pixmap for splash screen
        logo_path = os.path.join(get_assets_dir(), "logo.svg")
        loading_path = os.path.join(get_assets_dir(), "loading.svg")
        
        # Create a pixmap for the splash screen
        pixmap = QPixmap(500, 400)
        pixmap.fill(Qt.transparent)
        
        # Create painter for the pixmap
        painter = QPainter(pixmap)
        
        # Draw background
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 220))
        
        # Draw logo if it exists
        if os.path.exists(logo_path):
            logo_renderer = QSvgRenderer(logo_path)
            logo_renderer.render(painter, QRectF(50, 50, 400, 200))
        
        # Draw loading animation if it exists
        if os.path.exists(loading_path):
            self.loading_renderer = QSvgRenderer(loading_path)
            self.loading_renderer.render(painter, QRectF(150, 250, 200, 100))
        
        # End painting
        painter.end()
        
        super().__init__(pixmap)
        
        # Create progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, pixmap.height() - 40, pixmap.width() - 40, 20)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    
    def drawContents(self, painter):
        """
        Draw the contents of the splash screen.
        
        Args:
            painter: QPainter object.
        """
        # Get application name and version
        app_name = self.config.get("app_name", "ASL Translator and Emotion Communicator")
        app_version = self.config.get("app_version", "1.0.0")
        
        # Draw background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        
        # Draw application name
        painter.setPen(Qt.white)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, f"\n{app_name}")
        
        # Draw version
        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignHCenter | Qt.AlignTop, f"\n\n\nVersion {app_version}")
        
        # Draw loading message
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -50),
            Qt.AlignHCenter | Qt.AlignBottom,
            f"Loading... {self.progress_bar.value()}%"
        )
    
    def set_progress(self, value):
        """
        Set the progress value.
        
        Args:
            value: Progress value (0-100).
        """
        self.progress_bar.setValue(value)
        self.repaint()
    
    def show_message(self, message):
        """
        Show a message on the splash screen.
        
        Args:
            message: Message to display.
        """
        self.showMessage(message, Qt.AlignBottom | Qt.AlignHCenter, Qt.white)


def create_splash_screen():
    """
    Create and return a splash screen.
    
    Returns:
        SplashScreen: The created splash screen.
    """
    return SplashScreen()


def show_splash_screen(splash_screen, app, main_window_class, loading_tasks=None):
    """
    Show the splash screen and perform loading tasks.
    
    Args:
        splash_screen: The splash screen to show.
        app: The QApplication instance.
        main_window_class: The main window class to instantiate.
        loading_tasks: List of loading tasks to perform.
        
    Returns:
        The created main window instance.
    """
    # Show splash screen
    splash_screen.show()
    app.processEvents()
    
    # Default loading tasks
    if loading_tasks is None:
        loading_tasks = [
            ("Initializing application...", 10),
            ("Loading configuration...", 20),
            ("Initializing camera...", 30),
            ("Loading ASL detection model...", 50),
            ("Loading emotion detection model...", 70),
            ("Initializing user interface...", 90),
            ("Ready!", 100)
        ]
    
    # Perform loading tasks
    for message, progress in loading_tasks:
        splash_screen.show_message(message)
        splash_screen.set_progress(progress)
        app.processEvents()
        
        # Simulate loading time
        QTimer.singleShot(300, lambda: None)
        app.processEvents()
    
    # Create main window
    main_window = main_window_class()
    
    # Finish splash screen
    splash_screen.finish(main_window)
    
    # Show main window
    main_window.show()
    
    return main_window