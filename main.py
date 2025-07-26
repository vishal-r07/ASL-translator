#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASL Translator and Emotion Communicator

A desktop application that translates American Sign Language (ASL) gestures and
facial emotions in real-time using a webcam.
"""

import sys
import os
import logging

# Add the project directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gui.main_window import MainWindow
from src.gui.splash_screen import create_splash_screen, show_splash_screen
from PyQt5.QtWidgets import QApplication
import qdarktheme


def run_app():
    """Run the application with splash screen."""
    # Create the application
    app = QApplication(sys.argv)
    
    # Apply dark theme
    qdarktheme.setup_theme("dark")
    
    # Create splash screen
    splash = create_splash_screen()
    
    # Define loading tasks
    loading_tasks = [
        ("Initializing application...", 10),
        ("Loading configuration...", 20),
        ("Initializing camera...", 30),
        ("Loading ASL detection model...", 50),
        ("Loading emotion detection model...", 70),
        ("Initializing user interface...", 90),
        ("Ready!", 100)
    ]
    
    # Show splash screen and create main window
    window = show_splash_screen(splash, app, MainWindow, loading_tasks)
    
    # Start the event loop
    return app.exec_()


def main():
    """Main entry point for the application."""
    sys.exit(run_app())


if __name__ == "__main__":
    main()