#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Utility for ASL Translator and Emotion Communicator

This module provides a Camera class for handling webcam operations.
"""

import os
import sys
import cv2
import numpy as np
import time
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_assets_dir


class Camera:
    """Camera class for handling webcam operations."""
    
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize the camera.
        
        Args:
            camera_id (int): The ID of the camera to use (default: 0 for primary webcam)
            width (int): The width to set for the camera capture (default: 640)
            height (int): The height to set for the camera capture (default: 480)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.initialize()
    
    def initialize(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        
        # Allow camera to warm up
        time.sleep(0.5)
        return True
    
    def get_frame(self):
        """Get a frame from the camera.
        
        Returns:
            tuple: (success, frame) where success is a boolean indicating if the frame was
                  successfully captured, and frame is the captured frame as a numpy array.
                  If no camera is available, returns a placeholder image.
        """
        if self.cap is None or not self.cap.isOpened():
            # Return a placeholder image when no camera is available
            return True, self.get_no_camera_image()
        
        # Read a frame
        ret, frame = self.cap.read()
        
        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Failed to capture frame.")
            # Return a placeholder image when frame capture fails
            return True, self.get_no_camera_image()
        
        return ret, frame
    
    def get_no_camera_image(self):
        """Get a placeholder image when no camera is available.
        
        Returns:
            numpy.ndarray: A placeholder image as a numpy array.
        """
        # Path to the no_camera SVG image
        no_camera_path = os.path.join(get_assets_dir(), "no_camera.svg")
        
        # Create a placeholder image
        if os.path.exists(no_camera_path):
            # Load SVG and convert to QImage
            renderer = QSvgRenderer(no_camera_path)
            image = QImage(self.width, self.height, QImage.Format_ARGB32)
            image.fill(0)  # Transparent background
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            
            # Convert QImage to numpy array
            ptr = image.constBits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(image.height(), image.width(), 4)  # RGBA
            
            # Convert RGBA to BGR (OpenCV format)
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            return bgr_image
        else:
            # If SVG file doesn't exist, create a simple placeholder
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Add text to the image
            cv2.putText(image, "No Camera Available", (int(self.width/2) - 150, int(self.height/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, "Please connect a camera and restart the application", 
                        (int(self.width/2) - 250, int(self.height/2) + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            return image
    
    def release(self):
        """Release the camera resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        """Destructor to ensure camera resources are released."""
        self.release()
    
    def get_camera_properties(self):
        """Get the properties of the camera.
        
        Returns:
            dict: A dictionary of camera properties.
        """
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        properties = {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "hue": self.cap.get(cv2.CAP_PROP_HUE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
        }
        
        return properties
    
    def set_camera_property(self, property_id, value):
        """Set a camera property.
        
        Args:
            property_id: The OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: The value to set for the property
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return False
        
        return self.cap.set(property_id, value)
    
    def list_available_cameras(self):
        """List all available cameras on the system.
        
        Returns:
            list: A list of available camera IDs
        """
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        return available_cameras