#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Utilities for ASL Translator and Emotion Communicator

This module provides common utility functions and constants used throughout the application.
"""

import os
import sys
import time
import datetime
import cv2
import numpy as np


# Application constants
APP_NAME = "ASL Translator and Emotion Communicator"
APP_VERSION = "1.0.0"
APP_AUTHOR = "ASL Translator Team"


def get_app_root_dir():
    """
    Get the root directory of the application.
    
    Returns:
        str: The absolute path to the application root directory.
    """
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as a script
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_models_dir():
    """
    Get the directory containing the models.
    
    Returns:
        str: The absolute path to the models directory.
    """
    return os.path.join(get_app_root_dir(), "models")


def get_assets_dir():
    """
    Get the directory containing the assets.
    
    Returns:
        str: The absolute path to the assets directory.
    """
    return os.path.join(get_app_root_dir(), "assets")


def get_data_dir():
    """
    Get the directory containing the data.
    
    Returns:
        str: The absolute path to the data directory.
    """
    return os.path.join(get_app_root_dir(), "data")


def get_user_data_dir():
    """
    Get the directory for storing user data.
    
    Returns:
        str: The absolute path to the user data directory.
    """
    # Use the user's documents folder for storing user data
    user_data_dir = os.path.join(os.path.expanduser("~"), "Documents", APP_NAME)
    
    # Create the directory if it doesn't exist
    os.makedirs(user_data_dir, exist_ok=True)
    
    return user_data_dir


def get_timestamp():
    """
    Get a formatted timestamp for the current time.
    
    Returns:
        str: A formatted timestamp string.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_filename_timestamp():
    """
    Get a formatted timestamp suitable for use in filenames.
    
    Returns:
        str: A formatted timestamp string for filenames.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image to the specified width and height.
    
    Args:
        image (numpy.ndarray): The image to resize.
        width (int, optional): The desired width. If None, the aspect ratio is preserved.
        height (int, optional): The desired height. If None, the aspect ratio is preserved.
        inter (int, optional): The interpolation method to use.
        
    Returns:
        numpy.ndarray: The resized image.
    """
    # Initialize the dimensions of the image to be resized
    (h, w) = image.shape[:2]
    
    # If both width and height are None, return the original image
    if width is None and height is None:
        return image
    
    # If width is None, calculate it from the height while preserving aspect ratio
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # If height is None, calculate it from the width while preserving aspect ratio
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    # If both width and height are specified, use them directly
    else:
        dim = (width, height)
    
    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized


def draw_text(image, text, position, font_scale=1.0, color=(255, 255, 255),
              thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, background=None):
    """
    Draw text on an image with optional background.
    
    Args:
        image (numpy.ndarray): The image to draw on.
        text (str): The text to draw.
        position (tuple): The position (x, y) to draw the text.
        font_scale (float, optional): The font scale.
        color (tuple, optional): The text color (B, G, R).
        thickness (int, optional): The text thickness.
        font (int, optional): The font to use.
        background (tuple, optional): The background color (B, G, R). If None, no background is drawn.
        
    Returns:
        numpy.ndarray: The image with the text drawn on it.
    """
    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw the background if specified
    if background is not None:
        cv2.rectangle(
            image,
            (position[0], position[1] - text_height - baseline),
            (position[0] + text_width, position[1] + baseline),
            background,
            -1
        )
    
    # Draw the text
    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness
    )
    
    return image


def overlay_transparent(background, overlay, x, y):
    """
    Overlay a transparent PNG image on top of another image.
    
    Args:
        background (numpy.ndarray): The background image.
        overlay (numpy.ndarray): The overlay image with transparency (BGRA).
        x (int): The x-coordinate to place the overlay.
        y (int): The y-coordinate to place the overlay.
        
    Returns:
        numpy.ndarray: The combined image.
    """
    # Extract the alpha channel from the overlay
    if overlay.shape[2] == 4:
        # The overlay has an alpha channel (BGRA)
        overlay_alpha = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]
    else:
        # The overlay doesn't have an alpha channel (BGR)
        overlay_alpha = np.ones(overlay.shape[:2], dtype=float)
        overlay_rgb = overlay
    
    # Image ranges
    y1, y2 = max(0, y), min(background.shape[0], y + overlay_rgb.shape[0])
    x1, x2 = max(0, x), min(background.shape[1], x + overlay_rgb.shape[1])
    
    # Overlay ranges
    y1o, y2o = max(0, -y), min(overlay_rgb.shape[0], background.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay_rgb.shape[1], background.shape[1] - x)
    
    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return background
    
    # Blend the overlay with the background
    alpha = overlay_alpha[y1o:y2o, x1o:x2o][..., np.newaxis]
    background[y1:y2, x1:x2] = (1 - alpha) * background[y1:y2, x1:x2] + alpha * overlay_rgb[y1o:y2o, x1o:x2o]
    
    return background


def create_animated_text(text, duration=2.0, fade_in=0.5, fade_out=0.5):
    """
    Create an animated text effect with fade in and fade out.
    
    Args:
        text (str): The text to animate.
        duration (float, optional): The total duration of the animation in seconds.
        fade_in (float, optional): The fade in duration in seconds.
        fade_out (float, optional): The fade out duration in seconds.
        
    Returns:
        generator: A generator that yields (text, alpha) tuples over time.
    """
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        elapsed = time.time() - start_time
        remaining = end_time - time.time()
        
        # Calculate alpha based on elapsed time
        if elapsed < fade_in:
            # Fade in
            alpha = elapsed / fade_in
        elif remaining < fade_out:
            # Fade out
            alpha = remaining / fade_out
        else:
            # Full opacity
            alpha = 1.0
        
        yield text, alpha
        
        # Sleep a bit to avoid consuming too much CPU
        time.sleep(0.03)  # ~30 fps