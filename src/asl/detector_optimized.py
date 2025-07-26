#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized ASL Detector for High-Accuracy Translation

This module provides the OptimizedASLDetector class, designed to work with the
high-accuracy model trained by `train_model_optimized.py`.
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import time

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_models_dir

class OptimizedASLDetector:
    """Optimized ASL Detector for high-accuracy gesture classification."""
    
    def __init__(self, model_name="asl_model_optimized.h5", confidence_threshold=0.9):
        """
        Initialize the optimized ASL detector.
        
        Args:
            model_name (str): Name of the model file to load.
            confidence_threshold (float): Threshold for detection confidence.
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Optimized for single-hand gestures
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.confidence_threshold = confidence_threshold
        self.model_loaded = False
        self.model = None
        self.label_encoder = None
        
        # Load the model and label encoder
        self._load_model_and_encoder(model_name)
        
        # Temporal filtering for stable predictions
        self.last_gesture = None
        self.gesture_start_time = None
        self.gesture_hold_time = 0.2  # Increased hold time for stability
        self.last_confirmed_time = 0
        self.cooldown_period = 1.5  # Cooldown to prevent rapid re-triggering

    def _load_model_and_encoder(self, model_name):
        """Load the TensorFlow model and the label encoder."""
        models_dir = get_models_dir()
        model_path = os.path.join(models_dir, model_name)
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found at {model_path}")
            return
        
        if not os.path.exists(encoder_path):
            print(f"❌ Error: Label encoder not found at {encoder_path}")
            return

        try:
            self.model = load_model(model_path)
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.model_loaded = True
            print(f"✅ Optimized ASL model '{model_name}' loaded successfully.")
            print(f"✅ Loaded label encoder with classes: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"❌ Error loading model or encoder: {e}")

    def process_frame(self, frame):
        """
        Process a single frame to detect and classify an ASL gesture.
        
        Args:
            frame (numpy.ndarray): The input video frame.
            
        Returns:
            tuple: (confirmed_gesture, annotated_frame)
        """
        if not self.model_loaded:
            return None, frame

        # Flip the frame horizontally for a later selfie-view display
        # and convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        detected_gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract and classify landmarks
                landmarks = self._extract_landmarks(hand_landmarks)
                if landmarks is not None:
                    detected_gesture, confidence = self._classify_gesture(landmarks)
                    
                    # Display prediction on frame
                    if detected_gesture:
                        label = f"{detected_gesture} ({confidence:.2f})"
                        cv2.putText(annotated_frame, label, (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Apply temporal filtering for stable results
        confirmed_gesture = self._apply_temporal_filtering(detected_gesture)
        
        return confirmed_gesture, annotated_frame

    def _extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks, dtype=np.float32).flatten()

    def _classify_gesture(self, landmarks):
        """
        Classify landmarks into an ASL gesture using the optimized model.
        
        Returns:
            tuple: (predicted_word, confidence)
        """
        # Reshape for model input
        landmarks_reshaped = landmarks.reshape(1, -1)
        
        # Get model prediction
        predictions = self.model.predict(landmarks_reshaped, verbose=0)[0]
        
        predicted_class_index = np.argmax(predictions)
        confidence = predictions[predicted_class_index]
        
        if confidence >= self.confidence_threshold:
            predicted_word = self.label_encoder.classes_[predicted_class_index]
            return predicted_word, confidence
        
        return None, 0.0

    def _apply_temporal_filtering(self, gesture):
        """Apply temporal filtering to stabilize gesture detection."""
        current_time = time.time()
        
        if gesture is None:
            self.last_gesture = None
            self.gesture_start_time = None
            return None
        
        if self.last_gesture != gesture:
            self.last_gesture = gesture
            self.gesture_start_time = current_time
            return None
        
        if (self.gesture_start_time is not None and 
                current_time - self.gesture_start_time >= self.gesture_hold_time):
            
            if current_time - self.last_confirmed_time >= self.cooldown_period:
                self.last_confirmed_time = current_time
                return gesture
        
        return None

    def get_available_gestures(self):
        """Return the list of gestures the model can recognize."""
        if self.label_encoder:
            return self.label_encoder.classes_
        return []

# Example usage (for testing)
if __name__ == '__main__':
    print("Running Optimized ASL Detector Test...")
    detector = OptimizedASLDetector()
    
    if not detector.model_loaded:
        print("\nDetector could not be initialized. Exiting test.")
    else:
        print(f"\nAvailable gestures: {detector.get_available_gestures()}")
        print("Simulating a dummy landmark input...")
        
        # Create a dummy input of the correct shape (21 landmarks * 3 coords)
        dummy_landmarks = np.random.rand(63).astype(np.float32)
        
        # Classify the dummy input
        word, conf = detector._classify_gesture(dummy_landmarks)
        
        if word:
            print(f"Dummy classification result: '{word}' with confidence {conf:.2f}")
        else:
            print("Dummy classification confidence was below threshold.")
        
        print("\nTest complete. This detector is ready to be integrated into the main application.")
