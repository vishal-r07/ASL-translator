#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emotion Detector for ASL Translator and Emotion Communicator

This module provides the EmotionDetector class for detecting and classifying
facial emotions using MediaPipe FaceMesh and TensorFlow.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time


class EmotionDetector:
    """Emotion Detector class for detecting and classifying facial emotions."""
    
    def __init__(self, model_path=None, confidence_threshold=0.7):
        """
        Initialize the emotion detector.
        
        Args:
            model_path (str): Path to the pre-trained model. If None, a default model will be used.
            confidence_threshold (float): Threshold for confidence in emotion detection (0.0 to 1.0).
        """
        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create FaceMesh object for detecting facial landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Set confidence threshold for emotion classification
        self.confidence_threshold = 0.3
        
        # Load the emotion classification model
        if model_path is None:
            # Use default model path
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "models", "emotion_model")
            model_path = os.path.join(model_dir, "emotion_model.h5")
        
        # Check if model exists, if not, we'll use a placeholder for now
        self.model = None
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                self.model_loaded = True
                print(f"Loaded emotion model from {model_path}")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
        else:
            print(f"Emotion model not found at {model_path}. Using placeholder detection.")
        
        # Load the list of emotions that the model can recognize from the class_names.txt file
        # If the file doesn't exist, use a default list
        class_names_path = os.path.join(model_dir, "class_names.txt")
        if os.path.exists(class_names_path):
            try:
                with open(class_names_path, 'r') as f:
                    self.emotions = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Loaded {len(self.emotions)} emotions from {class_names_path}")
            except Exception as e:
                print(f"Error loading emotion class names: {e}")
                # Fall back to default list
                self._set_default_emotions()
        else:
            print(f"Emotion class names file not found at {class_names_path}. Using default list.")
            self._set_default_emotions()
        
        # Initialize variables for tracking emotions over time
        self.last_emotion = None
        self.emotion_start_time = None
        self.emotion_hold_time = 0.1  # seconds to hold an emotion before confirming
        self.last_confirmed_time = 0
        self.cooldown_period = 2.0  # seconds to wait before confirming the same emotion again
        # For demonstration purposes (remove in production)
        self.demo_counter = 0
        self.demo_emotions = ["neutral", "happy", "sad", "angry", "surprised"]
    
    def _set_default_emotions(self):
        """Set the default list of emotions."""
        self.emotions = [
            "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"
        ]
        print(f"Using default list of {len(self.emotions)} emotions.")
    
    def process_frame(self, frame, current_word=None, prediction_enhancer=None):
        """
        Process a frame to detect and classify facial emotions.
        
        Args:
            frame (numpy.ndarray): The frame to process.
            current_word (str, optional): Current detected ASL word for prediction enhancement.
            prediction_enhancer (PredictionEnhancer, optional): Enhancer for predictions.
            
        Returns:
            str: The detected emotion or None if no face is detected with high confidence.
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe FaceMesh
        results = self.face_mesh.process(rgb_frame)
        
        # Draw face landmarks on the frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Make the frame square by padding (if not already square)
                h, w, _ = frame.shape
                if h != w:
                    size = max(h, w)
                    padded_frame = np.zeros((size, size, 3), dtype=frame.dtype)
                    y_offset = (size - h) // 2
                    x_offset = (size - w) // 2
                    padded_frame[y_offset:y_offset+h, x_offset:x_offset+w] = frame
                    frame = padded_frame
                # Draw the face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Extract facial landmarks for classification
                landmarks = self._extract_landmarks(face_landmarks)
                
                # Classify the emotion
                emotion = None
                confidence_scores = {}
                if self.model_loaded and self.model is not None:
                    # Use the actual model for classification
                    emotion, confidence_scores = self._classify_emotion(landmarks)
                    
                    # Enhance prediction if enhancer and word are provided
                    if prediction_enhancer and current_word and confidence_scores:
                        enhanced_scores = prediction_enhancer.enhance_emotion_prediction(confidence_scores, current_word)
                        
                        # Find the emotion with highest confidence after enhancement
                        if enhanced_scores:
                            max_emotion = max(enhanced_scores.items(), key=lambda x: x[1])
                            if max_emotion[1] >= self.confidence_threshold:
                                emotion = max_emotion[0]
                else:
                    # Use a placeholder for demonstration
                    emotion = self._demo_classify_emotion()
                    confidence_scores = {emotion: 1.0} if emotion else {}
                
                # Apply temporal filtering to reduce false positives
                confirmed_emotion = self._apply_temporal_filtering(emotion)
                
                # Update prediction enhancer with the detected emotion and word
                if prediction_enhancer and confirmed_emotion and current_word:
                    prediction_enhancer.update_mapping(current_word, confirmed_emotion)
                
                return confirmed_emotion
        
        # Reset emotion tracking if no face is detected
        self.last_emotion = None
        self.emotion_start_time = None
        
        return None
    
    def _extract_landmarks(self, face_landmarks):
        """
        Extract facial landmarks from MediaPipe results and normalize them.
        
        Args:
            face_landmarks: MediaPipe face landmarks.
            
        Returns:
            numpy.ndarray: Normalized facial landmarks as a flat array.
        """
        # Extract the x, y, z coordinates of each landmark
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Normalize the landmarks
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Reshape for the model input (assuming the model expects a specific shape)
        # This may need to be adjusted based on the actual model architecture
        landmarks = landmarks.reshape(1, -1)
        
        return landmarks
    
    def _classify_emotion(self, landmarks):
        """
        Classify facial landmarks into an emotion.
        
        Args:
            landmarks (numpy.ndarray): Normalized facial landmarks.
            
        Returns:
            tuple: (predicted_emotion, confidence_scores) where predicted_emotion is the classified emotion 
                  or None if confidence is below threshold, and confidence_scores is a dictionary 
                  of {emotion: confidence} for all emotions.
        """
        # Make prediction with the model
        predictions = self.model.predict(landmarks, verbose=0)  # Set verbose=0 to reduce output noise
        
        # Get the index of the highest confidence prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        
        # Create a dictionary of all emotion confidences
        confidence_scores = {}
        for i, score in enumerate(predictions[0]):
            if i < len(self.emotions):
                confidence_scores[self.emotions[i]] = float(score)
        
        # Check if confidence is above threshold
        predicted_emotion = None
        if confidence >= self.confidence_threshold:
            # Ensure the index is within range of available emotions
            if predicted_class_index < len(self.emotions):
                predicted_emotion = self.emotions[predicted_class_index]
            else:
                print(f"Warning: Predicted class index {predicted_class_index} is out of range for available emotions.")
        
        # Print top-3 predictions for debugging
        top_indices = np.argsort(predictions[0])[::-1][:3]
        print('Emotion Top-3 predictions:')
        for idx in top_indices:
            if idx < len(self.emotions):
                print(f'  {self.emotions[idx]}: {predictions[0][idx]:.3f}')
        
        return predicted_emotion, confidence_scores
    
    def _demo_classify_emotion(self):
        """
        A placeholder method for demonstration when no model is available.
        
        Returns:
            str: A demo emotion.
        """
        # Increment counter every 30 frames (about 1 second at 30 fps)
        if self.demo_counter % 30 == 0:
            emotion_index = (self.demo_counter // 30) % len(self.demo_emotions)
            return self.demo_emotions[emotion_index]
        
        self.demo_counter += 1
        return None
    
    def _apply_temporal_filtering(self, emotion):
        """
        Apply temporal filtering to reduce false positives and ensure emotion stability.
        
        Args:
            emotion (str): The current detected emotion.
            
        Returns:
            str: The confirmed emotion or None if not confirmed.
        """
        current_time = time.time()
        
        # If the emotion is None, reset tracking
        if emotion is None:
            self.last_emotion = None
            self.emotion_start_time = None
            return None
        
        # If this is a new emotion, start tracking it
        if self.last_emotion != emotion:
            self.last_emotion = emotion
            self.emotion_start_time = current_time
            return None
        
        # If the same emotion has been held for long enough, confirm it
        if (self.emotion_start_time is not None and 
                current_time - self.emotion_start_time >= self.emotion_hold_time):
            
            # Check if we're in the cooldown period for this emotion
            if current_time - self.last_confirmed_time >= self.cooldown_period:
                self.last_confirmed_time = current_time
                return emotion
        
        return None
    
    def get_available_emotions(self):
        """
        Get the list of available emotions that the model can recognize.
        
        Returns:
            list: A list of emotions.
        """
        return self.emotions