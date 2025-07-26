#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction Enhancer Module

This module provides functionality to enhance ASL and emotion predictions
by linking words with emotions and applying confidence boosting.
"""

import os
import json
import numpy as np
from collections import defaultdict


class PredictionEnhancer:
    """Class to enhance ASL and emotion predictions by linking words with emotions."""
    
    def __init__(self, mapping_file=None):
        """
        Initialize the PredictionEnhancer.
        
        Args:
            mapping_file (str, optional): Path to a JSON file containing word-emotion mappings.
                If not provided, starts with an empty mapping.
        """
        self.word_emotion_map = defaultdict(list)
        self.emotion_word_map = defaultdict(list)
        self.word_confidence_boost = 0.2  # Default confidence boost
        self.emotion_confidence_boost = 0.2  # Default confidence boost
        
        # Load existing mapping if provided
        if mapping_file and os.path.exists(mapping_file):
            self.load_mapping(mapping_file)
    
    def load_mapping(self, mapping_file):
        """
        Load word-emotion mappings from a JSON file.
        
        Args:
            mapping_file (str): Path to the JSON file containing mappings.
        """
        try:
            with open(mapping_file, 'r') as f:
                data = json.load(f)
                
                # Handle the new format with 'mappings' key
                if 'mappings' in data:
                    mappings = data.get('mappings', {})
                    # Convert mappings to our internal format
                    for word, emotions in mappings.items():
                        self.word_emotion_map[word] = emotions
                        for emotion in emotions:
                            if word not in self.emotion_word_map[emotion]:
                                self.emotion_word_map[emotion].append(word)
                    
                    self.word_confidence_boost = data.get('confidence_boost', 0.2)
                    self.emotion_confidence_boost = data.get('confidence_boost', 0.2)
                # Handle the old format with separate maps
                else:
                    self.word_emotion_map = defaultdict(list, data.get('word_emotion_map', {}))
                    self.emotion_word_map = defaultdict(list, data.get('emotion_word_map', {}))
                    self.word_confidence_boost = data.get('word_confidence_boost', 0.2)
                    self.emotion_confidence_boost = data.get('emotion_confidence_boost', 0.2)
                
            print(f"Loaded prediction enhancement mappings from {mapping_file}")
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            # Initialize with empty mappings
            self.word_emotion_map = defaultdict(list)
            self.emotion_word_map = defaultdict(list)
    
    def save_mapping(self, mapping_file):
        """
        Save word-emotion mappings to a JSON file.
        
        Args:
            mapping_file (str): Path to save the JSON file.
        """
        try:
            # Convert to the new format with 'mappings' key
            mappings = {}
            for word, emotions in self.word_emotion_map.items():
                if emotions:  # Only save non-empty mappings
                    mappings[word] = emotions
            
            data = {
                'mappings': mappings,
                'confidence_boost': self.word_confidence_boost,  # Use same boost for both
                'min_occurrences': 3,  # Default value
                'version': '1.0'
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            
            with open(mapping_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved prediction enhancement mappings to {mapping_file}")
        except Exception as e:
            print(f"Error saving mapping file: {e}")
    
    def update_mapping(self, word, emotion, strength=1.0):
        """
        Update the word-emotion mapping with a new observation.
        
        Args:
            word (str): The ASL word.
            emotion (str): The emotion.
            strength (float): The strength of the association (0.0 to 1.0).
        """
        # Check if the word-emotion pair already exists
        if emotion not in self.word_emotion_map[word]:
            self.word_emotion_map[word].append(emotion)
        
        # Check if the emotion-word pair already exists
        if word not in self.emotion_word_map[emotion]:
            self.emotion_word_map[emotion].append(word)
    
    def enhance_word_prediction(self, word_predictions, current_emotion):
        """
        Enhance word predictions based on the current emotion.
        
        Args:
            word_predictions (dict): Dictionary of {word: confidence} predictions.
            current_emotion (str): The current detected emotion.
            
        Returns:
            dict: Enhanced word predictions.
        """
        if not current_emotion or not word_predictions:
            return word_predictions
        
        enhanced_predictions = word_predictions.copy()
        
        # Get words associated with the current emotion
        associated_words = self.emotion_word_map.get(current_emotion, [])
        
        # Boost confidence for associated words
        for word in associated_words:
            if word in enhanced_predictions:
                enhanced_predictions[word] = min(1.0, enhanced_predictions[word] + self.word_confidence_boost)
        
        return enhanced_predictions
    
    def enhance_emotion_prediction(self, emotion_predictions, current_word):
        """
        Enhance emotion predictions based on the current word.
        
        Args:
            emotion_predictions (dict): Dictionary of {emotion: confidence} predictions.
            current_word (str): The current detected word.
            
        Returns:
            dict: Enhanced emotion predictions.
        """
        if not current_word or not emotion_predictions:
            return emotion_predictions
        
        enhanced_predictions = emotion_predictions.copy()
        
        # Get emotions associated with the current word
        associated_emotions = self.word_emotion_map.get(current_word, [])
        
        # Boost confidence for associated emotions
        for emotion in associated_emotions:
            if emotion in enhanced_predictions:
                enhanced_predictions[emotion] = min(1.0, enhanced_predictions[emotion] + self.emotion_confidence_boost)
        
        return enhanced_predictions
    
    def get_word_emotion_stats(self):
        """
        Get statistics about word-emotion mappings.
        
        Returns:
            dict: Statistics about the mappings.
        """
        stats = {
            'total_words': len(self.word_emotion_map),
            'total_emotions': len(self.emotion_word_map),
            'total_mappings': sum(len(emotions) for emotions in self.word_emotion_map.values()),
            'words_with_most_emotions': [],
            'emotions_with_most_words': []
        }
        
        # Find words with the most emotions
        word_counts = [(word, len(emotions)) for word, emotions in self.word_emotion_map.items()]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        stats['words_with_most_emotions'] = word_counts[:5]  # Top 5
        
        # Find emotions with the most words
        emotion_counts = [(emotion, len(words)) for emotion, words in self.emotion_word_map.items()]
        emotion_counts.sort(key=lambda x: x[1], reverse=True)
        stats['emotions_with_most_words'] = emotion_counts[:5]  # Top 5
        
        return stats