#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASL Data Collection Script

This script is used to collect hand landmark data for training the ASL model.
It captures data from a webcam, processes it with MediaPipe Hands, and saves
the normalized landmarks for a specified ASL word.
"""

import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.common import get_data_dir

def collect_asl_data(word, num_samples=200, sample_duration=2):
    """
    Collects and saves hand landmark data for a specific ASL word.

    Args:
        word (str): The ASL word to collect data for.
        num_samples (int): The number of samples to collect.
        sample_duration (int): The duration in seconds to record each sample.
    """
    data_dir = os.path.join(get_data_dir(), "asl_data", word)
    os.makedirs(data_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print(f"\nStarting data collection for the word: '{word}'")
    print(f"You will collect {num_samples} samples.")
    print("Get ready to sign. The recording will start after the countdown.")

    for i in range(num_samples):
        print(f"\n--- Sample {i + 1}/{num_samples} ---")
        
        # Countdown before each sample
        for t in range(5, 0, -1):
            print(f"Starting in {t}...", end='\r')
            time.sleep(1)
        print("RECORDING... Sign now!")

        start_time = time.time()
        landmarks_list = []

        while time.time() - start_time < sample_duration:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks_list.append(landmarks)
            
            # Display a preview window
            cv2.putText(frame, f"Sample {i+1}: RECORDING...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if landmarks_list:
            # Save the collected landmarks as a single sample
            # We'll use the average landmarks over the duration for consistency
            avg_landmarks = np.mean(landmarks_list, axis=0)
            sample_path = os.path.join(data_dir, f"{word}_{int(time.time())}.npy")
            np.save(sample_path, avg_landmarks)
            print(f"Saved sample to {sample_path}")
        else:
            print("Warning: No hand landmarks were detected for this sample.")

    print("\nData collection complete!")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

def main():
    """Main function to run the data collection script interactively."""
    # --- Interactive Data Collection Setup ---
    data_dir_base = os.path.join(get_data_dir(), "asl_data")
    os.makedirs(data_dir_base, exist_ok=True)

    # List existing words
    try:
        existing_words = [d for d in os.listdir(data_dir_base) if os.path.isdir(os.path.join(data_dir_base, d))]
        if existing_words:
            print("Existing words found:")
            for word in existing_words:
                num_files = len(os.listdir(os.path.join(data_dir_base, word)))
                print(f"  - {word} ({num_files} samples)")
        else:
            print("No existing word data found.")
    except FileNotFoundError:
        print("Data directory not found. A new one will be created.")

    print("\n--- New Data Collection ---")
    
    # Get word from user
    target_word = input("Enter the word to collect data for (e.g., 'hello', 'thanks'): ").strip().lower()
    if not target_word:
        print("Error: Word cannot be empty.")
        return

    # Get number of samples from user
    while True:
        try:
            num_samples_str = input(f"Enter the number of samples (takes) to collect for '{target_word}' (default: 200): ")
            if not num_samples_str:
                num_samples = 200
                break
            num_samples = int(num_samples_str)
            if num_samples > 0:
                break
            else:
                print("Error: Please enter a positive number.")
        except ValueError:
            print("Error: Invalid input. Please enter a number.")

    collect_asl_data(target_word, num_samples)

if __name__ == "__main__":
    main()
