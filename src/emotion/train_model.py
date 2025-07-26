#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Emotion Model Training Script

This script is used to train a custom facial emotion recognition model using
TensorFlow/Keras and MediaPipe FaceMesh landmarks.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import time
import pickle

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_data_dir, get_models_dir


def create_model(input_shape, num_classes):
    """
    Create a neural network model for emotion classification.
    
    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of output classes.
        
    Returns:
        tensorflow.keras.Model: The created model.
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def collect_data(output_dir, num_samples=100, sample_duration=3, display_preview=True):
    """
    Collect facial landmark data for training the emotion model.
    
    Args:
        output_dir (str): Directory to save the collected data.
        num_samples (int): Number of samples to collect per emotion.
        sample_duration (int): Duration in seconds to collect each sample.
        display_preview (bool): Whether to display a preview window during collection.
        
    Returns:
        bool: True if data collection was successful, False otherwise.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Get list of emotions to collect
    default_emotions = [
        "neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"
    ]
    
    # Allow user to select specific emotions or use custom ones
    print("\nEmotion Selection Options:")
    print("1. Use all default emotions")
    print("2. Select specific emotions from the default list")
    print("3. Add custom emotions")
    print("4. Combine selected default emotions with custom ones")
    
    selection = input("Enter your choice (1-4): ")
    
    emotions = []
    if selection == "1":
        emotions = default_emotions
    elif selection == "2":
        print("\nAvailable emotions:")
        for i, emotion in enumerate(default_emotions):
            print(f"{i+1}. {emotion}")
        
        indices = input("\nEnter the numbers of emotions to collect (comma-separated): ")
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in indices.split(",")]
            emotions = [default_emotions[idx] for idx in selected_indices if 0 <= idx < len(default_emotions)]
        except:
            print("Invalid input. Using all default emotions.")
            emotions = default_emotions
    elif selection == "3":
        custom_input = input("\nEnter custom emotions (comma-separated): ")
        emotions = [e.strip() for e in custom_input.split(",") if e.strip()]
    elif selection == "4":
        print("\nAvailable default emotions:")
        for i, emotion in enumerate(default_emotions):
            print(f"{i+1}. {emotion}")
        
        indices = input("\nEnter the numbers of default emotions to collect (comma-separated): ")
        try:
            selected_indices = [int(idx.strip()) - 1 for idx in indices.split(",")]
            selected_defaults = [default_emotions[idx] for idx in selected_indices if 0 <= idx < len(default_emotions)]
        except:
            print("Invalid input. No default emotions selected.")
            selected_defaults = []
        
        custom_input = input("\nEnter custom emotions (comma-separated): ")
        custom_emotions = [e.strip() for e in custom_input.split(",") if e.strip()]
        
        emotions = selected_defaults + custom_emotions
    else:
        print("Invalid choice. Using all default emotions.")
        emotions = default_emotions
    
    if not emotions:
        print("No emotions selected. Using all default emotions.")
        emotions = default_emotions
    
    print(f"\nCollecting data for {len(emotions)} emotions: {', '.join(emotions)}")
    time.sleep(2)  # Give user time to read the list
    
    # Collect data for each emotion
    for emotion in emotions:
        print(f"\nPreparing to collect data for emotion: {emotion}")
        print(f"Press 'SPACE' when ready to start collecting {num_samples} samples.")
        
        # Wait for user to press space
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                return False
            
            # Display instructions
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press SPACE to start collecting",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            if display_preview:
                cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                break
            elif key == 27:  # Esc key
                print("Data collection cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        # Create directory for this emotion
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Collect samples
        sample_count = 0
        while sample_count < num_samples:
            print(f"Collecting sample {sample_count + 1}/{num_samples} for {emotion}...")
            
            # Countdown
            for i in range(3, 0, -1):
                print(f"Starting in {i}...")
                time.sleep(1)
            
            print("GO! Express the emotion now.")
            
            # Collect data for the specified duration
            start_time = time.time()
            landmarks_list = []
            
            while time.time() - start_time < sample_duration:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    continue
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with MediaPipe FaceMesh
                results = face_mesh.process(rgb_frame)
                
                # Draw face landmarks on the frame
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        
                        # Extract landmarks
                        landmarks = []
                        for landmark in face_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        landmarks_list.append(landmarks)
                
                # Display countdown
                remaining = sample_duration - (time.time() - start_time)
                cv2.putText(
                    frame,
                    f"Time remaining: {remaining:.1f}s",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                if display_preview:
                    cv2.imshow("Data Collection", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                    print("Data collection cancelled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            # Save the collected landmarks
            if landmarks_list:
                # Calculate average landmarks over the sample duration
                avg_landmarks = np.mean(landmarks_list, axis=0)
                
                # Save to file
                sample_file = os.path.join(emotion_dir, f"sample_{sample_count}.npy")
                np.save(sample_file, avg_landmarks)
                # Save metadata
                with open(os.path.join(emotion_dir, f"sample_{sample_count}_meta.txt"), 'w') as meta_f:
                    meta_f.write(f"timestamp: {time.time()}\n")
                # Optional: Data augmentation (rotation, brightness)
                # (Add your augmentation code here if desired)
                
                sample_count += 1
                print(f"Saved sample {sample_count}/{num_samples} for {emotion}")
            else:
                print("No face detected in this sample. Retrying...")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nData collection completed successfully!")
    return True


def prepare_dataset(data_dir):
    """
    Prepare the dataset for training from collected data.
    
    Args:
        data_dir (str): Directory containing the collected data.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder) if successful, None otherwise.
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return None
    
    # Get list of emotion directories
    emotion_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not emotion_dirs:
        print(f"Error: No emotion directories found in {data_dir}.")
        return None
    
    # Collect data and labels
    X = []
    y = []
    
    for emotion in emotion_dirs:
        emotion_dir = os.path.join(data_dir, emotion)
        sample_files = [f for f in os.listdir(emotion_dir) if f.endswith('.npy')]
        
        for sample_file in sample_files:
            sample_path = os.path.join(emotion_dir, sample_file)
            try:
                # Load the sample data
                sample_data = np.load(sample_path)
                
                # Add to dataset
                X.append(sample_data)
                y.append(emotion)
            except Exception as e:
                print(f"Error loading sample {sample_path}: {e}")
    
    if not X or not y:
        print("Error: No valid samples found.")
        return None
    
    # Convert to numpy arrays
    X = np.array(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
    )
    
    print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} testing samples")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def train_model(X_train, X_test, y_train, y_test, label_encoder, output_dir, epochs=100, batch_size=32):
    """
    Train the emotion recognition model.
    
    Args:
        X_train (numpy.ndarray): Training data.
        X_test (numpy.ndarray): Testing data.
        y_train (numpy.ndarray): Training labels (one-hot encoded).
        y_test (numpy.ndarray): Testing labels (one-hot encoded).
        label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used for classes.
        output_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        
    Returns:
        bool: True if training was successful, False otherwise.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    # Create the model
    model = create_model(input_shape, num_classes)
    
    # Define callbacks
    checkpoint_path = os.path.join(output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "emotion_model.h5")
    model.save(final_model_path)
    
    # Save the label encoder
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save the class names
    class_names_path = os.path.join(output_dir, "class_names.txt")
    with open(class_names_path, 'w') as f:
        for class_name in label_encoder.classes_:
            f.write(f"{class_name}\n")
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    
    # Evaluate the model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # In train_model, after training and evaluation, print and plot confusion matrix and per-class accuracy
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    print(f"\nModel saved to {final_model_path}")
    print(f"Label encoder saved to {label_encoder_path}")
    print(f"Class names saved to {class_names_path}")
    
    return True


def main():
    """
    Main function to run the emotion model training script.
    """
    parser = argparse.ArgumentParser(description="Emotion Model Training Script")
    parser.add_argument(
        "--collect", action="store_true",
        help="Collect training data from webcam"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the model using collected data"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory for storing/loading training data"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Directory for saving the trained model"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of samples to collect per emotion (default: 100)"
    )
    parser.add_argument(
        "--duration", type=int, default=3,
        help="Duration in seconds to collect each sample (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable preview window during data collection"
    )
    
    args = parser.parse_args()
    
    # Set default directories if not specified
    if args.data_dir is None:
        args.data_dir = os.path.join(get_data_dir(), "emotion_data")
    
    if args.model_dir is None:
        args.model_dir = os.path.join(get_models_dir(), "emotion_model")
    
    # Collect data if requested
    if args.collect:
        print(f"Collecting data to {args.data_dir}")
        success = collect_data(
            args.data_dir,
            num_samples=args.samples,
            sample_duration=args.duration,
            display_preview=not args.no_preview
        )
        
        if not success:
            print("Data collection failed.")
            return 1
    
    # Train model if requested
    if args.train:
        print(f"Training model using data from {args.data_dir}")
        
        # Prepare dataset
        dataset = prepare_dataset(args.data_dir)
        if dataset is None:
            print("Dataset preparation failed.")
            return 1
        
        X_train, X_test, y_train, y_test, label_encoder = dataset
        
        # Train the model
        success = train_model(
            X_train, X_test, y_train, y_test, label_encoder,
            args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        if not success:
            print("Model training failed.")
            return 1
    
    # If neither collect nor train was specified, print help
    if not args.collect and not args.train:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())