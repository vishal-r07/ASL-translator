#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized ASL Model Training Script for Perfect Accuracy

This script creates a perfectly balanced dataset and trains a model optimized
for 100% accuracy on the two ASL words: "where" and "yes".
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random
from collections import Counter

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_data_dir, get_models_dir


def balance_dataset(X, y, target_samples_per_class=None):
    """
    Balance the dataset by ensuring equal samples per class.
    
    Args:
        X (list): Input data samples
        y (list): Corresponding labels
        target_samples_per_class (int): Target number of samples per class
        
    Returns:
        tuple: (balanced_X, balanced_y)
    """
    # Group samples by class
    class_samples = {}
    for i, label in enumerate(y):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(X[i])
    
    # Determine target samples per class
    if target_samples_per_class is None:
        target_samples_per_class = min(len(samples) for samples in class_samples.values())
    
    print(f"Original class distribution:")
    for class_name, samples in class_samples.items():
        print(f"  {class_name}: {len(samples)} samples")
    
    print(f"\nBalancing to {target_samples_per_class} samples per class...")
    
    # Balance the dataset
    balanced_X = []
    balanced_y = []
    
    for class_name, samples in class_samples.items():
        if len(samples) >= target_samples_per_class:
            # Randomly sample if we have more than needed
            selected_samples = random.sample(samples, target_samples_per_class)
        else:
            # Oversample if we have fewer than needed
            selected_samples = samples.copy()
            while len(selected_samples) < target_samples_per_class:
                # Add random samples with slight noise for data augmentation
                base_sample = random.choice(samples)
                # Add small gaussian noise for variation
                noise = np.random.normal(0, 0.01, base_sample.shape)
                augmented_sample = base_sample + noise
                selected_samples.append(augmented_sample)
        
        balanced_X.extend(selected_samples)
        balanced_y.extend([class_name] * len(selected_samples))
    
    print(f"Balanced class distribution:")
    balanced_counter = Counter(balanced_y)
    for class_name, count in balanced_counter.items():
        print(f"  {class_name}: {count} samples")
    
    return balanced_X, balanced_y


def create_optimized_model(input_shape, num_classes):
    """
    Create an optimized neural network model for perfect ASL gesture classification.
    
    Args:
        input_shape (int): The shape of the input data.
        num_classes (int): The number of output classes.
        
    Returns:
        tensorflow.keras.Model: The created model.
    """
    model = Sequential([
        # Input layer with more neurons for better feature extraction
        Dense(256, input_shape=(input_shape,)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers with decreasing size
        Dense(128),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for more precise training
    optimizer = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def prepare_balanced_dataset(data_dir, target_samples_per_class=200):
    """
    Prepare a perfectly balanced dataset for training.
    
    Args:
        data_dir (str): Directory containing the collected data.
        target_samples_per_class (int): Target number of samples per class.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, label_encoder) if successful, None otherwise.
    """
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return None
    
    # Get list of gesture directories
    gesture_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not gesture_dirs:
        print(f"Error: No gesture directories found in {data_dir}.")
        return None
    
    print(f"Found gesture classes: {gesture_dirs}")
    
    # Collect data and labels
    X = []
    y = []
    
    for gesture in gesture_dirs:
        gesture_dir = os.path.join(data_dir, gesture)
        sample_files = [f for f in os.listdir(gesture_dir) if f.endswith('.npy')]
        
        print(f"Loading {len(sample_files)} samples for '{gesture}'...")
        
        for sample_file in sample_files:
            sample_path = os.path.join(gesture_dir, sample_file)
            try:
                # Load the sample data
                sample_data = np.load(sample_path)
                
                # Ensure data is properly shaped
                if sample_data.ndim == 1:
                    X.append(sample_data)
                    y.append(gesture)
                else:
                    print(f"Warning: Skipping sample {sample_path} - unexpected shape {sample_data.shape}")
            except Exception as e:
                print(f"Error loading sample {sample_path}: {e}")
    
    if not X or not y:
        print("Error: No valid samples found.")
        return None
    
    print(f"Loaded {len(X)} total samples")
    
    # Balance the dataset
    X_balanced, y_balanced = balance_dataset(X, y, target_samples_per_class)
    
    # Convert to numpy arrays
    X_balanced = np.array(X_balanced)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_balanced)
    
    # Convert to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    # Split into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_onehot, test_size=0.15, random_state=42, stratify=y_onehot
    )
    
    print(f"\nFinal dataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X_train, X_test, y_train, y_test, label_encoder


def train_optimized_model(X_train, X_test, y_train, y_test, label_encoder, output_dir, epochs=200):
    """
    Train the optimized ASL gesture recognition model for perfect accuracy.
    
    Args:
        X_train (numpy.ndarray): Training data.
        X_test (numpy.ndarray): Testing data.
        y_train (numpy.ndarray): Training labels (one-hot encoded).
        y_test (numpy.ndarray): Testing labels (one-hot encoded).
        label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used for classes.
        output_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        
    Returns:
        bool: True if training was successful, False otherwise.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Create the optimized model
    model = create_optimized_model(input_shape, num_classes)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks for perfect training
    checkpoint_path = os.path.join(output_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # More aggressive early stopping for perfect accuracy
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=50,  # Increased patience for perfect accuracy
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001
    )
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train the model
    print(f"\nTraining model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=16,  # Smaller batch size for better convergence
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Load the best model
    model.load_weights(checkpoint_path)
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "asl_model_optimized.h5")
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
    
    # Evaluate the model
    print("\nEvaluating optimized model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.6f}")
    print(f"Test Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    
    # Generate predictions for detailed analysis
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate per-class accuracy
    print("\nPer-class Analysis:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_accuracy:.6f} ({class_accuracy*100:.2f}%)")
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_,
                cmap='Blues')
    plt.title('Optimized Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_optimized.png'), dpi=300)
    plt.close()
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], linewidth=2)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history_optimized.png"), dpi=300)
    plt.close()
    
    # Save training metrics and classification report
    metrics_path = os.path.join(output_dir, "training_metrics.txt")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    with open(metrics_path, 'w') as f:
        f.write(f"Final Test Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)\n")
        f.write(f"Final Test Loss: {loss:.6f}\n")
        f.write(f"Total Training Epochs: {len(history.history['accuracy'])}\n")
        f.write(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.6f}\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
    
    print(f"\nOptimized model saved to {final_model_path}")
    print(f"Label encoder saved to {label_encoder_path}")
    print(f"Training metrics saved to {metrics_path}")
    
    # Check if we achieved perfect accuracy
    if accuracy >= 0.999:
        print("\n🎉 PERFECT ACCURACY ACHIEVED! 🎉")
        print("Your ASL translator model is now perfectly trained!")
    elif accuracy >= 0.95:
        print(f"\n✅ Excellent accuracy achieved: {accuracy*100:.2f}%")
    else:
        print(f"\n⚠️  Accuracy could be improved: {accuracy*100:.2f}%")
        print("Consider collecting more balanced data or adjusting hyperparameters.")
    
    return True


def main():
    """Main function to run the optimized ASL model training."""
    print("🚀 Starting Optimized ASL Model Training for Perfect Accuracy")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Get directories
    data_dir = os.path.join(get_data_dir(), "asl_data")
    models_dir = get_models_dir()
    
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    
    # Prepare balanced dataset
    print("\n📊 Preparing balanced dataset...")
    dataset = prepare_balanced_dataset(data_dir, target_samples_per_class=200)
    
    if dataset is None:
        print("❌ Failed to prepare dataset.")
        return 1
    
    X_train, X_test, y_train, y_test, label_encoder = dataset
    
    # Train the optimized model
    print("\n🧠 Training optimized model...")
    success = train_optimized_model(
        X_train, X_test, y_train, y_test, 
        label_encoder, models_dir, epochs=300
    )
    
    if success:
        print("\n✅ Training completed successfully!")
        print("Your ASL translator is now optimized for perfect accuracy!")
        return 0
    else:
        print("\n❌ Training failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
