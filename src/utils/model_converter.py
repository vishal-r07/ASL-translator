#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Converter Utility

This script provides utilities to convert TensorFlow models to TensorFlow Lite format
for better performance on resource-constrained devices.
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_models_dir


def convert_to_tflite(model_path, output_path=None, quantize=True):
    """
    Convert a TensorFlow model to TensorFlow Lite format.
    
    Args:
        model_path (str): Path to the TensorFlow model (.h5 or SavedModel).
        output_path (str, optional): Path to save the TFLite model. If None, will use
                                     the same path as the input model but with .tflite extension.
        quantize (bool): Whether to apply quantization to reduce model size.
        
    Returns:
        str: Path to the converted TFLite model if successful, None otherwise.
    """
    try:
        # Load the model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Apply optimization if requested
        if quantize:
            print("Applying post-training quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        print("Converting model to TFLite format...")
        tflite_model = converter.convert()
        
        # Determine output path if not provided
        if output_path is None:
            if model_path.endswith('.h5'):
                output_path = model_path.replace('.h5', '.tflite')
            else:
                output_path = os.path.join(os.path.dirname(model_path), "model.tflite")
        
        # Save the model
        print(f"Saving TFLite model to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"TFLite model size: {model_size:.2f} MB")
        
        return output_path
    
    except Exception as e:
        print(f"Error converting model: {e}")
        return None


def benchmark_tflite_model(tflite_model_path, input_shape, num_runs=50):
    """
    Benchmark the performance of a TFLite model.
    
    Args:
        tflite_model_path (str): Path to the TFLite model.
        input_shape (tuple): Shape of the input tensor (excluding batch dimension).
        num_runs (int): Number of inference runs for benchmarking.
        
    Returns:
        dict: Dictionary containing benchmark results.
    """
    try:
        # Load the TFLite model
        print(f"Loading TFLite model from {tflite_model_path}...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Create a random input tensor
        input_data = np.random.random((1,) + input_shape).astype(np.float32)
        
        # Warm-up run
        print("Performing warm-up inference...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Benchmark inference time
        print(f"Benchmarking model with {num_runs} runs...")
        inference_times = []
        
        for i in range(num_runs):
            start_time = tf.timestamp()
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            end_time = tf.timestamp()
            inference_time = (end_time - start_time).numpy() * 1000  # Convert to ms
            inference_times.append(inference_time)
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        p95_time = np.percentile(inference_times, 95)
        
        # Print results
        print(f"\nBenchmark Results (over {num_runs} runs):")
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Min inference time: {min_time:.2f} ms")
        print(f"Max inference time: {max_time:.2f} ms")
        print(f"95th percentile inference time: {p95_time:.2f} ms")
        print(f"Theoretical max throughput: {1000/avg_time:.2f} inferences/second")
        
        return {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "p95_time": p95_time,
            "throughput": 1000/avg_time
        }
    
    except Exception as e:
        print(f"Error benchmarking model: {e}")
        return None


def main():
    """
    Main function to run the model converter utility.
    """
    parser = argparse.ArgumentParser(description="TensorFlow to TFLite Model Converter")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the TensorFlow model (.h5 or SavedModel)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save the TFLite model (default: same as input with .tflite extension)"
    )
    parser.add_argument(
        "--no-quantize", action="store_true",
        help="Disable post-training quantization"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark the converted model"
    )
    parser.add_argument(
        "--input-shape", type=str, default=None,
        help="Input shape for benchmarking, comma-separated (e.g., '468,3' for FaceMesh)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=50,
        help="Number of inference runs for benchmarking (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Convert the model
    tflite_path = convert_to_tflite(
        args.model,
        args.output,
        not args.no_quantize
    )
    
    if tflite_path is None:
        print("Model conversion failed.")
        return 1
    
    # Benchmark if requested
    if args.benchmark:
        if args.input_shape is None:
            print("Error: --input-shape is required for benchmarking.")
            return 1
        
        # Parse input shape
        try:
            input_shape = tuple(map(int, args.input_shape.split(',')))
        except ValueError:
            print("Error: Invalid input shape format. Use comma-separated integers.")
            return 1
        
        # Run benchmark
        benchmark_results = benchmark_tflite_model(
            tflite_path,
            input_shape,
            args.num_runs
        )
        
        if benchmark_results is None:
            print("Benchmarking failed.")
            return 1
    
    print("\nOperation completed successfully.")
    return 0


def convert_asl_model():
    """
    Convenience function to convert the ASL model to TFLite format.
    """
    models_dir = get_models_dir()
    asl_model_path = os.path.join(models_dir, "asl_model", "asl_model.h5")
    
    if not os.path.exists(asl_model_path):
        print(f"Error: ASL model not found at {asl_model_path}")
        return False
    
    output_path = os.path.join(models_dir, "asl_model", "asl_model.tflite")
    
    result = convert_to_tflite(asl_model_path, output_path, quantize=True)
    return result is not None


def convert_emotion_model():
    """
    Convenience function to convert the emotion model to TFLite format.
    """
    models_dir = get_models_dir()
    emotion_model_path = os.path.join(models_dir, "emotion_model", "emotion_model.h5")
    
    if not os.path.exists(emotion_model_path):
        print(f"Error: Emotion model not found at {emotion_model_path}")
        return False
    
    output_path = os.path.join(models_dir, "emotion_model", "emotion_model.tflite")
    
    result = convert_to_tflite(emotion_model_path, output_path, quantize=True)
    return result is not None


if __name__ == "__main__":
    sys.exit(main())