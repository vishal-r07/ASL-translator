#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Downloader Utility

This script downloads pre-trained models for ASL and emotion recognition
to provide a starting point for users who don't want to train their own models.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import hashlib
import shutil
import time

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import utility functions
from src.utils.common import get_models_dir

# URLs for pre-trained models
# Note: These are placeholder URLs and would need to be replaced with actual model URLs
MODEL_URLS = {
    "asl": {
        "url": "https://github.com/yourusername/asl-translator-models/releases/download/v1.0/asl_model.zip",
        "md5": "0123456789abcdef0123456789abcdef",  # Placeholder MD5 hash
        "size": 25000000  # Approximate size in bytes (25 MB)
    },
    "emotion": {
        "url": "https://github.com/yourusername/emotion-recognition-models/releases/download/v1.0/emotion_model.zip",
        "md5": "fedcba9876543210fedcba9876543210",  # Placeholder MD5 hash
        "size": 15000000  # Approximate size in bytes (15 MB)
    }
}


def calculate_md5(file_path):
    """
    Calculate the MD5 hash of a file.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str: MD5 hash of the file.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_with_progress(url, output_path, expected_size=None):
    """
    Download a file with progress reporting.
    
    Args:
        url (str): URL to download from.
        output_path (str): Path to save the downloaded file.
        expected_size (int, optional): Expected file size in bytes.
        
    Returns:
        bool: True if download was successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define a progress hook to display download progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 / total_size)
                downloaded = block_num * block_size
                if expected_size is not None:
                    total_size = expected_size
                
                # Calculate download speed
                current_time = time.time()
                if not hasattr(report_progress, "start_time"):
                    report_progress.start_time = current_time
                    report_progress.last_time = current_time
                    report_progress.last_downloaded = 0
                
                time_diff = current_time - report_progress.last_time
                if time_diff >= 1.0:  # Update every second
                    speed = (downloaded - report_progress.last_downloaded) / time_diff
                    report_progress.last_time = current_time
                    report_progress.last_downloaded = downloaded
                    
                    # Format speed string
                    if speed < 1024:
                        speed_str = f"{speed:.1f} B/s"
                    elif speed < 1024 * 1024:
                        speed_str = f"{speed/1024:.1f} KB/s"
                    else:
                        speed_str = f"{speed/(1024*1024):.1f} MB/s"
                    
                    # Format size strings
                    if total_size < 1024:
                        size_str = f"{downloaded} / {total_size} B"
                    elif total_size < 1024 * 1024:
                        size_str = f"{downloaded/1024:.1f} / {total_size/1024:.1f} KB"
                    else:
                        size_str = f"{downloaded/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB"
                    
                    # Print progress
                    print(f"\rDownloading: {percent:.1f}% | {size_str} | {speed_str}", end="")
        
        # Download the file
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path, report_progress)
        print("\nDownload completed.")
        
        return True
    
    except Exception as e:
        print(f"\nError downloading file: {e}")
        # Remove partially downloaded file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def extract_zip(zip_path, extract_dir):
    """
    Extract a ZIP file.
    
    Args:
        zip_path (str): Path to the ZIP file.
        extract_dir (str): Directory to extract to.
        
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        # Create extraction directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the ZIP file
        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("Extraction completed.")
        return True
    
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        return False


def download_model(model_type, output_dir=None, force=False):
    """
    Download and extract a pre-trained model.
    
    Args:
        model_type (str): Type of model to download ('asl' or 'emotion').
        output_dir (str, optional): Directory to save the model. If None, will use the default models directory.
        force (bool): Whether to force download even if the model already exists.
        
    Returns:
        bool: True if download and extraction were successful, False otherwise.
    """
    # Check if model type is valid
    if model_type not in MODEL_URLS:
        print(f"Error: Invalid model type '{model_type}'. Valid types are: {', '.join(MODEL_URLS.keys())}")
        return False
    
    # Set output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(get_models_dir(), f"{model_type}_model")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model already exists
    model_file = os.path.join(output_dir, f"{model_type}_model.h5")
    if os.path.exists(model_file) and not force:
        print(f"Model already exists at {model_file}. Use --force to download again.")
        return True
    
    # Get model URL and MD5 hash
    model_info = MODEL_URLS[model_type]
    url = model_info["url"]
    expected_md5 = model_info["md5"]
    expected_size = model_info.get("size")
    
    # Download the model
    zip_path = os.path.join(output_dir, f"{model_type}_model.zip")
    success = download_with_progress(url, zip_path, expected_size)
    if not success:
        return False
    
    # Verify MD5 hash
    print("Verifying download integrity...")
    actual_md5 = calculate_md5(zip_path)
    if actual_md5 != expected_md5:
        print(f"Error: MD5 hash mismatch. Expected {expected_md5}, got {actual_md5}.")
        os.remove(zip_path)
        return False
    
    # Extract the ZIP file
    temp_extract_dir = os.path.join(output_dir, "temp_extract")
    success = extract_zip(zip_path, temp_extract_dir)
    if not success:
        # Clean up
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        return False
    
    # Move extracted files to the output directory
    try:
        for item in os.listdir(temp_extract_dir):
            src_path = os.path.join(temp_extract_dir, item)
            dst_path = os.path.join(output_dir, item)
            
            # Remove existing file/directory if it exists
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)
            
            # Move the file/directory
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
    
    except Exception as e:
        print(f"Error moving extracted files: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
    
    print(f"Model downloaded and extracted to {output_dir}")
    return True


def main():
    """
    Main function to run the model downloader utility.
    """
    parser = argparse.ArgumentParser(description="Pre-trained Model Downloader")
    parser.add_argument(
        "--model", type=str, choices=["asl", "emotion", "all"],
        default="all", help="Type of model to download (default: all)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save the models (default: project's models directory)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force download even if the model already exists"
    )
    
    args = parser.parse_args()
    
    # Set output directory if not provided
    if args.output_dir is None:
        args.output_dir = get_models_dir()
    
    # Download the requested models
    if args.model == "all":
        success = True
        for model_type in MODEL_URLS.keys():
            model_dir = os.path.join(args.output_dir, f"{model_type}_model")
            success = success and download_model(model_type, model_dir, args.force)
    else:
        model_dir = os.path.join(args.output_dir, f"{args.model}_model")
        success = download_model(args.model, model_dir, args.force)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())