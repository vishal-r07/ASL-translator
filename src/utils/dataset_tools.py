#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Visualization and Balancing Tool

- Counts and prints the number of samples per class
- Plots class distribution
- Warns if classes are imbalanced
- Optionally balances dataset by undersampling/oversampling
- Works for both ASL and emotion datasets
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import shutil
import random

def analyze_dataset(data_dir):
    class_counts = {}
    class_samples = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        class_counts[class_name] = len(npy_files)
        class_samples[class_name] = npy_files[:5]  # Preview up to 5
    return class_counts, class_samples

def plot_distribution(class_counts, title="Class Distribution"):
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts, color='skyblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def warn_imbalance(class_counts, threshold=0.2):
    counts = list(class_counts.values())
    if not counts:
        print("No data found.")
        return
    min_c, max_c = min(counts), max(counts)
    if min_c == 0:
        print("Warning: At least one class has zero samples!")
    if max_c == 0:
        print("No samples in any class.")
        return
    if (max_c - min_c) / max_c > threshold:
        print(f"Warning: Classes are imbalanced (min: {min_c}, max: {max_c})")
    else:
        print("Classes are reasonably balanced.")

def balance_dataset(data_dir, method='undersample'):
    class_counts, _ = analyze_dataset(data_dir)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Balancing dataset using method: {method}")
    for class_name, count in class_counts.items():
        class_path = os.path.join(data_dir, class_name)
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        if method == 'undersample' and count > min_count:
            to_remove = random.sample(npy_files, count - min_count)
            for f in to_remove:
                os.remove(os.path.join(class_path, f))
            print(f"Undersampled {class_name}: now {min_count} samples.")
        elif method == 'oversample' and count < max_count:
            to_copy = random.choices(npy_files, k=max_count - count)
            for i, f in enumerate(to_copy):
                src = os.path.join(class_path, f)
                dst = os.path.join(class_path, f"aug_{i}_{f}")
                shutil.copy(src, dst)
            print(f"Oversampled {class_name}: now {max_count} samples.")
    print("Balancing complete.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Visualization and Balancing Tool")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--plot', action='store_true', help='Plot class distribution')
    parser.add_argument('--balance', choices=['undersample', 'oversample'], help='Balance dataset')
    args = parser.parse_args()

    class_counts, class_samples = analyze_dataset(args.data_dir)
    print("Class counts:")
    for c, n in class_counts.items():
        print(f"  {c}: {n}")
    warn_imbalance(class_counts)
    print("Sample files per class:")
    for c, samples in class_samples.items():
        print(f"  {c}: {samples}")
    if args.plot:
        plot_distribution(class_counts, title=f"Class Distribution in {args.data_dir}")
    if args.balance:
        balance_dataset(args.data_dir, method=args.balance)

if __name__ == "__main__":
    main() 