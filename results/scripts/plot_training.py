import os
import re
import matplotlib.pyplot as plt
import argparse

def plot_training_log(log_path, output_path):
    epochs = []
    losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: 2026-02-28 23:05:21 - training - INFO - Epoch 50 Completed. Avg Loss: 1.3753
            match = re.search(r"Epoch (\d+) Completed\. Avg Loss: ([\d\.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not epochs:
        print(f"No training data found in {log_path}")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    plt.title("Training Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.grid(True)
    
    # Optional: highlight the trend if there's enough data
    if len(epochs) > 5:
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "r--", alpha=0.8, label="Trend")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss from a log file.")
    parser.add_argument("log_file", help="Path to the training log file.")
    parser.add_argument("--output", "-o", default="training_loss_plot.png", help="Output path for the PNG plot.")
    args = parser.parse_args()
    
    # Requires numpy for trend line
    import numpy as np
    
    if os.path.exists(args.log_file):
        plot_training_log(args.log_file, args.output)
    else:
        print(f"File not found: {args.log_file}")
