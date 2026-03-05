import os
import re
import matplotlib.pyplot as plt
import argparse

def plot_training_log(log_path, output_path):
    epochs = []
    losses = []
    val_sisdrs = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # First try matching the new format with Val SI-SDR
            # e.g.: 2026-03-04 12:00:00 - training - INFO - Epoch 1 Completed. Avg Loss: 1.234 | Val SI-SDR: -12.34 dB
            match_full = re.search(r"Epoch (\d+) Completed\. Avg Loss: ([\d\.]+)\s*\|\s*Val SI-SDR:\s*([-\d\.]+)\s*dB", line)
            if match_full:
                epochs.append(int(match_full.group(1)))
                losses.append(float(match_full.group(2)))
                val_sisdrs.append(float(match_full.group(3)))
            else:
                # Fallback to old format without validation
                match_old = re.search(r"Epoch (\d+) Completed\. Avg Loss: ([\d\.]+)", line)
                if match_old:
                    epochs.append(int(match_old.group(1)))
                    losses.append(float(match_old.group(2)))
                    val_sisdrs.append(None) # Mark as missing
                
    if not epochs:
        print(f"No training data found in {log_path}")
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Composite Loss', color=color)
    ax1.plot(epochs, losses, marker='o', linestyle='-', color=color, label="Training Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Only plot SI-SDR if we have data for it
    valid_sisdrs = [s for s in val_sisdrs if s is not None]
    if valid_sisdrs:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Validation SI-SDR (dB)', color=color)
        ax2.plot(epochs, val_sisdrs, marker='s', linestyle='-', color=color, label="Val SI-SDR")
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Adding legends for both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')
    else:
        ax1.legend(loc='upper right')
        
    plt.title("Training Progress (Loss & Accuracy)")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training loss from a log file.")
    parser.add_argument("log_file", help="Path to the training log file.")
    parser.add_argument("--output", "-o", default="training_loss_plot.png", help="Output path for the PNG plot.")
    args = parser.parse_args()
    
    if os.path.exists(args.log_file):
        plot_training_log(args.log_file, args.output)
    else:
        print(f"File not found: {args.log_file}")
