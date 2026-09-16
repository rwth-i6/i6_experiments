import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ast

def parse_learning_rates(lr_file):
    # Parses the learning_rates file to get epoch -> global_step mapping
    # Returns a list of tuples (epoch, step_start, step_end)
    epochs = []
    if not os.path.exists(lr_file):
        return epochs
    with open(lr_file, "r") as f:
        content = f.read()
        # the file is like a python dict: { 1: EpochData(..., error={':meta:global_train_step': 0, ':meta:global_train_step_end': 1554, ...}) }
        # Let's extract using regex
        import re
        epoch_pattern = re.compile(r"(\d+):\s*EpochData.*?'\:meta\:global_train_step'\:\s*(\d+).*?'\:meta\:global_train_step_end'\:\s*(\d+)", re.DOTALL)
        for match in epoch_pattern.finditer(content):
            ep = int(match.group(1))
            start = int(match.group(2))
            end = int(match.group(3))
            epochs.append((ep, start, end))
    return sorted(epochs, key=lambda x: x[0])

def main():
    if len(sys.argv) != 4:
        print("Usage: python plot_disc_loss.py <learning_rates_file> <out_plot.png> <out_data.csv>")
        sys.exit(1)

    lr_file = sys.argv[1]
    out_plot = sys.argv[2]
    out_csv = sys.argv[3]

    train_job_dir = os.path.dirname(os.path.dirname(lr_file))
    runs_dir = os.path.join(train_job_dir, "work", "runs")

    epoch_mapping = parse_learning_rates(lr_file)

    event_files = []
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))

    audio_losses = {} # step -> val
    text_losses = {} # step -> val

    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        if 'train_loss_adv_audio_disc' in ea.scalars.Keys():
            for s in ea.Scalars('train_loss_adv_audio_disc'):
                audio_losses[s.step] = s.value
        if 'train_loss_adv_text_disc' in ea.scalars.Keys():
            for s in ea.Scalars('train_loss_adv_text_disc'):
                text_losses[s.step] = s.value

    # Aggregate by epoch
    epoch_stats = []
    for ep, start, end in epoch_mapping:
        ep_audio = [v for s, v in audio_losses.items() if start <= s < end]
        ep_text = [v for s, v in text_losses.items() if start <= s < end]
        
        stat = {
            "epoch": ep,
            "audio_mean": np.mean(ep_audio) if ep_audio else np.nan,
            "audio_std": np.std(ep_audio) if ep_audio else np.nan,
            "text_mean": np.mean(ep_text) if ep_text else np.nan,
            "text_std": np.std(ep_text) if ep_text else np.nan,
        }
        epoch_stats.append(stat)

    # Plot
    epochs = [s["epoch"] for s in epoch_stats]
    audio_means = np.array([s["audio_mean"] for s in epoch_stats])
    audio_stds = np.array([s["audio_std"] for s in epoch_stats])
    text_means = np.array([s["text_mean"] for s in epoch_stats])
    text_stds = np.array([s["text_std"] for s in epoch_stats])

    plt.figure(figsize=(10, 6))
    if not np.isnan(audio_means).all():
        plt.plot(epochs, audio_means, label='Audio Disc Loss', color='blue')
        plt.fill_between(epochs, audio_means - audio_stds, audio_means + audio_stds, color='blue', alpha=0.2)
    
    if not np.isnan(text_means).all():
        plt.plot(epochs, text_means, label='Text Disc Loss', color='red')
        plt.fill_between(epochs, text_means - text_stds, text_means + text_stds, color='red', alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")
    plt.title("Per Epoch Average Discriminator Loss with Variance Bounds")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_plot)
    
    # Save CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "audio_mean", "audio_std", "text_mean", "text_std"])
        writer.writeheader()
        writer.writerows(epoch_stats)

if __name__ == "__main__":
    main()
