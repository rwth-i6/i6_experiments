import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

def parse_learning_rates(lr_file, max_epoch):
    losses = {}
    if not os.path.exists(lr_file):
        return losses
        
    with open(lr_file, "r") as f:
        content = f.read()
        
    # Extract epoch and the dict part
    epoch_blocks = re.findall(r"(\d+):\s*EpochData.*?(?:error|metrics)=\{(.*?)\}", content, re.DOTALL)
    for ep_str, metrics_str in epoch_blocks:
        ep = int(ep_str)
        if ep > max_epoch:
            continue
            
        audio_match = re.search(r"'train_loss_adv_audio_disc':\s*([0-9.]+)", metrics_str)
        text_match = re.search(r"'train_loss_adv_text_disc':\s*([0-9.]+)", metrics_str)
        
        audio_loss = float(audio_match.group(1)) if audio_match else np.nan
        text_loss = float(text_match.group(1)) if text_match else np.nan
        
        losses[ep] = {"audio": audio_loss, "text": text_loss}
        
    return losses

def main():
    if len(sys.argv) < 4:
        print("Usage: python plot_disc_loss_summary.py <max_epoch> <out_plot.png> <lr_file_1> [<lr_file_2> ...]")
        sys.exit(1)

    max_epoch = int(sys.argv[1])
    out_plot = sys.argv[2]
    lr_files = sys.argv[3:]

    all_audio_losses = {} # ep -> [losses]
    all_text_losses = {} # ep -> [losses]

    for lr_file in lr_files:
        losses = parse_learning_rates(lr_file, max_epoch)
        for ep, vals in losses.items():
            if ep not in all_audio_losses:
                all_audio_losses[ep] = []
                all_text_losses[ep] = []
            
            if not np.isnan(vals["audio"]):
                all_audio_losses[ep].append(vals["audio"])
            if not np.isnan(vals["text"]):
                all_text_losses[ep].append(vals["text"])

    epochs = sorted(list(all_audio_losses.keys()))
    
    audio_means = np.array([np.mean(all_audio_losses[ep]) if all_audio_losses[ep] else np.nan for ep in epochs])
    audio_stds = np.array([np.std(all_audio_losses[ep]) if all_audio_losses[ep] else np.nan for ep in epochs])
    
    text_means = np.array([np.mean(all_text_losses[ep]) if all_text_losses[ep] else np.nan for ep in epochs])
    text_stds = np.array([np.std(all_text_losses[ep]) if all_text_losses[ep] else np.nan for ep in epochs])

    plt.figure(figsize=(10, 6))
    
    # Plot Audio Disc Loss
    if not np.isnan(audio_means).all():
        plt.plot(epochs, audio_means, label='Audio Disc Loss', color='blue')
        plt.fill_between(epochs, audio_means - audio_stds, audio_means + audio_stds, color='blue', alpha=0.2)
        
    # Plot Text Disc Loss
    if not np.isnan(text_means).all():
        plt.plot(epochs, text_means, label='Text Disc Loss', color='red')
        plt.fill_between(epochs, text_means - text_stds, text_means + text_stds, color='red', alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")
    plt.title(f"Average Discriminator Loss (up to Epoch {max_epoch})")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_plot)

if __name__ == "__main__":
    main()
