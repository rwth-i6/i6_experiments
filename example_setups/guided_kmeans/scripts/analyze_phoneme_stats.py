"""
Analyse per-epoch phoneme statistics against LibriSpeech-100 unigram priors.

Reads epoch_statistics.json from OUTPUT_DIR and produces:
  1. Line plot  – L1 distance to unigram priors per epoch
  2. Two-panel barplot – unigram priors (top) vs. last-epoch training counts (bottom)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/work/i6_core/returnn/forward/ReturnnForwardJobV2.J8q9N2q1Geyj/output"
)

PRIORS_FILE = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/phoneme_frequencies/phoneme_frequencies_ls_100.txt"
)

PLOT_DIR = Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/guided_kmeans/testing_experimental/")

BARPLOT_EPOCH = 9


def load_priors(path: Path) -> dict[str, float]:
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count_str, phoneme = line.split("\t")
            counts[phoneme] = int(count_str)
    total = sum(counts.values())
    return {p: c / total for p, c in counts.items()}


def l1_distance(epoch_counts: dict[str, int], priors: dict[str, float]) -> float:
    total = sum(epoch_counts.get(p, 0) for p in priors)
    if total == 0:
        return float("nan")
    return sum(
        abs(epoch_counts.get(p, 0) / total - prior_freq)
        for p, prior_freq in priors.items()
    )


def main() -> None:
    priors = load_priors(PRIORS_FILE)
    phoneme_order = sorted(priors, key=lambda p: priors[p], reverse=True)

    stats_path = OUTPUT_DIR / "epoch_statistics.json"
    with open(stats_path) as f:
        stats: dict = json.load(f)

    epochs = sorted(stats.keys(), key=int)

    # L1 distance per epoch
    l1_values = [
        l1_distance(stats[ep]["absolute_phoneme_counts"], priors)
        for ep in epochs
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([int(e) for e in epochs], l1_values, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 distance")
    ax.set_title("L1 distance to LibriSpeech-100 unigram priors per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "l1_to_priors.png", dpi=150)
    plt.show()

    # Barplot: priors (top) vs. selected epoch (bottom)
    SILENCE = "[SILENCE]"
    barplot_epoch = str(BARPLOT_EPOCH) if BARPLOT_EPOCH is not None else epochs[-1]
    last_counts = stats[barplot_epoch]["absolute_phoneme_counts"]
    # include silence in the denominator so frequencies sum to 1
    last_total = sum(last_counts.get(p, 0) for p in priors) + last_counts.get(SILENCE, 0)

    x_labels = phoneme_order + [SILENCE]
    colors = ["C0"] * len(phoneme_order) + ["C1"]

    prior_freqs = [priors[p] for p in phoneme_order] + [0.0]
    last_freqs = [last_counts.get(p, 0) / last_total if last_total else 0.0
                  for p in phoneme_order]
    last_freqs += [last_counts.get(SILENCE, 0) / last_total if last_total else 0.0]

    y_max = max(max(prior_freqs), max(last_freqs)) * 1.1

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    ax_top.bar(x_labels, prior_freqs, color=colors)
    ax_top.set_ylabel("Relative frequency")
    ax_top.set_title("Unigram priors (LibriSpeech-100)")
    ax_top.set_ylim(0, y_max)
    ax_top.grid(True, axis="y", alpha=0.3)

    ax_bot.bar(x_labels, last_freqs, color=colors)
    ax_bot.set_ylabel("Relative frequency")
    ax_bot.set_title(f"Training distribution – epoch {barplot_epoch}")
    ax_bot.set_xlabel("Phoneme")
    ax_bot.tick_params(axis="x", rotation=90)
    ax_bot.set_ylim(0, y_max)
    ax_bot.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "phoneme_freq_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
