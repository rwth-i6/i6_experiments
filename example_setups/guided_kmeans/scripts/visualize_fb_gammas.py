"""
Visualize forward-backward soft posteriors (gammas) across clustering epochs

Re-runs the FB search with each epoch's saved centroids on a sample of sequences
and produces four figures:
  1. occupancy.png  — average posterior per phoneme × epoch (heatmap)
  2. entropy.png    — per-frame entropy distribution per epoch (boxplot)
  3. example.png    — gamma heatmap for one utterance, one panel per epoch
  4. alignment.png  — AED-style plot: γ(t, ref_phoneme[n]) for the reference
                       sequence of one utterance, one panel per epoch
"""

import pickle
import sys
sys.path.insert(0, "/work/asr4/lkleppel/rasr_dev/forward-backward/rasr/arch/linux-x86_64-standard")
sys.path.insert(0, "/u/lkleppel/experiments/20260520_unsupervised_asr/recipe")

from itertools import groupby
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial.distance import cdist

FB_JOB = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/work/i6_core/returnn/forward/ReturnnForwardJobV2.6HMH1MexQw07"
)
RASR_CONFIG = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/work/i6_core/rasr/config/WriteRasrConfigJob.rZA2bkCFqHob/output/rasr.config"
)
FEATURES_HDF = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/features/filtered_features_train-clean-100-dbg.hdf"
)
ALIGNMENT_PKL = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids/train-clean-100-dbg/alignments.pkl"
)
OUT_DIR = Path("/u/lkleppel/experiments/20260520_unsupervised_asr/output/guided_kmeans/fb_test/fb_gamma_plots/7epochs")

# How many sequences to use for aggregate statistics (None = all)
MAX_SEQS = None
# Index of the sequence to use for the example/alignment plots (None = shortest sequence)
EXAMPLE_SEQ_IDX = None
# Epochs to include in the alignment plot (None = all epochs found), e.g. [5] or [3, 5, 10]
ALIGNMENT_PLOT_EPOCHS = None
# Scale applied to squared distances before passing to FB search
DISTANCE_SCALE = 0.0001

# ---------------------------------------------------------------------------

PHONEME_NAMES = [
    "[SILENCE]", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH",
    "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]


def normalize_gammas(gammas: np.ndarray) -> np.ndarray:
    # Per-frame L1 normalization to correct float32 accumulation drift
    row_sums = gammas.sum(axis=1, keepdims=True)
    return np.where(
        row_sums > 1e-30,
        gammas / np.maximum(row_sums, 1e-300),
        np.zeros_like(gammas),
    )


def posterior_entropy(posteriors: np.ndarray) -> np.ndarray:
    p = np.clip(posteriors, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def load_features(hdf_path: Path, max_seqs=None):
    with h5py.File(hdf_path, "r") as f:
        lengths = f["seqLengths"][:, 0]
        tags = [t.decode() if isinstance(t, bytes) else t for t in f["seqTags"][:]]
        data = f["inputs"][:]
    seqs = []
    offset = 0
    for tag, length in zip(tags, lengths):
        seqs.append((tag, data[offset : offset + length].astype(np.float32)))
        offset += length
        if max_seqs is not None and len(seqs) >= max_seqs:
            break
    return seqs


def compute_gammas_for_epoch(centroids, seqs, search_algo):
    # Run FB search for all sequences, return per-seq normalized gammas
    results = []
    for tag, features in seqs:
        dists = (cdist(features, centroids, metric="sqeuclidean") * DISTANCE_SCALE).astype(np.float32)
        result = search_algo.recognize_segment_forward_backward(dists)
        raw = np.asarray(result["label_gammas"], dtype=np.float64)
        gammas = normalize_gammas(raw[:, :len(PHONEME_NAMES)])
        results.append((tag, gammas))
    return results


def load_reference_segments(alignment_pkl: Path, full_tag: str):
    # Return the reference phoneme sequence for one utterance as a list of
    # (phoneme_name, start_frame, end_frame) tuples, derived from the GMM alignment
    with open(alignment_pkl, "rb") as f:
        alignment = pickle.load(f)

    short = "/".join(full_tag.split("/")[-2:])
    if short not in alignment:
        # Try the full tag as key
        if full_tag not in alignment:
            raise KeyError(f"Tag {short!r} not found in alignment pickle")
        frame_labels = alignment[full_tag]
    else:
        frame_labels = alignment[short]

    segments = []
    pos = 0
    for label_idx, group in groupby(frame_labels):
        length = sum(1 for _ in group)
        segments.append((PHONEME_NAMES[label_idx], pos, pos + length))
        pos += length
    return segments


def find_centroid_files(job_dir: Path):
    # Find all centroids.N.npy files
    files = {}
    for subdir in ("output", "work"):
        for p in (job_dir / subdir).glob("centroids.*.npy"):
            epoch = int(p.stem.split(".")[1])
            files[epoch] = p
    return dict(sorted(files.items()))


def main():
    OUT_DIR.mkdir(exist_ok=True)

    centroid_files = find_centroid_files(FB_JOB)
    if not centroid_files:
        raise FileNotFoundError(f"No centroid files found in {FB_JOB}")
    epochs = sorted(centroid_files)
    print(f"Found centroids for epochs: {epochs}")

    print(f"Loading features from {FEATURES_HDF}")
    seqs = load_features(FEATURES_HDF, MAX_SEQS)
    print(f"  {len(seqs)} sequences")

    if EXAMPLE_SEQ_IDX is None:
        shortest_seq_idx = min(range(len(seqs)), key=lambda i: seqs[i][1].shape[0])
    else:
        shortest_seq_idx = EXAMPLE_SEQ_IDX
    print(f"  Example sequence: idx={shortest_seq_idx}, tag={seqs[shortest_seq_idx][0]}, T={seqs[shortest_seq_idx][1].shape[0]}")

    from librasr import Configuration, SearchAlgorithm
    config = Configuration()
    config.set_from_file(str(RASR_CONFIG))
    search_algo = SearchAlgorithm(config=config)

    # Collect per-epoch statistics
    epoch_occupancy = []   # (num_epochs, num_phonemes) — mean posterior per phoneme
    epoch_entropy = []     # list of (num_epochs,) arrays of per-frame entropies
    example_gammas = []    # (num_epochs,) gamma matrices for one utterance

    for epoch in epochs:
        centroids = np.load(centroid_files[epoch]).astype(np.float32)
        print(f"Epoch {epoch}: centroids shape={centroids.shape}, norm_mean={np.linalg.norm(centroids, axis=1).mean():.1f}")

        gamma_results = compute_gammas_for_epoch(centroids, seqs, search_algo)

        all_gammas = np.concatenate([g for _, g in gamma_results], axis=0)
        epoch_occupancy.append(all_gammas.mean(axis=0))
        epoch_entropy.append(posterior_entropy(all_gammas))
        example_gammas.append(gamma_results[shortest_seq_idx][1])

        print(f"  mean entropy={epoch_entropy[-1].mean():.3f}, "
              f"top phoneme={PHONEME_NAMES[epoch_occupancy[-1].argmax()]} "
              f"({epoch_occupancy[-1].max()*100:.1f}%)")

    occupancy = np.array(epoch_occupancy)   # (num_epochs, num_phonemes)
    n_phonemes = len(PHONEME_NAMES)

    # Plot 1: occupancy heatmap
    fig, ax = plt.subplots(figsize=(max(10, n_phonemes * 0.4), len(epochs) * 0.6 + 2))
    im = ax.imshow(occupancy, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(n_phonemes))
    ax.set_xticklabels(PHONEME_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(epochs)))
    ax.set_yticklabels([f"Epoch {e}" for e in epochs])
    ax.set_xlabel("Phoneme")
    ax.set_title("Average posterior per phoneme per epoch")
    plt.colorbar(im, ax=ax, label="Mean γ")
    #plt.tight_layout()
    out = OUT_DIR / "occupancy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    # Plot 2: entropy distribution per epoch
    fig, ax = plt.subplots(figsize=(max(6, len(epochs) * 1.2), 4))
    ax.boxplot(
        [epoch_entropy[i] for i in range(len(epochs))],
        labels=[f"E{e}" for e in epochs],
        showfliers=False,
        medianprops=dict(color="steelblue", linewidth=2),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-frame entropy")
    ax.set_title("Posterior entropy distribution across epochs")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)
    #plt.tight_layout()
    out = OUT_DIR / "entropy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    # Load reference segments once for use in plots 3 and 4
    ex_tag = seqs[shortest_seq_idx][0]
    ref_segments = load_reference_segments(ALIGNMENT_PKL, ex_tag)

    # Build frame-level correct-phoneme index array from the reference segments
    ex_T = example_gammas[0].shape[0]
    correct_label = np.full(ex_T, -1, dtype=int)
    for phoneme_name, start, end in ref_segments:
        correct_label[start:min(end, ex_T)] = PHONEME_NAMES.index(phoneme_name)

    # Plot 3: example utterance heatmap (one file per epoch)
    for epoch, gammas in zip(epochs, example_gammas):
        fig, ax = plt.subplots(figsize=(min(20, gammas.shape[0] * 0.05 + 3), n_phonemes * 0.18 + 1))
        im = ax.imshow(gammas.T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(n_phonemes))
        ax.set_yticklabels(PHONEME_NAMES, fontsize=6)
        ax.set_xlabel("Frame")
        ax.set_title(f"Gamma heatmap — epoch {epoch} — {ex_tag.split('/')[-1]}", fontsize=9)
        # Mark the correct reference phoneme at each frame
        valid = correct_label >= 0
        ax.scatter(np.where(valid)[0], correct_label[valid],
                   color="red", s=60, marker=".", zorder=5, linewidths=0)
        plt.colorbar(im, ax=ax, label="γ")
        plt.tight_layout()
        out = OUT_DIR / f"example_epoch{epoch}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

    # Plot 4: AED-style alignment plot (one file per epoch)
    # Rows = reference phoneme occurrences in sequence order, Columns = frames
    # Color = γ(t, ref_phoneme[n]) within each segment's span, 0 elsewhere
    ref_labels = [PHONEME_NAMES.index(name) for name, _, _ in ref_segments]
    N = len(ref_segments)
    y_labels = [f"{name} ({start}-{end})" for name, start, end in ref_segments]

    align_epochs_gammas = [
        (e, g) for e, g in zip(epochs, example_gammas)
        if ALIGNMENT_PLOT_EPOCHS is None or e in ALIGNMENT_PLOT_EPOCHS
    ]
    if not align_epochs_gammas:
        print(f"Warning: ALIGNMENT_PLOT_EPOCHS={ALIGNMENT_PLOT_EPOCHS!r} matched no epochs {epochs}, skipping alignment plot")
    else:
        for epoch, gammas in align_epochs_gammas:
            T = gammas.shape[0]
            fig, ax = plt.subplots(figsize=(min(24, T * 0.04 + 2), max(3, N * 0.15) + 1))
            # A[n, t] = γ(t, ref_phoneme[n]) only within the segment's time span, 0 elsewhere
            A = np.zeros((N, T), dtype=np.float64)
            for n, (_, start, end) in enumerate(ref_segments):
                A[n, start:end] = gammas[start:end, ref_labels[n]]
            im = ax.imshow(A, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                           interpolation="nearest", origin="lower")
            for _, start, end in ref_segments:
                ax.axvline(start, color="gray", linewidth=0.3, alpha=0.5)
            ax.set_yticks(range(N))
            ax.set_yticklabels(y_labels, fontsize=5)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Reference phoneme")
            ax.set_title(
                f"Alignment — epoch {epoch} — {ex_tag.split('/')[-1]}",
                fontsize=9,
            )
            plt.colorbar(im, ax=ax, label="γ")
            plt.tight_layout()
            out = OUT_DIR / f"alignment_epoch{epoch}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved {out}")

    print(f"\nAll plots written to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
