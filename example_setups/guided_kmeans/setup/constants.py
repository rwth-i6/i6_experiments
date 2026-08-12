from sisyphus import tk

PHONEME_UNIGRAM_PRIORS = tk.Path(
    "/work/asr4/lkleppel/experiments/20260520_unsupervised_asr/work/i6_core/corpus/stats/CountCorpusWordFrequenciesJob.VGJUeKIZGWLa/output/counts"
)

_FEATURES_BASE = "/u/lkleppel/experiments/20260520_unsupervised_asr/output/features"
_SEGMENTS_BASE = "/u/lkleppel/experiments/20260520_unsupervised_asr/output/segments_list"
_CENTROIDS_BASE = "/u/lkleppel/experiments/20260520_unsupervised_asr/output/cheating_centroids"
_DANIEL_BASE = "/u/mann/experiments/2026-06-09--guided-k-means/test/cheating_centroids_larissa"

# --- Feature HDFs ---

FEATURES_TRAIN_CLEAN_100_DBG = tk.Path(f"{_FEATURES_BASE}/filtered_features_train-clean-100-dbg.hdf")
FEATURES_LS100H = tk.Path(f"{_FEATURES_BASE}/wav2vec2_ls100h.hdf")
FEATURES_LS100H_SHARDED = [
    tk.Path(f"/work/asr4/lkleppel/experiments/20260520_unsupervised_asr/wav2vec2_ls100h/wav2vec2_ls100h_part{i:02d}.hdf")
    for i in range(10)
]
FEATURES_LS100H_SEGMENTED = tk.Path(f"{_FEATURES_BASE}/segmented_features_wav2vec2_ls100h.hdf")
FEATURES_LS_CV = tk.Path(f"{_FEATURES_BASE}/ls-cv.hdf")
FEATURES_LS_CV_SEGMENTED = tk.Path(f"{_FEATURES_BASE}/segmented_features_ls-cv.hdf")

# --- Segment files ---
SEGMENTS_TRAIN_CLEAN_100_DBG = tk.Path(f"{_SEGMENTS_BASE}/train-clean-100-dbg-segments.txt")
SEGMENTS_LS100H = tk.Path(f"{_SEGMENTS_BASE}/ls100h-segments.txt")
SEGMENTS_LS_CV = tk.Path(f"{_SEGMENTS_BASE}/ls-cv-segments.txt")
SEGMENTS_LS_CV_SEGMENTED = tk.Path(f"{_SEGMENTS_BASE}/ls-cv-segmented-segments.txt")

# --- Cheating centroids and covariances ---
# Centroids computed on the train-clean-100-dbg subset specifically
CHEATING_CENTROIDS_DBG = tk.Path(f"{_CENTROIDS_BASE}/train-clean-100-dbg/centroids.npy")
# Centroids computed on the full LS-960h corpus
CHEATING_CENTROIDS_LS960 = tk.Path(f"{_CENTROIDS_BASE}/centroids.npy")
# Covariances
CHEATING_COVS = tk.Path(f"{_DANIEL_BASE}/covs.npy")

# --- GMM alignments ---
GMM_ALIGNMENT_DBG = tk.Path(f"{_CENTROIDS_BASE}/train-clean-100-dbg/alignments.pkl")
GMM_ALIGNMENT_CV = tk.Path(f"{_CENTROIDS_BASE}/ls-cv/alignments.pkl")

# --- Phoneme frequency reference ---
PHONEME_FREQUENCIES_LS100H = tk.Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/phoneme_frequencies/phoneme_frequencies_ls_100.txt"
)

# --- RASR binary path for librasr (used in cheated clustering configs) ---
RASR_PATH_LIBRASR = tk.Path(
    "/work/asr3/michel/mann/tools/rasr/librasr_recog2/arch/linux-x86_64-standard"
)

# --- Canonical input_data dict ---
# Configs that differ in centroid choice override the relevant entry:
#   clustering_base: train-clean-100-dbg → CHEATING_CENTROIDS_DBG
INPUT_DATA = {
    "train-clean-100-dbg": {
        "features": FEATURES_TRAIN_CLEAN_100_DBG,
        "cheating_centroids": CHEATING_CENTROIDS_LS960,
        "cheating_covs": CHEATING_COVS,
        "segment_file": SEGMENTS_TRAIN_CLEAN_100_DBG,
    },
    "ls-100": {
        "features": FEATURES_LS100H,
        "features_sharded": FEATURES_LS100H_SHARDED,
        "cheating_centroids": CHEATING_CENTROIDS_LS960,
        "cheating_covs": CHEATING_COVS,
        "segment_file": SEGMENTS_LS100H,
    },
    "ls-100-segmented": {
        "features": FEATURES_LS100H_SEGMENTED,
        "cheating_centroids": CHEATING_CENTROIDS_LS960,
        "cheating_covs": CHEATING_COVS,
        "segment_file": SEGMENTS_LS100H,
    },
    "cv": {
        "features": FEATURES_LS_CV,
        "segment_file": SEGMENTS_LS_CV,
    },
    "cv-segmented": {
        "features": FEATURES_LS_CV_SEGMENTED,
        "segment_file": SEGMENTS_LS_CV,
    },
}
