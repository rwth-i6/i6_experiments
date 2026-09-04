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
SHARED_COVS = tk.Path(f"{_DANIEL_BASE}/shared_cov.npy") # unclustered covariance matrix duplicated 40 times [40, 512, 512]

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

# --- Cross-setup inputs from a colleague's vector-quantized HMM work ---------
# Kept here rather than in a config because two configs now use them, and
# because the provenance matters more than the paths do.

#: FAISS k-means codebook, 512 centroids over 512-dim wav2vec2, trained on
#: *silence-free* features. FAISS minimizes plain L2, which is the metric
#: VectorQuantizedModel.quantize uses, so this codebook partitions here exactly
#: as it does in the setup it came from. Measured against a reference alignment
#: it supports 67.2% frame accuracy with an oracle table; our own stage-1
#: codebook reaches 65.6% at the same size.
COLLEAGUE_CENTROIDS_K512 = tk.Path(
    "/work/asr4/zyang/mini/work/i6_experiments/users/yang/experiments/generative_ctc"
    "/example_setups/librispeech/phmm/segment_clustering_jobs"
    "/TrainFaissKMeansJob.DHEI3eu0otkT/output/centers_k512.npy"
)

#: LibriSpeech-960 GMM segment representations, 20 shards: 278,400 sequences and
#: 32,666,275 vectors of dim 512, one per phoneme segment and already
#: silence-free (produced after VAD). The ``labels`` dataset in these files is
#: an empty placeholder - they carry no supervision.
COLLEAGUE_SEGMENT_FEATURES_LS960 = [
    tk.Path(f"/work/asr4/zyang/mini/share/gmm_segment_reps_k512_input/gmm_segment_reps.{i:03d}.hdf")
    for i in range(20)
]

#: A phoneme trigram distilled to ARPA from a convolutional LM (41 unigrams,
#: 1600 bigrams, 64000 trigrams; no end-of-word marker, so it pairs with
#: use_eow_phonemes=False). Sharper than this setup's count LM in the tail: it
#: assigns markedly smaller probabilities to unlikely trigrams. Pass it as
#: ``create_recog_rasr_config(lm_path=...)``, which overrides the order-indexed
#: default.
PHONEME_LM_ZIJIAN_3GRAM = tk.Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/phon_lm/lm_zijian"
    "/3gram/phoneme_trigram_no_eow_epoch200.arpa"
)

#: Frame-level GMM phoneme alignments for all of LibriSpeech-960, 20 RETURNN
#: shards. Silence is index 0 and the remaining indices are this setup's lexicon
#: order - established by comparison rather than assumed: 144 sequences occur in
#: both these shards and ``GMM_ALIGNMENT_CV``, and they agree on 100% of frames
#: at identical lengths. (The ``labels`` dataset inside the files is a numeric
#: placeholder '0'..'39' and carries no inventory.)
#:
#: **Tags use the ``train-other-960/`` corpus prefix**, while the ls-100h
#: feature file uses ``train-clean-100/`` - so there are zero exact tag matches
#: between them and the join has to strip that first segment. Doing so covers
#: all 28,234 ls-100h sequences with no length disagreement (15,160,794 frames,
#: 8.2% silence). SegmentedFeaturesFromAlignmentJob handles this.
GMM_ALIGNMENT_LS960_FRAME = [
    tk.Path(
        "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024"
        f"/ls960_fairseq_wav2vecu_frame_and_segment_reclustering/frame"
        f"/phoneme_ref_from_gmm_vad/phoneme_ref.{i:03d}.hdf"
    )
    for i in range(20)
]

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

INITIALIZATIONS = {
    "shared_covariances": SHARED_COVS,
    "cheating_centroids": CHEATING_CENTROIDS_LS960,
    "cheating_covs": CHEATING_COVS,
}
