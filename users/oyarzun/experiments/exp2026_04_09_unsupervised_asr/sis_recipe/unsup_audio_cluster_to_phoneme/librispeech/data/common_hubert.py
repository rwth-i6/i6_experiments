"""
Data preparation for HuBERT-based cluster sequences and cross-modal embedding alignment.
Integrates the HuBERT extraction, clustering, and embedding scheme from exp2026_09_15_self_learning_emb_asr.
"""

from typing import List, Optional, Dict, Tuple
import os

from sisyphus import tk
from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset

from ....data.librispeech import text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings

# -------------------------------------------------------------------------
# Abstracted Paths for Precomputed HuBERT Clusters & Cross-Modal Embeddings
# (Can be overwritten via arguments or rsync'd to these standard locations)
# -------------------------------------------------------------------------
HUBERT_DATA_ROOT = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/hubert_cluster_indices"
if not os.path.exists(HUBERT_DATA_ROOT) or not os.listdir(HUBERT_DATA_ROOT):
    HUBERT_DATA_ROOT = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/librispeech_cluster_indices"
EMBEDDINGS_ROOT = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/cross_modal_embeddings"
RSYNC_ALIAS_ROOT = "/rwthfs/rz/cluster/home/p0023999/experiments/2026_05_07_first_experiments/alias/experiments/self_learning_emb_asr"

DEFAULT_HUBERT_CLUSTERS_960_HDFS = [
    os.path.join(HUBERT_DATA_ROOT, f"train_other_960_data_{i}.hdf")
    for i in range(10)
]
DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS = [
    os.path.join(HUBERT_DATA_ROOT, "dev_other_data_0.hdf")
]
DEFAULT_HUBERT_CLUSTERS_DEV_CLEAN_HDFS = [
    os.path.join(HUBERT_DATA_ROOT, "dev_clean_data_0.hdf")
]

HPCWORK_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr/data/self_learning_emb_asr"
DEFAULT_AUDIO_CROSS_MODAL_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_ROOT, "mapped_audio_embeddings.npy")
DEFAULT_TEXT_CROSS_MODAL_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_ROOT, "phoneme_embeddings.npy")

def resolve_embedding_path(default_path: str, filename: str) -> str:
    """Finds embedding file if placed in default_path, hpcwork, or alias directory."""
    if os.path.exists(default_path):
        return default_path
    for search_root in [HPCWORK_DIR, EMBEDDINGS_ROOT, RSYNC_ALIAS_ROOT]:
        if os.path.exists(search_root):
            for r, _, files in os.walk(search_root):
                if filename in files:
                    found = os.path.join(r, filename)
                    return found
    return default_path

def get_audio_embeddings_path(
    cluster_mult: int = 1,
    init_method: str = "freq",
    seed: int = 42,
) -> str:
    """
    Finds mapped audio embeddings for the specified cluster ablation and alignment init method.
    Default: clus_1P_41 with frequency-based initialization.
    """
    cluster_dir_map = {1: "clus_1P_41", 2: "clus_2P_82", 4: "clus_4P_164"}
    c_name = cluster_dir_map.get(cluster_mult, f"clus_{cluster_mult}P_{cluster_mult * 41}")
    align_dir = "align_init_freq" if init_method == "freq" else f"align_init_rand_seed{seed}"

    for root in [HPCWORK_DIR, RSYNC_ALIAS_ROOT]:
        target = os.path.join(root, "train-clean-100", "hubert_l6", c_name, align_dir, "output", "mapped_audio_embeddings.npy")
        if os.path.exists(target):
            return target
        target_wstar = os.path.join(root, "train-clean-100", "hubert_l6", c_name, align_dir, "output", "W_star.npy")
        if os.path.exists(target_wstar):
            return target_wstar

    # Fallback to search-based resolution
    p = resolve_embedding_path(DEFAULT_AUDIO_CROSS_MODAL_EMBEDDINGS_PATH, "mapped_audio_embeddings.npy")
    if not os.path.exists(p):
        p = resolve_embedding_path(p, "W_star.npy")
    return p

def get_text_embeddings_path() -> str:
    p = resolve_embedding_path(DEFAULT_TEXT_CROSS_MODAL_EMBEDDINGS_PATH, "phoneme_embeddings.npy")
    if not os.path.exists(p):
        for root in [HPCWORK_DIR, RSYNC_ALIAS_ROOT]:
            candidate = os.path.join(root, "train-clean-100", "hubert_l6", "text_w2v_phonemes", "output", "embeddings.npy")
            if os.path.exists(candidate):
                return candidate
    return p



def build_training_datasets_hubert(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    num_clusters: int = 128,
    clusters_960_hdfs: Optional[List[str]] = None,
    clusters_dev_other_hdfs: Optional[List[str]] = None,
    clusters_dev_clean_hdfs: Optional[List[str]] = None,
) -> TrainingDatasets:
    """
    Builds TrainingDatasets where the acoustic stream ('data') consists of HuBERT-derived
    quantized cluster tokens, and the text stream ('target') consists of phonemes.
    """
    if clusters_960_hdfs is None:
        clusters_960_hdfs = DEFAULT_HUBERT_CLUSTERS_960_HDFS
    if clusters_dev_other_hdfs is None:
        clusters_dev_other_hdfs = DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS
    if clusters_dev_clean_hdfs is None:
        clusters_dev_clean_hdfs = DEFAULT_HUBERT_CLUSTERS_DEV_CLEAN_HDFS

    # Phonemized text corpus preparation
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out
    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    return TrainingDatasets(
        add_opts={"line_based_lexicon_file": lexicon_file},
        train=CombinedDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                ("phon_indices", "data"): "target",
                ("feature_clusters", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "target",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_dev_other_hdfs + clusters_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                },
                data_map={
                    ("phon_indices", "data"): "target",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=num_clusters,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_test_datasets_hubert(
    clusters_dev_other_hdfs: Optional[List[str]] = None,
) -> Dict[str, MetaDataset]:
    """
    Builds the test dataset for evaluation using HuBERT cluster tokens.
    """
    if clusters_dev_other_hdfs is None:
        clusters_dev_other_hdfs = DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS

    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_dev_hdfs, _, _, dev_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
    )

    return {
        "dev-other": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_dev_other_hdfs,
                    segment_file=dev_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_dev_hdfs,
                    segment_file=dev_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }
