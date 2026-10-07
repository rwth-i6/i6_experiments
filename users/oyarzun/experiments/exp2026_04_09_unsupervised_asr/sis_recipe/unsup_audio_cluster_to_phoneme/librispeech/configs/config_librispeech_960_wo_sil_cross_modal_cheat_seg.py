"""
Config: LibriSpeech 960 (without silence) - Cross-Modal Baseline Translation Quality with Cheating Segmentation

Main Goal:
Test whether cross-modal codes trained using HuBERT and clustering achieve good recognition performance
when using an oracle cheating segmentation.
No AED, no backtranslation, no denoising, no learnable parameters.
For each segment from the oracle cheating segmentation:
  1. Average representations in the segment to a single vector
  2. Find the nearest neighbor text phoneme via cosine similarity against text phoneme embeddings

Assumes cheating segmentation files are located in $HPCWORK/cheating_segmentation.
"""

import glob
import os
from typing import List, Optional
from sisyphus import tk

from ..data.common_hubert import (
    get_audio_embeddings_path,
    get_text_embeddings_path,
    DEFAULT_HUBERT_CLUSTERS_960_HDFS,
    DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS,
    DEFAULT_HUBERT_CLUSTERS_DEV_CLEAN_HDFS,
    HUBERT_DATA_ROOT,
    RSYNC_ALIAS_ROOT,
    HPCWORK_DIR,
)
from ....data.librispeech import text
from ....eval_cross_modal_cheating_seg import CheatingSegmentationNearestNeighborEvalJob
from ... import __setup_base_name__

# -------------------------------------------------------------------------
# Default Paths for Cheating Segmentation
# (Assumed to be in $HPCWORK, transfer from /u/zyang/... on other cluster)
# -------------------------------------------------------------------------
HPCWORK_PATH = "/rwthfs/rz/cluster/hpcwork/bpd03090"
CHEATING_SEG_DIR = os.path.join(HPCWORK_PATH, "cheating_segmentation")


def get_cheating_segmentation_path(k_clusters: int = 512) -> tk.Path:
    """
    Returns path to the cheating segmentation files transferred to $HPCWORK.
    Original source on previous cluster:
    /u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/ls960_gmm_oracle_segment_clustering/
    """
    specific_path = os.path.join(
        CHEATING_SEG_DIR,
        f"ls960_gmm_oracle_segment_clustering/clustering/k{k_clusters}/compat",
    )
    if os.path.exists(specific_path):
        return tk.Path(specific_path)
    for root in [CHEATING_SEG_DIR, HPCWORK_PATH]:
        candidate = os.path.join(root, f"k{k_clusters}")
        if os.path.exists(candidate):
            return tk.Path(candidate)
    # Default directory in $HPCWORK
    return tk.Path(CHEATING_SEG_DIR)


def get_hubert_cluster_hdfs(cluster_mult: int, k_clusters: int, split: str = "train_other_960") -> List[str]:
    """
    Finds cluster HDF files corresponding to a specific cluster ablation (e.g. 1P_41, 4P_164, 200_200).
    """
    clus_dir_map = {
        1: "clus_1P_41",
        2: "clus_2P_82",
        4: "clus_4P_164",
        200: "clus_200_200",
    }
    c_name = clus_dir_map.get(cluster_mult, f"clus_{cluster_mult}P_{k_clusters}")

    for root_dir in [HPCWORK_DIR, HUBERT_DATA_ROOT, os.path.dirname(HUBERT_DATA_ROOT)]:
        target_dir = os.path.join(root_dir, c_name)
        if os.path.exists(target_dir):
            if split == "dev_other":
                files = glob.glob(os.path.join(target_dir, "*dev_other*.hdf"))
            else:
                files = glob.glob(os.path.join(target_dir, "*train_other*.hdf"))
            if files:
                return sorted(files)

    # Fallback to default if cluster-specific HDFs are stored in standard locations
    if split == "dev_other":
        return DEFAULT_HUBERT_CLUSTERS_DEV_OTHER_HDFS
    return DEFAULT_HUBERT_CLUSTERS_960_HDFS


def get_phoneme_vocab_path() -> tk.Path:
    """Finds phoneme vocab.txt from text_w2v_phonemes output or standard lexicon."""
    for root in [HPCWORK_DIR, RSYNC_ALIAS_ROOT]:
        target = os.path.join(root, "train-clean-100", "hubert_l6", "text_w2v_phonemes", "output", "vocab.txt")
        if os.path.exists(target):
            return tk.Path(target)
    # Fallback to general data path
    return tk.Path(os.path.join(HPCWORK_PATH, "data/cross_modal_embeddings/vocab.txt"))


VERSION = "v2"


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}/{VERSION}"

    # Phoneme vocabulary and text embeddings
    text_emb_path = tk.Path(get_text_embeddings_path())
    phon_vocab_path = get_phoneme_vocab_path()

    # Ground-truth reference phonemes (wo_sil: sil_prob=0.0, surround_w_sil=False)
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=0.0,
        surround_w_sil=False,
    )

    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=0.0,
        surround_w_sil=False,
    )

    # Cross-modal cluster configurations: 1P (41), 2P (82), 4P (164), 200 clusters
    cluster_configs = [
        (1, 41, "1P_41"),
        (2, 82, "2P_82"),
        (4, 164, "4P_164"),
        (200, 200, "200_200"),
    ]

    for mult, k_clusters, clus_str in cluster_configs:
        audio_emb_path = tk.Path(get_audio_embeddings_path(cluster_mult=mult, init_method="freq"))
        dev_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=mult, k_clusters=k_clusters, split="dev_other")
        train_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=mult, k_clusters=k_clusters, split="train_other_960")
        seg_path = get_cheating_segmentation_path(k_clusters=k_clusters)

        # 1. Evaluation on dev-other split
        eval_dev_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=audio_emb_path,
            text_embeddings_npy=text_emb_path,
            phoneme_vocab_file=phon_vocab_path,
            cheating_segmentation_path=seg_path,
            audio_cluster_hdfs=dev_cluster_hdfs,
            reference_phonemes=phoneme_dev_other_hdfs,
            seq_tags_file=dev_other_seq_tags,
            ignore_silence=True,
            name_prefix=f"eval_dev_other_{clus_str}_{VERSION}",
        )
        eval_dev_job.add_alias(f"{prefix_name}/{clus_str}/eval_dev_other")
        tk.register_output(f"{prefix_name}/{clus_str}/dev_other_per_summary.json", eval_dev_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{clus_str}/dev_other_per_report.txt", eval_dev_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/dev_other_hypotheses.txt", eval_dev_job.out_hypotheses_txt)

        # 2. Evaluation on train-other-960 (or sampled subset)
        eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=audio_emb_path,
            text_embeddings_npy=text_emb_path,
            phoneme_vocab_file=phon_vocab_path,
            cheating_segmentation_path=seg_path,
            audio_cluster_hdfs=train_cluster_hdfs,
            reference_phonemes=phoneme_960_hdfs,
            seq_tags_file=train_seq_tags,
            max_seqs=3000,  # 3000 sequences standard subset matching devtrain
            ignore_silence=True,
            name_prefix=f"eval_train3k_{clus_str}_{VERSION}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{clus_str}/eval_train3k")
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)


