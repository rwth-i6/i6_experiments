"""
Config: LibriSpeech 960 (without silence) - Cross-Modal Baseline Translation Quality with Cheating Segmentation

Main Goal:
Test whether cross-modal codes trained using HuBERT and clustering achieve good recognition performance
when using an oracle cheating segmentation.
No AED, no backtranslation, no denoising, no learnable parameters.
For each segment from the oracle cheating segmentation:
  1. Average representations in the segment to a single vector
  2. Find the nearest neighbor text phoneme via cosine similarity against text phoneme embeddings

Evaluates:
- v1 (legacy): 1P_41, 2P_82, 4P_164, 200_200 monophones
- v2: 1P_40, 2P_80, 4P_160, 200_200 monophones (frequency init)
- v3: Diphone and Triphone codes across K in {200, 250, 500} and window sizes in {3, 5, 10}
"""

import glob
import os
from typing import List, Optional
from sisyphus import tk

from ....cluster_adapter import (
    get_cheating_segmentation_path,
    get_cheating_phoneme_ref_hdfs,
    get_hubert_cluster_hdfs,
    get_hubert_seq_tags_path,
    get_oracle_idx_to_phoneme_path,
    get_audio_embeddings_path,
    get_text_embeddings_path,
    get_phoneme_vocab_path,
    get_v2_audio_embeddings_path,
    get_v2_text_embeddings_path,
    get_v2_phoneme_vocab_path,
    get_v3_audio_embeddings_path,
    get_v3_text_embeddings_path,
    get_v3_vocab_path,
    get_cluster_counts_path,
    get_raw_clusters_path,
    get_collapsed_clusters_path,
    get_v3_counts_path,
    get_v4_cheat_seg_tokens_path,
    get_v5_cheat_seg_tokens_path,
    get_v5_w2v2_cheat_seg_tokens_path,
    V2_CLUSTER_CONFIGS,
    V3_CODE_TYPES,
    V3_CLUSTER_COUNTS,
    V3_WINDOWS,
    set_cluster,
    get_cluster,
)
from ....data.librispeech import text
from ....eval_cross_modal_cheating_seg import CheatingSegmentationNearestNeighborEvalJob
from ....analysis.distribution_analysis import (
    TextNgramDistributionAnalysisJob,
    SpeechClusterDistributionAnalysisJob,
)
from ... import __setup_base_name__


# =============================================================================
# v1 Evaluation Pipeline
# =============================================================================

def eval_v1(cluster: Optional[str] = None):
    """
    Evaluates v1 legacy cross-modal codes (1P_41, 2P_82, 4P_164, 200_200).
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v1"

    # Phoneme vocabulary and text embeddings
    text_emb_raw = get_text_embeddings_path()
    if not text_emb_raw or not os.path.exists(text_emb_raw):
        return
    text_emb_path = tk.Path(text_emb_raw)
    phon_vocab_path = get_phoneme_vocab_path()

    # Ground-truth reference phonemes
    lexicon_file = text.get_librispeech_lexicon()
    phoneme_vocab = get_phoneme_vocab_path()
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
        audio_emb_raw = get_audio_embeddings_path(cluster_mult=mult, init_method="freq")
        if not audio_emb_raw or not os.path.exists(audio_emb_raw):
            continue
        audio_emb_path = tk.Path(audio_emb_raw)
        dev_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=mult, k_clusters=k_clusters, split="dev_other")
        train_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=mult, k_clusters=k_clusters, split="train_other_960")
        seg_path = get_cheating_segmentation_path(k_clusters=k_clusters)

        # 1. Evaluation on dev-other split (Disabled: GMM oracle cheating segmentation is only available for train-960)
        # eval_dev_job = CheatingSegmentationNearestNeighborEvalJob(...)

        # 2. Evaluation on train-other-960 (or sampled subset)
        eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=audio_emb_path,
            text_embeddings_npy=text_emb_path,
            phoneme_vocab_file=phon_vocab_path,
            cheating_segmentation_path=seg_path,
            audio_cluster_hdfs=train_cluster_hdfs,
            reference_phonemes=phoneme_960_hdfs,
            reference_phoneme_vocab_file=phoneme_vocab,
            seq_tags_file=train_seq_tags,
            code_type="monophone",
            max_seqs=3000,
            ignore_silence=True,
            name_prefix=f"eval_train3k_{clus_str}_v1",
        )
        eval_train_job.add_alias(f"{prefix_name}/{clus_str}/eval_train3k")
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


# =============================================================================
# v2 Evaluation Pipeline (Monophone Ablations: 1P_40, 2P_80, 4P_160, 200_200)
# =============================================================================

def eval_v2(cluster: Optional[str] = None):
    """
    Evaluates v2 monophone cross-modal codes (1P_40, 2P_80, 4P_160, 200_200)
    with frequency initialization using cheating segmentation.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v2"

    text_emb_raw = get_v2_text_embeddings_path()
    if not text_emb_raw or not os.path.exists(text_emb_raw):
        return
    text_emb_path = tk.Path(text_emb_raw)
    phon_vocab_path = get_v2_phoneme_vocab_path()

    lexicon_file = text.get_librispeech_lexicon()
    phoneme_vocab = get_v2_phoneme_vocab_path()
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

    for mult, k_clusters, clus_str in V2_CLUSTER_CONFIGS:
        audio_emb_raw = get_v2_audio_embeddings_path(clus_str)
        if not audio_emb_raw or not os.path.exists(audio_emb_raw):
            continue
        audio_emb_path = tk.Path(audio_emb_raw)
        dev_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="dev_other")
        train_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="train_other_960")
        seg_path = get_cheating_segmentation_path(k_clusters=k_clusters)

        # 1. Dev-other (Disabled: GMM oracle cheating segmentation is only available for train-960)
        # eval_dev_job = CheatingSegmentationNearestNeighborEvalJob(...)

        # 2. Train-other-960 subset (3000 utts)
        eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=audio_emb_path,
            text_embeddings_npy=text_emb_path,
            phoneme_vocab_file=phon_vocab_path,
            cheating_segmentation_path=seg_path,
            audio_cluster_hdfs=train_cluster_hdfs,
            reference_phonemes=phoneme_960_hdfs,
            reference_phoneme_vocab_file=phoneme_vocab,
            seq_tags_file=train_seq_tags,
            code_type="monophone",
            max_seqs=3000,
            ignore_silence=True,
            name_prefix=f"eval_train3k_{clus_str}_v2",
        )
        eval_train_job.add_alias(f"{prefix_name}/{clus_str}/eval_train3k")
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{clus_str}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


# =============================================================================
# v3 Evaluation Pipeline (Diphone & Triphone Codes: K in {200, 250, 500}, win in {3, 5, 10})
# =============================================================================

def eval_v3(cluster: Optional[str] = None):
    """
    Evaluates v3 diphone and triphone cross-modal codes across cluster sizes
    K in {200, 250, 500} and context windows in {3, 5, 10} using cheating segmentation.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v3"

    lexicon_file = text.get_librispeech_lexicon()
    phoneme_vocab = get_phoneme_vocab_path()
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

    for code_type in V3_CODE_TYPES:
        for k_clusters, clus_str in V3_CLUSTER_COUNTS:
            for win in V3_WINDOWS:
                exp_key = f"{code_type}/{clus_str}/win{win}"
                audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win)
                text_emb_raw = get_v3_text_embeddings_path(code_type, win=win)
                if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                    continue
                if not text_emb_raw or not os.path.exists(text_emb_raw):
                    continue

                audio_emb_path = tk.Path(audio_emb_raw)
                text_emb_path = tk.Path(text_emb_raw)
                vocab_path = get_v3_vocab_path(code_type, win=win)

                dev_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="dev_other")
                train_cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="train_other_960")
                seg_path = get_cheating_segmentation_path(k_clusters=k_clusters)

                # 1. Dev-other (Disabled: GMM oracle cheating segmentation is only available for train-960)
                # eval_dev_job = CheatingSegmentationNearestNeighborEvalJob(...)

                # 2. Train-other-960 subset (3000 utts)
                eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                    audio_embeddings_npy=audio_emb_path,
                    text_embeddings_npy=text_emb_path,
                    phoneme_vocab_file=vocab_path,
                    cheating_segmentation_path=seg_path,
                    audio_cluster_hdfs=train_cluster_hdfs,
                    reference_phonemes=phoneme_960_hdfs,
                    reference_phoneme_vocab_file=phoneme_vocab,
                    seq_tags_file=train_seq_tags,
                    code_type=code_type,
                    max_seqs=3000,
                    ignore_silence=True,
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_win{win}_v3",
                )
                eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)
def eval_v3_matched_90pct(cluster: Optional[str] = None):
    """
    Evaluates v3 matched-dimensionality cross-modal codes where K_audio matches the 90% coverage:
    - diphones: K = 483 (483x483 square cross-modal matrix)
    - triphones: K = 6,161 (6,161x6,161 square cross-modal matrix)
    using cheating segmentation.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v3_matched_90pct"

    oracle_seg_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    matched_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
    ]
    for code_type, k_clusters, clus_str in matched_setups:
        raw_clus_file = get_raw_clusters_path(clus_str)
        train_cluster_seqs = [raw_clus_file] if raw_clus_file is not None else get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="train_clean_100")

        for dim in [64, 128, 300]:
            for win in V3_WINDOWS:
                for metric in ["cosine", "csls"]:
                    dw_tag = f"win{win}" if dim == 300 else f"dim{dim}_win{win}"
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}" if metric == "cosine" else f"{code_type}/{clus_str}/{dw_tag}_{metric}"

                    audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win, dim=dim, retrieval_metric=metric)
                    text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim)
                    if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                        continue
                    if not text_emb_raw or not os.path.exists(text_emb_raw):
                        continue

                    audio_emb_path = tk.Path(audio_emb_raw)
                    text_emb_path = tk.Path(text_emb_raw)
                    vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim)

                    # Option 1: Real Frame Pooling with HuBERT Sequences across Oracle Segment Boundaries (3000 utts subset)
                    eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                        audio_embeddings_npy=audio_emb_path,
                        text_embeddings_npy=text_emb_path,
                        phoneme_vocab_file=vocab_path,
                        cheating_segmentation_path=oracle_seg_hdfs,
                        audio_cluster_hdfs=train_cluster_seqs,
                        reference_phonemes=None,
                        reference_phoneme_vocab_file=oracle_idx_to_phoneme,
                        seq_tags_file=oracle_seq_tags,
                        code_type=code_type,
                        max_seqs=3000,
                        ignore_silence=True,
                        retrieval_metric=metric,
                        csls_k=10,
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_matched90",
                    )
                    eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


def eval_v3_matched_90pct_direct_seg(cluster: Optional[str] = None):
    """
    Evaluates v3 matched-dimensionality cross-modal codes using DIRECT SEGMENTATION
    (i.e. using HuBERT+KMeans collapsed tokens from the training, rather than oracle GMM phoneme boundaries).
    - diphones: K = 483 (483x483)
    - triphones: K = 6,161 (6,161x6,161)
    Focusing on smaller dimensions below 300 (dim in {64, 128}).
    Output directory: cross_modal_direct_seg_{cluster}/v3_matched_90pct
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_direct_seg_{active_cluster}/v3_matched_90pct"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    matched_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
    ]
    for code_type, k_clusters, clus_str in matched_setups:
        collapsed_clus_file = get_collapsed_clusters_path(clus_str)
        if collapsed_clus_file is None:
            continue

        for dim in [64, 128]:
            for win in V3_WINDOWS:
                for metric in ["cosine", "csls"]:
                    dw_tag = f"dim{dim}_win{win}"
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}" if metric == "cosine" else f"{code_type}/{clus_str}/{dw_tag}_{metric}"

                    audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win, dim=dim, retrieval_metric=metric)
                    text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim)
                    if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                        continue
                    if not text_emb_raw or not os.path.exists(text_emb_raw):
                        continue

                    audio_emb_path = tk.Path(audio_emb_raw)
                    text_emb_path = tk.Path(text_emb_raw)
                    vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim)

                    # Direct Segmentation Evaluation: token-by-token mapping from HuBERT+Kmeans collapsed tokens
                    eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                        audio_embeddings_npy=audio_emb_path,
                        text_embeddings_npy=text_emb_path,
                        phoneme_vocab_file=vocab_path,
                        cheating_segmentation_path=collapsed_clus_file,
                        audio_cluster_hdfs=None,
                        reference_phonemes=oracle_ref_hdfs,
                        reference_phoneme_vocab_file=oracle_idx_to_phoneme,
                        seq_tags_file=oracle_seq_tags,
                        code_type=code_type,
                        max_seqs=3000,
                        ignore_silence=True,
                        retrieval_metric=metric,
                        csls_k=10,
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_direct_seg",
                    )
                    eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


# =============================================================================
# v4 Evaluation Pipeline (Corrected Segment-Averaged HuBERT K-Means + CSLS)
# =============================================================================

def eval_v4_matched_90pct(cluster: Optional[str] = None):
    """
    Evaluates v4 corrected segment-averaged HuBERT K-Means setup:
    Continuous 768-dim HuBERT frames within oracle segmentation boundaries were averaged,
    quantized by K-Means, and aligned via Word2Vec + Artetxe CSLS into text space.
    Scope:
      - Matched 90% coverage setups: diphone (K=483) and triphone (K=6161)
      - Word2Vec dimensions: [64, 128] (smaller dimensions below 300)
      - Context windows: [1, 2, 3, 5] (smaller windows, omitting window 10)
      - Metric: CSLS only (k=10)
      - Normalization: center_text_embeddings=True (normalize_and_center)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v4_matched_90pct"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    matched_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
    ]
    dims = [64, 128]
    windows = [1, 2, 3, 5]
    metric = "csls"

    for code_type, k_clusters, clus_str in matched_setups:
        cheat_seg_tokens_file = get_v4_cheat_seg_tokens_path(clus_str)

        for dim in dims:
            for win in windows:
                dw_tag = f"dim{dim}_win{win}"
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"

                audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win, dim=dim, retrieval_metric=metric)
                text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim)
                if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                    continue
                if not text_emb_raw or not os.path.exists(text_emb_raw):
                    continue

                audio_emb_path = tk.Path(audio_emb_raw)
                text_emb_path = tk.Path(text_emb_raw)
                vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim)

                eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                    audio_embeddings_npy=audio_emb_path,
                    text_embeddings_npy=text_emb_path,
                    phoneme_vocab_file=vocab_path,
                    cheating_segmentation_path=cheat_seg_tokens_file,
                    audio_cluster_hdfs=None,
                    reference_phonemes=oracle_ref_hdfs,
                    reference_phoneme_vocab_file=oracle_idx_to_phoneme,
                    seq_tags_file=oracle_seq_tags,
                    code_type=code_type,
                    max_seqs=3000,
                    ignore_silence=True,
                    retrieval_metric=metric,
                    csls_k=10,
                    center_text_embeddings=True,
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v4",
                )
                eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


# =============================================================================
# v5 Evaluation Pipeline (Corrected Segment-Averaged HuBERT 512-dim PCA + K-Means + CSLS)
# =============================================================================

def eval_v5_matched_90pct(cluster: Optional[str] = None, pca_dim: int = 512):
    """
    Evaluates v5 setup:
    Continuous 768-dim HuBERT frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, quantized by K-Means,
    and aligned via Word2Vec + Artetxe CSLS into text space.
    Scope:
      - Matched 90% coverage setups: diphone (K=483) and triphone (K=6161)
      - Word2Vec dimensions: [64, 128]
      - Context windows: [1, 2, 3, 5]
      - Metric: CSLS only (k=10)
      - Normalization: center_text_embeddings=True (normalize_and_center)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5_matched_90pct_pca{pca_dim}"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    matched_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
    ]
    dims = [64, 128]
    windows = [1, 2, 3, 5]
    metric = "csls"

    for code_type, k_clusters, clus_str in matched_setups:
        cheat_seg_tokens_file = get_v5_cheat_seg_tokens_path(clus_str, cluster=active_cluster, pca_dim=pca_dim)
        if cheat_seg_tokens_file is None:
            continue

        for dim in dims:
            for win in windows:
                dw_tag = f"dim{dim}_win{win}"
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"

                audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win, dim=dim, retrieval_metric=metric)
                text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim)
                if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                    continue
                if not text_emb_raw or not os.path.exists(text_emb_raw):
                    continue

                audio_emb_path = tk.Path(audio_emb_raw)
                text_emb_path = tk.Path(text_emb_raw)
                vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim)

                eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                    audio_embeddings_npy=audio_emb_path,
                    text_embeddings_npy=text_emb_path,
                    phoneme_vocab_file=vocab_path,
                    cheating_segmentation_path=cheat_seg_tokens_file,
                    audio_cluster_hdfs=None,
                    reference_phonemes=oracle_ref_hdfs,
                    reference_phoneme_vocab_file=oracle_idx_to_phoneme,
                    seq_tags_file=oracle_seq_tags,
                    code_type=code_type,
                    max_seqs=3000,
                    ignore_silence=True,
                    retrieval_metric=metric,
                    csls_k=10,
                    center_text_embeddings=True,
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)


# =============================================================================
# v5-wav2vec2.0 Evaluation Pipeline (wav2vec 2.0 Large-60k Layer 15 + PCA 512 + K-Means + CSLS)
# =============================================================================

def eval_v5_w2v2_matched_90pct(cluster: Optional[str] = None, pca_dim: int = 512):
    """
    Evaluates v5-wav2vec2.0 setup:
    Continuous 1024-dim wav2vec 2.0 Large-60k layer 15 frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, quantized by K-Means,
    and aligned via Word2Vec + Artetxe CSLS into text space.
    Scope:
      - Matched 90% coverage setups: diphone (K=483) and triphone (K=6161)
      - Word2Vec dimensions: [64, 128]
      - Context windows: [1, 2, 3, 5]
      - Metric: CSLS only (k=10)
      - Normalization: center_text_embeddings=True (normalize_and_center)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5_w2v2_matched_90pct_pca{pca_dim}"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    matched_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
    ]
    dims = [64, 128]
    windows = [1, 2, 3, 5]
    metric = "csls"

    for code_type, k_clusters, clus_str in matched_setups:
        cheat_seg_tokens_file = get_v5_w2v2_cheat_seg_tokens_path(clus_str, cluster=active_cluster, pca_dim=pca_dim)
        if cheat_seg_tokens_file is None:
            continue

        for dim in dims:
            for win in windows:
                dw_tag = f"dim{dim}_win{win}"
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"

                audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win, dim=dim, retrieval_metric=metric)
                text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim)
                if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                    continue
                if not text_emb_raw or not os.path.exists(text_emb_raw):
                    continue

                audio_emb_path = tk.Path(audio_emb_raw)
                text_emb_path = tk.Path(text_emb_raw)
                vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim)

                eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
                    audio_embeddings_npy=audio_emb_path,
                    text_embeddings_npy=text_emb_path,
                    phoneme_vocab_file=vocab_path,
                    cheating_segmentation_path=cheat_seg_tokens_file,
                    audio_cluster_hdfs=None,
                    reference_phonemes=oracle_ref_hdfs,
                    reference_phoneme_vocab_file=oracle_idx_to_phoneme,
                    seq_tags_file=oracle_seq_tags,
                    code_type=code_type,
                    max_seqs=3000,
                    ignore_silence=True,
                    retrieval_metric=metric,
                    csls_k=10,
                    center_text_embeddings=True,
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5_w2v2_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)




# =============================================================================
# Corpus Distribution Analysis Pipeline
# =============================================================================

def run_distribution_analysis(cluster: Optional[str] = None):
    """
    Computes and plots frequency distribution histograms for:
    1. Text bigrams and trigrams in the corpus (sorted descending: most frequent to least frequent)
    2. Discovered speech clusters (frame assignments per cluster across the overall corpus)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/analysis"

    # 1. Text N-gram Frequency Analysis (Diphones & Triphones)
    raw_text_data, _ = text.get_text("train-other-960")
    diphone_counts = get_v3_counts_path("diphone")
    triphone_counts = get_v3_counts_path("triphone")
    text_analysis_job = TextNgramDistributionAnalysisJob(
        text_file=raw_text_data if diphone_counts is None else None,
        bigram_counts_file=diphone_counts,
        trigram_counts_file=triphone_counts,
        top_k=50,
        ignore_silence=True,
        title_prefix="LibriSpeech Phoneme N-grams (Diphones & Triphones)",
        name_prefix="ngram_analysis_phonemes",
    )
    text_analysis_job.add_alias(f"{prefix_name}/text_ngrams_train_other_960")
    tk.register_output(f"{prefix_name}/text/bigram_histogram.png", text_analysis_job.out_bigram_plot)
    tk.register_output(f"{prefix_name}/text/trigram_histogram.png", text_analysis_job.out_trigram_plot)
    tk.register_output(f"{prefix_name}/text/ngram_combined_histogram.png", text_analysis_job.out_combined_plot)
    tk.register_output(f"{prefix_name}/text/ngram_cumulative_coverage.png", text_analysis_job.out_coverage_plot)
    tk.register_output(f"{prefix_name}/text/ngram_counts.txt", text_analysis_job.out_counts_txt)
    tk.register_output(f"{prefix_name}/text/ngram_distribution.json", text_analysis_job.out_summary_json)
    tk.register_output(f"{prefix_name}/text/ngram_report.txt", text_analysis_job.out_report_txt)

    # 2. Speech Cluster Distribution Analysis
    # Evaluates discovered clusters across both monophone (v2) and multi-token (v3) sets
    cluster_configs = [
        ("1P_40", 40, "v2 Monophone 40 Clusters"),
        ("2P_80", 80, "v2 Monophone 80 Clusters"),
        ("4P_160", 160, "v2 Monophone 160 Clusters"),
        ("200_200", 200, "v2/v3 200 Clusters"),
        ("250_250", 250, "v3 250 Clusters"),
        ("500_500", 500, "v3 500 Clusters"),
    ]

    for clus_str, k_clusters, label in cluster_configs:
        counts_path = get_cluster_counts_path(clus_str)
        raw_clus_path = get_raw_clusters_path(clus_str)
        cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split="train_other_960")

        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
            cluster_counts_file=counts_path,
            raw_clusters_file=raw_clus_path,
            cluster_hdfs=cluster_hdfs,
            k_clusters=k_clusters,
            top_k=50,
            title_prefix=f"{label} ({clus_str})",
            name_prefix=f"speech_cluster_dist_{clus_str}",
        )
        cluster_analysis_job.add_alias(f"{prefix_name}/speech_clusters/{clus_str}")
        tk.register_output(f"{prefix_name}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
        tk.register_output(f"{prefix_name}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
        tk.register_output(f"{prefix_name}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
        tk.register_output(f"{prefix_name}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
        tk.register_output(f"{prefix_name}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# Main Entry Point
# =============================================================================

def py(cluster: Optional[str] = None, version: str = "all", run_analysis: bool = True):
    """
    Runs Cheating Segmentation Nearest Neighbor evaluation pipeline and corpus distribution analysis.
    :param cluster: 'curium' for local cluster or 'rwth-hpc' / 'rwth-hcp' for RWTH HPC cluster.
    :param version: 'v1', 'v2', 'v3', 'v2_v3', 'all', or 'analysis'.
    :param run_analysis: Whether to run text N-gram and speech cluster distribution analysis.
    """
    v_norm = version.lower()
    if v_norm in ("v1", "legacy"):
        eval_v1(cluster=cluster)
    elif v_norm in ("v2",):
        eval_v2(cluster=cluster)
    elif v_norm in ("v3",):
        eval_v3(cluster=cluster)
    elif v_norm in ("v3_matched", "v3_matched_90", "matched_90", "matched", "v3_90"):
        eval_v3_matched_90pct(cluster=cluster)
    elif v_norm in ("v3_matched_90_direct_seg", "v3_direct_seg", "direct_seg", "direct", "direct_segmentation"):
        eval_v3_matched_90pct_direct_seg(cluster=cluster)
    elif v_norm in ("v4", "v4_matched_90pct", "v4_matched_90", "v4_90", "v4_matched"):
        eval_v4_matched_90pct(cluster=cluster)
    elif v_norm in ("v2_v3", "v2+v3", "new"):
        eval_v2(cluster=cluster)
        eval_v3(cluster=cluster)
    elif v_norm == "all":
        eval_v1(cluster=cluster)
        eval_v2(cluster=cluster)
        eval_v3(cluster=cluster)
        eval_v3_matched_90pct(cluster=cluster)
        eval_v4_matched_90pct(cluster=cluster)
    elif v_norm in ("analysis", "dist", "distribution"):
        run_distribution_analysis(cluster=cluster)
        return
    else:
        raise ValueError(f"Unknown version: {version}. Choose from 'v1', 'v2', 'v3', 'v3_matched_90', 'v3_matched_90_direct_seg', 'v4', 'v2_v3', 'all', 'analysis'.")

    if run_analysis:
        run_distribution_analysis(cluster=cluster)

