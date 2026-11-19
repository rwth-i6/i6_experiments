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
from typing import List, Optional, Tuple
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
    get_v5_3_gan_audio_embeddings_path,
    get_v5_4_cheat_seg_tokens_path,
    get_v5_4_gan_audio_embeddings_path,
    get_v5_w2v2_cheat_seg_tokens_path,
    get_phoneme_text_corpus_path,
    V2_CLUSTER_CONFIGS,
    V3_CODE_TYPES,
    V3_CLUSTER_COUNTS,
    V3_WINDOWS,
    set_cluster,
    get_cluster,
)
from ....data.librispeech import text
from ....eval_cross_modal_cheating_seg import (
    CheatingSegmentationNearestNeighborEvalJob,
    PhonemeNgramRescoreJob,
    JoinCheatingSegScoresJob,
)
from ....analysis.distribution_analysis import (
    TextNgramDistributionAnalysisJob,
    SpeechClusterDistributionAnalysisJob,
    TopClusterNgramsAnalysisJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.models.analysis import (
    EmbeddingAlignmentAnalysisJob,
    Word2VecIsotropyAndPcaAnalysisJob,
)
from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.sis_recipe.train_exp import (
    build_self_learning_asr_alignment_pipeline,
    build_triphone_asymmetric_sweep_pipeline,
    DEFAULT_TRIPHONE_ASYMMETRIC_SWEEP_PAIRS,
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

                # 4-Gram Language Model Rescoring Job
                text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)


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

                # 4-Gram Language Model Rescoring Job
                text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5_w2v2_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)




# =============================================================================
# v5.3 GAN Evaluation Pipeline (HuBERT L6 + 512-dim PCA + K-Means + GAN Alignment + CSLS)
# =============================================================================

def eval_v5_3_gan_pca512(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
):
    """
    Evaluates v5.3 GAN alignment setup (Chung et al., 2018 + CSLS Procrustes refinement):
    Continuous 768-dim HuBERT Layer 6 frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, quantized by K-Means,
    and aligned via Word2Vec (dim=128, win=2) + GAN domain-adversarial training & CSLS refinement.

    Target Configurations (best config dim=128, win=2 only):
      1. Diphone clus_483_483 (483 audio clusters -> 483 text diphones)
      2. Triphone clus_6161_6161 (6161 audio clusters -> 6161 text triphones)
      3. Diphone clus_500_483 (500 audio clusters -> 483 text diphones)
      4. Triphone clus_500_6161 (500 audio clusters -> 6161 text triphones)

    Evaluates:
      - CheatingSegmentationNearestNeighborEvalJob (CSLS k=10, center_text_embeddings=True)
      - PhonemeNgramRescoreJob (4-gram phoneme LM rescoring)
      - SpeechClusterDistributionAnalysisJob & TextNgramDistributionAnalysisJob for v5.3 clusters
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.3_gan_pca{pca_dim}"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    target_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
        ("diphone", 500, "clus_500_483"),
        ("triphone", 500, "clus_500_6161"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_k = {}

    for code_type, k_clusters, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )

        if res_entry is not None:
            a_clus = res_entry["cluster_job"]
            t_w2v = res_entry["text_w2v_job"]
            align_gan = res_entry["align_gan_job"]
            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus
        else:
            cheat_seg_tokens_file = get_v5_cheat_seg_tokens_path(clus_str, cluster=active_cluster, pca_dim=pca_dim)
            if cheat_seg_tokens_file is None:
                continue

            audio_emb_raw = get_v5_3_gan_audio_embeddings_path(
                code_type, clus_str, win=win, dim=dim, retrieval_metric=metric, pca_dim=pca_dim, cluster=active_cluster
            )
            text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim, cluster=active_cluster)
            if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                continue
            if not text_emb_raw or not os.path.exists(text_emb_raw):
                continue

            audio_emb_path = tk.Path(audio_emb_raw)
            text_emb_path = tk.Path(text_emb_raw)
            vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim, cluster=active_cluster)

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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.3_gan_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 4-Gram Language Model Rescoring Job
        text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.3_gan_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.3, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # Cross-Modal Embedding Alignment & Representation Analysis (PCA, Frequency Gradient, Unaligned & Aligned Spaces)
        analysis_gan = res_entry.get("analysis_gan_job") if res_entry is not None else None
        if analysis_gan is None and res_entry is not None:
            a_w2v = res_entry.get("audio_w2v_job")
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy if a_w2v is not None else None,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.3 Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label="Audio Clusters",
                trg_label="Text Phonemes",
                _version=4,
            )

        if analysis_gan is not None:
            analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        # 1. Text N-gram Frequency Analysis (Diphones & Triphones)
        diphone_counts = get_v3_counts_path("diphone", cluster=active_cluster)
        triphone_counts = get_v3_counts_path("triphone", cluster=active_cluster)
        raw_text_data = text.get_text("train-other-960")[0] if diphone_counts is None else None
        text_analysis_job = TextNgramDistributionAnalysisJob(
            text_file=raw_text_data,
            bigram_counts_file=diphone_counts,
            trigram_counts_file=triphone_counts,
            top_k=50,
            ignore_silence=True,
            title_prefix="LibriSpeech Phoneme N-grams (Diphones & Triphones)",
            name_prefix="ngram_analysis_phonemes_v5.3",
        )
        text_analysis_job.add_alias(f"{analysis_prefix}/text_ngrams_train_other_960")
        tk.register_output(f"{analysis_prefix}/text/bigram_histogram.png", text_analysis_job.out_bigram_plot)
        tk.register_output(f"{analysis_prefix}/text/trigram_histogram.png", text_analysis_job.out_trigram_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_combined_histogram.png", text_analysis_job.out_combined_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_cumulative_coverage.png", text_analysis_job.out_coverage_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_counts.txt", text_analysis_job.out_counts_txt)
        tk.register_output(f"{analysis_prefix}/text/ngram_distribution.json", text_analysis_job.out_summary_json)
        tk.register_output(f"{analysis_prefix}/text/ngram_report.txt", text_analysis_job.out_report_txt)

        # 2. Speech Cluster Distribution Analysis for v5.3 PCA-512 clusters (K in {483, 500, 6161})
        v5_3_cluster_configs = [
            ("clus_483_483", 483, f"v5.3 PCA-{pca_dim} 483 Speech Clusters"),
            ("clus_500_500", 500, f"v5.3 PCA-{pca_dim} 500 Speech Clusters"),
            ("clus_6161_6161", 6161, f"v5.3 PCA-{pca_dim} 6161 Speech Clusters"),
        ]
        for c_str, k_clus, label in v5_3_cluster_configs:
            if k_clus in cluster_jobs_by_k:
                tokens_file = cluster_jobs_by_k[k_clus].out_collapsed_tokens_txt
                counts_path = cluster_jobs_by_k[k_clus].out_cluster_counts_txt
            else:
                tokens_file = get_v5_cheat_seg_tokens_path(c_str, cluster=active_cluster, pca_dim=pca_dim)
                counts_path = get_cluster_counts_path(c_str, cluster=active_cluster)
            if tokens_file is None and counts_path is None:
                continue
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                cluster_counts_file=counts_path,
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"{label} ({c_str})",
                name_prefix=f"speech_cluster_dist_v5.3_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# v5.4 GAN Evaluation Pipeline (HuBERT L6 + 512-dim PCA + Unit Sphere Norm + K-Means + GAN + CSLS)
# =============================================================================

def eval_v5_4_gan_pca512(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
):
    """
    Evaluates v5.4 GAN alignment setup (Chung et al., 2018 + CSLS Procrustes refinement):
    Continuous 768-dim HuBERT Layer 6 frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, normalized (unit sphere or z-score),
    quantized by K-Means, and aligned via Word2Vec (dim=128, win=2) + GAN domain-adversarial training & CSLS refinement.

    Target Configurations (best config dim=128, win=2 only):
      1. Diphone clus_483_483 (483 audio clusters -> 483 text diphones)
      2. Triphone clus_6161_6161 (6161 audio clusters -> 6161 text triphones)
      3. Diphone clus_500_483 (500 audio clusters -> 483 text diphones)
      4. Triphone clus_500_6161 (500 audio clusters -> 6161 text triphones)

    Evaluates:
      - CheatingSegmentationNearestNeighborEvalJob (CSLS k=10, center_text_embeddings=True)
      - PhonemeNgramRescoreJob (4-gram phoneme LM rescoring)
      - SpeechClusterDistributionAnalysisJob & TextNgramDistributionAnalysisJob for v5.4 clusters
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else ("Z-Score" if "z" in norm_clean else norm_tag)
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.4_{norm_clean}_gan_pca{pca_dim}"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    target_setups = [
        ("diphone", 483, "clus_483_483"),
        ("triphone", 6161, "clus_6161_6161"),
        ("diphone", 500, "clus_500_483"),
        ("triphone", 500, "clus_500_6161"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_k = {}

    for code_type, k_clusters, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )

        if res_entry is not None:
            a_clus = res_entry["cluster_job"]
            t_w2v = res_entry["text_w2v_job"]
            align_gan = res_entry["align_gan_job"]
            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus
        else:
            cheat_seg_tokens_file = get_v5_4_cheat_seg_tokens_path(clus_str, cluster=active_cluster, pca_dim=pca_dim, norm_tag=norm_clean)
            if cheat_seg_tokens_file is None:
                continue

            audio_emb_raw = get_v5_4_gan_audio_embeddings_path(
                code_type, clus_str, win=win, dim=dim, retrieval_metric=metric, pca_dim=pca_dim, cluster=active_cluster, norm_tag=norm_clean
            )
            text_emb_raw = get_v3_text_embeddings_path(code_type, win=win, dim=dim, cluster=active_cluster)
            if not audio_emb_raw or not os.path.exists(audio_emb_raw):
                continue
            if not text_emb_raw or not os.path.exists(text_emb_raw):
                continue

            audio_emb_path = tk.Path(audio_emb_raw)
            text_emb_path = tk.Path(text_emb_raw)
            vocab_path = get_v3_vocab_path(code_type, win=win, dim=dim, cluster=active_cluster)

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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.4_{norm_clean}_gan_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 4-Gram Language Model Rescoring Job
        text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.4_{norm_clean}_gan_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.4_{norm_clean}, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # Cross-Modal Embedding Alignment & Representation Analysis (PCA, Frequency Gradient, Unaligned & Aligned Spaces)
        analysis_gan = res_entry.get("analysis_gan_job") if res_entry is not None else None
        if analysis_gan is None and res_entry is not None:
            a_w2v = res_entry.get("audio_w2v_job")
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy if a_w2v is not None else None,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.4 Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label="Audio Clusters",
                trg_label="Text Phonemes",
                _version=4,
            )

        if analysis_gan is not None:
            analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        # 1. Text N-gram Frequency Analysis (Diphones & Triphones)
        diphone_counts = get_v3_counts_path("diphone", cluster=active_cluster)
        triphone_counts = get_v3_counts_path("triphone", cluster=active_cluster)
        raw_text_data = text.get_text("train-other-960")[0] if diphone_counts is None else None
        text_analysis_job = TextNgramDistributionAnalysisJob(
            text_file=raw_text_data,
            bigram_counts_file=diphone_counts,
            trigram_counts_file=triphone_counts,
            top_k=50,
            ignore_silence=True,
            title_prefix=f"LibriSpeech Phoneme N-grams ({norm_title})",
            name_prefix=f"ngram_analysis_phonemes_v5.4_{norm_clean}",
        )
        text_analysis_job.add_alias(f"{analysis_prefix}/text_ngrams_train_other_960")
        tk.register_output(f"{analysis_prefix}/text/bigram_histogram.png", text_analysis_job.out_bigram_plot)
        tk.register_output(f"{analysis_prefix}/text/trigram_histogram.png", text_analysis_job.out_trigram_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_combined_histogram.png", text_analysis_job.out_combined_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_cumulative_coverage.png", text_analysis_job.out_coverage_plot)
        tk.register_output(f"{analysis_prefix}/text/ngram_counts.txt", text_analysis_job.out_counts_txt)
        tk.register_output(f"{analysis_prefix}/text/ngram_distribution.json", text_analysis_job.out_summary_json)
        tk.register_output(f"{analysis_prefix}/text/ngram_report.txt", text_analysis_job.out_report_txt)

        # 2. Speech Cluster Distribution Analysis for v5.4 PCA-512 clusters (K in {483, 500, 6161})
        v5_4_cluster_configs = [
            ("clus_483_483", 483, f"v5.4 PCA-{pca_dim} 483 Speech Clusters ({norm_title})"),
            ("clus_500_500", 500, f"v5.4 PCA-{pca_dim} 500 Speech Clusters ({norm_title})"),
            ("clus_6161_6161", 6161, f"v5.4 PCA-{pca_dim} 6161 Speech Clusters ({norm_title})"),
        ]
        for c_str, k_clus, label in v5_4_cluster_configs:
            if k_clus in cluster_jobs_by_k:
                tokens_file = cluster_jobs_by_k[k_clus].out_collapsed_tokens_txt
                counts_path = cluster_jobs_by_k[k_clus].out_cluster_counts_txt
            else:
                tokens_file = get_v5_4_cheat_seg_tokens_path(c_str, cluster=active_cluster, pca_dim=pca_dim, norm_tag=norm_clean)
                counts_path = get_cluster_counts_path(c_str, cluster=active_cluster)
            if tokens_file is None and counts_path is None:
                continue
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                cluster_counts_file=counts_path,
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"{label} ({c_str})",
                name_prefix=f"speech_cluster_dist_v5.4_{norm_clean}_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)



# =============================================================================
# v5.5 GAN Evaluation Pipeline (HuBERT L6 + 512-dim PCA + Unit Sphere Norm + K=1500 diphone, K=10k triphone + 80% coverage text)
# =============================================================================

def eval_v5_5_gan_pca512(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
):
    """
    Evaluates v5.5 GAN alignment setup (Chung et al., 2018 + CSLS Procrustes refinement):
    Continuous 768-dim HuBERT Layer 6 frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, unit-sphere normalized,
    quantized by K-Means (K=1500 for diphone, K=10,000 for triphone), and aligned via Word2Vec (dim=128, win=2)
    with the 80% most frequent text diphones / triphones via GAN domain-adversarial training & CSLS refinement.

    Target Configurations (exactly two trainings):
      1. Diphone clus_1500_1500 (1500 audio clusters -> 80% text diphones)
      2. Triphone clus_10000_10000 (10000 audio clusters -> 80% text triphones)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.5_{norm_clean}_gan_pca{pca_dim}"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    target_setups = [
        ("diphone", 1500, "clus_1500_1500"),
        ("triphone", 10000, "clus_10000_10000"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_k = {}

    for code_type, k_clusters, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )

        if res_entry is not None:
            a_clus = res_entry["cluster_job"]
            t_w2v = res_entry["text_w2v_job"]
            align_gan = res_entry["align_gan_job"]
            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus
        else:
            continue

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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.5_{norm_clean}_gan_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 4-Gram Language Model Rescoring Job
        text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.5_{norm_clean}_gan_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.5, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # Cross-Modal Embedding Alignment & Representation Analysis (PCA, Frequency Gradient, Unaligned & Aligned Spaces)
        analysis_gan = res_entry.get("analysis_gan_job") if res_entry is not None else None
        if analysis_gan is None and res_entry is not None:
            a_w2v = res_entry.get("audio_w2v_job")
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy if a_w2v is not None else None,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.5 Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label="Audio Clusters",
                trg_label="Text Phonemes",
                _version=4,
            )

        if analysis_gan is not None:
            analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        v5_5_cluster_configs = [
            ("clus_1500_1500", 1500, f"v5.5 PCA-{pca_dim} 1500 Speech Clusters ({norm_title})"),
            ("clus_10000_10000", 10000, f"v5.5 PCA-{pca_dim} 10k Speech Clusters ({norm_title})"),
        ]
        for c_str, k_clus, label in v5_5_cluster_configs:
            if k_clus in cluster_jobs_by_k:
                tokens_file = cluster_jobs_by_k[k_clus].out_collapsed_tokens_txt
                counts_path = cluster_jobs_by_k[k_clus].out_cluster_counts_txt
            else:
                continue
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                cluster_counts_file=counts_path,
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"{label} ({c_str})",
                name_prefix=f"speech_cluster_dist_v5.5_{norm_clean}_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# v5.6 GAN Alignment Evaluation Pipeline (Structured Skip-Gram + Unit Sphere)
# =============================================================================

def eval_v5_6_gan_pca512(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
):
    """
    Evaluates v5.6 Structured Skip-Gram GAN alignment setup:
    Continuous 768-dim HuBERT Layer 6 frames projected to 512-dim via frame-fitted PCA,
    averaged within oracle segmentation boundaries, unit-sphere normalized,
    quantized by K-Means (K=1500 for diphone, K=10,000 for triphone), and trained via
    Structured Skip-Gram (order-aware position-dependent Skip-Gram, dim=128, win=2)
    aligned with the 80% most frequent text diphones / triphones via GAN & CSLS refinement.

    Target Configurations (exactly two trainings):
      1. Diphone clus_1500_1500 (1500 audio clusters -> 80% text diphones)
      2. Triphone clus_10000_10000 (10000 audio clusters -> 80% text triphones)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag
    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.6_{norm_clean}_gan_pca{pca_dim}"

    if alignment_results is None:
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_layer=6,
            word2vec_dims=[128],
            word2vec_windows=[2],
            word2vec_epochs=20,
            cluster_counts=[1500, 10000],
            code_types=["diphone", "triphone"],
            version=f"v5.6_{norm_clean}",
            text_version="v5.6",
            coverage_threshold=0.8,
            segmentation_type="cheat_seg",
            pca_dim=pca_dim,
            retrieval_metrics=["csls"],
            csls_k=10,
            alignment_method="gan",
            code_cluster_map={
                "diphone": [(1500, "clus_1500_1500")],
                "triphone": [(10000, "clus_10000_10000")],
            },
            normalize_segments=norm_clean,
            embedding_model="structured_skipgram",
            word2vec_postprocessing="none",
            all_but_the_top_D=0,
            run_isotropy_analysis=True,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()

    target_setups = [
        ("diphone", 1500, "clus_1500_1500"),
        ("triphone", 10000, "clus_10000_10000"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_k = {}

    for code_type, k_clusters, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )

        if res_entry is not None:
            a_clus = res_entry["cluster_job"]
            t_w2v = res_entry["text_w2v_job"]
            align_gan = res_entry["align_gan_job"]
            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus
        else:
            continue

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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.6_{norm_clean}_gan_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 4-Gram Language Model Rescoring Job
        text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.6_{norm_clean}_gan_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.6, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # Cross-Modal Embedding Alignment & Representation Analysis (PCA, Frequency Gradient, Unaligned & Aligned Spaces)
        analysis_gan = res_entry.get("analysis_gan_job") if res_entry is not None else None
        if analysis_gan is None and res_entry is not None:
            a_w2v = res_entry.get("audio_w2v_job")
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy if a_w2v is not None else None,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.6 Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label="Audio Clusters",
                trg_label="Text Phonemes",
                _version=4,
            )

        if analysis_gan is not None:
            analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

        # Baseline Isotropy & PCA Analysis (Mu, Bhat, & Viswanath 2018) for direct comparison with v5.6_isotropic
        w2v_isotropy_job = res_entry.get("w2v_isotropy_job") if res_entry is not None else None
        if w2v_isotropy_job is None and res_entry is not None:
            a_w2v = res_entry.get("audio_w2v_job")
            if a_w2v is not None:
                w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                    src_raw_embeddings_npy=a_w2v.out_embeddings_npy,
                    trg_raw_embeddings_npy=t_w2v.out_embeddings_npy,
                    src_post_embeddings_npy=None,
                    trg_post_embeddings_npy=None,
                    src_vocab_file=a_clus.out_cluster_vocab_txt,
                    trg_vocab_file=t_w2v.out_vocab_txt,
                    src_counts_file=a_clus.out_cluster_counts_txt,
                    trg_counts_file=t_w2v.out_counts_txt,
                    d_remove=1,
                    title_prefix=f"v5.6 Baseline Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                    src_label="Audio Clusters",
                    trg_label=f"Text {code_type.capitalize()}s",
                    is_postprocessed_active=False,
                )

        if w2v_isotropy_job is not None:
            w2v_isotropy_job.add_alias(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        v5_6_cluster_configs = [
            ("clus_1500_1500", 1500, f"v5.6 PCA-{pca_dim} 1500 Speech Clusters ({norm_title})"),
            ("clus_10000_10000", 10000, f"v5.6 PCA-{pca_dim} 10k Speech Clusters ({norm_title})"),
        ]
        for c_str, k_clus, label in v5_6_cluster_configs:
            a_clus = cluster_jobs_by_k.get(k_clus)
            tokens_file = a_clus.out_collapsed_tokens_txt if a_clus is not None else None
            if tokens_file is None:
                continue

            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"{label} ({c_str})",
                name_prefix=f"speech_cluster_dist_v5.6_{norm_clean}_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# v5.6 Isotropic GAN Alignment Pipeline (All-but-the-Top, Mu et al. 2018)
# =============================================================================

def eval_v5_6_isotropic(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    d_values: Optional[List[int]] = None,
):
    """
    Evaluates v5.6_isotropic:
      - Stage A (HuBERT Layer 6 -> PCA-512 -> segment averaging -> Unit-Sphere normalization
        -> K-Means with K=1500 for diphone and K=10000 for triphone) is kept identical to
        v5.6_unit_sphere_gan_pca512, reusing the exact same cluster sequences and raw
        Structured Skip-Gram (dim=128, win=2, 80% coverage text diphones/triphones) jobs.
      - Stage B applies Mu, Bhat, & Viswanath (2018, arXiv:1702.01417) "All-but-the-Top"
        postprocessing with D in {1, 2, 3} removed top principal components to both the
        audio cluster Structured Skip-Gram embeddings and text diphone/triphone Structured
        Skip-Gram embeddings before GAN alignment and CSLS nearest-neighbor evaluation.
      - Runs all evaluations and analyses:
          1. GAN & CSLS cross-modal alignment (GANEmbeddingAlignmentJob)
          2. Cheating segmentation nearest-neighbor evaluation (CheatingSegmentationNearestNeighborEvalJob)
          3. 4-gram phoneme LM rescoring (PhonemeNgramRescoreJob)
          4. Cross-modal 2D PCA & alignment representation analysis (EmbeddingAlignmentAnalysisJob)
          5. Before vs. After All-but-the-Top PCA & Mu et al. (2018) isotropy analysis (Word2VecIsotropyAndPcaAnalysisJob)
          6. Speech cluster distribution analysis (SpeechClusterDistributionAnalysisJob)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag
    if d_values is None:
        d_values = [1, 2, 3]

    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.6_isotropic"

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", 1500, "clus_1500_1500"),
        ("triphone", 10000, "clus_10000_10000"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_k = {}

    for d_rem in d_values:
        d_rem_int = int(d_rem)
        d_tag = f"D{d_rem_int}"
        d_prefix = f"{prefix_name}/{d_tag}"

        if alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
            alignment_results = alignment_results_by_d[d_rem_int]
        else:
            alignment_results = build_self_learning_asr_alignment_pipeline(
                librispeech_key="train-clean-100",
                audio_layer=6,
                word2vec_dims=[dim],
                word2vec_windows=[win],
                word2vec_epochs=20,
                cluster_counts=[1500, 10000],
                code_types=["diphone", "triphone"],
                version=f"v5.6_isotropic_{d_tag}",
                text_version="v5.6",
                coverage_threshold=0.8,
                segmentation_type="cheat_seg",
                pca_dim=pca_dim,
                retrieval_metrics=[metric],
                csls_k=10,
                alignment_method="gan",
                code_cluster_map={
                    "diphone": [(1500, "clus_1500_1500")],
                    "triphone": [(10000, "clus_10000_10000")],
                },
                normalize_segments=norm_clean,
                embedding_model="structured_skipgram",
                word2vec_postprocessing="all_but_the_top",
                all_but_the_top_D=d_rem_int,
                run_isotropy_analysis=True,
            )

        for code_type, k_clusters, clus_str in target_setups:
            exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
            res_entry = (
                alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                if alignment_results is not None
                else None
            )
            if res_entry is None:
                continue

            a_clus = res_entry["cluster_job"]
            a_w2v = res_entry["audio_w2v_job"]
            t_w2v = res_entry["text_w2v_job"]
            raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
            raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
            align_gan = res_entry["align_gan_job"]

            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus

            # Register postprocessed embeddings & GAN alignment outputs under v5.6_isotropic/D{D}/...
            tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
            tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
            if hasattr(a_w2v, "out_isotropy_json"):
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
            if hasattr(t_w2v, "out_isotropy_json"):
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
            tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

            # 1. Cheating Segmentation Nearest-Neighbor Evaluation
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
                name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.6_isotropic_{d_tag}",
            )
            eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

            # 2. 4-Gram Language Model Rescoring Job
            rescore_train_job = PhonemeNgramRescoreJob(
                hypotheses_file=eval_train_job.out_hypotheses_txt,
                reference_file=eval_train_job.out_reference_txt,
                text_corpus_file=text_corpus_path,
                lm_order=4,
                backoff_alpha=0.4,
                lm_weight=0.5,
                match_bonus=2.0,
                beam_width=10,
                collapse_before=True,
                name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.6_isotropic_{d_tag}",
            )
            rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

            # Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
            join_asr_scores = JoinCheatingSegScoresJob(
                scores={
                    "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                    "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                },
                title=f"Cheating Segmentation Evaluation (v5.6_isotropic {d_tag}, {code_type}, {clus_str})",
            )
            join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
            tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

            # 3. Cross-Modal Embedding Alignment & Representation Analysis (PCA, Frequency Gradient, Unaligned & Aligned Spaces)
            analysis_gan = res_entry.get("analysis_gan_job")
            if analysis_gan is None:
                analysis_gan = EmbeddingAlignmentAnalysisJob(
                    mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                    text_embeddings_npy=t_w2v.out_embeddings_npy,
                    dictionary_json=align_gan.out_dictionary_json,
                    raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                    raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                    audio_vocab_file=a_clus.out_cluster_vocab_txt,
                    text_vocab_file=t_w2v.out_vocab_txt,
                    audio_counts_file=a_clus.out_cluster_counts_txt,
                    text_counts_file=t_w2v.out_counts_txt,
                    alignment_report_json=align_gan.out_alignment_report,
                    title_prefix=f"v5.6_isotropic ({d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                    src_label="Audio Clusters",
                    trg_label="Text Phonemes",
                    _version=4,
                )

            analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

            # 4. Word2Vec / Structured Skip-Gram Isotropy & PCA Before vs. After All-but-the-Top Analysis (Mu et al. 2018)
            w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
            if w2v_isotropy_job is None:
                w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                    src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                    trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                    src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                    trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                    src_vocab_file=a_clus.out_cluster_vocab_txt,
                    trg_vocab_file=t_w2v.out_vocab_txt,
                    src_counts_file=a_clus.out_cluster_counts_txt,
                    trg_counts_file=t_w2v.out_counts_txt,
                    d_remove=d_rem_int,
                    title_prefix=f"v5.6_isotropic ({d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                    src_label="Audio Clusters",
                    trg_label=f"Text {code_type.capitalize()}s",
                    is_postprocessed_active=True,
                )

            w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        v5_6_cluster_configs = [
            ("clus_1500_1500", 1500, f"v5.6 PCA-{pca_dim} 1500 Speech Clusters ({norm_title})"),
            ("clus_10000_10000", 10000, f"v5.6 PCA-{pca_dim} 10k Speech Clusters ({norm_title})"),
        ]
        for c_str, k_clus, label in v5_6_cluster_configs:
            a_clus = cluster_jobs_by_k.get(k_clus)
            tokens_file = a_clus.out_collapsed_tokens_txt if a_clus is not None else None
            if tokens_file is None:
                continue

            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"{label} ({c_str})",
                name_prefix=f"speech_cluster_dist_v5.6_{norm_clean}_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# v5.6 Triphone Asymmetric Parameter Sweep Evaluation Pipeline
# Pure Cosine Nearest-Neighbor Retrieval (No CSLS)
# =============================================================================

def eval_v5_6_triphone_asymmetric_sweep(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    pairs: Optional[List[Tuple[int, int]]] = None,
):
    """
    Evaluates the 9 asymmetric triphone sweep ablations across (K, N) pairs:
      - Uses pure Cosine nearest-neighbor retrieval (No CSLS)
      - Computes PER with oracle cheating segmentation on train-clean-100 (3k sequences)
      - 4-gram phoneme LM rescoring
      - Cross-modal 2D PCA representation analysis
      - Word2Vec/SSG isotropy analysis
      - Speech cluster distribution analysis for unique K
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    if pairs is None:
        pairs = DEFAULT_TRIPHONE_ASYMMETRIC_SWEEP_PAIRS

    prefix_name = f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/v5.6_triphone_sweep"

    if alignment_results is None:
        alignment_results = build_triphone_asymmetric_sweep_pipeline(
            librispeech_key="train-clean-100",
            audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
            audio_layer=6,
            pca_dim=pca_dim,
            word2vec_dim=128,
            word2vec_win=2,
            word2vec_epochs=20,
            pairs=pairs,
            normalize_segments=norm_clean,
            run_isotropy_analysis=True,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    dim = 128
    win = 2
    dw_tag = f"dim{dim}_win{win}"
    cluster_jobs_by_k = {}

    for K, N in pairs:
        pair_tag = f"K{K}_N{N}"
        exp_key = f"{pair_tag}/{dw_tag}_cosine"

        res_entry = alignment_results.get((K, N))
        if res_entry is None:
            continue

        a_clus = res_entry["cluster_job"]
        a_ssg = res_entry["audio_ssg_job"]
        t_ngram = res_entry["ngram_job"]
        t_ssg = res_entry["text_ssg_job"]
        align_gan = res_entry["align_gan_job"]
        cluster_jobs_by_k[K] = a_clus

        cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
        audio_emb_path = align_gan.out_mapped_audio_emb_npy
        text_emb_path = t_ssg.out_embeddings_npy
        vocab_path = t_ngram.out_vocab_txt

        # Register alignment outputs under output/...
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_embeddings.npy", a_ssg.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/text_embeddings.npy", t_ssg.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

        # 1. Cheating Segmentation Nearest-Neighbor Evaluation: Pure Cosine (NO CSLS)
        eval_train_job = CheatingSegmentationNearestNeighborEvalJob(
            audio_embeddings_npy=audio_emb_path,
            text_embeddings_npy=text_emb_path,
            phoneme_vocab_file=vocab_path,
            cheating_segmentation_path=cheat_seg_tokens_file,
            audio_cluster_hdfs=None,
            reference_phonemes=oracle_ref_hdfs,
            reference_phoneme_vocab_file=oracle_idx_to_phoneme,
            seq_tags_file=oracle_seq_tags,
            code_type="triphone",
            max_seqs=3000,
            ignore_silence=True,
            retrieval_metric="cosine",  # Cosine only, NO CSLS
            center_text_embeddings=True,
            name_prefix=f"eval_train3k_triphone_{pair_tag}_{dw_tag}_cosine_v5.6",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 2. 4-Gram Language Model Rescoring Job
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_triphone_{pair_tag}_{dw_tag}_cosine_v5.6",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # 3. Joined Score Results (Nearest-Neighbor PER + 4-gram LM Rescoring PER)
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_cosine_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation ({pair_tag}, Cosine)",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # 4. Cross-Modal 2D PCA Representation Analysis
        analysis_gan = res_entry.get("analysis_gan_job")
        if analysis_gan is not None:
            analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

        # 4. Isotropy Analysis
        w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
        if w2v_isotropy_job is not None:
            w2v_isotropy_job.add_alias(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis")
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
            tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        for k_clus, a_clus in sorted(cluster_jobs_by_k.items()):
            tokens_file = a_clus.out_collapsed_tokens_txt if a_clus is not None else None
            if tokens_file is None:
                continue

            c_str = f"clus_{k_clus}_{k_clus}"
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                raw_clusters_file=tokens_file,
                cluster_hdfs=[],
                k_clusters=k_clus,
                top_k=50,
                title_prefix=f"v5.6 PCA-{pca_dim} {k_clus} Speech Clusters ({norm_title})",
                name_prefix=f"speech_cluster_dist_v5.6_triphone_sweep_pca{pca_dim}_{c_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# v5.7 Spin-HuBERT GAN Alignment Evaluation Pipeline
# (Chang et al., Interspeech 2023, arXiv:2305.11072, Codebooks {256, 512, 1024})
# =============================================================================

def eval_v5_7_spin_gan_pca512(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_spin: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    spin_codebook_sizes: Optional[List[int]] = None,
    audio_layer: int = 12,
):
    """
    Evaluates v5.7 Spin-HuBERT setup (based on v5.5_unit_sphere_gan_pca512):
      - Swaps the vanilla HuBERT-base encoder with Spin fine-tuned HuBERT encoders
        (Chang et al., 2023, arXiv:2305.11072) trained with speaker-invariant clustering
        at codebook sizes {256, 512, 1024} (extracting Layer 12 representations).
      - Frame-level PCA projection (768 -> 512) -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means (K=1500 for diphone, K=10000 for triphone).
      - Word2Vec (dim=128, win=2) + GAN domain-adversarial alignment with CSLS Procrustes refinement
        against 80% coverage text diphones and triphones.
      - Runs:
          1. Cheating segmentation nearest-neighbor CSLS PER evaluation (CheatingSegmentationNearestNeighborEvalJob)
          2. 4-gram phoneme LM rescoring (PhonemeNgramRescoreJob)
          3. Joined score summary (JoinCheatingSegScoresJob)
          4. Cross-modal 2D PCA & alignment representation analysis (EmbeddingAlignmentAnalysisJob)
          5. Embedding isotropy & PCA eigenvalue spectrum analysis (Word2VecIsotropyAndPcaAnalysisJob)
          6. Speech cluster distribution analysis (SpeechClusterDistributionAnalysisJob)
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag
    if spin_codebook_sizes is None:
        spin_codebook_sizes = [1024, 2048]

    prefix_name = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.7_spin_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", 1500, "clus_1500_1500"),
        ("triphone", 10000, "clus_10000_10000"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"
        spin_prefix = f"{prefix_name}/{spin_tag}"

        if alignment_results_by_spin is not None and spin_k_int in alignment_results_by_spin:
            alignment_results = alignment_results_by_spin[spin_k_int]
        else:
            alignment_results = build_self_learning_asr_alignment_pipeline(
                librispeech_key="train-clean-100",
                audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                audio_layer=audio_layer,
                word2vec_dims=[dim],
                word2vec_windows=[win],
                word2vec_epochs=20,
                cluster_counts=[1500, 10000],
                coverage_threshold=0.80,
                code_types=["diphone", "triphone"],
                code_cluster_map={
                    "diphone": [(1500, "clus_1500_1500")],
                    "triphone": [(10000, "clus_10000_10000")],
                },
                segmentation_type="cheating",
                pca_dim=pca_dim,
                version=f"v5.7_{spin_tag}_{norm_clean}",
                text_version="v5.5",
                text_model_tag="hubert_l6",
                normalize_segments=norm_clean,
                retrieval_metrics=[metric],
                csls_k=10,
                alignment_method="gan",
                embedding_model="word2vec",
                run_isotropy_analysis=True,
                spin_codebook_size=spin_k_int,
            )

        cluster_jobs_by_k = {}

        for code_type, k_clusters, clus_str in target_setups:
            exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
            res_entry = (
                alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                if alignment_results is not None
                else None
            )
            if res_entry is None:
                continue

            a_clus = res_entry["cluster_job"]
            a_w2v = res_entry["audio_w2v_job"]
            t_w2v = res_entry["text_w2v_job"]
            align_gan = res_entry["align_gan_job"]

            cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
            audio_emb_path = align_gan.out_mapped_audio_emb_npy
            text_emb_path = t_w2v.out_embeddings_npy
            vocab_path = t_w2v.out_vocab_txt
            cluster_jobs_by_k[k_clusters] = a_clus

            # Register embeddings & GAN alignment outputs under v5.7_spin_unit_sphere_gan_pca512/spin_{K}/...
            tk.register_output(f"{spin_prefix}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
            tk.register_output(f"{spin_prefix}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
            tk.register_output(f"{spin_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

            # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.7_{spin_tag}_{norm_clean}_gan_pca{pca_dim}",
            )
            eval_train_job.add_alias(f"{spin_prefix}/{exp_key}/eval_train3k")
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

            # 2. 4-Gram Language Model Rescoring Job
            rescore_train_job = PhonemeNgramRescoreJob(
                hypotheses_file=eval_train_job.out_hypotheses_txt,
                reference_file=eval_train_job.out_reference_txt,
                text_corpus_file=text_corpus_path,
                lm_order=4,
                backoff_alpha=0.4,
                lm_weight=0.5,
                match_bonus=2.0,
                beam_width=10,
                collapse_before=True,
                name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.7_{spin_tag}_{norm_clean}_gan_pca{pca_dim}",
            )
            rescore_train_job.add_alias(f"{spin_prefix}/{exp_key}/rescore_4gram_train3k")
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

            # 3. Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
            join_asr_scores = JoinCheatingSegScoresJob(
                scores={
                    "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                    "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                },
                title=f"Cheating Segmentation Evaluation (v5.7 {spin_tag} L{audio_layer}, {code_type}, {clus_str})",
            )
            join_asr_scores.add_alias(f"{spin_prefix}/{exp_key}/score_results")
            tk.register_output(f"{spin_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

            # 4. Cross-Modal 2D PCA & Representation Analysis
            analysis_gan = res_entry.get("analysis_gan_job")
            if analysis_gan is None:
                analysis_gan = EmbeddingAlignmentAnalysisJob(
                    mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                    text_embeddings_npy=t_w2v.out_embeddings_npy,
                    dictionary_json=align_gan.out_dictionary_json,
                    raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                    raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                    audio_vocab_file=a_clus.out_cluster_vocab_txt,
                    text_vocab_file=t_w2v.out_vocab_txt,
                    audio_counts_file=a_clus.out_cluster_counts_txt,
                    text_counts_file=t_w2v.out_counts_txt,
                    alignment_report_json=align_gan.out_alignment_report,
                    title_prefix=f"v5.7 ({spin_tag} L{audio_layer}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                    src_label=f"Audio Clusters ({spin_tag})",
                    trg_label="Text Phonemes",
                    _version=4,
                )

            analysis_gan.add_alias(f"{spin_prefix}/{exp_key}/pca_analysis")
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

            # 5. Embedding Isotropy & PCA Spectrum Analysis
            w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
            if w2v_isotropy_job is None:
                w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                    src_raw_embeddings_npy=a_w2v.out_embeddings_npy,
                    trg_raw_embeddings_npy=t_w2v.out_embeddings_npy,
                    src_post_embeddings_npy=None,
                    trg_post_embeddings_npy=None,
                    src_vocab_file=a_clus.out_cluster_vocab_txt,
                    trg_vocab_file=t_w2v.out_vocab_txt,
                    src_counts_file=a_clus.out_cluster_counts_txt,
                    trg_counts_file=t_w2v.out_counts_txt,
                    d_remove=1,
                    title_prefix=f"v5.7 ({spin_tag} L{audio_layer}) Embedding Isotropy & PCA ({code_type}, {clus_str})",
                    src_label=f"Audio Clusters ({spin_tag})",
                    trg_label=f"Text {code_type.capitalize()}s",
                    is_postprocessed_active=False,
                )

            w2v_isotropy_job.add_alias(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis")
            tk.register_output(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
            tk.register_output(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
            tk.register_output(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
            tk.register_output(f"{spin_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

        if run_analysis:
            analysis_prefix = f"{spin_prefix}/analysis"
            v5_7_cluster_configs = [
                ("clus_1500_1500", 1500, f"v5.7 {spin_tag} L{audio_layer} PCA-{pca_dim} 1500 Speech Clusters ({norm_title})"),
                ("clus_10000_10000", 10000, f"v5.7 {spin_tag} L{audio_layer} PCA-{pca_dim} 10k Speech Clusters ({norm_title})"),
            ]
            for c_str, k_clus, label in v5_7_cluster_configs:
                a_clus = cluster_jobs_by_k.get(k_clus)
                if a_clus is None:
                    continue
                cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                    cluster_counts_file=a_clus.out_cluster_counts_txt,
                    raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                    cluster_hdfs=[],
                    k_clusters=k_clus,
                    top_k=50,
                    title_prefix=f"{label} ({c_str})",
                    name_prefix=f"speech_cluster_dist_v5.7_{spin_tag}_{norm_clean}_pca{pca_dim}_{c_str}",
                )
                cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
                tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_8_spin_isotropic(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.8 Spin-HuBERT Isotropic setup:
      - Combines Spin-HuBERT (Chang et al., 2023, arXiv:2305.11072) with codebook size 2048
        (Layer 12 representations with intrinsic isotropy / isoperimetric regularization)
        with Mu, Bhat, & Viswanath (2018) "All-but-the-Top" isotropic postprocessing (D in {1, 2, 3})
        applied to Word2Vec embeddings.
      - Frame-level PCA projection (768 -> 512) -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means (K=1500 for diphone, K=10000 for triphone).
      - Audio and text Word2Vec (dim=128, win=2) postprocessed with All-but-the-Top (removing top D components).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Cheating segmentation nearest-neighbor CSLS PER evaluation + 4-gram phoneme LM rescoring.
      - Cross-modal 2D PCA representation analysis, Before vs. After All-but-the-Top isotropy analysis,
        and speech cluster distribution analysis.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.8_spin_isotropic_{norm_clean}_gan_pca{pca_dim}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [1, 2, 3]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.8_spin_isotropic_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", 1500, "clus_1500_1500"),
        ("triphone", 10000, "clus_10000_10000"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[1500, 10000],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(1500, "clus_1500_1500")],
                        "triphone": [(10000, "clus_10000_10000")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.8_{spin_tag}_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.5" if embedding_model == "word2vec" else "v5.6",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="gan",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                )

            cluster_jobs_by_k = {}

            for code_type, k_clusters, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gan = res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gan.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_k[k_clusters] = a_clus

                # Register postprocessed embeddings & GAN alignment outputs under v5.8_spin_isotropic/...
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.8_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.8_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results (Nearest-Neighbor CSLS PER + 4-gram LM Rescoring PER)
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.8 Spin-{spin_k_int} L{audio_layer} {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gan = res_entry.get("analysis_gan_job")
                if analysis_gan is None:
                    analysis_gan = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gan.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gan.out_alignment_report,
                        title_prefix=f"v5.8 (Spin-{spin_k_int} L{audio_layer} {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio Clusters (Spin-{spin_k_int})",
                        trg_label="Text Phonemes",
                        _version=4,
                    )

                analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Before vs. After All-but-the-Top Analysis (Mu et al. 2018)
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.8 (Spin-{spin_k_int} L{audio_layer} {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio Clusters (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

            if run_analysis:
                analysis_prefix = f"{d_prefix}/analysis"
                cluster_configs = [
                    ("clus_1500_1500", 1500, f"v5.8 Spin-{spin_k_int} L{audio_layer} PCA-{pca_dim} {d_tag} 1500 Speech Clusters ({norm_title})"),
                    ("clus_10000_10000", 10000, f"v5.8 Spin-{spin_k_int} L{audio_layer} PCA-{pca_dim} {d_tag} 10k Speech Clusters ({norm_title})"),
                ]
                for c_str, k_clus, label in cluster_configs:
                    a_clus = cluster_jobs_by_k.get(k_clus)
                    if a_clus is None:
                        continue
                    cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                        cluster_counts_file=a_clus.out_cluster_counts_txt,
                        raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                        cluster_hdfs=[],
                        k_clusters=k_clus,
                        top_k=50,
                        title_prefix=f"{label} ({c_str})",
                        name_prefix=f"speech_cluster_dist_v5.8_spin_{spin_k_int}_{norm_clean}_pca{pca_dim}_{d_tag}_{c_str}",
                    )
                    cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{c_str}")
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{c_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


eval_v5_8_spin_2048_isotropic = eval_v5_8_spin_isotropic
eval_v5_7_spin_isotropic = eval_v5_8_spin_isotropic


def eval_v5_9_audio_ngrams(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
):
    """
    Evaluates v5.9 Compositional Audio N-Grams (1A):
      - Stage 1: Vanilla HuBERT-base Layer 6 -> 512-dim PCA -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means at monophone scale (K_base = 40).
      - Stage 2: Builds sliding-window audio 2-grams (diphones) and 3-grams (triphones)
        from the base cluster sequences with 80% cumulative coverage filtering
        (GenerateAudioClusterNgramsJob), mirroring text diphone/triphone generation.
      - Stage 3: Standard Skip-Gram Word2Vec (dim=128, win=2, epochs=20; no Spin, no ABTT)
        aligned against 80% coverage text diphones and triphones via GAN + CSLS Procrustes refinement.
      - Runs nearest-neighbor CSLS PER evaluation, 4-gram LM rescoring, cross-modal 2D PCA analysis,
        and audio n-gram frequency distribution analysis.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9_audio_ngrams_{norm_clean}_gan_pca{pca_dim}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)
    prefix_name = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    if alignment_results is None:
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
            audio_layer=audio_layer,
            word2vec_dims=[128],
            word2vec_windows=[2],
            word2vec_epochs=20,
            cluster_counts=[k_base],
            coverage_threshold=0.80,
            code_types=["diphone", "triphone"],
            code_cluster_map={
                "diphone": [(k_base, f"clus_{k_base}_2gram")],
                "triphone": [(k_base, f"clus_{k_base}_3gram")],
            },
            segmentation_type="cheating",
            pca_dim=pca_dim,
            version=f"v5.9_audio_ngrams_base{k_base}_{norm_clean}",
            text_version="v5.5",
            text_model_tag="hubert_l6",
            normalize_segments=norm_clean,
            retrieval_metrics=["csls"],
            csls_k=10,
            alignment_method="gan",
            embedding_model="word2vec",
            run_isotropy_analysis=True,
            use_audio_ngrams=True,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_name = {}
    base_cluster_job = None

    for code_type, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )
        if res_entry is None:
            continue

        a_clus = res_entry["cluster_job"]
        a_w2v = res_entry["audio_w2v_job"]
        t_w2v = res_entry["text_w2v_job"]
        align_gan = res_entry["align_gan_job"]

        cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
        audio_emb_path = align_gan.out_mapped_audio_emb_npy
        text_emb_path = t_w2v.out_embeddings_npy
        vocab_path = t_w2v.out_vocab_txt
        cluster_jobs_by_name[clus_str] = a_clus

        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

        # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9_audio_ngrams_{norm_clean}_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 2. 4-Gram Language Model Rescoring Job
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9_audio_ngrams_{norm_clean}_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # 3. Joined Score Results
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.9 Audio N-Grams, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # 4. Cross-Modal 2D PCA & Representation Analysis
        analysis_gan = res_entry.get("analysis_gan_job")
        if analysis_gan is None:
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.9 Compositional Audio N-Grams Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                trg_label=f"Text {code_type.capitalize()}s",
                _version=4,
            )

        analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

        # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
        w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
        if w2v_isotropy_job is None:
            w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                src_raw_embeddings_npy=a_w2v.out_embeddings_npy,
                trg_raw_embeddings_npy=t_w2v.out_embeddings_npy,
                src_post_embeddings_npy=None,
                trg_post_embeddings_npy=None,
                src_vocab_file=a_clus.out_cluster_vocab_txt,
                trg_vocab_file=t_w2v.out_vocab_txt,
                src_counts_file=a_clus.out_cluster_counts_txt,
                trg_counts_file=t_w2v.out_counts_txt,
                d_remove=1,
                title_prefix=f"v5.9 Audio N-Grams Embedding Isotropy & PCA ({code_type}, {clus_str})",
                src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                trg_label=f"Text {code_type.capitalize()}s",
                is_postprocessed_active=False,
            )

        w2v_isotropy_job.add_alias(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis")
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

        # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
        top_ngrams_job = TopClusterNgramsAnalysisJob(
            cluster_vocab_file=a_clus.out_cluster_vocab_txt,
            cluster_counts_file=a_clus.out_ngram_counts_txt,
            dictionary_json=align_gan.out_dictionary_json,
            mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
            text_embeddings_npy=t_w2v.out_embeddings_npy,
            text_vocab_file=vocab_path,
            top_k=100,
            title_prefix=f"v5.9.3 ({code_type}, {clus_str}, {norm_title})",
            name_prefix=f"top_cluster_ngrams_v5.9.3_{code_type}_{clus_str}",
        )
        top_ngrams_job.add_alias(f"{prefix_name}/{exp_key}/top_100_ngrams")
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        for code_type, clus_str in target_setups:
            a_clus = cluster_jobs_by_name.get(clus_str)
            if a_clus is None:
                continue
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                cluster_counts_file=a_clus.out_cluster_counts_txt,
                raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                cluster_hdfs=[],
                k_clusters=None,
                top_k=50,
                title_prefix=f"v5.9 Compositional Audio {code_type.capitalize()}s ({clus_str}, {norm_title})",
                name_prefix=f"speech_cluster_dist_v5.9_audio_ngrams_{norm_clean}_pca{pca_dim}_{clus_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_9_1_spin_audio_ngrams_isotropic(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.1 Spin-HuBERT Compositional Audio N-Grams with All-but-the-Top Isotropy:
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default codebook size 2048, arXiv:2305.11072)
        with intrinsic speaker invariance and isotropy.
      - Frame-level PCA projection (768 -> 512) -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means at monophone scale (K_base = 40).
      - Compositional Audio N-Grams: Builds sliding-window audio 2-grams (diphones) and
        3-grams (triphones) from base cluster sequences with 80% coverage pruning.
      - Embeddings: Skip-Gram Word2Vec (dim=128, win=2, epochs=20).
      - Isotropic Postprocessing: All-but-the-Top (Mu, Bhat, & Viswanath, 2018) removing
        top D principal components (default D=2, or swept D in {1, 2, 3}) from both audio
        n-gram embeddings and text n-gram embeddings.
      - Alignment: GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram phoneme LM rescoring.
      - Analyses: Cross-modal 2D PCA representation analysis, Before vs. After All-but-the-Top
        isotropy & PCA spectrum analysis, and speech cluster distribution analysis.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.1_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [2]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.1_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.1_{spin_tag}_base{k_base}_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.5",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="gan",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gan = res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gan.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                # Register postprocessed embeddings & GAN alignment outputs
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.1 Spin-{spin_k_int} L{audio_layer} Audio N-Grams {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gan = res_entry.get("analysis_gan_job")
                if analysis_gan is None:
                    analysis_gan = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gan.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gan.out_alignment_report,
                        title_prefix=f"v5.9.1 (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        _version=4,
                    )

                analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Before vs. After All-but-the-Top Analysis (Mu et al. 2018)
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.9.1 (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

            if run_analysis:
                analysis_prefix = f"{d_prefix}/analysis"
                for code_type, clus_str in target_setups:
                    a_clus = cluster_jobs_by_name.get(clus_str)
                    if a_clus is None:
                        continue
                    cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                        cluster_counts_file=a_clus.out_cluster_counts_txt,
                        raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                        cluster_hdfs=[],
                        k_clusters=None,
                        top_k=50,
                        title_prefix=f"v5.9.1 Compositional Audio {code_type.capitalize()}s ({clus_str}, Spin-{spin_k_int}, {norm_title})",
                        name_prefix=f"speech_cluster_dist_v5.9.1_spin_{spin_k_int}_audio_ngrams_{norm_clean}_pca{pca_dim}_{d_tag}_{clus_str}",
                    )
                    cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_9_2_wo_sil_spin_audio_ngrams_isotropic(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    audio_backbone: str = "spin_hubert",
):
    """
    Evaluates v5.9.2 Compositional Audio N-Grams without Silence (100% symmetric) with ABTT isotropy:
      - Text phonemization with sil_prob=0.0 and surround_w_sil=False:
        Text diphones and triphones contain NO artificial <SIL> symbols, eliminating fake
        boundary hubs and frequency distortion.
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default) or Vanilla HuBERT Layer 6 (audio_backbone='hubert_l6').
      - Base K-Means at monophone scale (K_base = 40) with Unit-Sphere normalization on 512-dim PCA.
      - Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - All-but-the-Top Word2Vec postprocessing (default D=2).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram LM rescoring.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)
    is_spin = audio_backbone == "spin_hubert"

    if not is_spin:
        actual_audio_layer = audio_layer if audio_layer is not None else 6
        codebook_loop = [None]
        audio_tag = f"hubert_l{actual_audio_layer}"
        prefix_base = (
            f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
            f"v5.9.2_wo_sil_{audio_tag}_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
        )
    else:
        actual_audio_layer = audio_layer if audio_layer is not None else 12
        if spin_codebook_sizes is None:
            codebook_loop = [spin_codebook_size] if spin_codebook_size is not None else [2048]
        else:
            codebook_loop = spin_codebook_sizes
        prefix_base = (
            f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
            f"v5.9.2_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
        )

    if d_values is None:
        d_values = [2]

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in codebook_loop:
        if spin_k is not None:
            spin_k_int = int(spin_k)
            spin_tag = f"spin_{spin_k_int}"
            backbone_title = f"Spin-{spin_k_int}"
            backbone_name = f"spin_{spin_k_int}"
            model_subpath = spin_tag
        else:
            spin_k_int = None
            spin_tag = audio_tag
            backbone_title = f"HuBERT-L{actual_audio_layer}"
            backbone_name = audio_tag
            model_subpath = None

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{model_subpath}/{d_tag}" if model_subpath else f"{prefix_base}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=actual_audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.2_{spin_tag}_base{k_base}_wo_sil_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.9.2_wo_sil",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="gan",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                    sil_prob=0.0,
                    surround_w_sil=False,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gan = res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gan.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.2_wo_sil_{backbone_name}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.2_wo_sil_{backbone_name}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.2 Without Silence {backbone_title} L{actual_audio_layer} N-Grams {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gan = res_entry.get("analysis_gan_job")
                if analysis_gan is None:
                    analysis_gan = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gan.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gan.out_alignment_report,
                        title_prefix=f"v5.9.2 (Without Silence {backbone_title} L{actual_audio_layer} N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio {code_type.capitalize()}s ({backbone_title})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        _version=4,
                    )

                analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Before vs. After All-but-the-Top Analysis (Mu et al. 2018)
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.9.2 (Without Silence {backbone_title} L{actual_audio_layer} N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio {code_type.capitalize()}s ({backbone_title})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

            if run_analysis:
                analysis_prefix = f"{d_prefix}/analysis"
                for code_type, clus_str in target_setups:
                    a_clus = cluster_jobs_by_name.get(clus_str)
                    if a_clus is None:
                        continue
                    cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                        cluster_counts_file=a_clus.out_cluster_counts_txt,
                        raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                        cluster_hdfs=[],
                        k_clusters=None,
                        top_k=50,
                        title_prefix=f"v5.9.2 Compositional Audio {code_type.capitalize()}s ({clus_str}, {backbone_title}, {norm_title})",
                        name_prefix=f"speech_cluster_dist_v5.9.2_wo_sil_{backbone_name}_audio_ngrams_{norm_clean}_pca{pca_dim}_{d_tag}_{clus_str}",
                    )
                    cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_9_2_wo_sil_hubert_l6_audio_ngrams_isotropic(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.2 Compositional Audio N-Grams without Silence (collapsed) using Vanilla HuBERT Layer 6 + ABTT.
    Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.2_wo_sil_hubert_l6_audio_ngrams_base40_unit_sphere_gan_pca512/D{D}/...
    """
    return eval_v5_9_2_wo_sil_spin_audio_ngrams_isotropic(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        audio_backbone="hubert_l6",
    )


def eval_v5_9_3_wo_sil_audio_ngrams(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
):
    """
    Evaluates v5.9.3 Compositional Audio N-Grams without Silence (1A baseline: no Spin, no ABTT):
      - Stage 1: Vanilla HuBERT-base Layer 6 -> 512-dim PCA -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means at monophone scale (K_base = 40).
      - Stage 2: Builds sliding-window audio 2-grams (diphones) and 3-grams (triphones)
        from base cluster sequences with 80% coverage filtering.
      - Stage 3: Text phonemization with sil_prob=0.0 and surround_w_sil=False:
        diphones and triphones have NO artificial <SIL> symbols on either side.
      - Stage 4: Standard Skip-Gram Word2Vec (dim=128, win=2, epochs=20; no Spin, no ABTT)
        aligned against text diphones and triphones via GAN + CSLS Procrustes refinement.
      - Runs nearest-neighbor CSLS PER evaluation, 4-gram LM rescoring, cross-modal 2D PCA analysis,
        and audio n-gram frequency distribution analysis.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.3_wo_sil_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)
    prefix_name = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.3_wo_sil_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    if alignment_results is None:
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
            audio_layer=audio_layer,
            word2vec_dims=[128],
            word2vec_windows=[2],
            word2vec_epochs=20,
            cluster_counts=[k_base],
            coverage_threshold=0.80,
            code_types=["diphone", "triphone"],
            code_cluster_map={
                "diphone": [(k_base, f"clus_{k_base}_2gram")],
                "triphone": [(k_base, f"clus_{k_base}_3gram")],
            },
            segmentation_type="cheating",
            pca_dim=pca_dim,
            version=f"v5.9.3_wo_sil_audio_ngrams_base{k_base}_{norm_clean}",
            text_version="v5.9.2_wo_sil",
            text_model_tag="hubert_l6",
            normalize_segments=norm_clean,
            retrieval_metrics=["csls"],
            csls_k=10,
            alignment_method="gan",
            embedding_model="word2vec",
            run_isotropy_analysis=True,
            use_audio_ngrams=True,
            sil_prob=0.0,
            surround_w_sil=False,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_name = {}

    for code_type, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )
        if res_entry is None:
            continue

        a_clus = res_entry["cluster_job"]
        a_w2v = res_entry["audio_w2v_job"]
        t_w2v = res_entry["text_w2v_job"]
        align_gan = res_entry["align_gan_job"]

        cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
        audio_emb_path = align_gan.out_mapped_audio_emb_npy
        text_emb_path = t_w2v.out_embeddings_npy
        vocab_path = t_w2v.out_vocab_txt
        cluster_jobs_by_name[clus_str] = a_clus

        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

        # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.3_wo_sil_audio_ngrams_{norm_clean}_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 2. 4-Gram Language Model Rescoring Job
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.3_wo_sil_audio_ngrams_{norm_clean}_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # 3. Joined Score Results
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.9.3 Without Silence Audio N-Grams, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # 4. Cross-Modal 2D PCA & Representation Analysis
        analysis_gan = res_entry.get("analysis_gan_job")
        if analysis_gan is None:
            analysis_gan = EmbeddingAlignmentAnalysisJob(
                mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                text_embeddings_npy=t_w2v.out_embeddings_npy,
                dictionary_json=align_gan.out_dictionary_json,
                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                text_vocab_file=t_w2v.out_vocab_txt,
                audio_counts_file=a_clus.out_cluster_counts_txt,
                text_counts_file=t_w2v.out_counts_txt,
                alignment_report_json=align_gan.out_alignment_report,
                title_prefix=f"v5.9.3 Compositional Audio N-Grams Without Silence Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                trg_label=f"Text {code_type.capitalize()}s",
                _version=4,
            )

        analysis_gan.add_alias(f"{prefix_name}/{exp_key}/pca_analysis")
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

        # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
        w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
        if w2v_isotropy_job is None:
            w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                src_raw_embeddings_npy=a_w2v.out_embeddings_npy,
                trg_raw_embeddings_npy=t_w2v.out_embeddings_npy,
                src_post_embeddings_npy=None,
                trg_post_embeddings_npy=None,
                src_vocab_file=a_clus.out_cluster_vocab_txt,
                trg_vocab_file=t_w2v.out_vocab_txt,
                src_counts_file=a_clus.out_cluster_counts_txt,
                trg_counts_file=t_w2v.out_counts_txt,
                d_remove=1,
                title_prefix=f"v5.9.3 Audio N-Grams Without Silence Embedding Isotropy & PCA ({code_type}, {clus_str})",
                src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                trg_label=f"Text {code_type.capitalize()}s",
                is_postprocessed_active=False,
            )

        w2v_isotropy_job.add_alias(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis")
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
        tk.register_output(f"{prefix_name}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

        # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
        top_ngrams_job = TopClusterNgramsAnalysisJob(
            cluster_vocab_file=a_clus.out_cluster_vocab_txt,
            cluster_counts_file=a_clus.out_ngram_counts_txt,
            dictionary_json=align_gan.out_dictionary_json,
            mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
            text_embeddings_npy=t_w2v.out_embeddings_npy,
            text_vocab_file=vocab_path,
            top_k=100,
            title_prefix=f"v5.9.3 ({code_type}, {clus_str}, {norm_title})",
            name_prefix=f"top_cluster_ngrams_v5.9.3_{code_type}_{clus_str}",
        )
        top_ngrams_job.add_alias(f"{prefix_name}/{exp_key}/top_100_ngrams")
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
        tk.register_output(f"{prefix_name}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

    if run_analysis:
        analysis_prefix = f"{prefix_name}/analysis"
        for code_type, clus_str in target_setups:
            a_clus = cluster_jobs_by_name.get(clus_str)
            if a_clus is None:
                continue
            cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                cluster_counts_file=a_clus.out_cluster_counts_txt,
                raw_clusters_file=a_clus.out_collapsed_tokens_txt,
                cluster_hdfs=[],
                k_clusters=None,
                top_k=50,
                title_prefix=f"v5.9.3 Compositional Audio {code_type.capitalize()}s Without Silence ({clus_str}, {norm_title})",
                name_prefix=f"speech_cluster_dist_v5.9.3_wo_sil_audio_ngrams_{norm_clean}_pca{pca_dim}_{clus_str}",
            )
            cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
            tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_backbone: str = "spin_hubert",
    audio_layer: Optional[int] = None,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    structural_constraint: Optional[str] = None,
    lambda_struct: float = 0.5,
    struct_margin: float = 0.0,
    compatibility_mode: str = "collapsed_length",
    version_tag: Optional[str] = None,
):
    """
    Evaluates Compositional Audio N-Grams without Silence and WITHOUT Collapsing:
      - Text phonemization with sil_prob=0.0 and surround_w_sil=False:
        Text diphones and triphones contain NO artificial <SIL> symbols.
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default codebook 2048) or HuBERT Layer 6.
      - Base K-Means at monophone scale (K_base = 40) or sub-phonemic allophonic scale (K_base = 120)
        with Unit-Sphere normalization on 512-dim PCA.
      - collapse_consecutive=False: Adjacent oracle segments are PRESERVED rather than collapsed.
      - Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - All-but-the-Top Word2Vec postprocessing (default D=2).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Structural Constraint: Optional 'masked_csls' (.1) or 'reg_gan_and_masked_csls' (.2).
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram LM rescoring (collapse_before=False).
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    is_spin = audio_backbone == "spin_hubert" and (spin_codebook_size is not None or spin_codebook_sizes is not None or (audio_layer is not None and audio_layer != 6))
    if not is_spin:
        actual_audio_layer = audio_layer if audio_layer is not None else 6
        codebook_loop = [None]
        audio_tag = f"hubert_l{actual_audio_layer}"
    else:
        actual_audio_layer = audio_layer if audio_layer is not None else 12
        if spin_codebook_sizes is not None:
            codebook_loop = spin_codebook_sizes
        elif spin_codebook_size is not None:
            codebook_loop = [spin_codebook_size]
        else:
            codebook_loop = [2048]

    if d_values is None:
        d_values = [2]

    v_tag = version_tag or ("v5.10" if k_base >= 120 else "v5.9.4")
    if is_spin:
        prefix_base = (
            f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
            f"{v_tag}_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
        )
    elif v_tag.startswith("v5.10"):
        prefix_base = (
            f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
            f"{v_tag}_wo_sil_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
        )
    else:
        prefix_base = (
            f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
            f"{v_tag}_wo_sil_hubert_l6_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in codebook_loop:
        if spin_k is not None:
            spin_k_int = int(spin_k)
            spin_tag = f"spin_{spin_k_int}"
            model_subpath = spin_tag
            ver_tag_model = spin_tag
        else:
            spin_k_int = None
            spin_tag = audio_tag
            model_subpath = None
            ver_tag_model = audio_tag

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            if model_subpath:
                d_prefix = f"{prefix_base}/{model_subpath}/{d_tag}"
            else:
                d_prefix = f"{prefix_base}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=actual_audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"{v_tag}_{ver_tag_model}_base{k_base}_wo_sil_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.9.2_wo_sil",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="gan",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                    sil_prob=0.0,
                    surround_w_sil=False,
                    collapse_consecutive=False,
                    structural_constraint=structural_constraint,
                    lambda_struct=lambda_struct,
                    struct_margin=struct_margin,
                    compatibility_mode=compatibility_mode,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gan = res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gan.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_{v_tag}_wo_sil_{ver_tag_model}_base{k_base}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job (collapse_before=False)
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=False,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_{v_tag}_wo_sil_{ver_tag_model}_base{k_base}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation ({v_tag} Without Silence No-Collapse {ver_tag_model} Base-{k_base} N-Grams {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gan = res_entry.get("analysis_gan_job")
                if analysis_gan is None:
                    analysis_gan = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=audio_emb_path,
                        text_embeddings_npy=text_emb_path,
                        dictionary_json=align_gan.out_dictionary_json,
                        raw_audio_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=vocab_path,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gan.out_alignment_report,
                        title_prefix=f"{v_tag} No-Collapse ({ver_tag_model}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                        src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        name_prefix=f"analysis_{v_tag}_{ver_tag_model}_base{k_base}_{d_tag}_{code_type}_{clus_str}_{dw_tag}_{metric}_{norm_clean}_pca{pca_dim}",
                        _version=4,
                    )
                analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is not None:
                    w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                    if hasattr(w2v_isotropy_job, "out_src_after_pca_plot"):
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                    if hasattr(w2v_isotropy_job, "out_trg_after_pca_plot"):
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                    if hasattr(w2v_isotropy_job, "out_before_after_panel_plot"):
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

                # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
                top_ngrams_job = TopClusterNgramsAnalysisJob(
                    cluster_vocab_file=a_clus.out_cluster_vocab_txt,
                    cluster_counts_file=a_clus.out_ngram_counts_txt,
                    dictionary_json=align_gan.out_dictionary_json,
                    mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                    text_embeddings_npy=t_w2v.out_embeddings_npy,
                    text_vocab_file=vocab_path,
                    top_k=100,
                    title_prefix=f"{v_tag} ({ver_tag_model}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                    name_prefix=f"top_cluster_ngrams_{v_tag}_{ver_tag_model}_base{k_base}_{d_tag}_{code_type}_{clus_str}",
                )
                top_ngrams_job.add_alias(f"{d_prefix}/{exp_key}/top_100_ngrams")
                tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
                tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

            if run_analysis:
                for code_type, clus_str in target_setups:
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                    a_clus = cluster_jobs_by_name.get(clus_str)
                    if a_clus is None:
                        continue
                    analysis_prefix = f"{d_prefix}/{exp_key}/analysis"
                    cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                        cluster_counts_file=a_clus.out_ngram_counts_txt,
                        raw_clusters_file=a_clus.out_ngram_text,
                        cluster_hdfs=None,
                        k_clusters=None,
                        top_k=50,
                        title_prefix=f"{v_tag} No-Collapse Compositional Audio {code_type.capitalize()}s Without Silence ({ver_tag_model}, D={d_rem_int}, {clus_str}, {norm_title})",
                        name_prefix=f"speech_cluster_dist_{v_tag}_{ver_tag_model}_base{k_base}_{d_tag}_wo_sil_{norm_clean}_pca{pca_dim}_{clus_str}",
                    )
                    cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                    tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


def eval_v5_9_4_1_wo_sil_spin_audio_ngrams_masked_csls(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.9.4.1 Subablation: Structurally-Masked CSLS & Procrustes Refinement (Method B).
    Precludes impossible pairs (e.g. 24_24_24 mapped to V_DH_AH) from dictionary induction and SVD rotation.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        spin_codebook_sizes=spin_codebook_sizes,
        spin_codebook_size=spin_codebook_size,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="masked_csls",
        compatibility_mode=compatibility_mode,
        version_tag="v5.9.4.1_masked_csls",
    )


def eval_v5_9_4_2_wo_sil_spin_audio_ngrams_struct_reg_gan(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    lambda_struct: float = 0.5,
    struct_margin: float = 0.0,
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.9.4.2 Subablation: Structural Hinge Repulsion during GAN + Masked Refinement (Method A1 + B).
    Actively pushes mapped impossible pairs apart during GAN training via margin hinge loss,
    followed by masked Procrustes refinement.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        spin_codebook_sizes=spin_codebook_sizes,
        spin_codebook_size=spin_codebook_size,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="reg_gan_and_masked_csls",
        lambda_struct=lambda_struct,
        struct_margin=struct_margin,
        compatibility_mode=compatibility_mode,
        version_tag="v5.9.4.2_struct_reg_gan",
    )

# ---------------------------------------------------------------------------
# HuBERT-Base Layer 6 Replications of 5.9.4+ (No-Collapse, Masked CSLS, Regularized GAN)
# ---------------------------------------------------------------------------

def eval_v5_9_4_wo_sil_hubert_l6_audio_ngrams_no_collapse(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.4 Compositional Audio N-Grams without Silence and without Collapsing using Vanilla HuBERT Layer 6.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone="hubert_l6",
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        version_tag="v5.9.4",
    )


def eval_v5_9_4_1_wo_sil_hubert_l6_audio_ngrams_masked_csls(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.9.4.1 HuBERT Layer 6 Compositional Audio N-Grams with Structurally-Masked CSLS & Procrustes Refinement.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone="hubert_l6",
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="masked_csls",
        compatibility_mode=compatibility_mode,
        version_tag="v5.9.4.1_masked_csls",
    )


def eval_v5_9_4_2_wo_sil_hubert_l6_audio_ngrams_struct_reg_gan(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    lambda_struct: float = 0.5,
    struct_margin: float = 0.0,
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.9.4.2 HuBERT Layer 6 with Structural Hinge Repulsion during GAN + Masked Refinement.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone="hubert_l6",
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="reg_gan_and_masked_csls",
        lambda_struct=lambda_struct,
        struct_margin=struct_margin,
        compatibility_mode=compatibility_mode,
        version_tag="v5.9.4.2_struct_reg_gan",
    )


# ---------------------------------------------------------------------------
# Setups 5.10 (Approach A: Finer Allophonic Resolution K_base >= 120, No-Collapse)
# ---------------------------------------------------------------------------

def eval_v5_10_wo_sil_audio_ngrams_no_collapse(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 120,
    audio_backbone: str = "hubert_l6",
    spin_codebook_size: Optional[int] = None,
    audio_layer: Optional[int] = None,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.10 Sub-Phonemic / Allophonic Audio N-Grams (Approach A baseline, K_base=120):
    - Uses K_base = 120 centroids on pooled phoneme segments to prevent adjacent centroid collisions.
    - Preserves 1-to-1 phoneme segment length (collapse_consecutive=False) without artificial self-loops.
    - Defaults to HuBERT-Base Layer 6 (or Spin-HuBERT Layer 12 if audio_backbone='spin_hubert').
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone=audio_backbone,
        spin_codebook_size=spin_codebook_size,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        version_tag="v5.10",
    )


def eval_v5_10_1_wo_sil_audio_ngrams_masked_csls(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 120,
    audio_backbone: str = "hubert_l6",
    spin_codebook_size: Optional[int] = None,
    audio_layer: Optional[int] = None,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.10.1 Sub-Phonemic Audio N-Grams (K_base=120) with Structurally-Masked CSLS & Procrustes Refinement.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone=audio_backbone,
        spin_codebook_size=spin_codebook_size,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="masked_csls",
        compatibility_mode=compatibility_mode,
        version_tag="v5.10.1_masked_csls",
    )


def eval_v5_10_2_wo_sil_audio_ngrams_struct_reg_gan(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 120,
    audio_backbone: str = "hubert_l6",
    spin_codebook_size: Optional[int] = None,
    audio_layer: Optional[int] = None,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    lambda_struct: float = 0.5,
    struct_margin: float = 0.0,
    compatibility_mode: str = "collapsed_length",
):
    """
    Evaluates v5.10.2 Sub-Phonemic Audio N-Grams (K_base=120) with Structural Hinge Regularized GAN + Masked Refinement.
    """
    return eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(
        cluster=cluster,
        pca_dim=pca_dim,
        run_analysis=run_analysis,
        alignment_results_by_d=alignment_results_by_d,
        norm_tag=norm_tag,
        base_cluster_count=base_cluster_count,
        audio_backbone=audio_backbone,
        spin_codebook_size=spin_codebook_size,
        audio_layer=audio_layer,
        d_values=d_values,
        embedding_model=embedding_model,
        structural_constraint="reg_gan_and_masked_csls",
        lambda_struct=lambda_struct,
        struct_margin=struct_margin,
        compatibility_mode=compatibility_mode,
        version_tag="v5.10.2_struct_reg_gan",
    )

def eval_v5_9_5_wo_sil_spin_audio_ngrams_viterbi_regularized(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_pen_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    repetition_penalties: Optional[List[float]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.5 Spin-HuBERT Compositional Audio N-Grams without Silence and
    with Markov Soft Repetition Penalty Regularization (Viterbi decoding) on base K=40 clustering:
      - Prevents artificial duplicate runs (c_t == c_{t-1}) without collapsing sequence lengths.
      - Ablates soft penalty values lambda in repetition_penalties (default: [0.1, 0.25, 0.5]).
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default codebook size 2048, arXiv:2305.11072).
      - Base K-Means at monophone scale (K_base = 40) with Unit-Sphere normalization on 512-dim PCA.
      - collapse_consecutive=False: Adjacent oracle segments are PRESERVED (no length loss).
      - Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Text phonemization with sil_prob=0.0 and surround_w_sil=False (no artificial <SIL> symbols).
      - All-but-the-Top Word2Vec postprocessing (default D=2).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram LM rescoring (collapse_before=False).
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.5_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}/spin_{spin_k}/pen_{pen}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [2]

    if repetition_penalties is None:
        repetition_penalties = [0.1, 0.25, 0.5]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.5_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for pen in repetition_penalties:
            pen_float = float(pen)
            pen_tag = f"pen_{pen_float}"

            for d_rem in d_values:
                d_rem_int = int(d_rem)
                d_tag = f"D{d_rem_int}"
                d_prefix = f"{prefix_base}/{spin_tag}/{pen_tag}/{d_tag}"

                lookup_key = (spin_k_int, pen_float, d_rem_int)
                if alignment_results_by_pen_d is not None and lookup_key in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[lookup_key]
                elif alignment_results_by_pen_d is not None and (pen_float, d_rem_int) in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[(pen_float, d_rem_int)]
                elif alignment_results_by_pen_d is not None and pen_float in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[pen_float]
                else:
                    alignment_results = build_self_learning_asr_alignment_pipeline(
                        librispeech_key="train-clean-100",
                        audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                        audio_layer=audio_layer,
                        word2vec_dims=[dim],
                        word2vec_windows=[win],
                        word2vec_epochs=20,
                        cluster_counts=[k_base],
                        coverage_threshold=0.80,
                        code_types=["diphone", "triphone"],
                        code_cluster_map={
                            "diphone": [(k_base, f"clus_{k_base}_2gram")],
                            "triphone": [(k_base, f"clus_{k_base}_3gram")],
                        },
                        segmentation_type="cheating",
                        pca_dim=pca_dim,
                        version=f"v5.9.5_{spin_tag}_{pen_tag}_base{k_base}_wo_sil_{norm_clean}_isotropic_{d_tag}",
                        text_version="v5.9.2_wo_sil",
                        text_model_tag="hubert_l6",
                        normalize_segments=norm_clean,
                        retrieval_metrics=[metric],
                        csls_k=10,
                        alignment_method="gan",
                        embedding_model=embedding_model,
                        word2vec_postprocessing="all_but_the_top",
                        all_but_the_top_D=d_rem_int,
                        run_isotropy_analysis=True,
                        spin_codebook_size=spin_k_int,
                        use_audio_ngrams=True,
                        sil_prob=0.0,
                        surround_w_sil=False,
                        collapse_consecutive=False,
                        repetition_penalty=pen_float,
                    )

                cluster_jobs_by_name = {}

                for code_type, clus_str in target_setups:
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                    res_entry = (
                        alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                        if alignment_results is not None
                        else None
                    )
                    if res_entry is None:
                        continue

                    a_clus = res_entry["cluster_job"]
                    a_w2v = res_entry["audio_w2v_job"]
                    t_w2v = res_entry["text_w2v_job"]
                    raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                    raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                    align_gan = res_entry["align_gan_job"]

                    cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                    audio_emb_path = align_gan.out_mapped_audio_emb_npy
                    text_emb_path = t_w2v.out_embeddings_npy
                    vocab_path = t_w2v.out_vocab_txt
                    cluster_jobs_by_name[clus_str] = a_clus

                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                    if hasattr(a_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                    if hasattr(t_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                    # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                    # 2. 4-Gram Language Model Rescoring Job (collapse_before=False)
                    rescore_train_job = PhonemeNgramRescoreJob(
                        hypotheses_file=eval_train_job.out_hypotheses_txt,
                        reference_file=eval_train_job.out_reference_txt,
                        text_corpus_file=text_corpus_path,
                        lm_order=4,
                        backoff_alpha=0.4,
                        lm_weight=0.5,
                        match_bonus=2.0,
                        beam_width=10,
                        collapse_before=False,
                        name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                    # 3. Joined Score Results
                    join_asr_scores = JoinCheatingSegScoresJob(
                        scores={
                            "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                            "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                        },
                        title=f"Cheating Segmentation Evaluation (v5.9.5 Without Silence Spin-{spin_k_int} L{audio_layer} {pen_tag} {d_tag}, {code_type}, {clus_str})",
                    )
                    join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                    # 4. Cross-Modal 2D PCA & Representation Analysis
                    analysis_gan = res_entry.get("analysis_gan_job")
                    if analysis_gan is None:
                        analysis_gan = EmbeddingAlignmentAnalysisJob(
                            mapped_audio_embeddings_npy=audio_emb_path,
                            text_embeddings_npy=text_emb_path,
                            dictionary_json=align_gan.out_dictionary_json,
                            raw_audio_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                            raw_text_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=vocab_path,
                            audio_counts_file=a_clus.out_cluster_counts_txt,
                            text_counts_file=t_w2v.out_counts_txt,
                            alignment_report_json=align_gan.out_alignment_report,
                            title_prefix=f"v5.9.5 ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                            src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                            trg_label=f"Text {code_type.capitalize()}s",
                            name_prefix=f"analysis_v5.9.5_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}_{dw_tag}_{metric}_{norm_clean}_pca{pca_dim}",
                            _version=4,
                        )
                    analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                    # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
                    w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                    if w2v_isotropy_job is not None:
                        w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_src_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_trg_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_before_after_panel_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

                    # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
                    top_ngrams_job = TopClusterNgramsAnalysisJob(
                        cluster_vocab_file=a_clus.out_cluster_vocab_txt,
                        cluster_counts_file=a_clus.out_ngram_counts_txt,
                        dictionary_json=align_gan.out_dictionary_json,
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        text_vocab_file=vocab_path,
                        top_k=100,
                        title_prefix=f"v5.9.5 ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                        name_prefix=f"top_cluster_ngrams_v5.9.5_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}",
                    )
                    top_ngrams_job.add_alias(f"{d_prefix}/{exp_key}/top_100_ngrams")
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

                if run_analysis:
                    for code_type, clus_str in target_setups:
                        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                        a_clus = cluster_jobs_by_name.get(clus_str)
                        if a_clus is None:
                            continue
                        analysis_prefix = f"{d_prefix}/{exp_key}/analysis"
                        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                            cluster_counts_file=a_clus.out_ngram_counts_txt,
                            raw_clusters_file=a_clus.out_ngram_text,
                            cluster_hdfs=None,
                            k_clusters=None,
                            top_k=50,
                            title_prefix=f"v5.9.5 Compositional Audio {code_type.capitalize()}s Without Silence ({spin_tag}, {pen_tag}, D={d_rem_int}, {clus_str}, {norm_title})",
                            name_prefix=f"speech_cluster_dist_v5.9.5_{spin_tag}_{pen_tag}_{d_tag}_wo_sil_{norm_clean}_pca{pca_dim}_{clus_str}",
                        )
                        cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)

def eval_v5_9_5_copkmeans_wo_sil_spin_audio_ngrams(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_pen_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    repetition_penalties: Optional[List[float]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.5_copkmeans Spin-HuBERT Compositional Audio N-Grams without Silence and
    with Constrained COP-KMeans Regularization (alternating Viterbi with repetition penalty + centroid re-estimation):
      - Prevents artificial duplicate runs (c_t == c_{t-1}) without collapsing sequence lengths.
      - Iteratively refines cluster centroids so that consecutive segments are pushed into distinct clusters.
      - Ablates soft penalty values lambda in repetition_penalties (default: [0.1, 0.25, 0.5]).
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default codebook size 2048, arXiv:2305.11072).
      - Base K-Means at monophone scale (K_base = 40) with Unit-Sphere normalization on 512-dim PCA.
      - collapse_consecutive=False: Adjacent oracle segments are PRESERVED (no length loss).
      - Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Text phonemization with sil_prob=0.0 and surround_w_sil=False (no artificial <SIL> symbols).
      - All-but-the-Top Word2Vec postprocessing (default D=2).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram LM rescoring (collapse_before=False).
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.5_copkmeans_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}/spin_{spin_k}/pen_{pen}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [2]

    if repetition_penalties is None:
        repetition_penalties = [0.1, 0.25, 0.5]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.5_copkmeans_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for pen in repetition_penalties:
            pen_float = float(pen)
            pen_tag = f"pen_{pen_float}"

            for d_rem in d_values:
                d_rem_int = int(d_rem)
                d_tag = f"D{d_rem_int}"
                d_prefix = f"{prefix_base}/{spin_tag}/{pen_tag}/{d_tag}"

                lookup_key = (spin_k_int, pen_float, d_rem_int)
                if alignment_results_by_pen_d is not None and lookup_key in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[lookup_key]
                elif alignment_results_by_pen_d is not None and (pen_float, d_rem_int) in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[(pen_float, d_rem_int)]
                elif alignment_results_by_pen_d is not None and pen_float in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[pen_float]
                else:
                    alignment_results = build_self_learning_asr_alignment_pipeline(
                        librispeech_key="train-clean-100",
                        audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                        audio_layer=audio_layer,
                        word2vec_dims=[dim],
                        word2vec_windows=[win],
                        word2vec_epochs=20,
                        cluster_counts=[k_base],
                        coverage_threshold=0.80,
                        code_types=["diphone", "triphone"],
                        code_cluster_map={
                            "diphone": [(k_base, f"clus_{k_base}_2gram")],
                            "triphone": [(k_base, f"clus_{k_base}_3gram")],
                        },
                        segmentation_type="cheating",
                        pca_dim=pca_dim,
                        version=f"v5.9.5_copkmeans_{spin_tag}_{pen_tag}_base{k_base}_wo_sil_{norm_clean}_isotropic_{d_tag}",
                        text_version="v5.9.2_wo_sil",
                        text_model_tag="hubert_l6",
                        normalize_segments=norm_clean,
                        retrieval_metrics=[metric],
                        csls_k=10,
                        alignment_method="gan",
                        embedding_model=embedding_model,
                        word2vec_postprocessing="all_but_the_top",
                        all_but_the_top_D=d_rem_int,
                        run_isotropy_analysis=True,
                        spin_codebook_size=spin_k_int,
                        use_audio_ngrams=True,
                        sil_prob=0.0,
                        surround_w_sil=False,
                        collapse_consecutive=False,
                        repetition_penalty=pen_float,
                        use_cop_kmeans=True,
                    )

                cluster_jobs_by_name = {}

                for code_type, clus_str in target_setups:
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                    res_entry = (
                        alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                        if alignment_results is not None
                        else None
                    )
                    if res_entry is None:
                        continue

                    a_clus = res_entry["cluster_job"]
                    a_w2v = res_entry["audio_w2v_job"]
                    t_w2v = res_entry["text_w2v_job"]
                    raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                    raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                    align_gan = res_entry["align_gan_job"]

                    cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                    audio_emb_path = align_gan.out_mapped_audio_emb_npy
                    text_emb_path = t_w2v.out_embeddings_npy
                    vocab_path = t_w2v.out_vocab_txt
                    cluster_jobs_by_name[clus_str] = a_clus

                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                    if hasattr(a_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                    if hasattr(t_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                    # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_copkmeans_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                    # 2. 4-Gram Language Model Rescoring Job (collapse_before=False)
                    rescore_train_job = PhonemeNgramRescoreJob(
                        hypotheses_file=eval_train_job.out_hypotheses_txt,
                        reference_file=eval_train_job.out_reference_txt,
                        text_corpus_file=text_corpus_path,
                        lm_order=4,
                        backoff_alpha=0.4,
                        lm_weight=0.5,
                        match_bonus=2.0,
                        beam_width=10,
                        collapse_before=False,
                        name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_copkmeans_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                    # 3. Joined Score Results
                    join_asr_scores = JoinCheatingSegScoresJob(
                        scores={
                            "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                            "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                        },
                        title=f"Cheating Segmentation Evaluation (v5.9.5 COP-KMeans Without Silence Spin-{spin_k_int} L{audio_layer} {pen_tag} {d_tag}, {code_type}, {clus_str})",
                    )
                    join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                    # 4. Cross-Modal 2D PCA & Representation Analysis
                    analysis_gan = res_entry.get("analysis_gan_job")
                    if analysis_gan is None:
                        analysis_gan = EmbeddingAlignmentAnalysisJob(
                            mapped_audio_embeddings_npy=audio_emb_path,
                            text_embeddings_npy=text_emb_path,
                            dictionary_json=align_gan.out_dictionary_json,
                            raw_audio_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                            raw_text_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=vocab_path,
                            audio_counts_file=a_clus.out_cluster_counts_txt,
                            text_counts_file=t_w2v.out_counts_txt,
                            alignment_report_json=align_gan.out_alignment_report,
                            title_prefix=f"v5.9.5 COP-KMeans ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                            src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                            trg_label=f"Text {code_type.capitalize()}s",
                            name_prefix=f"analysis_v5.9.5_copkmeans_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}_{dw_tag}_{metric}_{norm_clean}_pca{pca_dim}",
                            _version=4,
                        )
                    analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                    # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
                    w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                    if w2v_isotropy_job is not None:
                        w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_src_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_trg_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_before_after_panel_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

                    # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
                    top_ngrams_job = TopClusterNgramsAnalysisJob(
                        cluster_vocab_file=a_clus.out_cluster_vocab_txt,
                        cluster_counts_file=a_clus.out_ngram_counts_txt,
                        dictionary_json=align_gan.out_dictionary_json,
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        text_vocab_file=vocab_path,
                        top_k=100,
                        title_prefix=f"v5.9.5 COP-KMeans ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                        name_prefix=f"top_cluster_ngrams_v5.9.5_copkmeans_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}",
                    )
                    top_ngrams_job.add_alias(f"{d_prefix}/{exp_key}/top_100_ngrams")
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

                if run_analysis:
                    for code_type, clus_str in target_setups:
                        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                        a_clus = cluster_jobs_by_name.get(clus_str)
                        if a_clus is None:
                            continue
                        analysis_prefix = f"{d_prefix}/{exp_key}/analysis"
                        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                            cluster_counts_file=a_clus.out_ngram_counts_txt,
                            raw_clusters_file=a_clus.out_ngram_text,
                            cluster_hdfs=None,
                            k_clusters=None,
                            top_k=50,
                            title_prefix=f"v5.9.5 COP-KMeans Compositional Audio {code_type.capitalize()}s Without Silence ({spin_tag}, {pen_tag}, D={d_rem_int}, {clus_str}, {norm_title})",
                            name_prefix=f"speech_cluster_dist_v5.9.5_copkmeans_{spin_tag}_{pen_tag}_{d_tag}_wo_sil_{norm_clean}_pca{pca_dim}_{clus_str}",
                        )
                        cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# Corpus Distribution Analysis Pipeline
# =============================================================================

def eval_v5_9_5_pckmeans_wo_sil_spin_audio_ngrams(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_pen_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    repetition_penalties: Optional[List[float]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.5_pckmeans Spin-HuBERT Compositional Audio N-Grams without Silence and
    with Pairwise Constrained K-Means (PCK-Means, Basu et al., SDM 2004):
      - Soft-penalty pairwise cannot-link regularization on adjacent oracle segments.
      - Alternates between exact O(T * K) Viterbi assignment with soft repetition penalty lambda
        and centroid re-estimation over the unit sphere.
      - Ablates soft penalty values lambda in repetition_penalties (default: [0.1, 0.25, 0.5]).
      - Acoustic Backbone: Spin-HuBERT Layer 12 (default codebook size 2048, arXiv:2305.11072).
      - Base K-Means at monophone scale (K_base = 40) with Unit-Sphere normalization on 512-dim PCA.
      - collapse_consecutive=False: Adjacent oracle segments are PRESERVED (no length loss).
      - Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Text phonemization with sil_prob=0.0 and surround_w_sil=False (no artificial <SIL> symbols).
      - All-but-the-Top Word2Vec postprocessing (default D=2).
      - GAN domain-adversarial alignment with CSLS Procrustes refinement.
      - Evaluation: Cheating segmentation nearest-neighbor CSLS PER + 4-gram LM rescoring (collapse_before=False).
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.5_pckmeans_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}/spin_{spin_k}/pen_{pen}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [2]

    if repetition_penalties is None:
        repetition_penalties = [0.1, 0.25, 0.5]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.5_pckmeans_wo_sil_spin_audio_ngrams_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for pen in repetition_penalties:
            pen_float = float(pen)
            pen_tag = f"pen_{pen_float}"

            for d_rem in d_values:
                d_rem_int = int(d_rem)
                d_tag = f"D{d_rem_int}"
                d_prefix = f"{prefix_base}/{spin_tag}/{pen_tag}/{d_tag}"

                lookup_key = (spin_k_int, pen_float, d_rem_int)
                if alignment_results_by_pen_d is not None and lookup_key in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[lookup_key]
                elif alignment_results_by_pen_d is not None and (pen_float, d_rem_int) in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[(pen_float, d_rem_int)]
                elif alignment_results_by_pen_d is not None and pen_float in alignment_results_by_pen_d:
                    alignment_results = alignment_results_by_pen_d[pen_float]
                else:
                    alignment_results = build_self_learning_asr_alignment_pipeline(
                        librispeech_key="train-clean-100",
                        audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                        audio_layer=audio_layer,
                        word2vec_dims=[dim],
                        word2vec_windows=[win],
                        word2vec_epochs=20,
                        cluster_counts=[k_base],
                        coverage_threshold=0.80,
                        code_types=["diphone", "triphone"],
                        code_cluster_map={
                            "diphone": [(k_base, f"clus_{k_base}_2gram")],
                            "triphone": [(k_base, f"clus_{k_base}_3gram")],
                        },
                        segmentation_type="cheating",
                        pca_dim=pca_dim,
                        version=f"v5.9.5_pckmeans_{spin_tag}_{pen_tag}_base{k_base}_wo_sil_{norm_clean}_isotropic_{d_tag}",
                        text_version="v5.9.2_wo_sil",
                        text_model_tag="hubert_l6",
                        normalize_segments=norm_clean,
                        retrieval_metrics=[metric],
                        csls_k=10,
                        alignment_method="gan",
                        embedding_model=embedding_model,
                        word2vec_postprocessing="all_but_the_top",
                        all_but_the_top_D=d_rem_int,
                        run_isotropy_analysis=True,
                        spin_codebook_size=spin_k_int,
                        use_audio_ngrams=True,
                        sil_prob=0.0,
                        surround_w_sil=False,
                        collapse_consecutive=False,
                        repetition_penalty=pen_float,
                        use_pck_kmeans=True,
                    )

                cluster_jobs_by_name = {}

                for code_type, clus_str in target_setups:
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                    res_entry = (
                        alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                        if alignment_results is not None
                        else None
                    )
                    if res_entry is None:
                        continue

                    a_clus = res_entry["cluster_job"]
                    a_w2v = res_entry["audio_w2v_job"]
                    t_w2v = res_entry["text_w2v_job"]
                    raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                    raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                    align_gan = res_entry["align_gan_job"]

                    cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                    audio_emb_path = align_gan.out_mapped_audio_emb_npy
                    text_emb_path = t_w2v.out_embeddings_npy
                    vocab_path = t_w2v.out_vocab_txt
                    cluster_jobs_by_name[clus_str] = a_clus

                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                    if hasattr(a_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                    if hasattr(t_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                    # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_pckmeans_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                    # 2. 4-Gram Language Model Rescoring Job (collapse_before=False)
                    rescore_train_job = PhonemeNgramRescoreJob(
                        hypotheses_file=eval_train_job.out_hypotheses_txt,
                        reference_file=eval_train_job.out_reference_txt,
                        text_corpus_file=text_corpus_path,
                        lm_order=4,
                        backoff_alpha=0.4,
                        lm_weight=0.5,
                        match_bonus=2.0,
                        beam_width=10,
                        collapse_before=False,
                        name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.5_pckmeans_wo_sil_spin_{spin_k_int}_{pen_tag}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                    # 3. Joined Score Results
                    join_asr_scores = JoinCheatingSegScoresJob(
                        scores={
                            "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                            "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                        },
                        title=f"Cheating Segmentation Evaluation (v5.9.5 PCK-Means Without Silence Spin-{spin_k_int} L{audio_layer} {pen_tag} {d_tag}, {code_type}, {clus_str})",
                    )
                    join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                    # 4. Cross-Modal 2D PCA & Representation Analysis
                    analysis_gan = res_entry.get("analysis_gan_job")
                    if analysis_gan is None:
                        analysis_gan = EmbeddingAlignmentAnalysisJob(
                            mapped_audio_embeddings_npy=audio_emb_path,
                            text_embeddings_npy=text_emb_path,
                            dictionary_json=align_gan.out_dictionary_json,
                            raw_audio_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                            raw_text_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=vocab_path,
                            audio_counts_file=a_clus.out_cluster_counts_txt,
                            text_counts_file=t_w2v.out_counts_txt,
                            alignment_report_json=align_gan.out_alignment_report,
                            title_prefix=f"v5.9.5 PCK-Means ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                            src_label=f"Audio {code_type.capitalize()}s ({clus_str})",
                            trg_label=f"Text {code_type.capitalize()}s",
                            name_prefix=f"analysis_v5.9.5_pckmeans_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}_{dw_tag}_{metric}_{norm_clean}_pca{pca_dim}",
                            _version=4,
                        )
                    analysis_gan.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gan.out_cross_modal_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gan.out_cross_modal_pca_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gan.out_source_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gan.out_target_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gan.out_cross_modal_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gan.out_source_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gan.out_target_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gan.out_unaligned_joint_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gan.out_alignment_comparison_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gan.out_analysis_report_json)

                    # 5. Baseline Embedding Isotropy & PCA Spectrum Analysis
                    w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                    if w2v_isotropy_job is not None:
                        w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_src_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_trg_after_pca_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                        if hasattr(w2v_isotropy_job, "out_before_after_panel_plot"):
                            tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                        tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

                    # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
                    top_ngrams_job = TopClusterNgramsAnalysisJob(
                        cluster_vocab_file=a_clus.out_cluster_vocab_txt,
                        cluster_counts_file=a_clus.out_ngram_counts_txt,
                        dictionary_json=align_gan.out_dictionary_json,
                        mapped_audio_embeddings_npy=align_gan.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        text_vocab_file=vocab_path,
                        top_k=100,
                        title_prefix=f"v5.9.5 PCK-Means ({spin_tag}, {pen_tag}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                        name_prefix=f"top_cluster_ngrams_v5.9.5_pckmeans_{spin_tag}_{pen_tag}_{d_tag}_{code_type}_{clus_str}",
                    )
                    top_ngrams_job.add_alias(f"{d_prefix}/{exp_key}/top_100_ngrams")
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

                if run_analysis:
                    for code_type, clus_str in target_setups:
                        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                        a_clus = cluster_jobs_by_name.get(clus_str)
                        if a_clus is None:
                            continue
                        analysis_prefix = f"{d_prefix}/{exp_key}/analysis"
                        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                            cluster_counts_file=a_clus.out_ngram_counts_txt,
                            raw_clusters_file=a_clus.out_ngram_text,
                            cluster_hdfs=None,
                            k_clusters=None,
                            top_k=50,
                            title_prefix=f"v5.9.5 PCK-Means Compositional Audio {code_type.capitalize()}s Without Silence ({spin_tag}, {pen_tag}, D={d_rem_int}, {clus_str}, {norm_title})",
                            name_prefix=f"speech_cluster_dist_v5.9.5_pckmeans_{spin_tag}_{pen_tag}_{d_tag}_wo_sil_{norm_clean}_pca{pca_dim}_{clus_str}",
                        )
                        cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)


# =============================================================================
# Corpus Distribution Analysis Pipeline
# =============================================================================

def eval_v5_9_audio_word_boundaries(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    audio_layer: int = 6,
    word_boundary_token: str = "<w>",
):
    """
    Evaluates v5.9 Compositional Audio N-Grams with Explicit Word Boundary Delimiters (<w>):
      - Uses GMM oracle cheating segmentation to pool HuBERT Layer 6 frame representations.
      - Aligns word boundaries using transcript lexicon DP and inserts `<w>` tokens between words.
      - Generates audio and text 2-grams (diphones) and 3-grams (triphones) with boundary token `<w>`.
      - Skip-Gram Word2Vec (dim=128, win=2) aligned via GAN with CSLS Procrustes refinement.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)
    prefix_name = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9_audio_word_boundaries_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    if alignment_results is None:
        alignment_results = build_self_learning_asr_alignment_pipeline(
            librispeech_key="train-clean-100",
            audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
            audio_layer=audio_layer,
            word2vec_dims=[128],
            word2vec_windows=[2],
            word2vec_epochs=20,
            cluster_counts=[k_base],
            coverage_threshold=0.80,
            code_types=["diphone", "triphone"],
            code_cluster_map={
                "diphone": [(k_base, f"clus_{k_base}_2gram")],
                "triphone": [(k_base, f"clus_{k_base}_3gram")],
            },
            segmentation_type="cheating",
            pca_dim=pca_dim,
            version=f"v5.9_audio_word_boundaries_base{k_base}_{norm_clean}",
            text_version="v5.5_audio_word_boundaries",
            text_model_tag="hubert_l6",
            normalize_segments=norm_clean,
            retrieval_metrics=["csls"],
            csls_k=10,
            alignment_method="gan",
            embedding_model="word2vec",
            run_isotropy_analysis=True,
            use_audio_ngrams=True,
            audio_word_boundaries=True,
            word_boundary_token=word_boundary_token,
        )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    cluster_jobs_by_name = {}

    for code_type, clus_str in target_setups:
        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
        res_entry = (
            alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
            if alignment_results is not None
            else None
        )
        if res_entry is None:
            continue

        a_clus = res_entry["cluster_job"]
        a_w2v = res_entry["audio_w2v_job"]
        t_w2v = res_entry["text_w2v_job"]
        align_gan = res_entry["align_gan_job"]

        cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
        audio_emb_path = align_gan.out_mapped_audio_emb_npy
        text_emb_path = t_w2v.out_embeddings_npy
        vocab_path = t_w2v.out_vocab_txt
        cluster_jobs_by_name[clus_str] = a_clus

        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
        tk.register_output(f"{prefix_name}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

        # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
            name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9_audio_word_boundaries_{norm_clean}_pca{pca_dim}",
        )
        eval_train_job.add_alias(f"{prefix_name}/{exp_key}/eval_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

        # 2. 4-Gram Language Model Rescoring Job
        rescore_train_job = PhonemeNgramRescoreJob(
            hypotheses_file=eval_train_job.out_hypotheses_txt,
            reference_file=eval_train_job.out_reference_txt,
            text_corpus_file=text_corpus_path,
            lm_order=4,
            backoff_alpha=0.4,
            lm_weight=0.5,
            match_bonus=2.0,
            beam_width=10,
            collapse_before=True,
            name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9_audio_wb_{norm_clean}_pca{pca_dim}",
        )
        rescore_train_job.add_alias(f"{prefix_name}/{exp_key}/rescore_4gram_train3k")
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

        # 3. Joined Score Results
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.9 Audio Word Boundaries, {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

        # 3. Joined Score Results
        join_asr_scores = JoinCheatingSegScoresJob(
            scores={
                "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
            },
            title=f"Cheating Segmentation Evaluation (v5.9 Audio Word Boundaries {code_type}, {clus_str})",
        )
        join_asr_scores.add_alias(f"{prefix_name}/{exp_key}/score_results")
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
        tk.register_output(f"{prefix_name}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

    return alignment_results


def eval_v5_9_1_spin_audio_word_boundaries(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    word_boundary_token: str = "<w>",
):
    """
    Evaluates v5.9.1 Spin-HuBERT Compositional Audio N-Grams with Audio Word Boundaries (<w>) and All-but-the-Top Isotropy:
      - Uses GMM oracle cheating segmentation to pool Spin-HuBERT Layer 12 frame representations.
      - Aligns word boundaries using transcript lexicon DP and inserts `<w>` tokens between words.
      - Skip-Gram Word2Vec (dim=128, win=2) with All-but-the-Top (D=2) aligned via GAN + CSLS Procrustes.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)

    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]

    if d_values is None:
        d_values = [2]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.1_spin_audio_word_boundaries_base{k_base}_{norm_clean}_gan_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.1_spin_audio_word_boundaries_base{k_base}_{norm_clean}_{d_tag}",
                    text_version="v5.5_audio_word_boundaries",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="gan",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                    audio_word_boundaries=True,
                    word_boundary_token=word_boundary_token,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                align_gan = res_entry["align_gan_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gan.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)

                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/dict.json", align_gan.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/report.json", align_gan.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_star.npy", align_gan.out_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/W_adv.npy", align_gan.out_adv_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/mapped_audio_embeddings.npy", align_gan.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/gan_alignment/gan_training_curves.png", align_gan.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_spin_wb_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_spin_wb_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.1 Spin Audio Word Boundaries {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

    return alignment_results


def eval_v5_9_1_geowarp(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.9.1_geowarp: Spin-HuBERT Compositional Audio N-Grams with All-but-the-Top Isotropy
    aligned via GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes (Schnaus et al., 2026, arXiv:2610.09411).
      - Acoustic Backbone: Spin-HuBERT Layer 12 (codebook size 2048).
      - Frame-level PCA projection (768 -> 512) -> oracle segment averaging ->
        unit-sphere (L2) normalization -> K-Means at monophone scale (K_base = 40).
      - Compositional Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Embeddings: Skip-Gram Word2Vec (dim=128, win=2, epochs=20) + All-but-the-Top (default D=2).
      - Alignment: GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes + CSLS MNN refinement.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.1_geowarp_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()

    k_base = int(base_cluster_count)
    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]
    if d_values is None:
        d_values = [2]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.1_geowarp_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.1_geowarp_{spin_tag}_base{k_base}_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.5",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="geowarp",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"geowarp_{metric}"))
                    or alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gw = res_entry.get("align_geowarp_job") or res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gw.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                # Register postprocessed embeddings & GeoWarP alignment outputs
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/dict.json", align_gw.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/report.json", align_gw.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_star.npy", align_gw.out_mapping_matrix_npy)
                if hasattr(align_gw, "out_init_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_init.npy", align_gw.out_init_mapping_matrix_npy)
                if hasattr(align_gw, "out_wp_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_wp.npy", align_gw.out_wp_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/mapped_audio_embeddings.npy", align_gw.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/geowarp_training_curves.png", align_gw.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.1_geowarp Spin-{spin_k_int} L{audio_layer} Audio N-Grams {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gw = res_entry.get("analysis_geowarp_job") or res_entry.get("analysis_gan_job")
                if analysis_gw is None:
                    analysis_gw = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gw.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gw.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gw.out_alignment_report,
                        title_prefix=f"v5.9.1_geowarp (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        _version=4,
                    )

                analysis_gw.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gw.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gw.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gw.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gw.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gw.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gw.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gw.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gw.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gw.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gw.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Analysis
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.9.1_geowarp (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    return alignment_results


def eval_v5_9_1_geowarp_spectral(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    spectral_n_neighbors: int = 30,
    spectral_sample_size: int = 25000,
):
    """
    Evaluates v5.9.1_geowarp_spectral: Swapping K-Means segment clustering with Spectral Clustering
    (nearest-neighbors affinity graph on pooled segment representations), followed by compositional
    audio N-grams (2-grams & 3-grams, 80% coverage), Skip-Gram Word2Vec (dim=128, win=2) with All-but-the-Top (D=2),
    and GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes (Schnaus et al., 2026, arXiv:2610.09411).
      - Acoustic Backbone: Spin-HuBERT Layer 12 (codebook size 2048).
      - Frame-level PCA projection (768 -> 512) -> oracle segment averaging ->
        unit-sphere (L2) normalization -> Spectral Clustering at monophone scale (K_base = 40).
      - Compositional Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Embeddings: Skip-Gram Word2Vec (dim=128, win=2, epochs=20) + All-but-the-Top (default D=2).
      - Alignment: GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes + CSLS MNN refinement.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.1_geowarp_spectral_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()

    k_base = int(base_cluster_count)
    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]
    if d_values is None:
        d_values = [2]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.1_geowarp_spectral_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.1_geowarp_spectral_{spin_tag}_base{k_base}_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.5",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="geowarp",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                    clustering_method="spectral",
                    spectral_n_neighbors=spectral_n_neighbors,
                    spectral_sample_size=spectral_sample_size,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"geowarp_{metric}"))
                    or alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gw = res_entry.get("align_geowarp_job") or res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gw.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                # Register postprocessed embeddings & GeoWarP alignment outputs
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/dict.json", align_gw.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/report.json", align_gw.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_star.npy", align_gw.out_mapping_matrix_npy)
                if hasattr(align_gw, "out_init_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_init.npy", align_gw.out_init_mapping_matrix_npy)
                if hasattr(align_gw, "out_wp_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_wp.npy", align_gw.out_wp_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/mapped_audio_embeddings.npy", align_gw.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/geowarp_training_curves.png", align_gw.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spectral_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spectral_spin_{spin_k_int}_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.1_geowarp_spectral Spin-{spin_k_int} L{audio_layer} Audio N-Grams {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gw = res_entry.get("analysis_geowarp_job") or res_entry.get("analysis_gan_job")
                if analysis_gw is None:
                    analysis_gw = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gw.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gw.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gw.out_alignment_report,
                        title_prefix=f"v5.9.1_geowarp_spectral (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        _version=4,
                    )

                analysis_gw.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gw.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gw.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gw.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gw.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gw.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gw.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gw.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gw.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gw.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gw.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Analysis
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.9.1_geowarp_spectral (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int})",
                        trg_label=f"Text {code_type.capitalize()}s",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    return alignment_results


def eval_v5_9_1_geowarp_audio_word_boundaries(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 40,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    embedding_model: str = "word2vec",
    word_boundary_token: str = "<w>",
):
    """
    Evaluates v5.9.1_geowarp_audio_word_boundaries: Spin-HuBERT Compositional Audio N-Grams with Audio Word Boundaries (<w>)
    and All-but-the-Top Isotropy, aligned via GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes (arXiv:2610.09411).
      - Uses GMM oracle cheating segmentation to pool Spin-HuBERT Layer 12 frame representations.
      - Aligns word boundaries using transcript lexicon DP and inserts `<w>` tokens between words.
      - Skip-Gram Word2Vec (dim=128, win=2) with All-but-the-Top (D=2) aligned via GeoWarP + Wasserstein Procrustes + CSLS.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.9.1_geowarp_spin_audio_word_boundaries_base{k_base}_{norm_clean}_pca{pca_dim}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()

    k_base = int(base_cluster_count)
    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]
    if d_values is None:
        d_values = [2]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.9.1_geowarp_spin_audio_word_boundaries_base{k_base}_{norm_clean}_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    for spin_k in spin_codebook_sizes:
        spin_k_int = int(spin_k)
        spin_tag = f"spin_{spin_k_int}"

        for d_rem in d_values:
            d_rem_int = int(d_rem)
            d_tag = f"D{d_rem_int}"
            d_prefix = f"{prefix_base}/{spin_tag}/{d_tag}"

            if alignment_results_by_d is not None and (spin_k_int, d_rem_int) in alignment_results_by_d:
                alignment_results = alignment_results_by_d[(spin_k_int, d_rem_int)]
            elif alignment_results_by_d is not None and d_rem_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[d_rem_int]
            elif alignment_results_by_d is not None and spin_k_int in alignment_results_by_d:
                alignment_results = alignment_results_by_d[spin_k_int]
            else:
                alignment_results = build_self_learning_asr_alignment_pipeline(
                    librispeech_key="train-clean-100",
                    audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                    audio_layer=audio_layer,
                    word2vec_dims=[dim],
                    word2vec_windows=[win],
                    word2vec_epochs=20,
                    cluster_counts=[k_base],
                    coverage_threshold=0.80,
                    code_types=["diphone", "triphone"],
                    code_cluster_map={
                        "diphone": [(k_base, f"clus_{k_base}_2gram")],
                        "triphone": [(k_base, f"clus_{k_base}_3gram")],
                    },
                    segmentation_type="cheating",
                    pca_dim=pca_dim,
                    version=f"v5.9.1_geowarp_spin_audio_word_boundaries_{spin_tag}_base{k_base}_{norm_clean}_isotropic_{d_tag}",
                    text_version="v5.9.1_spin_audio_word_boundaries",
                    text_model_tag="hubert_l6",
                    normalize_segments=norm_clean,
                    retrieval_metrics=[metric],
                    csls_k=10,
                    alignment_method="geowarp",
                    embedding_model=embedding_model,
                    word2vec_postprocessing="all_but_the_top",
                    all_but_the_top_D=d_rem_int,
                    run_isotropy_analysis=True,
                    spin_codebook_size=spin_k_int,
                    use_audio_ngrams=True,
                    audio_word_boundaries=True,
                    word_boundary_token=word_boundary_token,
                )

            cluster_jobs_by_name = {}

            for code_type, clus_str in target_setups:
                exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                res_entry = (
                    alignment_results.get((code_type, clus_str, dim, win, f"geowarp_{metric}"))
                    or alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                    if alignment_results is not None
                    else None
                )
                if res_entry is None:
                    continue

                a_clus = res_entry["cluster_job"]
                a_w2v = res_entry["audio_w2v_job"]
                t_w2v = res_entry["text_w2v_job"]
                raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                align_gw = res_entry.get("align_geowarp_job") or res_entry["align_gan_job"]

                cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                audio_emb_path = align_gw.out_mapped_audio_emb_npy
                text_emb_path = t_w2v.out_embeddings_npy
                vocab_path = t_w2v.out_vocab_txt
                cluster_jobs_by_name[clus_str] = a_clus

                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings.npy", a_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings.npy", t_w2v.out_embeddings_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                if hasattr(a_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                if hasattr(t_w2v, "out_isotropy_json"):
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/dict.json", align_gw.out_dictionary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/report.json", align_gw.out_alignment_report)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_star.npy", align_gw.out_mapping_matrix_npy)
                if hasattr(align_gw, "out_init_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_init.npy", align_gw.out_init_mapping_matrix_npy)
                if hasattr(align_gw, "out_wp_mapping_matrix_npy"):
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_wp.npy", align_gw.out_wp_mapping_matrix_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/mapped_audio_embeddings.npy", align_gw.out_mapped_audio_emb_npy)
                tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/geowarp_training_curves.png", align_gw.out_gan_loss_plot)

                # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                    name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spin_wb_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                # 2. 4-Gram Language Model Rescoring Job
                rescore_train_job = PhonemeNgramRescoreJob(
                    hypotheses_file=eval_train_job.out_hypotheses_txt,
                    reference_file=eval_train_job.out_reference_txt,
                    text_corpus_file=text_corpus_path,
                    lm_order=4,
                    backoff_alpha=0.4,
                    lm_weight=0.5,
                    match_bonus=2.0,
                    beam_width=10,
                    collapse_before=True,
                    name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.9.1_geowarp_spin_wb_{norm_clean}_{d_tag}_pca{pca_dim}",
                )
                rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                # 3. Joined Score Results
                join_asr_scores = JoinCheatingSegScoresJob(
                    scores={
                        "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                        "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                    },
                    title=f"Cheating Segmentation Evaluation (v5.9.1_geowarp Spin Audio Word Boundaries {d_tag}, {code_type}, {clus_str})",
                )
                join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                # 4. Cross-Modal 2D PCA & Representation Analysis
                analysis_gw = res_entry.get("analysis_geowarp_job") or res_entry.get("analysis_gan_job")
                if analysis_gw is None:
                    analysis_gw = EmbeddingAlignmentAnalysisJob(
                        mapped_audio_embeddings_npy=align_gw.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        dictionary_json=align_gw.out_dictionary_json,
                        raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                        raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                        audio_vocab_file=a_clus.out_cluster_vocab_txt,
                        text_vocab_file=t_w2v.out_vocab_txt,
                        audio_counts_file=a_clus.out_cluster_counts_txt,
                        text_counts_file=t_w2v.out_counts_txt,
                        alignment_report_json=align_gw.out_alignment_report,
                        title_prefix=f"v5.9.1_geowarp_wb (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int} <w>)",
                        trg_label=f"Text {code_type.capitalize()}s (<w>)",
                        _version=4,
                    )

                analysis_gw.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gw.out_cross_modal_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gw.out_cross_modal_pca_json)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gw.out_source_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gw.out_target_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gw.out_cross_modal_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gw.out_source_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gw.out_target_raw_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gw.out_unaligned_joint_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gw.out_alignment_comparison_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gw.out_analysis_report_json)

                # 5. Word2Vec Isotropy & PCA Analysis
                w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                if w2v_isotropy_job is None:
                    w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                        src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                        trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                        src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                        trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                        src_vocab_file=a_clus.out_cluster_vocab_txt,
                        trg_vocab_file=t_w2v.out_vocab_txt,
                        src_counts_file=a_clus.out_cluster_counts_txt,
                        trg_counts_file=t_w2v.out_counts_txt,
                        d_remove=d_rem_int,
                        title_prefix=f"v5.9.1_geowarp_wb (Spin-{spin_k_int} L{audio_layer} N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                        src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int} <w>)",
                        trg_label=f"Text {code_type.capitalize()}s (<w>)",
                        is_postprocessed_active=True,
                    )

                w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

    return alignment_results


def eval_v5_10_geowarp_spin_audio_ngrams(
    cluster: Optional[str] = None,
    pca_dim: int = 512,
    run_analysis: bool = True,
    alignment_results_by_pool_d: Optional[dict] = None,
    norm_tag: str = "unit_sphere",
    base_cluster_count: int = 512,
    spin_codebook_sizes: Optional[List[int]] = None,
    spin_codebook_size: Optional[int] = None,
    audio_layer: int = 12,
    d_values: Optional[List[int]] = None,
    pooling_methods: Optional[List[str]] = None,
    embedding_model: str = "word2vec",
):
    """
    Evaluates v5.10_geowarp (K_base = 512 with mean and center_weighted segment pooling ablations),
    copying the best v5.9.1_geowarp configuration:
      - Acoustic Backbone: Spin-HuBERT Layer 12 (codebook size 2048).
      - Frame-level PCA projection (768 -> 512) -> oracle segment pooling (mean vs center_weighted) ->
        unit-sphere (L2) normalization -> K-Means at fine scale (K_base = 512).
      - Compositional Audio N-Grams: 2-grams (diphones) and 3-grams (triphones) with 80% coverage pruning.
      - Embeddings: Skip-Gram Word2Vec (dim=128, win=2, epochs=20) + All-but-the-Top (default D=2).
      - Alignment: GeoWarP Coarse Geometric Initialization + Wasserstein Procrustes + CSLS MNN refinement.
      - Output prefix: output/.../cross_modal_cheat_seg_{cluster}/v5.10_geowarp_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}/{pooling_method}/spin_{spin_k}/D{D}/...
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()
    norm_clean = norm_tag.strip().lower()
    norm_title = "Unit Sphere" if norm_clean in ("unit_sphere", "sphere", "l2") else norm_tag

    k_base = int(base_cluster_count)
    if spin_codebook_sizes is None:
        spin_codebook_sizes = [spin_codebook_size] if spin_codebook_size is not None else [2048]
    if d_values is None:
        d_values = [2]
    if pooling_methods is None:
        pooling_methods = ["mean", "center_weighted"]

    prefix_base = (
        f"{__setup_base_name__}/librispeech/cross_modal_cheat_seg_{active_cluster}/"
        f"v5.10_geowarp_spin_audio_ngrams_base{k_base}_{norm_clean}_pca{pca_dim}"
    )

    oracle_ref_hdfs = get_cheating_phoneme_ref_hdfs()
    oracle_seq_tags = get_hubert_seq_tags_path()
    oracle_idx_to_phoneme = get_oracle_idx_to_phoneme_path()
    text_corpus_path = get_phoneme_text_corpus_path(cluster=active_cluster)

    target_setups = [
        ("diphone", f"clus_{k_base}_2gram"),
        ("triphone", f"clus_{k_base}_3gram"),
    ]
    dim = 128
    win = 2
    metric = "csls"
    dw_tag = f"dim{dim}_win{win}"

    all_results = {}

    for p_method in pooling_methods:
        p_clean = p_method.strip().lower()
        for spin_k in spin_codebook_sizes:
            spin_k_int = int(spin_k)
            spin_tag = f"spin_{spin_k_int}"

            for d_rem in d_values:
                d_rem_int = int(d_rem)
                d_tag = f"D{d_rem_int}"
                d_prefix = f"{prefix_base}/{p_clean}/{spin_tag}/{d_tag}"

                if alignment_results_by_pool_d is not None and (p_clean, spin_k_int, d_rem_int) in alignment_results_by_pool_d:
                    alignment_results = alignment_results_by_pool_d[(p_clean, spin_k_int, d_rem_int)]
                elif alignment_results_by_pool_d is not None and (p_clean, d_rem_int) in alignment_results_by_pool_d:
                    alignment_results = alignment_results_by_pool_d[(p_clean, d_rem_int)]
                elif alignment_results_by_pool_d is not None and p_clean in alignment_results_by_pool_d:
                    alignment_results = alignment_results_by_pool_d[p_clean]
                else:
                    alignment_results = build_self_learning_asr_alignment_pipeline(
                        librispeech_key="train-clean-100",
                        audio_model_name="/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
                        audio_layer=audio_layer,
                        word2vec_dims=[dim],
                        word2vec_windows=[win],
                        word2vec_epochs=20,
                        cluster_counts=[k_base],
                        coverage_threshold=0.80,
                        code_types=["diphone", "triphone"],
                        code_cluster_map={
                            "diphone": [(k_base, f"clus_{k_base}_2gram")],
                            "triphone": [(k_base, f"clus_{k_base}_3gram")],
                        },
                        segmentation_type="cheating",
                        pca_dim=pca_dim,
                        version=f"v5.10_geowarp_{spin_tag}_base{k_base}_{p_clean}_{norm_clean}_isotropic_{d_tag}",
                        text_version="v5.5",
                        text_model_tag="hubert_l6",
                        normalize_segments=norm_clean,
                        retrieval_metrics=[metric],
                        csls_k=10,
                        alignment_method="geowarp",
                        embedding_model=embedding_model,
                        word2vec_postprocessing="all_but_the_top",
                        all_but_the_top_D=d_rem_int,
                        run_isotropy_analysis=True,
                        spin_codebook_size=spin_k_int,
                        use_audio_ngrams=True,
                        pooling_method=p_clean,
                    )

                all_results[(p_clean, spin_k_int, d_rem_int)] = alignment_results
                cluster_jobs_by_name = {}

                for code_type, clus_str in target_setups:
                    exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                    res_entry = (
                        alignment_results.get((code_type, clus_str, dim, win, f"geowarp_{metric}"))
                        or alignment_results.get((code_type, clus_str, dim, win, f"gan_{metric}"))
                        if alignment_results is not None
                        else None
                    )
                    if res_entry is None:
                        continue

                    a_clus = res_entry["cluster_job"]
                    a_w2v = res_entry["audio_w2v_job"]
                    t_w2v = res_entry["text_w2v_job"]
                    raw_a_w2v = res_entry.get("raw_audio_w2v_job", a_w2v)
                    raw_t_w2v = res_entry.get("raw_text_w2v_job", t_w2v)
                    align_gw = res_entry.get("align_geowarp_job") or res_entry["align_gan_job"]

                    cheat_seg_tokens_file = a_clus.out_collapsed_tokens_txt
                    audio_emb_path = align_gw.out_mapped_audio_emb_npy
                    text_emb_path = t_w2v.out_embeddings_npy
                    vocab_path = t_w2v.out_vocab_txt
                    cluster_jobs_by_name[clus_str] = a_clus

                    # Register postprocessed embeddings & GeoWarP alignment outputs
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_embeddings_abtt.npy", a_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_embeddings_abtt.npy", t_w2v.out_embeddings_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_vocab.txt", a_clus.out_cluster_vocab_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_ngram_counts.txt", a_clus.out_ngram_counts_txt)
                    if hasattr(a_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/audio_isotropy_stats.json", a_w2v.out_isotropy_json)
                    if hasattr(t_w2v, "out_isotropy_json"):
                        tk.register_output(f"{d_prefix}/{exp_key}/embeddings/text_isotropy_stats.json", t_w2v.out_isotropy_json)

                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/dict.json", align_gw.out_dictionary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/report.json", align_gw.out_alignment_report)
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_star.npy", align_gw.out_mapping_matrix_npy)
                    if hasattr(align_gw, "out_init_mapping_matrix_npy"):
                        tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_init.npy", align_gw.out_init_mapping_matrix_npy)
                    if hasattr(align_gw, "out_wp_mapping_matrix_npy"):
                        tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/W_wp.npy", align_gw.out_wp_mapping_matrix_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/mapped_audio_embeddings.npy", align_gw.out_mapped_audio_emb_npy)
                    tk.register_output(f"{d_prefix}/{exp_key}/geowarp_alignment/geowarp_training_curves.png", align_gw.out_gan_loss_plot)

                    # 1. Cheating Segmentation Nearest-Neighbor CSLS Evaluation
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
                        name_prefix=f"eval_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.10_geowarp_spin_{spin_k_int}_base{k_base}_{p_clean}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    eval_train_job.add_alias(f"{d_prefix}/{exp_key}/eval_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_summary.json", eval_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_per_report.txt", eval_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_hypotheses.txt", eval_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_reference.txt", eval_train_job.out_reference_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_sclite.pra", eval_train_job.out_pra_report_txt)

                    # 2. 4-Gram Language Model Rescoring Job
                    rescore_train_job = PhonemeNgramRescoreJob(
                        hypotheses_file=eval_train_job.out_hypotheses_txt,
                        reference_file=eval_train_job.out_reference_txt,
                        text_corpus_file=text_corpus_path,
                        lm_order=4,
                        backoff_alpha=0.4,
                        lm_weight=0.5,
                        match_bonus=2.0,
                        beam_width=10,
                        collapse_before=True,
                        name_prefix=f"rescore_train3k_{code_type}_{clus_str}_{dw_tag}_{metric}_v5.10_geowarp_spin_{spin_k_int}_base{k_base}_{p_clean}_{norm_clean}_{d_tag}_pca{pca_dim}",
                    )
                    rescore_train_job.add_alias(f"{d_prefix}/{exp_key}/rescore_4gram_train3k")
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_hypotheses.txt", rescore_train_job.out_hypotheses_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_summary.json", rescore_train_job.out_per_summary_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_per_report.txt", rescore_train_job.out_per_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/train3k_4gram_sclite.pra", rescore_train_job.out_pra_report_txt)

                    # 3. Joined Score Results
                    join_asr_scores = JoinCheatingSegScoresJob(
                        scores={
                            "nearest_neighbor_csls_per": eval_train_job.out_per_summary_json,
                            "lm_rescore_4gram_per": rescore_train_job.out_per_summary_json,
                        },
                        title=f"Cheating Segmentation Evaluation (v5.10_geowarp Spin-{spin_k_int} L{audio_layer} Base-{k_base} [{p_clean}] Audio N-Grams {d_tag}, {code_type}, {clus_str})",
                    )
                    join_asr_scores.add_alias(f"{d_prefix}/{exp_key}/score_results")
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.txt", join_asr_scores.out_summary_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/score_results.json", join_asr_scores.out_summary_json)

                    # 4. Cross-Modal 2D PCA & Representation Analysis
                    analysis_gw = res_entry.get("analysis_geowarp_job") or res_entry.get("analysis_gan_job")
                    if analysis_gw is None:
                        analysis_gw = EmbeddingAlignmentAnalysisJob(
                            mapped_audio_embeddings_npy=align_gw.out_mapped_audio_emb_npy,
                            text_embeddings_npy=t_w2v.out_embeddings_npy,
                            dictionary_json=align_gw.out_dictionary_json,
                            raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                            raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=t_w2v.out_vocab_txt,
                            audio_counts_file=a_clus.out_cluster_counts_txt,
                            text_counts_file=t_w2v.out_counts_txt,
                            alignment_report_json=align_gw.out_alignment_report,
                            title_prefix=f"v5.10_geowarp (Spin-{spin_k_int} L{audio_layer} Base-{k_base} [{p_clean}] N-Grams {d_tag}) Cross-Modal 2D PCA ({code_type}, {clus_str}, {dw_tag}, {metric})",
                            src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int} K={k_base} {p_clean})",
                            trg_label=f"Text {code_type.capitalize()}s",
                            _version=4,
                        )

                    analysis_gw.add_alias(f"{d_prefix}/{exp_key}/pca_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.png", analysis_gw.out_cross_modal_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca.json", analysis_gw.out_cross_modal_pca_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_audio_pca.png", analysis_gw.out_source_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_text_pca.png", analysis_gw.out_target_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/cross_modal_pca_panel.png", analysis_gw.out_cross_modal_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/source_raw_pca.png", analysis_gw.out_source_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/target_raw_pca.png", analysis_gw.out_target_raw_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/unaligned_cross_modal_pca.png", analysis_gw.out_unaligned_joint_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/alignment_comparison_panel.png", analysis_gw.out_alignment_comparison_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/pca_analysis/analysis_report.json", analysis_gw.out_analysis_report_json)

                    # 5. Word2Vec Isotropy & PCA Analysis
                    w2v_isotropy_job = res_entry.get("w2v_isotropy_job")
                    if w2v_isotropy_job is None:
                        w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                            src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                            trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                            src_post_embeddings_npy=a_w2v.out_embeddings_npy,
                            trg_post_embeddings_npy=t_w2v.out_embeddings_npy,
                            src_vocab_file=a_clus.out_cluster_vocab_txt,
                            trg_vocab_file=t_w2v.out_vocab_txt,
                            src_counts_file=a_clus.out_cluster_counts_txt,
                            trg_counts_file=t_w2v.out_counts_txt,
                            d_remove=d_rem_int,
                            title_prefix=f"v5.10_geowarp (Spin-{spin_k_int} L{audio_layer} Base-{k_base} [{p_clean}] N-Grams {d_tag}) Embedding Isotropy & PCA ({code_type}, {clus_str}, {dw_tag})",
                            src_label=f"Audio {code_type.capitalize()}s (Spin-{spin_k_int} K={k_base} {p_clean})",
                            trg_label=f"Text {code_type.capitalize()}s",
                            is_postprocessed_active=True,
                        )

                    w2v_isotropy_job.add_alias(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis")
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_report.txt", w2v_isotropy_job.out_isotropy_report_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png", w2v_isotropy_job.out_src_before_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png", w2v_isotropy_job.out_src_after_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_before_abtt.png", w2v_isotropy_job.out_trg_before_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/target_text_pca_after_abtt.png", w2v_isotropy_job.out_trg_after_pca_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png", w2v_isotropy_job.out_before_after_panel_plot)
                    tk.register_output(f"{d_prefix}/{exp_key}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

                    # 6. Top 100 Cluster N-Grams and Post-Alignment Mapping Analysis
                    top_ngrams_job = TopClusterNgramsAnalysisJob(
                        cluster_vocab_file=a_clus.out_cluster_vocab_txt,
                        cluster_counts_file=a_clus.out_ngram_counts_txt,
                        dictionary_json=align_gw.out_dictionary_json,
                        mapped_audio_embeddings_npy=align_gw.out_mapped_audio_emb_npy,
                        text_embeddings_npy=t_w2v.out_embeddings_npy,
                        text_vocab_file=vocab_path,
                        top_k=100,
                        title_prefix=f"v5.10_geowarp ({spin_tag}, Base-{k_base}, {p_clean}, D={d_rem_int}, {code_type}, {clus_str}, {norm_title})",
                        name_prefix=f"top_cluster_ngrams_v5.10_geowarp_{spin_tag}_base{k_base}_{p_clean}_{d_tag}_{code_type}_{clus_str}",
                    )
                    top_ngrams_job.add_alias(f"{d_prefix}/{exp_key}/top_100_ngrams")
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.txt", top_ngrams_job.out_top_ngrams_txt)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.json", top_ngrams_job.out_top_ngrams_json)
                    tk.register_output(f"{d_prefix}/{exp_key}/top_100_cluster_ngrams.png", top_ngrams_job.out_top_ngrams_plot)

                if run_analysis:
                    for code_type, clus_str in target_setups:
                        exp_key = f"{code_type}/{clus_str}/{dw_tag}_{metric}"
                        a_clus = cluster_jobs_by_name.get(clus_str)
                        if a_clus is None:
                            continue
                        analysis_prefix = f"{d_prefix}/{exp_key}/analysis"
                        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
                            cluster_counts_file=a_clus.out_ngram_counts_txt,
                            raw_clusters_file=a_clus.out_ngram_text,
                            cluster_hdfs=None,
                            k_clusters=None,
                            top_k=50,
                            title_prefix=f"v5.10_geowarp Compositional Audio {code_type.capitalize()}s ({spin_tag}, Base-{k_base}, {p_clean}, D={d_rem_int}, {clus_str}, {norm_title})",
                            name_prefix=f"speech_cluster_dist_v5.10_geowarp_{spin_tag}_base{k_base}_{p_clean}_{d_tag}_{norm_clean}_pca{pca_dim}_{clus_str}",
                        )
                        cluster_analysis_job.add_alias(f"{analysis_prefix}/speech_clusters/{clus_str}")
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_distribution.json", cluster_analysis_job.out_summary_json)
                        tk.register_output(f"{analysis_prefix}/speech_clusters/{clus_str}/cluster_report.txt", cluster_analysis_job.out_report_txt)

    return all_results

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
    # Evaluates discovered clusters across both monophone (v2) and multi-token (v3/v5/v5.3) sets
    cluster_configs = [
        ("1P_40", 40, "v2 Monophone 40 Clusters"),
        ("2P_80", 80, "v2 Monophone 80 Clusters"),
        ("4P_160", 160, "v2 Monophone 160 Clusters"),
        ("200_200", 200, "v2/v3 200 Clusters"),
        ("250_250", 250, "v3 250 Clusters"),
        ("500_500", 500, "v3/v5.3 500 Clusters"),
        ("483_483", 483, "v5/v5.3 483 Clusters"),
        ("6161_6161", 6161, "v5/v5.3 6161 Clusters"),
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


# =============================================================================
# Main Entry Point
# =============================================================================

def py(cluster: Optional[str] = None, version: str = all, run_analysis: bool = True):
    """
    Runs Cheating Segmentation Nearest Neighbor evaluation pipeline and corpus distribution analysis.
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
    elif v_norm in ("v5", "v5_pca", "v5_matched_90pct", "v5_matched_90", "v5_90"):
        eval_v5_matched_90pct(cluster=cluster)
    elif v_norm in ("v5.3", "v5_3", "v5.3_gan", "v5_3_gan", "gan"):
        eval_v5_3_gan_pca512(cluster=cluster, run_analysis=run_analysis)
        return
    elif v_norm in ("v5.4_unit_sphere", "v5.4_sphere", "v5.4_l2", "v5.4", "v5_4", "v5.4_gan"):
        eval_v5_4_gan_pca512(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.4_zscore", "v5.4_znorm", "v5.4_z_transform", "v5.4_ztransform", "v5_4_zscore", "v5.4_z"):
        eval_v5_4_gan_pca512(cluster=cluster, run_analysis=run_analysis, norm_tag="zscore")
        return
    elif v_norm in ("v5.5", "v5_5", "v5.5_unit_sphere", "v5.5_gan", "v5.5_sphere"):
        eval_v5_5_gan_pca512(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.6", "v5_6", "v5.6_unit_sphere", "v5.6_gan", "v5.6_sphere", "v5.6_ssg", "v5.6_unit_sphere_gan_pca512"):
        eval_v5_6_gan_pca512(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.6_isotropic", "v5_6_isotropic", "v5.6_abtt", "v5_6_abtt", "isotropic"):
        eval_v5_6_isotropic(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere", d_values=[1, 2, 3])
        return
    elif v_norm in ("v5.6_both", "v5_6_both", "v5.6_all"):
        eval_v5_6_gan_pca512(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere")
        eval_v5_6_isotropic(cluster=cluster, run_analysis=run_analysis, norm_tag="unit_sphere", d_values=[1, 2, 3])
        return
    elif v_norm in ("v5.6_triphone_sweep", "v5_6_triphone_sweep", "triphone_sweep", "sweep", "asymmetric_sweep"):
        eval_v5_6_triphone_asymmetric_sweep(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.7", "v5_7", "v5.7_spin", "v5_7_spin", "spin"):
        eval_v5_7_spin_gan_pca512(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.8", "v5_8", "v5.8_spin_isotropic", "v5_8_spin_isotropic", "spin_isotropic"):
        eval_v5_8_spin_isotropic(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9", "v5_9", "v5.9_audio_ngrams", "v5_9_audio_ngrams", "audio_ngrams"):
        eval_v5_9_audio_ngrams(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9_audio_word_boundaries", "v5_9_audio_word_boundaries", "5.9_audio_word_boundaries", "v5.9_wb"):
        eval_v5_9_audio_word_boundaries(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.1", "v5_9_1", "v5.9.1_spin_audio_ngrams", "v5.9.1_spin_ngrams"):
        eval_v5_9_1_spin_audio_ngrams_isotropic(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.1_audio_word_boundaries", "v5_9_1_audio_word_boundaries", "5.9.1_audio_word_boundaries", "v5.9.1_spin_audio_word_boundaries", "v5_9_1_spin_audio_word_boundaries", "v5.9.1_wb"):
        eval_v5_9_1_spin_audio_word_boundaries(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.1_geowarp", "v5_9_1_geowarp", "5.9.1_geowarp", "geowarp"):
        eval_v5_9_1_geowarp(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.1_geowarp_spectral", "v5_9_1_geowarp_spectral", "5.9.1_geowarp_spectral", "v5.9.1_spectral", "v5_9_1_spectral", "spectral", "geowarp_spectral"):
        eval_v5_9_1_geowarp_spectral(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.1_geowarp_audio_word_boundaries", "v5_9_1_geowarp_audio_word_boundaries", "5.9.1_geowarp_audio_word_boundaries", "v5.9.1_geowarp_wb", "v5_9_1_geowarp_wb"):
        eval_v5_9_1_geowarp_audio_word_boundaries(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.2", "v5_9_2", "v5.9.2_wo_sil", "spin_audio_ngrams_wo_sil", "v5.9.2_spin_audio_ngrams"):
        eval_v5_9_2_wo_sil_spin_audio_ngrams_isotropic(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.2_hubert_l6", "v5_9_2_hubert_l6", "v5.9.2_l6", "hubert_l6_abtt"):
        eval_v5_9_2_wo_sil_hubert_l6_audio_ngrams_isotropic(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.3", "v5_9_3", "v5.9.3_wo_sil", "audio_ngrams_wo_sil"):
        eval_v5_9_3_wo_sil_audio_ngrams(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4", "v5_9_4", "v5.9.4_no_collapse", "no_collapse", "v5.9.4_wo_sil_no_collapse", "v5.9.4_spin"):
        eval_v5_9_4_wo_sil_spin_audio_ngrams_no_collapse(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4_hubert_l6", "v5_9_4_hubert_l6", "v5.9.4_l6", "v5_9_4_l6"):
        eval_v5_9_4_wo_sil_hubert_l6_audio_ngrams_no_collapse(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4.1", "v5_9_4_1", "v5.9.4_1", "v5_9_4_masked_csls", "masked_csls", "v5.9.4.1_spin"):
        eval_v5_9_4_1_wo_sil_spin_audio_ngrams_masked_csls(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4.1_hubert_l6", "v5_9_4_1_hubert_l6", "v5.9.4.1_l6"):
        eval_v5_9_4_1_wo_sil_hubert_l6_audio_ngrams_masked_csls(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4.2", "v5_9_4_2", "v5.9.4_2", "v5_9_4_struct_reg", "struct_reg_gan", "reg_gan", "v5.9.4.2_spin"):
        eval_v5_9_4_2_wo_sil_spin_audio_ngrams_struct_reg_gan(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.4.2_hubert_l6", "v5_9_4_2_hubert_l6", "v5.9.4.2_l6"):
        eval_v5_9_4_2_wo_sil_hubert_l6_audio_ngrams_struct_reg_gan(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.10", "v5_10", "v5.10_base120", "v5_10_base120", "approach_a"):
        eval_v5_10_wo_sil_audio_ngrams_no_collapse(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.10.1", "v5_10_1", "v5.10_1", "v5.10.1_masked_csls", "v5_10_1_masked_csls"):
        eval_v5_10_1_wo_sil_audio_ngrams_masked_csls(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.10.2", "v5_10_2", "v5.10_2", "v5.10.2_struct_reg", "v5_10_2_struct_reg_gan", "v5.10.2_reg_gan"):
        eval_v5_10_2_wo_sil_audio_ngrams_struct_reg_gan(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.10_geowarp", "v5_10_geowarp", "v5.10_base512", "v5_10_base512", "v5.10_k512", "v5_10_k512"):
        eval_v5_10_geowarp_spin_audio_ngrams(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere", base_cluster_count=512)
        return
    elif v_norm in ("v5.9.5", "v5_9_5", "v5.9.5_viterbi", "viterbi_regularized"):
        eval_v5_9_5_wo_sil_spin_audio_ngrams_viterbi_regularized(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.5_copkmeans", "v5_9_5_copkmeans", "copkmeans"):
        eval_v5_9_5_copkmeans_wo_sil_spin_audio_ngrams(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5.9.5_pckmeans", "v5_9_5_pckmeans", "pckmeans"):
        eval_v5_9_5_pckmeans_wo_sil_spin_audio_ngrams(cluster=cluster, pca_dim=512, run_analysis=run_analysis, norm_tag="unit_sphere")
        return
    elif v_norm in ("v5_w2v2", "v5_w2v2_matched_90pct", "v5_w2v2_matched_90", "v5_w2v2_90"):
        eval_v5_w2v2_matched_90pct(cluster=cluster)
    elif v_norm in ("v2_v3", "v2+v3", "new"):
        eval_v2(cluster=cluster)
        eval_v3(cluster=cluster)
    elif v_norm == "all":
        eval_v1(cluster=cluster)
        eval_v2(cluster=cluster)
        eval_v3(cluster=cluster)
        eval_v3_matched_90pct(cluster=cluster)
        eval_v4_matched_90pct(cluster=cluster)
        eval_v5_matched_90pct(cluster=cluster)
        eval_v5_3_gan_pca512(cluster=cluster, run_analysis=False)
        eval_v5_4_gan_pca512(cluster=cluster, run_analysis=False, norm_tag="unit_sphere")
        eval_v5_4_gan_pca512(cluster=cluster, run_analysis=False, norm_tag="zscore")
        eval_v5_5_gan_pca512(cluster=cluster, run_analysis=False, norm_tag="unit_sphere")
        eval_v5_6_gan_pca512(cluster=cluster, run_analysis=False, norm_tag="unit_sphere")
        eval_v5_6_isotropic(cluster=cluster, run_analysis=False, norm_tag="unit_sphere", d_values=[1, 2, 3])
        eval_v5_w2v2_matched_90pct(cluster=cluster)
    elif v_norm in ("analysis", "dist", "distribution"):
        run_distribution_analysis(cluster=cluster)
        return
    else:
        raise ValueError(f"Unknown version: {version}")

    if run_analysis:
        run_distribution_analysis(cluster=cluster)
