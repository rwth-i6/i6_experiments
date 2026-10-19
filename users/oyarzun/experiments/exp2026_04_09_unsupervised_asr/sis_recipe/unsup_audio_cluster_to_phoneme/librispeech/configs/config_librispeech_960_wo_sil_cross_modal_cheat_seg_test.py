"""
Test Config: Single GPU Debug Evaluation for New Cross-Modal Codes with Cheating Segmentation
and Corpus Distribution Analysis (Text Bigrams/Trigrams & Discovered Speech Clusters).

Designed for interactive debug sessions (1 GPU) to test the nearest-neighbor
cheating segmentation evaluation on a single new ablation (default: v3 diphone, K=200, win=3),
along with text N-gram frequency histograms and speech cluster frame assignment distributions.
"""

import os
from typing import Optional
from sisyphus import tk

from ....cluster_adapter import (
    get_cheating_segmentation_path,
    get_hubert_cluster_hdfs,
    get_phoneme_vocab_path,
    get_v3_audio_embeddings_path,
    get_v3_text_embeddings_path,
    get_v3_vocab_path,
    get_v2_audio_embeddings_path,
    get_v2_text_embeddings_path,
    get_v2_phoneme_vocab_path,
    get_cluster_counts_path,
    get_raw_clusters_path,
    get_v3_counts_path,
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


def py(
    version: str = "v3",
    code_type: str = "diphone",
    k_clusters: int = 200,
    win: int = 3,
    split: str = "train_other_960",
    max_seqs: Optional[int] = 100,
    cluster: Optional[str] = "curium",
    include_analysis: bool = True,
):
    """
    Constructs a single-GPU test evaluation job for quick debugging.

    :param version: 'v3' (diphone/triphone) or 'v2' (monophone).
    :param code_type: 'diphone' or 'triphone' (for v3).
    :param k_clusters: Cluster count (e.g. 200, 250, 500 for v3; 40, 80, 160, 200 for v2).
    :param win: Context window size (3, 5, 10).
    :param split: 'dev_other' or 'train_other_960'.
    :param max_seqs: Maximum number of utterances to evaluate (e.g. 100 for fast debug, None for full).
    :param cluster: 'curium' for local cluster or 'rwth-hpc' / 'rwth-hcp' for RWTH cluster.
    :param include_analysis: Whether to instantiate text N-gram and speech cluster distribution jobs.
    """
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()

    if version in ("v3_matched_90", "matched_90", "matched"):
        if k_clusters == 200:
            k_clusters = 483 if code_type == "diphone" else 6161
        version = "v3"

    clus_str = f"clus_{k_clusters}_{k_clusters}" if k_clusters in (200, 250, 500, 483, 6161) else f"clus_{k_clusters}"
    prefix_name = f"{__setup_base_name__}/librispeech/debug_cheat_seg_{active_cluster}"

    # 1. Ground-Truth Reference Phonemes
    lexicon_file = text.get_librispeech_lexicon()
    phoneme_vocab = get_phoneme_vocab_path()
    raw_text_data, _ = text.get_text(split.replace("_", "-"))
    phoneme_hdfs, _, _, seq_tags = text.get_phonemized_text(
        split.replace("_", "-"),
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=0.0,
        surround_w_sil=False,
    )

    # 2. Select Ablation Embeddings & Vocabularies
    if version == "v3":
        audio_emb_raw = get_v3_audio_embeddings_path(code_type, clus_str, win=win)
        text_emb_raw = get_v3_text_embeddings_path(code_type, win=win)
        vocab_path = get_v3_vocab_path(code_type, win=win)
        eval_code_type = code_type
        ablation_tag = f"v3_{code_type}_{clus_str}_win{win}"
    else:  # v2
        audio_emb_raw = get_v2_audio_embeddings_path(clus_str)
        text_emb_raw = get_v2_text_embeddings_path()
        vocab_path = get_v2_phoneme_vocab_path()
        eval_code_type = "monophone"
        ablation_tag = f"v2_{clus_str}"

    if not audio_emb_raw or not os.path.exists(audio_emb_raw):
        raise FileNotFoundError(
            f"Audio embeddings not found for {ablation_tag}: path resolved to '{audio_emb_raw}'. "
            f"Ensure the upstream clustering and alignment jobs have completed."
        )
    if not text_emb_raw or not os.path.exists(text_emb_raw):
        raise FileNotFoundError(
            f"Text embeddings not found for {ablation_tag}: path resolved to '{text_emb_raw}'. "
            f"Ensure text embeddings have been generated for {code_type} win={win}."
        )

    audio_emb_path = tk.Path(audio_emb_raw)
    text_emb_path = tk.Path(text_emb_raw)

    # 3. Audio Frame Clusters & Oracle Cheating Segmentation
    cluster_hdfs = get_hubert_cluster_hdfs(cluster_mult=clus_str, k_clusters=k_clusters, split=split)
    seg_path = get_cheating_segmentation_path(k_clusters=k_clusters)

    # 4. Construct Single Debug Evaluation Job
    debug_eval_job = CheatingSegmentationNearestNeighborEvalJob(
        audio_embeddings_npy=audio_emb_path,
        text_embeddings_npy=text_emb_path,
        phoneme_vocab_file=vocab_path,
        cheating_segmentation_path=seg_path,
        audio_cluster_hdfs=cluster_hdfs,
        reference_phonemes=phoneme_hdfs,
        reference_phoneme_vocab_file=phoneme_vocab,
        seq_tags_file=seq_tags,
        code_type=eval_code_type,
        max_seqs=max_seqs,
        ignore_silence=True,
        name_prefix=f"debug_eval_{ablation_tag}_{split}",
    )

    debug_eval_job.add_alias(f"{prefix_name}/{ablation_tag}/eval_{split}")
    tk.register_output(f"{prefix_name}/{ablation_tag}/{split}_per_summary.json", debug_eval_job.out_per_summary_json)
    tk.register_output(f"{prefix_name}/{ablation_tag}/{split}_per_report.txt", debug_eval_job.out_per_report_txt)
    tk.register_output(f"{prefix_name}/{ablation_tag}/{split}_hypotheses.txt", debug_eval_job.out_hypotheses_txt)
    tk.register_output(f"{prefix_name}/{ablation_tag}/{split}_reference.txt", debug_eval_job.out_reference_txt)

    # 5. Corpus Distribution Analysis Jobs
    if include_analysis:
        # A. Text N-gram Frequency Analysis (Diphones and Triphones sorted descending)
        diphone_counts = get_v3_counts_path("diphone")
        triphone_counts = get_v3_counts_path("triphone")
        text_analysis_job = TextNgramDistributionAnalysisJob(
            text_file=raw_text_data if diphone_counts is None else None,
            bigram_counts_file=diphone_counts,
            trigram_counts_file=triphone_counts,
            top_k=50,
            ignore_silence=True,
            title_prefix=f"LibriSpeech Phoneme N-grams (Diphones & Triphones)",
            name_prefix=f"ngram_analysis_{split}",
        )
        text_analysis_job.add_alias(f"{prefix_name}/{ablation_tag}/analysis/text_ngrams")
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/bigram_histogram.png", text_analysis_job.out_bigram_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/trigram_histogram.png", text_analysis_job.out_trigram_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/ngram_combined_histogram.png", text_analysis_job.out_combined_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/ngram_cumulative_coverage.png", text_analysis_job.out_coverage_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/ngram_counts.txt", text_analysis_job.out_counts_txt)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/ngram_distribution.json", text_analysis_job.out_summary_json)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/text/ngram_report.txt", text_analysis_job.out_report_txt)

        # B. Speech Cluster Distribution Analysis (Frame counts assigned per cluster across corpus)
        counts_path = get_cluster_counts_path(clus_str)
        raw_clus_path = get_raw_clusters_path(clus_str)
        cluster_analysis_job = SpeechClusterDistributionAnalysisJob(
            cluster_counts_file=counts_path,
            raw_clusters_file=raw_clus_path,
            cluster_hdfs=cluster_hdfs,
            k_clusters=k_clusters,
            top_k=50,
            title_prefix=f"Discovered Speech Clusters ({clus_str}, K={k_clusters})",
            name_prefix=f"speech_cluster_dist_{clus_str}",
        )
        cluster_analysis_job.add_alias(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters")
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters/cluster_distribution.png", cluster_analysis_job.out_distribution_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters/cluster_cumulative_coverage.png", cluster_analysis_job.out_cumulative_plot)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters/cluster_counts.txt", cluster_analysis_job.out_counts_txt)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters/cluster_distribution.json", cluster_analysis_job.out_summary_json)
        tk.register_output(f"{prefix_name}/{ablation_tag}/analysis/speech_clusters/cluster_report.txt", cluster_analysis_job.out_report_txt)

    return debug_eval_job
