"""
Main experiment pipeline for Self-Learning Embedding Alignment (Artetxe et al. 2017) applied to ASR.
Aligns HuBERT Layer 6 audio cluster Word2Vec embeddings with text Word2Vec embeddings across diphone and triphone codes.
Ablates over cluster counts K in {200, 250, 500} using frequency-based initialization.
"""

from typing import List, Dict, Optional, Tuple
import os
from sisyphus import tk

from .data.librispeech import text as lbs_text
from .data.librispeech import audio as lbs_audio
from .data.hubert import (
    ExtractHuBERTLayerFeaturesJob,
    HuBERTClusterAndCollapseJob,
    HuBERTCheatingSegmentAveragingAndClusterJob,
    HuBERTFramePCAFitJob,
)
from .data.text import GenerateTextNgramsJob
from .data.word2vec import TrainWord2VecJob
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob
from ..models.analysis import EmbeddingAlignmentAnalysisJob
from recipe.i6_experiments.users.oyarzun.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.cluster_adapter import (
    get_cheating_phoneme_ref_hdfs,
)


def _make_dw_tag(dim: int, win: int) -> str:
    """Formats directory / alias tag for dimension and window."""
    if dim == 300:
        return f"win{win}"
    return f"dim{dim}_win{win}"


def build_self_learning_asr_alignment_pipeline(
    librispeech_key: str = "train-clean-100",
    hubert_model_name: Optional[str] = None,
    hubert_layer: Optional[int] = None,
    audio_model_name: str = "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
    audio_layer: int = 6,
    word2vec_dim: int = 300,
    word2vec_dims: Optional[List[int]] = None,
    word2vec_windows: Optional[List[int]] = None,
    word2vec_epochs: int = 20,
    cluster_counts: Optional[List[int]] = None,
    code_types: Optional[List[str]] = None,
    version: str = "v3",
    coverage_threshold: Optional[float] = None,
    max_vocab_size: Optional[int] = None,
    segmentation_type: str = "kmeans",
    pca_dim: Optional[int] = None,
    retrieval_metrics: Optional[List[str]] = None,
    csls_k: int = 10,
    model_tag: Optional[str] = None,
):
    """
    Constructs the end-to-end self-learning embedding alignment ASR pipeline across
    diphone and triphone cross-modal codes, cluster sizes K, dimensions d, windows w,
    and dictionary retrieval metrics (Cosine & CSLS) with frequency-based initialization.
    Supports both HuBERT and Wav2Vec 2.0 backbones.
    """
    if hubert_model_name is not None:
        audio_model_name = hubert_model_name
    if hubert_layer is not None:
        audio_layer = hubert_layer

    if model_tag is None:
        if "wav2vec2" in audio_model_name or "lv60" in audio_model_name:
            model_tag = f"w2v2_large60k_l{audio_layer}"
        elif "hubert" in audio_model_name:
            model_tag = f"hubert_l{audio_layer}"
        else:
            model_tag = f"model_l{audio_layer}"

    if coverage_threshold is not None:
        cov_tag = f"_matched_cov{int(coverage_threshold*100)}"
        if cluster_counts is None:
            if code_types == ["diphone"]:
                cluster_counts = [483]
            elif code_types == ["triphone"]:
                cluster_counts = [6161]
            else:
                cluster_counts = [483, 6161]
    else:
        cov_tag = ""

    if cluster_counts is None:
        cluster_counts = [200, 250, 500]
    if code_types is None:
        code_types = ["diphone", "triphone"]
    if word2vec_dims is None:
        word2vec_dims = [word2vec_dim]
    if word2vec_windows is None:
        word2vec_windows = [1, 2, 3, 5] if segmentation_type not in ("cheating", "cheat_seg") else [1, 2, 3, 5]
    if retrieval_metrics is None:
        retrieval_metrics = ["cosine", "csls"]

    if segmentation_type in ("cheating", "cheat_seg"):
        seg_tag = "_cheat_seg"
    else:
        seg_tag = ""

    pca_tag = f"_pca{pca_dim}" if pca_dim else ""
    ver_suffix = f"_{version}{cov_tag}" if version else cov_tag
    text_alias_prefix = f"experiments/self_learning_emb_asr{ver_suffix}/{librispeech_key}/{model_tag}"
    alias_prefix = f"experiments/self_learning_emb_asr{ver_suffix}{seg_tag}{pca_tag}/{librispeech_key}/{model_tag}"
    audio_shared_prefix = f"experiments/self_learning_emb_asr{seg_tag}{pca_tag}/{librispeech_key}/{model_tag}"

    # -------------------------------------------------------------
    # 1. TEXT PIPELINE: Phonemization, N-grams & Text Word2Vec
    # -------------------------------------------------------------
    phonemized_text_file, phoneme_vocab, lexicon_file, lm_seq_tags = lbs_text.get_phonemized_text_corpus(
        librispeech_key,
    )

    text_pipelines = {}

    for code_type in code_types:
        if code_type == "diphone":
            ngram_job = GenerateTextNgramsJob(
                token_text_file=phonemized_text_file,
                ngram_order=2,
                delimiter="_",
                min_count=1,
                coverage_threshold=coverage_threshold,
                max_vocab_size=max_vocab_size,
            )
            ngram_job.add_alias(f"{text_alias_prefix}/text_diphones")
            tk.register_output(f"{text_alias_prefix}/text/diphones/tokens.txt", ngram_job.out_ngram_text)
            tk.register_output(f"{text_alias_prefix}/text/diphones/vocab.txt", ngram_job.out_vocab_txt)
            token_text_file = ngram_job.out_ngram_text
            fixed_vocab_file = ngram_job.out_vocab_txt

        elif code_type == "triphone":
            ngram_job = GenerateTextNgramsJob(
                token_text_file=phonemized_text_file,
                ngram_order=3,
                delimiter="_",
                min_count=1,
                coverage_threshold=coverage_threshold,
                max_vocab_size=max_vocab_size,
            )
            ngram_job.add_alias(f"{text_alias_prefix}/text_triphones")
            tk.register_output(f"{text_alias_prefix}/text/triphones/tokens.txt", ngram_job.out_ngram_text)
            tk.register_output(f"{text_alias_prefix}/text/triphones/vocab.txt", ngram_job.out_vocab_txt)
            token_text_file = ngram_job.out_ngram_text
            fixed_vocab_file = ngram_job.out_vocab_txt

        else:
            raise ValueError(f"Unknown code_type: {code_type}. Expected diphone, or triphone.")

        for d in word2vec_dims:
            for w in word2vec_windows:
                dw_tag = _make_dw_tag(d, w)
                train_text_w2v_job = TrainWord2VecJob(
                    token_text_file=token_text_file,
                    vector_size=d,
                    window=w,
                    min_count=1,
                    sg=1,
                    epochs=word2vec_epochs,
                    fixed_vocab_file=fixed_vocab_file,
                    _version=2,
                )
                train_text_w2v_job.add_alias(f"{text_alias_prefix}/text_w2v_{code_type}s_{dw_tag}")

                tk.register_output(f"{text_alias_prefix}/text/{code_type}/{dw_tag}/embeddings.npy", train_text_w2v_job.out_embeddings_npy)
                tk.register_output(f"{text_alias_prefix}/text/{code_type}/{dw_tag}/vocab.txt", train_text_w2v_job.out_vocab_txt)
                tk.register_output(f"{text_alias_prefix}/text/{code_type}/{dw_tag}/counts.txt", train_text_w2v_job.out_counts_txt)

                text_pipelines[(code_type, d, w)] = {
                    "w2v_job": train_text_w2v_job,
                    "vocab_file": fixed_vocab_file,
                }

    # -------------------------------------------------------------
    # 2. AUDIO PIPELINE: Feature Extraction & Optional PCA
    # -------------------------------------------------------------
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))

    feat_job = ExtractHuBERTLayerFeaturesJob(
        audio_dir=audio_dir,
        model_name=audio_model_name,
        layer=audio_layer,
    )
    feat_job.add_alias(f"{audio_shared_prefix}/{model_tag}_features")

    # Optional 2b. Frame-level PCA Fitting (before segment pooling / clustering)
    pca_components = None
    pca_mean = None
    if pca_dim is not None:
        pca_fit_job = HuBERTFramePCAFitJob(
            features_npy=feat_job.out_features_npy,
            n_components=pca_dim,
        )
        pca_fit_job.add_alias(f"{audio_shared_prefix}/pca_{pca_dim}")
        tk.register_output(f"{alias_prefix}/pca/components.npy", pca_fit_job.out_components_npy)
        tk.register_output(f"{alias_prefix}/pca/mean.npy", pca_fit_job.out_mean_npy)
        pca_components = pca_fit_job.out_components_npy
        pca_mean = pca_fit_job.out_mean_npy

    # -------------------------------------------------------------
    # 3. CLUSTERING & WORD2VEC: K in cluster_counts
    # -------------------------------------------------------------
    audio_pipelines = {}

    for K in cluster_counts:
        cluster_name = f"clus_{K}_{K}"

        if segmentation_type in ("cheating", "cheat_seg"):
            # 3a. Ablation: Pool audio features over GMM oracle segments and fit K-Means
            cluster_job = HuBERTCheatingSegmentAveragingAndClusterJob(
                features_npy=feat_job.out_features_npy,
                lengths_txt=feat_job.out_lengths_txt,
                seq_tags_txt=feat_job.out_seq_tags_txt,
                segmentation_hdfs=get_cheating_phoneme_ref_hdfs(),
                num_clusters=K,
                pca_components_npy=pca_components,
                pca_mean_npy=pca_mean,
            )
            cluster_job.add_alias(f"{audio_shared_prefix}/{cluster_name}/cheat_seg_average_kmeans")
        else:
            # 3a. Cluster frames and collapse consecutive runs
            cluster_job = HuBERTClusterAndCollapseJob(
                features_npy=feat_job.out_features_npy,
                lengths_txt=feat_job.out_lengths_txt,
                seq_tags_txt=feat_job.out_seq_tags_txt,
                num_clusters=K,
            )
            cluster_job.add_alias(f"{audio_shared_prefix}/{cluster_name}/kmeans_collapse")

        tk.register_output(
            f"{alias_prefix}/{cluster_name}/collapsed_tokens.txt",
            cluster_job.out_collapsed_tokens_txt,
        )
        tk.register_output(
            f"{alias_prefix}/{cluster_name}/cluster_vocab.txt",
            cluster_job.out_cluster_vocab_txt,
        )
        tk.register_output(
            f"{alias_prefix}/{cluster_name}/cluster_counts.txt",
            cluster_job.out_cluster_counts_txt,
        )

        for d in word2vec_dims:
            for w in word2vec_windows:
                dw_tag = _make_dw_tag(d, w)
                train_audio_w2v_job = TrainWord2VecJob(
                    token_text_file=cluster_job.out_collapsed_tokens_txt,
                    vector_size=d,
                    window=w,
                    min_count=1,
                    sg=1,
                    epochs=word2vec_epochs,
                    fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
                    _version=1,
                )
                train_audio_w2v_job.add_alias(f"{audio_shared_prefix}/{cluster_name}/audio_w2v_{dw_tag}")

                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_embeddings.npy",
                    train_audio_w2v_job.out_embeddings_npy,
                )

                audio_pipelines[(cluster_name, d, w)] = {
                    "cluster_job": cluster_job,
                    "w2v_job": train_audio_w2v_job,
                }

    # -------------------------------------------------------------
    # 4. ARTETXE SELF-LEARNING EMBEDDING ALIGNMENT (Cosine & CSLS)
    # -------------------------------------------------------------
    alignment_results = {}

    for code_type in code_types:
        if coverage_threshold is not None and set(cluster_counts) == {483, 6161}:
            matched_k_list = [483] if code_type == "diphone" else [6161]
        else:
            matched_k_list = cluster_counts

        for K in matched_k_list:
            cluster_name = f"clus_{K}_{K}"

            for d in word2vec_dims:
                for w in word2vec_windows:
                    dw_tag = _make_dw_tag(d, w)
                    t_info = text_pipelines[(code_type, d, w)]
                    t_w2v = t_info["w2v_job"]
                    t_vocab = t_info["vocab_file"]

                    a_info = audio_pipelines[(cluster_name, d, w)]
                    a_clus = a_info["cluster_job"]
                    a_w2v = a_info["w2v_job"]

                    for metric in retrieval_metrics:
                        if metric == "cosine":
                            m_sub = "align_freq"
                            m_alias = "align_init_freq"
                            an_sub = "analysis_v2"
                            an_alias = "analysis_freq_v2"
                        else:
                            m_sub = f"align_freq_{metric}"
                            m_alias = f"align_init_freq_{metric}"
                            an_sub = f"analysis_v2_{metric}"
                            an_alias = f"analysis_freq_v2_{metric}"

                        target_path_prefix = f"{alias_prefix}/{code_type}/{cluster_name}/{dw_tag}/{m_sub}"
                        alias_job_prefix = f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{m_alias}"

                        # Frequency-Rank Matching Initialization with specified metric
                        align_freq_job = SelfLearningEmbeddingAlignmentJob(
                            audio_embeddings_npy=a_w2v.out_embeddings_npy,
                            text_embeddings_npy=t_w2v.out_embeddings_npy,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=t_vocab,
                            audio_counts_file=a_clus.out_cluster_counts_txt,
                            text_counts_file=t_w2v.out_counts_txt,
                            init_method="frequency",
                            top_n_freq=None,
                            retrieval_metric=metric,
                            csls_k=csls_k,
                        )
                        align_freq_job.add_alias(alias_job_prefix)

                        tk.register_output(
                            f"{target_path_prefix}/dict.json",
                            align_freq_job.out_dictionary_json,
                        )
                        tk.register_output(
                            f"{target_path_prefix}/report.json",
                            align_freq_job.out_alignment_report,
                        )
                        tk.register_output(
                            f"{target_path_prefix}/W_star.npy",
                            align_freq_job.out_mapping_matrix_npy,
                        )
                        tk.register_output(
                            f"{target_path_prefix}/mapped_audio_embeddings.npy",
                            align_freq_job.out_mapped_audio_emb_npy,
                        )

                        # Standalone Analysis Job (Metric 1A & 3A)
                        analysis_freq_job = EmbeddingAlignmentAnalysisJob(
                            mapped_audio_embeddings_npy=align_freq_job.out_mapped_audio_emb_npy,
                            text_embeddings_npy=t_w2v.out_embeddings_npy,
                            dictionary_json=align_freq_job.out_dictionary_json,
                            audio_vocab_file=a_clus.out_cluster_vocab_txt,
                            text_vocab_file=t_vocab,
                            alignment_report_json=align_freq_job.out_alignment_report,
                            title_prefix=f"Cross-Modal 2D PCA ({code_type}, {cluster_name}, {dw_tag}, {metric})",
                            _version=2,
                        )
                        analysis_freq_job.add_alias(f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{an_alias}")

                        tk.register_output(
                            f"{target_path_prefix}/{an_sub}/analysis_report.json",
                            analysis_freq_job.out_analysis_report_json,
                        )
                        tk.register_output(
                            f"{target_path_prefix}/{an_sub}/cross_modal_pca.json",
                            analysis_freq_job.out_cross_modal_pca_json,
                        )
                        tk.register_output(
                            f"{target_path_prefix}/{an_sub}/cross_modal_pca.png",
                            analysis_freq_job.out_cross_modal_pca_plot,
                        )

                        alignment_results[(code_type, cluster_name, d, w, metric)] = {
                            "cluster_job": a_clus,
                            "audio_w2v_job": a_w2v,
                            "text_w2v_job": t_w2v,
                            "align_freq_job": align_freq_job,
                            "analysis_freq_job": analysis_freq_job,
                        }

    return alignment_results


if __name__ == "__main__":
    pipeline_jobs = build_self_learning_asr_alignment_pipeline()
