"""
Main experiment pipeline for Self-Learning Embedding Alignment (Artetxe et al. 2017) applied to ASR.
Aligns HuBERT Layer 6 audio cluster Word2Vec embeddings with text Word2Vec embeddings across diphone and triphone codes.
Ablates over cluster counts K in {200, 250, 500} using frequency-based initialization.
"""

from typing import List, Dict, Optional, Tuple, Any
import os
from sisyphus import tk

from .data.librispeech import text as lbs_text
from .data.hubert import (
    DownloadSpinHuBERTCheckpointJob,
    ExtractHuBERTLayerFeaturesJob,
    HuBERTClusterAndCollapseJob,
    HuBERTCheatingSegmentAveragingAndClusterJob,
    HuBERTCheatingSegmentCOPClusterJob,
    HuBERTCheatingSegmentPCKClusterJob,
    HuBERTFramePCAFitJob,
    GenerateAudioClusterNgramsJob,
)
from .data.text import GenerateTextNgramsJob
from .data.word2vec import (
    TrainWord2VecJob,
    TrainStructuredSkipGramJob,
    PostprocessWord2VecAllButTheTopJob,
)
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob
from ..models.gan_alignment import GANEmbeddingAlignmentJob
from ..models.geowarp_alignment import GeoWarPEmbeddingAlignmentJob
from ..models.analysis import (
    EmbeddingAlignmentAnalysisJob,
    Word2VecIsotropyAndPcaAnalysisJob,
)
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
    alignment_method: str = "artetxe",
    text_version: Optional[str] = None,
    code_cluster_map: Optional[Dict[str, List[Tuple[int, str]]]] = None,
    normalize_segments: Optional[str] = None,
    embedding_model: str = "word2vec",
    word2vec_postprocessing: Optional[str] = None,
    all_but_the_top_D: Optional[int] = None,
    run_isotropy_analysis: bool = False,
    spin_codebook_size: Optional[int] = None,
    text_model_tag: Optional[str] = None,
    use_audio_ngrams: bool = False,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    collapse_consecutive: bool = True,
    repetition_penalty: float = 0.0,
    use_cop_kmeans: bool = False,
    use_pck_kmeans: bool = False,
    structural_constraint: Optional[str] = None,
    lambda_struct: float = 0.5,
    struct_margin: float = 0.0,
    compatibility_mode: str = "collapsed_length",
    audio_word_boundaries: bool = False,
    word_boundary_token: str = "<w>",
    pooling_method: str = "mean",
    clustering_method: str = "kmeans",
    spectral_n_neighbors: int = 30,
    spectral_sample_size: int = 25000,
):
    """
    Constructs the end-to-end cross-modal embedding alignment ASR pipeline across
    diphone and triphone cross-modal codes, cluster sizes K, dimensions d, windows w,
    and dictionary retrieval metrics (Cosine & CSLS) with Artetxe frequency-based initialization
    or Domain-Adversarial (GAN) alignment (Chung et al., 2018, arXiv:1805.07467).
    Supports HuBERT, Spin-HuBERT (Chang et al., 2023, arXiv:2305.11072), and Wav2Vec 2.0 backbones.
    """
    if hubert_model_name is not None:
        audio_model_name = hubert_model_name
    if hubert_layer is not None:
        audio_layer = hubert_layer

    if model_tag is None:
        if spin_codebook_size is not None:
            model_tag = f"spin_hubert_{int(spin_codebook_size)}_l{audio_layer}"
        elif "wav2vec2" in audio_model_name or "lv60" in audio_model_name:
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
    txt_ver = text_version if text_version is not None else version
    txt_ver_suffix = f"_{txt_ver}{cov_tag}" if txt_ver else cov_tag
    txt_m_tag = text_model_tag if text_model_tag is not None else model_tag
    text_alias_prefix = f"experiments/self_learning_emb_asr{txt_ver_suffix}/{librispeech_key}/{txt_m_tag}"
    alias_prefix = f"experiments/self_learning_emb_asr{ver_suffix}{seg_tag}{pca_tag}/{librispeech_key}/{model_tag}"
    audio_shared_prefix = f"experiments/self_learning_emb_asr{seg_tag}{pca_tag}/{librispeech_key}/{model_tag}"
    audio_cluster_prefix = alias_prefix if normalize_segments is not None else audio_shared_prefix

    # -------------------------------------------------------------
    # 1. TEXT PIPELINE: Phonemization, N-grams & Text Word2Vec
    # -------------------------------------------------------------
    raw_text_data, _ = lbs_text.get_text(librispeech_key)
    phonemized_text_file, phoneme_vocab, lexicon_file, lm_seq_tags = lbs_text.get_phonemized_text_corpus(
        librispeech_key,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
        word_boundary_token=word_boundary_token if audio_word_boundaries else None,
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
                if embedding_model in ("structured_skipgram", "ssg"):
                    train_text_w2v_job = TrainStructuredSkipGramJob(
                        token_text_file=token_text_file,
                        vector_size=d,
                        window=w,
                        min_count=1,
                        negative=5,
                        epochs=word2vec_epochs,
                        fixed_vocab_file=fixed_vocab_file,
                    )
                    train_text_w2v_job.add_alias(f"{text_alias_prefix}/text_{code_type}_ssg_{dw_tag}")
                    tk.register_output(f"{text_alias_prefix}/text/{code_type}/{dw_tag}/ssg_loss.png", train_text_w2v_job.out_loss_plot)
                else:
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

                raw_text_w2v_job = train_text_w2v_job
                use_abtt = (
                    (all_but_the_top_D is not None and all_but_the_top_D > 0)
                    or (
                        word2vec_postprocessing is not None
                        and word2vec_postprocessing.lower() in ("all_but_the_top", "abtt", "isotropic", "mu2018")
                    )
                )
                if use_abtt:
                    d_rem = all_but_the_top_D if (all_but_the_top_D is not None and all_but_the_top_D > 0) else max(1, d // 100)
                    abtt_text_job = PostprocessWord2VecAllButTheTopJob(
                        embeddings_npy=raw_text_w2v_job.out_embeddings_npy,
                        vocab_txt=raw_text_w2v_job.out_vocab_txt,
                        counts_txt=raw_text_w2v_job.out_counts_txt,
                        num_components_to_remove=d_rem,
                    )
                    abtt_text_job.add_alias(f"{alias_prefix}/text_{code_type}_abtt_D{d_rem}_{dw_tag}")
                    tk.register_output(
                        f"{alias_prefix}/text/{code_type}/{dw_tag}/abtt_D{d_rem}/embeddings.npy",
                        abtt_text_job.out_embeddings_npy,
                    )
                    tk.register_output(
                        f"{alias_prefix}/text/{code_type}/{dw_tag}/abtt_D{d_rem}/isotropy_report.json",
                        abtt_text_job.out_isotropy_report_json,
                    )
                    tk.register_output(
                        f"{alias_prefix}/text/{code_type}/{dw_tag}/abtt_D{d_rem}/isotropy_report.txt",
                        abtt_text_job.out_isotropy_report_txt,
                    )
                    active_text_w2v_job = abtt_text_job
                else:
                    active_text_w2v_job = raw_text_w2v_job

                text_pipelines[(code_type, d, w)] = {
                    "w2v_job": active_text_w2v_job,
                    "raw_w2v_job": raw_text_w2v_job,
                    "vocab_file": fixed_vocab_file,
                }

    # -------------------------------------------------------------
    # 2. AUDIO PIPELINE: Feature Extraction & Optional PCA
    # -------------------------------------------------------------
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))

    spin_ckpt_path = None
    if spin_codebook_size is not None:
        dl_spin_job = DownloadSpinHuBERTCheckpointJob(codebook_size=int(spin_codebook_size))
        dl_spin_job.add_alias(f"{audio_shared_prefix}/download_spin_hubert_{int(spin_codebook_size)}")
        tk.register_output(
            f"{alias_prefix}/spin_ckpt/spin_hubert_{int(spin_codebook_size)}.ckpt",
            dl_spin_job.out_ckpt,
        )
        spin_ckpt_path = dl_spin_job.out_ckpt

    feat_job = ExtractHuBERTLayerFeaturesJob(
        audio_dir=audio_dir,
        model_name=audio_model_name,
        layer=audio_layer,
        spin_ckpt=spin_ckpt_path,
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
        base_cluster_name = f"base_clus_{K}" if use_audio_ngrams else f"clus_{K}_{K}"

        if segmentation_type in ("cheating", "cheat_seg"):
            # 3a. Ablation: Pool audio features over GMM oracle segments and fit K-Means
            if use_pck_kmeans:
                base_cluster_job = HuBERTCheatingSegmentPCKClusterJob(
                    features_npy=feat_job.out_features_npy,
                    lengths_txt=feat_job.out_lengths_txt,
                    seq_tags_txt=feat_job.out_seq_tags_txt,
                    segmentation_hdfs=get_cheating_phoneme_ref_hdfs(),
                    num_clusters=K,
                    pca_components_npy=pca_components,
                    pca_mean_npy=pca_mean,
                    normalize_segments=normalize_segments,
                    collapse_consecutive=collapse_consecutive,
                    repetition_penalty=repetition_penalty,
                )
                base_cluster_job.add_alias(f"{audio_cluster_prefix}/{base_cluster_name}/cheat_seg_pck_kmeans")
            elif use_cop_kmeans:
                base_cluster_job = HuBERTCheatingSegmentCOPClusterJob(
                    features_npy=feat_job.out_features_npy,
                    lengths_txt=feat_job.out_lengths_txt,
                    seq_tags_txt=feat_job.out_seq_tags_txt,
                    segmentation_hdfs=get_cheating_phoneme_ref_hdfs(),
                    num_clusters=K,
                    pca_components_npy=pca_components,
                    pca_mean_npy=pca_mean,
                    normalize_segments=normalize_segments,
                    collapse_consecutive=collapse_consecutive,
                    repetition_penalty=repetition_penalty,
                )
                base_cluster_job.add_alias(f"{audio_cluster_prefix}/{base_cluster_name}/cheat_seg_cop_kmeans")
            else:
                base_cluster_job = HuBERTCheatingSegmentAveragingAndClusterJob(
                    features_npy=feat_job.out_features_npy,
                    lengths_txt=feat_job.out_lengths_txt,
                    seq_tags_txt=feat_job.out_seq_tags_txt,
                    segmentation_hdfs=get_cheating_phoneme_ref_hdfs(),
                    num_clusters=K,
                    pca_components_npy=pca_components,
                    pca_mean_npy=pca_mean,
                    normalize_segments=normalize_segments,
                    collapse_consecutive=collapse_consecutive,
                    repetition_penalty=repetition_penalty,
                    word_boundary_token=word_boundary_token if audio_word_boundaries else None,
                    transcript_text_file=raw_text_data if audio_word_boundaries else None,
                    lexicon_file=lexicon_file if audio_word_boundaries else None,
                    pooling_method=pooling_method,
                    clustering_method=clustering_method,
                    spectral_n_neighbors=spectral_n_neighbors,
                    spectral_sample_size=spectral_sample_size,
                )
                clus_alias_tag = "spectral" if clustering_method in ("spectral", "spectral_clustering") else "kmeans"
                base_cluster_job.add_alias(f"{audio_cluster_prefix}/{base_cluster_name}/cheat_seg_average_{clus_alias_tag}")
        else:
            # 3a. Cluster frames and collapse consecutive runs
            base_cluster_job = HuBERTClusterAndCollapseJob(
                features_npy=feat_job.out_features_npy,
                lengths_txt=feat_job.out_lengths_txt,
                seq_tags_txt=feat_job.out_seq_tags_txt,
                num_clusters=K,
            )
            base_cluster_job.add_alias(f"{audio_cluster_prefix}/{base_cluster_name}/kmeans_collapse")

        tk.register_output(
            f"{alias_prefix}/{base_cluster_name}/collapsed_tokens.txt",
            base_cluster_job.out_collapsed_tokens_txt,
        )
        tk.register_output(
            f"{alias_prefix}/{base_cluster_name}/raw_clusters.txt",
            base_cluster_job.out_raw_clusters_txt,
        )
        tk.register_output(
            f"{alias_prefix}/{base_cluster_name}/cluster_vocab.txt",
            base_cluster_job.out_cluster_vocab_txt,
        )
        tk.register_output(
            f"{alias_prefix}/{base_cluster_name}/cluster_counts.txt",
            base_cluster_job.out_cluster_counts_txt,
        )

        if use_audio_ngrams:
            # 3b. Compositional Audio N-Grams (2-grams / 3-grams built on top of base monophone clusters)
            for code_type in code_types:
                ngram_order = 2 if code_type == "diphone" else 3
                cluster_name = f"clus_{K}_{ngram_order}gram"
                cluster_job = GenerateAudioClusterNgramsJob(
                    base_clusters_txt=base_cluster_job.out_collapsed_tokens_txt,
                    ngram_order=ngram_order,
                    delimiter="_",
                    min_count=1,
                    coverage_threshold=coverage_threshold,
                    max_vocab_size=max_vocab_size,
                )
                cluster_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_ngrams_{ngram_order}")
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/audio_ngrams.txt",
                    cluster_job.out_ngram_text,
                )
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/collapsed_tokens.txt",
                    cluster_job.out_collapsed_tokens_txt,
                )
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/raw_clusters.txt",
                    cluster_job.out_raw_clusters_txt,
                )
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/cluster_vocab.txt",
                    cluster_job.out_cluster_vocab_txt,
                )
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/cluster_counts.txt",
                    cluster_job.out_cluster_counts_txt,
                )
                tk.register_output(
                    f"{alias_prefix}/{cluster_name}/ngram_counts.txt",
                    cluster_job.out_ngram_counts_txt,
                )

                for d in word2vec_dims:
                    for w in word2vec_windows:
                        dw_tag = _make_dw_tag(d, w)
                        if embedding_model in ("structured_skipgram", "ssg"):
                            train_audio_w2v_job = TrainStructuredSkipGramJob(
                                token_text_file=cluster_job.out_ngram_text,
                                vector_size=d,
                                window=w,
                                min_count=1,
                                negative=5,
                                epochs=word2vec_epochs,
                                fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
                            )
                            train_audio_w2v_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_ssg_{dw_tag}")
                            tk.register_output(
                                f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_ssg_loss.png",
                                train_audio_w2v_job.out_loss_plot,
                            )
                        else:
                            train_audio_w2v_job = TrainWord2VecJob(
                                token_text_file=cluster_job.out_ngram_text,
                                vector_size=d,
                                window=w,
                                min_count=1,
                                sg=1,
                                epochs=word2vec_epochs,
                                fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
                                _version=1,
                            )
                            train_audio_w2v_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_w2v_{dw_tag}")

                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_embeddings.npy",
                            train_audio_w2v_job.out_embeddings_npy,
                        )

                        raw_audio_w2v_job = train_audio_w2v_job
                        active_audio_w2v_job = raw_audio_w2v_job
                        if use_abtt:
                            d_rem = all_but_the_top_D if (all_but_the_top_D is not None and all_but_the_top_D > 0) else max(1, d // 100)
                            abtt_audio_job = PostprocessWord2VecAllButTheTopJob(
                                embeddings_npy=raw_audio_w2v_job.out_embeddings_npy,
                                vocab_txt=cluster_job.out_cluster_vocab_txt,
                                counts_txt=cluster_job.out_cluster_counts_txt,
                                num_components_to_remove=d_rem,
                            )
                            abtt_audio_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_w2v_{dw_tag}_abtt_D{d_rem}")
                            tk.register_output(
                                f"{alias_prefix}/{cluster_name}/{dw_tag}/abtt_D{d_rem}/audio_embeddings.npy",
                                abtt_audio_job.out_embeddings_npy,
                            )
                            tk.register_output(
                                f"{alias_prefix}/{cluster_name}/{dw_tag}/abtt_D{d_rem}/isotropy_stats.json",
                                abtt_audio_job.out_isotropy_json,
                            )
                            active_audio_w2v_job = abtt_audio_job

                        audio_pipelines[(code_type, K, d, w)] = {
                            "cluster_job": cluster_job,
                            "base_cluster_job": base_cluster_job,
                            "w2v_job": active_audio_w2v_job,
                            "raw_w2v_job": raw_audio_w2v_job,
                        }
        else:
            cluster_name = base_cluster_name
            cluster_job = base_cluster_job
            for d in word2vec_dims:
                for w in word2vec_windows:
                    dw_tag = _make_dw_tag(d, w)
                    if embedding_model in ("structured_skipgram", "ssg"):
                        train_audio_w2v_job = TrainStructuredSkipGramJob(
                            token_text_file=cluster_job.out_collapsed_tokens_txt,
                            vector_size=d,
                            window=w,
                            min_count=1,
                            negative=5,
                            epochs=word2vec_epochs,
                            fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
                        )
                        train_audio_w2v_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_ssg_{dw_tag}")
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_ssg_loss.png",
                            train_audio_w2v_job.out_loss_plot,
                        )
                    else:
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
                        train_audio_w2v_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_w2v_{dw_tag}")

                    tk.register_output(
                        f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_embeddings.npy",
                        train_audio_w2v_job.out_embeddings_npy,
                    )

                    raw_audio_w2v_job = train_audio_w2v_job
                    active_audio_w2v_job = raw_audio_w2v_job
                    if use_abtt:
                        d_rem = all_but_the_top_D if (all_but_the_top_D is not None and all_but_the_top_D > 0) else max(1, d // 100)
                        abtt_audio_job = PostprocessWord2VecAllButTheTopJob(
                            embeddings_npy=raw_audio_w2v_job.out_embeddings_npy,
                            vocab_txt=cluster_job.out_cluster_vocab_txt,
                            counts_txt=cluster_job.out_cluster_counts_txt,
                            num_components_to_remove=d_rem,
                        )
                        abtt_audio_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_w2v_{dw_tag}_abtt_D{d_rem}")
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/{dw_tag}/abtt_D{d_rem}/audio_embeddings.npy",
                            abtt_audio_job.out_embeddings_npy,
                        )
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/{dw_tag}/abtt_D{d_rem}/isotropy_stats.json",
                            abtt_audio_job.out_isotropy_json,
                        )
                        active_audio_w2v_job = abtt_audio_job

                    audio_pipelines[(K, d, w)] = {
                        "cluster_job": cluster_job,
                        "w2v_job": active_audio_w2v_job,
                        "raw_w2v_job": raw_audio_w2v_job,
                    }

    # -------------------------------------------------------------
    # 4. EMBEDDING ALIGNMENT (Artetxe Self-Learning & GAN Domain-Adversarial)
    # -------------------------------------------------------------
    alignment_results = {}

    for code_type in code_types:
        if code_cluster_map is not None and code_type in code_cluster_map:
            target_pairs = code_cluster_map[code_type]
        elif use_audio_ngrams:
            ngram_order = 2 if code_type == "diphone" else 3
            target_pairs = [(k, f"clus_{k}_{ngram_order}gram") for k in cluster_counts]
        elif coverage_threshold is not None and set(cluster_counts) == {483, 6161}:
            matched_k_list = [483] if code_type == "diphone" else [6161]
            target_pairs = [(k, f"clus_{k}_{k}") for k in matched_k_list]
        elif coverage_threshold is not None and set(cluster_counts) == {483, 500, 6161}:
            if code_type == "diphone":
                target_pairs = [(483, "clus_483_483"), (500, "clus_500_483")]
            else:
                target_pairs = [(6161, "clus_6161_6161"), (500, "clus_500_6161")]
        else:
            target_pairs = [(k, f"clus_{k}_{k}") for k in cluster_counts]

        for K, cluster_name in target_pairs:
            ngram_order = 2 if code_type == "diphone" else 3
            base_audio_clus = f"clus_{K}_{ngram_order}gram" if use_audio_ngrams else f"clus_{K}_{K}"

            for d in word2vec_dims:
                for w in word2vec_windows:
                    dw_tag = _make_dw_tag(d, w)
                    t_info = text_pipelines[(code_type, d, w)]
                    t_w2v = t_info["w2v_job"]
                    raw_t_w2v = t_info.get("raw_w2v_job", t_w2v)
                    t_vocab = t_info["vocab_file"]

                    a_info = audio_pipelines.get((code_type, K, d, w)) or audio_pipelines[(K, d, w)]
                    a_clus = a_info["cluster_job"]
                    a_w2v = a_info["w2v_job"]
                    raw_a_w2v = a_info.get("raw_w2v_job", a_w2v)

                    # Also register alias links under cluster_name if asymmetric (e.g., clus_500_483 / clus_500_6161)
                    if cluster_name != base_audio_clus:
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/collapsed_tokens.txt",
                            a_clus.out_collapsed_tokens_txt,
                        )
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/raw_clusters.txt",
                            a_clus.out_raw_clusters_txt,
                        )
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/cluster_vocab.txt",
                            a_clus.out_cluster_vocab_txt,
                        )
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/cluster_counts.txt",
                            a_clus.out_cluster_counts_txt,
                        )
                        tk.register_output(
                            f"{alias_prefix}/{cluster_name}/{dw_tag}/audio_embeddings.npy",
                            a_w2v.out_embeddings_npy,
                        )

                    w2v_isotropy_job = None
                    if use_abtt or run_isotropy_analysis:
                        d_rem_val = (
                            int(all_but_the_top_D)
                            if (all_but_the_top_D is not None and all_but_the_top_D > 0)
                            else max(1, d // 100)
                        )
                        w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                            src_raw_embeddings_npy=raw_a_w2v.out_embeddings_npy,
                            trg_raw_embeddings_npy=raw_t_w2v.out_embeddings_npy,
                            src_post_embeddings_npy=a_w2v.out_embeddings_npy if use_abtt else None,
                            trg_post_embeddings_npy=t_w2v.out_embeddings_npy if use_abtt else None,
                            src_vocab_file=a_clus.out_cluster_vocab_txt,
                            trg_vocab_file=t_vocab,
                            src_counts_file=a_clus.out_cluster_counts_txt,
                            trg_counts_file=t_w2v.out_counts_txt,
                            d_remove=d_rem_val,
                            title_prefix=f"Embedding Isotropy & PCA ({code_type}, {cluster_name}, {dw_tag})",
                            src_label="Audio Clusters",
                            trg_label=f"Text {code_type.capitalize()}s",
                            is_postprocessed_active=bool(use_abtt),
                        )
                        w2v_isotropy_job.add_alias(
                            f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/w2v_isotropy_pca"
                        )

                    for metric in retrieval_metrics:
                        if alignment_method in ("artetxe", "both"):
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

                            analysis_freq_job = EmbeddingAlignmentAnalysisJob(
                                mapped_audio_embeddings_npy=align_freq_job.out_mapped_audio_emb_npy,
                                text_embeddings_npy=t_w2v.out_embeddings_npy,
                                dictionary_json=align_freq_job.out_dictionary_json,
                                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                                text_vocab_file=t_vocab,
                                audio_counts_file=a_clus.out_cluster_counts_txt,
                                text_counts_file=t_w2v.out_counts_txt,
                                alignment_report_json=align_freq_job.out_alignment_report,
                                title_prefix=f"Cross-Modal 2D PCA ({code_type}, {cluster_name}, {dw_tag}, {metric})",
                                src_label="Audio Clusters",
                                trg_label="Text Phonemes",
                                _version=4,
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
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/source_audio_pca.png",
                                analysis_freq_job.out_source_pca_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/target_text_pca.png",
                                analysis_freq_job.out_target_pca_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/cross_modal_pca_panel.png",
                                analysis_freq_job.out_cross_modal_panel_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/source_raw_pca.png",
                                analysis_freq_job.out_source_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/target_raw_pca.png",
                                analysis_freq_job.out_target_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/unaligned_cross_modal_pca.png",
                                analysis_freq_job.out_unaligned_joint_pca_plot,
                            )
                            tk.register_output(
                                f"{target_path_prefix}/{an_sub}/alignment_comparison_panel.png",
                                analysis_freq_job.out_alignment_comparison_panel_plot,
                            )

                            if w2v_isotropy_job is not None:
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/isotropy_report.json",
                                    w2v_isotropy_job.out_isotropy_report_json,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/isotropy_report.txt",
                                    w2v_isotropy_job.out_isotropy_report_txt,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png",
                                    w2v_isotropy_job.out_src_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png",
                                    w2v_isotropy_job.out_src_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/target_text_pca_before_abtt.png",
                                    w2v_isotropy_job.out_trg_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/target_text_pca_after_abtt.png",
                                    w2v_isotropy_job.out_trg_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png",
                                    w2v_isotropy_job.out_before_after_panel_plot,
                                )
                                tk.register_output(
                                    f"{target_path_prefix}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png",
                                    w2v_isotropy_job.out_isotropy_spectrum_plot,
                                )

                            alignment_results[(code_type, cluster_name, d, w, metric)] = {
                                "cluster_job": a_clus,
                                "audio_w2v_job": a_w2v,
                                "text_w2v_job": t_w2v,
                                "raw_audio_w2v_job": raw_a_w2v,
                                "raw_text_w2v_job": raw_t_w2v,
                                "align_freq_job": align_freq_job,
                                "analysis_freq_job": analysis_freq_job,
                                "w2v_isotropy_job": w2v_isotropy_job,
                            }

                        if alignment_method in ("gan", "both"):
                            if metric == "cosine":
                                gan_sub = "align_gan"
                                gan_alias = "align_init_gan"
                                gan_an_sub = "analysis_v2"
                                gan_an_alias = "analysis_gan_v2"
                            else:
                                gan_sub = f"align_gan_{metric}"
                                gan_alias = f"align_init_gan_{metric}"
                                gan_an_sub = f"analysis_v2_{metric}"
                                gan_an_alias = f"analysis_gan_v2_{metric}"

                            gan_target_prefix = f"{alias_prefix}/{code_type}/{cluster_name}/{dw_tag}/{gan_sub}"
                            gan_job_prefix = f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{gan_alias}"

                            align_gan_job = GANEmbeddingAlignmentJob(
                                audio_embeddings_npy=a_w2v.out_embeddings_npy,
                                text_embeddings_npy=t_w2v.out_embeddings_npy,
                                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                                text_vocab_file=t_vocab,
                                audio_counts_file=a_clus.out_cluster_counts_txt,
                                text_counts_file=t_w2v.out_counts_txt,
                                disc_hidden_dim=512,
                                disc_num_layers=2,
                                lr=1e-3,
                                retrieval_metric=metric,
                                csls_k=csls_k,
                                structural_constraint=structural_constraint,
                                lambda_struct=lambda_struct,
                                struct_margin=struct_margin,
                                compatibility_mode=compatibility_mode,
                            )
                            align_gan_job.add_alias(gan_job_prefix)

                            tk.register_output(
                                f"{gan_target_prefix}/dict.json",
                                align_gan_job.out_dictionary_json,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/report.json",
                                align_gan_job.out_alignment_report,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/W_star.npy",
                                align_gan_job.out_mapping_matrix_npy,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/W_adv.npy",
                                align_gan_job.out_adv_mapping_matrix_npy,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/mapped_audio_embeddings.npy",
                                align_gan_job.out_mapped_audio_emb_npy,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/gan_training_curves.png",
                                align_gan_job.out_gan_loss_plot,
                            )

                            analysis_gan_job = EmbeddingAlignmentAnalysisJob(
                                mapped_audio_embeddings_npy=align_gan_job.out_mapped_audio_emb_npy,
                                text_embeddings_npy=t_w2v.out_embeddings_npy,
                                dictionary_json=align_gan_job.out_dictionary_json,
                                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                                text_vocab_file=t_vocab,
                                audio_counts_file=a_clus.out_cluster_counts_txt,
                                text_counts_file=t_w2v.out_counts_txt,
                                alignment_report_json=align_gan_job.out_alignment_report,
                                title_prefix=f"GAN Cross-Modal 2D PCA ({code_type}, {cluster_name}, {dw_tag}, {metric})",
                                src_label="Audio Clusters",
                                trg_label="Text Phonemes",
                                _version=4,
                            )
                            analysis_gan_job.add_alias(f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{gan_an_alias}")

                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/analysis_report.json",
                                analysis_gan_job.out_analysis_report_json,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/cross_modal_pca.json",
                                analysis_gan_job.out_cross_modal_pca_json,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/cross_modal_pca.png",
                                analysis_gan_job.out_cross_modal_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/source_audio_pca.png",
                                analysis_gan_job.out_source_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/target_text_pca.png",
                                analysis_gan_job.out_target_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/cross_modal_pca_panel.png",
                                analysis_gan_job.out_cross_modal_panel_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/source_raw_pca.png",
                                analysis_gan_job.out_source_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/target_raw_pca.png",
                                analysis_gan_job.out_target_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/unaligned_cross_modal_pca.png",
                                analysis_gan_job.out_unaligned_joint_pca_plot,
                            )
                            tk.register_output(
                                f"{gan_target_prefix}/{gan_an_sub}/alignment_comparison_panel.png",
                                analysis_gan_job.out_alignment_comparison_panel_plot,
                            )

                            if w2v_isotropy_job is not None:
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/isotropy_report.json",
                                    w2v_isotropy_job.out_isotropy_report_json,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/isotropy_report.txt",
                                    w2v_isotropy_job.out_isotropy_report_txt,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png",
                                    w2v_isotropy_job.out_src_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png",
                                    w2v_isotropy_job.out_src_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/target_text_pca_before_abtt.png",
                                    w2v_isotropy_job.out_trg_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/target_text_pca_after_abtt.png",
                                    w2v_isotropy_job.out_trg_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png",
                                    w2v_isotropy_job.out_before_after_panel_plot,
                                )
                                tk.register_output(
                                    f"{gan_target_prefix}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png",
                                    w2v_isotropy_job.out_isotropy_spectrum_plot,
                                )

                            alignment_results[(code_type, cluster_name, d, w, f"gan_{metric}")] = {
                                "cluster_job": a_clus,
                                "audio_w2v_job": a_w2v,
                                "text_w2v_job": t_w2v,
                                "raw_audio_w2v_job": raw_a_w2v,
                                "raw_text_w2v_job": raw_t_w2v,
                                "align_gan_job": align_gan_job,
                                "analysis_gan_job": analysis_gan_job,
                                "w2v_isotropy_job": w2v_isotropy_job,
                            }

                        if alignment_method in ("geowarp",):
                            if metric == "cosine":
                                gw_sub = "align_geowarp"
                                gw_alias = "align_init_geowarp"
                                gw_an_sub = "analysis_v2"
                                gw_an_alias = "analysis_geowarp_v2"
                            else:
                                gw_sub = f"align_geowarp_{metric}"
                                gw_alias = f"align_init_geowarp_{metric}"
                                gw_an_sub = f"analysis_v2_{metric}"
                                gw_an_alias = f"analysis_geowarp_v2_{metric}"

                            gw_target_prefix = f"{alias_prefix}/{code_type}/{cluster_name}/{dw_tag}/{gw_sub}"
                            gw_job_prefix = f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{gw_alias}"

                            align_geowarp_job = GeoWarPEmbeddingAlignmentJob(
                                audio_embeddings_npy=a_w2v.out_embeddings_npy,
                                text_embeddings_npy=t_w2v.out_embeddings_npy,
                                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                                text_vocab_file=t_vocab,
                                audio_counts_file=a_clus.out_cluster_counts_txt,
                                text_counts_file=t_w2v.out_counts_txt,
                                retrieval_metric=metric,
                                csls_k=csls_k,
                                structural_constraint=structural_constraint,
                                compatibility_mode=compatibility_mode,
                            )
                            align_geowarp_job.add_alias(gw_job_prefix)

                            tk.register_output(
                                f"{gw_target_prefix}/dict.json",
                                align_geowarp_job.out_dictionary_json,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/report.json",
                                align_geowarp_job.out_alignment_report,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/W_star.npy",
                                align_geowarp_job.out_mapping_matrix_npy,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/W_init.npy",
                                align_geowarp_job.out_init_mapping_matrix_npy,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/W_wp.npy",
                                align_geowarp_job.out_wp_mapping_matrix_npy,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/mapped_audio_embeddings.npy",
                                align_geowarp_job.out_mapped_audio_emb_npy,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/geowarp_training_curves.png",
                                align_geowarp_job.out_geowarp_loss_plot,
                            )

                            analysis_geowarp_job = EmbeddingAlignmentAnalysisJob(
                                mapped_audio_embeddings_npy=align_geowarp_job.out_mapped_audio_emb_npy,
                                text_embeddings_npy=t_w2v.out_embeddings_npy,
                                dictionary_json=align_geowarp_job.out_dictionary_json,
                                raw_audio_embeddings_npy=a_w2v.out_embeddings_npy,
                                raw_text_embeddings_npy=t_w2v.out_embeddings_npy,
                                audio_vocab_file=a_clus.out_cluster_vocab_txt,
                                text_vocab_file=t_vocab,
                                audio_counts_file=a_clus.out_cluster_counts_txt,
                                text_counts_file=t_w2v.out_counts_txt,
                                alignment_report_json=align_geowarp_job.out_alignment_report,
                                title_prefix=f"GeoWarP Cross-Modal 2D PCA ({code_type}, {cluster_name}, {dw_tag}, {metric})",
                                src_label="Audio Clusters",
                                trg_label="Text Phonemes",
                                _version=4,
                            )
                            analysis_geowarp_job.add_alias(f"{alias_prefix}/{code_type}_{cluster_name}_{dw_tag}/{gw_an_alias}")

                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/analysis_report.json",
                                analysis_geowarp_job.out_analysis_report_json,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/cross_modal_pca.json",
                                analysis_geowarp_job.out_cross_modal_pca_json,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/cross_modal_pca.png",
                                analysis_geowarp_job.out_cross_modal_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/source_audio_pca.png",
                                analysis_geowarp_job.out_source_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/target_text_pca.png",
                                analysis_geowarp_job.out_target_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/cross_modal_pca_panel.png",
                                analysis_geowarp_job.out_cross_modal_panel_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/source_raw_pca.png",
                                analysis_geowarp_job.out_source_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/target_raw_pca.png",
                                analysis_geowarp_job.out_target_raw_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/unaligned_cross_modal_pca.png",
                                analysis_geowarp_job.out_unaligned_joint_pca_plot,
                            )
                            tk.register_output(
                                f"{gw_target_prefix}/{gw_an_sub}/alignment_comparison_panel.png",
                                analysis_geowarp_job.out_alignment_comparison_panel_plot,
                            )

                            if w2v_isotropy_job is not None:
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/isotropy_report.json",
                                    w2v_isotropy_job.out_isotropy_report_json,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/isotropy_report.txt",
                                    w2v_isotropy_job.out_isotropy_report_txt,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/source_audio_pca_before_abtt.png",
                                    w2v_isotropy_job.out_src_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/source_audio_pca_after_abtt.png",
                                    w2v_isotropy_job.out_src_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/target_text_pca_before_abtt.png",
                                    w2v_isotropy_job.out_trg_before_pca_plot,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/target_text_pca_after_abtt.png",
                                    w2v_isotropy_job.out_trg_after_pca_plot,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/pca_before_vs_after_abtt_panel.png",
                                    w2v_isotropy_job.out_before_after_panel_plot,
                                )
                                tk.register_output(
                                    f"{gw_target_prefix}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png",
                                    w2v_isotropy_job.out_isotropy_spectrum_plot,
                                )

                            entry_dict = {
                                "cluster_job": a_clus,
                                "audio_w2v_job": a_w2v,
                                "text_w2v_job": t_w2v,
                                "raw_audio_w2v_job": raw_a_w2v,
                                "raw_text_w2v_job": raw_t_w2v,
                                "align_geowarp_job": align_geowarp_job,
                                "align_gan_job": align_geowarp_job,
                                "analysis_geowarp_job": analysis_geowarp_job,
                                "analysis_gan_job": analysis_geowarp_job,
                                "w2v_isotropy_job": w2v_isotropy_job,
                            }
                            alignment_results[(code_type, cluster_name, d, w, f"geowarp_{metric}")] = entry_dict
                            alignment_results[(code_type, cluster_name, d, w, f"gan_{metric}")] = entry_dict

    return alignment_results


# =============================================================================
# Asymmetric Triphone Parameter Sweep Pipeline (K audio clusters, N text triphones)
# Pure Cosine Retrieval (No CSLS)
# =============================================================================

DEFAULT_TRIPHONE_ASYMMETRIC_SWEEP_PAIRS = [
    # Regime 1: Acoustic Oversampling (K > N)
    (10000, 4000),   # 2.50x
    (10000, 6000),   # 1.67x
    (15000, 6000),   # 2.50x
    (15000, 10000),  # 1.50x
    # Regime 2: Symmetric Anchors (K = N)
    (6000, 6000),    # 1.00x
    (10000, 10000),  # 1.00x
    # Regime 3: Lexical Richness (K < N)
    (4000, 10000),   # 0.40x
    (6000, 12000),   # 0.50x
    (10000, 15000),  # 0.67x
]


def build_triphone_asymmetric_sweep_pipeline(
    librispeech_key: str = "train-clean-100",
    audio_model_name: str = "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
    audio_layer: int = 6,
    pca_dim: int = 512,
    word2vec_dim: int = 128,
    word2vec_win: int = 2,
    word2vec_epochs: int = 20,
    pairs: Optional[List[Tuple[int, int]]] = None,
    normalize_segments: str = "unit_sphere",
    run_isotropy_analysis: bool = True,
    version_tag: str = "v5.6_triphone_sweep",
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Constructs an asymmetric parameter sweep over audio cluster count K and text triphone vocabulary N:
      - 9 target ablations across oversampling (K > N), symmetric (K = N), and lexical richness (K < N).
      - Reuses frame-level HuBERT features + PCA-512 across all jobs.
      - Fits 4 unique K-Means clusterings (K in {4k, 6k, 10k, 15k}) with unit-sphere normalization.
      - Trains 5 unique text triphone vocabularies (N in {4k, 6k, 10k, 12k, 15k}).
      - Trains Structured Skip-Gram (SSG) embeddings (dim=128, win=2) for each unique K and N.
      - Performs GAN domain-adversarial alignment with Cosine nearest-neighbor Procrustes refinement (NO CSLS).
    """
    if pairs is None:
        pairs = DEFAULT_TRIPHONE_ASYMMETRIC_SWEEP_PAIRS

    model_tag = f"hubert_l{audio_layer}"
    audio_shared_prefix = f"experiments/self_learning_emb_asr_cheat_seg_pca{pca_dim}/{librispeech_key}/{model_tag}"
    audio_cluster_prefix = f"experiments/self_learning_emb_asr_v5.6_unit_sphere_cheat_seg_pca{pca_dim}/{librispeech_key}/{model_tag}"
    text_alias_prefix = f"experiments/self_learning_emb_asr_v5.6/{librispeech_key}/{model_tag}"
    sweep_alias_prefix = f"experiments/self_learning_emb_asr_{version_tag}_pca{pca_dim}/{librispeech_key}/{model_tag}"
    dw_tag = _make_dw_tag(word2vec_dim, word2vec_win)

    # 1. Feature extraction & frame PCA
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))
    feat_job = ExtractHuBERTLayerFeaturesJob(
        audio_dir=audio_dir,
        model_name=audio_model_name,
        layer=audio_layer,
    )
    feat_job.add_alias(f"{audio_shared_prefix}/{model_tag}_features")

    pca_fit_job = HuBERTFramePCAFitJob(
        features_npy=feat_job.out_features_npy,
        n_components=pca_dim,
    )
    pca_fit_job.add_alias(f"{audio_shared_prefix}/pca_{pca_dim}")

    # 2. Audio clustering & Structured Skip-Gram (unique K)
    unique_k = sorted(list({k for k, _ in pairs}))
    audio_pipelines = {}
    for K in unique_k:
        cluster_name = f"clus_{K}_{K}"
        cluster_job = HuBERTCheatingSegmentAveragingAndClusterJob(
            features_npy=feat_job.out_features_npy,
            lengths_txt=feat_job.out_lengths_txt,
            seq_tags_txt=feat_job.out_seq_tags_txt,
            segmentation_hdfs=get_cheating_phoneme_ref_hdfs(),
            num_clusters=K,
            pca_components_npy=pca_fit_job.out_components_npy,
            pca_mean_npy=pca_fit_job.out_mean_npy,
            normalize_segments=normalize_segments,
        )
        cluster_job.add_alias(f"{audio_cluster_prefix}/{cluster_name}/cheat_seg_average_kmeans")

        train_audio_ssg = TrainStructuredSkipGramJob(
            token_text_file=cluster_job.out_collapsed_tokens_txt,
            vector_size=word2vec_dim,
            window=word2vec_win,
            min_count=1,
            negative=5,
            epochs=word2vec_epochs,
            fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
        )
        train_audio_ssg.add_alias(f"{audio_cluster_prefix}/{cluster_name}/audio_ssg_{dw_tag}")

        tk.register_output(f"{audio_cluster_prefix}/{cluster_name}/collapsed_tokens.txt", cluster_job.out_collapsed_tokens_txt)
        tk.register_output(f"{audio_cluster_prefix}/{cluster_name}/raw_clusters.txt", cluster_job.out_raw_clusters_txt)
        tk.register_output(f"{audio_cluster_prefix}/{cluster_name}/cluster_vocab.txt", cluster_job.out_cluster_vocab_txt)
        tk.register_output(f"{audio_cluster_prefix}/{cluster_name}/cluster_counts.txt", cluster_job.out_cluster_counts_txt)
        tk.register_output(f"{audio_cluster_prefix}/{cluster_name}/{dw_tag}/audio_embeddings.npy", train_audio_ssg.out_embeddings_npy)

        audio_pipelines[K] = {
            "cluster_job": cluster_job,
            "audio_ssg_job": train_audio_ssg,
        }

    # 3. Text triphone N-grams & Structured Skip-Gram (unique N)
    phonemized_text_file, phoneme_vocab, lexicon_file, lm_seq_tags = lbs_text.get_phonemized_text_corpus(
        librispeech_key,
    )
    unique_n = sorted(list({n for _, n in pairs}))
    text_pipelines = {}
    for N in unique_n:
        text_tag = f"triphones_N{N}"
        ngram_job = GenerateTextNgramsJob(
            token_text_file=phonemized_text_file,
            ngram_order=3,
            delimiter="_",
            min_count=1,
            max_vocab_size=N,
        )
        ngram_job.add_alias(f"{text_alias_prefix}/text_{text_tag}")

        train_text_ssg = TrainStructuredSkipGramJob(
            token_text_file=ngram_job.out_ngram_text,
            vector_size=word2vec_dim,
            window=word2vec_win,
            min_count=1,
            negative=5,
            epochs=word2vec_epochs,
            fixed_vocab_file=ngram_job.out_vocab_txt,
        )
        train_text_ssg.add_alias(f"{text_alias_prefix}/text_{text_tag}_ssg_{dw_tag}")

        tk.register_output(f"{text_alias_prefix}/text/{text_tag}/tokens.txt", ngram_job.out_ngram_text)
        tk.register_output(f"{text_alias_prefix}/text/{text_tag}/vocab.txt", ngram_job.out_vocab_txt)
        tk.register_output(f"{text_alias_prefix}/text/{text_tag}/{dw_tag}/embeddings.npy", train_text_ssg.out_embeddings_npy)
        tk.register_output(f"{text_alias_prefix}/text/{text_tag}/{dw_tag}/counts.txt", train_text_ssg.out_counts_txt)

        text_pipelines[N] = {
            "ngram_job": ngram_job,
            "text_ssg_job": train_text_ssg,
        }

    # 4. Alignment for each (K, N) pair: Pure Cosine Retrieval (No CSLS)
    alignment_results = {}
    for K, N in pairs:
        pair_tag = f"K{K}_N{N}"
        a_info = audio_pipelines[K]
        t_info = text_pipelines[N]

        a_clus = a_info["cluster_job"]
        a_ssg = a_info["audio_ssg_job"]
        t_ngram = t_info["ngram_job"]
        t_ssg = t_info["text_ssg_job"]

        gan_target_prefix = f"{sweep_alias_prefix}/{pair_tag}/{dw_tag}/align_gan_cosine"
        gan_job_prefix = f"{sweep_alias_prefix}/{pair_tag}_{dw_tag}/align_init_gan_cosine"

        align_gan_job = GANEmbeddingAlignmentJob(
            audio_embeddings_npy=a_ssg.out_embeddings_npy,
            text_embeddings_npy=t_ssg.out_embeddings_npy,
            audio_vocab_file=a_clus.out_cluster_vocab_txt,
            text_vocab_file=t_ngram.out_vocab_txt,
            audio_counts_file=a_clus.out_cluster_counts_txt,
            text_counts_file=t_ssg.out_counts_txt,
            disc_hidden_dim=512,
            disc_num_layers=2,
            lr=1e-3,
            retrieval_metric="cosine",  # Strictly Cosine, removing CSLS
        )
        align_gan_job.add_alias(gan_job_prefix)

        tk.register_output(f"{gan_target_prefix}/dict.json", align_gan_job.out_dictionary_json)
        tk.register_output(f"{gan_target_prefix}/report.json", align_gan_job.out_alignment_report)
        tk.register_output(f"{gan_target_prefix}/W_star.npy", align_gan_job.out_mapping_matrix_npy)
        tk.register_output(f"{gan_target_prefix}/W_adv.npy", align_gan_job.out_adv_mapping_matrix_npy)
        tk.register_output(f"{gan_target_prefix}/mapped_audio_embeddings.npy", align_gan_job.out_mapped_audio_emb_npy)
        tk.register_output(f"{gan_target_prefix}/gan_training_curves.png", align_gan_job.out_gan_loss_plot)

        # Cross-modal 2D PCA & representation analysis
        an_target_prefix = f"{gan_target_prefix}/analysis_cosine"
        analysis_gan_job = EmbeddingAlignmentAnalysisJob(
            mapped_audio_embeddings_npy=align_gan_job.out_mapped_audio_emb_npy,
            text_embeddings_npy=t_ssg.out_embeddings_npy,
            dictionary_json=align_gan_job.out_dictionary_json,
            raw_audio_embeddings_npy=a_ssg.out_embeddings_npy,
            raw_text_embeddings_npy=t_ssg.out_embeddings_npy,
            audio_vocab_file=a_clus.out_cluster_vocab_txt,
            text_vocab_file=t_ngram.out_vocab_txt,
            audio_counts_file=a_clus.out_cluster_counts_txt,
            text_counts_file=t_ssg.out_counts_txt,
            alignment_report_json=align_gan_job.out_alignment_report,
            title_prefix=f"Cross-Modal 2D PCA (Triphones, K={K}, N={N}, Cosine)",
            src_label=f"Audio Clusters (K={K})",
            trg_label=f"Text Triphones (N={N})",
            _version=4,
        )
        analysis_gan_job.add_alias(f"{sweep_alias_prefix}/{pair_tag}_{dw_tag}/analysis_gan_cosine")
        tk.register_output(f"{an_target_prefix}/cross_modal_pca.png", analysis_gan_job.out_cross_modal_pca_plot)
        tk.register_output(f"{an_target_prefix}/cross_modal_pca.json", analysis_gan_job.out_cross_modal_pca_json)
        tk.register_output(f"{an_target_prefix}/alignment_comparison_panel.png", analysis_gan_job.out_alignment_comparison_panel_plot)
        tk.register_output(f"{an_target_prefix}/analysis_report.json", analysis_gan_job.out_analysis_report_json)

        w2v_isotropy_job = None
        if run_isotropy_analysis:
            w2v_isotropy_job = Word2VecIsotropyAndPcaAnalysisJob(
                src_raw_embeddings_npy=a_ssg.out_embeddings_npy,
                trg_raw_embeddings_npy=t_ssg.out_embeddings_npy,
                src_post_embeddings_npy=a_ssg.out_embeddings_npy,
                trg_post_embeddings_npy=t_ssg.out_embeddings_npy,
                src_vocab_file=a_clus.out_cluster_vocab_txt,
                trg_vocab_file=t_ngram.out_vocab_txt,
                src_counts_file=a_clus.out_cluster_counts_txt,
                trg_counts_file=t_ssg.out_counts_txt,
                d_remove=0,
                title_prefix=f"Embedding Isotropy (K={K}, N={N})",
                src_label=f"Audio Clusters (K={K})",
                trg_label=f"Text Triphones (N={N})",
                is_postprocessed_active=False,
            )
            w2v_isotropy_job.add_alias(f"{sweep_alias_prefix}/{pair_tag}_{dw_tag}/w2v_isotropy_analysis")
            tk.register_output(f"{gan_target_prefix}/w2v_isotropy_analysis/isotropy_report.json", w2v_isotropy_job.out_isotropy_report_json)
            tk.register_output(f"{gan_target_prefix}/w2v_isotropy_analysis/isotropy_and_spectrum_panel.png", w2v_isotropy_job.out_isotropy_spectrum_plot)

        alignment_results[(K, N)] = {
            "cluster_job": a_clus,
            "audio_ssg_job": a_ssg,
            "ngram_job": t_ngram,
            "text_ssg_job": t_ssg,
            "align_gan_job": align_gan_job,
            "analysis_gan_job": analysis_gan_job,
            "w2v_isotropy_job": w2v_isotropy_job,
        }

    return alignment_results


if __name__ == "__main__":
    pipeline_jobs = build_self_learning_asr_alignment_pipeline()

