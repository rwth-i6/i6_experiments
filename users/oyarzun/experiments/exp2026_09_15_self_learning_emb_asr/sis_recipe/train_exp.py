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
from .data.hubert import ExtractHuBERTLayerFeaturesJob, HuBERTClusterAndCollapseJob
from .data.text import GenerateTextNgramsJob
from .data.word2vec import TrainWord2VecJob
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob
from ..models.analysis import EmbeddingAlignmentAnalysisJob


def build_self_learning_asr_alignment_pipeline(
    librispeech_key: str = "train-clean-100",
    hubert_model_name: str = "facebook/hubert-base-ls960",
    hubert_layer: int = 6,
    word2vec_dim: int = 300,
    word2vec_windows: Optional[List[int]] = None,
    word2vec_epochs: int = 20,
    cluster_counts: Optional[List[int]] = None,
    code_types: Optional[List[str]] = None,
    version: str = "v3",
):
    """
    Constructs the end-to-end self-learning embedding alignment ASR pipeline across
    diphone and triphone cross-modal codes and cluster sizes K in {200, 250, 500}
    with frequency-based initialization.
    """
    if cluster_counts is None:
        cluster_counts = [200, 250, 500]
    if code_types is None:
        code_types = ["diphone", "triphone"]
    if word2vec_windows is None:
        word2vec_windows = [3, 5, 10]

    ver_suffix = f"_{version}" if version else ""
    alias_prefix = f"experiments/self_learning_emb_asr{ver_suffix}/{librispeech_key}/hubert_l{hubert_layer}"
    audio_shared_prefix = f"experiments/self_learning_emb_asr/{librispeech_key}/hubert_l{hubert_layer}"

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
            )
            ngram_job.add_alias(f"{alias_prefix}/text_diphones")
            tk.register_output(f"{alias_prefix}/text/diphones/tokens.txt", ngram_job.out_ngram_text)
            tk.register_output(f"{alias_prefix}/text/diphones/vocab.txt", ngram_job.out_vocab_txt)
            token_text_file = ngram_job.out_ngram_text
            fixed_vocab_file = ngram_job.out_vocab_txt

        elif code_type == "triphone":
            ngram_job = GenerateTextNgramsJob(
                token_text_file=phonemized_text_file,
                ngram_order=3,
                delimiter="_",
                min_count=1,
            )
            ngram_job.add_alias(f"{alias_prefix}/text_triphones")
            tk.register_output(f"{alias_prefix}/text/triphones/tokens.txt", ngram_job.out_ngram_text)
            tk.register_output(f"{alias_prefix}/text/triphones/vocab.txt", ngram_job.out_vocab_txt)
            token_text_file = ngram_job.out_ngram_text
            fixed_vocab_file = ngram_job.out_vocab_txt

        else:
            raise ValueError(f"Unknown code_type: {code_type}. Expected diphone, or triphone.")

        for w in word2vec_windows:
            train_text_w2v_job = TrainWord2VecJob(
                token_text_file=token_text_file,
                vector_size=word2vec_dim,
                window=w,
                min_count=1,
                sg=1,
                epochs=word2vec_epochs,
                fixed_vocab_file=fixed_vocab_file,
                _version=2,
            )
            train_text_w2v_job.add_alias(f"{alias_prefix}/text_w2v_{code_type}s_win{w}")

            tk.register_output(f"{alias_prefix}/text/{code_type}/win{w}/embeddings.npy", train_text_w2v_job.out_embeddings_npy)
            tk.register_output(f"{alias_prefix}/text/{code_type}/win{w}/vocab.txt", train_text_w2v_job.out_vocab_txt)
            tk.register_output(f"{alias_prefix}/text/{code_type}/win{w}/counts.txt", train_text_w2v_job.out_counts_txt)

            text_pipelines[(code_type, w)] = {
                "w2v_job": train_text_w2v_job,
                "vocab_file": fixed_vocab_file,
            }

    # -------------------------------------------------------------
    # 2. AUDIO PIPELINE: HuBERT Layer 6 Feature Extraction (Reused)
    # -------------------------------------------------------------
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))

    hubert_feat_job = ExtractHuBERTLayerFeaturesJob(
        audio_dir=audio_dir,
        model_name=hubert_model_name,
        layer=hubert_layer,
    )
    hubert_feat_job.add_alias(f"{audio_shared_prefix}/hubert_features_l{hubert_layer}")

    # -------------------------------------------------------------
    # 3. CLUSTERING & WORD2VEC: K in {200, 250, 500}
    # -------------------------------------------------------------
    audio_pipelines = {}

    for K in cluster_counts:
        cluster_name = f"clus_{K}_{K}"

        # 3a. Cluster frames and collapse consecutive runs (Reused from audio pipeline)
        cluster_job = HuBERTClusterAndCollapseJob(
            features_npy=hubert_feat_job.out_features_npy,
            lengths_txt=hubert_feat_job.out_lengths_txt,
            seq_tags_txt=hubert_feat_job.out_seq_tags_txt,
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

        for w in word2vec_windows:
            # 3b. Train Audio Cluster Word2Vec (Reused from audio pipeline)
            train_audio_w2v_job = TrainWord2VecJob(
                token_text_file=cluster_job.out_collapsed_tokens_txt,
                vector_size=word2vec_dim,
                window=w,
                min_count=1,
                sg=1,
                epochs=word2vec_epochs,
                fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
                _version=1,
            )
            train_audio_w2v_job.add_alias(f"{audio_shared_prefix}/{cluster_name}/audio_w2v_win{w}")

            tk.register_output(
                f"{alias_prefix}/{cluster_name}/win{w}/audio_embeddings.npy",
                train_audio_w2v_job.out_embeddings_npy,
            )

            audio_pipelines[(cluster_name, w)] = {
                "cluster_job": cluster_job,
                "w2v_job": train_audio_w2v_job,
            }

    # -------------------------------------------------------------
    # 4. ARTETXE SELF-LEARNING EMBEDDING ALIGNMENT (Frequency Init Only)
    # -------------------------------------------------------------
    alignment_results = {}

    for code_type in code_types:
        for K in cluster_counts:
            cluster_name = f"clus_{K}_{K}"

            for w in word2vec_windows:
                t_info = text_pipelines[(code_type, w)]
                t_w2v = t_info["w2v_job"]
                t_vocab = t_info["vocab_file"]

                a_info = audio_pipelines[(cluster_name, w)]
                a_clus = a_info["cluster_job"]
                a_w2v = a_info["w2v_job"]

                target_path_prefix = f"{alias_prefix}/{code_type}/{cluster_name}/win{w}"
                alias_job_prefix = f"{alias_prefix}/{code_type}_{cluster_name}_win{w}"

                # Frequency-Rank Matching Initialization
                align_freq_job = SelfLearningEmbeddingAlignmentJob(
                    audio_embeddings_npy=a_w2v.out_embeddings_npy,
                    text_embeddings_npy=t_w2v.out_embeddings_npy,
                    audio_vocab_file=a_clus.out_cluster_vocab_txt,
                    text_vocab_file=t_vocab,
                    audio_counts_file=a_clus.out_cluster_counts_txt,
                    text_counts_file=t_w2v.out_counts_txt,
                    init_method="frequency",
                    top_n_freq=None,
                )
                align_freq_job.add_alias(f"{alias_job_prefix}/align_init_freq")

                tk.register_output(
                    f"{target_path_prefix}/align_freq/dict.json",
                    align_freq_job.out_dictionary_json,
                )
                tk.register_output(
                    f"{target_path_prefix}/align_freq/report.json",
                    align_freq_job.out_alignment_report,
                )
                tk.register_output(
                    f"{target_path_prefix}/align_freq/W_star.npy",
                    align_freq_job.out_mapping_matrix_npy,
                )

                # Standalone Analysis Job (Metric 1A & 3A) for Frequency Init
                analysis_freq_job = EmbeddingAlignmentAnalysisJob(
                    mapped_audio_embeddings_npy=align_freq_job.out_mapped_audio_emb_npy,
                    text_embeddings_npy=t_w2v.out_embeddings_npy,
                    dictionary_json=align_freq_job.out_dictionary_json,
                    audio_vocab_file=a_clus.out_cluster_vocab_txt,
                    text_vocab_file=t_vocab,
                    alignment_report_json=align_freq_job.out_alignment_report,
                    title_prefix=f"Cross-Modal 2D PCA ({code_type}, {cluster_name}, win{w}, freq-init)",
                    _version=2,
                )
                analysis_freq_job.add_alias(f"{alias_job_prefix}/analysis_freq_v2")

                tk.register_output(
                    f"{target_path_prefix}/align_freq/analysis_v2/analysis_report.json",
                    analysis_freq_job.out_analysis_report_json,
                )
                tk.register_output(
                    f"{target_path_prefix}/align_freq/analysis_v2/cross_modal_pca.json",
                    analysis_freq_job.out_cross_modal_pca_json,
                )
                tk.register_output(
                    f"{target_path_prefix}/align_freq/analysis_v2/cross_modal_pca.png",
                    analysis_freq_job.out_cross_modal_pca_plot,
                )

                alignment_results[(code_type, cluster_name, w)] = {
                    "cluster_job": a_clus,
                    "audio_w2v_job": a_w2v,
                    "text_w2v_job": t_w2v,
                    "align_freq_job": align_freq_job,
                    "analysis_freq_job": analysis_freq_job,
                }

    return alignment_results


if __name__ == "__main__":
    pipeline_jobs = build_self_learning_asr_alignment_pipeline()
