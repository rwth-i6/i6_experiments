"""
Main experiment pipeline for Self-Learning Embedding Alignment (Artetxe et al. 2017) applied to ASR.
Aligns HuBERT Layer 6 audio cluster Word2Vec embeddings with text phoneme Word2Vec embeddings.
Ablates over cluster counts K in {P, 2P, 4P}.
"""

from typing import List, Dict, Optional
import os
from sisyphus import tk

from .data.librispeech import text as lbs_text
from .data.librispeech import audio as lbs_audio
from .data.hubert import ExtractHuBERTLayerFeaturesJob, HuBERTClusterAndCollapseJob
from .data.word2vec import TrainWord2VecJob
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob


def build_self_learning_asr_alignment_pipeline(
    librispeech_key: str = "train-clean-100",
    hubert_model_name: str = "facebook/hubert-base-ls960",
    hubert_layer: int = 6,
    word2vec_dim: int = 300,
    word2vec_window: int = 10,
    word2vec_epochs: int = 20,
    base_phoneme_count: int = 41,  # P for LibriSpeech G2P phoneme inventory
):
    """
    Constructs the end-to-end self-learning embedding alignment ASR pipeline.
    """
    alias_prefix = f"experiments/self_learning_emb_asr/{librispeech_key}/hubert_l{hubert_layer}"

    # -------------------------------------------------------------
    # 1. TEXT PIPELINE: Phonemization & Text Word2Vec
    # -------------------------------------------------------------
    # Get phonemized text sequences and vocabulary directly from job output
    phonemized_text_file, phoneme_vocab, lexicon_file, lm_seq_tags = lbs_text.get_phonemized_text_corpus(
        librispeech_key,
    )

    train_text_w2v_job = TrainWord2VecJob(
        token_text_file=phonemized_text_file,
        vector_size=word2vec_dim,
        window=word2vec_window,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=word2vec_epochs,
        fixed_vocab_file=phoneme_vocab,
    )
    train_text_w2v_job.add_alias(f"{alias_prefix}/text_w2v_phonemes")

    tk.register_output(f"{alias_prefix}/text/phoneme_embeddings.npy", train_text_w2v_job.out_embeddings_npy)
    tk.register_output(f"{alias_prefix}/text/phoneme_vocab.txt", train_text_w2v_job.out_vocab_txt)

    # -------------------------------------------------------------
    # 2. AUDIO PIPELINE: HuBERT Layer 6 Feature Extraction
    # -------------------------------------------------------------
    audio_dir = tk.Path(os.path.join("/u/corpora/speech/LibriSpeech/LibriSpeech", librispeech_key))

    hubert_feat_job = ExtractHuBERTLayerFeaturesJob(
        audio_dir=audio_dir,
        model_name=hubert_model_name,
        layer=hubert_layer,
    )
    hubert_feat_job.add_alias(f"{alias_prefix}/hubert_features_l{hubert_layer}")

    # -------------------------------------------------------------
    # 3. CLUSTERING & WORD2VEC ABLATIONS: K in {P, 2P, 4P}
    # -------------------------------------------------------------
    P = base_phoneme_count
    cluster_multipliers = [1, 2, 4]

    alignment_results = {}

    for mult in cluster_multipliers:
        K = mult * P
        cluster_name = f"clus_{mult}P_{K}"

        # 3a. Cluster frames and collapse consecutive runs
        cluster_job = HuBERTClusterAndCollapseJob(
            features_npy=hubert_feat_job.out_features_npy,
            lengths_txt=hubert_feat_job.out_lengths_txt,
            seq_tags_txt=hubert_feat_job.out_seq_tags_txt,
            num_clusters=K,
        )
        cluster_job.add_alias(f"{alias_prefix}/{cluster_name}/kmeans_collapse")

        tk.register_output(
            f"{alias_prefix}/{cluster_name}/collapsed_tokens.txt",
            cluster_job.out_collapsed_tokens_txt,
        )

        # 3b. Train Audio Cluster Word2Vec
        train_audio_w2v_job = TrainWord2VecJob(
            token_text_file=cluster_job.out_collapsed_tokens_txt,
            vector_size=word2vec_dim,
            window=word2vec_window,
            min_count=1,
            sg=1,
            epochs=word2vec_epochs,
            fixed_vocab_file=cluster_job.out_cluster_vocab_txt,
        )
        train_audio_w2v_job.add_alias(f"{alias_prefix}/{cluster_name}/audio_w2v")

        tk.register_output(
            f"{alias_prefix}/{cluster_name}/audio_embeddings.npy",
            train_audio_w2v_job.out_embeddings_npy,
        )

        # -------------------------------------------------------------
        # 4. ARTETXE SELF-LEARNING EMBEDDING ALIGNMENT
        # -------------------------------------------------------------
        # Ablation 1: Frequency-Rank Matching Initialization
        align_freq_job = SelfLearningEmbeddingAlignmentJob(
            audio_embeddings_npy=train_audio_w2v_job.out_embeddings_npy,
            text_embeddings_npy=train_text_w2v_job.out_embeddings_npy,
            audio_vocab_file=cluster_job.out_cluster_vocab_txt,
            text_vocab_file=phoneme_vocab,
            audio_counts_file=cluster_job.out_cluster_counts_txt,
            text_counts_file=train_text_w2v_job.out_counts_txt,
            init_method="frequency",
            top_n_freq=min(K, P),
        )
        align_freq_job.add_alias(f"{alias_prefix}/{cluster_name}/align_init_freq")

        tk.register_output(
            f"{alias_prefix}/{cluster_name}/align_freq/dict.json",
            align_freq_job.out_dictionary_json,
        )
        tk.register_output(
            f"{alias_prefix}/{cluster_name}/align_freq/report.json",
            align_freq_job.out_alignment_report,
        )
        tk.register_output(
            f"{alias_prefix}/{cluster_name}/align_freq/W_star.npy",
            align_freq_job.out_mapping_matrix_npy,
        )

        # Ablation 2: Random Orthogonal Initializations (Seeds 42, 43, 44)
        for seed in [42, 43, 44]:
            align_rand_job = SelfLearningEmbeddingAlignmentJob(
                audio_embeddings_npy=train_audio_w2v_job.out_embeddings_npy,
                text_embeddings_npy=train_text_w2v_job.out_embeddings_npy,
                audio_vocab_file=cluster_job.out_cluster_vocab_txt,
                text_vocab_file=phoneme_vocab,
                audio_counts_file=cluster_job.out_cluster_counts_txt,
                text_counts_file=train_text_w2v_job.out_counts_txt,
                init_method="random",
                seed=seed,
            )
            align_rand_job.add_alias(f"{alias_prefix}/{cluster_name}/align_init_rand_seed{seed}")

            tk.register_output(
                f"{alias_prefix}/{cluster_name}/align_rand_seed{seed}/dict.json",
                align_rand_job.out_dictionary_json,
            )
            tk.register_output(
                f"{alias_prefix}/{cluster_name}/align_rand_seed{seed}/report.json",
                align_rand_job.out_alignment_report,
            )

        alignment_results[cluster_name] = {
            "cluster_job": cluster_job,
            "audio_w2v_job": train_audio_w2v_job,
            "align_freq_job": align_freq_job,
        }

    return alignment_results


if __name__ == "__main__":
    pipeline_jobs = build_self_learning_asr_alignment_pipeline()

