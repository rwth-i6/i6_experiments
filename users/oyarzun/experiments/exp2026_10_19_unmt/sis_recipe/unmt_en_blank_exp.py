"""
End-to-End Phoneme-Level Unsupervised Neural Machine Translation (UNMT) Pipeline for EN+Blank:
  - Setup 1 (`target_lang="en"`): Uncollapsed English Phonemes + Blank (`EN+Blank`) <-> Pure English ARPAbet Phonemes (`EN`).
  - Setup 2 (`target_lang="de"`): Uncollapsed English Phonemes + Blank (`EN+Blank`) <-> Pure German IPA Phonemes (`DE`).

Both setups implement the complete two-stage UNMT architecture (Artetxe et al., ICLR 2018 / UNdreaMT):
  Stage 1 — Cross-Modal Codes Learning & Alignment:
    - Extracts uncollapsed 20ms frame-level `EN+Blank` sequences (`BLK` + 39 ARPAbet phones, V=40)
      from zyang's LibriSpeech 960h GMM VAD-resampled HDFs (`phoneme_ref.{000..019}.hdf`).
    - Phonemizes WMT14 News Crawl monolingual text into pure ARPAbet (`EN`, V=39) or German-specific
      IPA (`DE`, V=46) monophone sequences.
    - Trains independent 300-dim Skip-gram Word2Vec embeddings (`sg=1, negative=10, dim=300`) on
      `EN+Blank` and on `EN` / `DE` phoneme corpora.
    - Aligns the cross-modal embedding spaces via `SelfLearningEmbeddingAlignmentJob` (orthogonal
      Procrustes self-learning with frequency-rank initialization and CSLS / cosine retrieval).
    - Evaluates the non-neural nearest-neighbor cross-modal code baseline (`WordByWordCrossLingualEvalJob`)
      with BLEU-4, Phoneme Error Rate (PER), and Sequence Length/Difference metrics.
  Stage 2 — Denoising Autoencoding + On-the-Fly Backtranslation (`UNdreaMT`):
    - Initializes `UNdreaMTModel` (shared 2-layer BiGRU encoder with frozen aligned cross-modal
      code embeddings + language-specific 2-layer StackedGRU attention decoders with input feeding).
    - Trains jointly with token-swap denoising autoencoding (`src->src`, `trg->trg`) and on-the-fly
      greedy backtranslation (`src -> syn_trg -> src`, `trg -> syn_src -> trg`) using asymmetric
      length ratios (`bt_src2trg_max_ratio=1.0`, `bt_trg2src_max_ratio=6.0`) tailored to uncollapsed
      50 Hz `EN+Blank` sequences.
    - Decodes test sets in both directions with beam search and scores BLEU-4, PER (exact & CTC-collapsed),
      sequence length statistics, and shared-encoder PCA state alignment visualizations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

from sisyphus import tk
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import SearchTakeBestJob
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset

from .cluster_adapter import (
    get_cluster,
    get_commonvoice_en_audio_dir,
    get_covost2_en_de_tsv,
    get_en_blank_dev_other_hdfs,
    get_en_blank_train_hdfs,
    get_hf_cache_dir,
    get_librispeech_corpus_dir,
    get_wav2vec2_phoneme_ctc_checkpoint,
    get_wav2vec2_phoneme_ctc_prior,
    get_wmt14_dir,
    set_cluster,
)
from .data.common import TrainingDatasets
from .data.en_blank import (
    BLANK_TOKEN,
    EN_BLANK_PHONEMES,
    ExtractCoVoST2AudioEnBlankJob,
    ExtractEnBlankFromPhonemeRefHdfJob,
    FilterAlignedParallelCorpusJob,
    FilterNonEmptyMonolingualTextJob,
    PrepareLibriSpeechDevOtherEnBlankEvalJob,
)
from .data.phonemize import (
    ENGLISH_ARPABET_PHONEMES,
    GERMAN_IPA_PHONEMES,
    PhonemizeEnglishTextJob,
    PhonemizeGermanTextJob,
    WriteFixedVocabJob,
)
from .data.wmt14_de_en import (
    DumpTextToReturnnHdfJob,
    PrepareWmt14MonolingualNewsCrawlJob,
)
from .data.word2vec import TrainWord2VecJob
from .bleu_eval import ComputeBleuJob
from .per_eval import ComputePerJob
from .eval_word_by_word_baseline import WordByWordCrossLingualEvalJob
from .eval_sequence_difference import SequenceDifferenceEvalJob
from .comet_eval import CometEvalJob
from .pipeline import training, forward_single, get_checkpoint
from .config import get_forward_config
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from ..models.self_learning_alignment import SelfLearningEmbeddingAlignmentJob
from ..models.analysis import EmbeddingAlignmentAnalysisJob


def run_en_blank_phoneme_unmt_pipeline(
    target_lang: str = "en",  # "en" (EN+Blank <-> EN) or "de" (EN+Blank <-> DE)
    prefix_name: Optional[str] = None,
    train_name: str = "undreamt_rnn_aed",
    cluster: Optional[str] = None,
    wmt14_dir: Optional[str] = None,
    news_crawl_years: Tuple[int, ...] = (2013,),
    max_mono_sentences: int = 2000000,
    unmt_train_src_sentences: Optional[int] = 278400,
    unmt_train_trg_sentences: int = 300000,
    max_en_blank_frames: int = 1000,
    dev_max_sentences: Optional[int] = 500,
    covost_dev_max_utterances: int = 500,
    covost_test_max_utterances: int = 2000,
    w2v_dim: int = 300,
    src_w2v_window: int = 25,
    trg_w2v_window: int = 10,
    w2v_negative: int = 10,
    w2v_sg: int = 1,
    w2v_epochs: int = 10,
    w2v_sample: Optional[float] = 1e-3,
    alignment_retrieval_metric: str = "csls",
    alignment_csls_k: int = 5,
    num_epochs: int = 20,
    eval_epochs: Optional[Sequence[int]] = None,
    partition_epoch: int = 10,
    batch_size: int = 6000,
    batch_max_seqs: int = 24,
    learning_rate: float = 0.0002,
    bt_src2trg_max_ratio: float = 1.0,
    bt_trg2src_max_ratio: float = 6.0,
    beam_size: int = 8,
    lp_alpha: float = 0.6,
    device: str = "gpu",
    run_undreamt_training: bool = True,
    enable_comet: bool = False,
) -> Dict[str, Any]:
    """
    Builds the complete Sisyphus graph for Phoneme-Level EN+Blank <-> EN or EN+Blank <-> DE UNMT.
    """
    assert target_lang in ("en", "de"), f"target_lang must be 'en' or 'de', got {target_lang!r}"
    if cluster is not None:
        set_cluster(cluster)
    active_cluster = get_cluster()

    if wmt14_dir is None:
        wmt14_dir = get_wmt14_dir(active_cluster)
    if prefix_name is None:
        prefix_name = f"exp2026_10_19_unmt/en_blank_{target_lang}_phoneme"

    if eval_epochs is None:
        eval_epochs = [5, 10, 15, num_epochs] if num_epochs >= 15 else [num_epochs]
    eval_epochs = sorted(set(e for e in eval_epochs if 1 <= e <= num_epochs))
    if not eval_epochs:
        eval_epochs = [num_epochs]

    # -------------------------------------------------------------------------
    # 1. Source Monolingual Corpus: Uncollapsed 20ms EN+Blank from LibriSpeech 960h
    # -------------------------------------------------------------------------
    en_blank_train_hdfs = get_en_blank_train_hdfs(active_cluster)
    extract_en_blank_job = ExtractEnBlankFromPhonemeRefHdfJob(
        hdf_files=en_blank_train_hdfs,
        collapse=False,
        min_frames=2,
        max_frames=max_en_blank_frames,
        max_sentences=None,
    )
    extract_en_blank_job.add_alias(f"{prefix_name}/data/extract_ls960_en_blank")
    tk.register_output(f"{prefix_name}/data/ls960.en_blank.txt", extract_en_blank_job.out_en_blank_text)
    tk.register_output(f"{prefix_name}/data/ls960.en_blank.stats.json", extract_en_blank_job.out_stats_json)
    src_train_txt = extract_en_blank_job.out_en_blank_text

    src_vocab_symbols = list(EN_BLANK_PHONEMES)  # 40 symbols: BLK + 39 ARPAbet
    src_vocab_size = len(src_vocab_symbols)
    fixed_src_vocab_job = WriteFixedVocabJob(symbols=src_vocab_symbols)
    fixed_src_vocab_job.add_alias(f"{prefix_name}/data/fixed_vocab_en_blank")

    # -------------------------------------------------------------------------
    # 2. Target Monolingual Corpus: Pure Phonemes (EN ARPAbet or DE IPA) from WMT14 News Crawl
    # -------------------------------------------------------------------------
    mono_raw_files = [
        tk.Path(os.path.join(wmt14_dir, "monolingual", f"news.{y}.{target_lang}.shuffled.gz"))
        for y in news_crawl_years
    ]
    mono_tok_job = PrepareWmt14MonolingualNewsCrawlJob(
        input_files=mono_raw_files,
        lang=target_lang,
        min_tokens=3,
        max_tokens=50,
        max_sentences=max_mono_sentences,
        lowercase=False,
    )
    mono_tok_job.add_alias(f"{prefix_name}/data/prepare_mono_{target_lang}_tok")
    tk.register_output(f"{prefix_name}/data/news_crawl.{target_lang}.tok.txt", mono_tok_job.out_tokenized_txt)

    if target_lang == "en":
        mono_phn_job = PhonemizeEnglishTextJob(text_file=mono_tok_job.out_tokenized_txt)
        trg_vocab_symbols = list(ENGLISH_ARPABET_PHONEMES)  # 39 ARPAbet symbols
    else:
        mono_phn_job = PhonemizeGermanTextJob(text_file=mono_tok_job.out_tokenized_txt, njobs=4)
        trg_vocab_symbols = list(GERMAN_IPA_PHONEMES)  # 46 German IPA symbols

    mono_phn_job.add_alias(f"{prefix_name}/data/phonemize_mono_{target_lang}")
    mono_phn_filter_job = FilterNonEmptyMonolingualTextJob(
        text_file=mono_phn_job.out_phoneme_text,
        min_tokens=2,
        max_tokens=400,
    )
    mono_phn_filter_job.add_alias(f"{prefix_name}/data/filter_mono_{target_lang}_phn")
    tk.register_output(f"{prefix_name}/data/news_crawl.{target_lang}.phn.txt", mono_phn_filter_job.out_text)
    trg_train_txt = mono_phn_filter_job.out_text

    trg_vocab_size = len(trg_vocab_symbols)
    fixed_trg_vocab_job = WriteFixedVocabJob(symbols=trg_vocab_symbols)
    fixed_trg_vocab_job.add_alias(f"{prefix_name}/data/fixed_vocab_{target_lang}_phn")

    # -------------------------------------------------------------------------
    # 3. Dev & Test Parallel Evaluation Sets
    #    - EN+Blank <-> EN: LibriSpeech dev-other (500 dev / ~2,364 test)
    #    - EN+Blank <-> DE: CoVoST-2 en_de Speech Translation (CommonVoice EN audio -> DE phonemes)
    # -------------------------------------------------------------------------
    eval_benchmark_name: str
    if target_lang == "en":
        eval_benchmark_name = "librispeech_dev_other"
        ls_eval_job = PrepareLibriSpeechDevOtherEnBlankEvalJob(
            dev_other_align_hdfs=get_en_blank_dev_other_hdfs(active_cluster),
            librispeech_corpus_dir=get_librispeech_corpus_dir(active_cluster),
            subset_name="dev-other",
            dev_split_size=dev_max_sentences or 500,
            max_frames=1200,
        )
        ls_eval_job.add_alias(f"{prefix_name}/data/prepare_librispeech_dev_other_eval")
        tk.register_output(f"{prefix_name}/data/eval_stats.json", ls_eval_job.out_stats_json)
        tk.register_output(f"{prefix_name}/data/test.en_blank.txt", ls_eval_job.out_test_en_blank_txt)
        tk.register_output(f"{prefix_name}/data/test.en.phn.txt", ls_eval_job.out_test_en_phn_txt)

        src_dev_txt = ls_eval_job.out_dev_en_blank_txt
        trg_dev_txt = ls_eval_job.out_dev_en_phn_txt
        src_test_txt = ls_eval_job.out_test_en_blank_txt
        trg_test_txt = ls_eval_job.out_test_en_phn_txt
        dev_tags_file = ls_eval_job.out_dev_utt_ids_txt
        test_tags_file = ls_eval_job.out_test_utt_ids_txt
    else:
        eval_benchmark_name = "covost2_en_de"
        covost_tsv = get_covost2_en_de_tsv(active_cluster)
        cv_audio_dir = get_commonvoice_en_audio_dir(active_cluster)
        w2v2_ckpt = get_wav2vec2_phoneme_ctc_checkpoint(active_cluster)
        w2v2_prior = get_wav2vec2_phoneme_ctc_prior(active_cluster)
        hf_cache = get_hf_cache_dir(active_cluster)

        covost_dev_audio_job = ExtractCoVoST2AudioEnBlankJob(
            covost2_tsv=covost_tsv,
            commonvoice_audio_dir=cv_audio_dir,
            wav2vec2_ctc_ckpt=w2v2_ckpt,
            wav2vec2_ctc_prior=w2v2_prior,
            hf_cache_dir=hf_cache,
            split="dev",
            max_utterances=covost_dev_max_utterances,
            prior_scale=0.7,
            max_frames=1200,
            use_gpu=(device == "gpu"),
        )
        covost_dev_audio_job.add_alias(f"{prefix_name}/data/extract_covost2_dev_en_blank")
        covost_dev_de_phn_job = PhonemizeGermanTextJob(
            text_file=covost_dev_audio_job.out_de_tok_txt,
            njobs=2,
        )
        covost_dev_de_phn_job.add_alias(f"{prefix_name}/data/phonemize_covost2_dev_de")
        covost_dev_aligned_job = FilterAlignedParallelCorpusJob(
            src_txt=covost_dev_audio_job.out_en_blank_txt,
            trg_txt=covost_dev_de_phn_job.out_phoneme_text,
            raw_trg_txt=covost_dev_audio_job.out_de_raw_txt,
            clip_ids_txt=covost_dev_audio_job.out_clip_ids_txt,
            min_src_len=2,
            min_trg_len=2,
        )
        covost_dev_aligned_job.add_alias(f"{prefix_name}/data/align_covost2_dev")

        covost_test_audio_job = ExtractCoVoST2AudioEnBlankJob(
            covost2_tsv=covost_tsv,
            commonvoice_audio_dir=cv_audio_dir,
            wav2vec2_ctc_ckpt=w2v2_ckpt,
            wav2vec2_ctc_prior=w2v2_prior,
            hf_cache_dir=hf_cache,
            split="test",
            max_utterances=covost_test_max_utterances,
            prior_scale=0.7,
            max_frames=1200,
            use_gpu=(device == "gpu"),
        )
        covost_test_audio_job.add_alias(f"{prefix_name}/data/extract_covost2_test_en_blank")
        covost_test_de_phn_job = PhonemizeGermanTextJob(
            text_file=covost_test_audio_job.out_de_tok_txt,
            njobs=2,
        )
        covost_test_de_phn_job.add_alias(f"{prefix_name}/data/phonemize_covost2_test_de")
        covost_test_aligned_job = FilterAlignedParallelCorpusJob(
            src_txt=covost_test_audio_job.out_en_blank_txt,
            trg_txt=covost_test_de_phn_job.out_phoneme_text,
            raw_trg_txt=covost_test_audio_job.out_de_raw_txt,
            clip_ids_txt=covost_test_audio_job.out_clip_ids_txt,
            min_src_len=2,
            min_trg_len=2,
        )
        covost_test_aligned_job.add_alias(f"{prefix_name}/data/align_covost2_test")
        tk.register_output(f"{prefix_name}/data/covost2_test.stats.json", covost_test_audio_job.out_stats_json)
        tk.register_output(f"{prefix_name}/data/test.en_blank.txt", covost_test_aligned_job.out_src_txt)
        tk.register_output(f"{prefix_name}/data/test.de.phn.txt", covost_test_aligned_job.out_trg_txt)

        src_dev_txt = covost_dev_aligned_job.out_src_txt
        trg_dev_txt = covost_dev_aligned_job.out_trg_txt
        src_test_txt = covost_test_aligned_job.out_src_txt
        trg_test_txt = covost_test_aligned_job.out_trg_txt
        dev_tags_file = covost_dev_aligned_job.out_clip_ids_txt
        test_tags_file = covost_test_aligned_job.out_clip_ids_txt

    # -------------------------------------------------------------------------
    # 4. Stage 1 — Cross-Modal Codes Learning (Independent 300-dim Skip-gram Word2Vec
    #    + Self-Learning Orthogonal Procrustes Alignment)
    # -------------------------------------------------------------------------
    w2v_src_job = TrainWord2VecJob(
        token_text_file=src_train_txt,
        vector_size=w2v_dim,
        window=src_w2v_window,
        min_count=1,
        max_vocab_size=None,
        sg=w2v_sg,
        negative=w2v_negative,
        epochs=w2v_epochs,
        sample=w2v_sample,
        workers=8,
        fixed_vocab_file=fixed_src_vocab_job.out_vocab_txt,
    )
    w2v_src_job.add_alias(f"{prefix_name}/word2vec/en_blank_win{src_w2v_window}_d{w2v_dim}")
    tk.register_output(f"{prefix_name}/word2vec/en_blank_embeddings.vec", w2v_src_job.out_embeddings_vec)
    tk.register_output(f"{prefix_name}/word2vec/en_blank_vocab.txt", w2v_src_job.out_vocab_txt)
    tk.register_output(f"{prefix_name}/word2vec/en_blank_counts.txt", w2v_src_job.out_counts_txt)

    w2v_trg_job = TrainWord2VecJob(
        token_text_file=trg_train_txt,
        vector_size=w2v_dim,
        window=trg_w2v_window,
        min_count=1,
        max_vocab_size=None,
        sg=w2v_sg,
        negative=w2v_negative,
        epochs=w2v_epochs,
        sample=w2v_sample,
        workers=8,
        fixed_vocab_file=fixed_trg_vocab_job.out_vocab_txt,
    )
    w2v_trg_job.add_alias(f"{prefix_name}/word2vec/{target_lang}_phn_win{trg_w2v_window}_d{w2v_dim}")
    tk.register_output(f"{prefix_name}/word2vec/{target_lang}_phn_embeddings.vec", w2v_trg_job.out_embeddings_vec)
    tk.register_output(f"{prefix_name}/word2vec/{target_lang}_phn_vocab.txt", w2v_trg_job.out_vocab_txt)
    tk.register_output(f"{prefix_name}/word2vec/{target_lang}_phn_counts.txt", w2v_trg_job.out_counts_txt)

    align_job = SelfLearningEmbeddingAlignmentJob(
        src_embeddings_npy=w2v_src_job.out_embeddings_npy,
        trg_embeddings_npy=w2v_trg_job.out_embeddings_npy,
        src_vocab_file=w2v_src_job.out_vocab_txt,
        trg_vocab_file=w2v_trg_job.out_vocab_txt,
        src_counts_file=w2v_src_job.out_counts_txt,
        trg_counts_file=w2v_trg_job.out_counts_txt,
        init_method="frequency",
        max_iters=100,
        tol=1e-6,
        retrieval_metric=alignment_retrieval_metric,
        csls_k=alignment_csls_k,
    )
    align_job.add_alias(f"{prefix_name}/alignment/en_blank_{target_lang}_freq_{alignment_retrieval_metric}")
    tk.register_output(f"{prefix_name}/alignment/mapped_en_blank_embeddings.npy", align_job.out_mapped_src_emb_npy)
    tk.register_output(f"{prefix_name}/alignment/normalized_{target_lang}_embeddings.npy", align_job.out_normalized_trg_emb_npy)
    tk.register_output(f"{prefix_name}/alignment/mapped_en_blank_embeddings.vec", align_job.out_mapped_src_vec_txt)
    tk.register_output(f"{prefix_name}/alignment/normalized_{target_lang}_embeddings.vec", align_job.out_normalized_trg_vec_txt)
    tk.register_output(f"{prefix_name}/alignment/alignment_report.json", align_job.out_alignment_report)
    tk.register_output(f"{prefix_name}/alignment/en_blank_to_{target_lang}_dict.json", align_job.out_dictionary_json)

    align_analysis_job = EmbeddingAlignmentAnalysisJob(
        mapped_audio_embeddings_npy=align_job.out_mapped_src_emb_npy,
        text_embeddings_npy=align_job.out_normalized_trg_emb_npy,
        dictionary_json=align_job.out_dictionary_json,
        raw_src_embeddings_npy=w2v_src_job.out_embeddings_npy,
        raw_trg_embeddings_npy=w2v_trg_job.out_embeddings_npy,
        audio_vocab_file=w2v_src_job.out_returnn_vocab,
        text_vocab_file=w2v_trg_job.out_returnn_vocab,
        audio_counts_file=w2v_src_job.out_counts_txt,
        text_counts_file=w2v_trg_job.out_counts_txt,
        alignment_report_json=align_job.out_alignment_report,
        title_prefix=f"Cross-Modal 2D PCA (en_blank vs {target_lang}_phn)",
        src_label="EN+Blank Codes",
        trg_label=f"{target_lang.upper()} Phonemes",
        _version=4,
    )
    align_analysis_job.add_alias(f"{prefix_name}/alignment/pca_analysis")
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/cross_modal_pca.png", align_analysis_job.out_cross_modal_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/source_modality_pca.png", align_analysis_job.out_source_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/target_modality_pca.png", align_analysis_job.out_target_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/cross_modal_pca_panel.png", align_analysis_job.out_cross_modal_panel_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/source_raw_pca.png", align_analysis_job.out_source_raw_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/target_raw_pca.png", align_analysis_job.out_target_raw_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/unaligned_cross_modal_pca.png", align_analysis_job.out_unaligned_joint_pca_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/alignment_comparison_panel.png", align_analysis_job.out_alignment_comparison_panel_plot)
    tk.register_output(f"{prefix_name}/alignment/pca_analysis/analysis_report.json", align_analysis_job.out_analysis_report_json)

    # -------------------------------------------------------------------------
    # 5. Cross-Modal Code Nearest-Neighbor Baseline Evaluation (Stage 1 Eval)
    # -------------------------------------------------------------------------
    src2trg_tag = f"en_blank2{target_lang}"
    trg2src_tag = f"{target_lang}2en_blank"

    for metric in ("cosine", "csls"):
        # en_blank -> target_lang
        wbw_s2t = WordByWordCrossLingualEvalJob(
            src_embeddings_npy=align_job.out_mapped_src_emb_npy,
            trg_embeddings_npy=align_job.out_normalized_trg_emb_npy,
            src_vocab_file=w2v_src_job.out_vocab_txt,
            trg_vocab_file=w2v_trg_job.out_vocab_txt,
            src_test_txt=src_test_txt,
            trg_ref_txt=trg_test_txt,
            direction="src2trg",
            retrieval_metric=metric,
            csls_k=alignment_csls_k,
            copy_oov=False,
        )
        wbw_s2t.add_alias(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/bleu", wbw_s2t.out_bleu)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/report", wbw_s2t.out_bleu_report_txt)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/hypotheses.txt", wbw_s2t.out_hypotheses_txt)

        per_wbw_s2t = ComputePerJob(
            hyp_file=wbw_s2t.out_hypotheses_txt,
            ref_file=trg_test_txt,
            remove_unk=True,
            collapse_blank_token=None,
        )
        per_wbw_s2t.add_alias(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/per_eval")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/per", per_wbw_s2t.out_per)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/per_report.txt", per_wbw_s2t.out_report)

        # Also evaluate CTC-collapsed nearest-neighbor hypothesis (collapsing repeated frame predictions)
        per_wbw_s2t_col = ComputePerJob(
            hyp_file=wbw_s2t.out_hypotheses_txt,
            ref_file=trg_test_txt,
            remove_unk=True,
            collapse_blank_token=BLANK_TOKEN,
        )
        per_wbw_s2t_col.add_alias(f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/per_collapsed_eval")
        tk.register_output(
            f"{prefix_name}/baseline_word_by_word/{src2trg_tag}_{metric}/per_collapsed_summary.json",
            per_wbw_s2t_col.out_summary_json,
        )

        # target_lang -> en_blank
        wbw_t2s = WordByWordCrossLingualEvalJob(
            src_embeddings_npy=align_job.out_mapped_src_emb_npy,
            trg_embeddings_npy=align_job.out_normalized_trg_emb_npy,
            src_vocab_file=w2v_src_job.out_vocab_txt,
            trg_vocab_file=w2v_trg_job.out_vocab_txt,
            src_test_txt=trg_test_txt,
            trg_ref_txt=src_test_txt,
            direction="trg2src",
            retrieval_metric=metric,
            csls_k=alignment_csls_k,
            copy_oov=False,
        )
        wbw_t2s.add_alias(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}/bleu", wbw_t2s.out_bleu)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}/report", wbw_t2s.out_bleu_report_txt)

        per_wbw_t2s = ComputePerJob(
            hyp_file=wbw_t2s.out_hypotheses_txt,
            ref_file=src_test_txt,
            remove_unk=True,
            collapse_blank_token=BLANK_TOKEN,
        )
        per_wbw_t2s.add_alias(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}/per_eval")
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}/per", per_wbw_t2s.out_per)
        tk.register_output(f"{prefix_name}/baseline_word_by_word/{trg2src_tag}_{metric}/per_report.txt", per_wbw_t2s.out_report)

    # -------------------------------------------------------------------------
    # 6. Stage 2 — UNdreaMT Denoising + On-the-Fly Backtranslation Training
    # -------------------------------------------------------------------------
    train_job = None
    pca_jobs = None

    if run_undreamt_training:
        total_src_vocab_dim = src_vocab_size + 4  # PAD=0, OOV=1, EOS=2, SOS=3, tokens=4..V+3
        total_trg_vocab_dim = trg_vocab_size + 4

        train_src_hdf_job = DumpTextToReturnnHdfJob(
            text_file=src_train_txt,
            returnn_vocab_file=w2v_src_job.out_returnn_vocab,
            max_sentences=unmt_train_src_sentences,
            seq_tag_prefix="line",
        )
        train_src_hdf_job.add_alias(f"{prefix_name}/hdf/train_mono_en_blank")

        train_trg_hdf_job = DumpTextToReturnnHdfJob(
            text_file=trg_train_txt,
            returnn_vocab_file=w2v_trg_job.out_returnn_vocab,
            max_sentences=unmt_train_trg_sentences,
            seq_tag_prefix="line",
        )
        train_trg_hdf_job.add_alias(f"{prefix_name}/hdf/train_mono_{target_lang}_phn")

        dev_src_hdf_job = DumpTextToReturnnHdfJob(
            text_file=src_dev_txt,
            returnn_vocab_file=w2v_src_job.out_returnn_vocab,
            max_sentences=dev_max_sentences,
            seq_tag_prefix="line",
            seq_tags_file=dev_tags_file,
        )
        dev_trg_hdf_job = DumpTextToReturnnHdfJob(
            text_file=trg_dev_txt,
            returnn_vocab_file=w2v_trg_job.out_returnn_vocab,
            max_sentences=dev_max_sentences,
            seq_tag_prefix="line",
            seq_tags_file=dev_tags_file,
        )

        test_src_hdf_job = DumpTextToReturnnHdfJob(
            text_file=src_test_txt,
            returnn_vocab_file=w2v_src_job.out_returnn_vocab,
            seq_tag_prefix="line",
            seq_tags_file=test_tags_file,
        )
        test_trg_hdf_job = DumpTextToReturnnHdfJob(
            text_file=trg_test_txt,
            returnn_vocab_file=w2v_trg_job.out_returnn_vocab,
            seq_tag_prefix="line",
            seq_tags_file=test_tags_file,
        )

        train_dataset = MetaDataset(
            datasets={
                "src_mono": HdfDataset(
                    files=[train_src_hdf_job.out_hdf],
                    partition_epoch=partition_epoch,
                    seq_ordering="random",
                ),
                "trg_mono": HdfDataset(
                    files=[train_trg_hdf_job.out_hdf],
                    partition_epoch=partition_epoch,
                    seq_ordering="random",
                ),
            },
            data_map={
                "data": ("src_mono", "data"),
                "target": ("trg_mono", "data"),
            },
            seq_order_control_dataset="src_mono",
        )

        dev_dataset = MetaDataset(
            datasets={
                "src_dev": HdfDataset(
                    files=[dev_src_hdf_job.out_hdf],
                    partition_epoch=1,
                    seq_ordering="sorted_reverse",
                ),
                "trg_dev": HdfDataset(
                    files=[dev_trg_hdf_job.out_hdf],
                    partition_epoch=1,
                    seq_ordering="sorted_reverse",
                ),
            },
            data_map={
                "data": ("src_dev", "data"),
                "target": ("trg_dev", "data"),
            },
            seq_order_control_dataset="src_dev",
        )

        test_s2t_dataset = MetaDataset(
            datasets={
                "src_test": HdfDataset(files=[test_src_hdf_job.out_hdf]),
                "trg_test": HdfDataset(files=[test_trg_hdf_job.out_hdf]),
            },
            data_map={
                "data": ("src_test", "data"),
                "target": ("trg_test", "data"),
            },
            seq_order_control_dataset="src_test",
        )

        test_t2s_dataset = MetaDataset(
            datasets={
                "trg_test": HdfDataset(files=[test_trg_hdf_job.out_hdf]),
                "src_test": HdfDataset(files=[test_src_hdf_job.out_hdf]),
            },
            data_map={
                "data": ("trg_test", "data"),
                "target": ("src_test", "data"),
            },
            seq_order_control_dataset="trg_test",
        )

        datastreams = {
            "data": LabelDatastream(
                available_for_inference=True,
                vocab=w2v_src_job.out_returnn_vocab,
                vocab_size=total_src_vocab_dim,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=w2v_trg_job.out_returnn_vocab,
                vocab_size=total_trg_vocab_dim,
            ),
        }

        training_datasets = TrainingDatasets(
            train=train_dataset,
            eval_datasets={"dev": dev_dataset},
            datastreams=datastreams,
        )

        network_module = "definitions.undreamt_rnn_aed_v1.Model"
        train_step_module = "train_steps.undreamt_backtranslation_train_step.train_step"
        forward_step_module = "recognition.undreamt_mt.forward_step.forward_step"
        callback_module = "recognition.undreamt_mt.callback.RecognitionToTextDictCallback"

        net_args = {
            "src_vocab_size": src_vocab_size,
            "trg_vocab_size": trg_vocab_size,
            "embedding_size": w2v_dim,
            "hidden_size": 600,
            "layers": 2,
            "dropout": 0.3,
            "param_init": 0.1,
            "src_pretrained_emb_path": align_job.out_mapped_src_emb_npy,
            "trg_pretrained_emb_path": align_job.out_normalized_trg_emb_npy,
            "freeze_encoder_embeddings": True,
            "freeze_decoder_embeddings": False,
            "use_embedding_generator": False,
            "alignment_function": "general",
        }

        train_step_args = {
            "default_data_key": "data",
            "default_target_key": "target",
            "enable_denoising": True,
            "enable_backtranslation": True,
            "denoise_backtranslation": True,
            "bt_max_ratio": 2.0,
            "bt_src2trg_max_ratio": bt_src2trg_max_ratio,
            "bt_trg2src_max_ratio": bt_trg2src_max_ratio,
            "denoise_scale": 1.0,
            "bt_scale": 1.0,
        }

        returnn_train_config_dict = {
            "default_data_key": "data",
            "default_target_key": "target",
            "batch_size": batch_size,
            "max_seqs": batch_max_seqs,
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "learning_rate": learning_rate,
            "gradient_clip_global_norm": 5.0,
        }

        train_rqmt: Dict[str, Any] = {
            "mem_rqmt": 24,
            "time_rqmt": 24,
            "cpu_rqmt": 4,
            "gpu_mem": 24,
        }
        if device != "gpu":
            train_rqmt["device"] = device

        train_args = {
            "config": returnn_train_config_dict,
            "post_config": {
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 4,
                    "keep": sorted(set(list(eval_epochs) + [num_epochs])),
                }
            },
            "python_prolog": [
                "import os",
                "import sys",
                'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"',
            ],
            "network_module": network_module,
            "train_step_module": train_step_module,
            "net_args": net_args,
            "train_args": train_step_args,
            "rqmt": train_rqmt,
        }

        undreamt_train_name = f"{prefix_name}/{train_name}"
        train_job = training(
            training_name=undreamt_train_name,
            datasets=training_datasets,
            train_args=train_args,
            num_epochs=num_epochs,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
        )

        swapped_datastreams = {
            "data": datastreams["target"],
            "target": datastreams["data"],
        }

        # Evaluate checkpoints in both directions (en_blank -> trg and trg -> en_blank)
        for eval_ep in eval_epochs:
            ckpt = get_checkpoint(
                training_name=undreamt_train_name,
                train_job=train_job,
                get_specific_checkpoint=eval_ep,
            )

            for (
                direction_name,
                src_lang_id,
                trg_lang_id,
                test_ds,
                fwd_datastreams,
                dir_max_ratio,
                src_file,
                ref_file,
                collapse_blk,
            ) in [
                (
                    src2trg_tag,
                    "src",
                    "trg",
                    test_s2t_dataset,
                    datastreams,
                    bt_src2trg_max_ratio,
                    src_test_txt,
                    trg_test_txt,
                    None,
                ),
                (
                    trg2src_tag,
                    "trg",
                    "src",
                    test_t2s_dataset,
                    swapped_datastreams,
                    bt_trg2src_max_ratio,
                    trg_test_txt,
                    src_test_txt,
                    BLANK_TOKEN,
                ),
            ]:
                fwd_cfg = get_forward_config(
                    config={
                        "default_data_key": "data",
                        "default_target_key": "target",
                        "batch_size": min(batch_size, 4000),
                        "max_seqs": 16,
                    },
                    network_module=network_module,
                    extra_config=ReturnnConfig(
                        config={},
                        python_prolog=[
                            "import os, sys",
                            'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                        ],
                    ),
                    net_args=net_args,
                    decoder=forward_step_module,
                    callback_module=callback_module,
                    callback_opts={"include_beam": True},
                    decoder_args={
                        "beam_size": beam_size,
                        "lp_alpha": lp_alpha,
                        "max_ratio": dir_max_ratio,
                        "src_lang": src_lang_id,
                        "trg_lang": trg_lang_id,
                        "input_data_key": "data",
                    },
                    datastreams=fwd_datastreams,
                    vocab_key="target",
                )

                recog_prefix = f"{undreamt_train_name}/recog_{direction_name}_ep{eval_ep}/{eval_benchmark_name}"
                search_out = forward_single(
                    prefix_name=recog_prefix,
                    returnn_config=fwd_cfg,
                    checkpoint=ckpt,
                    dataset_dict=test_ds.as_returnn_opts(),
                    returnn_exe=RETURNN_EXE,
                    returnn_root=RETURNN_ROOT,
                    rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 16, "time": 2},
                )
                best_hyp = SearchTakeBestJob(search_out, output_gzip=True).out_best_search_results

                bleu_job = ComputeBleuJob(
                    hyp_file=best_hyp,
                    ref_file=ref_file,
                    lowercase=False,
                    remove_unk=True,
                )
                bleu_job.add_alias(f"{recog_prefix}/bleu_eval")
                tk.register_output(f"{recog_prefix}/bleu", bleu_job.out_bleu)
                tk.register_output(f"{recog_prefix}/bleu_report.txt", bleu_job.out_report)
                tk.register_output(f"{recog_prefix}/hypotheses.txt", bleu_job.out_hyp_txt)

                per_job = ComputePerJob(
                    hyp_file=best_hyp,
                    ref_file=ref_file,
                    remove_unk=True,
                    collapse_blank_token=collapse_blk,
                )
                per_job.add_alias(f"{recog_prefix}/per_eval")
                tk.register_output(f"{recog_prefix}/per", per_job.out_per)
                tk.register_output(f"{recog_prefix}/per_report.txt", per_job.out_report)
                tk.register_output(f"{recog_prefix}/per_summary.json", per_job.out_summary_json)

                seq_diff_job = SequenceDifferenceEvalJob(
                    input_txt=src_file,
                    hypothesis_txt=best_hyp,
                    reference_txt=ref_file,
                    name=f"{direction_name}_ep{eval_ep}",
                    input_lang="en_blank" if src_lang_id == "src" else target_lang,
                    target_lang=target_lang if trg_lang_id == "trg" else "en_blank",
                )
                seq_diff_job.add_alias(f"{recog_prefix}/length_analysis")
                tk.register_output(f"{recog_prefix}/length_analysis/report.txt", seq_diff_job.out_report_txt)
                tk.register_output(f"{recog_prefix}/length_analysis/summary.json", seq_diff_job.out_summary_json)

                if enable_comet:
                    comet_job = CometEvalJob(
                        src_file=src_file,
                        hyp_file=best_hyp,
                        ref_file=ref_file,
                        name=f"{direction_name}_ep{eval_ep}",
                        gpus=1 if device == "gpu" else 0,
                    )
                    comet_job.add_alias(f"{recog_prefix}/comet_eval")
                    tk.register_output(f"{recog_prefix}/comet", comet_job.out_comet)

        # Shared Encoder PCA State Visualization (EN+Blank vs EN / DE)
        from .analysis import analyze_encoder_states

        pca_jobs = analyze_encoder_states(
            config={
                "default_data_key": "data",
                "default_target_key": "target",
                "batch_size": 3000,
                "max_seqs": 16,
            },
            training_name=undreamt_train_name,
            train_job=train_job,
            train_args=train_args,
            train_data=training_datasets,
            test_data_dict={eval_benchmark_name: test_s2t_dataset, f"{eval_benchmark_name}_dev": dev_dataset},
            checkpoints=sorted(set(list(eval_epochs) + [num_epochs])),
            analysis_name=f"vis_encoder_pca_en_blank_vs_{target_lang}",
            cosine_similarity_summary=True,
            src_label="en_blank",
            trg_label=target_lang,
            src_vocab_file=w2v_src_job.out_returnn_vocab,
            trg_vocab_file=w2v_trg_job.out_returnn_vocab,
            src_text_file=src_test_txt,
            trg_text_file=trg_test_txt,
            seq_tags_file=test_tags_file,
            extra_forward_config=ReturnnConfig(
                config={},
                python_prolog=[
                    "import os, sys",
                    'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                ],
            ),
            rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 16, "time": 2},
        )

        pca_hyp_jobs = analyze_encoder_states(
            config={
                "default_data_key": "data",
                "default_target_key": "target",
                "batch_size": 3000,
                "max_seqs": 16,
            },
            training_name=undreamt_train_name,
            train_job=train_job,
            train_args=train_args,
            train_data=training_datasets,
            test_data_dict={eval_benchmark_name: test_s2t_dataset, f"{eval_benchmark_name}_dev": dev_dataset},
            checkpoints=sorted(set(list(eval_epochs) + [num_epochs])),
            analysis_name=f"vis_encoder_pca_en_blank_vs_hyp_{target_lang}",
            compare_to_hypothesis=True,
            cosine_similarity_summary=True,
            src_label="en_blank",
            trg_label=f"{target_lang} (hyp)",
            src_vocab_file=w2v_src_job.out_returnn_vocab,
            trg_vocab_file=w2v_trg_job.out_returnn_vocab,
            src_text_file=src_test_txt,
            trg_text_file=trg_test_txt,
            seq_tags_file=test_tags_file,
            extra_forward_config=ReturnnConfig(
                config={},
                python_prolog=[
                    "import os, sys",
                    'os.environ["PYTHONPATH"] = ":".join(sys.path)',
                ],
            ),
            rqmt={"device": device, "gpu": 1 if device == "gpu" else 0, "cpu": 4, "mem": 16, "time": 2},
        )

    return {
        "extract_en_blank_job": extract_en_blank_job,
        "mono_tok_job": mono_tok_job,
        "mono_phn_job": mono_phn_job,
        "w2v_src_job": w2v_src_job,
        "w2v_trg_job": w2v_trg_job,
        "align_job": align_job,
        "train_job": train_job,
        "pca_jobs": pca_jobs,
        "pca_hyp_jobs": pca_hyp_jobs,
    }
