"""
Sisyphus Configuration Entrypoint for Phoneme-Level EN+Blank UNMT Experiments:
  1. EN+Blank <-> EN Phonemes UNMT:
     - Source: Uncollapsed 20ms English phoneme + blank frame sequences (`BLK` + 39 ARPAbet, V=40)
       from LibriSpeech 960h GMM VAD-resampled alignments (`phoneme_ref.{000..019}.hdf`).
     - Target: Pure English ARPAbet monophones (V=39) from WMT14 English News Crawl.
     - Eval: LibriSpeech `dev-other` (`EN+Blank <-> EN`).
  2. EN+Blank <-> DE Phonemes UNMT:
     - Source: Uncollapsed 20ms English phoneme + blank frame sequences (`BLK` + 39 ARPAbet, V=40)
       from LibriSpeech 960h (`phoneme_ref.{000..019}.hdf`).
     - Target: Pure German-specific IPA monophones (V=46) from WMT14 German News Crawl.
     - Eval: CoVoST-2 English-to-German Speech Translation (`EN+Blank <-> DE`).

Both pipelines execute:
  - Stage 1: Cross-modal 300-dim Skip-gram Word2Vec + Orthogonal Procrustes alignment + Nearest-Neighbor baseline.
  - Stage 2: UNdreaMT denoising autoencoding + on-the-fly greedy backtranslation + beam search decoding.
"""

from typing import Any, Dict, Optional
from ..unmt_en_blank_exp import run_en_blank_phoneme_unmt_pipeline


def run_en_blank_to_en_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
) -> Dict[str, Any]:
    """ Launches the EN+Blank <-> EN phoneme UNMT setup (cross-modal codes + UNdreaMT denoising/BT). """
    return run_en_blank_phoneme_unmt_pipeline(
        target_lang="en",
        prefix_name="exp2026_10_19_unmt/en_blank_en_phoneme_d300",
        train_name="undreamt_rnn_aed_20ep",
        cluster=cluster,
        news_crawl_years=(2013,),
        max_mono_sentences=2000000,
        unmt_train_src_sentences=278400,
        unmt_train_trg_sentences=300000,
        max_en_blank_frames=1000,
        w2v_dim=300,
        src_w2v_window=25,
        trg_w2v_window=10,
        w2v_negative=10,
        w2v_sg=1,
        w2v_epochs=10,
        w2v_sample=1e-3,
        alignment_retrieval_metric="csls",
        alignment_csls_k=5,
        num_epochs=20,
        eval_epochs=[5, 10, 15, 20],
        partition_epoch=10,
        batch_size=6000,
        batch_max_seqs=24,
        learning_rate=0.0002,
        bt_src2trg_max_ratio=1.0,
        bt_trg2src_max_ratio=6.0,
        beam_size=8,
        lp_alpha=0.6,
        device="gpu",
        run_undreamt_training=run_undreamt_training,
        enable_comet=False,
    )


def run_en_blank_to_de_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
) -> Dict[str, Any]:
    """ Launches the EN+Blank <-> DE phoneme UNMT setup (cross-modal codes + UNdreaMT denoising/BT). """
    return run_en_blank_phoneme_unmt_pipeline(
        target_lang="de",
        prefix_name="exp2026_10_19_unmt/en_blank_de_phoneme_d300",
        train_name="undreamt_rnn_aed_20ep",
        cluster=cluster,
        news_crawl_years=(2013,),
        max_mono_sentences=2000000,
        unmt_train_src_sentences=278400,
        unmt_train_trg_sentences=300000,
        max_en_blank_frames=1000,
        covost_dev_max_utterances=500,
        covost_test_max_utterances=2000,
        w2v_dim=300,
        src_w2v_window=25,
        trg_w2v_window=10,
        w2v_negative=10,
        w2v_sg=1,
        w2v_epochs=10,
        w2v_sample=1e-3,
        alignment_retrieval_metric="csls",
        alignment_csls_k=5,
        num_epochs=20,
        eval_epochs=[5, 10, 15, 20],
        partition_epoch=10,
        batch_size=6000,
        batch_max_seqs=24,
        learning_rate=0.0002,
        bt_src2trg_max_ratio=1.0,
        bt_trg2src_max_ratio=6.0,
        beam_size=8,
        lp_alpha=0.6,
        device="gpu",
        run_undreamt_training=run_undreamt_training,
        enable_comet=False,
    )


def run_all_en_blank_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Entrypoint registering both:
      1. EN+Blank <-> EN phoneme UNMT
      2. EN+Blank <-> DE phoneme UNMT
    """
    en_res = run_en_blank_to_en_phoneme_unmt(cluster=cluster, run_undreamt_training=run_undreamt_training)
    de_res = run_en_blank_to_de_phoneme_unmt(cluster=cluster, run_undreamt_training=run_undreamt_training)
    return {
        "en_blank_en": en_res,
        "en_blank_de": de_res,
    }


py = run_all_en_blank_phoneme_unmt
