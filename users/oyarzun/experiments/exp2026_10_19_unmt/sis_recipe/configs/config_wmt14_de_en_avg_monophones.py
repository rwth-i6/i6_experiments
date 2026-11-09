"""
Config for WMT14 German-English UNMT:
German Words (50k) <-> English Words (50k) with averaged Monophone / Phoneme embeddings.
Symmetric word-to-word setup:
For each word in English, we average its constituent monophone (unigram phoneme) embeddings
from a monophone Word2Vec model, producing word-level codes for the top 50,000 most frequent
English words to align symmetrically with 50,000 German word codes.

Includes ablations across:
  1. English phonemized Word2Vec window size (win=5, win=7, win=10).
  2. Tokenization scheme (standard vs clean: lowercase + punctuation removed).
"""

from typing import Dict, Any, Optional
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def run_avg_monophone_setup(
    en_w2v_window: int = 10,
    clean_tok: bool = False,
    run_undreamt_training: bool = True,
) -> Dict[str, Any]:
    """
    Sets up a single ablation of the English monophone-averaged pipeline.
    """
    prefix = "exp2026_10_19_unmt/wmt14_de_en_avg_monophones"
    if clean_tok:
        prefix += "_clean_tok"
    if en_w2v_window != 10 or clean_tok:
        prefix += f"_win{en_w2v_window}"

    return run_wmt14_de_en_unmt_pipeline(
        prefix_name=prefix,
        train_name="undreamt_rnn_aed",
        de_code_type="word",
        en_code_type="avg_monophone",
        vocab_size=50000,
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0003,
        run_undreamt_training=run_undreamt_training,
        lowercase=clean_tok,
        remove_punctuation=clean_tok,
        en_w2v_window=en_w2v_window,
    )


def run_all_avg_monophone_ablations(run_undreamt_training: bool = True) -> Dict[str, Any]:
    """
    Runs the full grid of ablations:
      - Window sizes: [10, 7, 5]
      - Tokenization: standard vs clean_tok (lowercase + remove_punctuation)
    """
    results = {}
    for clean_tok in (False, True):
        tok_str = "clean_tok" if clean_tok else "standard"
        for win in (10, 7, 5):
            key = f"{tok_str}_win{win}"
            results[key] = run_avg_monophone_setup(
                en_w2v_window=win,
                clean_tok=clean_tok,
                run_undreamt_training=run_undreamt_training,
            )
    return results


def py(run_undreamt_training: bool = True):
    return run_all_avg_monophone_ablations(run_undreamt_training=run_undreamt_training)
