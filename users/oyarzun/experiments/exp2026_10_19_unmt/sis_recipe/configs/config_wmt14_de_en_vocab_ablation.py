"""
Config for WMT14 German-English UNMT Vocabulary Size Ablation:
Ablations reducing vocabulary size (number of words) from English and German
to the (10k, 1k, 100) most frequent words.

Each ablation sets up both:
  1. Initialization ("init"):
     - Monolingual Skip-gram Word2Vec embeddings (dim=300, negative=10, window=10)
       trained with max_vocab_size in {10000, 1000, 100} on German and English.
     - Self-learning cross-lingual embedding alignment (numeral-initialized with frequency fallback).
     - Cross-lingual word-by-word baseline translation evaluation (BLEU-4, COMET, sequence difference)
       on newstest2014 in both directions (de->en and en->de).
  2. Posterior Denoising-Backtranslation Training ("posterior"):
     - Full UNdreaMT 2-layer BiGRU encoder + 2-layer StackedGRU attention decoders initialized
       with the aligned cross-lingual word embeddings.
     - Word-swap denoising autoencoding + on-the-fly greedy backtranslation training (100 epochs).
     - Periodic checkpoint evaluation (epochs 20, 40, 60, 80, 100) on newstest2014 (de->en and en->de)
       with beam search (beam_size=12, lp_alpha=0.6), BLEU-4, COMET, length analysis, and encoder PCA.
"""

from typing import Dict, Any
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


VOCAB_SIZE_ABLATIONS = [
    ("10k", 10000),
    ("1k", 1000),
    ("100", 100),
]


def run_vocab_ablation(vocab_tag: str, vocab_size: int) -> Dict[str, Any]:
    """
    Sets up a single vocabulary ablation for German and English.
    """
    return run_wmt14_de_en_unmt_pipeline(
        prefix_name=f"exp2026_10_19_unmt/wmt14_de_en_vocab{vocab_tag}",
        train_name="undreamt_rnn_aed",
        vocab_size=vocab_size,
        en_vocab_size=vocab_size,
        de_code_type="word",
        en_code_type="word",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0002,
        run_undreamt_training=True,
    )


def py():
    """
    Entrypoint called by config/__init__.py or Sisyphus manager.
    Runs all 3 vocabulary size ablations: 10k, 1k, and 100 words.
    """
    jobs = {}
    for v_tag, v_size in VOCAB_SIZE_ABLATIONS:
        jobs[v_tag] = run_vocab_ablation(vocab_tag=v_tag, vocab_size=v_size)
    return jobs
