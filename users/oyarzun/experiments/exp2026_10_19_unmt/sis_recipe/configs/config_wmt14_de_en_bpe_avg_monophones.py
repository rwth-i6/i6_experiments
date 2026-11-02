"""
Config for WMT14 German-English UNMT:
German BPE 50k <-> English Words (50k) with averaged Monophone / Phoneme embeddings.

- German (de): Segmented with 50,000 Byte-Pair Encoding (BPE) subword operations.
- English (en): 50,000 word-level codes constructed by decomposing each English word into its
  constituent monophones (single ARPAbet phonemes) and averaging their Word2Vec embeddings.
"""

from typing import Optional
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py(run_undreamt_training: bool = True):
    return run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_bpe_en_avg_monophones",
        train_name="undreamt_rnn_aed",
        de_code_type="bpe",
        en_code_type="avg_monophone",
        bpe_operations=50000,
        bpe_mode="separate",
        vocab_size=50000,
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0003,
        run_undreamt_training=run_undreamt_training,
    )

