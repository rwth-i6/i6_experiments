"""
Config for WMT14 German-English UNMT:
German Words (50k) <-> English 50k Cluster Indices (from averaged monophones).

Ablation 1: K = 50,000
  1. Decompose English words into constituent monophones and train monophone Word2Vec.
  2. Compute averaged monophone representations for English words.
  3. Cluster word representations into K=50,000 centroids using MiniBatchKMeans.
  4. Quantize English News Crawl into cluster index tokens (c0..c49999).
  5. Train new 300-dim Skip-gram Word2Vec on the English cluster tokens.
  6. German vocabulary matched to 50,000 words with Word2Vec.
  7. Cross-lingual alignment via frequency-based initialization.
  8. UNdreaMT training, beam search decoding, and word-level BLEU evaluation (clusters mapped back to words).
"""

from typing import Optional
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py(run_undreamt_training: bool = True):
    return run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_avg_monophone_cluster_50k",
        train_name="undreamt_rnn_aed",
        de_code_type="word",
        en_code_type="cluster_avg_monophone",
        vocab_size=50000,
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0003,
        run_undreamt_training=run_undreamt_training,
    )

