"""
Config for WMT14 German-English UNMT:
German Words (50k) <-> English Words (50k) with averaged Triphone embeddings.
For each word in English, we average its constituent triphone embeddings from a triphone Word2Vec model,
producing word-level codes for the top 50,000 most frequent English words to align with 50,000 German word codes.
"""

from typing import Optional
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py(run_undreamt_training: bool = True):
    return run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_avg_triphones",
        train_name="undreamt_rnn_aed",
        en_code_type="avg_triphone",
        vocab_size=50000,
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0003,
        run_undreamt_training=run_undreamt_training,
    )

