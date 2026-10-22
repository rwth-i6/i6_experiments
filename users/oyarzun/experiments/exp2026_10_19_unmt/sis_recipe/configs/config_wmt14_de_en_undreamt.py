"""
Experiment config entrypoint for WMT14 German-English Unsupervised NMT (UNdreaMT in RETURNN).
Can be called directly from config/__init__.py via py().
"""

from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py():
    # 1. Baseline UNdreaMT training (batch_size=2500, max_seqs=50, 5.1GB VRAM) - active run
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en",
        train_name="undreamt_rnn_aed",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=2500,
        batch_max_seqs=50,
        learning_rate=0.0002,
    )

    # 2. High-throughput ablation for 80GB H100 (batch_size=25000, max_seqs=500, ~30-45GB VRAM, ~10x throughput)
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_b25k",
        train_name="undreamt_rnn_aed",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=25000,
        batch_max_seqs=500,
        learning_rate=0.0003,
    )

