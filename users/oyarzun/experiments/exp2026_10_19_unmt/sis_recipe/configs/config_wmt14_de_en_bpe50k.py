"""
Config for WMT14 German-English UNMT: German & English Joint BPE 50k (Shared Vocabulary).
Single BPE ablation with full UNdreaMT denoising autoencoding and backtranslation training.
"""

from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py():
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_joint_bpe50k",
        train_name="undreamt_rnn_aed",
        de_code_type="bpe",
        en_code_type="bpe",
        bpe_operations=50000,
        bpe_mode="joint",
        vocab_size=50000,
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=20000,
        batch_max_seqs=400,
        learning_rate=0.0003,
        run_undreamt_training=True,
    )
