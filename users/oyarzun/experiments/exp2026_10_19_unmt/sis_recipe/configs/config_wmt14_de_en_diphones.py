"""
Config for WMT14 German-English UNMT: German Words (50k) <-> English Diphones (2k).
Single Diphone ablation with full UNdreaMT denoising autoencoding and backtranslation training.
"""

from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py():
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_diphones",
        train_name="undreamt_rnn_aed",
        en_code_type="diphone",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=12000,
        batch_max_seqs=200,
        learning_rate=0.0003,
        run_undreamt_training=True,
    )
