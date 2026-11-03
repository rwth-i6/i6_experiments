"""
Config for WMT14 German-English UNMT: German Triphones (20k) <-> English Triphones (20k).
Full UNdreaMT denoising autoencoding and backtranslation training on triphone representations for both languages.
"""

from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py():
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_triphone_triphone",
        train_name="undreamt_rnn_aed",
        de_code_type="triphone",
        en_code_type="triphone",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=12000,
        batch_max_seqs=200,
        learning_rate=0.0003,
        run_undreamt_training=True,
    )

