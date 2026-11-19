"""
Config for WMT14 German-English UNMT: German Triphones (20k) <-> English Triphones (20k).
Full UNdreaMT denoising autoencoding and backtranslation training on triphone representations for both languages.
Includes:
  - Clean tokenization (lowercasing and punctuation removal)
  - 2D PCA cross-lingual alignment analysis and shared encoder state PCA
  - Baseline vs. All-but-the-Top (ABTT) isotropy postprocessing ablation (Mu et al., 2018)
"""

from typing import Dict, Any, Optional
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def run_triphone_triphone_setup(
    clean_tok: bool = True,
    w2v_all_but_the_top: bool = False,
    w2v_abtt_d: Optional[int] = None,
    run_undreamt_training: bool = True,
    gpu_mem: Optional[int] = 24,
) -> Dict[str, Any]:
    """
    Builds the German-English triphone-triphone UNMT pipeline.

    :param clean_tok: Whether to lowercase and remove punctuation before triphone generation.
    :param w2v_all_but_the_top: Whether to apply All-but-the-Top (ABTT) isotropy post-processing to Word2Vec.
    :param w2v_abtt_d: Number of top principal components to remove (defaults to 3 for 300-d embeddings).
    :param run_undreamt_training: Whether to run UNdreaMT seq2seq training and evaluation.
    :param gpu_mem: Minimum GPU memory in GB.
    """
    prefix = "exp2026_10_19_unmt/wmt14_de_en_triphone_triphone"
    if clean_tok:
        prefix += "_clean_tok"
    if w2v_all_but_the_top:
        d_val = w2v_abtt_d if w2v_abtt_d is not None else 3
        prefix += f"_abtt_d{d_val}"

    return run_wmt14_de_en_unmt_pipeline(
        prefix_name=prefix,
        train_name="undreamt_rnn_aed",
        de_code_type="triphone",
        en_code_type="triphone",
        num_epochs=100,
        eval_epochs=(20, 40, 60, 80, 100),
        batch_size=12000,
        batch_max_seqs=200,
        learning_rate=0.0003,
        run_undreamt_training=run_undreamt_training,
        lowercase=clean_tok,
        remove_punctuation=clean_tok,
        w2v_all_but_the_top=w2v_all_but_the_top,
        w2v_abtt_d=w2v_abtt_d,
        run_w2v_isotropy_analysis=True,
        gpu_mem=gpu_mem,
    )


def py(run_undreamt_training: bool = True):
    return {
        "clean_tok": run_triphone_triphone_setup(
            clean_tok=True,
            w2v_all_but_the_top=False,
            run_undreamt_training=run_undreamt_training,
        ),
        "clean_tok_abtt": run_triphone_triphone_setup(
            clean_tok=True,
            w2v_all_but_the_top=True,
            w2v_abtt_d=3,
            run_undreamt_training=run_undreamt_training,
        ),
    }
