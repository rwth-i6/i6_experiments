"""
Experiment config entrypoint for WMT14 German-English Unsupervised NMT (UNdreaMT in RETURNN).
Supports word-level training with isotropy (All-but-the-Top) and cross-lingual alignment
optimizers (Artetxe numerals Procrustes, GAN, GeoWarP).
"""

from typing import Any, Dict, Optional, Sequence, Tuple
from sisyphus import tk
from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)
from ...models.analysis import CompareAlignmentAblationsPcaPanelJob


def run_wmt14_de_en_word_unmt(
    prefix_name: Optional[str] = None,
    train_name: str = "undreamt_rnn_aed",
    num_epochs: int = 100,
    eval_epochs: Sequence[int] = (20, 40, 60, 80, 100),
    batch_size: int = 2500,
    batch_max_seqs: int = 50,
    learning_rate: float = 0.0002,
    w2v_all_but_the_top: bool = False,
    w2v_abtt_d: Optional[int] = 3,
    alignment_method: str = "artetxe",
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Runs the vanilla word-level WMT14 German-English UNMT pipeline with configurable
    All-but-the-Top (ABTT) isotropy postprocessing and alignment optimizer ('artetxe', 'gan', 'geowarp').
    """
    align_clean = alignment_method.strip().lower()
    d_val = int(w2v_abtt_d) if w2v_abtt_d is not None else 3

    if prefix_name is None:
        prefix_name = "exp2026_10_19_unmt/wmt14_de_en_word_d300"
        if w2v_all_but_the_top:
            prefix_name += f"_abtt_d{d_val}"
        if align_clean not in ("artetxe", "self_learning", "numerals", "frequency", "freq"):
            prefix_name += f"_{align_clean}"

    return run_wmt14_de_en_unmt_pipeline(
        prefix_name=prefix_name,
        train_name=train_name,
        num_epochs=num_epochs,
        eval_epochs=eval_epochs,
        batch_size=batch_size,
        batch_max_seqs=batch_max_seqs,
        learning_rate=learning_rate,
        de_code_type="word",
        en_code_type="word",
        w2v_dim=300,
        w2v_window=10,
        w2v_negative=10,
        w2v_sg=1,
        w2v_epochs=5,
        w2v_all_but_the_top=w2v_all_but_the_top,
        w2v_abtt_d=d_val if w2v_all_but_the_top else w2v_abtt_d,
        run_w2v_isotropy_analysis=True,
        alignment_method=align_clean,
        alignment_retrieval_metric="cosine",
        alignment_csls_k=10,
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_baseline(
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Baseline: Raw Word2Vec (d=300) + Artetxe numerals initialization Procrustes alignment.
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_word_d300",
        w2v_all_but_the_top=False,
        alignment_method="artetxe",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_abtt(
    w2v_abtt_d: int = 3,
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ablation 1: All-but-the-Top (ABTT, D=3) isotropy postprocessing + Artetxe numerals Procrustes alignment.
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d3
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name=f"exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d{w2v_abtt_d}",
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="artetxe",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_gan(
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ablation 2: Domain-Adversarial GAN alignment optimizer (Chung et al. 2018).
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300_gan
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_word_d300_gan",
        w2v_all_but_the_top=False,
        alignment_method="gan",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_geowarp(
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ablation 3: GeoWarP Wasserstein Procrustes alignment optimizer (Schnaus et al. 2026).
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300_geowarp
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_word_d300_geowarp",
        w2v_all_but_the_top=False,
        alignment_method="geowarp",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_abtt_gan(
    w2v_abtt_d: int = 3,
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ablation 4: All-but-the-Top (ABTT, D=3) + Domain-Adversarial GAN alignment.
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d3_gan
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name=f"exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d{w2v_abtt_d}_gan",
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="gan",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_abtt_geowarp(
    w2v_abtt_d: int = 3,
    run_undreamt_training: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Ablation 5: All-but-the-Top (ABTT, D=3) + GeoWarP Wasserstein Procrustes alignment.
    Output prefix: exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d3_geowarp
    """
    return run_wmt14_de_en_word_unmt(
        prefix_name=f"exp2026_10_19_unmt/wmt14_de_en_word_d300_abtt_d{w2v_abtt_d}_geowarp",
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="geowarp",
        run_undreamt_training=run_undreamt_training,
        **kwargs,
    )


def run_wmt14_de_en_word_ablations(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_abtt_d: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Executes the full suite of 6 word-level WMT14 DE <-> EN UNMT ablations:
      1. Baseline: Raw Word2Vec + Artetxe Numerals Procrustes
      2. ABTT (D=3): All-but-the-Top + Artetxe Numerals Procrustes
      3. GAN: Raw Word2Vec + GAN Alignment
      4. GeoWarP: Raw Word2Vec + GeoWarP Wasserstein Procrustes
      5. ABTT (D=3) + GAN: All-but-the-Top (D=3) + GAN Alignment
      6. ABTT (D=3) + GeoWarP: All-but-the-Top (D=3) + GeoWarP Alignment
    And generates a unified 6-panel 2D PCA alignment comparison figure.
    """
    res_baseline = run_wmt14_de_en_word_baseline(
        run_undreamt_training=run_undreamt_training,
    )
    res_abtt = run_wmt14_de_en_word_abtt(
        w2v_abtt_d=w2v_abtt_d,
        run_undreamt_training=run_undreamt_training,
    )
    res_gan = run_wmt14_de_en_word_gan(
        run_undreamt_training=run_undreamt_training,
    )
    res_geowarp = run_wmt14_de_en_word_geowarp(
        run_undreamt_training=run_undreamt_training,
    )
    res_abtt_gan = run_wmt14_de_en_word_abtt_gan(
        w2v_abtt_d=w2v_abtt_d,
        run_undreamt_training=run_undreamt_training,
    )
    res_abtt_geowarp = run_wmt14_de_en_word_abtt_geowarp(
        w2v_abtt_d=w2v_abtt_d,
        run_undreamt_training=run_undreamt_training,
    )

    ablation_map = {
        "1. Baseline (Artetxe Numerals)": res_baseline,
        f"2. ABTT (D={w2v_abtt_d}, Artetxe)": res_abtt,
        "3. GAN (Chung et al. 2018)": res_gan,
        "4. GeoWarP (Wasserstein Procrustes)": res_geowarp,
        f"5. ABTT (D={w2v_abtt_d}) + GAN": res_abtt_gan,
        f"6. ABTT (D={w2v_abtt_d}) + GeoWarP": res_abtt_geowarp,
    }

    ablation_entries = {}
    for title_key, r_dict in ablation_map.items():
        al_job = r_dict["align_job"]
        ablation_entries[title_key] = {
            "mapped_src_embeddings_npy": al_job.out_mapped_src_emb_npy,
            "normalized_trg_embeddings_npy": al_job.out_normalized_trg_emb_npy,
            "dictionary_json": al_job.out_dictionary_json,
            "alignment_report_json": al_job.out_alignment_report,
        }

    raw_src_w2v = res_baseline["raw_w2v_de_job"]
    raw_trg_w2v = res_baseline["raw_w2v_en_job"]

    compare_pca_job = CompareAlignmentAblationsPcaPanelJob(
        ablation_entries=ablation_entries,
        src_vocab_file=raw_src_w2v.out_vocab_txt,
        trg_vocab_file=raw_trg_w2v.out_vocab_txt,
        src_counts_file=raw_src_w2v.out_counts_txt,
        trg_counts_file=raw_trg_w2v.out_counts_txt,
        src_label="German Words",
        trg_label="English Words",
        title=f"WMT14 DE <-> EN Words (d=300): All-but-the-Top (D={w2v_abtt_d}) & Optimizer (Artetxe / GAN / GeoWarP) 2D PCA Comparison",
    )
    compare_pca_job.add_alias("exp2026_10_19_unmt/wmt14_de_en_word_d300/alignment/compare_ablations_pca_panel")
    for out_prefix in (
        "exp2026_10_19_unmt/wmt14_de_en_word_d300/alignment/pca_analysis",
        "exp2026_10_19_unmt/wmt14_de_en_word_d300_ablation_comparison",
    ):
        tk.register_output(
            f"{out_prefix}/optimizer_and_isotropy_ablation_pca_panel.png",
            compare_pca_job.out_comparison_panel_plot,
        )
        tk.register_output(
            f"{out_prefix}/ablation_comparison_report.json",
            compare_pca_job.out_comparison_report_json,
        )
        tk.register_output(
            f"{out_prefix}/ablation_comparison_report.txt",
            compare_pca_job.out_comparison_report_txt,
        )

    return {
        "baseline": res_baseline,
        "abtt": res_abtt,
        "gan": res_gan,
        "geowarp": res_geowarp,
        "abtt_gan": res_abtt_gan,
        "abtt_geowarp": res_abtt_geowarp,
        "compare_pca_job": compare_pca_job,
    }


def py():
    """Default runner for WMT14 DE-EN word UNMT ablations."""
    return run_wmt14_de_en_word_ablations()
