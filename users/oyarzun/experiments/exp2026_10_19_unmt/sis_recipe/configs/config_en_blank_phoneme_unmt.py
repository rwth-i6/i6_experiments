"""
Sisyphus Configuration Entrypoint for Phoneme-Level EN+Blank UNMT Experiments:
  1. EN+Blank <-> EN Phonemes UNMT (`en_blank_en_phoneme_d300`) & Ablations:
     - Source: Uncollapsed 20ms English phoneme + blank frame sequences (`BLK` + 39 ARPAbet, V=40)
       from LibriSpeech 960h GMM VAD-resampled alignments (`phoneme_ref.{000..019}.hdf`).
     - Target: Pure English ARPAbet monophones (V=39) from WMT14 English News Crawl.
     - Eval: LibriSpeech `dev-other` (`EN+Blank <-> EN`).
     - Ablations:
       * Baseline (`en_blank_en_phoneme_d300`): Raw Word2Vec (d=300) + Artetxe Frequency-Init Procrustes + CSLS.
       * All-but-the-Top Isotropy (`en_blank_en_phoneme_d300_abtt_d3`): Mu et al. (2018) All-but-the-Top (D=3) + Artetxe Procrustes + CSLS.
       * GAN Optimizer (`en_blank_en_phoneme_d300_gan`): Chung et al. (2018) Domain-Adversarial GAN + CSLS Procrustes Refinement.
       * GeoWarP Optimizer (`en_blank_en_phoneme_d300_geowarp`): Schnaus et al. (2026) Coarse Geometric Init + Wasserstein Procrustes + CSLS Refinement.
       * ABTT + GAN (`en_blank_en_phoneme_d300_abtt_d3_gan`): All-but-the-Top (D=3) + GAN + CSLS Procrustes Refinement.
       * ABTT + GeoWarP (`en_blank_en_phoneme_d300_abtt_d3_geowarp`): All-but-the-Top (D=3) + GeoWarP + Wasserstein Procrustes + CSLS Refinement.
  2. EN+Blank <-> DE Phonemes UNMT (`en_blank_de_phoneme_d300`):
     - Source: Uncollapsed 20ms English phoneme + blank frame sequences (`BLK` + 39 ARPAbet, V=40)
       from LibriSpeech 960h (`phoneme_ref.{000..019}.hdf`).
     - Target: Pure German-specific IPA monophones (V=46) from WMT14 German News Crawl.
     - Eval: CoVoST-2 English-to-German Speech Translation (`EN+Blank <-> DE`).

All pipelines execute:
  - Stage 1: Cross-modal 300-dim Skip-gram Word2Vec (+ optional ABTT isotropy postprocessing) +
    Cross-modal Alignment (Artetxe / GAN / GeoWarP) + ASR-style 2D PCA & Isotropy Visualization Panels +
    Nearest-Neighbor baseline evaluation.
  - Stage 2: UNdreaMT denoising autoencoding + on-the-fly greedy backtranslation + beam search decoding +
    Shared-Encoder 2D PCA, Contrastive BERTScore PCA, & Cross-Attention Visualizations.
"""

from typing import Any, Dict, Optional
from sisyphus import tk
from ..unmt_en_blank_exp import run_en_blank_phoneme_unmt_pipeline
from ...models.analysis import CompareAlignmentAblationsPcaPanelJob


def run_en_blank_to_en_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_all_but_the_top: bool = False,
    w2v_abtt_d: Optional[int] = None,
    alignment_method: str = "artetxe",
    prefix_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Launches the EN+Blank <-> EN phoneme UNMT setup (cross-modal codes + UNdreaMT denoising/BT)
    with configurable All-but-the-Top isotropy postprocessing and alignment optimizer ('artetxe', 'gan', 'geowarp').
    """
    align_clean = alignment_method.strip().lower()
    d_val = int(w2v_abtt_d) if w2v_abtt_d is not None else 3

    if prefix_name is None:
        prefix_name = "exp2026_10_19_unmt/en_blank_en_phoneme_d300"
        if w2v_all_but_the_top:
            prefix_name += f"_abtt_d{d_val}"
        if align_clean not in ("artetxe", "self_learning", "frequency", "freq"):
            prefix_name += f"_{align_clean}"

    return run_en_blank_phoneme_unmt_pipeline(
        target_lang="en",
        prefix_name=prefix_name,
        train_name="undreamt_rnn_aed_20ep",
        cluster=cluster,
        news_crawl_years=(2013,),
        max_mono_sentences=2000000,
        unmt_train_src_sentences=278400,
        unmt_train_trg_sentences=300000,
        max_en_blank_frames=1000,
        w2v_dim=300,
        src_w2v_window=25,
        trg_w2v_window=10,
        w2v_negative=10,
        w2v_sg=1,
        w2v_epochs=10,
        w2v_sample=1e-3,
        w2v_all_but_the_top=w2v_all_but_the_top,
        w2v_abtt_d=d_val if w2v_all_but_the_top else w2v_abtt_d,
        run_w2v_isotropy_analysis=True,
        alignment_method=align_clean,
        alignment_retrieval_metric="csls",
        alignment_csls_k=5,
        num_epochs=20,
        eval_epochs=[5, 10, 15, 20],
        partition_epoch=10,
        batch_size=6000,
        batch_max_seqs=24,
        learning_rate=0.0002,
        bt_src2trg_max_ratio=1.0,
        bt_trg2src_max_ratio=6.0,
        beam_size=8,
        lp_alpha=0.6,
        device="gpu",
        run_undreamt_training=run_undreamt_training,
        enable_comet=False,
    )


def run_en_blank_to_en_phoneme_unmt_abtt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_abtt_d: int = 3,
) -> Dict[str, Any]:
    """
    Ablation 1: EN+Blank <-> EN phoneme UNMT with All-but-the-Top (ABTT, D=3) isotropy postprocessing
    and Artetxe frequency-init self-learning Procrustes alignment.
    Output prefix: exp2026_10_19_unmt/en_blank_en_phoneme_d300_abtt_d3
    """
    return run_en_blank_to_en_phoneme_unmt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="artetxe",
    )


def run_en_blank_to_en_phoneme_unmt_gan(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_all_but_the_top: bool = False,
    w2v_abtt_d: int = 3,
) -> Dict[str, Any]:
    """
    Ablation 2: EN+Blank <-> EN phoneme UNMT swapping the alignment optimizer to Domain-Adversarial GAN
    (Chung et al., 2018 + CSLS Procrustes refinement).
    Output prefix:
      - exp2026_10_19_unmt/en_blank_en_phoneme_d300_gan (when w2v_all_but_the_top=False)
      - exp2026_10_19_unmt/en_blank_en_phoneme_d300_abtt_d3_gan (when w2v_all_but_the_top=True)
    """
    return run_en_blank_to_en_phoneme_unmt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=w2v_all_but_the_top,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="gan",
    )


def run_en_blank_to_en_phoneme_unmt_geowarp(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_all_but_the_top: bool = False,
    w2v_abtt_d: int = 3,
) -> Dict[str, Any]:
    """
    Ablation 3: EN+Blank <-> EN phoneme UNMT swapping the alignment optimizer to GeoWarP
    (Schnaus et al., 2026 Coarse Geometric Init + Wasserstein Procrustes + CSLS MNN refinement).
    Output prefix:
      - exp2026_10_19_unmt/en_blank_en_phoneme_d300_geowarp (when w2v_all_but_the_top=False)
      - exp2026_10_19_unmt/en_blank_en_phoneme_d300_abtt_d3_geowarp (when w2v_all_but_the_top=True)
    """
    return run_en_blank_to_en_phoneme_unmt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=w2v_all_but_the_top,
        w2v_abtt_d=w2v_abtt_d,
        alignment_method="geowarp",
    )


def run_en_blank_to_en_phoneme_unmt_abtt_gan(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_abtt_d: int = 3,
) -> Dict[str, Any]:
    """
    Ablation 4: EN+Blank <-> EN phoneme UNMT with All-but-the-Top (ABTT, D=3) + GAN alignment optimizer.
    Output prefix: exp2026_10_19_unmt/en_blank_en_phoneme_d300_abtt_d3_gan
    """
    return run_en_blank_to_en_phoneme_unmt_gan(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
    )


def run_en_blank_to_en_phoneme_unmt_abtt_geowarp(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_abtt_d: int = 3,
) -> Dict[str, Any]:
    """
    Ablation 5: EN+Blank <-> EN phoneme UNMT with All-but-the-Top (ABTT, D=3) + GeoWarP alignment optimizer.
    Output prefix: exp2026_10_19_unmt/en_blank_en_phoneme_d300_abtt_d3_geowarp
    """
    return run_en_blank_to_en_phoneme_unmt_geowarp(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=True,
        w2v_abtt_d=w2v_abtt_d,
    )


def run_en_blank_to_en_phoneme_unmt_ablations(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
    w2v_abtt_d: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs the complete EN+Blank <-> EN Phoneme (d=300) UNMT ablation grid:
      1. Baseline (`en_blank_en_phoneme_d300`): Raw Word2Vec + Artetxe Freq-Init Procrustes
      2. ABTT (`en_blank_en_phoneme_d300_abtt_d3`): All-but-the-Top (D=3) + Artetxe Freq-Init Procrustes
      3. GAN (`en_blank_en_phoneme_d300_gan`): Raw Word2Vec + GAN Domain-Adversarial Alignment
      4. GeoWarP (`en_blank_en_phoneme_d300_geowarp`): Raw Word2Vec + GeoWarP Wasserstein Procrustes
      5. ABTT + GAN (`en_blank_en_phoneme_d300_abtt_d3_gan`): All-but-the-Top (D=3) + GAN Alignment
      6. ABTT + GeoWarP (`en_blank_en_phoneme_d300_abtt_d3_geowarp`): All-but-the-Top (D=3) + GeoWarP Alignment
    Also generates a unified 6-panel 2D PCA comparison figure across all ablations.
    """
    res_baseline = run_en_blank_to_en_phoneme_unmt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=False,
        alignment_method="artetxe",
    )
    res_abtt = run_en_blank_to_en_phoneme_unmt_abtt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_abtt_d=w2v_abtt_d,
    )
    res_gan = run_en_blank_to_en_phoneme_unmt_gan(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=False,
    )
    res_geowarp = run_en_blank_to_en_phoneme_unmt_geowarp(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_all_but_the_top=False,
    )
    res_abtt_gan = run_en_blank_to_en_phoneme_unmt_abtt_gan(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_abtt_d=w2v_abtt_d,
    )
    res_abtt_geowarp = run_en_blank_to_en_phoneme_unmt_abtt_geowarp(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
        w2v_abtt_d=w2v_abtt_d,
    )

    ablation_map = {
        "1. Baseline (Artetxe Freq)": res_baseline,
        f"2. ABTT (D={w2v_abtt_d}, Artetxe Freq)": res_abtt,
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

    raw_src_w2v = res_baseline["raw_w2v_src_job"]
    raw_trg_w2v = res_baseline["raw_w2v_trg_job"]

    compare_pca_job = CompareAlignmentAblationsPcaPanelJob(
        ablation_entries=ablation_entries,
        src_vocab_file=raw_src_w2v.out_vocab_txt,
        trg_vocab_file=raw_trg_w2v.out_vocab_txt,
        src_counts_file=raw_src_w2v.out_counts_txt,
        trg_counts_file=raw_trg_w2v.out_counts_txt,
        src_label="EN+Blank Codes",
        trg_label="EN Phonemes",
        title=f"EN+Blank <-> EN Phonemes (d=300): All-but-the-Top (D={w2v_abtt_d}) & Optimizer (Artetxe / GAN / GeoWarP) 2D PCA Comparison",
    )
    compare_pca_job.add_alias("exp2026_10_19_unmt/en_blank_en_phoneme_d300/alignment/compare_ablations_pca_panel")
    for out_prefix in (
        "exp2026_10_19_unmt/en_blank_en_phoneme_d300/alignment/pca_analysis",
        "exp2026_10_19_unmt/en_blank_en_phoneme_d300_ablation_comparison",
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
        "en_blank_en_baseline": res_baseline,
        "en_blank_en_abtt": res_abtt,
        "en_blank_en_gan": res_gan,
        "en_blank_en_geowarp": res_geowarp,
        "en_blank_en_abtt_gan": res_abtt_gan,
        "en_blank_en_abtt_geowarp": res_abtt_geowarp,
        "compare_pca_job": compare_pca_job,
    }


def run_en_blank_to_de_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
) -> Dict[str, Any]:
    """ Launches the EN+Blank <-> DE phoneme UNMT setup (cross-modal codes + UNdreaMT denoising/BT). """
    return run_en_blank_phoneme_unmt_pipeline(
        target_lang="de",
        prefix_name="exp2026_10_19_unmt/en_blank_de_phoneme_d300",
        train_name="undreamt_rnn_aed_20ep",
        cluster=cluster,
        news_crawl_years=(2013,),
        max_mono_sentences=2000000,
        unmt_train_src_sentences=278400,
        unmt_train_trg_sentences=300000,
        max_en_blank_frames=1000,
        covost_dev_max_utterances=500,
        covost_test_max_utterances=2000,
        w2v_dim=300,
        src_w2v_window=25,
        trg_w2v_window=10,
        w2v_negative=10,
        w2v_sg=1,
        w2v_epochs=10,
        w2v_sample=1e-3,
        alignment_retrieval_metric="csls",
        alignment_csls_k=5,
        num_epochs=20,
        eval_epochs=[5, 10, 15, 20],
        partition_epoch=10,
        batch_size=6000,
        batch_max_seqs=24,
        learning_rate=0.0002,
        bt_src2trg_max_ratio=1.0,
        bt_trg2src_max_ratio=6.0,
        beam_size=8,
        lp_alpha=0.6,
        device="gpu",
        run_undreamt_training=run_undreamt_training,
        enable_comet=False,
    )


def run_all_en_blank_phoneme_unmt(
    cluster: Optional[str] = None,
    run_undreamt_training: bool = True,
) -> Dict[str, Any]:
    """
    Entrypoint registering:
      1. EN+Blank <-> EN phoneme UNMT baseline + ABTT isotropy & GAN / GeoWarP optimizer ablations + PCA panels
      2. EN+Blank <-> DE phoneme UNMT
    """
    en_ablations = run_en_blank_to_en_phoneme_unmt_ablations(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
    )
    de_res = run_en_blank_to_de_phoneme_unmt(
        cluster=cluster,
        run_undreamt_training=run_undreamt_training,
    )
    return {
        "en_blank_en": en_ablations["en_blank_en_baseline"],
        "en_blank_en_ablations": en_ablations,
        "en_blank_de": de_res,
    }


py = run_all_en_blank_phoneme_unmt
