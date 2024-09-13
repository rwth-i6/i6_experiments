"""
Alignments
"""

from __future__ import annotations
from typing import Optional, Union, Any, Dict
import os
import sys
from sisyphus import tk, Job, Task, Path


def py():
    prefix = "exp2024_09_09_grad_align/"

    from i6_experiments.users.zeyer.datasets.librispeech import LibrispeechOggZip, Bpe

    bpe1k_num_labels_with_blank = 1057  # incl blank
    bpe1k_blank_idx = 1056  # at the end
    bpe1k_vocab = Path(
        "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"
    )
    returnn_dataset = LibrispeechOggZip(
        vocab=Bpe(
            codes=Path(
                "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes"
            ),
            vocab=bpe1k_vocab,
            dim=1056,
        ),
        train_epoch_split=1,
    ).get_dataset("train")

    # gmm_alignment_hdf = Path(
    #     "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_core/returnn/hdf/ReturnnDumpHDFJob.nQ1YkjerObMO/output/data.hdf"
    # )
    gmm_alignment_allophones = Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    gmm_alignment_sprint_cache = Path(
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
    )
    features_sprint_cache = Path(  # for exact timings
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
    )
    seq_list = Path(
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
    )

    grads = {
        "base-mid-60ms": (  # non-flipped
            6,
            Path(
                "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/forward/ReturnnForwardJobV2.KKMedG4R3uf4/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC auf single gpu mit random seed 1337 (flipped)
        "base-flip-early141-60ms": (
            6,
            Path(
                "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/forward/ReturnnForwardJobV2.RgWrrTtM4Ljf/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC auf single gpu mit random seed 1337 (flipped)
        # epoch 141/2000
        "base-flip-early141-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2_rand-seed-1337/returnn_decoding/epoch-141-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # epoch 646/2000
        "base-flip-mid646-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2_rand-seed-1337/returnn_decoding/epoch-646-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline mit CTC auf single-gpu (epoch 919/2000) (nicht flipped)
        "base-ctc-mid919-60ms": (
            6,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2_ce-aux-4-8/returnn_decoding/epoch-919-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_encoder_input/ground-truth/output/gradients.hdf"
            ),
        ),
        "base-ctc-mid919-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2_ce-aux-4-8/returnn_decoding/epoch-919-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC auf single-gpu (epoch 1676/2000) (nicht flipped)
        "base-far1676-60ms": (
            6,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2/returnn_decoding/epoch-1676-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_encoder_input/ground-truth/output/gradients.hdf"
            ),
        ),
        "base-far1676-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2/returnn_decoding/epoch-1676-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC + ohne filtering auf multi-gpu (epoch 406/500) (flipped)
        "base-flip-far406-60ms": (
            6,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_accum-4/returnn_decoding/epoch-406-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_encoder_input/ground-truth/output/gradients.hdf"
            ),
        ),
        "base-flip-far406-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_accum-4/returnn_decoding/epoch-406-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC + zero padding bei conv module im conformer und im frontend auf multi-gpu
        # (epoch 61/500; att weights gerade konvergiert) (nicht flipped)
        "base-convMask-early61-60ms": (
            6,
            Path(
                "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-conv-w-zero-padding-conv-frontend-w-zero-padding/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_encoder_input/ground-truth/output/gradients.hdf"
            ),
        ),
        "base-convMask-early61-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-conv-w-zero-padding-conv-frontend-w-zero-padding/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        "base-convMask-mid225-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-conv-w-zero-padding-conv-frontend-w-zero-padding/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-225-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC auf single gpu (nicht flipped) (epoch 1743/2000)
        # ohne zero padding
        "base-far1743-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2/returnn_decoding/epoch-1743-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # mit zero padding
        "base-convMaskForward-far1743-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-conv-w-zero-padding-conv-frontend-w-zero-padding/import_1k-baseline-wo-ctc/returnn_decoding/epoch-1743-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
        # 1k baseline ohne CTC auf single gpu (nicht flipped) (wie davor) (epoch 141/2000)
        "base-early141-10ms": (
            1,
            Path(
                "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/2000-ep_bs-35000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-2/returnn_decoding/epoch-141-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
            ),
        ),
    }

    # Specifying the TSE metric for the word bound/pos here in the comments (cutting off all decimals, not rounded).
    for opts in [
        {"grad_name": "base-mid-60ms", "sm": True},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": True, "apply_log": False},  # 108/81.2ms
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": -1.0},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": -2},  # 106/79.3ms
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": -3},  # 88/69.8ms
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": -4},  # 65.5/54.4ms (!)
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": -5},  # 68/57.0ms
        {"grad_name": "base-mid-60ms", "sm": True, "norm_scores": True},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": True, "norm_scores": True, "blank_score": -4},  # 65/54.4ms
        {"grad_name": "base-mid-60ms", "sm": True, "blank_score": "calc"},  # 107/80.2ms
        {"grad_name": "base-mid-60ms", "sm": True, "norm_scores": True, "blank_score": "calc"},  # 107/80.2ms
        {
            "grad_name": "base-mid-60ms",
            "sm": True,
            "norm_scores": True,
            "apply_softmax_over_labels": True,
            "blank_score": "calc",
        },
        {
            "grad_name": "base-mid-60ms",
            "sm": True,
            "norm_scores": True,
            "apply_softmax_over_labels": True,
            "blank_score": -4,
        },
        {"grad_name": "base-mid-60ms", "sm": False},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": False, "blank_score": -1.0},  # 106/78.8ms
        {"grad_name": "base-mid-60ms", "sm": False, "blank_score": -2.0},  # 104/78.2ms
        {"grad_name": "base-mid-60ms", "sm": False, "blank_score": -3.0},  # 108/81.2ms
        {"grad_name": "base-mid-60ms", "sm": False, "apply_log": False},  # 108/81.2ms
        {"grad_name": "base-mid-60ms", "sm": False, "norm_scores": True, "blank_score": -1},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": False, "norm_scores": True, "blank_score": -3},  # 106/79.4ms
        {"grad_name": "base-mid-60ms", "sm": False, "norm_scores": True, "blank_score": "calc"},  # 108/81.8ms
        {"grad_name": "base-mid-60ms", "sm": False, "blank_score": "calc"},  # 108/81.8ms
        {"grad_name": "base-mid-60ms", "sm": False, "apply_softmax_over_labels": True},  # 106/78.4ms
        {
            "grad_name": "base-mid-60ms",
            "sm": False,
            "apply_softmax_over_labels": True,
            "blank_score": "calc",
        },  # 108/81.8ms
        {"grad_name": "base-flip-early141-60ms", "sm": True},  # 111/85.1ms
        {"grad_name": "base-flip-early141-60ms", "sm": True, "blank_score": -4},  # 75.0/60.8
        {"grad_name": "base-flip-early141-60ms", "sm": True, "blank_score": -6},  # 91.0/74.7
        {"grad_name": "base-flip-early141-10ms", "sm": True, "blank_score": -6},  # 72.3/53.4
        {"grad_name": "base-flip-mid646-10ms", "sm": True, "blank_score": -6},  # 91.6/65.9
        {"grad_name": "base-flip-far406-10ms", "sm": True, "blank_score": -4},  # 176.3/132.5
        {"grad_name": "base-flip-far406-10ms", "sm": True, "blank_score": -6},  # 101.3/73.7
        {"grad_name": "base-flip-far406-60ms", "sm": True, "blank_score": -4},  # 102.2/83.2
        {"grad_name": "base-ctc-mid919-10ms", "sm": False, "blank_score": 0},  # 144.3/102.7
        {"grad_name": "base-ctc-mid919-10ms", "sm": False, "blank_score": -1},  # 143.2/101.6
        {"grad_name": "base-ctc-mid919-10ms", "sm": False, "blank_score": -6},  # 1439.0
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": 0},  # 144.3/102.7
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -2},  # 144.3/102.7
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -4},  # 139.7/98.4
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -5},  # 96.4/68.1
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -6},  # 70.6/55.0
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -7},  # 82.8/66.3
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -8},  # 93.2/75.9
        {"grad_name": "base-ctc-mid919-10ms", "sm": True, "blank_score": -10},  # 94.1/76.8
        {"grad_name": "base-ctc-mid919-60ms", "sm": True, "blank_score": -2},  # 110.4/89.8
        {"grad_name": "base-ctc-mid919-60ms", "sm": True, "blank_score": -4},  # 78.2/66.5
        {"grad_name": "base-ctc-mid919-60ms", "sm": True, "blank_score": -6},  # 96.5/82.6
        {"grad_name": "base-far1676-10ms", "sm": True, "blank_score": -4},  # 128.6/77.9
        {"grad_name": "base-far1676-10ms", "sm": True, "blank_score": -6},  # 61.0/45.8 (!!)
        {"grad_name": "base-far1676-10ms", "sm": True, "blank_score": -7},  # 72.4/57.3
        {"grad_name": "base-far1676-60ms", "sm": True, "blank_score": -4},  # 67.0/54.8
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -4},  # 117.5/72.4
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -5.9},  # 56.1/42.6
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6},  # 55.7/42.5 (!!)
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6.1},  # 55.6/42.5
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6.2},  # 55.6/42.6
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6.3},  # 55.7/42.8
        {"grad_name": "base-convMask-early61-60ms", "sm": True, "blank_score": -3},  # 84.7/65.9
        {"grad_name": "base-convMask-early61-60ms", "sm": True, "blank_score": -4},  # 61.0/50.3 (!)
        {"grad_name": "base-convMask-early61-60ms", "sm": True, "blank_score": -5},  # 66.2/54.9
        {"grad_name": "base-convMask-mid225-10ms", "sm": True, "blank_score": -6},  # 65.2/50.5
        {"grad_name": "base-far1743-10ms", "sm": True, "blank_score": -6},  # 61.6/46.3
        {"grad_name": "base-convMaskForward-far1743-10ms", "sm": True, "blank_score": -6},  # 61.6/46.3
        {"grad_name": "base-early141-10ms", "sm": True, "blank_score": -6},  # 59.8/45.8
        # Testing more on calculated blank.
        # baseline: {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6},  # 55.7/42.5 (!!)
        {
            "grad_name": "base-convMask-early61-10ms",
            "sm": True,
            "apply_softmax_over_time_est_blank": False,
            "blank_score": "calc",
        },  # 144.6/96.0
        {
            "grad_name": "base-convMask-early61-10ms",
            "sm": True,
            "apply_softmax_over_time_est_blank": False,
            "blank_score": "calc",
            "apply_softmax_over_labels": True,
        },  # 144.6/96.0
        {
            "grad_name": "base-convMask-early61-10ms",
            "sm": True,
            "blank_score": "calc",
            "apply_softmax_over_labels": True,
        },
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -6, "apply_softmax_over_labels": True},
        {"grad_name": "base-convMask-early61-10ms", "sm": True, "blank_score": -3, "apply_softmax_over_labels": True},
    ]:
        opts = opts.copy()
        apply_softmax_over_time = opts.pop("sm", False)
        grad_name = opts.pop("grad_name")
        factor, grad_hdf = grads[grad_name]

        # The dumped grads cover about 9.6h audio from train.
        name = f"grad-align-{grad_name}-sm{apply_softmax_over_time}"
        if opts:
            for k, v in opts.items():
                name += f"-{k}{v}"
        job = ForcedAlignOnScoreMatrixJob(
            score_matrix_hdf=grad_hdf,
            apply_softmax_over_time=apply_softmax_over_time,
            num_labels=bpe1k_num_labels_with_blank,
            blank_idx=bpe1k_blank_idx,
            returnn_dataset=returnn_dataset,
            **opts,
        )
        job.add_alias(prefix + name + "/align")
        tk.register_output(prefix + name + "/align.hdf", job.out_align)
        alignment_hdf = job.out_align

        name += "/metrics"
        job = CalcAlignmentMetrics(
            seq_list=seq_list,
            alignment_hdf=alignment_hdf,
            alignment_bpe_vocab=bpe1k_vocab,
            alignment_blank_idx=bpe1k_blank_idx,
            features_sprint_cache=features_sprint_cache,
            ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
            ref_alignment_allophones=gmm_alignment_allophones,
            ref_alignment_len_factor=factor,
        )
        job.add_alias(prefix + name)
        tk.register_output(prefix + name + ".json", job.out_scores)

    name = "ctc-1k-align/metrics"  # 83.0/60.6ms
    job = CalcAlignmentMetrics(
        seq_list=seq_list,
        alignment_hdf=Path(
            "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/ctc/baseline_v1/baseline_rf/bpe1056/8-layer_standard-conformer/import_glob.conformer.luca.bpe1k.w-ctc/returnn_realignment/best-checkpoint/realignment_train/output/realignment.hdf"
        ),
        alignment_label_topology="ctc",
        alignment_bpe_vocab=bpe1k_vocab,
        alignment_blank_idx=bpe1k_blank_idx,
        features_sprint_cache=features_sprint_cache,
        ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
        ref_alignment_allophones=gmm_alignment_allophones,
        ref_alignment_len_factor=6,
    )
    job.add_alias(prefix + name)
    tk.register_output(prefix + name + ".json", job.out_scores)

    name = "ctc-10k-align/metrics"  # 312.2/306.1ms
    job = CalcAlignmentMetrics(
        seq_list=seq_list,
        alignment_hdf=Path(
            "/u/schmitt/experiments/segmental_models_2022_23_rf/alias/models/ls_conformer/global_att/baseline_v1/baseline/no-finetuning/ctc_alignments/train/output/alignments.hdf"
        ),
        alignment_label_topology="ctc",
        alignment_bpe_vocab=Path(
            "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
        ),
        alignment_blank_idx=10_025,
        features_sprint_cache=features_sprint_cache,
        ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
        ref_alignment_allophones=gmm_alignment_allophones,
        ref_alignment_len_factor=6,
    )
    job.add_alias(prefix + name)
    tk.register_output(prefix + name + ".json", job.out_scores)


def visualize_grad_scores():
    # to run:
    # Fish: set -x PYTHONPATH tools/espnet:tools/returnn:tools/sisyphus:recipe
    # Bash: export PYTHONPATH="tools/espnet:tools/returnn:tools/sisyphus:recipe"
    # Then: `python3 -c "from i6_experiments.users.zeyer.experiments.exp2024_09_09_grad_align import visualize_grad_scores as vis; vis()"`  # noqa
    # play around here...

    seq_list = Path(
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
    )
    seq_list = open(seq_list.get_path()).read().splitlines()

    # base-convMask-early61-10ms:
    score_matrix_hdf = Path(
        "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-conv-w-zero-padding-conv-frontend-w-zero-padding/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/dump_gradients_wrt_frontend_input/ground-truth/output/gradients.hdf"
    )

    plot_dir = "output/exp2024_09_09_grad_align/visualize_grad_scores"
    os.makedirs(plot_dir, exist_ok=True)

    from i6_experiments.users.schmitt.hdf import load_hdf_data
    import i6_core.util as util

    returnn_root = util.get_returnn_root(None)

    sys.path.insert(0, returnn_root.get_path())

    import numpy as np

    def _log_softmax(x: np.ndarray, *, axis: Optional[int] = None) -> np.ndarray:
        max_score = np.max(x, axis=axis, keepdims=True)
        x = x - max_score
        return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))

    def _y_to_mat(y, y_num_pixels=100):
        x_num_pixels = len(y)
        y_min, y_max = np.min(y), np.max(y)
        mat = np.full((x_num_pixels, y_num_pixels), y_min)
        for x_, y_ in enumerate(y):
            y__ = int((y_ - y_min) / max(y_max - y_min, 1) * (y_num_pixels - 1))
            mat[x_, y__] = y_
        return mat.T

    def _rescale(y, clip_min, clip_max):
        y = np.clip(y, clip_min, clip_max)
        return (y - clip_min) / max(clip_max - clip_min, 1)

    score_matrix_data_dict = load_hdf_data(score_matrix_hdf, num_dims=2)
    for i, seq_tag in enumerate(seq_list):
        if i >= 2:
            break

        score_matrix = score_matrix_data_dict[seq_tag]  # [S, T]
        S, T = score_matrix.shape  # noqa
        score_matrix = score_matrix[:-1]  # cut off EOS

        log_sm_over_time = _log_softmax(np.log(score_matrix), axis=1)  # [S, T]

        non_blank_score = np.max(np.exp(log_sm_over_time), axis=0)  # [T]
        blank_score = 1.0 - non_blank_score
        blank_score = np.log(blank_score)

        # log_scores = np.log(score_matrix)
        log_scores = log_sm_over_time
        # mean or max, both seem ok. opt percentile changes.
        # log_non_blank_score = np.max(log_scores, axis=0)  # [T]
        log_non_blank_score = np.mean(log_scores, axis=0)  # [T]
        flip_point = np.percentile(log_non_blank_score, 30)  # for max, 10 enough. for mean: 30 or so.
        blank_score___ = 2 * flip_point - log_non_blank_score  # [T]
        # blank_score___ = np.full_like(blank_score___, -1e10)

        # Concat blank score to the end, to include it in the softmax.
        score_matrix_ = np.concatenate([log_scores, blank_score___[None, :]], axis=0)  # [S+1, T]
        score_matrix_ = _log_softmax(score_matrix_, axis=0)
        # score_matrix_ += np.min(score_matrix_) + 1e-10
        # score_matrix_ /= np.sum(score_matrix_, axis=0, keepdims=True)
        # score_matrix_ = np.log(score_matrix_)
        log_sm_over_labels_incl_blank, blank_score_ = score_matrix_[:-1], score_matrix_[-1]

        non_blank_score = np.max(score_matrix, axis=0)  # [T]
        blank_score__ = np.max(score_matrix) - non_blank_score  # [T]

        from matplotlib import pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        rows = [
            # ("score matrix", score_matrix),
            ("log score matrix", np.log(score_matrix)),
            # ("log softmax over all", _log_softmax(np.log(score_matrix))),
            ("log softmax over time", log_sm_over_time),
            ("log blank score", blank_score),
            ("log blank score___", blank_score___),
            # ("log softmax over labels", _log_softmax(np.log(score_matrix), axis=0)),  # bad
            ("log softmax over time first, then labels", _log_softmax(log_sm_over_time, axis=0)),
            ("log softmax over labels incl blank", log_sm_over_labels_incl_blank),
            ("log blank score_", blank_score_),
            ("log blank score__", blank_score__),
            ("label[0] log sm scores", log_sm_over_time[0]),
            # ("log non blank scores", _y_to_mat(np.log(non_blank_score))),
        ]
        fig, ax = plt.subplots(nrows=len(rows), ncols=1, figsize=(20, 5 * len(rows)))
        for i, (alias, mat) in enumerate(rows):
            # mat is [S|Y,T] or just [T]
            if mat.ndim == 1:
                assert mat.shape == (T,)
                mat = _y_to_mat(mat)  # [Y,T]
            else:
                assert mat.ndim == 2 and mat.shape[1] == T
            mat_ = ax[i].matshow(mat, cmap="Blues", aspect="auto")
            ax[i].set_title(f"{alias} for seq {seq_tag}")
            ax[i].set_xlabel("time")
            ax[i].set_ylabel("labels")
            ax[i].set_ylim(ax[i].get_ylim()[::-1])

            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(mat_, cax=cax, orientation="vertical")

        plt.tight_layout()
        fn = f"{plot_dir}/alignment_{seq_tag.replace('/', '_')}.pdf"
        print("save to:", fn)
        plt.savefig(fn)


class ForcedAlignOnScoreMatrixJob(Job):
    """Calculate the Viterbi alignment for a given score matrix."""

    __sis_hash_exclude__ = {
        "blank_score": 0.0,
        "substract": "max_gt0",
        "norm_scores": False,
        "apply_softmax_over_time_est_blank": True,
        "apply_softmax_over_labels": False,
    }

    def __init__(
        self,
        *,
        score_matrix_hdf: Path,
        cut_off_eos: bool = True,
        norm_scores: bool = False,
        apply_log: bool = True,
        substract: Optional[Union[str, float]] = "max_gt0",
        apply_softmax_over_time: bool = False,
        apply_softmax_over_time_est_blank: bool = True,
        apply_softmax_over_labels: bool = False,
        blank_score: Union[float, str] = 0.0,  # or "calc"
        num_seqs: int = -1,
        num_labels: Optional[int] = None,
        blank_idx: int,
        returnn_dataset: Dict[str, Any],  # for BPE labels
        returnn_dataset_key: str = "classes",
        returnn_root: Optional[tk.Path] = None,
    ):
        self.score_matrix_hdf = score_matrix_hdf
        self.cut_off_eos = cut_off_eos
        self.norm_scores = norm_scores
        self.apply_log = apply_log
        self.substract = substract
        self.apply_softmax_over_time = apply_softmax_over_time
        self.apply_softmax_over_time_est_blank = apply_softmax_over_time_est_blank
        self.apply_softmax_over_labels = apply_softmax_over_labels
        self.blank_score = blank_score
        self.num_seqs = num_seqs
        self.num_labels = num_labels
        self.blank_idx = blank_idx
        self.returnn_dataset = returnn_dataset
        self.returnn_dataset_key = returnn_dataset_key
        self.returnn_root = returnn_root

        self.out_align = self.output_path("out_align")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        from typing import List, Tuple
        import numpy as np
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        from i6_experiments.users.schmitt.hdf import load_hdf_data
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)

        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import SimpleHDFWriter

        score_matrix_data_dict = load_hdf_data(self.score_matrix_hdf, num_dims=2)
        hdf_writer = SimpleHDFWriter(
            self.out_align.get_path(), dim=self.num_labels, ndim=1, extra_type={"states": (1, 1, "int32")}
        )
        seq_list = list(score_matrix_data_dict.keys())

        from returnn.config import set_global_config, Config
        from returnn.datasets import init_dataset
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 4
        log.init_by_config(config)

        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, Path) else x, dataset_dict)
        print("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)

        # We might want "train-other-960/1034-121119-0049/1034-121119-0049",
        # but it's actually "train-clean-100/1034-121119-0049/1034-121119-0049" in the RETURNN dataset.
        # Transform the seq tags for the RETURNN dataset.
        all_tags = set(dataset.get_all_tags())
        all_tags_wo_prefix = {}
        for tag in all_tags:
            tag_wo_prefix = tag.split("/", 2)[-1]
            assert tag_wo_prefix not in all_tags_wo_prefix
            all_tags_wo_prefix[tag_wo_prefix] = tag
        seq_list_ = []
        for seq_tag in seq_list:
            tag_wo_prefix = seq_tag.split("/", 2)[-1]
            if seq_tag in all_tags:
                seq_list_.append(seq_tag)
            elif tag_wo_prefix in all_tags_wo_prefix:
                seq_list_.append(all_tags_wo_prefix[tag_wo_prefix])
            else:
                print(f"seq tag {seq_tag} not found in dataset")

        dataset.init_seq_order(epoch=1, seq_list=seq_list_)

        def _log_softmax(x: np.ndarray, *, axis: Optional[int]) -> np.ndarray:
            max_score = np.max(x, axis=axis, keepdims=True)
            x = x - max_score
            return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))

        for i, seq_tag in enumerate(seq_list):
            if 0 < self.num_seqs <= i:
                break

            print("seq tag:", seq_tag)

            dataset.load_seqs(i, i + 1)
            assert dataset.get_tag(i) == seq_list_[i]
            labels = dataset.get_data(i, self.returnn_dataset_key)
            print("labels:", labels, f"(len {len(labels)})")

            score_matrix = score_matrix_data_dict[seq_tag]  # [S, T]
            print("score matrix shape (S x T):", score_matrix.shape)
            if self.cut_off_eos:
                # Last row is EOS, remove it.
                score_matrix = score_matrix[:-1]
            assert len(score_matrix) == len(labels)
            T = score_matrix.shape[1]  # noqa
            S = score_matrix.shape[0]  # noqa

            if self.norm_scores:  # norm such that sum over whole matrix is 1
                score_matrix = score_matrix / np.sum(score_matrix)

            non_blank_score = np.max(score_matrix, axis=0)  # [T]
            blank_score = np.max(score_matrix) - non_blank_score

            # Note: We are going to search the alignment path with the highest score.
            if self.apply_log:
                # Assuming L2 norm scores (i.e. >0).
                score_matrix = np.log(score_matrix)
                blank_score = np.log(blank_score)
            # Otherwise assume already in log space.
            # Make sure they are all negative or zero max.
            m = np.max(score_matrix)
            print("score matrix max:", m)
            if self.substract == "max_gt0":
                score_matrix = score_matrix - max(m, 0.0)
                blank_score = blank_score - max(m, 0.0)
            elif isinstance(self.substract, float):
                score_matrix = score_matrix - self.substract
                blank_score = blank_score - self.substract
            elif not self.substract:
                pass
            else:
                raise ValueError(f"invalid substract {self.substract!r}")
            if self.apply_softmax_over_time:
                score_matrix = _log_softmax(score_matrix, axis=1)
                non_blank_score = np.max(np.exp(score_matrix), axis=0)  # [T]
                if self.apply_softmax_over_time_est_blank:
                    blank_score = 1.0 - non_blank_score
                    blank_score = np.log(blank_score)
            if self.apply_softmax_over_labels:
                # Concat blank score to the end, to include it in the softmax.
                score_matrix = np.concatenate([score_matrix, blank_score[None, :]], axis=0)  # [S+1, T]
                score_matrix = _log_softmax(score_matrix, axis=0)
                score_matrix, blank_score = score_matrix[:-1], score_matrix[-1]

            # scores/backpointers over the states and time steps.
            # states = blank/sil + labels. whether we give scores to blank (and what score) or not is to be configured.
            # [T, S*2+1]
            backpointers = np.full(
                (T, S * 2 + 1), 3, dtype=np.int32
            )  # 0: diagonal-skip, 1: diagonal, 2: left, 3: undefined
            align_scores = np.full((T, S * 2 + 1), -np.infty, dtype=np.float32)

            score_matrix_ = np.zeros((T, S * 2 + 1), dtype=np.float32)  # [T, S*2+1]
            score_matrix_[:, 1::2] = score_matrix.T
            if isinstance(self.blank_score, (int, float)):
                score_matrix_[:, 0::2] = self.blank_score  # blank score
            elif self.blank_score == "calc":
                score_matrix_[:, 0::2] = blank_score[:, None]
            else:
                raise ValueError(f"invalid blank_score {self.blank_score!r} setting")

            # The first two states are valid start states.
            align_scores[0, :2] = score_matrix_[0, :2]
            backpointers[0, :] = 0  # doesn't really matter

            # calculate align_scores and backpointers
            for t in range(1, T):
                scores_diagonal_skip = np.full([2 * S + 1], -np.infty)
                scores_diagonal_skip[2:] = align_scores[t - 1, :-2] + score_matrix_[t, 2:]  # [2*S-1]
                scores_diagonal_skip[::2] = -np.infty  # diagonal skip is not allowed in blank
                scores_diagonal = np.full([2 * S + 1], -np.infty)
                scores_diagonal[1:] = align_scores[t - 1, :-1] + score_matrix_[t, 1:]  # [2*S]
                scores_horizontal = align_scores[t - 1, :] + score_matrix_[t, :]  # [2*S+1]

                score_cases = np.stack([scores_diagonal_skip, scores_diagonal, scores_horizontal], axis=0)  # [3, 2*S+1]
                backpointers[t] = np.argmax(score_cases, axis=0)  # [2*S+1]->[0,1,2]
                align_scores[t : t + 1] = np.take_along_axis(score_cases, backpointers[t : t + 1], axis=0)  # [1,2*S+1]

            # All but the last two states are not valid final states.
            align_scores[-1, :-2] = -np.infty

            # backtrace
            best_final = np.argmax(align_scores[-1])  # scalar, S*2 or S*2-1
            s = best_final
            t = T - 1
            alignment: List[Tuple[int, int]] = []
            while True:
                assert 0 <= s < S * 2 + 1 and 0 <= t < T
                alignment.append((t, s))
                if t == 0 and s <= 1:  # we reached some start state
                    break

                b = backpointers[t, s]
                if b == 0:
                    s -= 2
                    t -= 1
                elif b == 1:
                    s -= 1
                    t -= 1
                elif b == 2:
                    t -= 1
                else:
                    raise ValueError(f"invalid backpointer {b} at s={s}, t={t}")

            assert len(alignment) == T
            alignment.reverse()
            alignment_ = []
            for t, s in alignment:
                if s % 2 == 0:
                    alignment_.append(self.blank_idx)
                else:
                    alignment_.append(labels[s // 2])
            alignment_ = np.array(alignment_, dtype=np.int32)  # [T]
            assert len(alignment_) == T

            hdf_writer.insert_batch(
                alignment_[None, :], seq_len=[T], seq_tag=[seq_tag], extra={"states": np.array(alignment)[None, :, 1]}
            )

            if i < 10:  # plot the first 10 for debugging
                plot_dir = Path("alignment-plots", self).get_path()
                os.makedirs(plot_dir, exist_ok=True)

                from matplotlib import pyplot as plt
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                alignment_map = np.zeros([T, S], dtype=np.int32)  # [T, S]
                for t, s in alignment:
                    if s % 2 == 1:
                        alignment_map[t, s // 2] = 1

                fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 10))
                for i, (alias, mat) in enumerate(
                    [
                        ("log(gradients) (local scores d)", score_matrix.T),
                        ("Partial scores D", -1 * align_scores),
                        ("backpointers", -1 * backpointers),
                        ("alignment", alignment_map),
                    ]
                ):
                    # mat is [T,S*2+1] or [T,S]
                    mat_ = ax[i].matshow(mat.T, cmap="Blues", aspect="auto")
                    ax[i].set_title(f"{alias} for seq {seq_tag}")
                    ax[i].set_xlabel("time")
                    ax[i].set_ylabel("labels")

                    divider = make_axes_locatable(ax[i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    if alias == "backpointers":
                        cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, -1, -2, -3])
                        cbar.ax.set_yticklabels(["diagonal-skip", "diagonal", "left", "unreachable"])
                    elif alias == "alignment":
                        cbar = fig.colorbar(mat_, cax=cax, orientation="vertical", ticks=[0, 1])
                        cbar.ax.set_yticklabels(["", "label"])
                    else:
                        fig.colorbar(mat_, cax=cax, orientation="vertical")

                plt.tight_layout()
                plt.savefig(f"{plot_dir}/alignment_{seq_tag.replace('/', '_')}.png")

        hdf_writer.close()


class CalcAlignmentMetrics(Job):
    """Calculate alignment metrics, e.g. time-stamp-error (TSE) for word boundaries and for word positions."""

    def __init__(
        self,
        *,
        seq_list: Optional[tk.Path] = None,
        alignment_hdf: Path,
        alignment_label_topology: str = "explicit",
        alignment_bpe_vocab: Path,
        alignment_blank_idx: int,
        ref_alignment_sprint_cache: Path,
        ref_alignment_allophones: Path,
        ref_alignment_len_factor: int,
        features_sprint_cache: Optional[Path] = None,  # for exact timings
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()

        self.seq_list = seq_list
        self.alignment_hdf = alignment_hdf
        self.alignment_label_topology = alignment_label_topology
        self.alignment_bpe_vocab = alignment_bpe_vocab
        self.alignment_blank_idx = alignment_blank_idx
        self.ref_alignment_sprint_cache = ref_alignment_sprint_cache
        self.ref_alignment_allophones = ref_alignment_allophones
        self.ref_alignment_len_factor = ref_alignment_len_factor
        self.features_sprint_cache = features_sprint_cache
        self.returnn_root = returnn_root

        self.out_scores = self.output_path("out_scores.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        from typing import List, Tuple
        import numpy as np
        import subprocess
        from itertools import zip_longest
        import i6_experiments

        def _cf(path: Path) -> str:
            return path.get_path()

            try:
                return subprocess.check_output(["cf", path.get_path()]).decode(sys.stdout.encoding).strip()
            except subprocess.CalledProcessError:
                return path.get_path()

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)

        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        from returnn.sprint.cache import open_file_archive
        from returnn.datasets.util.vocabulary import Vocabulary

        print("Loading alignment HDF...")
        alignments_ds = HDFDataset([self.alignment_hdf.get_path()])
        alignments_ds.initialize()
        seq_list = None
        if self.seq_list is not None:
            seq_list = open(self.seq_list.get_path()).read().splitlines()
        alignments_ds.init_seq_order(epoch=1, seq_list=seq_list)

        print("Loading BPE vocab...")
        bpe_vocab = Vocabulary(self.alignment_bpe_vocab.get_path(), unknown_label=None)
        bpe_labels_with_blank = bpe_vocab.labels + ["[BLANK]"]

        # noinspection PyShadowingNames
        def _is_word_end(t: int) -> bool:
            label_idx = alignment[t]
            state_idx = align_states[t]
            if label_idx == self.alignment_blank_idx:
                return False
            if bpe_vocab.labels[label_idx].endswith("@@"):
                return False
            if t == len(alignment) - 1:
                return True
            return state_idx != align_states[t + 1]

        print("Loading ref alignment Sprint cache...")
        ref_align_sprint_cache = open_file_archive(_cf(self.ref_alignment_sprint_cache))
        print("Loading ref alignment allophones...")
        ref_align_sprint_cache.set_allophones(_cf(self.ref_alignment_allophones))
        allophones = ref_align_sprint_cache.get_allophones_list()

        print("Loading features Sprint cache...")
        features_sprint_cache = open_file_archive(_cf(self.features_sprint_cache))

        def _ceil_div(a: int, b: int) -> int:
            return -(-a // b)

        # noinspection PyShadowingNames
        def _start_end_time_for_align_frame_idx(t: int) -> Tuple[float, float]:
            """in seconds"""
            # For the downsampling, assume same padding, thus pad:
            stride = win_size = self.ref_alignment_len_factor
            pad_total = win_size - 1
            pad_left = pad_total // 2
            t0 = t * stride - pad_left  # inclusive
            t1 = t0 + win_size - 1  # inclusive
            # Now about the log mel features.
            window_len = 0.025  # 25 ms
            step_len = 0.010  # 10 ms
            sampling_rate = 16_000
            window_num_frames = int(window_len * sampling_rate)
            step_num_frames = int(step_len * sampling_rate)
            t0 *= step_num_frames
            t1 *= step_num_frames
            t1 += window_num_frames  # exclusive
            return max(0.0, t0 / sampling_rate), t1 / sampling_rate

        out_scores = {
            # "per_seq": {"tse_word_boundaries": {}, "tse_word_positions": {}},
            # "total": {"tse_word_boundaries": 0.0, "tse_word_positions": 0.0},
            "avg": {"tse_word_boundaries": -1.0, "tse_word_positions": -1.0},
            "total_num_words": 0,
            "total_num_seqs": 0,
            "total_duration": 0.0,
        }

        total_tse_word_boundaries = 0.0
        total_tse_word_positions = 0.0
        seq_idx = 0
        while alignments_ds.is_less_than_num_seqs(seq_idx):
            alignments_ds.load_seqs(seq_idx, seq_idx + 1)
            key = alignments_ds.get_tag(seq_idx)
            alignment = alignments_ds.get_data(seq_idx, "data")
            if self.alignment_label_topology == "explicit":
                align_states = alignments_ds.get_data(seq_idx, "states")
            elif self.alignment_label_topology == "ctc":
                align_states = []
                s = 0
                prev_label_idx = self.alignment_blank_idx
                for label_idx in alignment:
                    if label_idx == prev_label_idx:
                        align_states.append(s)
                    elif label_idx == self.alignment_blank_idx:  # and label_idx != prev_label_idx
                        # Was in label, went into blank.
                        s += 1
                        assert s % 2 == 0
                        align_states.append(s)
                    else:  # label_idx != blank_idx and label_idx != prev_label_idx
                        # Went into new label.
                        if prev_label_idx == self.alignment_blank_idx:
                            assert s % 2 == 0
                            s += 1
                        else:  # was in other label before
                            assert s % 2 == 1
                            s += 2  # skip over blank state
                        align_states.append(s)
                    prev_label_idx = label_idx
                align_states = np.array(align_states)  # [T]
            else:
                raise ValueError(f"alignment_label_topology {self.alignment_label_topology!r} not supported")
            assert len(alignment) == len(align_states)

            print("seq tag:", key)
            feat_times, _ = features_sprint_cache.read(key, typ="feat")
            ref_align = ref_align_sprint_cache.read(key, typ="align")
            assert len(feat_times) == len(ref_align), f"feat len {len(feat_times)} vs ref align len {len(ref_align)}"
            print(f"  start time: {feat_times[0][0]} sec")
            print(f"  end time: {feat_times[-1][1]} sec")
            if seq_idx == 0:
                for t in [
                    0,
                    1,
                    2,
                    3,
                    4,
                    len(feat_times) - 4,
                    len(feat_times) - 3,
                    len(feat_times) - 2,
                    len(feat_times) - 1,
                ]:
                    start_time, end_time = feat_times[t]
                    print(f"  ref align frame {t}: start {start_time} sec, end {end_time} sec")
            ref_start_time = ref_align[0][0]
            duration_sec = feat_times[-1][1] - ref_start_time
            sampling_rate = 16_000
            len_samples = round(duration_sec * sampling_rate)  # 16 kHz
            print(f"  num samples: {len_samples} (rounded from {duration_sec * sampling_rate})")
            # RETURNN uses log mel filterbank features, 10ms frame shift, via stft (valid padding)
            window_len = 0.025  # 25 ms
            step_len = 0.010  # 10 ms
            window_num_frames = int(window_len * sampling_rate)
            step_num_frames = int(step_len * sampling_rate)
            len_feat = _ceil_div(len_samples - (window_num_frames - 1), step_num_frames)
            print(f"  num features (10ms): {len_feat} (window {window_num_frames} step {step_num_frames})")
            len_feat_downsampled = _ceil_div(len_feat, self.ref_alignment_len_factor)
            print(f"  downsampled num features: {len_feat_downsampled} (factor {self.ref_alignment_len_factor})")
            print(f"  actual align len: {len(alignment)}")

            last_frame_start, align_dur = _start_end_time_for_align_frame_idx(len(alignment) - 1)
            print(f"  last frame start: {last_frame_start} sec")
            print(f"  align duration: {align_dur} sec")
            out_scores["total_duration"] += duration_sec

            # I'm not really sure on the calculation above, and also not really sure about the limit here...
            # assert (
            #     abs(align_dur - duration_sec) < 0.0301
            # ), f"align duration {align_dur} vs duration {duration_sec}, diff {abs(align_dur - duration_sec)}"
            assert last_frame_start <= duration_sec <= align_dur + 0.0301

            cur_word_start_frame = None
            word_boundaries = []
            words_bpe = []
            prev_state_idx = 0
            for t, (label_idx, state_idx) in enumerate(zip(alignment, align_states)):
                if label_idx == self.alignment_blank_idx:
                    continue
                if cur_word_start_frame is None:
                    cur_word_start_frame = t  # new word starts here
                    words_bpe.append([])
                if state_idx != prev_state_idx:
                    words_bpe[-1].append(bpe_vocab.labels[label_idx])
                if _is_word_end(t):
                    assert cur_word_start_frame is not None
                    word_frame_start, _ = _start_end_time_for_align_frame_idx(cur_word_start_frame)
                    _, word_frame_end = _start_end_time_for_align_frame_idx(t)
                    word_boundaries.append((word_frame_start, word_frame_end))
                    cur_word_start_frame = None
                prev_state_idx = state_idx
            assert cur_word_start_frame is None  # word should have ended
            num_words = len(word_boundaries)
            assert num_words == len(words_bpe)
            print("  num words:", num_words)

            ref_word_boundaries = []
            cur_word_start_frame = None
            prev_allophone_idx = None
            ref_words_phones = []
            for t, (t_, allophone_idx, hmm_state_idx, _) in enumerate(ref_align):
                assert t == t_
                if "[SILENCE]" in allophones[allophone_idx]:
                    continue
                if cur_word_start_frame is None:
                    cur_word_start_frame = t  # new word starts here
                    ref_words_phones.append([])
                if prev_allophone_idx != allophone_idx:
                    ref_words_phones[-1].append(allophones[allophone_idx])
                if "@f" in allophones[allophone_idx] and (
                    t == len(ref_align) - 1
                    or ref_align[t + 1][1] != allophone_idx
                    or ref_align[t + 1][2] < hmm_state_idx
                ):
                    # end of word
                    start_time = feat_times[cur_word_start_frame][0] - ref_start_time
                    end_time = feat_times[t][1] - ref_start_time
                    # take center 10ms of the 25ms window
                    # (or not, as we also don't do for the alignment)
                    # start_time += (window_len - step_len) / 2
                    # end_time -= (window_len - step_len) / 2
                    ref_word_boundaries.append((start_time, end_time))
                    cur_word_start_frame = None
                prev_allophone_idx = allophone_idx
            assert cur_word_start_frame is None  # word should have ended
            assert len(ref_words_phones) == len(ref_word_boundaries)
            assert num_words == len(word_boundaries) == len(ref_word_boundaries), (
                f"seq idx {seq_idx}, tag {key},"
                f" num word mismatch: {len(word_boundaries)} vs {len(ref_word_boundaries)},"
                f" words (BPE vs ref phones):\n"
                + "\n".join(f"{w1} vs {w2}" for w1, w2 in zip_longest(words_bpe, ref_words_phones))
                + f"\nwords BPE alignment: {' '.join(bpe_labels_with_blank[i] for i in alignment)}"
            )

            seq_tse_word_boundaries = 0.0
            seq_tse_word_positions = 0.0
            for i, (word_boundary, ref_word_boundary) in enumerate(zip(word_boundaries, ref_word_boundaries)):
                tse_bounds = (
                    abs(word_boundary[0] - ref_word_boundary[0]) + abs(word_boundary[1] - ref_word_boundary[1])
                ) / 2
                # center pos of words
                tse_pos = abs(
                    (word_boundary[0] + word_boundary[1]) / 2 - (ref_word_boundary[0] + ref_word_boundary[1]) / 2
                )
                seq_tse_word_boundaries += tse_bounds
                seq_tse_word_positions += tse_pos
                # out_scores["per_seq"]["tse_word_boundaries"].setdefault(key, []).append(tse_bounds)
                # out_scores["per_seq"]["tse_word_positions"].setdefault(key, []).append(tse_pos)
                # out_scores["total"]["tse_word_boundaries"] += tse_bounds
                # out_scores["total"]["tse_word_positions"] += tse_pos
                total_tse_word_boundaries += tse_bounds
                total_tse_word_positions += tse_pos

            print("  TSE word boundaries:", seq_tse_word_boundaries / num_words)
            print("  TSE word positions:", seq_tse_word_positions / num_words)
            out_scores["total_num_words"] += num_words
            out_scores["total_num_seqs"] += 1
            seq_idx += 1

        assert out_scores["total_num_words"] > 0, out_scores
        out_scores["avg"]["tse_word_boundaries"] = total_tse_word_boundaries / out_scores["total_num_words"]
        out_scores["avg"]["tse_word_positions"] = total_tse_word_positions / out_scores["total_num_words"]
        print(out_scores)

        import json

        json.dump(out_scores, open(self.out_scores.get_path(), "w"))

        with open(Path("short-report-string.txt", self).get_path(), "w") as f:
            print(
                "%s/%s"
                % (
                    round(out_scores["avg"]["tse_word_boundaries"] * 1000, 1),
                    round(out_scores["avg"]["tse_word_positions"] * 1000, 1),
                ),
                file=f,
            )
