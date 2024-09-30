"""
More on grad align
"""


from __future__ import annotations

from typing import Optional, Any, List, Sequence, Dict
from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import ModelWithCheckpoint

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    train_exp,
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    speed_pert_librosa_config,
    Model,
    ctc_model_def,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    _get_cfg_lrlin_oclr_by_bs_nep,
    _log_mel_feature_dim,
    _batch_size_factor,
    _get_cfg_lrlin_oclr_by_bs_nep,
    _get_bos_idx,
    _get_eos_idx,
)
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoder
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ForwardRFDef
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from sisyphus import tk, Path


def py():
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        seq_list_960_to_split_100_360_500,
        seq_list_split_100_360_500_to_single_960,
    )
    from .exp2024_09_09_grad_align import CalcAlignmentMetrics, ForcedAlignOnScoreMatrixJob
    from i6_core.text.label.sentencepiece.vocab import ExtractSentencePieceVocabJob

    prefix = "exp2024_09_16_grad_align/"

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
    seq_list_ref = Path(
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
    )
    seq_list = seq_list_960_to_split_100_360_500(seq_list_ref)
    vocabs = {
        "spm10k": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm10k").model_file).out_vocab, 10_240),
        "spm512": ("spm", ExtractSentencePieceVocabJob(get_vocab_by_str("spm512").model_file).out_vocab, 512),
        "bpe10k": ("bpe", get_vocab_by_str("bpe10k").vocab, 10_025),
    }

    for shortname, fullname, vocab in [
        (  # 110.7/43.7ms
            "noBias",  # 5.65, better baseline
            "v6-relPosAttDef-noBias"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            "spm10k",
        ),
        (  # 111.5/52.9ms
            "base",  # 5.77
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            "spm10k",
        ),
        (  # 116.8/74.4ms
            "lpNormedGradC05_11P1",  # 5.71
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-lpNormedGradC05_11P1",
            "spm10k",
        ),
        (  # 98.5/77.6ms
            "blankSep",  # 5.73
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-blankSep",
            "spm10k",
        ),
        (  # 75.4/42.7ms
            "base-spm512",  # 6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm512-bpeSample001",
            "spm512",
        ),
        (  # 59.6/48.5ms
            "base-spm512-blankSep",  # 6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm512-bpeSample001"
            "-blankSep",
            "spm512",
        ),
        (  # 113.9/68.1ms
            "base-bpe10k",  # 6.18
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-bpe10k-bpeSample001",
            "bpe10k",
        ),
        (  # 84.9/64.2ms
            "base-bpe10k-blankSep",  # 5.98
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-bpe10k-bpeSample001"
            "-blankSep",
            "bpe10k",
        ),
    ]:
        # Note: task hardcoded... (and also not needed, I just need the train dataset...)
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        # train_dataset.main_dataset["fixed_random_subset"] = 1000  # for debugging...
        train_dataset.main_dataset["seq_list_filter_file"] = seq_list

        ctc_model = sis_get_model(fullname)

        alignment = ctc_forced_align(ctc_model, train_dataset)
        alignment.creator.add_alias(f"{prefix}ctc_{shortname}_forced_align/align")
        tk.register_output(f"{prefix}ctc_{shortname}_forced_align/align.hdf", alignment)

        name = f"ctc_{shortname}_forced_align/metrics"
        job = CalcAlignmentMetrics(
            seq_list=seq_list,
            seq_list_ref=seq_list_ref,
            alignment_hdf=alignment,
            alignment_label_topology="ctc",
            alignment_bpe_vocab=vocabs[vocab][1],
            alignment_bpe_style=vocabs[vocab][0],
            alignment_blank_idx=vocabs[vocab][2],
            features_sprint_cache=features_sprint_cache,
            ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
            ref_alignment_allophones=gmm_alignment_allophones,
            ref_alignment_len_factor=6,
        )
        job.add_alias(prefix + name)
        tk.register_output(prefix + name + ".json", job.out_scores)
        tk.register_output(prefix + name + ".short_report.txt", job.out_short_report_str)

        for extra_name, grad_opts in [
            ("", {}),
            ("-blankStopGrad", {"stop_grad_blank": True}),
            *([("-bs1", {"max_seqs": 1})] if shortname == "base" else []),  # test influence of batching
        ]:
            grad_opts = grad_opts.copy()
            # base model
            epoch = grad_opts.pop("epoch", -1)
            ctc_model = sis_get_model(fullname, epoch=epoch)

            # Now grad based align
            grads = get_input_grads(
                ctc_model,
                train_dataset,
                config={
                    **({"fixed_blank_sep_v1": True} if "blankSep" in shortname else {}),
                    **grad_opts,
                },
            )
            if shortname == "blankSep":
                # I get some strange CUDA error: an illegal memory access was encountered,
                # maybe this fixes it:
                grads.creator.set_env("CUDA_LAUNCH_BLOCKING", "1")
            tk.register_output(f"{prefix}ctc_{shortname}{extra_name}_input_grads/grads.hdf", grads)
            grads.creator.add_alias(f"{prefix}ctc_{shortname}{extra_name}_input_grads/grads")

            # see also exp2024_09_09_grad_align.py
            opts = {"grad_name": f"ctc_{shortname}{extra_name}_input_grads", "sm": True, "blank_score": -6}
            opts = opts.copy()
            apply_softmax_over_time = opts.pop("sm", False)
            grad_name = opts.pop("grad_name")
            # factor, grad_hdf = grads[grad_name]
            factor = 1
            grad_hdf = grads

            # The dumped grads cover about 9.6h audio from train.
            name = f"grad-align-{grad_name}-sm{apply_softmax_over_time}"
            if opts:
                for k, v in opts.items():
                    name += f"-{k}{v}"
            job = ForcedAlignOnScoreMatrixJob(
                score_matrix_hdf=grad_hdf,
                cut_off_eos=False,
                apply_softmax_over_time=apply_softmax_over_time,
                # Need to know blank idx for the generated output alignment.
                num_labels=vocabs[vocab][2] + 1,
                blank_idx=vocabs[vocab][2],
                returnn_dataset=train_dataset.get_main_dataset(),
                **opts,
            )
            job.add_alias(prefix + name + "/align")
            tk.register_output(prefix + name + "/align.hdf", job.out_align)
            alignment_hdf = job.out_align

            name += "/metrics"
            job = CalcAlignmentMetrics(
                seq_list=seq_list,
                seq_list_ref=seq_list_ref,
                alignment_hdf=alignment_hdf,
                alignment_bpe_vocab=vocabs[vocab][1],
                alignment_bpe_style=vocabs[vocab][0],
                alignment_blank_idx=vocabs[vocab][2],
                features_sprint_cache=features_sprint_cache,
                ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                ref_alignment_allophones=gmm_alignment_allophones,
                ref_alignment_len_factor=factor,
            )
            job.add_alias(prefix + name)
            tk.register_output(prefix + name + ".json", job.out_scores)
            tk.register_output(prefix + name + "_short_report.txt", job.out_short_report_str)

    # Grad align debug
    for name, grad_opts in [
        ("base", {}),  # 98.0/74.6
        # ("base", {"epoch": 80}),  # 113.4/93.9
        ("base-p0.1", {"grad_norm_p": 0.1}),
        ("base-multSource", {"source_grad_mult_with_source": True}),  # 101.2/79.8
        ("base-blankStopGrad", {"stop_grad_blank": True}),  # 97.3/76.7
        # ("base-blankStopGrad", {"stop_grad_blank": True, "epoch": 160}),  # 107.7/87.6
        # ("base-blankStopGrad", {"stop_grad_blank": True, "epoch": 320}),  # 103.1/81.7
        ("base-blankStopGrad-p0.1", {"stop_grad_blank": True, "grad_norm_p": 0.1}),  # 95.8/77.1
        ("base-blankStopGrad-p0.5", {"stop_grad_blank": True, "grad_norm_p": 0.5}),  # 96.0/76.9
        ("base-blankStopGrad-p1", {"stop_grad_blank": True, "grad_norm_p": 1}),  # 96.6/76.7. seems better than 2 or 3
        ("base-blankStopGrad-p3", {"stop_grad_blank": True, "grad_norm_p": 3}),  # 97.9/76.9
        (
            "base-blankStopGrad-inclBlankState",
            {"stop_grad_blank": True, "ctc_partial_scores_include_next_blank": True},
        ),  # 97.3/76.7
    ]:
        grad_opts = grad_opts.copy()
        # base model
        epoch = grad_opts.pop("epoch", -1)
        ctc_model = sis_get_model(
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            epoch=epoch,
        )
        vocab = "spm10k"
        task = get_librispeech_task_raw_v2(vocab=vocab)
        train_dataset = task.train_dataset.copy_train_as_static()
        train_dataset.main_dataset["fixed_random_subset"] = 100  # for debugging...
        # train_dataset.main_dataset["seq_list_filter_file"] = seq_list
        grads = get_input_grads(ctc_model, train_dataset, grad_opts)
        tk.register_output(f"{prefix}ctc_{name}_input_grads_debug-ep{epoch}/grads.hdf", grads)
        grads.creator.add_alias(f"{prefix}ctc_{name}_input_grads_debug-ep{epoch}/grads")

        # see also exp2024_09_09_grad_align.py
        for opts in [
            {"sm": True, "blank_score": -6},
            # Always a bit better (e.g. 94.0/75.3 vs 97.3/76.7), but more heuristic:
            {
                "sm": True,
                "blank_score": "calc",
                "blank_score_est": "flipped_after_softmax_over_time",
                "non_blank_score_reduce": "log_mean_exp",
                "blank_score_flipped_percentile": 60,
                "apply_softmax_over_labels": True,
            },
        ]:
            opts = opts.copy()
            apply_softmax_over_time = opts.pop("sm", False)
            grad_name = f"ctc_{name}_input_grads_debug-ep{epoch}"
            # factor, grad_hdf = grads[grad_name]
            factor = 1
            grad_hdf = grads

            # The dumped grads cover about 9.6h audio from train.
            name = f"grad-align-{grad_name}-sm{apply_softmax_over_time}"
            if opts:
                for k, v in opts.items():
                    name += f"-{k}{v}"
            job = ForcedAlignOnScoreMatrixJob(
                score_matrix_hdf=grad_hdf,
                cut_off_eos=False,
                apply_softmax_over_time=apply_softmax_over_time,
                # Need to know blank idx for the generated output alignment.
                num_labels=vocabs[vocab][2] + 1,
                blank_idx=vocabs[vocab][2],
                returnn_dataset=train_dataset.get_main_dataset(),
                **opts,
            )
            job.add_alias(prefix + name + "/align")
            tk.register_output(prefix + name + "/align.hdf", job.out_align)
            alignment_hdf = job.out_align

            from i6_experiments.users.zeyer.datasets.utils.extract_seq_list import ExtractSeqListJob

            ds = train_dataset.get_main_dataset().copy()
            ds["audio"] = None
            ds["targets"] = None

            seq_list_debug = ExtractSeqListJob(returnn_dataset=ds).out_seq_list
            seq_list_debug_ref = seq_list_split_100_360_500_to_single_960(seq_list_debug)

            name += "/metrics"
            job = CalcAlignmentMetrics(
                seq_list=seq_list_debug,
                seq_list_ref=seq_list_debug_ref,
                alignment_hdf=alignment_hdf,
                alignment_bpe_vocab=vocabs[vocab][1],
                alignment_bpe_style=vocabs[vocab][0],
                alignment_blank_idx=vocabs[vocab][2],
                features_sprint_cache=features_sprint_cache,
                ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
                ref_alignment_allophones=gmm_alignment_allophones,
                ref_alignment_len_factor=factor,
            )
            job.add_alias(prefix + name)
            tk.register_output(prefix + name + ".json", job.out_scores)
            tk.register_output(prefix + name + "_short_report.txt", job.out_short_report_str)

    # TODO job to dump grads, diff variants:
    #  - using prob entropy instead of ground truth log prob
    pass

    # TODO align using att weights


_called_ctc_py_once = False


def sis_get_model(name: str, *, epoch: int = -1) -> ModelWithCheckpoint:
    if (
        name == "v6-relPosAttDef-noBias"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
    ):
        exp = train_exp(  # 5.65 (!!!)
            "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
            "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
            config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            model_config={
                "enc_conformer_layer": rf.build_dict(
                    rf.encoder.conformer.ConformerEncoderLayer,
                    ff=rf.build_dict(
                        rf.encoder.conformer.ConformerPositionwiseFeedForward,
                        activation=rf.build_dict(rf.relu_square),
                        with_bias=False,
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            },
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        )

    else:

        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import py as ctc_py, _train_experiments

        global _called_ctc_py_once

        if not _called_ctc_py_once:
            from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

            with disable_register_output():
                ctc_py()
            _called_ctc_py_once = True

        exp = _train_experiments[name]

    if epoch < 0:
        return exp.get_last_fixed_epoch()
    return exp.get_epoch(epoch)


def ctc_forced_align(model: ModelWithCheckpoint, dataset: DatasetConfig) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_dict = dataset.get_extern_data()
    default_input_dict = extern_data_dict[dataset.get_default_input()]
    input_dims: Sequence[Dim] = (
        default_input_dict["dims"] if "dims" in default_input_dict else default_input_dict["dim_tags"]
    )
    assert isinstance(input_dims, (tuple, list)) and all(isinstance(dim, Dim) for dim in input_dims)
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    classes_dim = default_target_dict["sparse_dim"]
    assert isinstance(classes_dim, Dim)
    classes_with_blank_dim = classes_dim + 1

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_ctc_model_forced_align_step,
        config={
            "model_outputs": {
                "output": {"shape": (None,), "sparse_dim": classes_with_blank_dim},
                "scores": {"shape": ()},
            }
        },
        forward_rqmt={"time": 12},
    )


def _ctc_model_forced_align_step(*, model: Model, extern_data: TensorDict, **_kwargs):
    """
    :param model: model with batch size 1
    """
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.fsa import best_path_ctc

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    out_spatial_dim = expected_output.dims[-1]

    logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
    path, score = best_path_ctc(
        logits=logits,
        input_spatial_dim=enc_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets.get_time_dim_tag(),
        blank_index=model.blank_idx,
    )
    out_spatial_dim.declare_same_as(enc_spatial_dim)
    path.mark_as_default_output(shape=[batch_dim, enc_spatial_dim])
    score.mark_as_output("scores", shape=[batch_dim])


def get_input_grads(
    model: ModelWithCheckpoint, dataset: DatasetConfig, config: Optional[Dict[str, Any]] = None
) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_dict = dataset.get_extern_data()
    default_input_dict = extern_data_dict[dataset.get_default_input()]
    input_dims: Sequence[Dim] = (
        default_input_dict["dims"] if "dims" in default_input_dict else default_input_dict["dim_tags"]
    )
    assert isinstance(input_dims, (tuple, list)) and all(isinstance(dim, Dim) for dim in input_dims)
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    batch_dim, out_spatial_dim = default_target_dict["dim_tags"]
    feat_spatial_dim = Dim(None, name="feat_spatial_dim")  # it's not the input raw audio spatial dim...

    if config:
        config = config.copy()
    else:
        config = {}
    config.setdefault("__batch_size_dependent", True)
    config.setdefault("batch_size", 10_000 * _batch_size_factor)  # grads need more mem

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_ctc_model_get_input_grads_step,
        config={
            "model_outputs": {
                "output": {"dims": [batch_dim, out_spatial_dim, feat_spatial_dim], "dtype": "float32"},
                "feat_size": {"dims": [batch_dim], "dtype": "int32"},
                "targets_size": {"dims": [batch_dim], "dtype": "int32"},
                "partial_scores": {"dims": [batch_dim, out_spatial_dim], "dtype": "float32"},
            },
            **config,
        },
        forward_rqmt={"time": 24},
    )


def _ctc_model_get_input_grads_step(*, model: Model, extern_data: TensorDict, **_kwargs):
    import torch
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.fsa import ctc_partial_scores
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    feat_spatial_dim_ = expected_output.dims[-1]

    # Call: logits, enc, enc_spatial_dim = model(source, in_spatial_dim=source.get_time_dim_tag())
    in_spatial_dim = source.get_time_dim_tag()

    # log mel filterbank features
    source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
        source,
        in_spatial_dim=in_spatial_dim,
        out_dim=model.in_dim,
        sampling_rate=16_000,
    )
    in_spatial_dim.get_size_tensor().mark_as_output("feat_size")
    feat_spatial_dim_.declare_same_as(in_spatial_dim)
    if model.feature_batch_norm:
        source = model.feature_batch_norm(source)
    if model.feature_norm:
        source = rf.normalize(source, axis=in_spatial_dim)
    if model.feature_stats:
        source = (source - model.feature_stats.mean) / model.feature_stats.std_dev

    with torch.enable_grad():
        # Just that we have well-defined dim order, for the grad logic below.
        source = source.copy_transpose((batch_dim, in_spatial_dim, model.in_dim))  # [B,T_in,D]
        source.raw_tensor.requires_grad = True

        # (No Mixup, no SpecAugment)

        # Encoder including convolutional frontend
        enc, enc_spatial_dim = model.encoder(source, in_spatial_dim=in_spatial_dim)
        logits = model.enc_logits(enc)
        if config.bool("stop_grad_blank", False):
            # Just that we have well-defined dim order, for the grad logic.
            logits = logits.copy_transpose((batch_dim, enc_spatial_dim, model.wb_target_dim))

            # noinspection PyShadowingNames
            def _zero_grad_blank_hook(grad):
                grad = grad.clone()
                grad[:, :, model.blank_idx] = 0
                return grad

            logits.raw_tensor.register_hook(_zero_grad_blank_hook)

        if model.out_blank_separated:
            assert config.bool("fixed_blank_sep_v1", False)
        log_probs = model.log_probs_wb_from_logits(logits)

        targets_spatial_dim = targets.get_time_dim_tag()
        targets_spatial_dim.get_size_tensor().mark_as_output("targets_size")
        scores = ctc_partial_scores(
            logits=log_probs,
            logits_normalized=True,
            input_spatial_dim=enc_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            blank_index=model.blank_idx,
            include_next_blank=config.bool("ctc_partial_scores_include_next_blank", False),
        )  # [B,T_out]
        scores.mark_as_output("partial_scores")
        scores_ta = TensorArray.unstack(scores, axis=targets_spatial_dim)

        source_grad_mult_with_source = config.bool("source_grad_mult_with_source", False)
        grad_norm_p = config.float("grad_norm_p", 2.0)

        grad_norms = []
        for t in range(targets_spatial_dim.get_dim_value()):
            source.raw_tensor.grad = None
            scores_t = scores_ta[t]  # [B], +log prob
            partial_loss = scores_t.raw_tensor.sum()
            partial_loss.backward(retain_graph=True)
            grad: torch.Tensor = source.raw_tensor.grad  # [B,T_in,D]  # noqa
            if source_grad_mult_with_source:
                grad = grad * source.raw_tensor
            grad_norm = torch.norm(grad, p=grad_norm_p, dim=2)  # [B,T_in]
            grad_norms.append(grad_norm)
        grad_norms = torch.stack(grad_norms, dim=0)  # [T_out,B,T_in]
        grad_norms_ = rf.convert_to_tensor(grad_norms, dims=[targets_spatial_dim, batch_dim, in_spatial_dim])
        grad_norms_.mark_as_default_output()
