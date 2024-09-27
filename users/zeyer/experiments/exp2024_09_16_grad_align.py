"""
More on grad align
"""


from __future__ import annotations

from typing import List, Sequence
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
    )
    from .exp2024_09_09_grad_align import CalcAlignmentMetrics
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
        (
            "blankSep",  # 5.73
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-blankSep",
            "spm10k",
        ),
        (
            "base-spm512",  # 6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001",
            "spm512",
        ),
        (
            "base-spm512-blankSep",  # 6.02
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-spm10k-bpeSample001"
            "-blankSep",
            "spm512",
        ),
        (
            "base-bpe10k",  # 6.18
            "v6-relPosAttDef"
            "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
            "-featBN-speedpertV2-bpe10k-bpeSample001",
            "bpe10k",
        ),
        (
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

    # TODO job to dump grads, diff variants:
    #  - x * grad
    #  - using prob entropy instead of ground truth log prob
    pass

    # TODO force align CTC, calc TSE
    #   any of the new variants have influence on TSE?

    # TODO align using att weights


_called_ctc_py_once = False


def sis_get_model(name: str) -> ModelWithCheckpoint:
    if (
        name == "v6-relPosAttDef-noBias"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
    ):
        return train_exp(  # 5.65 (!!!)
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
        ).get_last_fixed_epoch()

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import py as ctc_py, _train_experiments

    global _called_ctc_py_once

    if not _called_ctc_py_once:
        ctc_py()
        _called_ctc_py_once = True

    return _train_experiments[name].get_last_fixed_epoch()


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
