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
from sisyphus import tk


def py():
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    prefix = "exp2024_09_16_grad_align/"

    ctc_model = sis_get_model(
        "v6-relPosAttDef-noBias"
        "-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
        "-featBN-speedpertV2-spm10k-bpeSample001"
    )

    # Note: task hardcoded... (and also not needed, I just need the train dataset...)
    # Note: spm10k hardcoded...
    task = get_librispeech_task_raw_v2(vocab="spm10k")
    train_dataset = task.train_dataset.copy_train_as_static()
    # train_dataset.main_dataset["fixed_random_subset"] = 1000  # for debugging...
    train_dataset.main_dataset["seq_list_filter_file"] = ...  # TODO
    # TODO with seq_list...
    # TODO probably need to translate robins seq list ... 960 to mixed 100/360/460

    alignment = ctc_forced_align(ctc_model, train_dataset)
    alignment.creator.add_alias(f"{prefix}ctc_forced_align")
    tk.register_output(f"{prefix}ctc_forced_align.hdf", alignment)

    # TODO job to dump grads, diff variants:
    #  - x * grad
    #  - using prob entropy instead of ground truth log prob
    pass

    # TODO force align CTC, calc TSE
    #   any of the new variants have influence on TSE?

    # TODO align using att weights


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

    raise ValueError(f"unknown encoder {name}")


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
