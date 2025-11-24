from __future__ import annotations

from typing import Union, Literal, Dict, Tuple
from dataclasses import dataclass

from i6_core.returnn import ReturnnTrainingJob
from sisyphus import tk
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg, ModelWithCheckpoint
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.plots.scaling_laws import ScalingLawPlotJob

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
    get_loquacious_text_only_dataset_v2,
    get_loquacious_train_subset_dataset_v2,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)


__setup_root_prefix__ = "exp2025_11_22_lm_scaling_laws_loquacious"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    stats = get_lm_scaling_stats(only_available=True)  # TODO remove only_available...

    if stats:
        tk.register_output(
            f"{prefix}/lm_scaling_plot.pdf",
            ScalingLawPlotJob(
                x_label="Train time [h]",
                y_label="WER [%]",
                points={"CTC+LM": [(stat.train_time_hours, stat.wer) for stat in stats.values()]},
                filter_outliers=True,
            ).out_plot_pdf,
        )


@dataclass
class LmScalingStats:
    num_params: tk.Variable
    train_time_hours: tk.Variable
    wer: tk.Variable


def get_lm_scaling_stats(*, only_available: bool = False) -> Dict[str, LmScalingStats]:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_recog_recomb_labelwise_prior_auto_scale,
        ctc_labelwise_recog_auto_scale,
    )

    prefix = get_setup_prefix_for_module(__name__)
    asr_task_spm10k = get_loquacious_task_raw_v2(vocab="spm10k")

    ctc_model_name, ctc_model = get_ctc_model()
    lms = train_lms()

    # Scaling laws for std LMs (num params, num epochs, total training time)

    from i6_experiments.users.zeyer.returnn.model_num_params_from_config import GetNumParamsFromReturnnConfigJob
    from i6_experiments.users.zeyer.returnn.total_runtime_from_training import GetTotalRuntimeFromReturnnTrainingJob

    out = {}
    for lm_name, lm in lms.items():
        train_job: ReturnnTrainingJob = lm.checkpoint.path.creator
        num_params = GetNumParamsFromReturnnConfigJob(train_job.returnn_config).out_num_params
        tk.register_output(f"{prefix}/lm/{lm_name}/num_params.txt", num_params)

        train_time_secs = GetTotalRuntimeFromReturnnTrainingJob(train_job.out_learning_rates).out_train_time_secs
        tk.register_output(f"{prefix}/lm/{lm_name}/total_train_time_secs.txt", train_time_secs)

        res = ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{ctc_model_name}/ctc+lm-v2/{lm_name}",
            task=asr_task_spm10k,
            ctc_model=ctc_model,
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k"),
        )
        res_wer = res.get_main_measure_value_as_variable()
        if not only_available or res_wer.available():
            out[lm_name] = LmScalingStats(
                num_params=num_params,
                train_time_hours=train_time_secs / 60 / 60,
                wer=res_wer,
            )

    return out


def get_ctc_model(
    *, subset: str = "large", total_k_hours: int = 250, epoch: Union[Literal["max", "min"], int] = "max"
) -> Tuple[str, ModelWithCheckpoint]:
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
        train_exp as aed_train_exp,
        _raw_sample_rate,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        aed_ctc_timesync_recog_recomb_auto_scale,
    )

    prefix = get_setup_prefix_for_module(__name__)
    task_spm10k = get_loquacious_task_raw_v2(vocab="spm10k")
    # i6_experiments.users.zeyer.experiments.exp2025_10_04_loquacious.py
    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)
    name = f"base-v2-{subset}-nFullEp{num_full_ep:.1f}-nEp{n_ep}-totalHours{total_k_hours}k"
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=get_loquacious_task_raw_v2(vocab="spm10k", subset_name=subset, train_epoch_split=train_epoch_split),
        model_config={
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # "__train_audio_preprocess": speed_pert_librosa_config,
            # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    if epoch == "max":
        epoch = max(exp.fixed_epochs)
    elif epoch == "min":
        epoch = min(exp.fixed_epochs)
    assert isinstance(epoch, int)
    model = exp.get_epoch(epoch)
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=f"{prefix}/aed/{name}/ep{epoch}/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=model,
        aux_ctc_layer=16,
    )
    model_def = model.definition
    extra_config = {"aux_loss_layers": [16]}
    if isinstance(model_def, ModelDefWithCfg):
        cfg = model_def.config
        cfg = dict_update_deep(cfg, extra_config)
        model_def = ModelDefWithCfg(model_def.model_def, cfg)
    else:
        model_def = ModelDefWithCfg(model_def, extra_config)
    model = ModelWithCheckpoint(definition=model_def, checkpoint=model.checkpoint)
    return name, model


def train_lms() -> Dict[str, ModelWithCheckpoint]:
    from i6_experiments.users.zeyer.train_v4 import train
    from i6_experiments.users.zeyer.experiments.exp2025_10_04_loquacious import (
        lm_model_def,
        lm_train_def,
        train_lms as base_lm_exps,
    )

    prefix = get_setup_prefix_for_module(__name__)

    lms = base_lm_exps().copy()

    # Try various configurations.
    # The goal is to have, for e.g. a given number of training time, or given amount of parameters, etc,
    # some optimal configuration,
    # and we can see a nice pareto front when plotting them all.

    for opts in [
        {"n": 3},
        {"n": 6},
        {"n": 12},
        {"n": 16},
        {"n": 24},
        # {"n": 32},  # baseline
        # {"d": 1024},  # baseline
        {"d": 768},
        {"d": 512},
        {"d": 256},
        {"n": 16, "d": 512},
        {"n": 8, "d": 512},
        {"n": 4, "d": 512},
        {"n": 4, "d": 256},
        {"n": 4, "d": 768},
        {"n": 4},
        {"n": 4, "lr": 0.5},
        {"n": 4, "lr": 2.0},
        {"n": 4, "nEp": 100},
        # {"n": 4, "nEp": 150},
        {"n": 4, "nEp": 25},
        {"n": 4, "d": 512, "nEp": 25},
        {"n": 3},
        {"n": 2},
        {"n": 3, "nEp": 25},
        {"n": 2, "nEp": 25},
        {"nEp": 10},
        {"nEp": 25},
        # {"nEp": 50},  # baseline
        {"nEp": 100},
        {"nEp": 150},
        {"nEp": 200},
        # {"drop": 0.1, "nEp": 25},
        {"drop": 0.1, "nEp": 50},
        {"drop": 0.1, "nEp": 100},
        {"drop": 0.1, "nEp": 150},
        {"drop": 0.1, "nEp": 200},
        {"drop": 0.1, "nEp": 250},
        {"drop": 0.1, "nEp": 300},
        {"drop": 0.1, "nEp": 400},
        {"drop": 0.1, "nEp": 500},
        {"drop": 0.1, "nEp": 600},
        {"n": 24, "d": 1280, "drop": 0.1, "nEp": 100},
        {"n": 24, "d": 1280, "drop": 0.1, "nEp": 150},
        # intended to be similar as Dorians DLMs (wrong, but anyway keep that now...)
        {"n": 6, "a": 2, "d": 512, "lr": 0.5, "nEp": 100},
        # {"n": 18, "a": 6, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 9, "a": 4, "d": 512, "lr": 1.25, "nEp": 25},
        {"n": 3, "a": 1, "d": 512, "lr": 1.25, "nEp": 50},
        {"n": 3, "a": 1, "d": 256, "lr": 0.5, "nEp": 25},
        {"n": 12, "a": 4, "d": 768, "lr": 0.5, "nEp": 100},
        {"n": 3, "a": 1, "d": 128, "lr": 1.25, "nEp": 25},
        # {"n": 30, "a": 16, "d": 1024, "lr": 0.5, "nEp": 50},
        {"n": 18, "a": 6, "d": 768, "lr": 0.5, "nEp": 25},
        # {"n": 30, "a": 16, "d": 1024, "lr": 1.0, "nEp": 50},
        # similar as Dorians DLMs
        {"n": 8, "d": 512, "lr": 0.5, "nEp": 100},
        # {"n": 24, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 12, "d": 512, "lr": 1.25, "nEp": 25},
        {"n": 4, "d": 512, "lr": 1.25, "nEp": 50},
        {"n": 4, "d": 256, "lr": 0.5, "nEp": 25},
        {"n": 16, "d": 768, "lr": 0.5, "nEp": 100},
        {"n": 4, "d": 128, "lr": 1.25, "nEp": 25},
        # {"n": 40, "d": 1024, "lr": 0.5, "nEp": 50},
        {"n": 24, "d": 768, "lr": 0.5, "nEp": 25},
        # {"n": 40, "d": 1024, "lr": 1.0, "nEp": 50},
        {"n": 2, "d": 128},
        # similar as from MDLM
        {"n": 12, "a": 12, "d": 768, "drop": 0.1, "nEp": 50},  # like small mdlm
        # {"n": 24, "a": 16, "d": 1024, "drop": 0.1, "nEp": 50},  # like medium mdlm
    ]:
        # n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-nEp100 by default
        # reorder and set defaults
        opts = {"n": opts.pop("n", 32), "d": opts.pop("d", 1024), **opts}
        name = f"trafo-v2-{_name_for_dict(opts)}-spm10k"
        n_l = opts.pop("n")
        dim = opts.pop("d")
        # Note on nEp: 10 subepochs -> 1 full epoch
        n_ep = opts.pop("nEp", 50)
        lr = opts.pop("lr", 1.0)
        num_heads = opts.pop("a", None)
        drop = opts.pop("drop", 0.0)
        att_drop = opts.pop("adrop", drop)
        assert not opts
        exp = train(
            f"{prefix}/lm/{name}",
            config=dict_update_deep(
                config_96gb_bf16_accgrad1,
                {
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(20_000, n_ep, base_lr=lr, batch_size_factor=1),
                    "max_seqs": 400,
                    "optimizer.weight_decay": 1e-2,
                    "calculate_exp_loss": True,
                },
            ),
            train_dataset=get_loquacious_text_only_dataset_v2(vocab="spm10k", train_epoch_split=10),
            model_def=ModelDefWithCfg(
                lm_model_def,
                {
                    "_model_def_dict": rf.build_dict(
                        TransformerDecoder,
                        encoder_dim=None,
                        num_layers=n_l,
                        model_dim=dim,
                        pos_enc=None,
                        norm=rf.build_dict(rf.RMSNorm),
                        ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                        decoder_layer_opts=dict(
                            self_att=rf.build_dict(
                                rf.RotaryPosCausalSelfAttention,
                                with_bias=False,
                            ),
                            **({"num_heads": num_heads} if num_heads is not None else {}),
                        ),
                        dropout=drop,
                        att_dropout=att_drop,
                    )
                },
            ),
            train_def=lm_train_def,
        )
        lms[name] = exp.get_last_fixed_epoch()

    return lms


def _name_for_dict(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        v = str(v).replace(".", "_").replace("-", "_")
        parts.append(f"{k}{v}")
    return "-".join(parts)
