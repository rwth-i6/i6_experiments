from sisyphus import tk
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
)

from i6_experiments.users.zeyer.datasets.loquacious import get_loquacious_text_only_dataset_v2

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


__setup_root_prefix__ = "exp2025_11_22_lm_scaling_laws_loquacious"


def py():
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023 import (
        recog_ext_with_lm,
        recog_ext_labelwise_with_lm,
    )

    train_base_asr_models()
    train_lms()

    # Scaling laws for std LMs (num params, num epochs, total training time)

    from i6_experiments.users.zeyer.train_v4 import train_models_by_prefix as train_v4_models
    from i6_experiments.users.zeyer.returnn.model_num_params_from_config import GetNumParamsFromReturnnConfigJob
    from i6_experiments.users.zeyer.returnn.total_runtime_from_training import GetTotalRuntimeFromReturnnTrainingJob
    from i6_experiments.users.zeyer.plots.scaling_laws import ScalingLawPlotJob

    # baselines
    params_points = {}
    time_points = {}

    ctc_model_name = "L16-D1024-spm10k-auxAED-b100k"

    params_points_ = params_points["CTC+LM"] = []
    time_points_ = time_points["CTC+LM"] = []

    for lm_name, exp in train_v4_models.items():
        if not lm_name.startswith("lm/"):
            continue

        num_params = GetNumParamsFromReturnnConfigJob(exp.get_training_job().returnn_config).out_num_params
        tk.register_output(f"{lm_name}/num_params.txt", num_params)

        train_time_secs = GetTotalRuntimeFromReturnnTrainingJob(exp.scores_and_learning_rates).out_train_time_secs
        tk.register_output(f"{lm_name}/total_train_time_secs.txt", train_time_secs)

        res = recog_ext_with_lm(
            ctc_model_name=ctc_model_name,
            lm_name=lm_name[len("lm/") :],
            lm=exp.get_last_fixed_epoch(),
            ctc_soft_collapse_threshold=0.9,
        )
        res_wer = res.get_main_measure_value_as_variable()
        if res_wer.available():  # TODO remove this check...
            params_points_.append((num_params, res_wer))
            time_points_.append((train_time_secs / 60 / 60, res_wer))

    tk.register_output(
        "lm/scaling_plot.pdf",
        ScalingLawPlotJob(
            x_label="Train time [h]", y_label="WER [%]", points=time_points, filter_outliers=True
        ).out_plot_pdf,
    )


def train_base_asr_models():
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        train_exp as ctc_train_exp,
        _raw_sample_rate,
        _batch_size_factor,
        speed_pert_librosa_config,
    )

    for num_layers, num_dims, batch_size in [(16, 1024, 100_000)]:
        # Baseline (without using TTS).
        # Warning: this keeps aux_loss_layers=[4, 8], not sure if this is optimal...
        ctc_train_exp(
            f"L{num_layers}-D{num_dims}-spm10k-auxAED-b{batch_size // 1000}k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_build_dict": rf.build_dict(
                    # ConformerEncoder(in_dim, enc_model_dim, **enc_opts)
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=num_layers,
                    out_dim=num_dims,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(batch_size, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                # purely used for training
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )


def train_lms():
    from i6_experiments.users.zeyer.train_v4 import train
    from i6_experiments.users.zeyer.experiments.exp2025_10_04_loquacious import (
        lm_model_def,
        lm_train_def,
        train_lms as base_lm_exps,
    )

    prefix = get_setup_prefix_for_module(__name__)

    base_lm_exps()

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
        {"n": 32},  # baseline
        {"d": 1024},  # baseline
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
        {"n": 4, "nEp": 150},
        {"n": 4, "nEp": 25},
        {"n": 4, "d": 512, "nEp": 25},
        {"n": 3},
        {"n": 2},
        {"n": 3, "nEp": 25},
        {"n": 2, "nEp": 25},
        {"nEp": 10},
        {"nEp": 25},
        {"nEp": 50},  # baseline
        {"nEp": 100},
        {"nEp": 150},
        {"nEp": 200},
        {"drop": 0.1, "nEp": 25},
        {"drop": 0.1, "nEp": 50},
        {"drop": 0.1, "nEp": 100},
        {"drop": 0.1, "nEp": 150},
        {"drop": 0.1, "nEp": 200},
        {"n": 24, "d": 1280, "drop": 0.1, "nEp": 100},
        {"n": 24, "d": 1280, "drop": 0.1, "nEp": 150},
        # intended to be similar as Dorians DLMs (wrong, but anyway keep that now...)
        {"n": 6, "a": 2, "d": 512, "lr": 0.5, "nEp": 100},
        {"n": 18, "a": 6, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 9, "a": 4, "d": 512, "lr": 1.25, "nEp": 25},
        {"n": 3, "a": 1, "d": 512, "lr": 1.25, "nEp": 50},
        {"n": 3, "a": 1, "d": 256, "lr": 0.5, "nEp": 25},
        {"n": 12, "a": 4, "d": 768, "lr": 0.5, "nEp": 100},
        {"n": 3, "a": 1, "d": 128, "lr": 1.25, "nEp": 25},
        {"n": 30, "a": 16, "d": 1024, "lr": 0.5, "nEp": 50},
        {"n": 18, "a": 6, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 30, "a": 16, "d": 1024, "lr": 1.0, "nEp": 50},
        # similar as Dorians DLMs
        {"n": 8, "d": 512, "lr": 0.5, "nEp": 100},
        {"n": 24, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 12, "d": 512, "lr": 1.25, "nEp": 25},
        {"n": 4, "d": 512, "lr": 1.25, "nEp": 50},
        {"n": 4, "d": 256, "lr": 0.5, "nEp": 25},
        {"n": 16, "d": 768, "lr": 0.5, "nEp": 100},
        {"n": 4, "d": 128, "lr": 1.25, "nEp": 25},
        {"n": 40, "d": 1024, "lr": 0.5, "nEp": 50},
        {"n": 24, "d": 768, "lr": 0.5, "nEp": 25},
        {"n": 40, "d": 1024, "lr": 1.0, "nEp": 50},
        {"n": 2, "d": 128},
        # similar as from MDLM
        {"n": 12, "a": 12, "d": 768, "drop": 0.1, "nEp": 50},  # like small mdlm
        {"n": 24, "a": 16, "d": 1024, "drop": 0.1, "nEp": 50},  # like medium mdlm
    ]:
        # n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop0-b400_20k-nEp100 by default
        # reorder and set defaults
        opts = {"n": opts.pop("n", 32), "d": opts.pop("d", 1024), **opts}
        name = f"{prefix}/lm/trafo-v2-{_name_for_dict(opts)}-spm10k"
        n_l = opts.pop("n")
        dim = opts.pop("d")
        n_ep = opts.pop("nEp", 50)
        lr = opts.pop("lr", 1.0)
        num_heads = opts.pop("a", None)
        drop = opts.pop("drop", 0.0)
        att_drop = opts.pop("adrop", drop)
        assert not opts
        train(
            name,
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


def _name_for_dict(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        v = str(v).replace(".", "_").replace("-", "_")
        parts.append(f"{k}{v}")
    return "-".join(parts)
