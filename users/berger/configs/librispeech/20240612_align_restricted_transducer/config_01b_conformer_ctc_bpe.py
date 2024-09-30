import copy
import os
from typing import List, Optional

import i6_core.rasr as rasr
from i6_models.parts.conformer.norm import LayerNormNC
import torch
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data_bpe
from i6_experiments.users.berger.pytorch.custom_parts.specaugment import (
    SpecaugmentByLengthConfigV1,
    SpecaugmentByLengthModuleV1,
)
from i6_experiments.users.berger.pytorch.models import conformer_ctc_minireturnn as conformer_ctc
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, ReturnnConfigs, SummaryKey
from i6_experiments.users.berger.systems.returnn_native_system import ReturnnNativeSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_models.assemblies.conformer import (
    ConformerBlockV2Config,
    ConformerEncoderV2,
    ConformerEncoderV2Config,
)
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1Config,
)
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import (
    LogMelFeatureExtractionV1,
    LogMelFeatureExtractionV1Config,
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

bpe_size = 128
target_size = 185
num_subepochs = 1000
sub_checkpoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 960, 970, 980, 990, 1000]

tools = copy.deepcopy(default_tools_v2)
assert tools.returnn_root is not None
normal_returnn = tools.returnn_root
tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfig:
    feature_extraction = ModuleFactoryV1(
        module_class=RasrCompatibleLogMelFeatureExtractionV1,
        cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            min_amp=1.175494e-38,
            num_filters=80,
            alpha=0.97 if kwargs.get("preemphasis", False) else 0.0,
        ),
    )
    # feature_extraction = ModuleFactoryV1(
    #     module_class=LogMelFeatureExtractionV1,
    #     cfg=LogMelFeatureExtractionV1Config(
    #         sample_rate=16000,
    #         win_size=0.025,
    #         hop_size=0.01,
    #         f_min=60,
    #         f_max=7600,
    #         min_amp=1e-10,
    #         num_filters=80,
    #         center=False,
    #         n_fft=400,
    #     ),
    # )

    specaugment = ModuleFactoryV1(
        module_class=SpecaugmentByLengthModuleV1,
        cfg=SpecaugmentByLengthConfigV1(
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=5,
            freq_mask_max_size=16,
        ),
    )

    frontend = ModuleFactoryV1(
        GenericFrontendV1,
        GenericFrontendV1Config(
            in_features=80,
            layer_ordering=[
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Activation,
                FrontendLayerType.Pool2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Conv2d,
                FrontendLayerType.Activation,
                FrontendLayerType.Pool2d,
            ],
            conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
            conv_out_dims=[32, 64, 64, 32],
            conv_strides=None,
            conv_paddings=None,
            pool_kernel_sizes=[(2, 1), (2, 1)],
            pool_strides=None,
            pool_paddings=None,
            activations=[torch.nn.ReLU(), torch.nn.ReLU()],
            out_features=512,
        ),
    )

    ff_cfg = ConformerPositionwiseFeedForwardV1Config(
        input_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        activation=torch.nn.SiLU(),
    )

    mhsa_cfg = ConformerMHSAV1Config(
        input_dim=512,
        num_att_heads=8,
        att_weights_dropout=0.1,
        dropout=0.1,
    )

    conv_cfg = ConformerConvolutionV1Config(
        channels=512,
        kernel_size=31,
        dropout=0.1,
        activation=torch.nn.SiLU(),
        norm=LayerNormNC(512),
    )

    block_cfg = ConformerBlockV2Config(
        ff_cfg=ff_cfg,
        mhsa_cfg=mhsa_cfg,
        conv_cfg=conv_cfg,
        modules=["ff", "conv", "mhsa", "ff"],
        scales=[0.5, 1.0, 1.0, 0.5],
    )

    conformer_cfg = ConformerEncoderV2Config(
        num_layers=12,
        frontend=frontend,
        block_cfg=block_cfg,
    )

    model_config = conformer_ctc.ConformerCTCConfig(
        feature_extraction=feature_extraction,
        specaugment=specaugment,
        conformer=ModuleFactoryV1(ConformerEncoderV2, cfg=conformer_cfg),
        dim=512,
        target_size=target_size,
        dropout=0.1,
        specaug_start_epoch=11,
    )

    if variant == ConfigVariant.TRAIN:
        extra_config: dict = {
            "train": train_data_config,
            "dev": dev_data_config,
            "max_seq_length": {"data": 35 * 16000},
            "torch_amp_options": {"dtype": "bfloat16"},
            "stop_on_nonfinite_train_score": True,
            "num_workers_per_gpu": 2,
        }
    if variant == ConfigVariant.PRIOR:
        extra_config: dict = {
            "forward": train_data_config,
            "torch_amp_options": {"dtype": "bfloat16"},
        }
    if variant == ConfigVariant.RECOG:
        extra_config = {}

    if variant == ConfigVariant.TRAIN:
        serializer_kwargs = {"train_type": conformer_ctc.TrainType.TORCH_CTC_LOSS, "blank_idx": target_size - 1}
    if variant == ConfigVariant.PRIOR:
        serializer_kwargs = {}
    if variant == ConfigVariant.RECOG:
        serializer_kwargs = {
            "recog_type": conformer_ctc.RecogType.FLASHLIGHT,
            # "recog_type": conformer_ctc.RecogType.GREEDY,
            "beam_size": kwargs.get("beam_size", 1024),
            "beam_threshold": kwargs.get("beam_threshold", 14.0),
            "silence_token": "<blank>",
        }
        if kwargs.get("beam_size_token", None):
            serializer_kwargs["beam_size_token"] = kwargs.get("beam_size_token", None)

    return get_returnn_config(
        num_epochs=num_subepochs,
        target="classes",
        python_prolog=[
            "import sys",
            "sys.path.insert(0, '/u/berger/asr-exps/librispeech/20240612_align_restricted_transducer/recipe')",
            "sys.path.insert(0, '/work/asr4/berger/repositories/rasr_versions/master/lib/linux-x86_64-standard')",
            Import("i6_experiments.users.berger.corpus.general.speed_perturbation.legacy_speed_perturbation"),
        ],
        extra_python=[
            conformer_ctc.get_serializer(model_config, variant=variant, **serializer_kwargs),
        ],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        use_lovely_tensors=False,
        grad_noise=None,
        grad_clip=1.0,
        optimizer=Optimizers.AdamW,
        weight_decay=kwargs.get("weight_decay", 0.01),
        schedule=LearningRateSchedules.OCLR_V2,
        keep_last_n=1,
        keep_best_n=0,
        keep=sub_checkpoints,
        inc_epochs=480,
        initial_lr=kwargs.get("initial_lr", 7e-06),
        peak_lr=kwargs.get("peak_lr", 5e-04),
        decayed_lr=kwargs.get("decay_lr", 5e-05),
        final_lr=1e-07,
        batch_size=kwargs.get("batch_size", 36000 * 160),
        use_chunking=False,
        extra_config=extra_config,
        use_base_config=False,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    beam_sizes: Optional[List[int]] = None,
    beam_thresholds: Optional[List[float]] = None,
    beam_sizes_token: Optional[List[int]] = None,
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    if beam_sizes and beam_thresholds and beam_sizes_token:
        recog_configs = {
            f"recog_bs-{beam_size}_bt-{beam_threshold}_bst-{beam_size_token}": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                beam_size=beam_size,
                beam_threshold=beam_threshold,
                beam_size_token=beam_size_token,
                **kwargs,
            )
            for beam_size in beam_sizes
            for beam_threshold in beam_thresholds
            for beam_size_token in beam_sizes_token
        }
    else:
        recog_configs = {
            "recog": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                **kwargs,
            )
        }

    return ReturnnConfigs(
        train_config=returnn_config_generator(
            variant=ConfigVariant.TRAIN,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            **kwargs,
        ),
        prior_config=returnn_config_generator(
            variant=ConfigVariant.PRIOR,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            **kwargs,
        ),
        recog_configs=recog_configs,
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_librispeech_data_bpe(
        bpe_size=bpe_size,
        returnn_root=normal_returnn,
        returnn_python_exe=tools.returnn_python_exe,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
        partition_epoch=10,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_ctc_flashlight_bpe_recog_step_args(
        epochs=sub_checkpoints,
        prior_scales=[0.3],
        lm_scales=[2.0],
        ogg_dataset=True,
    )
    # recog_args = exp_args.get_ctc_greedy_bpe_recog_step_args(
    #     epochs=sub_checkpoints,
    #     prior_scales=[0.3],
    #     ogg_dataset=True,
    # )

    # ********** System **********

    system = ReturnnNativeSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.RECOG_NAME,
            SummaryKey.CORPUS,
            SummaryKey.EPOCH,
            SummaryKey.LM,
            SummaryKey.PRIOR,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.DEL,
            SummaryKey.INS,
            SummaryKey.ERR,
            SummaryKey.RTF,
        ],
        summary_sort_keys=[SummaryKey.ERR, SummaryKey.CORPUS],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring()

    data.train_data_config = copy.deepcopy(data.train_data_config)
    data.train_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.97
    data.train_data_config["datasets"]["data"]["audio"]["pre_process"] = CodeWrapper("legacy_speed_perturbation")

    data.cv_data_config = copy.deepcopy(data.cv_data_config)
    data.cv_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.97

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_bpe-128",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            preemphasis=False,
        ),
    )

    data.train_data_config = copy.deepcopy(data.train_data_config)
    data.train_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.0
    data.cv_data_config = copy.deepcopy(data.cv_data_config)
    data.cv_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.0

    system.add_experiment_configs(
        "Conformer_CTC_bpe-128_model-preemph",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            preemphasis=True,
        ),
    )

    data.train_data_config = copy.deepcopy(data.train_data_config)
    data.train_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.97
    data.cv_data_config = copy.deepcopy(data.cv_data_config)
    data.cv_data_config["datasets"]["data"]["audio"]["preemphasis"] = 0.97

    system.run_train_step(**train_args)
    system.run_dev_recog_step(
        exp_names=["Conformer_CTC_bpe-128"],
        extra_audio_config={"preemphasis": 0.97},
        **recog_args,
    )
    system.run_dev_recog_step(
        exp_names=["Conformer_CTC_bpe-128_model-preemph"],
        **recog_args,
    )

    # recog_args["epochs"] = sub_checkpoints[-1:]
    # recog_args["lm_scales"] = [1.4, 1.6, 1.8, 2.0, 2.2]
    # system.run_dev_recog_step(
    #     exp_names=["Conformer_CTC_bpe-128"],
    #     extra_audio_config={"preemphasis": 0.97},
    #     **recog_args,
    # )
    # recog_args["epochs"] = sub_checkpoints[-1:]
    # recog_args["prior_scales"] = [0.2, 0.3, 0.4]
    # recog_args["lm_scales"] = [1.5, 1.6, 1.7]
    # system.run_dev_recog_step(
    #     exp_names=["Conformer_CTC_bpe-128"],
    #     extra_audio_config={"preemphasis": 0.97},
    #     **recog_args,
    # )

    system.add_experiment_configs(
        "Conformer_CTC_bpe-128",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            preemphasis=False,
            beam_sizes=[512, 1024],
            beam_sizes_token=[0, 16, 32],
            beam_thresholds=[10.0, 50.0, 100.0, 500.0],
        ),
    )
    recog_args["epochs"] = sub_checkpoints[-1:]
    recog_args["prior_scales"] = [0.2]
    recog_args["lm_scales"] = [1.5]
    system.run_dev_recog_step(
        exp_names=["Conformer_CTC_bpe-128"],
        extra_audio_config={"preemphasis": 0.97},
        **recog_args,
    )

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
