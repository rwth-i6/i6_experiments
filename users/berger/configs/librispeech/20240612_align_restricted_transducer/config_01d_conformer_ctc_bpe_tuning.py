import copy
import os
from pathlib import Path
from typing import List

import i6_core.rasr as rasr
from i6_models.parts.conformer.norm import LayerNormNC
import optuna
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
from i6_experiments.users.berger.systems.optuna_returnn_native_system import OptunaReturnnNativeSystem
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
    RasrCompatibleLogMelFeatureExtractionV1,
    RasrCompatibleLogMelFeatureExtractionV1Config,
)
from sisyphus import gs, tk
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

bpe_size = 128
target_size = 185
num_subepochs = 500
sub_checkpoints = [100, 200, 300, 400, 420, 440, 480, 490, 500]

storage_path = Path(__file__).parent.parent / "optuna_studies" / "storage.db"
storage = f"sqlite:///{storage_path.as_posix()}"

tools = copy.deepcopy(default_tools_v2)
assert tools.returnn_root is not None
normal_returnn = tools.returnn_root
tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


# ********** Return Config generators **********


def tune_specaugment(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    model_config.specaugment.cfg.time_max_mask_per_n_frames = trial.suggest_int(
        "time_max_mask_per_n_frames", 15, 40, step=5
    )
    model_config.specaugment.cfg.time_mask_max_size = trial.suggest_int("time_mask_max_size", 10, 30, step=5)

    freq_max_num_masks = trial.suggest_categorical("freq_max_num_masks", [4, 5, 8])
    model_config.specaugment.cfg.freq_max_num_masks = freq_max_num_masks
    model_config.specaugment.cfg.freq_mask_max_size = 80 // freq_max_num_masks

    return {}


def tune_oclr_schedule(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    peak_lr = trial.suggest_float("peak_lr", 1e-04, 1e-03, log=True)
    initial_lr = peak_lr / 10
    return {"decayed_lr": initial_lr, "peak_lr": peak_lr}


def tune_oclr_schedule_v2(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    peak_lr = trial.suggest_float("peak_lr", 2e-04, 6e-04, log=True)
    decayed_lr = peak_lr / 10
    return {"decayed_lr": decayed_lr, "peak_lr": peak_lr}


def tune_dropout(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    model_config.dropout = dropout
    model_config.conformer.cfg.block_cfg.ff_cfg.dropout = dropout
    model_config.conformer.cfg.block_cfg.mhsa_cfg.dropout = dropout
    model_config.conformer.cfg.block_cfg.mhsa_cfg.att_weights_dropout = dropout
    model_config.conformer.cfg.block_cfg.conv_cfg.dropout = dropout

    return {}


def tune_grad_clip(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"grad_clip": trial.suggest_categorical("grad_clip", [1.0, 100.0])}


def tune_weight_decay(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"weight_decay": trial.suggest_float("weight_decay", 1e-04, 5e-02, log=True)}


def tune_weight_decay_v2(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"weight_decay": trial.suggest_float("weight_decay", 5e-03, 3e-02, log=True)}


def tune_batch_size(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"batch_size": trial.suggest_int("batch_frames", 10000, 40000, step=2000) * 160}


def tuned_specaugment(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    model_config.specaugment.cfg.time_max_mask_per_n_frames = 30
    model_config.specaugment.cfg.time_mask_max_size = 20
    model_config.specaugment.cfg.freq_max_num_masks = 5
    model_config.specaugment.cfg.freq_mask_max_size = 16

    return {}


def tuned_oclr_schedule(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"decayed_lr": 2.6e-05, "peak_lr": 2.6e-04}


def tuned_dropout(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    model_config.dropout = 0.02
    model_config.conformer.cfg.block_cfg.ff_cfg.dropout = 0.02
    model_config.conformer.cfg.block_cfg.mhsa_cfg.dropout = 0.02
    model_config.conformer.cfg.block_cfg.mhsa_cfg.att_weights_dropout = 0.02
    model_config.conformer.cfg.block_cfg.conv_cfg.dropout = 0.02

    return {}


def tuned_grad_clip(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"grad_clip": 100.0}


def tuned_weight_decay(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"weight_decay": 0.025}


def tuned_batch_size(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {"batch_size": 22000}


def increased_epochs(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    return {
        "num_epochs": 1000,
        "inc_epochs": 480,
        "keep": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }


tuning_functions = {
    "specaugment": tune_specaugment,
    "oclr_schedule": tune_oclr_schedule,
    "oclr_schedule_v2": tune_oclr_schedule_v2,
    "dropout": tune_dropout,
    "grad_clip": tune_grad_clip,
    "weight_decay": tune_weight_decay,
    "weight_decay_v2": tune_weight_decay_v2,
    "batch_size": tune_batch_size,
    "tuned_specaugment": tuned_specaugment,
    "tuned_oclr_schedule": tuned_oclr_schedule,
    "tuned_dropout": tuned_dropout,
    "tuned_grad_clip": tuned_grad_clip,
    "tuned_weight_decay": tuned_weight_decay,
    "tuned_batch_size": tuned_batch_size,
    "increased_epochs": increased_epochs,
}


def returnn_config_generator(
    trial: optuna.Trial,
    tuning_names: List[str],
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
            alpha=0.97,
        ),
    )

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

    tuning_kwargs = {}
    for tuning_name in tuning_names:
        tuning_kwargs.update(tuning_functions[tuning_name](trial, model_config))

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
            "beam_size": 128,
            "beam_threshold": 14.0,
            "silence_token": "<blank>",
        }

    kwargs = {
        "num_epochs": num_subepochs,
        "target": "classes",
        "python_prolog": [
            "import sys",
            "sys.path.insert(0, '/u/berger/asr-exps/librispeech/20240612_align_restricted_transducer/recipe')",
            "sys.path.insert(0, '/work/asr4/berger/repositories/rasr_versions/master/lib/linux-x86_64-standard')",
            Import("i6_experiments.users.berger.corpus.general.speed_perturbation.legacy_speed_perturbation"),
        ],
        "extra_python": [
            conformer_ctc.get_serializer(model_config, variant=variant, **serializer_kwargs),
        ],
        "extern_data_config": False,
        "backend": Backend.PYTORCH,
        "use_lovely_tensors": False,
        "grad_noise": None,
        "grad_clip": 1.0,
        "optimizer": Optimizers.AdamW,
        "weight_decay": 0.01,
        "schedule": LearningRateSchedules.OCLR_V2,
        "keep_last_n": 1,
        "keep_best_n": 0,
        "keep": sub_checkpoints,
        "inc_epochs": 240,
        "initial_lr": 7e-06,
        "peak_lr": 5e-04,
        "decayed_lr": 5e-05,
        "final_lr": 1e-07,
        "batch_size": 36000 * 160,
        "use_chunking": False,
        "extra_config": extra_config,
        "use_base_config": False,
    }
    kwargs.update(tuning_kwargs)
    return get_returnn_config(**kwargs)


def get_returnn_config_collection(
    tuning_names: List[str],
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfigs[OptunaReturnnConfig]:
    generator_kwargs = {
        "tuning_names": tuning_names,
        "train_data_config": train_data_config,
        "dev_data_config": dev_data_config,
    }
    return ReturnnConfigs(
        train_config=OptunaReturnnConfig(
            returnn_config_generator, {"variant": ConfigVariant.TRAIN, **generator_kwargs}
        ),
        prior_config=OptunaReturnnConfig(
            returnn_config_generator, {"variant": ConfigVariant.PRIOR, **generator_kwargs}
        ),
        recog_configs={
            "recog": OptunaReturnnConfig(
                returnn_config_generator, {"variant": ConfigVariant.RECOG, **generator_kwargs}
            ),
        },
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

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        gpu_mem_rqmt=24,
        study_storage=storage,
        backend=Backend.PYTORCH,
    )
    recog_args = exp_args.get_ctc_flashlight_bpe_recog_step_args(
        epochs=sub_checkpoints,
        prior_scales=[0.3],
        lm_scales=[2.0],
        ogg_dataset=True,
    )

    # ********** System **********

    system = OptunaReturnnNativeSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.RECOG_NAME,
            SummaryKey.CORPUS,
            SummaryKey.TRIAL,
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
    data.train_data_config["datasets"]["data"]["audio"]["pre_process"] = CodeWrapper("legacy_speed_perturbation")

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_bpe-128_tune",
        get_returnn_config_collection(
            tuning_names=["specaugment", "oclr_schedule", "dropout", "grad_clip", "weight_decay", "batch_size"],
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
        ),
    )

    system.run_train_step(**train_args, num_trials=15, num_parallel=3)
    system.run_recog_step_for_corpora(corpora=["dev-other_4gram"], **recog_args, trial_nums=list(range(15)))

    system.cleanup_experiments()

    system.add_experiment_configs(
        "Conformer_CTC_bpe-128_tune-v2",
        get_returnn_config_collection(
            tuning_names=[
                "increased_epochs",
                "tuned_specaugment",
                "oclr_schedule_v2",
                "dropout",
                "tuned_grad_clip",
                "weight_decay_v2",
                "tuned_batch_size",
            ],
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
        ),
    )
    train_args["num_epochs"] = 1000
    recog_args["epochs"] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    system.run_train_step(**train_args, num_trials=10, num_parallel=5)
    system.run_recog_step_for_corpora(corpora=["dev-other_4gram"], **recog_args, trial_nums=list(range(10)))

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
