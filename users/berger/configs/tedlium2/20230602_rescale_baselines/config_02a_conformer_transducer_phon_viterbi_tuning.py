import copy
from enum import Enum, auto
import torch
import os
from pathlib import Path
from typing import Callable, Dict, List

import optuna

import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.viterbi_transducer_data import get_tedlium2_data
from i6_experiments.users.berger.pytorch.custom_parts.identity import IdentityConfig, IdentityModule
from i6_experiments.users.berger.pytorch.models import conformer_transducer_v2 as model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    AlignmentData,
    EncDecConfig,
    FeatureType,
    ReturnnConfigs,
    SummaryKey,
)
from i6_experiments.users.berger.util import default_tools_v2
from i6_models.config import ModuleFactoryV1
from sisyphus import gs, tk
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig
from i6_experiments.users.berger.systems.optuna_returnn_seq2seq_system import OptunaReturnnSeq2SeqSystem
from i6_experiments.users.berger.recipe.returnn.hdf import MatchLengthsJob
from i6_experiments.users.berger.pytorch.custom_parts.vgg_frontend import (
    VGG4LayerActFrontendCeilPoolV1,
    VGG4LayerActFrontendCeilPoolV1Config,
)

from .config_01_conformer_ctc import py as py_ctc

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

storage_path = Path(__file__).parent.parent / "optuna_studies" / "storage.db"
storage = f"sqlite:///{storage_path.as_posix()}"

num_outputs = 79
num_subepochs = 300
sub_checkpoints = [50, 100, 150, 200, 220, 240, 260, 280, 290, 300]

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def subsample_by_4_ceil(x: int) -> int:
    return -(-x // 4)


def tune_specaugment(trial: optuna.Trial, model_config: model.FFNNTransducerConfig) -> dict:
    model_config.transcriber_cfg.specaugment.cfg.time_max_mask_per_n_frames = trial.suggest_int(
        "time_max_mask_per_n_frames", 20, 40, step=5
    )
    model_config.transcriber_cfg.specaugment.cfg.time_max_mask_size = trial.suggest_int(
        "time_max_mask_size", 10, 30, step=5
    )
    freq_max_num_masks = trial.suggest_categorical("freq_max_num_masks", [8, 10, 16, 20])
    model_config.transcriber_cfg.specaugment.cfg.freq_max_num_masks = freq_max_num_masks
    model_config.transcriber_cfg.specaugment.cfg.freq_mask_max_size = 80 // freq_max_num_masks

    return {}


def tune_model(trial: optuna.Trial, model_config: model.FFNNTransducerConfig) -> dict:
    num_att_heads = trial.suggest_int("att_heads", 6, 8, step=2)
    dim_per_head = trial.suggest_int("dim_per_head", 64, 96, step=32)

    total_dim = num_att_heads * dim_per_head

    model_config.transcriber_cfg.encoder.cfg.frontend.cfg.out_features = total_dim
    model_config.transcriber_cfg.layer_size = total_dim
    model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.input_dim = total_dim
    model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.hidden_dim = 4 * total_dim
    model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.input_dim = total_dim
    model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.num_att_heads = num_att_heads
    model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.channels = total_dim
    model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.norm = torch.nn.BatchNorm1d(
        num_features=total_dim, affine=False
    )

    model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.kernel_size = trial.suggest_categorical(
        "conv_kernel_size", [7, 15, 31]
    )

    model_config.predictor_cfg.layers = trial.suggest_int("predictor_layers", 1, 2)

    join_combination_mode = trial.suggest_categorical("joiner_combination", ["add", "concat"])
    if join_combination_mode == "add":
        predictor_layer_size = total_dim
        model_config.joiner_cfg.combination_mode = model.CombinationMode.SUM
        model_config.joiner_cfg.input_size = total_dim
    else:
        predictor_layer_size = trial.suggest_int("predictor_layer_size", 384, 640, step=128)
        model_config.joiner_cfg.combination_mode = model.CombinationMode.CONCAT
        model_config.joiner_cfg.input_size = total_dim + predictor_layer_size

    model_config.predictor_cfg.layer_size = predictor_layer_size

    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])

    model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.dropout = dropout
    model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.dropout = dropout
    model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.att_weights_dropout = dropout
    model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.dropout = dropout

    layer_order = trial.suggest_categorical("layer_order", ["conv_first", "mhsa_first"])
    if layer_order == "conv_first":
        model_config.transcriber_cfg.encoder.cfg.block_cfg.modules = ["ff", "conv", "mhsa", "ff"]
    else:
        model_config.transcriber_cfg.encoder.cfg.block_cfg.modules = ["ff", "mhsa", "conv", "ff"]

    return {}


def tune_model_broad(trial: optuna.Trial, model_config: model.FFNNTransducerConfig) -> dict:
    size = trial.suggest_categorical("size", ["small", "medium", "large"])

    def build_model(
        att_heads: int, dim_per_head: int, conv_kernel_size: int, predictor_layers: int, predictor_layer_size: int
    ) -> None:
        total_dim = att_heads * dim_per_head
        model_config.transcriber_cfg.encoder.cfg.frontend = ModuleFactoryV1(
            VGG4LayerActFrontendCeilPoolV1,
            VGG4LayerActFrontendCeilPoolV1Config(
                in_features=80,
                conv1_channels=32,
                conv2_channels=64,
                conv3_channels=32,
                conv4_channels=32,
                conv_kernel_size=(3, 3),
                conv_padding=None,
                pool1_kernel_size=(2, 1),
                pool1_stride=(2, 1),
                pool1_padding=None,
                pool2_kernel_size=(2, 1),
                pool2_stride=(2, 1),
                pool2_padding=None,
                activation=torch.nn.ReLU(),
                out_features=total_dim,
            ),
        )

        model_config.transcriber_cfg.encoder.cfg.frontend.cfg.out_features = total_dim
        model_config.transcriber_cfg.layer_size = total_dim
        model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.input_dim = total_dim
        model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.hidden_dim = 4 * total_dim
        model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.input_dim = total_dim
        model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.num_att_heads = att_heads
        model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.channels = total_dim
        model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.norm = torch.nn.BatchNorm1d(
            num_features=total_dim, affine=False
        )

        model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.kernel_size = conv_kernel_size

        model_config.predictor_cfg.layers = predictor_layers

        model_config.joiner_cfg.combination_mode = model.CombinationMode.CONCAT
        model_config.joiner_cfg.input_size = total_dim + predictor_layer_size

        model_config.predictor_cfg.layer_size = predictor_layer_size

        model_config.transcriber_cfg.encoder.cfg.block_cfg.ff_cfg.dropout = 0.3
        model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.dropout = 0.3
        model_config.transcriber_cfg.encoder.cfg.block_cfg.mhsa_cfg.att_weights_dropout = 0.3
        model_config.transcriber_cfg.encoder.cfg.block_cfg.conv_cfg.dropout = 0.3

        model_config.transcriber_cfg.encoder.cfg.block_cfg.modules = ["ff", "conv", "mhsa", "ff"]

    if size == "small":
        build_model(
            att_heads=6,
            dim_per_head=64,
            conv_kernel_size=7,
            predictor_layers=2,
            predictor_layer_size=384,
        )
    elif size == "medium":
        build_model(
            att_heads=8,
            dim_per_head=64,
            conv_kernel_size=7,
            predictor_layers=2,
            predictor_layer_size=384,
        )
    elif size == "large":
        build_model(
            att_heads=8,
            dim_per_head=96,
            conv_kernel_size=7,
            predictor_layers=2,
            predictor_layer_size=384,
        )

    return {}


def tune_learn_schedule(trial: optuna.Trial, _: model.FFNNTransducerConfig) -> dict:
    batch_size = trial.suggest_int("batch_size", 10000, 30000, step=5000)
    peak_lr = trial.suggest_float("peak_lr", 3e-04, 1e-03)
    initial_lr = peak_lr / 10
    return {"initial_lr": initial_lr, "peak_lr": peak_lr, "batch_size": batch_size * 160}


class TuningOption(Enum):
    SPECAUGMENT = auto()
    MODEL = auto()
    MODEL_BROAD = auto()
    LEARN_SCHEDULE = auto()


def map_tuning_option(tuning_option: TuningOption) -> Callable[[optuna.Trial, model.FFNNTransducerConfig], dict]:
    if tuning_option == TuningOption.SPECAUGMENT:
        return tune_specaugment
    if tuning_option == TuningOption.MODEL:
        return tune_model
    if tuning_option == TuningOption.MODEL_BROAD:
        return tune_model_broad
    if tuning_option == TuningOption.LEARN_SCHEDULE:
        return tune_learn_schedule


def returnn_config_generator(
    trial: optuna.Trial,
    tuning_options: List[TuningOption],
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfig:
    model_config = model.get_default_config_v1(num_outputs=num_outputs)
    model_config.transcriber_cfg.feature_extraction = ModuleFactoryV1(
        IdentityModule,
        IdentityConfig(),
    )

    tuning_kwargs = {}
    for tuning_option in tuning_options:
        tuning_kwargs.update(map_tuning_option(tuning_option)(trial, model_config))

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seq_length": {"audio_features": 560000},
        "torch_amp": {"dtype": "bfloat16"},
        "chunking": (
            {
                "data": 256,
                "classes": 64,
            },
            {
                "data": 128,
                "classes": 32,
            },
        ),
    }
    serializer = model.get_viterbi_train_serializer(model_config, enc_loss_scales={5: 0.3, 11: 0.7})

    kwargs = {
        "num_epochs": num_subepochs,
        "num_inputs": 80,
        "num_outputs": num_outputs,
        "target": "classes",
        "extra_python": [serializer],
        "extern_data_config": True,
        "backend": Backend.PYTORCH,
        "grad_noise": 0.0,
        "grad_clip": 0.0,
        "keep_last_n": 1,
        "keep_best_n": 0,
        "keep": sub_checkpoints,
        "optimizer": Optimizers.AdamW,
        "weight_decay": 5e-06,
        "schedule": LearningRateSchedules.OCLR,
        "initial_lr": 8e-05,
        "peak_lr": 8e-04,
        "decayed_lr": 1e-05,
        "final_lr": 1e-07,
        "batch_size": 30000 * 160,
        "use_chunking": False,
        "extra_config": extra_config,
    }
    kwargs.update(tuning_kwargs)
    return get_returnn_config(**kwargs)


def recog_enc_returnn_config_generator(
    trial: optuna.Trial,
    tuning_options: List[TuningOption],
    ilm_scale: float = 0.0,
) -> ReturnnConfig:
    model_config = model.get_default_config_v1(num_outputs=num_outputs)
    model_config.transcriber_cfg.feature_extraction = ModuleFactoryV1(
        IdentityModule,
        IdentityConfig(),
    )
    for tuning_option in tuning_options:
        map_tuning_option(tuning_option)(trial, model_config)
    if ilm_scale != 0:
        model_config = model.FFNNTransducerWithIlmConfig(
            transcriber_cfg=model_config.transcriber_cfg,
            predictor_cfg=model_config.predictor_cfg,
            joiner_cfg=model_config.joiner_cfg,
            ilm_scale=ilm_scale,
        )

    enc_extra_config = {
        "extern_data": {
            "sources": {"dim": 80, "dtype": "float32"},
        },
        "model_outputs": {
            "source_encodings": {
                "dim": model_config.transcriber_cfg.layer_size,
                "dtype": "float32",
            },
        },
    }
    enc_serializer = model.get_encoder_recog_serializer(model_config)

    return get_returnn_config(
        num_inputs=80,
        num_outputs=num_outputs,
        target=None,
        extra_python=[enc_serializer],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        extra_config=enc_extra_config,
    )


def recog_dec_returnn_config_generator(
    trial: optuna.Trial,
    tuning_options: List[TuningOption],
    ilm_scale: float = 0.0,
) -> ReturnnConfig:
    model_config = model.get_default_config_v1(num_outputs=num_outputs)
    for tuning_option in tuning_options:
        map_tuning_option(tuning_option)(trial, model_config)
    model_config.transcriber_cfg.feature_extraction = ModuleFactoryV1(
        IdentityModule,
        IdentityConfig(),
    )
    if ilm_scale != 0:
        model_config = model.FFNNTransducerWithIlmConfig(
            transcriber_cfg=model_config.transcriber_cfg,
            predictor_cfg=model_config.predictor_cfg,
            joiner_cfg=model_config.joiner_cfg,
            ilm_scale=ilm_scale,
        )

    dec_extra_config = {
        "extern_data": {
            "source_encodings": {
                "dim": model_config.transcriber_cfg.layer_size,
                "time_dim_axis": None,
                "dtype": "float32",
            },
            "history": {
                "dim": num_outputs,
                "time_dim_axis": None,
                "sparse": True,
                "shape": (1,),
                "dtype": "int32",
            },
        },
        "model_outputs": {
            "log_probs": {
                "dim": num_outputs,
                "time_dim_axis": None,
                "dtype": "float32",
            }
        },
    }
    dec_serializer = model.get_decoder_recog_serializer(model_config)

    return get_returnn_config(
        num_inputs=80,
        num_outputs=num_outputs,
        target=None,
        extra_python=[dec_serializer],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        extra_config=dec_extra_config,
    )


def get_returnn_config_collection(
    tuning_options: List[TuningOption],
    train_data_config: dict,
    dev_data_config: dict,
    ilm_scales: List[float] = [0.2],
) -> ReturnnConfigs[OptunaReturnnConfig]:
    return ReturnnConfigs(
        train_config=OptunaReturnnConfig(
            returnn_config_generator,
            {
                "train_data_config": train_data_config,
                "dev_data_config": dev_data_config,
                "tuning_options": tuning_options,
            },
        ),
        recog_configs={
            f"recog_ilm-{ilm_scale}": EncDecConfig(
                encoder_config=OptunaReturnnConfig(
                    recog_enc_returnn_config_generator, {"ilm_scale": ilm_scale, "tuning_options": tuning_options}
                ),
                decoder_config=OptunaReturnnConfig(
                    recog_dec_returnn_config_generator, {"ilm_scale": ilm_scale, "tuning_options": tuning_options}
                ),
            )
            for ilm_scale in ilm_scales
        },
    )


def run_exp(alignments: Dict[str, AlignmentData]) -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data(
        alignments=alignments,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.LOGMEL_16K,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    for data_config in [data.train_data_config, data.cv_data_config]:
        data_config["datasets"]["classes"]["files"] = [
            MatchLengthsJob(
                hdf_file,
                match_hdfs=data_config["datasets"]["data"]["files"],
                match_len_transform_func=subsample_by_4_ceil,
            ).out_hdf
            for hdf_file in data_config["datasets"]["classes"]["files"]
        ]

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=num_subepochs,
        study_storage=storage,
        num_parallel=5,
        gpu_mem_rqmt=24,
        backend=Backend.PYTORCH,
    )
    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes=num_outputs,
        epochs=sub_checkpoints,
        trial_nums=list(range(30)),
        label_scorer_type="onnx-ffnn-transducer",
        label_scorer_args={"extra_args": {"start_label_index": 0}},
        search_parameters={"blank-label-penalty": 1.0},
        reduction_subtrahend=3,
        reduction_factor=4,
        feature_type=FeatureType.LOGMEL_16K,
        seq2seq_v2=True,
        backend=Backend.PYTORCH,
    )

    # ********** System **********

    system = OptunaReturnnSeq2SeqSystem(
        tool_paths=tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.RECOG_NAME,
            SummaryKey.CORPUS,
            SummaryKey.TRIAL,
            SummaryKey.EPOCH,
            SummaryKey.LM,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
            SummaryKey.ERR,
        ],
        summary_sort_keys=[SummaryKey.ERR, SummaryKey.CORPUS],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    train_args["num_trials"] = 30
    recog_args["trial_nums"] = list(range(30))
    system.add_experiment_configs(
        "Conformer_Transducer_Viterbi_tuning",
        get_returnn_config_collection(
            tuning_options=[TuningOption.MODEL_BROAD, TuningOption.LEARN_SCHEDULE, TuningOption.SPECAUGMENT],
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
        ),
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    _, alignments = py_ctc()
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    report = run_exp(alignments)

    summary_report.merge_report(report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
