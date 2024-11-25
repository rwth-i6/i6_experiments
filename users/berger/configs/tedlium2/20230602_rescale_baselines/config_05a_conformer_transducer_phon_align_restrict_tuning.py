import copy
import os
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List

import i6_core.rasr as rasr
import optuna
from i6_core.returnn import CodeWrapper
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.viterbi_transducer_data import get_tedlium2_data
from i6_experiments.users.berger.pytorch.custom_parts.identity import IdentityConfig, IdentityModule
from i6_experiments.users.berger.pytorch.models import conformer_transducer_v2 as model
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    AlignmentData,
    EncDecConfig,
    FeatureType,
    ReturnnConfigs,
    SummaryKey,
)
from i6_experiments.users.berger.systems.optuna_returnn_seq2seq_system import OptunaReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_models.config import ModuleFactoryV1
from sisyphus import gs, tk

from .config_01_conformer_ctc import py as py_ctc

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

storage_path = Path(__file__).parent.parent / "optuna_studies" / "storage.db"
storage = f"sqlite:///{storage_path.as_posix()}"

num_outputs = 79
num_subepochs = 500
sub_checkpoints = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def tune_constant_schedule(trial: optuna.Trial) -> CodeWrapper:
    shift = trial.suggest_int("shift", 0, 25)

    return CodeWrapper(str(shift))


def tune_linear_widening_schedule(trial: optuna.Trial) -> CodeWrapper:
    begin_shift = trial.suggest_int("begin_shift", 0, 5)
    increase_amount = trial.suggest_int("increase_amount", 1, 25)

    return CodeWrapper(f"lambda epoch: {begin_shift} + ({increase_amount} * epoch) // {num_subepochs}")


def tune_linear_narrowing_schedule(trial: optuna.Trial) -> CodeWrapper:
    final_shift = trial.suggest_int("final_shift", 0, 5)
    decrease_amount = trial.suggest_int("decrease_amount", 1, 25)

    return CodeWrapper(
        f"lambda epoch: {final_shift} + ({decrease_amount} * ({num_subepochs} - epoch) // {num_subepochs}"
    )


def tune_diamond_schedule(trial: optuna.Trial) -> CodeWrapper:
    begin_shift = trial.suggest_int("begin_shift", 0, 5)
    peak_shift = trial.suggest_int("peak_shift", 5, 25)
    final_shift = trial.suggest_int("final_shift", 0, 5)
    peak_pos = trial.suggest_int("peak_pos", 100, num_subepochs - 100)

    return CodeWrapper(
        f"lambda epoch: ({begin_shift} + ({peak_shift - begin_shift} * epoch) // {peak_pos}) if epoch <= {peak_pos} else ({final_shift} + ({peak_shift - final_shift} * ({num_subepochs} - epoch)) // {num_subepochs - peak_pos})"
    )


def tune_hourglass_schedule(trial: optuna.Trial) -> CodeWrapper:
    begin_shift = trial.suggest_int("begin_shift", 5, 25)
    valley_shift = trial.suggest_int("valley_shift", 0, 5)
    final_shift = trial.suggest_int("final_shift", 5, 25)
    valley_pos = trial.suggest_int("valley_pos", 100, num_subepochs - 100)

    return CodeWrapper(
        f"lambda epoch: ({valley_shift} + ({begin_shift - valley_shift} * ({valley_pos} - epoch) // {valley_pos})) if epoch <= {valley_pos} else ({valley_shift} + ({final_shift - valley_shift} * (epoch - {valley_pos})) // {num_subepochs - valley_pos})"
    )


class ScheduleShape(Enum):
    CONSTANT = auto()
    LINEAR_WIDENING = auto()
    LINEAR_NARROWING = auto()
    DIAMOND = auto()
    HOURGLASS = auto()


def map_schedule_shape(schedule_shape: ScheduleShape) -> Callable[[optuna.Trial], CodeWrapper]:
    if schedule_shape == ScheduleShape.CONSTANT:
        return tune_constant_schedule
    if schedule_shape == ScheduleShape.LINEAR_WIDENING:
        return tune_linear_widening_schedule
    if schedule_shape == ScheduleShape.LINEAR_NARROWING:
        return tune_linear_narrowing_schedule
    if schedule_shape == ScheduleShape.DIAMOND:
        return tune_diamond_schedule
    if schedule_shape == ScheduleShape.HOURGLASS:
        return tune_hourglass_schedule


def returnn_config_generator(
    trial: optuna.Trial,
    schedule_shape: ScheduleShape,
    train_data_config: dict,
    dev_data_config: dict,
    **_,
) -> ReturnnConfig:
    model_config = model.get_default_config_v3(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seqs": 60,
        "max_seq_length": {"audio_features": 560000},
        "torch_amp": {"dtype": "bfloat16"},
    }
    serializer = model.get_align_restrict_train_serializer(
        model_config,
        max_distance_from_alignment=map_schedule_shape(schedule_shape)(trial),
        enc_loss_scales={5: 0.3, 11: 1.0},
        blank_idx=0,
    )

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[serializer],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        weight_decay=5e-06,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=9e-05,
        peak_lr=9e-04,
        decayed_lr=1e-05,
        final_lr=1e-07,
        batch_size=10000 * 160,
        accum_grad=2,
        use_chunking=False,
        extra_config=extra_config,
    )


def recog_enc_returnn_config_generator(
    _: optuna.Trial,
    ilm_scale: float = 0.0,
    **kwargs,
) -> ReturnnConfig:
    model_config = model.get_default_config_v3(num_outputs=num_outputs)
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

    enc_extra_config = {
        "extern_data": {
            "sources": {"dim": 80, "dtype": "float32"},
        },
        "model_outputs": {
            "source_encodings": {
                "dim": 512,
                "dtype": "float32",
            },
        },
    }
    enc_serializer = model.get_encoder_recog_serializer(model_config, **kwargs)

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
    _: optuna.Trial,
    ilm_scale: float = 0.0,
    **kwargs,
) -> ReturnnConfig:
    model_config = model.get_default_config_v3(num_outputs=num_outputs)
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
                "dim": 512,
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
    dec_serializer = model.get_decoder_recog_serializer(model_config, **kwargs)

    return get_returnn_config(
        num_inputs=1,
        num_outputs=num_outputs,
        target=None,
        extra_python=[dec_serializer],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        extra_config=dec_extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    schedule_shape: ScheduleShape,
    ilm_scales: List[float] = [0.2],
    **kwargs,
) -> ReturnnConfigs[OptunaReturnnConfig]:
    return ReturnnConfigs(
        train_config=OptunaReturnnConfig(
            returnn_config_generator,
            {
                "train_data_config": train_data_config,
                "dev_data_config": dev_data_config,
                "schedule_shape": schedule_shape,
                **kwargs,
            },
        ),
        recog_configs={
            f"recog_ilm-{ilm_scale}": EncDecConfig(
                encoder_config=OptunaReturnnConfig(recog_enc_returnn_config_generator, {"ilm_scale": ilm_scale}),
                decoder_config=OptunaReturnnConfig(recog_dec_returnn_config_generator, {"ilm_scale": ilm_scale}),
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
        feature_type=FeatureType.SAMPLES,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    common_train_args = exp_args.get_transducer_train_step_args(
        num_epochs=num_subepochs, study_storage=storage, gpu_mem_rqmt=24, backend=Backend.PYTORCH
    )
    common_recog_args = exp_args.get_transducer_recog_step_args(
        num_classes=num_outputs,
        epochs=sub_checkpoints,
        lm_scales=[0.7],
        label_scorer_type="onnx-ffnn-transducer",
        label_scorer_args={"extra_args": {"start_label_index": 0}},
        search_parameters={"blank-label-penalty": 1.0},
        reduction_factor=4,
        reduction_subtrahend=3,
        feature_type=FeatureType.LOGMEL_16K,
        backend=Backend.PYTORCH,
        seq2seq_v2=True,
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

    system.add_experiment_configs(
        "Conformer_Transducer_Viterbi_const",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            schedule_shape=ScheduleShape.CONSTANT,
        ),
    )
    system.run_train_step(num_trials=8, num_parallel=1, **common_train_args)
    system.run_dev_recog_step(trial_nums=list(range(8)), **common_recog_args)

    system.cleanup_experiments()

    system.add_experiment_configs(
        "Conformer_Transducer_Viterbi_linear-widening",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            schedule_shape=ScheduleShape.LINEAR_WIDENING,
        ),
    )
    system.run_train_step(num_trials=10, num_parallel=2, **common_train_args)
    system.run_dev_recog_step(trial_nums=list(range(10)), **common_recog_args)

    system.cleanup_experiments()

    # system.add_experiment_configs(
    #     "Conformer_Transducer_Viterbi_linear-narrowing",
    #     get_returnn_config_collection(
    #         data.train_data_config,
    #         data.cv_data_config,
    #         schedule_shape=ScheduleShape.LINEAR_NARROWING,
    #     ),
    # )
    # system.run_train_step(num_trials=10, num_parallel=2, **common_train_args)
    # system.run_dev_recog_step(trial_nums=list(range(10)), **common_recog_args)
    #
    # system.cleanup_experiments()

    system.add_experiment_configs(
        "Conformer_Transducer_Viterbi_diamond",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            schedule_shape=ScheduleShape.DIAMOND,
        ),
    )
    system.run_train_step(num_trials=15, num_parallel=3, **common_train_args)
    system.run_dev_recog_step(trial_nums=list(range(15)), **common_recog_args)

    system.cleanup_experiments()

    system.add_experiment_configs(
        "Conformer_Transducer_Viterbi_hourglass",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
            schedule_shape=ScheduleShape.HOURGLASS,
        ),
    )
    system.run_train_step(num_trials=15, num_parallel=3, **common_train_args)
    system.run_dev_recog_step(trial_nums=list(range(15)), **common_recog_args)

    system.cleanup_experiments()

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
