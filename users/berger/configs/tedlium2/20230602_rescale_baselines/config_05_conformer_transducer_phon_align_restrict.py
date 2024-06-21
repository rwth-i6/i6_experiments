import copy
import os
from typing import Dict, List, Tuple

import i6_core.rasr as rasr
from i6_core.returnn import PtCheckpoint
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
from i6_experiments.users.berger.systems.returnn_seq2seq_system import ReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_models.config import ModuleFactoryV1
from sisyphus import gs, tk

from .config_01_conformer_ctc import py as py_ctc

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def returnn_config_generator(
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfig:
    model_config = model.get_default_config_v2(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seqs": 60,
        "max_seq_length": {"audio_features": 560000},
        "torch_amp": {"dtype": "bfloat16"},
    }
    serializer = model.get_align_restrict_train_serializer(
        model_config,
        max_distance_from_alignment=kwargs.get("max_shift", 0),
        enc_loss_scales={5: 0.2, 11: 0.5},
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
        initial_lr=8e-05,
        peak_lr=kwargs.get("peak_lr", 8e-04),
        decayed_lr=1e-05,
        final_lr=1e-07,
        batch_size=10000 * 160,
        accum_grad=3,
        use_chunking=False,
        extra_config=extra_config,
    )


def recog_returnn_configs_generator(
    ilm_scale: float = 0.0,
    **kwargs,
) -> EncDecConfig[ReturnnConfig]:
    model_config = model.get_default_config_v2(num_outputs=num_outputs)
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
                "dim": 768,
                "dtype": "float32",
            },
        },
    }
    dec_extra_config = {
        "extern_data": {
            "source_encodings": {
                "dim": 768,
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
    enc_serializer = model.get_encoder_recog_serializer(model_config, **kwargs)
    dec_serializer = model.get_decoder_recog_serializer(model_config, **kwargs)

    return EncDecConfig(
        encoder_config=get_returnn_config(
            num_inputs=80,
            num_outputs=num_outputs,
            target=None,
            extra_python=[enc_serializer],
            extern_data_config=False,
            backend=Backend.PYTORCH,
            extra_config=enc_extra_config,
        ),
        decoder_config=get_returnn_config(
            num_inputs=1,
            num_outputs=num_outputs,
            target=None,
            extra_python=[dec_serializer],
            extern_data_config=False,
            backend=Backend.PYTORCH,
            extra_config=dec_extra_config,
        ),
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    ilm_scales: List[float] = [0.0, 0.2],
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            **kwargs,
        ),
        recog_configs={
            f"recog_ilm-{ilm_scale}": recog_returnn_configs_generator(ilm_scale=ilm_scale, **kwargs)
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

    train_args = exp_args.get_transducer_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes=num_outputs,
        epochs=[20, 40, 80, 160, 320, num_subepochs],
        lm_scales=[0.7],
        label_scorer_type="onnx-ffnn-transducer",
        label_scorer_args={"extra_args": {"start_label_index": 0}},
        search_parameters={"blank-label-penalty": 1.0},
        reduction_factor=4,
        reduction_subtrahend=3,
        feature_type=FeatureType.LOGMEL_16K,
        seq2seq_v2=True,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(
        tool_paths=tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.RECOG_NAME,
            SummaryKey.CORPUS,
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
        am_args=exp_args.transducer_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    for max_shift in [0, 1, 5, 10, 10000]:
        system.add_experiment_configs(
            f"Conformer_Transducer_Viterbi_shift-{max_shift}",
            get_returnn_config_collection(
                data.train_data_config,
                data.cv_data_config,
                max_shift=max_shift,
                ilm_scales=[0.2],
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
