import copy
import os
from typing import List, Optional
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.bpe_transducer_data import get_tedlium2_data_dumped_bpe_labels
from i6_experiments.users.berger.pytorch.models import conformer_transducer_v2 as model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, EncDecConfig, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import ReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.systems.functors.recognition.returnn_search import LexiconType
from i6_experiments.users.berger.systems.functors.rasr_base import RecognitionScoringType

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 1068
num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def returnn_config_generator(
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfig:
    model_config = model.get_default_config_v1(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seq_length": {"audio_features": 560000},
        "torch_amp": {"dtype": "bfloat16"},
    }
    serializer = model.get_train_serializer(model_config, **kwargs)

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
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-06,
        peak_lr=8e-05,
        decayed_lr=1e-05,
        final_lr=1e-08,
        batch_size=10000 * 160,
        use_chunking=False,
        extra_config=extra_config,
    )


def recog_returnn_configs_generator(
    **kwargs,
) -> EncDecConfig[ReturnnConfig]:
    model_config = model.get_default_config_v1(num_outputs=num_outputs)

    enc_extra_config = {
        "extern_data": {
            "sources": {"dim": 80, "dtype": "float32"},
        },
        "model_outputs": {
            "source_encodings": {
                "dim": 384,
                "dtype": "float32",
            },
        },
    }
    dec_extra_config = {
        "extern_data": {
            "source_encodings": {
                "dim": 384,
                "time_dim_axis": None,
                "dtype": "float32",
            },
            "targets": {
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
            # python_prolog=["from returnn.tensor.dim import Dim, batch_dim"],
            extra_python=[dec_serializer],
            extern_data_config=False,
            backend=Backend.PYTORCH,
            extra_config=dec_extra_config,
        ),
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            blank_id=0,
            **kwargs,
        ),
        recog_configs={
            "recog": recog_returnn_configs_generator(
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                **kwargs,
            )
        },
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data_dumped_bpe_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes=num_outputs,
        epochs=[500],
        lm_scales=[0.5],
        label_scorer_type="onnx-ffnn-transducer",
        label_scorer_args={"extra_args": {"start_label_index": 0}},
        reduction_subtrahend=3,
        reduction_factor=4,
        feature_type=FeatureType.LOGMEL_16K,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.transducer_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_Transducer",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
        ),
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
