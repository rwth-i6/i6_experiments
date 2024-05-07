import copy
import os
from typing import List, Optional
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.phon_transducer_data import get_tedlium2_data_dumped_labels
from i6_experiments.users.berger.pytorch.models import conformer_transducer_v2 as model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import (
    EncDecConfig,
    FeatureType,
    ReturnnConfigs,
    SummaryKey,
)
from i6_experiments.users.berger.systems.returnn_seq2seq_system import ReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.systems.functors.recognition.returnn_search import LexiconType

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
    model_config = model.get_default_config_v1(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seq_length": {"audio_features": 560000},
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
        "model_outputs": {
            "encoder": {
                "dim": model_config.transcriber_cfg.dim,
            }
        },
    }
    dec_extra_config = {
        "model_outputs": {
            "log_probs": {
                "dim": num_outputs,
            }
        },
    }
    enc_serializer = model.get_encoder_recog_serializer(model_config, **kwargs)
    dec_serializer = model.get_decoder_recog_serializer(model_config, **kwargs)

    return EncDecConfig(
        encoder_config=get_returnn_config(
            num_inputs=1,
            num_outputs=num_outputs,
            target=None,
            extra_python=[enc_serializer],
            extern_data_config=True,
            backend=Backend.PYTORCH,
            extra_config=enc_extra_config,
        ),
        decoder_config=get_returnn_config(
            num_inputs=1,
            num_outputs=num_outputs,
            target=None,
            extra_python=[dec_serializer],
            extern_data_config=True,
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
        recog_configs={"recog": recog_returnn_configs_generator(**kwargs)},
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = {
        "epochs": [500],
        "prior_scales": [0.0],
        "lm_scales": [0.0],
    }

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

    system.add_experiment_configs(
        "Conformer_Transducer",
        get_returnn_config_collection(
            data.train_data_config,
            data.cv_data_config,
        ),
    )

    system.run_train_step(**train_args)

    system.run_dev_recog_step(
        recog_exp_names={
            exp_name: [
                recog_exp_name for recog_exp_name in system.get_recog_exp_names()[exp_name] if dev_key in recog_exp_name
            ]
            for dev_key in data.dev_keys
            for exp_name in system.get_exp_names()
        },
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
