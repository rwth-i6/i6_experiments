import copy
import os
from typing import Any, Dict, Optional
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_data_dumped_labels
from i6_experiments.users.berger.pytorch.models import conformer_ctc
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_native_system import ReturnnNativeSystem
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.systems.functors.recognition.returnn_search import LexiconType, LmType, VocabType

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 250

tools = copy.deepcopy(default_tools_v2)

# ********** Return Config generators **********


def returnn_config_generator(
    *,
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
    forward_data_config: Optional[dict] = None,
    **kwargs,
) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v3(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }

    if variant == ConfigVariant.TRAIN:
        extra_config["max_seq_length"] = {"audio_features": 560000}

    if variant == ConfigVariant.PRIOR:
        extra_config = {"forward_data": train_data_config}

    if variant == ConfigVariant.RECOG:
        assert forward_data_config is not None
        extra_config = {
            "forward_data": forward_data_config,
            "model_outputs": {
                "tokens": {
                    "dtype": "string",
                    "feature_dim_axis": None,
                }
            },
        }

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes" if variant != ConfigVariant.RECOG else None,
        extra_python=[
            conformer_ctc.get_serializer(
                model_config, variant=variant, recog_type=conformer_ctc.RecogType.FLASHLIGHT, **kwargs
            )
        ],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        initial_lr=7e-06,
        peak_lr=7e-04,
        decayed_lr=7e-05,
        final_lr=1e-08,
        batch_size=360 * 16000,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    *, recog_variations: Optional[Dict[str, Any]] = None, forward_data_configs: dict, **kwargs
) -> ReturnnConfigs[ReturnnConfig]:
    if recog_variations is None:
        return ReturnnConfigs(
            train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **kwargs),
            prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **kwargs),
            recog_configs={
                f"recog_{key}": returnn_config_generator(
                    variant=ConfigVariant.RECOG, forward_data_config=forward_data_config, **kwargs
                )
                for key, forward_data_config in forward_data_configs.items()
            },
        )
    else:
        return ReturnnConfigs(
            train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **kwargs),
            prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **kwargs),
            recog_configs={
                f"recog_{key}_{variation_name}": returnn_config_generator(
                    variant=ConfigVariant.RECOG, forward_data_config=forward_data_config, **variation_kwargs, **kwargs
                )
                for key, forward_data_config in forward_data_configs.items()
                for variation_name, variation_kwargs in recog_variations.items()
            },
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

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = {
        "epochs": [num_subepochs],
        "lm_scales": [2.0],
        "prior_scales": [0.5],
        "lexicon_type": LexiconType.FLASHLIGHT,
        "vocab_type": VocabType.RETURNN,
        "lm_type": LmType.ARPA_FILE,
    }
    # ********** System **********

    system = ReturnnNativeSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_logmel",
        get_returnn_config_collection(
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            forward_data_configs=data.forward_data_config,
            beam_size=64,
            beam_threshold=14.0,
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
