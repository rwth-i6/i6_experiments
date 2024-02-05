import copy
import os
from typing import Dict, Tuple

import i6_core.rasr as rasr
from i6_core.recognition import Hub5ScoreJob
from i6_core.returnn import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
import i6_experiments.users.berger.network.models.fullsum_ctc as ctc_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.util import default_tools_v2 as tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.corpus.switchboard.ctc_data import (
    get_switchboard_data,
)
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 88


# ********** Return Config generators **********


def generate_returnn_config(
    train: bool,
    *,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
    am_args: dict,
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfig:
    if train:
        network_dict, extra_python = ctc_model.make_conformer_fullsum_ctc_model(
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
            },
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 1e-04,
            },
            output_args={
                "rasr_binary_path": tools.rasr_binary_path,
                "loss_corpus_path": loss_corpus,
                "loss_lexicon_path": loss_lexicon,
                "am_args": am_args,
            },
        )
    else:
        network_dict, extra_python = ctc_model.make_conformer_ctc_recog_model(
            num_outputs=num_classes,
            conformer_args={
                "num_blocks": 12,
                "size": 512,
            },
        )

    returnn_config = get_returnn_config(
        network=network_dict,
        target=None,
        num_epochs=300,
        num_inputs=40,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        extra_python=extra_python,
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=4e-04,
        final_lr=1e-05,
        batch_size=10000,
        use_chunking=False,
        extra_config={
            "train": train_data_config,
            "dev": dev_data_config,
        },
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp() -> Tuple[SummaryReport, Checkpoint, Dict[str, AlignmentData]]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_switchboard_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        feature_type=FeatureType.GAMMATONE_8K,
        test_keys=["hub5e01"],
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=300,
        gpu_mem_rqmt=11,
    )

    recog_args = exp_args.get_ctc_recog_step_args(num_classes)
    align_args = exp_args.get_ctc_align_step_args(num_classes)
    recog_args["epochs"] = [20, 40, 80, 160, 240, 300, "best"]
    recog_args["feature_type"] = FeatureType.GAMMATONE_8K
    recog_args["prior_scales"] = [0.3]
    recog_args["lm_scales"] = [0.7]
    align_args["feature_type"] = FeatureType.GAMMATONE_8K

    recog_am_args = copy.deepcopy(exp_args.ctc_recog_am_args)
    recog_am_args.update(
        {
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
        }
    )
    loss_am_args = copy.deepcopy(exp_args.ctc_loss_am_args)
    loss_am_args.update(
        {
            "state_tying": "lookup",
            "state_tying_file": tk.Path("/work/asr4/berger/dependencies/switchboard/state_tying/eow-state-tying"),
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
            "phon_history_length": 0,
            "phon_future_length": 0,
        }
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(
        scorer_type=Hub5ScoreJob,
        stm_kwargs={"non_speech_tokens": ["[NOISE]", "[LAUGHTER]", "[VOCALIZED-NOISE]"]},
        score_kwargs={"glm": tk.Path("/u/corpora/speech/hub-5-00/raw/transcriptions/reference/en20000405_hub5.glm")},
    )

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "loss_corpus": data.loss_corpus,
        "loss_lexicon": data.loss_lexicon,
        "am_args": loss_am_args,
        "dev_data_config": data.cv_data_config,
    }

    for ordering in ["laplace:.1000", "laplace:.100", "laplace:.50", "laplace:.25", "laplace:.10", "random"]:
        mod_train_data_config = copy.deepcopy(data.train_data_config)
        mod_train_data_config["seq_ordering"] = ordering

        train_config = generate_returnn_config(
            train=True, train_data_config=mod_train_data_config, **config_generator_kwargs
        )
        recog_config = generate_returnn_config(
            train=False, train_data_config=mod_train_data_config, **config_generator_kwargs
        )

        returnn_configs = ReturnnConfigs(
            train_config=train_config,
            recog_configs={"recog": recog_config},
        )

        system.add_experiment_configs(f"Conformer_CTC_order-{ordering}", returnn_configs)

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)
    alignments = next(
        iter(system.run_align_step(exp_names=["Conformer_CTC_order-laplace:.1000"], **align_args).values())
    )

    model = system.get_train_job("Conformer_CTC_order-laplace:.1000").out_checkpoints[300]
    assert isinstance(model, Checkpoint)

    assert system.summary_report
    return system.summary_report, model, alignments


def py() -> Tuple[SummaryReport, Checkpoint, Dict[str, AlignmentData]]:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report, model, alignments = run_exp()

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, model, alignments
