import copy
import os
from typing import Any, Dict, Tuple
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.sms_wsj.data import get_data_inputs
from i6_experiments.users.berger.systems.transducer_system import (
    TransducerSystem,
)
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
    make_blstm_ctc_recog_model,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import (
    get_returnn_rasr_data_inputs,
)
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

speechsource_train_key = "sms_train_si284_speechsource"
speechsource_dev_key = "sms_cv_dev93_speechsource"

f_name = "gt"

frequency = 8

num_inputs = 40
num_classes = 87


def run_exp(**kwargs) -> Tuple[Dict[str, Any], SummaryReport]:
    am_args = {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        "phon_history_length": 0,
        "phon_future_length": 0,
        "tdp_scale": 1.0,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
    }

    # ********** Data inputs **********
    (train_data_inputs, dev_data_inputs, test_data_inputs, align_data_inputs,) = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        align_keys=[train_key, dev_key, speechsource_train_key, speechsource_dev_key]
        if kwargs.get("align", False)
        else [],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=True,
    )

    nn_data_inputs = get_returnn_rasr_data_inputs(
        train_data_inputs=train_data_inputs,
        cv_data_inputs=dev_data_inputs,
        dev_data_inputs=dev_data_inputs,
        test_data_inputs=test_data_inputs,
        align_data_inputs=align_data_inputs,
        am_args=am_args,
    )

    # ********** Neural Networks **********

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))

    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_model(
        num_outputs=num_classes,
        specaug_args={
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 4,
            "max_feature": 5,
        },
        blstm_args={
            "max_pool": max_pool,
            "num_layers": 6,
            "size": 400,
            "dropout": kwargs.get("dropout", 0.1),
            "l2": kwargs.get("l2", 5e-06),
        },
        mlp_args={
            "num_layers": kwargs.get("num_mlp_layers", 0),
            "size": 600,
            "dropout": kwargs.get("dropout", 0.1),
            "l2": kwargs.get("l2", 5e-06),
        },
    )
    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_blstm_ctc_recog_model(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "num_layers": 6,
            "size": 400,
        },
        mlp_args={
            "num_layers": kwargs.get("num_mlp_layers", 0),
            "size": 600,
        },
    )
    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_subepochs", 240)

    nn_args = get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        search_type=SearchTypes.LabelSyncSearch,
        returnn_train_config_args={
            "extra_python": train_python_code,
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "grad_clip": kwargs.get("grad_clip", 100.0),
            "batch_size": kwargs.get("batch_size", 15000),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-05,
            "n_steps_per_epoch": 1100,
            "use_chunking": False,
        },
        returnn_recog_config_args={"extra_python": recog_python_code},
        train_args={
            "partition_epochs": 3,
            "log_verbosity": 4,
            "use_rasr_ctc_loss": True,
            "rasr_ctc_loss_args": {"allow_label_loop": True},
        },
        prior_args={
            "num_classes": num_classes,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "lm_scales": kwargs.get("lm_scales", [1.4]),
            "prior_scales": kwargs.get("prior_scales", [0.8]),
            "use_gpu": False,
            "label_unit": "phoneme",
            "add_eow": True,
            "allow_blank": True,
            "allow_loop": True,
            "label_scorer_type": "precomputed-log-posterior",
            "lp": 15.0,
            "label_scorer_args": {
                "use_prior": True,
                "num_classes": num_classes,
                "extra_args": {
                    "blank_label_index": 0,
                    "reduction_factors": red_fact,
                },
            },
            "label_tree_args": {
                "use_transition_penalty": False,
                "skip_silence": True,
            },
        },
        align_args={
            "prior_scales": kwargs.get("prior_scales", [0.8]),
            "label_unit": "phoneme",
            "use_gpu": False,
            "add_eow": True,
            "allow_blank": True,
            "allow_loop": True,
            "label_scorer_type": "precomputed-log-posterior",
            "label_scorer_args": {
                "use_prior": True,
                "num_classes": num_classes,
                "extra_args": {
                    "blank_label_index": 0,
                    "reduction_factors": red_fact,
                },
            },
        },
    )

    # ********** Transducer System **********

    system = TransducerSystem(rasr_binary_path=rasr_binary_path)
    init_args = get_init_args(sample_rate_kHz=frequency)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        align_data=nn_data_inputs["align"],
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("extract", {"feature_key": f_name, **init_args.feature_extraction_args})
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)
    nn_steps.add_step("realign", nn_args)

    system.run(nn_steps)

    return system.alignments, system.get_summary_report()


def py() -> Tuple[Dict[str, Any], SummaryReport]:
    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
    summary_report = SummaryReport()

    # Newbob
    for lr in [3e-03]:
        name_suffix = f"newbob_lr-{lr}"
        alignments, summary = run_exp(
            name_suffix=name_suffix,
            schedule=LearningRateSchedules.Newbob,
            learning_rate=lr,
            lm_scales=[1.4],
            prior_scales=[0.6],
            align=True,
        )
        summary_report.merge_report(summary, update_structure=True)

    # OCLR
    for peak_lr in [8e-04]:
        name_suffix = f"oclr_lr-{peak_lr}"
        _, summary = run_exp(
            name_suffix=name_suffix,
            schedule=LearningRateSchedules.OCLR,
            peak_lr=peak_lr,
            lm_scales=[1.3],
            prior_scales=[0.6],
        )
        summary_report.merge_report(summary)

    tk.register_report(
        f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report",
        summary_report,
    )
    return alignments, summary_report
