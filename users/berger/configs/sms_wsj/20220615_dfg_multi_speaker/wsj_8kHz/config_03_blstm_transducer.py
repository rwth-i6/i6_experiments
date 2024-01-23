import copy
from functools import partial
import os
from typing import Any, Dict, Optional
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.context_1_transducer import (
    get_viterbi_transducer_alignment_config,
    make_context_1_blstm_transducer_blank,
    make_context_1_blstm_transducer_recog,
    pretrain_construction_algo,
)
from i6_experiments.users.berger.corpus.sms_wsj.data import get_data_inputs
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_inputs
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_core.returnn.config import CodeWrapper

from .config_02_blstm_ctc import py as run_ctc


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

frequency = 8

f_name = "gt"

num_inputs = 40
num_classes = 87


def run_exp(alignments: Dict[str, Any], **kwargs) -> SummaryReport:
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
        alignments=alignments,
        am_args=am_args,
    )

    # ********** Neural Networks **********

    name = "_".join(filter(None, ["BLSTM_Transducer", kwargs.get("name_suffix", "")]))

    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for p in max_pool:
        red_fact *= p

    train_networks = {}
    recog_networks = {}

    l2 = kwargs.get("l2", 5e-06)
    dropout = kwargs.get("dropout", 0.1)
    train_blstm_net, train_python_code = make_context_1_blstm_transducer_blank(
        num_outputs=num_classes,
        encoder_loss=kwargs.get("encoder_loss", True),
        loss_boost_scale=kwargs.get("loss_boost_scale", 5.0),
        specaug_args={
            "max_time_num": kwargs.get("time_num", 1),
            "max_time": kwargs.get("max_time", 15),
            "max_feature_num": 4,
            "max_feature": 5,
        },
        blstm_args={
            "max_pool": max_pool,
            "l2": l2,
            "dropout": dropout,
            "size": 400,
        },
        decoder_args={
            "combination_mode": kwargs.get("combination_mode", "concat"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 2),
                "size": 640,
                "l2": l2,
                "dropout": dropout,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 1),
                "size": 1024,
                "l2": l2,
                "dropout": dropout,
            },
        },
        output_args={"label_smoothing": 0.2},
    )

    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_context_1_blstm_transducer_recog(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "size": 400,
        },
        decoder_args={
            "combination_mode": kwargs.get("combination_mode", "concat"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 2),
                "size": 640,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 1),
                "size": 1024,
            },
        },
    )

    recog_networks[name] = recog_blstm_net

    alignment_config = get_viterbi_transducer_alignment_config(red_fact)

    num_subepochs = kwargs.get("num_subepochs", 180)

    nn_args = get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        search_type=SearchTypes.LabelSyncSearch,
        returnn_train_config_args={
            "extra_python": train_python_code,
            "batch_size": kwargs.get("batch_size", 15000),
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "grad_clip": kwargs.get("grad_clip", 20.0),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-06,
            "n_steps_per_epoch": 1410,
            "base_chunk_size": 256,
            "chunking_factors": {"data": 1, "classes": red_fact},
            "extra_config": {
                "pretrain": {
                    "repetitions": 6,
                    "construction_algo": CodeWrapper("pretrain_construction_algo"),
                }
                if kwargs.get("pretrain", False)
                else None,
                "train": {"reduce_target_factor": red_fact},
                "dev": {"reduce_target_factor": red_fact},
            },
            "python_prolog": [pretrain_construction_algo] if kwargs.get("pretrain", False) else [],
        },
        returnn_recog_config_args={
            "extra_python": recog_python_code,
        },
        train_args={"partition_epochs": 3, "extra_rasr_config": alignment_config},
        prior_args={
            "num_classes": num_classes,
            "use_python_control": False,
            "extra_rasr_config": alignment_config,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "lm_scales": kwargs.get("lm_scales", [0.6]),
            "prior_scales": kwargs.get("prior_scales", [0.0]),
            "use_gpu": True,
            "label_unit": "phoneme",
            "add_eow": True,
            "allow_blank": True,
            "allow_loop": False,
            "blank_penalty": 0.0,
            "recombination_limit": 1,
            "label_scorer_type": "tf-ffnn-transducer",
            "lp": 15.0,
            "label_scorer_args": {
                "use_prior": False,
                "num_classes": num_classes,
                "extra_args": {
                    "blank_label_index": 0,
                    "context_size": 1,
                    "max_batch_size": 256,
                    "reduction_factors": red_fact,
                    "use_start_label": True,
                    "start_label_index": num_classes,
                    "transform_output_negate": True,
                },
            },
            "label_tree_args": {
                "use_transition_penalty": False,
                "skip_silence": True,
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
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("extract", {"feature_key": f_name, **init_args.feature_extraction_args})
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)

    system.run(nn_steps)

    return system.get_summary_report()


def py(alignments: Optional[Dict[str, Any]] = None) -> SummaryReport:
    if alignments is None:
        alignments, _ = run_ctc()

    run_exp_partial = partial(run_exp, alignments)

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    # Specaug settings
    for time_num, max_time in [(1, 15), (3, 10), (2, 15), (3, 15)]:
        name_suffix = f"oclr-1e-03_pool-4_specaug-tn-{time_num}_mt-{max_time}"
        summary_report.merge_report(
            run_exp_partial(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=1e-03,
                lm_scales=[0.6],
                time_num=time_num,
                max_time=max_time,
            ),
            update_structure=True,
        )

    return summary_report
