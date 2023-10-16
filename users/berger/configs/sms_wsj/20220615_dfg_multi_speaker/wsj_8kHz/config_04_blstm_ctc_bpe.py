import copy
from functools import partial
import os
from typing import Optional
from i6_core.returnn.training import Checkpoint
from i6_experiments.users.berger.recipe.summary.report import SummaryReport

from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedFormat

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_bpe,
    get_data_inputs,
)
from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
    make_blstm_ctc_recog_model,
)
from i6_experiments.users.berger.network.models.lstm_lm import (
    make_lstm_lm_recog_model,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.returnn.pretrain import pretrain_construction_algo
from i6_experiments.users.berger.args.jobs.data import (
    get_returnn_rasr_data_inputs,
)
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from ..lm.config_01_lstm_bpe import py as run_lm


# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

rasr_binary_path = tk.Path("/u/berger/rasr_tf2/arch/linux-x86_64-standard")

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

frequency = 8

f_name = "gt"

num_inputs = 40

bpe_size = 100


def run_exp(lm_model: tk.Path, **kwargs) -> SummaryReport:

    lm_cleaning = kwargs.get("lm_cleaning", True)

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs, _ = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=True,
        lm_cleaning=lm_cleaning,
    )

    # ********** Data inputs **********

    nn_data_inputs = get_returnn_rasr_data_inputs(
        train_data_inputs=train_data_inputs,
        cv_data_inputs=dev_data_inputs,
        dev_data_inputs=dev_data_inputs,
        test_data_inputs=test_data_inputs,
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

    bpe_job = get_bpe(size=bpe_size, lm_cleaning=lm_cleaning)

    bpe_codes = bpe_job.out_bpe_codes
    bpe_vocab = bpe_job.out_bpe_vocab

    num_classes = bpe_job.out_vocab_size  # bpe count
    num_classes_b = num_classes + 1  # bpe count + blank

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC_BPE", kwargs.get("name_suffix", "")]))
    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_model(
        num_outputs=num_classes_b,
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
            "l2": kwargs.get("l2", 5e-06),
            "dropout": kwargs.get("dropout", 0.1),
        },
        mlp_args={
            "num_layers": kwargs.get("num_mlp_layers", 0),
            "size": 600,
            "l2": kwargs.get("l2", 5e-06),
            "dropout": kwargs.get("dropout", 0.1),
        },
    )
    train_blstm_net["output"] = {
        "class": "softmax",
        "from": train_blstm_net["output"]["from"],
        "n_out": num_classes_b,
    }
    train_blstm_net["ctc_loss"] = {
        "class": "fast_bw",
        "from": "output",
        "align_target_key": "bpe",
        "align_target": "ctc",
        "input_type": "prob",
        "tdp_scale": 0.0,
        "ctc_opts": {"blank_idx": num_classes},
    }
    train_blstm_net["output_loss"] = {
        "class": "copy",
        "from": "output",
        "loss": "via_layer",
        "loss_opts": {
            "loss_wrt_to_act_in": "softmax",
            "align_layer": "ctc_loss",
        },
    }
    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_blstm_ctc_recog_model(
        num_outputs=num_classes_b,
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
    recog_blstm_net["output"].update(
        {
            "class": "linear",
            "activation": "log_softmax",
        }
    )
    output_name = "output"

    blank_penalty = kwargs.get("blank_penalty", 0.0)
    if blank_penalty != 0:
        recog_blstm_net["output_bp"] = {
            "class": "eval",
            "from": output_name,
            "eval": DelayedFormat(
                "source(0) - tf.expand_dims(tf.one_hot([{}], {}, on_value={}, dtype=tf.float32), axis=0)",
                num_classes,
                num_classes_b,
                blank_penalty,
            ),
        }
        output_name = "output_bp"

    recog_blstm_net.update(
        {
            "beam_search": {
                "class": "rec",
                "from": output_name,
                "unit": {
                    "output": {
                        "class": "choice",
                        "from": "data:source",
                        "input_type": "log_prob",
                        "target": "bpe_b",
                        "beam_size": kwargs.get("beam_size", 16),
                        "explicit_search_source": "prev:output",
                        "initial_output": num_classes,
                    },
                },
            },
            "ctc_decode": {
                "class": "subnetwork",
                "is_output_layer": True,
                "target": "bpe",
                "subnetwork": {
                    "beam_search": {
                        "class": "reduce",
                        "from": "data:source",
                        "axis": "F",
                        "mode": "argmax",
                    },
                    "decision": {
                        "class": "decide",
                        "from": "base:beam_search",
                    },
                    "decision_shifted": {
                        "class": "shift_axis",
                        "from": "decision",
                        "axis": "T",
                        "amount": 1,
                        "pad_value": -1,
                        "adjust_size_info": False,
                    },
                    "mask_unique": {
                        "class": "compare",
                        "from": ["decision", "decision_shifted"],
                        "kind": "not_equal",
                    },
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "decision",
                        "kind": "not_equal",
                        "value": num_classes,
                    },
                    "mask_label": {
                        "class": "combine",
                        "from": ["mask_unique", "mask_non_blank"],
                        "kind": "logical_and",
                    },
                    "decision_unique_labels": {
                        "class": "masked_computation",
                        "from": "decision",
                        "mask": "mask_label",
                        "unit": {"class": "copy"},
                    },
                    "output": {
                        "class": "reinterpret_data",
                        "from": "decision_unique_labels",
                        "increase_sparse_dim": -1,
                        "target": "bpe",
                        "loss": "edit_distance",
                    },
                },
            },
        }
    )
    if kwargs.get("lm_scale", 0.7):
        recog_blstm_net["beam_search"]["unit"]["output"]["from"] = "combined_scores"
        recog_blstm_net["beam_search"]["unit"].update(
            {
                "mask_non_blank": {
                    "class": "compare",
                    "from": "output",
                    "value": num_classes,
                    "kind": "not_equal",
                    "initial_output": True,
                },
                "prev_output_reinterpret": {
                    "class": "reinterpret_data",
                    "from": "prev:output",
                    "increase_sparse_dim": -1,
                },
                "lm_masked": {
                    "class": "masked_computation",
                    "from": "prev_output_reinterpret",
                    "mask": "prev:mask_non_blank",
                    "unit": {
                        "class": "subnetwork",
                        "load_on_init": lm_model,
                        "subnetwork": make_lstm_lm_recog_model(
                            num_outputs=num_classes,
                            embedding_args={
                                "size": 256,
                            },
                            lstm_args={
                                "num_layers": 2,
                                "size": 2048,
                            },
                        ),
                    },
                },
                "lm_padded": {
                    "class": "pad",
                    "from": "lm_masked",
                    "axes": "f",
                    "padding": (0, 1),
                    "value": 0,
                    "mode": "constant",
                },
                "combined_scores": {
                    "class": "eval",
                    "from": ["data:source", "lm_padded"],
                    "eval": "source(0) + %f * source(1)" % kwargs.get("lm_scale", 0.7),
                },
            }
        )

    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_epochs", 240)

    nn_args = get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_epochs=num_subepochs,
        num_outputs=num_classes,
        search_type=SearchTypes.ReturnnSearch,
        returnn_train_config_args={
            "python_prolog": [pretrain_construction_algo],
            "extra_python": train_python_code,
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "grad_clip": kwargs.get("grad_clip", 100.0),
            "batch_size": 15000,
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-05,
            "n_steps_per_epoch": 1100,
            "use_chunking": False,
            "target": "bpe",
            "extra_config": {
                # "pretrain": {"repetitions": 3, "construction_algo": CodeWrapper("pretrain_construction_algo")},
                "train": {
                    "bpe": {
                        "bpe_file": bpe_codes,
                        "vocab_file": bpe_vocab,
                    }
                },
                "dev": {
                    "bpe": {
                        "bpe_file": bpe_codes,
                        "vocab_file": bpe_vocab,
                    }
                },
            },
        },
        returnn_recog_config_args={
            "extra_python": recog_python_code,
            "target": "bpe",
            "hash_full_python_code": False,
            "extra_config": {
                "search_data": {
                    "bpe": {
                        "bpe_file": bpe_codes,
                        "vocab_file": bpe_vocab,
                    }
                },
                "extern_data": {
                    "data": {
                        "dim": num_inputs,
                        "sparse": False,
                    },
                    "bpe": {
                        "dim": num_classes,
                        "sparse": True,
                        "available_for_inference": False,
                    },
                    "bpe_b": {
                        "dim": num_classes_b,
                        "sparse": True,
                        "available_for_inference": False,
                    },
                },
                "search_output_layer": "ctc_decode",
            },
        },
        train_args={
            "num_classes": None,
            "use_python_control": True,
            "partition_epochs": 3,
            "log_verbosity": 4,
        },
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("extract", {"feature_key": f_name, **init_args.feature_extraction_args})
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)

    system.run(nn_steps)

    return system.get_summary_report()


def py() -> SummaryReport:

    cleaned_text_lm = run_lm(lm_cleaning=True)
    lm = run_lm(lm_cleaning=False)

    run_exp_clean_partial = partial(run_exp, cleaned_text_lm, lm_cleaning=True)
    run_exp_partial = partial(run_exp, lm, lm_cleaning=False)

    dir_handle = os.path.dirname(__file__).split("config/")[1]
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"

    summary_report = SummaryReport()

    # Newbob
    for lr in [4e-04]:
        for bp in [0.6]:
            for lm in [1.1]:
                name_suffix = f"newbob_lr-{lr}_bpe-{bpe_size}"
                recog_name = f"bp-{bp}_lm-{lm}"
                summary_report.merge_report(
                    run_exp_partial(
                        name_suffix=name_suffix,
                        recog_name=recog_name,
                        schedule=LearningRateSchedules.Newbob,
                        learning_rate=lr,
                        beam_size=32,
                        blank_penalty=bp,
                        lm_scale=lm,
                    ),
                    update_structure=True,
                )

    for lr in [4e-04]:
        for bp in [0.6]:
            for lm in [1.1]:
                name_suffix = f"newbob_lr-{lr}_bpe-{bpe_size}_clean"
                recog_name = f"bp-{bp}_lm-{lm}"
                summary_report.merge_report(
                    run_exp_clean_partial(
                        name_suffix=name_suffix,
                        recog_name=recog_name,
                        schedule=LearningRateSchedules.Newbob,
                        learning_rate=lr,
                        beam_size=32,
                        blank_penalty=bp,
                        lm_scale=lm,
                    ),
                )

    # OCLR
    for peak_lr in [2e-04]:
        name_suffix = f"oclr_lr-{peak_lr}_bpe-{bpe_size}_clean"
        summary_report.merge_report(
            run_exp_clean_partial(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=peak_lr,
                beam_size=32,
                blank_penalty=0.6,
                lm_scale=1.1,
            ),
        )

    return summary_report
