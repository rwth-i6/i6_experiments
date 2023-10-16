import os
import copy

from sisyphus import gs, tk
from sisyphus.delayed_ops import DelayedFormat

import i6_core.rasr as rasr
import i6_core.text as text
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools import CloneGitRepositoryJob

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
    PreprocessWSJTranscriptionsJob,
)
from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
    make_blstm_ctc_recog_model,
)
from i6_experiments.users.berger.network.models.lstm_lm import (
    make_lstm_lm_model,
    make_lstm_lm_recog_model,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.returnn.pretrain import pretrain_construction_algo
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_input
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "sms_train_si284"
dev_key = "sms_cv_dev93"
test_key = "sms_test_eval92"

frequency = 16

f_name = "gt"

num_inputs = 50

lm_models = {
    100: tk.Path(
        # "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.OXKudR5Jxo0p/output/models/epoch.150.meta"
        "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.d43AUmy7n8xx/output/models/epoch.150.meta"
    ),
    1000: tk.Path(
        "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/training/ReturnnTrainingJob.Ol9oGphLUIx5/output/models/epoch.150.meta"
    ),
}


def run_exp(**kwargs) -> None:

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=True,
    )

    # initialize feature system and extract features
    init_args = get_init_args(
        sample_rate_kHz=frequency,
        stm_args={
            "non_speech_tokens": ["<NOISE>"],
        },
    )
    init_args.feature_extraction_args = {
        f_name: init_args.feature_extraction_args[f_name]
    }  # only keep fname and discard other features

    feature_system = gmm_system.GmmSystem(rasr_binary_path=None)
    feature_system.init_system(
        rasr_init_args=init_args,
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )
    feature_system.run(["extract"])

    train_corpus_object = train_data_inputs[train_key].corpus_object
    train_corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(
        train_corpus_object.corpus_file,
        lm_cleaning=True,
    ).out_corpus_file

    dev_corpus_object = dev_data_inputs[dev_key].corpus_object
    dev_corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(
        dev_corpus_object.corpus_file,
        lm_cleaning=True,
    ).out_corpus_file

    test_corpus_object = test_data_inputs[test_key].corpus_object
    test_corpus_object.corpus_file = PreprocessWSJTranscriptionsJob(
        test_corpus_object.corpus_file,
        lm_cleaning=True,
    ).out_corpus_file

    # ********** Data inputs **********

    nn_data_inputs = {}

    nn_data_inputs["train"] = {
        f"{train_key}.train": get_returnn_rasr_data_input(
            train_data_inputs[train_key],
            feature_flow=feature_system.feature_flows[train_key][f_name],
            features=feature_system.feature_caches[train_key][f_name],
            shuffle_data=True,
            concurrent=1,
        )
    }

    nn_data_inputs["cv"] = {
        f"{train_key}.cv": get_returnn_rasr_data_input(
            copy.deepcopy(dev_data_inputs[dev_key]),
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            shuffle_data=False,
            concurrent=1,
        )
    }

    nn_data_inputs["dev"] = {
        dev_key: get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            shuffle_data=False,
        )
    }
    nn_data_inputs["test"] = {
        test_key: get_returnn_rasr_data_input(
            test_data_inputs[test_key],
            feature_flow=feature_system.feature_flows[test_key][f_name],
            features=feature_system.feature_caches[test_key][f_name],
            shuffle_data=False,
        )
    }

    # ********** Transducer System **********

    system = TransducerSystem(rasr_binary_path=None)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )

    subword_nmt_repo = CloneGitRepositoryJob("https://github.com/albertz/subword-nmt.git").out_repository

    train_corpus = tk.Path("/work/asr4/berger/dependencies/sms_wsj/text/NAB-training-corpus-clean.txt")

    bpe_size = kwargs.get("bpe_size", 500)
    train_bpe_job = ReturnnTrainBpeJob(
        train_corpus,
        bpe_size,
        subword_nmt_repo=subword_nmt_repo,
    )
    bpe_codes = train_bpe_job.out_bpe_codes
    bpe_vocab = train_bpe_job.out_bpe_vocab

    num_classes = train_bpe_job.out_vocab_size  # bpe count
    num_classes_b = num_classes + 1  # bpe count + blank

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))
    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_model(
        num_outputs=num_classes_b,
        specaug_args={
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 5,
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

    recog_blstm_net["output_bp"] = {
        "class": "eval",
        "from": output_name,
        "eval": DelayedFormat(
            "source(0) - tf.expand_dims(tf.one_hot([{}], {}, on_value={}, dtype=tf.float32), axis=0)",
            num_classes,
            num_classes_b,
            kwargs.get("blank_penalty", 0.0),
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
                        "load_on_init": lm_models[bpe_size],
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
        recog_blstm_net["beam_search"]["unit"]["output"]["from"] = "combined_scores"

    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_epochs", 150)

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
            "n_steps_per_epoch": 2370,
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
        recog_name=kwargs.get("recog_name", None),
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
        prior_args={
            "num_classes": num_classes_b,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "epochs": [num_subepochs] if kwargs.get("recog_final_only", False) else None,
            "prior_scales": kwargs.get("prior_scales", None),
            "log_prob_layer": output_name,
        },
        test_recog_args={
            "epochs": [num_subepochs] if kwargs.get("recog_final_only", False) else None,
            "prior_scales": kwargs.get("prior_scales", None),
            "log_prob_layer": output_name,
        },
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("bpe_nn", nn_args)
    nn_steps.add_step("bpe_recog", nn_args)

    system.run(nn_steps)


def py() -> None:
    # Newbob
    if False:
        for bpe_size, lr in [(100, 4e-04), (1000, 1e-04), (1000, 8e-05), (1000, 6e-05)]:
            for bp in [0.0, 0.2, 0.4, 0.6]:
                for lm in [0.5, 0.7, 0.9, 1.1]:
                    name_suffix = f"newbob_lr-{lr}_bpe-{bpe_size}"
                    recog_name = f"bp-{bp}_lm-{lm}"
                    run_exp(
                        name_suffix=name_suffix,
                        recog_name=recog_name,
                        schedule=LearningRateSchedules.Newbob,
                        learning_rate=lr,
                        bpe_size=bpe_size,
                        beam_size=32,
                        blank_penalty=bp,
                        lm_scale=lm,
                        recog_final_only=True,
                    )

    # OCLR
    if False:
        for bpe_size in [100, 1000]:
            for peak_lr in [8e-05, 1e-04, 2e-04, 1e-03, 3e-03]:
                for bp in [0.6]:
                    for lm in [0.5]:
                        name_suffix = f"oclr_lr-{peak_lr}_bpe-{bpe_size}"
                        recog_name = f"bp-{bp}_lm-{lm}"
                        run_exp(
                            name_suffix=name_suffix,
                            schedule=LearningRateSchedules.OCLR,
                            peak_lr=peak_lr,
                            bpe_size=bpe_size,
                            beam_size=32,
                            blank_penalty=bp,
                            lm_scale=lm,
                            recog_final_only=True,
                        )

    # more agressive subsampling
    if False:
        for pool in [[2, 2, 2], [1, 2, 3], [1, 3, 3], [1, 2, 4], [1, 2, 5]]:
            for bp in [0.6]:
                for lm in [1.1]:
                    name_suffix = f"newbob_bpe-100_pool-{'-'.join([str(f) for f in pool])}"
                    recog_name = f"bp-{bp}_lm-{lm}"
                    run_exp(
                        name_suffix=name_suffix,
                        recog_name=recog_name,
                        schedule=LearningRateSchedules.Newbob,
                        max_pool=pool,
                        learning_rate=4e-04,
                        bpe_size=100,
                        beam_size=32,
                        blank_penalty=bp,
                        lm_scale=lm,
                        recog_final_only=True,
                    )

    # More learning rate tuning
    if True:
        for lr in [2e-04, 3e-04, 4e-04, 5e-04, 6e-04, 8e-04, 1e-03, 1e-03]:
            for bp in [0.0, 0.3, 0.6, 0.9]:
                for lm in [0.7, 0.9, 1.1, 1.3, 1.5]:
                    name_suffix = f"newbob_lr-{lr}_bpe-100"
                    recog_name = f"bp-{bp}_lm-{lm}"
                    run_exp(
                        name_suffix=name_suffix,
                        recog_name=recog_name,
                        schedule=LearningRateSchedules.Newbob,
                        learning_rate=lr,
                        bpe_size=100,
                        beam_size=32,
                        blank_penalty=bp,
                        lm_scale=lm,
                        prior_scales=[0.0, 0.3, 0.6, 0.9],
                        recog_final_only=True,
                    )
