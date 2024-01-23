import os
from i6_experiments.users.berger.args.returnn import learning_rates

from sisyphus import gs, tk

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.switchboard.data import get_data_inputs
from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.network.models.fullsum_ctc import (
    make_blstm_fullsum_ctc_model,
    make_blstm_ctc_recog_model,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_input
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)


# ********** Settings **********

filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "switchboard-300h"
dev_key = "hub5e-00"
test_key = "hub5-01"

stm_file = "/u/corpora/speech/hub5e_00/xml/hub5e_00.stm"
glm_file = "/u/corpora/speech/hub5e_00/xml/glm"

# Subset of hub5e-00 with OOV words removed
cv_segments = tk.Path("/work/asr4/berger/dependencies/switchboard/segments/hub5e-00.reduced")

f_name = "gt"

num_inputs = 40
num_classes = 88


def run_exp(**kwargs):

    am_args = {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER], [NOISE], [VOCALIZEDNOISE]",
        "phon_future_length": 0,
        "phon_history_length": 0,
        "tdp_scale": 1.0,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
    }

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(delete_empty_orth=True)

    # rename dev corpus to train corpus name
    # dev_data_inputs[dev_key].corpus_object.corpus_file = corpus_recipe.MergeCorporaJob(
    #     [dev_data_inputs[dev_key].corpus_object.corpus_file],
    #     name="switchboard-1",
    #     merge_strategy=corpus_recipe.MergeStrategy.CONCATENATE,
    # ).out_merged_corpus

    # initialize feature system and extract features
    init_args = get_init_args(sample_rate_kHz=8, scorer="hub5")
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
    feature_system.stm_files[dev_key] = stm_file
    feature_system.glm_files[dev_key] = glm_file
    feature_system.run(["extract"])

    # ********** Data inputs **********

    nn_data_inputs = {}

    nn_data_inputs["train"] = {
        f"{train_key}.train": get_returnn_rasr_data_input(
            train_data_inputs[train_key],
            feature_flow=feature_system.feature_flows[train_key][f_name],
            features=feature_system.feature_caches[train_key][f_name],
            am_args=am_args,
            shuffle_data=True,
            concurrent=1,
        )
    }

    nn_data_inputs["cv"] = {
        f"{train_key}.cv": get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            segment_path=cv_segments,
            am_args=am_args,
            shuffle_data=False,
            concurrent=1,
        )
    }

    nn_data_inputs["dev"] = {
        dev_key: get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            am_args=am_args,
            shuffle_data=False,
        )
    }
    nn_data_inputs["test"] = {}

    nn_data_inputs["align"] = {
        f"{train_key}.train": get_returnn_rasr_data_input(
            train_data_inputs[train_key],
            feature_flow=feature_system.feature_flows[train_key][f_name],
            features=feature_system.feature_caches[train_key][f_name],
            am_args=am_args,
            shuffle_data=False,
            concurrent=100,
        ),
        f"{train_key}.cv": get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            segment_path=cv_segments,
            am_args=am_args,
            shuffle_data=False,
            concurrent=10,
        ),
    }

    # ********** Transducer System **********

    system = TransducerSystem(rasr_binary_path=None)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        align_data=nn_data_inputs["align"],
    )

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
            "max_feature_num": 5,
            "max_feature": 4,
        },
        blstm_args={
            "max_pool": max_pool,
            "l2": kwargs.get("l2", 5e-06),
            "dropout": kwargs.get("dropout", 0.1),
            "size": 512,
        },
        mlp_args={
            "num_layers": 0,
        },
    )
    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_blstm_ctc_recog_model(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "size": 512,
        },
        mlp_args={
            "num_layers": 0,
        },
    )
    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_subepochs", 300)

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
            "use_chunking": False,
            "batch_size": kwargs.get("batch_size", 10000),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "final_lr": kwargs.get("final_lr", None),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-05,
            "n_steps_per_epoch": 2850,
        },
        returnn_recog_config_args={"extra_python": recog_python_code},
        train_args={
            "partition_epochs": 6,
            "use_rasr_ctc_loss": True,
            "rasr_ctc_loss_args": {"allow_label_loop": True},
        },
        prior_args={
            "num_classes": num_classes,
            "mem_rqmt": 6.0,
        },
        recog_name=kwargs.get("recog_name", ""),
        recog_args={
            "lm_scales": kwargs.get("lm_scales", [0.8]),
            "prior_scales": kwargs.get("prior_scales", [0.3]),
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
            "epochs": [num_epochs] if kwargs.get("align", False) else [],
            "prior_scales": kwargs.get("prior_scales", [0.3]),
            "use_gpu": False,
            "label_unit": "phoneme",
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

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("realign", nn_args)

    system.stm_files[dev_key] = stm_file
    system.glm_files[dev_key] = glm_file

    system.run(nn_steps)


def py():
    # Newbob
    if True:
        for lr in [1e-04, 3e-04, 5e-04, 1e-03, 3e-03]:
            name_suffix = "newbob"
            name_suffix += f"_lr-{lr}"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.Newbob,
                learning_rate=lr,
                lm_scales=[0.7, 0.8, 0.9],
                prior_scales=[0.3, 0.4, 0.5],
            )

    # OCLR
    if True:
        for peak_lr in [3e-04, 7e-04, 1e-03, 3e-03]:
            name_suffix = "oclr"
            name_suffix += f"_peak-lr-{peak_lr}"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=peak_lr,
                lm_scales=[0.7, 0.8, 0.9],
                prior_scales=[0.3, 0.4, 0.5],
                final_lr=1e-06,
            )

    # Other specaug
    if True:
        for time_num, max_time in [(3, 10), (1, 15)]:
            name_suffix = "newbob"
            name_suffix += f"_tn-{time_num}_mt-{max_time}"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.Newbob,
                learning_rate=5e-04,
                lm_scales=[0.7, 0.8, 0.9],
                prior_scales=[0.3, 0.4, 0.5],
                final_lr=1e-06,
            )
