import os

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.network.models.context_1_transducer import (
    make_context_1_blstm_transducer_blank,
    make_context_1_blstm_transducer_recog,
)
from i6_experiments.users.berger.corpus.switchboard.data import get_data_inputs
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_input
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)


def run_exp(**kwargs):
    # ********** Settings **********

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    train_key = "switchboard-300h"
    dev_key = "hub5e-00"
    test_key = "hub5-01"

    stm_file = tk.Path("/u/corpora/speech/hub5e_00/xml/hub5e_00.stm")
    glm_file = tk.Path("/u/corpora/speech/hub5e_00/xml/glm")

    alignments = {
        "gmm": {
            "train": tk.Path(
                "/work/asr4/berger/dependencies/switchboard/alignments/switchboard-300h_tuske/alignment.cache.bundle"
            ),
            "dev": tk.Path(
                "/work/asr4/berger/dependencies/switchboard/alignments/hub5e-00.reduced.tuske/alignment.cache.bundle"
            ),
        }
    }
    align_type = kwargs.get("align_type", "gmm")
    alignments = alignments[align_type]

    # Subset of hub5e-00 with OOV words removed
    dev_segments = tk.Path("/work/asr4/berger/dependencies/switchboard/segments/hub5e-00.reduced")

    f_name = "gt"

    num_inputs = 40
    num_classes = 88

    am_args = {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER], [NOISE], [VOCALIZEDNOISE]",
        "phon_history_length": 0 if align_type == "ctc" else 1,
        "phon_future_length": 0 if align_type == "ctc" else 1,
        "tdp_scale": 1.0,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
    }

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(delete_empty_orth=False)
    init_args = get_init_args(sample_rate_kHz=8, scorer="hub5", feature_args={"dc_detection": align_type == "gmm"})
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
            alignments=alignments["train"],
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
            segment_path=dev_segments,
            alignments=alignments["dev"],
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

    # ********** Transducer System **********

    max_pool = kwargs.get("max_pool", [1, 2, 2])
    train_networks = {}
    recog_networks = {}
    name = "_".join(filter(None, ["BLSTM_transducer", kwargs.get("name_suffix", "")]))
    train_blstm_net, train_python_code = make_context_1_blstm_transducer_blank(
        num_outputs=num_classes,
        encoder_loss=kwargs.get("encoder_loss", True),
        loss_boost_scale=5.0,
        specaug_args={
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 5,
            "max_feature": 4,
        },
        blstm_args={
            "max_pool": max_pool,
            "l2": 5e-06,
            "dropout": 0.1,
            "size": 512,
        },
        decoder_args={
            "combination_mode": "concat",
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 2),
                "size": 1024,
                "l2": 5e-06,
                "dropout": 0.1,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 1),
                "size": 1024,
                "l2": 5e-06,
                "dropout": 0.1,
            },
        },
        output_args={"label_smoothing": 0.2},
    )

    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_context_1_blstm_transducer_recog(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "size": 512,
        },
        decoder_args={
            "combination_mode": "concat",
            "dec_mlp_args": {"num_layers": kwargs.get("num_dec_layers", 2), "size": 1024},
            "joint_mlp_args": {"num_layers": kwargs.get("num_joint_layers", 1), "size": 1024},
        },
    )

    recog_networks[name] = recog_blstm_net

    red_fact = 1
    for p in max_pool:
        red_fact *= p

    alignment_config = rasr.RasrConfig()
    alignment_config.neural_network_trainer["*"].force_single_state = True
    alignment_config.neural_network_trainer["*"].reduce_alignment_factor = red_fact
    alignment_config.neural_network_trainer["*"].peaky_alignment = True
    alignment_config.neural_network_trainer["*"].peak_position = 1.0
    alignment_config.neural_network_trainer["*"].segments_to_skip = ["switchboard-1/sw04118A/sw4118A-ms98-a-0045"]

    if red_fact > 1:
        alignment_config.neural_network_trainer["*"].segments_to_skip += [
            "switchboard-1/sw02939A/sw2939A-ms98-a-0004",
            "switchboard-1/sw03458A/sw3458A-ms98-a-0015",
        ]

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
            "batch_size": 15000,
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "final_lr": kwargs.get("final_lr", None),
            "n_steps_per_epoch": 2750,
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-05,
            "n_steps_per_epoch": 2850,
            "grad_clip": kwargs.get("grad_clip", 20.0),
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "base_chunk_size": 256,
            "chunking_factors": {"data": 1, "classes": red_fact},
            "extra_config": {
                "train": {"reduce_target_factor": red_fact},
                "dev": {"reduce_target_factor": red_fact},
            },
        },
        returnn_recog_config_args={
            "extra_python": recog_python_code,
        },
        train_args={
            "partition_epochs": 6,
            "extra_rasr_config": alignment_config,
        },
        prior_args={
            "num_classes": num_classes,
            "use_python_control": False,
            "extra_rasr_config": alignment_config,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "epochs": [num_subepochs] if kwargs.get("recog_final_only", False) else None,
            "lm_scales": [0.6],
            "prior_scales": [0.0],
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

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)

    system = TransducerSystem(rasr_binary_path=None)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )
    system.stm_files[dev_key] = stm_file
    system.glm_files[dev_key] = glm_file

    system.run(nn_steps)


def py():
    # OCLR
    if True:
        run_exp(
            name_suffix="gmm-align_oclr_peak-lr-0.0008",
            schedule=LearningRateSchedules.OCLR,
            peak_lr=0.0008,
            lm_scales=[0.6, 0.7, 0.8, 0.9],
            final_lr=1e-06,
        )
