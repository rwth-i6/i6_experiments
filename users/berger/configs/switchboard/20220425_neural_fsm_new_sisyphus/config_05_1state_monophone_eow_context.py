import os
import copy

from sisyphus import gs, tk

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.text as text

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.rasr_systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.transducer_args import get_nn_args
from i6_experiments.users.berger.network.models.viterbi_context_1_transducer import (
    make_context_1_blstm_transducer_noblank,
    make_context_1_blstm_transducer_recog,
)
from i6_experiments.users.berger.corpus.switchboard.data import get_data_inputs
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import get_returnn_rasr_data_input
from i6_experiments.users.berger.recipe.datasets.switchboard import (
    PreprocessSwitchboardLexiconJob,
)


def py():
    # ********** Settings **********

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"
    rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

    train_key = "switchboard-300h"
    dev_key = "hub5e-00"
    test_key = "hub5-01"

    stm_file = "/u/corpora/speech/hub5e_00/xml/hub5e_00.stm"
    glm_file = "/u/corpora/speech/hub5e_00/xml/glm"

    train_alignment = tk.Path(
        "/work/asr4/berger/dependencies/switchboard/alignments/switchboard-300h_tuske/alignment.cache.bundle"
    )
    dev_alignment = tk.Path(
        "/work/asr4/berger/dependencies/switchboard/alignments/hub5e-00.reduced.tuske/alignment.cache.bundle"
    )

    # Subset of hub5e-00 with OOV words removed
    dev_segments = tk.Path("/work/asr4/berger/dependencies/switchboard/segments/hub5e-00.reduced")

    allophone_file = tk.Path("/work/asr4/berger/dependencies/switchboard/allophones/tuske_allophones")

    f_name = "gt"

    num_inputs = 40
    num_classes = 88

    nonword_labels = [2]  # LAUGHTER, NOISE, SILENCE, VOCALIZED-NOISE
    transform_func = "tf.where(tf.math.greater(source(0), 45), source(0)-42, source(0))"
    transformed_label_dim = 46

    am_args = {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER], [NOISE], [VOCALIZEDNOISE]",
    }

    recog_am_args = copy.deepcopy(am_args)
    recog_am_args.update(
        {
            "state_tying": "monophone-eow",
            # "state_tying_file": "/work/asr3/berger/asr-exps/switchboard/dependencies/state-tying/monophone-eow_1state_3tied",
            "states_per_phone": 1,
            "phon_history_length": 0,
            "phon_future_length": 0,
            "tdp_transition": (3.0, 0.0, 30.0, 0.0),
            "tdp_silence": (0.0, 3.0, "infinity", 15.0),
            "tdp_nonword": (0.0, 3.0, "infinity", 20.0),
        }
    )
    recog_am_args["tdp_scale"] = 0.1

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs()
    init_args = get_init_args(sample_rate_kHz=8, scorer="hub5")
    init_args.feature_extraction_args = {
        f_name: init_args.feature_extraction_args[f_name]
    }  # only keep fname and discard other features

    feature_system = gmm_system.GmmSystem()
    feature_system.init_system(
        hybrid_init_args=init_args,
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
            alignments=train_alignment,
            am_args=am_args,
            allophone_file=allophone_file,
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
            alignments=dev_alignment,
            am_args=am_args,
            allophone_file=allophone_file,
            shuffle_data=False,
            concurrent=1,
        )
    }

    nn_data_inputs["dev"] = {
        dev_key: get_returnn_rasr_data_input(
            dev_data_inputs[dev_key],
            feature_flow=feature_system.feature_flows[dev_key][f_name],
            features=feature_system.feature_caches[dev_key][f_name],
            am_args=recog_am_args,
            # allophone_file=allophone_file,
            shuffle_data=False,
        )
    }
    nn_data_inputs["test"] = {}

    # Allophone correction
    nn_data_inputs["dev"][dev_key].crp.acoustic_model_config.allophones.add_from_lexicon = False
    nn_data_inputs["dev"][dev_key].crp.acoustic_model_config.allophones.add_all = True

    # ********** Transducer System **********

    for context_trafo in [False]:
        for suffix, max_pool in [
            ("", []),
            ("_pool-2", [1, 1, 2]),
        ]:
            train_networks = {}
            recog_networks = {}
            name = "BLSTM_transducer" + suffix + ("_mono-context" if context_trafo else "_eow-context")
            (train_blstm_net, train_python_code,) = make_context_1_blstm_transducer_noblank(
                num_outputs=num_classes,
                nonword_labels=nonword_labels,
                context_transformation_func=transform_func if context_trafo else None,
                context_label_dim=transformed_label_dim if context_trafo else num_classes,
                loss_boost_scale=0,
                blstm_args={
                    "max_pool": max_pool,
                    "size": 512,
                },
                decoder_args={
                    "dec_mlp_args": {"num_layers": 1, "size": 1024},
                    "joint_mlp_args": {"num_layers": 1, "size": 1024},
                },
                output_args={"label_smoothing": 0.2, "focal_loss": 2.0},
            )

            train_networks[name] = train_blstm_net

            recog_blstm_net, recog_python_code = make_context_1_blstm_transducer_recog(
                num_outputs=num_classes,
                nonword_labels=nonword_labels,
                context_transformation_func=transform_func,
                context_label_dim=transformed_label_dim,
                blstm_args={
                    "size": 512,
                },
                decoder_args={
                    "dec_mlp_args": {"num_layers": 1, "size": 1024},
                    "joint_mlp_args": {"num_layers": 1, "size": 1024},
                },
            )

            recog_networks[name] = recog_blstm_net

            red_fact = 1
            for p in max_pool:
                red_fact *= p

            alignment_config = rasr.RasrConfig()
            alignment_config.neural_network_trainer["*"].force_single_state = True
            alignment_config.neural_network_trainer["*"].reduce_alignment_factor = red_fact

            if red_fact > 1:
                alignment_config.neural_network_trainer["*"].segments_to_skip = [
                    "switchboard-1/sw04118A/sw4118A-ms98-a-0045",
                    "switchboard-1/sw02939A/sw2939A-ms98-a-0004",
                ]

            nn_args = get_nn_args(
                train_networks=train_networks,
                recog_networks=recog_networks,
                num_inputs=num_inputs,
                num_outputs=num_classes,
                num_epochs=180,
                returnn_train_config_args={
                    "extra_python": train_python_code,
                    "chunking_factors": {"data": 1, "classes": red_fact},
                    "extra_config": {
                        "train": {"reduce_target_factor": red_fact},
                        "dev": {"reduce_target_factor": red_fact},
                    },
                },
                returnn_recog_config_args={"extra_python": recog_python_code},
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
                recog_keys=[dev_key],
                recog_args={
                    "lm_scales": [1.0],
                    "prior_scales": [0.25],
                    "epochs": [180],
                    "use_gpu": True,
                    "label_unit": "phoneme",
                    "add_eow": True,
                    "allow_blank": False,
                    "allow_loop": True,
                    # "recombination_limit": 1,
                    # "recog_lexicon": recog_lexicon,
                    "label_scorer_type": "tf-ffnn-transducer",
                    "lp": 25.0,
                    "label_scorer_args": {
                        "use_prior": True,
                        "num_classes": num_classes,
                        "extra_args": {
                            "context_size": 1,
                            "max_batch_size": 256,
                            "reduction_factors": red_fact,
                            "use_start_label": True,
                            "start_label_index": num_classes,
                            "transform_output_negate": True,
                        },
                    },
                    "label_tree_args": {
                        "use_transition_penalty": True,
                        "skip_silence": False,
                    },
                },
            )

            nn_steps = rasr_util.RasrSteps()
            nn_steps.add_step("nn", nn_args)

            system = TransducerSystem()
            system.init_system(
                hybrid_init_args=init_args,
                train_data=nn_data_inputs["train"],
                cv_data=nn_data_inputs["cv"],
                dev_data=nn_data_inputs["dev"],
                test_data=nn_data_inputs["test"],
                train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
            )
            system.stm_files[dev_key] = stm_file
            system.glm_files[dev_key] = glm_file

            system.run(nn_steps)
