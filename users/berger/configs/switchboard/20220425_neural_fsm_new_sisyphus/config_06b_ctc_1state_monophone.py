import os
import copy

from sisyphus import gs, tk

import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr
import i6_core.text as text
from i6_core.lib import corpus

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.corpus.switchboard.data import get_data_inputs
from i6_experiments.users.berger.rasr_systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.transducer_args import get_nn_args
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

# recog_lexicon = tk.Path("/work/asr3/berger/asr-exps/switchboard/dependencies/lexicon/recog-lexicon.1state.eow.gz")

# Subset of hub5e-00 with OOV words removed
cv_segments = tk.Path("/work/asr4/berger/dependencies/switchboard/segments/hub5e-00.renamed.reduced")

# allophone_file = tk.Path("/work/asr4/berger/dependencies/switchboard/allophones/tuske_allophones")

f_name = "gt"

num_inputs = 40
num_classes = 46


def run_exp(**kwargs):

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 1,
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER], [NOISE], [VOCALIZEDNOISE]",
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
        "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
    }

    recog_am_args = copy.deepcopy(am_args)
    # recog_am_args["state_tying"] = "lookup"
    # recog_am_args["state_tying_file"] = tk.Path("/work/asr3/berger/asr-exps/switchboard/dependencies/state-tying/monophone-eow_1state")

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(delete_empty_orth=True)

    # rename dev corpus to train corpus name
    train_corpus = corpus.Corpus()
    train_corpus_path = train_data_inputs[train_key].corpus_object.corpus_file.get_path()
    train_corpus.load(train_corpus_path)

    dev_corpus_path = dev_data_inputs[dev_key].corpus_object.corpus_file
    dev_corpus_path = corpus_recipe.MergeCorporaJob(
        [dev_corpus_path],
        name=train_corpus.name,
        merge_strategy=corpus_recipe.MergeStrategy.CONCATENATE,
    ).out_merged_corpus
    dev_data_inputs[dev_key].corpus_object.corpus_file = dev_corpus_path

    # initialize feature system and extract features
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
            am_args=am_args,
            # allophone_file=allophone_file,
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
            # allophone_file=allophone_file,
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

    # ********** Transducer System **********

    system = TransducerSystem()
    system.init_system(
        hybrid_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))
    max_pool = kwargs.get("max_pool", [1, 2, 1, 2])
    red_fact = 1
    for fact in max_pool:
        red_fact *= fact
    train_blstm_net, train_python_code = make_blstm_fullsum_ctc_model(
        num_outputs=num_classes,
        specaug_args={
            "max_time_num": 3,
            "max_time": 10,
            "max_feature_num": 5,
            "max_feature": 4,
        },
        blstm_args={
            "max_pool": max_pool,
            "l2": kwargs.get("l2", 0.01),
            "size": 512,
        },
        mlp_args={
            "num_layers": 2,
            "l2": kwargs.get("l2", 0.01),
            "size": 1024,
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
            "num_layers": 2,
            "size": 1024,
        },
    )
    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_subepochs", 180)

    nn_args = get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        returnn_train_config_args={
            "extra_python": train_python_code,
            "grad_noise": kwargs.get("grad_noise", 0.1),
            "grad_clip": kwargs.get("grad_clip", 20.0),
            "use_chunking": False,
            "batch_size": kwargs.get("batch_size", 10000),
            "schedule": LearningRateSchedules.OCLR,
            "initial_lr": kwargs.get("initial_lr", 1e-4),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "final_lr": kwargs.get("final_lr", 1e-6),
            "total_epochs": num_subepochs,
            "cycle_epoch": num_subepochs * 4 // 10,
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
        recog_names=kwargs.get("recog_name", ""),
        recog_args={
            "lm_scales": kwargs.get("lm_scales", [0.8]),
            "prior_scales": kwargs.get("prior_scales", [0.3]),
            "use_gpu": False,
            "label_unit": "phoneme",
            "add_eow": False,
            "allow_blank": True,
            "allow_loop": True,
            # "recog_lexicon": recog_lexicon,
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
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)

    system.stm_files[dev_key] = stm_file
    system.glm_files[dev_key] = glm_file

    system.run(nn_steps)


def py():
    for max_pool in [[], [1, 1, 2], [1, 2, 1, 2]]:
        for init_lr, peak_lr, final_lr in [
            (1e-4, 1e-3, 1e-6),
            (1e-5, 3e-4, 1e-5),
        ]:
            if max_pool == [1, 1, 2]:
                name_suffix = "pool-2"
            elif max_pool == [1, 2, 1, 2]:
                name_suffix = "pool-4"
            else:
                name_suffix = "no-pool"
            name_suffix += f"_peak-lr-{peak_lr}"
            run_exp(
                name_suffix=name_suffix,
                l2=0.0001,
                max_pool=max_pool,
                initial_lr=init_lr,
                peak_lr=peak_lr,
                final_lr=final_lr,
                num_subepochs=300,
                grad_noise=0.0,
                grad_clip=100.0,
                lm_scales=[0.8, 1.0, 1.2, 1.4],
                prior_scales=[0.3, 0.5, 0.7],
            )
