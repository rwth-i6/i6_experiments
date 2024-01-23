import os

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
import i6_experiments.users.berger.args.jobs.hybrid_args as hybrid_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.blstm_hybrid import (
    make_blstm_hybrid_model,
    make_blstm_hybrid_recog_model,
)
from i6_experiments.users.berger.corpus.sms_wsj.data import (
    get_data_inputs,
)
from i6_experiments.users.berger.args.jobs.rasr_init_args import get_init_args
from i6_experiments.users.berger.args.jobs.data import (
    get_returnn_rasr_data_inputs,
)
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

train_alignment = tk.Path(
    "/work/asr4/berger/dependencies/sms_wsj/alignment/gmm/sms_train_si284_speechsource/alignment.cache.bundle"
)
dev_alignment = tk.Path(
    "/work/asr4/berger/dependencies/sms_wsj/alignment/gmm/sms_cv_dev93_speechsource/alignment.cache.bundle"
)

f_name = "gt"

frequency = 16

num_inputs = 50
num_classes = 9001


def run_exp(**kwargs):

    am_args = {
        "state_tying": "cart",
        "state_tying_file": tk.Path("/work/asr4/berger/dependencies/sms_wsj/cart/cart.tree.xml.gz"),
    }

    # ********** GMM System **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        add_all_allophones=False,
    )
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

    # ********** Data inputs **********

    nn_data_inputs = get_returnn_rasr_data_inputs(
        train_data_inputs=train_data_inputs,
        cv_data_inputs=dev_data_inputs,
        dev_data_inputs=dev_data_inputs,
        test_data_inputs=test_data_inputs,
        feature_flows=feature_system.feature_flows,
        feature_caches=feature_system.feature_caches,
        train_alignments={train_key: train_alignment},
        cv_alignments={dev_key: dev_alignment},
        am_args=am_args,
    )

    # ********** Hybrid System **********

    train_networks = {}
    recog_networks = {}

    name = "_".join(filter(None, ["BLSTM_CTC", kwargs.get("name_suffix", "")]))

    train_blstm_net, train_python_code = make_blstm_hybrid_model(
        specaug_args={
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 5,
            "max_feature": 5,
        },
        blstm_args={
            "num_layers": 6,
            "size": 400,
        },
        mlp_args={
            "num_layers": 0,
        },
        num_outputs=num_classes,
    )
    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_blstm_hybrid_recog_model(
        blstm_args={
            "num_layers": 6,
            "size": 400,
        },
        mlp_args={
            "num_layers": 0,
        },
        num_outputs=num_classes,
    )
    recog_networks[name] = recog_blstm_net

    num_subepochs = kwargs.get("num_subepochs", 180)

    nn_args = hybrid_args.get_nn_args(
        train_networks=train_networks,
        recog_networks=recog_networks,
        num_inputs=num_inputs,
        num_outputs=num_classes,
        num_epochs=num_subepochs,
        search_type=SearchTypes.AdvancedTreeSearch,
        returnn_train_config_args={
            "extra_python": train_python_code,
            "batch_size": 15000,
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "n_steps_per_epoch": 2370,
        },
        returnn_recog_config_args={"extra_python": recog_python_code},
        train_args={"partition_epochs": 3},
        prior_args={
            "use_python_control": False,
            "num_classes": num_classes,
            "mem_rqmt": 6.0,
        },
        recog_args={
            "epochs": [num_subepochs] if kwargs.get("recog_final_epoch_only", False) else None,
            "lm_scales": kwargs.get("lm_scales", [16.0]),
            "prior_scales": kwargs.get("prior_scales", [0.7]),
            "use_gpu": False,
            "label_file_blank": False,
            "add_eow": False,
            "add_sow": False,
            "allow_blank": False,
            "label_unit": "hmm",
            "label_scorer_type": "precomputed-log-posterior",
            # "lp": 15.0,
            # "label_scorer_args": {
            #     "use_prior": True,
            #     "num_classes": num_classes,
            # },
            # "label_tree_args": {
            #     "use_transition_penalty": True,
            #     "skip_silence": False,
            # },
            "rtf": 20,
            "mem": 8,
        },
    )

    nn_steps = rasr_util.RasrSteps()
    nn_steps.add_step("nn", nn_args)
    nn_steps.add_step("nn_recog", nn_args)

    wsj_hybrid_system = TransducerSystem(rasr_binary_path=None)
    wsj_hybrid_system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        # train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )

    wsj_hybrid_system.run(nn_steps)


def py():
    run_exp()
