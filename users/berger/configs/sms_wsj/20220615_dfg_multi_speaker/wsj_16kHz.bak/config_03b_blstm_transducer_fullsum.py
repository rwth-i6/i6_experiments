import os

from sisyphus import gs, tk

import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.gmm_system as gmm_system
import i6_experiments.common.setups.rasr.util as rasr_util

from i6_experiments.users.berger.systems.transducer_system import TransducerSystem
from i6_experiments.users.berger.args.jobs.hybrid_args import get_nn_args
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from i6_experiments.users.berger.network.models.context_1_transducer import (
    get_viterbi_transducer_alignment_config,
    make_context_1_blstm_transducer_blank,
    make_context_1_blstm_transducer_fullsum,
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


# ********** Settings **********

dir_handle = os.path.dirname(__file__).split("config/")[1]
filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
gs.ALIAS_AND_OUTPUT_SUBDIR = f"{dir_handle}/{filename_handle}/"
rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

train_key = "train_si284"
dev_key = "cv_dev93"
test_key = "test_eval92"

cv_segments = tk.Path("/work/asr4/berger/dependencies/sms_wsj/segments/cv_dev93_16kHz.reduced")

# Original WER: 4.3
# viterbi_model = tk.Path(
#     "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/rasr_training/ReturnnRasrTrainingJob.DzUlhFloyCtK/output/models/epoch.240.index"
# )
# Original WER: 3.3
viterbi_model = tk.Path(
    "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/rasr_training/ReturnnRasrTrainingJob.i17PIYLSFkbp/output/models/epoch.230.meta"
)
train_alignment = tk.Path(
    "/work/asr4/berger/dependencies/sms_wsj/alignment/16kHz/train_si284_gmm/alignment.cache.bundle"
)
dev_alignment = tk.Path("/work/asr4/berger/dependencies/sms_wsj/alignment/16kHz/cv_dev93_gmm/alignment.cache.bundle")

gmm_allophones = tk.Path(
    "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/lexicon/allophones/StoreAllophonesJob.74FPxuoluGhv/output/allophones"
)
ctc_allophones = tk.Path(
    "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/lexicon/allophones/StoreAllophonesJob.KIW6XeiDZx8T/output/allophones"
)

frequency = 16

f_name = "gt"

num_inputs = 50
num_classes = 87


def run_exp(**kwargs):
    max_pool = kwargs.get("max_pool", [1, 2, 2])
    red_fact = 1
    for p in max_pool:
        red_fact *= p

    am_args = {
        "state_tying": "monophone-eow",
        "states_per_phone": 1,
        # "phon_history_length": 0,
        # "phon_future_length": 0,
        "tdp_scale": 1.0,
        "tdp_transition": (0.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 0.0, "infinity", 0.0),
    }

    # ********** Init args **********

    train_data_inputs, dev_data_inputs, test_data_inputs = get_data_inputs(
        train_keys=[train_key],
        dev_keys=[dev_key],
        test_keys=[test_key],
        freq=frequency,
        lm_name="64k_3gram",
        recog_lex_name="nab-64k",
        delete_empty_orth=False,
    )
    init_args = get_init_args(sample_rate_kHz=frequency)
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
        train_cv_pairing=[(train_key, dev_key)],
        feature_flows=feature_system.feature_flows,
        feature_caches=feature_system.feature_caches,
        am_args=am_args,
        train_alignment=train_alignment,
        cv_alignment=dev_alignment,
        cv_segments=cv_segments,
        allophone_file=gmm_allophones,
    )

    # ********** Transducer System **********

    name = "_".join(filter(None, ["BLSTM_transducer", kwargs.get("name_suffix", "")]))

    train_networks = {}
    recog_networks = {}

    l2 = kwargs.get("l2", 5e-06)
    dropout = kwargs.get("dropout", 0.1)
    train_blstm_net, train_python_code = make_context_1_blstm_transducer_fullsum(
        num_outputs=num_classes,
        compress_joint_input=kwargs.get("compressed_join", False),
        specaug_args={
            "max_time_num": 3,
            "max_time": 15,
            "max_feature_num": 5,
            "max_feature": 5,
        },
        blstm_args={
            "max_pool": max_pool,
            "l2": l2,
            "dropout": dropout,
            "size": 400,
        },
        decoder_args={
            "combination_mode": "concat",
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 2),
                "size": 800,
                "l2": l2,
                "dropout": dropout,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 1),
                "size": 600,
                "l2": l2,
                "dropout": dropout,
            },
        },
    )

    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_context_1_blstm_transducer_recog(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "size": 400,
        },
        decoder_args={
            "combination_mode": "concat",
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 2),
                "size": 800,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 1),
                "size": 600,
            },
        },
    )

    recog_networks[name] = recog_blstm_net

    alignment_config = get_viterbi_transducer_alignment_config(red_fact)

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
            "batch_size": kwargs.get("batch_size", 15000),
            "accum_grad": kwargs.get("accum_grad", 1),
            "grad_noise": kwargs.get("grad_noise", 0.0),
            "grad_clip": kwargs.get("grad_clip", 100.0),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-5),
            "const_lr": kwargs.get("const_lr", 1e-5),
            "learning_rate": kwargs.get("learning_rate", 1e-05),
            "min_learning_rate": 1e-06,
            "n_steps_per_epoch": kwargs.get("n_steps_per_epoch", 1100),
            "use_chunking": False,
            "extra_config": {
                "preload_from_files": {
                    "base": {
                        "init_for_train": True,
                        "ignore_missing": True,
                        "filename": viterbi_model,
                    }
                },
                "train": {"reduce_target_factor": red_fact},
                "dev": {"reduce_target_factor": red_fact},
            },
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
            "epochs": [num_subepochs] if kwargs.get("recog_final_epoch_only", False) else None,
            "lm_scales": kwargs.get("lm_scales", [0.6]),
            "prior_scales": kwargs.get("prior_scales", [0.0]),
            "use_gpu": True,
            "label_unit": "phoneme",
            "add_eow": True,
            "allow_blank": True,
            "allow_loop": False,
            "blank_penalty": kwargs.get("blank_penalty", 0.0),
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
    nn_steps.add_step("nn_recog", nn_args)

    system = TransducerSystem(rasr_binary_path=None)
    system.init_system(
        rasr_init_args=init_args,
        train_data=nn_data_inputs["train"],
        cv_data=nn_data_inputs["cv"],
        dev_data=nn_data_inputs["dev"],
        test_data=nn_data_inputs["test"],
        train_cv_pairing=[(f"{train_key}.train", f"{train_key}.cv")],
    )

    system.run(nn_steps)


def py():
    # OCLR
    # Peak-lr   WER
    # 5e-06     4.5
    # 1e-05     4.7
    # 2e-05     4.7
    # -> No improvements
    # Min/Ep: 17 at 5k Batch size
    if False:
        for peak_lr in [5e-06, 1e-05, 2e-05]:
            name_suffix = f"oclr-{peak_lr}_pool-4"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=peak_lr,
                max_pool=[1, 2, 2],
                lm_scales=[0.6],
                recog_final_epoch_only=False,
            )

    # compressed input
    # Batch WER  Min/Ep
    # 5000  4.7  14
    # 7500  4.8  12
    # 15000 4.8  10
    if False:
        for batch_size, accum_grad, n_steps_per_epoch in [
            (5000, 3, 3100),
            (7500, 2, 2000),
            (15000, 1, 1100),
        ]:
            name_suffix = f"oclr-{peak_lr}_pool-4_compress-joint_bs-{batch_size}"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=1e-05,
                max_pool=[1, 2, 2],
                lm_scales=[0.6],
                compressed_join=True,
                batch_size=batch_size,
                accum_grad=accum_grad,
                n_steps_per_epoch=n_steps_per_epoch,
                recog_final_epoch_only=False,
            )

    # Default parameters change here:
    # Viterbi model WER 4.3 -> 3.7, Dec-size 640 -> 800, Joint-size 1024 -> 600

    # Adapted learning rate schedule -> Const into decay
    if True:
        for const_lr in [5e-06, 8e-06, 1e-05, 3e-05, 5e-05, 7e-05, 9e-05, 1e-04, 3e-04]:
            name_suffix = f"const_decay-{const_lr}_pool-4"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.CONST_DECAY,
                compressed_join=True,
                batch_size=15000,
                accum_grad=1,
                n_steps_per_epoch=1100,
                const_lr=const_lr,
                max_pool=[1, 2, 2],
                lm_scales=[0.6],
                recog_final_epoch_only=False,
            )
