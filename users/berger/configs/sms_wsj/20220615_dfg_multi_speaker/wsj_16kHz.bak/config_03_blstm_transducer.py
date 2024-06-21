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

ctc_model = "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/rasr_training/ReturnnRasrTrainingJob.nSDi7F07VTlp/output/models/epoch.300.index"
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
        # "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
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
    train_blstm_net, train_python_code = make_context_1_blstm_transducer_blank(
        num_outputs=num_classes,
        encoder_loss=kwargs.get("encoder_loss", True),
        loss_boost_scale=kwargs.get("loss_boost_scale", 5.0),
        specaug_args={
            "max_time_num": kwargs.get("time_num", 1),
            "max_time": kwargs.get("max_time", 15),
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
            "combination_mode": kwargs.get("combination_mode", "add"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 1),
                "size": 800,
                "l2": l2,
                "dropout": dropout,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 0),
                "size": 600,
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
            "combination_mode": kwargs.get("combination_mode", "add"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 1),
                "size": 800,
            },
            "joint_mlp_args": {
                "num_layers": kwargs.get("num_joint_layers", 0),
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
            "grad_noise": kwargs.get("grad_noise", 0.1),
            "grad_clip": kwargs.get("grad_clip", 100.0),
            "schedule": kwargs.get("schedule", LearningRateSchedules.OCLR),
            "peak_lr": kwargs.get("peak_lr", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 1e-03),
            "min_learning_rate": 1e-05,
            "n_steps_per_epoch": 1410,
            "base_chunk_size": 256,
            "chunking_factors": {"data": 1, "classes": red_fact},
            "extra_config": {
                "pretrain": (
                    {
                        "repetitions": 6,
                        "construction_algo": CodeWrapper("pretrain_construction_algo"),
                    }
                    if kwargs.get("pretrain", False)
                    else None
                ),
                "preload_from_files": (
                    {
                        "base": {
                            "init_for_train": True,
                            "ignore_missing": True,
                            "filename": ctc_model,
                        }
                    }
                    if kwargs.get("ctc_init", False)
                    else None
                ),
                "train": {"reduce_target_factor": red_fact},
                "dev": {"reduce_target_factor": red_fact},
            },
            "python_prolog": [pretrain_construction_algo] if kwargs.get("pretrain", False) else None,
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
    # Default: Boost = True, Pretrain = False
    # Boost \ Pretrain  True    False
    #   True            4.3     4.3
    #   False           4.7     4.3
    if False:
        for pretrain in [True, False]:
            for loss_boost in [True, False]:
                name_suffix = f"oclr-0.0008_pool-4"
                if loss_boost:
                    name_suffix += "_boost"
                if pretrain:
                    name_suffix += "_pretrain"
                run_exp(
                    name_suffix=name_suffix,
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=0.0008,
                    max_pool=[1, 2, 2],
                    pretrain=pretrain,
                    loss_boost_scale=5.0 * float(loss_boost),
                    lm_scales=[0.6],
                    recog_final_epoch_only=True,
                )

    # Boost \ Pretrain  True    False
    #   True            4.5     3.9
    #   False           4.4     4.6
    if False:
        for pretrain in [True, False]:
            for loss_boost in [True, False]:
                name_suffix = f"oclr-0.0008_pool-2"
                if loss_boost:
                    name_suffix += "_boost"
                if pretrain:
                    name_suffix += "_pretrain"
                run_exp(
                    name_suffix=name_suffix,
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=0.0008,
                    max_pool=[1, 1, 2],
                    pretrain=pretrain,
                    loss_boost_scale=5.0 * float(loss_boost),
                    lm_scales=[0.6],
                    recog_final_epoch_only=True,
                )

    # OCLR
    # 4.3 -> No difference
    if False:
        name_suffix = f"oclr-0.0008_pool-4_ctc-init_dec-add-1x800"
        run_exp(
            name_suffix=name_suffix,
            schedule=LearningRateSchedules.OCLR,
            peak_lr=0.0008,
            lm_scales=[0.6],
            recog_final_epoch_only=False,
            ctc_init=True,
        )

    # Comb-mode Dec-L   Joint-L WER
    # Add       1       0       4.1
    #                   1       3.9
    #           2       0       4.1
    #                   1       4.1
    # Concat    1       0       4.1
    #                   1       3.8
    #           2       0       4.0
    #                   1       3.7 <- Best
    # Multiply  1       0       4.0
    #                   1       4.6
    #           2       0       3.9
    #                   1       4.0
    if False:
        for num_dec_layers in [2]:
            for num_joint_layers in [1]:
                for combination_mode in ["concat"]:
                    name_suffix = f"oclr-0.0008_pool-4_dec-{combination_mode}-{num_dec_layers}x800"
                    if num_joint_layers > 0:
                        name_suffix += f"_joint-{num_joint_layers}x600"
                    run_exp(
                        name_suffix=name_suffix,
                        schedule=LearningRateSchedules.OCLR,
                        peak_lr=0.0008,
                        lm_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        num_dec_layers=num_dec_layers,
                        num_joint_layers=num_joint_layers,
                        combination_mode=combination_mode,
                        recog_final_epoch_only=True,
                    )

    # Longer training
    # Peak LR   WER
    # 6e-04     4.4
    # 8e-04     4.0
    # 1e-03     3.9
    # => No improvement
    if False:
        for peak_lr in [6e-04, 8e-04, 1e-03]:
            name_suffix = f"oclr-{peak_lr}_pool-4_600-epochs"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=peak_lr,
                lm_scales=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                num_subepochs=600,
                num_dec_layers=2,
                num_joint_layers=1,
                combination_mode="concat",
            )

    # Specaug settings
    # TN    MT  WER
    # 1     15  3.5
    # 2     15  3.7
    # 3     15  3.3 <- Best
    # 4     15  3.7
    # 5     15  3.5
    # 2     10  3.9
    # 3     10  3.9
    # 4     10  3.8
    # 5     10  4.2
    # 2     20  4.0
    # 3     20  3.7
    if True:
        for time_num, max_time in [(3, 15)]:
            name_suffix = f"oclr-1e-03_pool-4_specaug-tn-{time_num}_mt-{max_time}"
            run_exp(
                name_suffix=name_suffix,
                schedule=LearningRateSchedules.OCLR,
                peak_lr=1e-03,
                lm_scales=[0.5, 0.6, 0.7, 0.8, 0.9],
                time_num=time_num,
                max_time=max_time,
                num_dec_layers=2,
                num_joint_layers=1,
                combination_mode="concat",
                recog_final_only=True,
            )
