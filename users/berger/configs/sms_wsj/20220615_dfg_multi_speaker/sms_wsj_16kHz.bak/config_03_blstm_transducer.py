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

train_key = "sms_train_si284"
dev_key = "sms_cv_dev93"
test_key = "sms_test_eval92"

cv_segments = tk.Path("/work/asr4/berger/dependencies/sms_wsj/segments/sms_cv_dev93_16kHz.reduced")

ctc_model = "/work/asr4/berger/sisyphus_work_dirs/sms_wsj/20220615_dfg_multi_speaker/i6_core/returnn/rasr_training/ReturnnRasrTrainingJob.nSDi7F07VTlp/output/models/epoch.300.index"
train_alignment = tk.Path(
    "/work/asr4/berger/dependencies/sms_wsj/alignment/8kHz/sms_train_si284_speechsource_gmm/alignment.cache.bundle"
)
dev_alignment = tk.Path(
    "/work/asr4/berger/dependencies/sms_wsj/alignment/8kHz/sms_cv_dev93_speechsource_gmm/alignment.cache.bundle"
)

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
            "size": kwargs.get("enc_layer_size", 400),
        },
        decoder_args={
            "combination_mode": kwargs.get("combination_mode", "add"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 1),
                "size": kwargs.get("dec_layer_size", 800),
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
        output_args={
            "label_smoothing": kwargs.get("label_smoothing", 0.2),
            "focal_loss_factor": kwargs.get("focal_loss", 2.0),
        },
    )

    train_networks[name] = train_blstm_net

    recog_blstm_net, recog_python_code = make_context_1_blstm_transducer_recog(
        num_outputs=num_classes,
        blstm_args={
            "max_pool": max_pool,
            "size": kwargs.get("enc_layer_size", 400),
        },
        decoder_args={
            "combination_mode": kwargs.get("combination_mode", "add"),
            "dec_mlp_args": {
                "num_layers": kwargs.get("num_dec_layers", 1),
                "size": kwargs.get("dec_layer_size", 800),
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
                "pretrain": {
                    "repetitions": 6,
                    "construction_algo": CodeWrapper("pretrain_construction_algo"),
                }
                if kwargs.get("pretrain", False)
                else None,
                "preload_from_files": {
                    "base": {
                        "init_for_train": True,
                        "ignore_missing": True,
                        "filename": ctc_model,
                    }
                }
                if kwargs.get("ctc_init", False)
                else None,
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
        # test_recog_args={
        #     "lm_scales": kwargs.get("lm_scales", [0.6]),
        #     "prior_scales": kwargs.get("prior_scales", [0.0]),
        # },
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
    if False:
        for max_pool in [2]:
            for time_num in [1, 2, 3]:
                run_exp(
                    name_suffix=f"specaug-tn-{time_num}_mt-15_pool-{max_pool}",
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=1e-03,
                    lm_scales=[0.9],
                    max_pool=[1, 1, 2] if max_pool == 2 else [1, 2, 2],
                    time_num=time_num,
                    max_time=15,
                    num_dec_layers=2,
                    num_joint_layers=1,
                    combination_mode="concat",
                    recog_final_epoch_only=False,
                )

    if False:
        for peak_lr in [3e-04, 7e-04, 1e-03]:
            run_exp(
                name_suffix=f"peak_lr-{peak_lr}",
                schedule=LearningRateSchedules.OCLR,
                peak_lr=peak_lr,
                lm_scales=[0.9],
                max_pool=[1, 1, 2],
                time_num=2,
                max_time=15,
                num_dec_layers=1,
                num_joint_layers=1,
                dropout=0.2,
                gradient_clip=15.0,
                combination_mode="concat",
                recog_final_epoch_only=False,
            )

    # LR \ Drop 0.1     0.2
    # 5e-04     21.5    20.3
    # 6e-04     21.6    19.6
    # 7e-04     21.4    20.1
    # 8e-04     21.6    20.8
    # 9e-04     20.5    20.6
    if True:
        for peak_lr in [5e-04, 6e-04, 7e-04, 8e-04, 9e-04]:
            for dropout in [0.1, 0.2]:
                run_exp(
                    name_suffix=f"peak_lr-{peak_lr}_drop-{dropout}",
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=peak_lr,
                    lm_scales=[0.9],
                    max_pool=[1, 1, 2],
                    time_num=2,
                    max_time=15,
                    num_dec_layers=1,
                    num_joint_layers=1,
                    dropout=dropout,
                    gradient_clip=100.0,
                    combination_mode="concat",
                    recog_final_epoch_only=False,
                )

    if True:
        # LS \ FL   0.0     1.0     2.0
        # 0.0       19.4    19.2    20.0
        # 0.1       19.1    20.8    20.5
        # 0.2       18.8    19.7    19.6
        for label_smoothing in [0.0, 0.1, 0.2]:
            for focal_loss in [0.0, 1.0, 2.0]:
                name_suffix = f"ls-{label_smoothing}_fl-{focal_loss}"

                run_exp(
                    name_suffix=name_suffix,
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=6e-04,
                    lm_scales=[0.9],
                    max_pool=[1, 1, 2],
                    time_num=2,
                    max_time=15,
                    num_dec_layers=1,
                    num_joint_layers=1,
                    dropout=0.2,
                    gradient_clip=100.0,
                    label_smoothing=label_smoothing,
                    focal_loss=focal_loss,
                    combination_mode="concat",
                    recog_final_epoch_only=False,
                )

    # Add:
    # Enc   WER
    # 400   20.6
    # 500   20.9
    # 600   21.1
    # Concat:
    # Enc \ Dec 400     800
    # 400       19.5    19.6
    # 500       19.6    20.6
    # 600       20.7    20.3
    if True:
        for combination_mode in ["add", "concat"]:
            for enc_size in [400, 500, 600]:
                for dec_size in [400, 800]:
                    if combination_mode == "add":
                        dec_size = 2 * enc_size
                    run_exp(
                        name_suffix=f"enc-{enc_size}_dec-{dec_size}_comb-{combination_mode}",
                        schedule=LearningRateSchedules.OCLR,
                        peak_lr=6e-04,
                        lm_scales=[0.9],
                        max_pool=[1, 1, 2],
                        time_num=2,
                        max_time=15,
                        enc_layer_size=enc_size,
                        dec_layer_size=dec_size,
                        num_dec_layers=1,
                        num_joint_layers=1,
                        dropout=0.2,
                        gradient_clip=100.0,
                        combination_mode=combination_mode,
                        recog_final_epoch_only=False,
                    )

    if True:
        run_exp(
            schedule=LearningRateSchedules.OCLR,
            peak_lr=6e-04,
            lm_scales=[0.9],
            max_pool=[1, 1, 2],
            time_num=2,
            max_time=15,
            enc_layer_size=400,
            dec_layer_size=400,
            num_dec_layers=1,
            num_joint_layers=1,
            dropout=0.2,
            gradient_clip=100.0,
            label_smoothing=0.2,
            focal_loss=0.0,
            combination_mode="concat",
            recog_final_epoch_only=False,
        )

    if True:
        for grad_noise in [0.0, 0.1, 0.2]:
            for pretrain in [True, False]:
                run_exp(
                    name_suffix=f"gn-{grad_noise}{'_pretrain' if pretrain else ''}",
                    schedule=LearningRateSchedules.OCLR,
                    peak_lr=6e-04,
                    lm_scales=[0.9],
                    max_pool=[1, 1, 2],
                    time_num=2,
                    max_time=15,
                    enc_layer_size=400,
                    dec_layer_size=400,
                    num_dec_layers=1,
                    num_joint_layers=1,
                    dropout=0.2,
                    grad_noise=grad_noise,
                    pretrain=pretrain,
                    gradient_clip=100.0,
                    label_smoothing=0.2,
                    focal_loss=0.0,
                    combination_mode="concat",
                    recog_final_epoch_only=False,
                )
