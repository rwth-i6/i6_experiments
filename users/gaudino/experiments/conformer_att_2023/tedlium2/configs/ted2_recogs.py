import copy, os

import numpy
from itertools import product
from string import Template
import numpy as np

import sisyphus.toolkit as tk

from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.attention_asr_config import (
    CTCDecoderArgs,
    create_config,
    ConformerEncoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ConformerDecoderArgs,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
    reset_params_init,
    apply_fairseq_init_to_transformer_decoder,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
    SCTK_BINARY_PATH,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import (
    log10_net_10ms,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.pipeline import (
    training,
    search,
    get_average_checkpoint,
    get_best_checkpoint,
    search_single,
)
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM

from i6_experiments.users.gaudino.experiments.conformer_att_2023.librispeech_960.search_helpers import (
    rescore_att_ctc_search,
)

from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.model_ckpt_info import (
    models,
)

from i6_core.returnn.training import Checkpoint


train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000
BPE_500 = 500

# train:
# ------
# Seq-length 'data' Stats:
#   92973 seqs
#   Mean: 819.1473868757647
#   Std dev: 434.7168733027807
#   Min/max: 26 / 2049

# --------------------------- LM --------------------------- #

# LM data (runnnig words)
# trans 2250417 ~ 2.25M
# external: 12688261 ~ 12.7M
# Total: 14.9M

lstm_10k_lm_opts = {
    "lm_subnet": generic_lm.libri_lstm_bpe10k_net,
    "lm_model": generic_lm.libri_lstm_bpe10k_model,
    "name": "lstm",
}

lstm_lm_opts_map = {
    BPE_10K: lstm_10k_lm_opts,
}

trafo_lm_net = TransformerLM(
    source="prev:output", num_layers=24, vocab_size=10025, use_as_ext_lm=True
)
trafo_lm_net.create_network()
trafo_10k_lm_opts = {
    "lm_subnet": trafo_lm_net.network.get_net(),
    "load_on_init_opts": {
        "filename": "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023",
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}

trafo_lm_opts_map = {
    BPE_10K: trafo_10k_lm_opts,
    # BPE_5K: trafo_5k_lm_opts,
}

prior_file = "work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt"

# ----------------------------------------------------------- #

def adjust_enc_args_to_model_name(enc_args, model_name):
    new_enc_args = copy.deepcopy(enc_args)
    if "enc_layer_w_ctc" in models[model_name].keys():
        new_enc_args.enc_layer_w_ctc = models[model_name]["enc_layer_w_ctc"]
    if "no_ctc" in models[model_name].keys():
        new_enc_args.with_ctc = not models[model_name]["no_ctc"]
    if "gauss_window" in models[model_name].keys():
        new_enc_args.ctc_att_weights_gauss = True
        new_enc_args.ctc_att_weights_gauss_stddev = models[model_name]["gauss_std"]
        new_enc_args.ctc_att_weights_gauss_window = models[model_name]["gauss_window"]
    if "use_enc" in models[model_name].keys():
        new_enc_args.ctc_att_weights_use_enc = models[model_name]["use_enc"]
    return new_enc_args

def otps_recogs_additonal_trainings():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def get_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev", "test"]:
            test_dataset_tuples[testset] = build_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
        return test_dataset_tuples

    def run_single_search(
        exp_name,
        train_data,
        search_args,
        checkpoint,
        feature_extraction_net,
        recog_dataset,
        recog_ref,
        recog_bliss,
        mem_rqmt=8,
        time_rqmt=4,
        two_pass_rescore=False,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )
        if two_pass_rescore:
            assert (
                "att_scale" in kwargs
                and "ctc_scale" in kwargs
                and "ctc_prior_scale" in kwargs
            ), "rescore requires scales."
            rescore_att_ctc_search(
                exp_prefix,
                returnn_search_config,
                checkpoint,
                recognition_dataset=recog_dataset,
                recognition_reference=recog_ref,
                recognition_bliss_corpus=recog_bliss,
                returnn_exe=RETURNN_CPU_EXE,
                returnn_root=RETURNN_ROOT,
                mem_rqmt=mem_rqmt,
                time_rqmt=time_rqmt,
                **kwargs,  # pass scales here
            )
        else:
            kwargs.pop("att_scale", None)
            kwargs.pop("ctc_scale", None)
            kwargs.pop("ctc_prior_scale", None)
            search_single(
                exp_prefix,
                returnn_search_config,
                checkpoint,
                recognition_dataset=recog_dataset,
                recognition_reference=recog_ref,
                recognition_bliss_corpus=recog_bliss,
                returnn_exe=RETURNN_CPU_EXE,
                returnn_root=RETURNN_ROOT,
                mem_rqmt=mem_rqmt,
                time_rqmt=time_rqmt,
                **kwargs,
            )

    def run_decoding(
        exp_name,
        train_data,
        checkpoint,
        search_args,
        bpe_size,
        test_sets: list,
        feature_extraction_net=log10_net_10ms,
        time_rqmt: float = 1.0,
        remove_label=None,
        two_pass_rescore=False,
        **kwargs,
    ):
        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)
        for test_set in test_sets:
            run_single_search(
                exp_name=exp_name + f"/recogs/{test_set}",
                train_data=train_data,
                search_args=search_args,
                checkpoint=checkpoint,
                feature_extraction_net=feature_extraction_net,
                recog_dataset=test_dataset_tuples[test_set][0],
                recog_ref=test_dataset_tuples[test_set][1],
                recog_bliss=test_dataset_tuples[test_set][2],
                time_rqmt=time_rqmt,
                remove_label=remove_label,
                two_pass_rescore=two_pass_rescore,
                **kwargs,
            )

    # TODO: check for late evaluation of jobs
    # def tune_decoding(
    #     exp_name_template: Template,
    #     train_data,
    #     checkpoint,
    #     search_args,
    #     bpe_size,
    #     test_sets: list,
    #     tune_param_dict: dict,
    #     feature_extraction_net=log10_net_10ms,
    #     time_rqmt: float = 1.0,
    #     remove_label=None,
    #     **kwargs,
    # ):
    #
    #     max_tuning_level = max(tune_param_dict.values()["level"])
    #
    #     for tune_level in range(max_tuning_level):
    #         if any([param_ranges["level"] == tune_level for param_ranges in tune_param_dict.values()]):
    #
    #             param_names = [key for key in tune_param_dict.keys() if tune_param_dict[key]["level"] == tune_level]
    #             param_ranges = []
    #             # np.arange(0.0, 1.0, 0.1)
    #
    #             for param_dict in [param_dict for param_dict in tune_param_dict.values() if param_dict["level"] == tune_level]:
    #                 param_ranges.append(np.arange(param_dict["start"], param_dict["end"], param_dict["step"]))
    #
    #             for param_comb in product(*param_ranges):
    #                 exp_name = exp_name_template.substitute(
    #                     tune_level=tune_level,
    #                     tune_params="_".join([str(param) for param in param_comb]),
    #                 )
    #                 run_decoding(
    #                     exp_name=exp_name,
    #                     train_data=train_data,
    #                     checkpoint=checkpoint,
    #                     search_args=search_args,
    #                     bpe_size=bpe_size,
    #                     test_sets=test_sets,
    #                     feature_extraction_net=feature_extraction_net,
    #                     time_rqmt= time_rqmt,
    #                     remove_label=remove_label,
    #                     **kwargs,
    #                 )

    def compute_ctc_prior(prior_exp_name, train_args, model_ckpt, bpe_size):
        exp_prefix = os.path.join(prefix_name, prior_exp_name)
        ctc_prior_train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,
            partition_epoch=1,
            seq_ordering="laplace:.1000",
        )
        returnn_config = create_config(
            training_datasets=ctc_prior_train_data,
            **train_args,
            feature_extraction_net=log10_net_10ms,
            with_pretrain=False,
        )
        returnn_config.config["network"]["output"] = {"class": "copy", "from": "ctc"}
        returnn_config.config["max_seq_length"] = -1
        from i6_core.returnn.extract_prior import ReturnnComputePriorJobV2

        prior_j = ReturnnComputePriorJobV2(
            model_checkpoint=model_ckpt,
            returnn_config=returnn_config,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
        )
        tk.register_output(
            exp_prefix + "/priors/ctc_prior_fix", prior_j.out_prior_txt_file
        )
        return prior_j.out_prior_txt_file

    # --------------------------- General Settings --------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12,
        input_layer="conv-6",
        att_num_heads=8,
        ff_dim=2048,
        enc_key_dim=512,
        conv_kernel_size=32,
        pos_enc="rel",
        dropout=0.1,
        att_dropout=0.1,
        l2=0.0001,
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6,
        embed_dropout=0.1,
        label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc="rel",
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    conformer_dec_args = ConformerDecoderArgs()
    apply_fairseq_init_to_conformer(conformer_dec_args)

    training_args = dict()

    # LR scheduling
    training_args["const_lr"] = [42, 100]  # use const LR during pretraining
    training_args["wup_start_lr"] = 0.0002
    training_args["wup"] = 20
    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 20000 * 160,
    }
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 12000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy(
        {
            **trafo_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": trafo_dec_args,
        }
    )

    conformer_dec_exp_args = copy.deepcopy(trafo_dec_exp_args)
    conformer_dec_exp_args["decoder_args"] = conformer_dec_args

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 22500 * 160,
    }
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    oclr_args = copy.deepcopy(lstm_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 9e-4,
        "final_lr": 1e-6,
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args["encoder_args"].use_sqrd_relu = True
    oclr_args["max_seq_length"] = None

    # add hardcoded paths because DelayedFormat breaks hashes otherwise
    # _, _, global_mean, global_std = compute_features_stats(output_dirname="logmel_80", feat_dim=80)
    global_mean = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/mean"
    )
    global_std = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/std_dev"
    )

    # --------------------- V1 ---------------------
    def get_base_v1_args(lr, ep, enc_drop=0.1, pretrain_reps=3, use_legacy_stats=True):
        #  base_bpe1000_peakLR0.0008_ep200_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.1_woDepthConvPre
        # Average ckpt: 8.19/7.64 (50 epochs)
        # - Epoch-based OCLR with peak LR 8e-4
        # - EncDrop 0.1, fixed zoneout
        # - Pretrain 3, no depthwise conv pretrain
        # - Feature global normalization

        base_v1_args = copy.deepcopy(oclr_args)
        base_v1_args.pop("oclr_opts")
        cyc_ep = int(0.45 * ep)
        # Epoch-based OCLR
        base_v1_args["learning_rates_list"] = (
            list(numpy.linspace(lr / 10, lr, cyc_ep))
            + list(numpy.linspace(lr, lr / 10, cyc_ep))
            + list(numpy.linspace(lr / 10, 1e-6, ep - 2 * cyc_ep))
        )
        base_v1_args["global_stats"] = {
            "mean": global_mean,
            "stddev": global_std,
            "use_legacy_version": use_legacy_stats,
        }
        base_v1_args["pretrain_reps"] = pretrain_reps
        base_v1_args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = [
            "conv_kernel_size"
        ]
        base_v1_args["encoder_args"].dropout = enc_drop
        base_v1_args["encoder_args"].dropout_in = enc_drop
        base_v1_args["encoder_args"].att_dropout = enc_drop
        base_v1_args["encoder_args"].num_blocks = 12
        base_v1_args["encoder_args"].mhsa_weight_dropout = 0.1
        base_v1_args["encoder_args"].ff_weight_dropout = 0.1
        base_v1_args["encoder_args"].conv_weight_dropout = 0.1

        base_v1_args["decoder_args"].use_zoneout_output = True
        base_v1_args["decoder_args"].embed_dim = 256
        base_v1_args["decoder_args"].att_dropout = 0.0

        exp_name = f"base_bpe1000_peakLR{lr}_ep{ep}_globalNorm_epochOCLR_pre{pretrain_reps}_fixZoneout_encDrop{enc_drop}_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12"
        return base_v1_args, exp_name

    ep = 100 * 4
    lr = 8e-4
    enc_drop = 0.15

    base_v1_args, exp_name = get_base_v1_args(lr, ep, enc_drop=enc_drop)
    args = copy.deepcopy(base_v1_args)

    def get_train_data(**kwargs):
        train_data = build_training_datasets(
            bpe_size=1000,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", None),
            link_speed_perturbation=kwargs.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
            partition_epoch=4,
            devtrain_subset=kwargs.get(
                "devtrain_subset", 507
            ),  # same as num of dev segments
        )
        return train_data

    train_data_baseline = get_train_data()

    # batch size
    # orig. batch size: 15000 * 160 -> bsf 120
    bsf = 10
    args["batch_size"] = bsf * 20000

    # att only decoding
    for model_name, beam_size in product(list(models.keys())[:-19], [12, 32]):
        # for model_name, beam_size, use_time_mask in product(["model_att_only_currL"], [12, 32], [False]):
        search_args = copy.deepcopy(args)
        search_args["encoder_args"] = adjust_enc_args_to_model_name(
            search_args["encoder_args"], model_name
        )
        search_args["beam_size"] = beam_size
        # if use_time_mask:
        #     search_args["encoder_args"].conv_use_time_mask = True

        run_decoding(
            # f"bsf{bsf}/" + model_name + f"/att_only_beam{beam_size}" + (f"_timeMask_2" if use_time_mask else ""),
            f"bsf{bsf}/" + model_name + f"/att_only_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    # ctc greedy decoding
    search_args = copy.deepcopy(args)
    search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)

    for model_name in []:  # list(models.keys())[:-3] + ["model_ctc_only"]:
        search_args = copy.deepcopy(args)
        search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)
        search_args["encoder_args"] = adjust_enc_args_to_model_name(
            search_args["encoder_args"], model_name
        )
        run_decoding(
            f"bsf{bsf}/" + model_name + f"/ctc_greedy",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    # ctc prior correction
    ctc_prior_model_names = {
        "model_baseline": {
            "prior_scale": [0.15],  # dev/test 8.39/8.01 -> 8.19/7.92
        },
        "model_ctc0.43_att1.0": {  # dev/test 8.62/7.97 -> 8.58/7.86
            "prior_scale": [0.15],
        },
        "model_ctc0.25_att1.0": {
            "prior_scale": [0.22],  # dev/test 9.03/8.32 -> 8.79/8.25
        },
        "model_ctc0.2_att1.0": {  # dev/test 9.56/8.67 -> 9.38/8.65
            "prior_scale": [0.2],
        },
        "model_ctc0.9_att0.1": {
            "prior_scale": [0.22],  # bsf 10 dev/test 9.04/8.33 -> 8.85/8.44
        },
        "model_ctc0.8_att0.2": {
            "prior_scale": [0.2],  # bsf 10 dev/test 9.03/8.24 -> 8.96/8.21
        },
        "model_ctc0.7_att0.3": {
            "prior_scale": [0.22],  # bsf 10 dev/test 8.67/8.00 -> 8.58/7.94
        },
        "model_ctc0.6_att0.4": {
            "prior_scale": [0.2],  # bsf 10 dev/test 8.65/8.04 -> 8.64/7.98
        },
        "model_ctc0.5_att0.5": {
            "prior_scale": [0.2],  # bsf 10 dev/test 8.50/8.03 -> 8.31/7.92
        },
        "model_ctc0.4_att0.6": {
            "prior_scale": [0.2],  # bsf 10 dev/test 8.55/7.76 -> 8.42/7.89
        },
        "model_ctc0.3_att0.7": {
            "prior_scale": [0.25],  # dev/test 8.58/8.15 -> 8.46/8.11
        },
        "model_ctc0.2_att0.8": {
            "prior_scale": [0.22],  # dev/test 9.05/8.35 -> 8.78/8.33
        },
        "model_ctc0.1_att0.9": {
            "prior_scale": [0.17],  # dev/test 9.92/9.22 -> 9.84/9.20
        },
        "model_ctc0.001_att0.999": {
            "prior_scale": [0.2],  # dev/test 27.00/25.10 -> 26.32/24.76
        },
        "model_ctc0.3_att0.7_lay6": {
            "prior_scale": [0.15],  # dev/test 10.75/9.83 -> 10.70/9.74
        },
        "model_ctc0.3_att0.7_lay8": {
            "prior_scale": [0.2],  # dev/test 9.68/9.11 -> 9.66/9.01
        },
        "model_ctc0.3_att0.7_lay10": {
            "prior_scale": [0.2],  # dev/test 9.26/8.44 -> 9.01/8.32
        },
        "model_ctc1.0_att1.0_lay6": {
            "prior_scale": [0.2],  # dev/test 10.34/9.51 -> 10.22/9.62
        },
        "model_ctc1.0_att1.0_lay8": {
            "prior_scale": [0.15],  # dev/test 9.53/8.73 -> 9.4/8.64
        },
        "model_ctc1.0_att1.0_lay10": {
            "prior_scale": [0.25],  # dev/test 9.38/8.56 -> 9.03/8.44
        },
        "model_ctc_only": {
            "prior_scale": [0.17],  # dev/test 9.27/8.46 -> 9.23/8.37
        },
        "model_ctc_only_gauss1.0_win5": { # dev/test 9.38/ -> 9.22/
            "prior_scale": [0.15],
        },
        "model_ctc_only_gauss1.0_win5_noEnc": { # dev/test 9.28/ -> 9.23/
            "prior_scale": [0.15],
        },
        "model_ctc_only_win1":{
            "prior_scale": [0.2],  # big jump ?
        },
        "model_ctc_only_gauss0.1_win5":{
            "prior_scale": [0.0, 0.13], # no improvement
        },
        "model_ctc_only_gauss0.5_win5": {
            "prior_scale": [0.17, 0.2],
        },
        "model_ctc_only_gauss2.0_win5": {
            "prior_scale": [0.2],
        },
        "model_ctc_only_gauss10.0_win5": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss1.0_win10": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss1.0_win20": {
            "prior_scale": [0.15],
        },
        "model_ctc_only_gauss1.0_win50": {
            "prior_scale": [0.17, 0.2],
        },
        "model_ctc_only_win1_noEnc": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss0.1_win5_noEnc": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss0.5_win5_noEnc": {
            "prior_scale": [0.15],
        },
        "model_ctc_only_gauss2.0_win5_noEnc": {
            "prior_scale": [0.15],
        },
        "model_ctc_only_gauss10.0_win5_noEnc": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss1.0_win10_noEnc": {
            "prior_scale": [0.17],
        },
        "model_ctc_only_gauss1.0_win20_noEnc": {
            "prior_scale": [0.15],
        },
        "model_ctc_only_gauss1.0_win50_noEnc": {
            "prior_scale": [0.15],
        },
    }

    for model_name in list(ctc_prior_model_names.keys())[-16:]:  # list(ctc_prior_model_names.keys())[4:10]:
        for prior_scale in ctc_prior_model_names[model_name]["prior_scale"]:
        # for prior_scale in [0.0, 0.1, 0.13, 0.15, 0.17, 0.2, 0.23]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            search_args["decoder_args"] = CTCDecoderArgs(
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                target_dim=1057,
            )
            run_decoding(
                f"bsf{bsf}/" + model_name + f"/ctc_greedy" +
                (f"_prior{prior_scale}" if prior_scale > 0 else "") + "_fixstd",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    from i6_experiments.users.gaudino.models.asr.lm.tedlium_lm import (
        tedlium_lm_net,
        tedlium_lm_model,
        tedlium_lm_load_on_init,
    )

    # rescoring att, ctc
    base_v2_args, exp_name = get_base_v1_args(
        lr, ep, enc_drop=enc_drop, use_legacy_stats=False
    )
    base_v2_args["batch_size"] = bsf * 20000
    for model_name, scales in product([], [(1, 0.008), (1, 0.01), (1, 0.1)]): # "model_ctc0.5_att0.5"
        att_scale, ctc_scale = scales
        prior_scale = 0.0
        beam_size = 12
        search_args = copy.deepcopy(base_v2_args)
        search_args["encoder_args"] = adjust_enc_args_to_model_name(
            search_args["encoder_args"], model_name
        )
        search_args["ctc_log_prior_file"] = models[model_name]["prior"]
        search_args["beam_size"] = beam_size
        run_decoding(
            f"bsf{bsf}/"
            + model_name
            + f"/two_pass_rescore_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior{prior_scale}" if prior_scale > 0 else "")
            + f"_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            ctc_prior_scale=prior_scale,
            two_pass_rescore=True,
        )

    # optsnr aed + ctc
    joint_training_model_names = {
        "model_baseline": {
            "scales": [(0.7, 0.3, 0.4)],
        },
        "model_ctc0.43_att1.0": {
            "scales": [(0.7, 0.3, 0.4)],
        },
        "model_ctc0.25_att1.0": {
            "scales": [(0.85, 0.15, 0.35)],
        },
        "model_ctc0.2_att1.0": {
            "scales": [(0.8, 0.2, 0.5)],
        },
        "model_ctc0.9_att0.1": {
            "scales": [(0.6, 0.4, 0.45)],  # bsf 10
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.8, 0.2, 0.55)],  # bsf 10
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.7, 0.3, 0.5)],  # bsf 10
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.7, 0.3, 0.45)],  # bsf 10
        },
        "model_ctc0.5_att0.5": {
            "scales": [(0.85, 0.15, 0.4)],  # bsf 10
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.75, 0.25, 0.55)],  # bsf 10
        },
        "model_ctc0.3_att0.7": {
            "scales": [(0.8, 0.2, 0.5)],
        },
        "model_ctc0.2_att0.8": {
            "scales": [(0.75, 0.25, 0.4)],
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.75, 0.25, 0.4)],
        },
        "model_ctc0.001_att0.999": {
            "scales": [],
        },
        "model_ctc0.3_att0.7_lay6": {
            "scales": [(0.85, 0.15, 0.3)],
        },
        "model_ctc0.3_att0.7_lay8": {
            "scales": [(0.85, 0.15, 0.55)],
        },
        "model_ctc0.3_att0.7_lay10": {
            "scales": [(0.8, 0.2, 0.45)],
        },
        "model_ctc1.0_att1.0_lay6": {
            "scales": [(0.8, 0.2, 0.3)],
        },
        "model_ctc1.0_att1.0_lay8": {
            "scales": [(0.9, 0.1, 0.45)],
        },
        "model_ctc1.0_att1.0_lay10": {
            "scales": [(0.9, 0.1, 0.2)],
        },
    }

    for model_name, beam_size in product(
        list(joint_training_model_names.keys())[4:10],
        [],
    ):
        for scales in joint_training_model_names[model_name]["scales"]:
            # for scales in joint_training_model_names[model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            att_scale, ctc_scale, prior_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
            )
            run_decoding(
                f"bsf{bsf}/"
                + model_name
                + f"/opts_ctc{ctc_scale}_att{att_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}",
                # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # ctc ts (with recombine) bsf 10
    # for model_name in list(models.keys())[:-3] + ["model_ctc_only"]:
    for model_name, beam_size in product(list(models.keys())[:-3], []):
        for prior_scale in ctc_prior_model_names[model_name]["prior_scale"] + [0]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            search_args["decoder_args"] = CTCDecoderArgs(
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                target_dim=1057,
                recombine=True,
            )
            run_decoding(
                f"bsf{bsf}/"
                + model_name
                + f"/ctc_ts"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # optsr (with recombine) + att bsf 10 (sum approx, omt 1.0, renorm after remove blank)
    dict_recombine = {
        # --tuning done--
        "model_baseline": {
            "scales": [(0.85, 0.15, 0.6)],
        },
        # "model_ctc0.9_att0.1": {
        #     "scales": [(0.65, 0.35, 0.75)],
        # },
        # "model_ctc0.8_att0.2": {
        #     "scales": [(0.75, 0.25, 0.75)],
        # },
        # "model_ctc0.7_att0.3": {
        #     "scales": [(0.7, 0.3, 0.45)],
        # },
        # "model_ctc0.6_att0.4": {
        #     "scales": [(0.8, 0.2, 0.45)],
        # },
        # "model_ctc0.5_att0.5": {
        #     "scales": [(0.85, 0.15, 0.75)],
        # },
        # "model_ctc0.4_att0.6": {
        #     "scales": [(0.75, 0.25, 0.35)],
        # },
        # "model_ctc0.3_att0.7": {
        #     "scales": [(0.65, 0.35, 0.6)],
        # },
        # "model_ctc0.2_att0.8": {
        #     "scales": [(0.85, 0.15, 0.55)],
        # },
        # "model_ctc0.1_att0.9": {
        #     "scales": [(0.9, 0.1, 0.3)],
        # },
        # "model_ctc0.001_att0.999": {
        #     "scales": [(0.9, 0.1, 0.0)],
        # },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [(0.9, 0.1, 0.5)],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [(0.95, 0.05, 0.45)],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [(0.97, 0.03, 0.4)],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales":[(0.97, 0.03, 0.5)],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales":[(0.85, 0.15, 0.6)],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales":[(0.97, 0.03, 0.45)],
        # },
        # ----
        # "model_ctc0.43_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [],
        # },
    }
    for model_name, beam_size, omt_scale, length_norm_scale in product(
        dict_recombine.keys(), [32], [1.0], [0.0]
    ):
        for scales in dict_recombine[model_name]["scales"]:
            # for scales in joint_training_model_names[model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            att_scale, ctc_scale, prior_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                recombine=True,
                one_minus_term_mul_scale=omt_scale,
                length_normalization=length_norm_scale > 0,
                length_normalization_scale=length_norm_scale,
            )
            run_decoding(
                f"bsf{bsf}/"
                + model_name
                + f"/optsr_ctc{ctc_scale}_att{att_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + (f"_lenNorm{length_norm_scale}" if length_norm_scale > 0 else "")
                + f"_omt{omt_scale}"
                + f"_beam{beam_size}_3",
                # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )


    # optsr 2.0 (max approx, vanilla, label scale)
    dict_recombine_optsr_max = {
        # --tuning done--
        # "model_baseline": {
        #     "scales": [(0.8, 0.2, 0.5, 0.9)],
        # },
        # ----
        "model_ctc0.9_att0.1": {
            "scales": [(0.7,0.3, 0.6, 1.0)],
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.75, 0.25, 0.3, 0.8), (0.75, 0.25, 0.55, 1.0)],
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.7, 0.3, 0.6, 1.0)],
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.7, 0.3, 0.4, 0.95)],
        },
        "model_ctc0.5_att0.5": {
            "scales": [(0.7, 0.3, 0.45, 1.0)],
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.75, 0.25, 0.4, 0.9), (0.75, 0.25, 0.5, 0.8), (0.75, 0.25, 0.5, 0.99)],
        },
        "model_ctc0.3_att0.7": {
            "scales": [(0.75, 0.25, 0.4, 0.8), (0.75, 0.25, 0.45, 0.9)],
        },
        "model_ctc0.2_att0.8": {
            "scales": [(0.85, 0.15, 0.4, 1.0), (0.8, 0.2, 0.3, 0.8)],
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.9, 0.1, 0.0, 0.95)],
        },
        "model_ctc0.001_att0.999": {
            "scales": [(0.9, 0.1, 0.0, 0.95)],
        },
        "model_ctc0.3_att0.7_lay6": {
            "scales": [(0.85, 0.15, 0.5, 1.0)],
        },
        "model_ctc0.3_att0.7_lay8": {
            "scales": [(0.9, 0.1, 0.4, 1.0)],
        },
        "model_ctc0.3_att0.7_lay10": {
            "scales": [(0.9, 0.1, 0.4, 1.0), (0.9, 0.1, 0.45, 0.9), (0.9, 0.1, 0.5, 0.9)],
        },
        "model_ctc1.0_att1.0_lay6": {
            "scales":[(0.85, 0.15, 0.4, 0.95)],
        },
        "model_ctc1.0_att1.0_lay8": {
            "scales":[(0.75, 0.25, 0.5, 1.0)],
        },
        "model_ctc1.0_att1.0_lay10": {
            "scales":[(0.95, 0.05, 0.0, 0.99)],
        },
        # "model_ctc0.43_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [],
        # },
    }

    for model_name, prior_scale, beam_size in product(
        ["model_baseline", "model_ctc0.5_att0.5", "model_ctc0.2_att0.8", "model_ctc_only"],
        # dict_recombine_optsr_max.keys(),
        [0.0, 0.05 ,0.1],
        [12, 32],
    ):
        for scales in [(0.0, 1.0)]:
        # for scales in dict_recombine_optsr_max[model_name]["scales"]:
            # for scales in joint_training_model_names[model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            label_scale = 1.0
            att_scale, ctc_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                recombine=True,
                max_approx=True,
                # normalization settings
                one_minus_term_mul_scale=0.0,
                renorm_after_remove_blank=False,
                blank_prob_scale=1.0,
                repeat_prob_scale=1.0,
                label_prob_scale=label_scale,
            )
            run_decoding(
                f"bsf{bsf}/"
                + model_name
                + f"/optsr_max_ctc{ctc_scale}"
                + (f"_att{att_scale}" if att_scale > 0.0 else "")
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + (f"_scales_l{label_scale}" if label_scale != 1.0 else "")
                + f"_vanilla"
                # + (f"_lenNorm{length_norm_scale}" if length_norm_scale > 0 else "")
                # + f"_omt{0.0}"
                + f"_beam{beam_size}",
                # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # optsr max approx + lenNorm fix 3
    for model_name, beam_size, label_scale, prior_scale, length_norm_scale in product(
        ["model_baseline"],
        [12],
        [1.0],
        [0.0, 0.5],
        [0.1, 0.2, 0.5, 0.8, 1.0],
    ):
        for scales in [(0.8, 0.2)]:
            # for scales in joint_training_model_names[model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            att_scale, ctc_scale = scales
            blank_scale = 1.0
            repeat_scale = 1.0
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                recombine=True,
                max_approx=True,
                # normalization settings
                one_minus_term_mul_scale=0.0,
                renorm_after_remove_blank=False,
                blank_prob_scale=blank_scale,
                repeat_prob_scale=repeat_scale,
                label_prob_scale=label_scale,
                # length_normalization=True,
                length_normalization=length_norm_scale > 0,
                length_normalization_scale=length_norm_scale,
            )
            run_decoding(
                f"bsf{bsf}/"
                + model_name
                + f"/optsr_max_ctc{ctc_scale}_att{att_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + (f"_scales_l{label_scale}_b{blank_scale}_r{repeat_scale}" if label_scale != 1.0 or blank_scale != 1.0 or repeat_scale != 1.0 else "")
                + f"_vanilla"
                + (f"_lenNorm{length_norm_scale}_fix_3" if length_norm_scale > 0 else "")
                # + f"_omt{0.0}"
                + f"_beam{beam_size}",
                # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # ------------------------- Separate Encoder (system combination) -------------------------
    # optsnr some model + ctc only

    joint_training_model_names_2 = {
        "model_baseline": {
            "scales": [(0.85, 0.15, 0.3)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.43_att1.0": {
            "scales": [(0.8, 0.2, 0.4)],
            "beam_sizes": [32, 70],
        },
        "model_ctc0.25_att1.0": {
            "scales": [(0.8, 0.2, 0.45), (0.8, 0.2, 0.5), (0.8, 0.2, 0.55)],
            "beam_sizes": [32, 64, 70],
        },
        "model_ctc0.2_att1.0": {
            "scales": [(0.75, 0.25, 0.45)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.9_att0.1": {
            "scales": [(0.75, 0.25, 0.7)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.8, 0.2, 0.6)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.8, 0.2, 0.5)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.75, 0.25, 0.45)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.5_att0.5": {
            "scales": [(0.8, 0.2, 0.45)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.8, 0.2, 0.4), (0.8, 0.2, 0.5)],  # bsf 10
            "beam_sizes": [],
        },
        "model_ctc0.3_att0.7": {
            "scales": [(0.8, 0.2, 0.6)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.2_att0.8": {
            "scales": [(0.8, 0.2, 0.55)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.8, 0.2, 0.6)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.001_att0.999": {
            "scales": [(0.65, 0.35, 0.5)],
            "beam_sizes": [32, 64],
        },
        "model_att_only_adjSpec": {
            "scales": [(0.75, 0.25, 0.6)],
            "beam_sizes": [32, 70],
        },
        "model_att_only_currL": {
            "scales": [(0.75, 0.25, 0.55)],
            "beam_sizes": [32, 64, 70],
        },
        "model_ctc0.3_att0.7_lay6": {
            "scales": [(0.8, 0.2, 0.6)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.3_att0.7_lay8": {
            "scales": [(0.75, 0.25, 0.65)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.3_att0.7_lay10": {
            "scales": [(0.75, 0.25, 0.65)],
            "beam_sizes": [32, 64],
        },
        "model_ctc1.0_att1.0_lay6": {
            "scales": [(0.65, 0.35, 0.65)],
            "beam_sizes": [32, 64],
        },
        "model_ctc1.0_att1.0_lay8": {
            "scales": [(0.75, 0.25, 0.75)],
            "beam_sizes": [32, 64],
        },
        "model_ctc1.0_att1.0_lay10": {
            "scales": [(0.6, 0.4, 0.7)],
            "beam_sizes": [32, 64],
        },
    }

    second_model_name = "model_ctc_only"

    for first_model_name in []:  # list(joint_training_model_names_2.keys())[4:10]:
        # for beam_size, scales in product(
        #     joint_training_model_names_2[first_model_name]["beam_sizes"],
        #     joint_training_model_names_2[first_model_name]["scales"],
        # ):
        for beam_size, scales in product(
            [32, 64],
            joint_training_model_names_2[first_model_name]["scales"],
        ):
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], first_model_name
            )
            search_args["second_encoder_args_update_dict"] = {
                "enc_layer_w_ctc": None,
                "with_ctc": True,
            }
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models["model_ctc_only"]["prior"]
            att_scale, ctc_scale, prior_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
            )
            search_args[
                "second_encoder_ckpt"
            ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
            # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
            run_decoding(
                f"bsf{bsf}/" + first_model_name
                # + f"__ctc_only/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                + f"__ctc_only/opts_ctc{ctc_scale}_att{att_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[first_model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
                time_rqmt=2.0,
            )

    # optsr max some model + ctc only bsf 10
    dict_sep_recombine = {
        # --tuning done--
        "model_baseline": {
            "scales": [(0.9, 0.1, 0.1, 1.0)],
            "beam_sizes": [32],
        },
        "model_ctc0.9_att0.1": {
            "scales": [(0.7, 0.3, 0.5, 0.9), (0.7, 0.3, 0.5, 1.0)],
            "beam_sizes": [],
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.8, 0.2, 0.4, 1.0)],
            "beam_sizes": [],
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.8, 0.2, 0.5, 1.0)],
            "beam_sizes": [],
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.8 ,0.2, 0.4, 1.0)],
            "beam_sizes": [],
        },
        "model_ctc0.5_att0.5": {
            "scales": [(0.85, 0.15, 0.3, 1.0)],
            "beam_sizes": [],
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.8, 0.2, 0.2, 0.9)],
            "beam_sizes": [],
        },
        "model_ctc0.3_att0.7": {
            "scales": [(0.8, 0.2, 0.5, 0.9)], # keep all
            "beam_sizes": [32, 64],
        },
        "model_ctc0.2_att0.8": {
            "scales": [ (0.9, 0.1, 0.3, 0.9)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.75, 0.25, 0.5, 1.0)],
            "beam_sizes": [32, 64],
        },
        "model_ctc0.001_att0.999": {
            "scales": [(0.8, 0.2, 0.5, 0.9)],
            "beam_sizes": [32, 64],
        },
        # "model_att_only_adjSpec": {
        #     "scales": [(0.7, 0.3, 0.65)],
        #     "beam_sizes": [32, 70],
        # },
        "model_att_only_currL": {
            "scales": [(0.7, 0.3, 0.3, 1.0)],
        },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [(0.8, 0.2, 0.55)],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [(0.8, 0.2, 0.8)],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [(0.7, 0.3, 0.7)],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales": [(0.7, 0.3, 0.65)],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales":[(0.75, 0.25, 0.75)],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales":[(0.75, 0.25, 0.7), (0.75, 0.25, 0.75)],
        # },
        # ----
        # "model_ctc0.43_att1.0": {
        #     "scales": [(0.8, 0.2, 0.4)],
        #     "beam_sizes": [32, 70],
        # },
        # "model_ctc0.25_att1.0": {
        #     "scales": [(0.8, 0.2, 0.45), (0.8, 0.2, 0.5), (0.8, 0.2, 0.55)],
        #     "beam_sizes": [32, 64, 70],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [(0.75, 0.25, 0.45)],
        #     "beam_sizes": [32, 64],
        # },
    }

    second_model_name = "model_ctc_only"

    for first_model_name in ["model_att_only_currL"]:
    # for first_model_name in list(dict_sep_recombine.keys()):
        # for beam_size, scales in product(
        #     joint_training_model_names_2[first_model_name]["beam_sizes"],
        #     joint_training_model_names_2[first_model_name]["scales"],
        # ):
        for beam_size, scales in product(
            [32],
            dict_sep_recombine[first_model_name]["scales"],
        ):
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], first_model_name
            )
            search_args["second_encoder_args_update_dict"] = {
                "enc_layer_w_ctc": None,
                "with_ctc": True,
            }
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models["model_ctc_only"]["prior"]
            att_scale, ctc_scale,prior_scale,_= scales
            label_scale = 1.0

            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=prior_scale > 0,
                prior_scale=prior_scale,
                recombine=True,
                max_approx=True,
                # normalization settings
                one_minus_term_mul_scale=0.0,
                renorm_after_remove_blank=False,
                blank_prob_scale=1.0,
                repeat_prob_scale=1.0,
                label_prob_scale=label_scale,
            )
            search_args[
                "second_encoder_ckpt"
            ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
            # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
            run_decoding(
                f"bsf{bsf}/" + first_model_name
                # + f"__ctc_only/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                + f"__ctc_only/optsr_max_ctc{ctc_scale}_att{att_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + (f"_scales_l{label_scale}" if label_scale != 1.0 else "")
                + f"_vanilla"
                + f"_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[first_model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
                time_rqmt=2.0,
            )
