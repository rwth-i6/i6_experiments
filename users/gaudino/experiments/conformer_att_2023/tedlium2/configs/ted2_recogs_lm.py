import copy, os

import numpy
from itertools import product

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

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.model_ckpt_info import models


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

# ----------------------------------------------------------- #


def ted2_recogs_lm():
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
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )
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
                # two_pass_rescore=two_pass_rescore,
                **kwargs,
            )

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

    from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.configs.ted2_recogs import (
        adjust_enc_args_to_model_name,
    )

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

    from i6_experiments.users.gaudino.models.asr.lm.tedlium_lm import (
        tedlium_lm_net,
        tedlium_lm_model,
        tedlium_lm_load_on_init,
    )

    train_data_baseline = get_train_data()

    bsf = 10
    args["batch_size"] = bsf * 20000

    # att + trafo lm
    for model_name, lm_scale, beam_size in product(
        ["model_ctc0.5_att0.5"], [0.25, 0.3, 0.35], [12]
    ):
        # for model_name, lm_scale, beam_size in product(list(models.keys())[:-1], [0.1, 0.2, 0.3, 0.4], [12]):
        search_args = copy.deepcopy(args)
        search_args["encoder_args"] = adjust_enc_args_to_model_name(
            search_args["encoder_args"], model_name
        )
        search_args["ext_lm_opts"] = {
            "lm_scale": lm_scale,
            "lm_subnet": tedlium_lm_net,
            "load_on_init_opts": tedlium_lm_load_on_init,
            "name": "trafo",
        }
        search_args["beam_size"] = beam_size

        run_decoding(
            f"bsf{bsf}/" + model_name + f"/att1.0_trafolm{lm_scale}_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
            time_rqmt=2.0,
        )

    # ctc + trafo lm
    ctc_prior_model_names = {
        "model_baseline": {
            "prior_scale": [0.15],  # dev/test 8.39/8.01 -> 8.19/7.92
            "lm_scale": [0.4],
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
            "lm_scale": [0.45],
        },
        "model_ctc0.4_att0.6": {
            "prior_scale": [0.2],  # bsf 10 dev/test 8.55/7.76 -> 8.42/7.89
            "lm_scale": [0.45],
        },
        "model_ctc0.3_att0.7": {
            "prior_scale": [0.25],  # dev/test 8.58/8.15 -> 8.46/8.11
            "lm_scale": [0.5, 0.6],
        },
        "model_ctc0.2_att0.8": {
            "prior_scale": [0.22],  # dev/test 9.05/8.35 -> 8.78/8.33
            "lm_scale": [0.5],
        },
        "model_ctc0.1_att0.9": {
            "prior_scale": [0.17],  # dev/test 9.92/9.22 -> 9.84/9.20
            "lm_scale": [0.45],
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
            "lm_scale": [0.4],
        },
        "model_ctc_only_gauss1.0_win5": {
            "prior_scale": [0.15],
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss1.0_win5_noEnc": {
            "prior_scale": [0.15],
            "lm_scale": [0.35],
        },
        "model_ctc_only_win1": {
            "prior_scale": [0.2],  # big jump ?
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss0.1_win5": {
            "prior_scale": [0.13],  # no improvement
            "lm_scale": [0.35],
        },
        "model_ctc_only_gauss0.5_win5": {
            "prior_scale": [0.2],
            "lm_scale": [0.45],
        },
        "model_ctc_only_gauss2.0_win5": {
            "prior_scale": [0.2],
            "lm_scale": [0.5],
        },
        "model_ctc_only_gauss10.0_win5": {
            "prior_scale": [0.17],
            "lm_scale": [0.6],
        },
        "model_ctc_only_gauss1.0_win10": {
            "prior_scale": [0.17],
            "lm_scale": [0.6],
        },
        "model_ctc_only_gauss1.0_win20": {
            "prior_scale": [0.15],
            "lm_scale": [0.45],
        },
        "model_ctc_only_gauss1.0_win50": {
            "prior_scale": [0.2],
            "lm_scale": [0.55],
        },
        "model_ctc_only_win1_noEnc": {
            "prior_scale": [0.17],
            "lm_scale": [0.5],
        },
        "model_ctc_only_gauss0.1_win5_noEnc": {
            "prior_scale": [0.17],
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss0.5_win5_noEnc": {
            "prior_scale": [0.15],
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss2.0_win5_noEnc": {
            "prior_scale": [0.15],
            "lm_scale": [0.5],
        },
        "model_ctc_only_gauss10.0_win5_noEnc": {
            "prior_scale": [0.17],
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss1.0_win10_noEnc": {
            "prior_scale": [0.17],
            "lm_scale": [0.55],
        },
        "model_ctc_only_gauss1.0_win20_noEnc": {
            "prior_scale": [0.15],
            "lm_scale": [0.45],
        },
        "model_ctc_only_gauss1.0_win50_noEnc": {
            "prior_scale": [0.15],
            "lm_scale": [0.55],
        },
    }

    for model_name in list(ctc_prior_model_names.keys())[-16:]:
        # for model_name in ctc_prior_model_names.keys():
        for prior_scale, lm_scale, beam_size in product(
            ctc_prior_model_names[model_name]["prior_scale"],
            # [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
            ctc_prior_model_names[model_name]["lm_scale"],
            [32],
        ):
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            search_args["beam_size"] = beam_size
            ctc_scale = 1.0
            label_scale = 1.0
            search_args["decoder_args"] = CTCDecoderArgs(
                ctc_scale=ctc_scale,
                add_ext_lm=True,
                lm_type="trafo_ted",
                ext_lm_opts={
                    "lm_subnet": tedlium_lm_net,
                    "load_on_init_opts": tedlium_lm_load_on_init,
                },
                recombine=True,
                max_approx=True,
                lm_scale=lm_scale,
                ctc_prior_correction=True,
                prior_scale=prior_scale,
                target_dim=1057,
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
                + f"/optsr_max_ctc{1.0}_trafolm{lm_scale}_prior{prior_scale}_vanilla_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
                time_rqmt=2.0,
            )

    # aed + ctc + trafo lm
    # att, ctc, ctc_prior, label_scale, lm_scale
    joint_training_model_names = {
        "model_baseline": {
            "scales": [
                (0.8, 0.2, 0.5, 0.9, 0.7),
                (0.8, 0.2, 0.5, 1.0, 0.7),
                (0.8, 0.2, 0.6, 0.9, 0.7),
                (0.8, 0.2, 0.4, 0.9, 0.7),
                (0.8, 0.2, 0.5, 0.9, 0.65),
                (0.8, 0.2, 0.5, 0.9, 0.75),
            ],
        },
        # "model_ctc0.43_att1.0",
        # "model_ctc0.25_att1.0": {
        #     "scales": [(0.85, 0.15, 0.35)],
        # },
        # "model_ctc0.2_att1.0": {
        #     "scales": [(0.8, 0.2, 0.5)],
        # },
        "model_ctc0.9_att0.1": {
            "scales": [(0.7, 0.3, 0.6, 1.0, 0.6)],
        },
        "model_ctc0.8_att0.2": {
            "scales": [(0.75, 0.25, 0.3, 0.8, 0.6)],
        },
        "model_ctc0.7_att0.3": {
            "scales": [(0.7, 0.3, 0.6, 1.0, 0.5)],
        },
        "model_ctc0.6_att0.4": {
            "scales": [(0.7, 0.3, 0.4, 0.95, 0.5)],
        },
        "model_ctc0.5_att0.5": {
            "scales": [
                (0.7, 0.3, 0.45, 1.0, 0.5),
                (0.7, 0.3, 0.45, 1.0, 0.45),
                (0.7, 0.3, 0.45, 1.0, 0.55),
                (0.7, 0.3, 0.5, 1.0, 0.5),
                (0.7, 0.3, 0.4, 1.0, 0.5),
            ],
        },
        "model_ctc0.4_att0.6": {
            "scales": [(0.75, 0.25, 0.4, 0.9, 0.6)],
        },
        "model_ctc0.3_att0.7": {
            "scales": [(0.75, 0.25, 0.45, 0.9, 0.45)],
        },
        "model_ctc0.2_att0.8": {
            "scales": [
                (0.8, 0.2, 0.3, 0.8, 0.5),
                (0.8, 0.2, 0.35, 0.8, 0.5),
                (0.8, 0.2, 0.25, 0.8, 0.5),
                (0.8, 0.2, 0.3, 0.8, 0.53),
                (0.8, 0.2, 0.3, 0.8, 0.55),
                (0.8, 0.2, 0.3, 0.8, 0.57),
            ],
        },
        "model_ctc0.1_att0.9": {
            "scales": [(0.9, 0.1, 0.0, 0.95, 0.4)],
        },
        # "model_ctc0.001_att0.999": {
        #     "scales": [(0.85, 0.15)],
        # },
        # "model_ctc0.3_att0.7_lay6": {
        #     "scales": [(0.85, 0.15)],
        # },
        # "model_ctc0.3_att0.7_lay8": {
        #     "scales": [(0.85, 0.15)],
        # },
        # "model_ctc0.3_att0.7_lay10": {
        #     "scales": [(0.8, 0.2)],
        # },
        # "model_ctc1.0_att1.0_lay6": {
        #     "scales": [(0.8, 0.2)],
        # },
        # "model_ctc1.0_att1.0_lay8": {
        #     "scales": [(0.9, 0.1)],
        # },
        # "model_ctc1.0_att1.0_lay10": {
        #     "scales": [(0.9, 0.1)],
        # },
    }

    for model_name in [
        "model_baseline",
        # "model_ctc0.9_att0.1",
        # "model_ctc0.8_att0.2",
        # "model_ctc0.7_att0.3",
        # "model_ctc0.6_att0.4",
        "model_ctc0.5_att0.5",
        # "model_ctc0.4_att0.6",
        # "model_ctc0.3_att0.7",
        "model_ctc0.2_att0.8",
        # "model_ctc0.1_att0.9",
    ]:
        for scales, beam_size in product(
            joint_training_model_names[model_name]["scales"],
            [], # 32
        ):
            # for scales in joint_training_model_names[model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], model_name
            )
            search_args["beam_size"] = beam_size

            search_args["ctc_log_prior_file"] = models[model_name]["prior"]
            att_scale, ctc_scale, prior_scale, label_scale, lm_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                add_ext_lm=True,
                lm_type="trafo_ted",
                ext_lm_opts={
                    "lm_subnet": tedlium_lm_net,
                    "load_on_init_opts": tedlium_lm_load_on_init,
                },
                lm_scale=lm_scale,
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
                + f"/optsr_max_ctc{ctc_scale}_att{att_scale}_trafolm{lm_scale}"
                + (f"_prior{prior_scale}" if prior_scale > 0 else "")
                + f"_scales_l{label_scale}"
                + f"_vanilla"
                + f"_beam{beam_size}",
                # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev", "test"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
                time_rqmt=2.0,
            )

    # separate encoders
    # some model + ctc only

    joint_training_model_names_2 = {
        "model_baseline": {
            "scales": [(0.85, 0.15, 0.3)],
        },
        # "model_ctc0.43_att1.0",
        "model_ctc0.25_att1.0": {
            "scales": [(0.8, 0.2, 0.45), (0.8, 0.2, 0.5), (0.8, 0.2, 0.55)],
            "beam_sizes": [32, 64, 70],
        },
        "model_ctc0.2_att1.0": {
            "scales": [(0.75, 0.25, 0.45)],
            "beam_sizes": [32, 64],
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
            "scales": [(0.8, 0.2)],
            "beam_sizes": [],
        },
        "model_ctc0.3_att0.7_lay8": {
            "scales": [(0.8, 0.2), (0.75, 0.25)],
            "beam_sizes": [],
        },
        "model_ctc0.3_att0.7_lay10": {
            "scales": [(0.75, 0.25)],
            "beam_sizes": [],
        },
        "model_ctc1.0_att1.0_lay6": {
            "scales": [(0.65, 0.35)],
            "beam_sizes": [],
        },
        "model_ctc1.0_att1.0_lay8": {
            "scales": [(0.75, 0.25)],
            "beam_sizes": [],
        },
        "model_ctc1.0_att1.0_lay10": {
            "scales": [(0.6, 0.4), (0.55, 0.45)],
            "beam_sizes": [],
        },
    }

    second_model_name = "model_ctc_only"

    bsf = 10

    for first_model_name, lm_scale, beam_size in product(
        ["model_baseline"],
        [0.2, 0.3, 0.4],
        [],
    ):
        for scales in joint_training_model_names_2[first_model_name]["scales"]:
            search_args = copy.deepcopy(args)
            search_args["encoder_args"] = adjust_enc_args_to_model_name(
                search_args["encoder_args"], first_model_name
            )
            search_args["second_encoder_args_update_dict"] = {"enc_layer_w_ctc": None}
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = models["model_ctc_only"]["prior"]
            att_scale, ctc_scale, prior_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                add_ext_lm=True,
                lm_type="trafo_ted",
                ext_lm_opts={
                    "lm_subnet": tedlium_lm_net,
                    "load_on_init_opts": tedlium_lm_load_on_init,
                },
                lm_scale=lm_scale,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=True,
                prior_scale=prior_scale,
            )
            search_args[
                "second_encoder_ckpt"
            ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
            # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
            # search_args["hash_override_version"] = 1
            run_decoding(
                f"bsf{bsf}/" + first_model_name
                # + f"__ctc_only/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                + f"__ctc_only/opts_att{att_scale}_ctc{ctc_scale}_trafolm{lm_scale}_prior{prior_scale}_beam{beam_size}",
                train_data_baseline,
                checkpoint=models[first_model_name]["ckpt"],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )
