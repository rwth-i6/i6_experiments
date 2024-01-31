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

    # models paths
    models = {
        "model_baseline": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt",
        },
        # ctcScale models
        "model_ctc0.43_att1.0": {  # ctcScale 0.3
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.nCrQhRfqIRiZ/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Yonvnwljktqh/output/prior.txt",
        },
        "model_ctc0.25_att1.0": {  # ctcScale 0.2
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.CknpN55pjOHo/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.MhkU9CYwTQy3/output/prior.txt",
        },
        "model_ctc0.2_att1.0": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.ro9g9W6DBJpW/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.gJiuTmxRwMVu/output/prior.txt",
        },
        # 1-y models
        "model_ctc0.3_att0.7": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.jGxeW6yzeoG7/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ypsBrM65Uj1k/output/prior.txt",
        },
        "model_ctc0.2_att0.8": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.6qWPnvXHalfJ/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.mHUoJaQFZ27b/output/prior.txt",
        },
        "model_ctc0.1_att0.9": {  # pre 4
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.MEtpESN5M4oD/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.I4aVOIk1CXmt/output/prior.txt",
        },
        "model_ctc0.001_att0.999": {  # pre 4
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.eEEAEAZQiFvO/output/model/average.index"
                )
            ),
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.mhjgjO6IUEPB/output/prior.txt",
        },
        # att only
        "model_att_only_currL": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.io6cKw6ETnHp/output/model/average.index"
                )
            ),
            "prior": "",
        },
        "model_att_only_adjSpec": {
            "ckpt": Checkpoint(
                tk.Path(
                    "work/i6_core/returnn/training/AverageTFCheckpointsJob.9f6nlw1UOxVO/output/model/average.index"
                )
            ),
            "prior": "",
        },
        # ctc only
        "model_ctc_only": {
            "ckpt": Checkpoint(
                tk.Path(
                    "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400.index"
                )
            ),  # last
            "prior": "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.Ow9jQN0VEdlo/output/prior.txt",  # how is this computed?
        },
    }

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

    # ctc greedy decoding
    search_args = copy.deepcopy(args)
    search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)

    for model_name in list(models.keys())[:-3] + ["model_ctc_only"]:
        run_decoding(
            model_name + f"/ctc_greedy",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    search_args = copy.deepcopy(args)
    for scales in [(0.7, 0.3)]:
        for beam_size in [55]:
            for prior_scale in [0.4]:
                search_args["beam_size"] = beam_size
                search_args["ctc_log_prior_file"] = models["model_ctc0.3_att0.7"][
                    "prior"
                ]
                att_scale, ctc_scale = scales
                search_args["decoder_args"] = CTCDecoderArgs(
                    add_att_dec=True,
                    att_scale=att_scale,
                    ctc_scale=ctc_scale,
                    att_masking_fix=True,
                    target_dim=1057,
                    target_embed_dim=256,
                    ctc_prior_correction=True,
                    prior_scale=prior_scale,
                )
                run_decoding(
                    f"model_ctc_0.43_att_1.0/opts_ctc{ctc_scale}_att{att_scale}_prior{prior_scale}_beam{beam_size}",
                    train_data_baseline,
                    checkpoint=models["model_ctc0.43_att1.0"]["ckpt"],
                    search_args=search_args,
                    bpe_size=BPE_1K,
                    test_sets=["dev"],
                    remove_label={"<s>", "<blank>"},
                    use_sclite=True,
                )

    from i6_experiments.users.gaudino.models.asr.lm.tedlium_lm import (
        tedlium_lm_net,
        tedlium_lm_model,
        tedlium_lm_load_on_init,
    )

    # try ctc + trafo lm
    search_args = copy.deepcopy(args)
    for scales, beam_size, model_name in product(
        [(1.0, 0.0), (1.0, 0.05), (1.0, 0.1), (1.0, 0.2), (1.0, 0.3), (1.0, 0.5)],
        [],
        ["model_baseline", "model_ctc_only"],
    ):
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = models[model_name]["prior"]
        search_args["batch_size"] = 2000 * 160
        ctc_scale, lm_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(
            ctc_scale=ctc_scale,
            add_ext_lm=True,  # TODO
            lm_type="trafo_ted",
            ext_lm_opts={
                "lm_subnet": tedlium_lm_net,
                "load_on_init_opts": tedlium_lm_load_on_init,
            },
            lm_scale=lm_scale,
            target_dim=1057,
            target_embed_dim=256,
            # ctc_prior_correction=True,
            # prior_scale=prior_scale,
        )
        run_decoding(
            model_name + f"/opts_ctc{ctc_scale}_trafolm{lm_scale}_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
            time_rqmt=4.0,
        )

    # ctc + trafo lm try different things
    search_args = copy.deepcopy(args)
    for scales, beam_size, model_name, blank_scale, rm_eos_lm in product(
        [(1.0, 0.1), (0.9, 0.1)],
        [12],
        ["model_baseline"], # , "model_ctc_only"
        [0.0, 0.5, 1.0, 2.0, 3.0],
        [True, False],
    ):
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = models[model_name]["prior"]
        search_args["batch_size"] = 2000 * 160
        ctc_scale, lm_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(
            ctc_scale=ctc_scale,
            add_ext_lm=True,
            lm_type="trafo_ted",
            ext_lm_opts={
                "lm_subnet": tedlium_lm_net,
                "load_on_init_opts": tedlium_lm_load_on_init,
            },
            lm_scale=lm_scale,
            target_dim=1057,
            target_embed_dim=256,
            remove_eos_from_ctc=True,
            remove_eos_from_ts=rm_eos_lm,
            add_eos_to_blank=True,
            blank_prob_scale=blank_scale, # substracted from log prob
            # rescore_last_eos=True,
            # blank_collapse=True,
            # ctc_prior_correction=True,
            # prior_scale=prior_scale,
        )
        run_decoding(
            model_name + f"/opts_ctc{ctc_scale}_trafolm{lm_scale}_beam{beam_size}" +
            (f"_blankScale{blank_scale}" if blank_scale != 0.0 else "") +
            (f"_rmEosLM" if rm_eos_lm else ""),
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
            time_rqmt=2.0,
        )

    # debug att + ctc rescore w eos
    search_args = copy.deepcopy(args)
    for scales, beam_size, model_name in product(
            [(0.65, 0.35)],
            [6, 12],
            ["model_baseline"],
    ):
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = models[model_name]["prior"]
        search_args["batch_size"] = 2000 * 160
        att_scale, ctc_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(
            add_att_dec=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            target_dim=1057,
            target_embed_dim=256,
            remove_eos_from_ctc=True,
            remove_eos_from_ts=True,
            add_eos_to_blank=True,
            rescore_last_eos=True,
            eos_postfix=True,
        )
        run_decoding(
            model_name + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}_rescore_eos_2",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
            time_rqmt=4.0,
        )

    joint_training_model_names = {
        # "model_ctc0.43_att1.0",
        "model_ctc0.25_att1.0": {
            "scales": [(0.85, 0.15, 0.35)],
        },
        "model_ctc0.2_att1.0": {
            "scales": [(0.8, 0.2, 0.5)],
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
            "scales": [(0.85, 0.15)],
        },
    }

    search_args = copy.deepcopy(args)

    for model_name, beam_size in product(
        ["model_ctc0.25_att1.0"],
        [],
    ):
        # for scales in joint_training_model_names[model_name]["scales"]:
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = models[model_name]["prior"]
        att_scale, ctc_scale, prior_scale = joint_training_model_names[model_name][
            "scales"
        ][0]
        search_args["decoder_args"] = CTCDecoderArgs(
            add_att_dec=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            att_masking_fix=True,
            target_dim=1057,
            target_embed_dim=256,
            ctc_prior_correction=True,
            prior_scale=prior_scale,
        )
        run_decoding(
            model_name
            + f"/opts_ctc{ctc_scale}_att{att_scale}_prior{prior_scale}_beam{beam_size}",
            # + f"/opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    # separate encoders
    # some model + ctc only
    # for name_scales, prior_scale, beam_size in product(
    #     [("model_ctc0.25_att1.0", (0.8, 0.2)), ("model_att_only_currL", (0.75, 0.25))],
    #     [0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
    #     [32]
    # ):

    joint_training_model_names = {
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
    }

    first_model_name = "model_ctc0.1_att0.9"

    for scales, prior_scale, beam_size in product(
        [(0.8, 0.2), (0.77, 0.23), (0.82, 0.18)],
        [0.55, 0.6, 0.65, 0.7, 0.75],
        [],
    ):
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = models["model_ctc_only"]["prior"]
        att_scale, ctc_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(
            add_att_dec=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            att_masking_fix=True,
            target_dim=1057,
            target_embed_dim=256,
            ctc_prior_correction=True,
            prior_scale=prior_scale,
        )
        search_args[
            "second_encoder_ckpt"
        ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
        # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
        run_decoding(
            first_model_name
            + f"__ctc_only/opts_ctc{ctc_scale}_att{att_scale}_prior{prior_scale}_beam{beam_size}",
            train_data_baseline,
            checkpoint=models[first_model_name]["ckpt"],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    # if True:
    #     # ctc greedy of separate encoder as sanity check
    #     search_args["beam_size"] = beam_size
    #     search_args["ctc_log_prior_file"] = prior_file_ctc_only
    #     att_scale, ctc_scale = scales
    #     search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)
    #     search_args[
    #         "second_encoder_ckpt"
    #     ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
    #     # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
    #     run_decoding(
    #         f"model_base__ctc_only/ctc_greedy",
    #         train_data,
    #         checkpoint=train_job_avg_ckpt[name],
    #         search_args=search_args,
    #         bpe_size=BPE_1K,
    #         test_sets=["dev"],
    #         remove_label={"<s>", "<blank>"},
    #         use_sclite=True,
    #     )
