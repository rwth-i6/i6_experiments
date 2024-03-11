import copy, os

import numpy
from itertools import product

from sisyphus import toolkit as tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.gaudino.experiments.conformer_att_2023.librispeech_960.attention_asr_config import (
    CTCDecoderArgs,
    create_config,
    ConformerEncoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ConformerDecoderArgs,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
    apply_fairseq_init_to_transformer_decoder,
    reset_params_init,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.gaudino.experiments.conformer_att_2023.librispeech_960.default_tools import (
    # RETURNN_EXE,
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import (
    log10_net_10ms,
    log10_net_10ms_long_bn,
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
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import (
    ilm_helpers,
)
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import (
    get_lm,
    ZeineldeenLM,
)

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.search_helpers import (
    rescore_att_ctc_search,
)

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000

# --------------------------- LM --------------------------- #

lstm_10k_lm_opts = {
    "lm_subnet": generic_lm.libri_lstm_bpe10k_net,
    "lm_model": generic_lm.libri_lstm_bpe10k_model,
    "name": "lstm",
}

lstm_lm_opts_map = {
    BPE_10K: lstm_10k_lm_opts,
}

trafo_lm_net = TransformerLM(
    source="prev:output",
    num_layers=24,
    vocab_size=10025,
    emb_cpu_lookup=False,
    use_as_ext_lm=True,
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

# bpe5k_lm = get_lm("ls960_trafo24_bs3000_5ep_5kbpe")  # type: ZeineldeenLM
# trafo_5k_lm_opts = {
#     "lm_subnet": bpe5k_lm.combination_network,
#     "load_on_init_opts": {
#         "filename": get_best_checkpoint(bpe5k_lm.train_job, key="dev_score_output/output"),
#         "params_prefix": "",
#         "load_if_prefix": "lm_output/",
#     },
#     "name": "trafo",
# }

trafo_lm_opts_map = {
    BPE_10K: trafo_10k_lm_opts,
    # BPE_5K: trafo_5k_lm_opts,
}

new_prior_file = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZeflcEHlQTjn/output/prior.txt"
old_prior_file = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


# ----------------------------------------------------------- #


def run_ctc_att_search():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def get_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            test_dataset_tuples[testset] = build_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
        return test_dataset_tuples

    def run_train(
        exp_name,
        train_args,
        train_data,
        feature_extraction_net,
        num_epochs,
        recog_epochs,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )
        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
        )
        return train_job

    def run_single_search(
        exp_name,
        train_data,
        search_args,
        checkpoint,
        feature_extraction_net,
        recog_dataset,
        recog_ref,
        recog_bliss,
        mem_rqmt: float = 8,
        time_rqmt: float = 4,
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
            assert "att_scale" in kwargs and "ctc_scale" in kwargs, "rescore requires scales."
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
        feature_extraction_net,
        bpe_size,
        test_sets: list,
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
        tk.register_output(exp_prefix + "/priors/ctc_prior_fix", prior_j.out_prior_txt_file)
        return prior_j.out_prior_txt_file

    def run_search(
        exp_name,
        train_args,
        train_data,
        train_job,
        feature_extraction_net,
        num_epochs,
        search_args,
        recog_epochs,
        bpe_size,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )

        num_avg = kwargs.get("num_avg", 4)
        averaged_checkpoint = get_average_checkpoint(
            train_job,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            num_average=num_avg,
        )
        if num_avg == 4:  # TODO: just for now to not break hashes
            train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            if num_epochs <= 100:
                default_recog_epochs = [20, 40]
            else:
                default_recog_epochs = []
            default_recog_epochs += [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}",
                returnn_search_config,
                train_job.out_checkpoints[ep],
                test_dataset_tuples,
                RETURNN_CPU_EXE,
                RETURNN_ROOT,
            )

        search(
            exp_prefix + "/default_last",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + f"/average_{num_avg}",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            use_sclite=True,
        )

    def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        bpe_size=10000,
        **kwargs,
    ):
        if train_args.get("retrain_checkpoint", None):
            assert kwargs.get("epoch_wise_filter", None) is None, "epoch_wise_filter should be disabled for retraining."
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        train_job = run_train(
            exp_name,
            train_args,
            train_data,
            feature_extraction_net,
            num_epochs,
            recog_epochs,
            **kwargs,
        )
        train_jobs_map[exp_name] = train_job

        run_search(
            exp_name,
            train_args,
            train_data,
            train_job,
            feature_extraction_net,
            num_epochs,
            search_args,
            recog_epochs,
            bpe_size=bpe_size,
            **kwargs,
        )
        return train_job, train_data

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
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame corresponds to bsf=120

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
        "cycle_ep": 915,
        "total_ep": 2035,  # 20 epochs
        "n_step": 1350,
        "learning_rates": [8e-5] * 35,
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args["encoder_args"].use_sqrd_relu = True

    # Att baseline with avg checkpoint: 2.27/5.39/2.41/5.51
    name = "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"
    train_j, train_data = run_exp(name, train_args=oclr_args, num_epochs=2035)
    retrain_args = copy.deepcopy(oclr_args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt[name]
    retrain_args["learning_rates_list"] = [1e-4] * 20 + list(numpy.linspace(1e-4, 1e-6, 580))
    retrain_args["lr_decay"] = 0.95
    train_j, train_data = run_exp(
        exp_name=name + f"_retrain1_const20_linDecay580_{1e-4}",
        train_args=retrain_args,
        num_epochs=600,
    )

    bsf = 10
    oclr_args["batch_size"] = bsf * 20000

    # --------------------------- Without LM --------------------------- #

    # att only
    for beam_size in [12]:
        search_args = copy.deepcopy(oclr_args)
        search_args["beam_size"] = beam_size
        run_decoding(
            exp_name=f"bsf{bsf}/" + f"att_only_beam{beam_size}",
            train_data=train_data,
            checkpoint=train_job_avg_ckpt[
                f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
            ],
            search_args={"beam_size": beam_size, **search_args},
            feature_extraction_net=log10_net_10ms,
            bpe_size=BPE_10K,
            # test_sets=["dev-other"],
            test_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            use_sclite=True,
            # att_scale=att_scale,
            # ctc_scale=ctc_scale,
            # two_pass_rescore=True,  # two-pass rescoring
            # time_rqmt=1.0,
        )

    # ctc ts
    for beam_size, prior_scale in product([12, 32], [0, 0.1]):
        search_args = copy.deepcopy(oclr_args)
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = new_prior_file
        search_args["decoder_args"] = CTCDecoderArgs(
            ctc_prior_correction=prior_scale > 0, prior_scale=prior_scale, recombine=True
        )
        run_decoding(
            exp_name=f"bsf{bsf}/"
            + f"ctc_ts"
            + (f"_prior_{prior_scale}" if prior_scale > 0 else "")
            + f"_beam{beam_size}",
            train_data=train_data,
            checkpoint=train_job_avg_ckpt[
                f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
            ],
            search_args=search_args,
            feature_extraction_net=log10_net_10ms,
            bpe_size=BPE_10K,
            test_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            # test_sets=["dev-other"],
            remove_label={"<s>", "<blank>"},  # blanks are removed in the network
            use_sclite=True,
        )

    # optsr att + ctc w prior
    for scales, prior_scale, beam_size in product([(0.65, 0.35), (0.7, 0.3)], [0, 0.1], [32]):
        search_args = copy.deepcopy(oclr_args)
        search_args["beam_size"] = beam_size
        att_scale, ctc_scale = scales
        search_args["ctc_log_prior_file"] = new_prior_file

        search_args["decoder_args"] = CTCDecoderArgs(
            add_att_dec=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            ctc_prior_correction=prior_scale > 0,
            prior_scale=prior_scale,
            recombine=True,
        )
        run_decoding(
            exp_name=f"bsf{bsf}/"
            + f"optsr_att{att_scale}_ctc{ctc_scale}"
            + (f"_prior_{prior_scale}" if prior_scale > 0 else "")
            + f"_beam{beam_size}",
            train_data=train_data,
            checkpoint=train_job_avg_ckpt[
                f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
            ],
            search_args=search_args,
            feature_extraction_net=log10_net_10ms,
            bpe_size=BPE_10K,
            test_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
            # test_sets=["dev-other"],
            remove_label={
                "<s>",
                "<blank>",
            },  # blanks are removed in the network
            use_sclite=True,
        )

    # --------------------------- With Lstm LM --------------------------- #
    # --------------------------- With Trafo LM --------------------------- #
