import copy
import os
from typing import Dict, Any

from sisyphus import gs, tk

from .system.attention_asr_config import (
    ConformerEncoderArgs,
    RNNDecoderArgs,
)
from .system.additional_config import (
    apply_fairseq_init_to_conformer,
)
from .system.pipeline import get_best_checkpoint
from .system.config_helpers import run_exp
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import (
    get_lm,
    ZeineldeenLM,
)
from i6_experiments.users.vieting.models.tf_networks.features import (
    NetworkDict,
    LogMelNetwork,
    GammatoneNetwork,
    ScfNetwork,
    PreemphasisNetwork,
)
from i6_experiments.users.vieting.tools.report import Report

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000

# LM
lstm_10k_lm_opts = {
    "lm_subnet": generic_lm.libri_lstm_bpe10k_net,
    "lm_model": generic_lm.libri_lstm_bpe10k_model,
    "name": "lstm",
}
lstm_lm_opts_map = {
    BPE_10K: lstm_10k_lm_opts,
}
trafo_lm_net = TransformerLM(source="prev:output", num_layers=24, vocab_size=10025, use_as_ext_lm=True)
trafo_lm_net.create_network()
trafo_10k_lm_opts = {
    "lm_subnet": trafo_lm_net.network.get_net(),
    "load_on_init_opts": {
        "filename": (
            "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/"
            "transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023"),
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}
bpe5k_lm = get_lm("ls960_trafo24_bs3000_5ep_5kbpe")  # type: ZeineldeenLM
trafo_5k_lm_opts = {
    "lm_subnet": bpe5k_lm.combination_network,
    "load_on_init_opts": {
        "filename": get_best_checkpoint(bpe5k_lm.train_job, key="dev_score_output/output"),
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}
trafo_lm_opts_map = {
    BPE_10K: trafo_10k_lm_opts,
    BPE_5K: trafo_5k_lm_opts,
}


def get_train_base_args() -> Dict[str, Any]:
    """
    Get basic arguments for training, i.e., model (encoder/decoder) and general (lr schedule, pretraining etc.) args.
    """
    # model (encoder/decoder) args
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
        ctc_loss_scale=1.0,
        use_sqrd_relu=True,
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)

    rnn_dec_args = RNNDecoderArgs(use_zoneout_output=True)

    # basic training args
    training_args = {
        "const_lr": [42, 100],  # use const LR during pretraining
        "wup_start_lr": 0.0002,
        "wup": 20,
        "with_staged_network": True,
        "speed_pert": True,
        "oclr_opts": {
            "peak_lr": 9e-4,
            "final_lr": 1e-6,
            "cycle_ep": 285,
            "total_ep": 635,
            "n_step": 1350,
            "learning_rates": [8e-5] * 35,
        },
        "batch_size": 15000 * 160,  # frames * samples per frame
        "pretrain_reps": 5,
        "pretrain_opts": {
            "variant": 3,
            "initial_batch_size": 22500 * 160,
        },
    }

    base_args = copy.deepcopy(
        {
            **training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    return base_args


def conformer_baseline():
    """
    Baseline attention system with conformer encoder and LSTM decoder. Based on Mohammad's setup.
    """
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_960_attention/feat/"

    # experiments
    args = get_train_base_args()
    _, _, report = run_exp(
        "baseline",
        train_args=args,
        num_epochs=635,
    )


def conformer_features():
    """
    Attention system with conformer encoder and LSTM decoder using different features.
    """
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_960_attention/feat/"

    def _get_feature_net(net: NetworkDict) -> Dict[str, Dict]:
        return {"features": net.get_as_subnetwork(source="data:audio_features")}

    # feature args
    feature_nets = {
        "logmel": LogMelNetwork(norm=False),
        "logmel_batchnorm": LogMelNetwork(),
        "scf": ScfNetwork(),
        "scf_batchnorm": ScfNetwork(normalization_env="batch"),
        "scf_nonorm": ScfNetwork(normalization_env=None),
        "scf_Tnorm": ScfNetwork(normalization_env="T"),
        "scf_TFnorm": ScfNetwork(normalization_env="TF"),
        "gammatone": GammatoneNetwork(),
    }

    # experiments
    report_list = []

    # vanilla experiments with different features and feature normalizations
    args = get_train_base_args()
    for name, feature_net in feature_nets.items():
        train_job, _, report = run_exp(
            name,
            train_args=args,
            num_epochs=635,
            feature_extraction_net=_get_feature_net(feature_net),
            feature_extraction_name="features",
        )
        report_list.append(report)
        if "scf" in name:
            train_job.rqmt.update({"gpu_mem": 24, "mem": 16, "cpu": 8, "sbatch_args": ["--gres=gpu:rtx_3090"]})

    # # scf with pretraining
    # # does not converge
    # args = get_train_base_args()
    # args["pretrain_opts"]["feature_extraction_pretraining"] = True
    # train_job, _, report = run_exp(
    #     "scf_pretrain",
    #     train_args=args,
    #     num_epochs=635,
    #     feature_extraction_net=_get_feature_net(feature_nets["scf"]),
    #     feature_extraction_name="features",
    # )
    # report_list.append(report)
    # train_job.rqmt.update({"gpu_mem": 24, "mem": 16, "cpu": 8, "sbatch_args": ["--gres=gpu:rtx_3090"]})
    #
    # # scf without pretraining of depthwise conv in conformer
    # # looks clearly worse than other trainings after ~100 sub-epochs
    # args = get_train_base_args()
    # args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
    # train_job, _, report = run_exp(
    #     "scf_no-conv-pretrain",
    #     train_args=args,
    #     num_epochs=635,
    #     feature_extraction_net=_get_feature_net(feature_nets["scf"]),
    #     feature_extraction_name="features",
    # )
    # report_list.append(report)
    # train_job.rqmt.update({"gpu_mem": 24, "mem": 16, "cpu": 8, "sbatch_args": ["--gres=gpu:rtx_3090"]})

    # scf with more SpecAugment
    for name_ext, max_time, freq_div in [("T25", 25, 5), ("F6", 20, 6), ("F4", 20, 4), ("F3", 20, 3)]:
        args = get_train_base_args()
        args["specaug_str_func_opts"] = {"max_time_dim": max_time, "freq_dim_factor": freq_div}
        train_job, _, report = run_exp(
            f"scf_specaug{name_ext}",
            train_args=args,
            num_epochs=635,
            feature_extraction_net=_get_feature_net(feature_nets["scf"]),
            feature_extraction_name="features",
        )
        report_list.append(report)
        train_job.rqmt.update({"gpu_mem": 24, "mem": 16, "cpu": 8})

    report = Report.merge_reports(report_list)
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
        values=report.get_values(),
        template=report.get_template())


def run_all():
    conformer_baseline()
    conformer_features()
