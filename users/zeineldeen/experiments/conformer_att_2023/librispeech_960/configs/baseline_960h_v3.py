import copy, os

import numpy

from sisyphus import *

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.attention_asr_config import (
    create_config,
    ConformerEncoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ConformerDecoderArgs,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
    apply_fairseq_init_to_transformer_decoder,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.librispeech_960.default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
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
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import (
    ilm_helpers,
)
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import (
    get_lm,
    ZeineldeenLM,
)

from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.data import (
    build_test_dataset as build_ted2_test_dataset,
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

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.seq_train_helpers import (
    get_trans_lstm_lm,
)

lbs_trans_subnet_cls = get_trans_lstm_lm()
lbs_density_ratio_lm_opts = {
    "lm_subset": lbs_trans_subnet_cls["subnetwork"],
    "lm_model": lbs_trans_subnet_cls["load_on_init"],
    "name": "lstm",
}

ted2_lstm_lm_opts = {
    "lm_subnet": generic_lm.libri_lstm_bpe10k_net,
    "lm_model": "/work/asr4/michel/setups-data/lm_training/data-train/tedlium_re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/net-model/network.020",
    "name": "lstm",
}

ted2_density_ratio_lm_opts = {}

lstm_lm_opts_map = {
    BPE_10K: lstm_10k_lm_opts,
}

trafo_lm_net = TransformerLM(source="prev:output", num_layers=24, vocab_size=10025, use_as_ext_lm=True)
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

TRAIN_AVG_ENC = tk.Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr/output/enc.mean.txt",
    hash_overwrite="best_global_aed_enc_avg",
)
TRAIN_AVG_ATT = tk.Path(
    "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr/output/att_ctx.mean.txt",
    hash_overwrite="best_global_aed_att_avg",
)

# ----------------------------------------------------------- #


def conformer_baseline():
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

    def get_ted2_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev", "test"]:
            test_dataset_tuples[testset] = build_ted2_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
            # override to use LibriSpeech bpe labels
            test_dataset_tuples[testset][0].datasets["zip_dataset"]["targets"]["bpe_file"] = tk.Path(
                "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"
            )
            test_dataset_tuples[testset][0].datasets["zip_dataset"]["targets"]["vocab_file"] = tk.Path(
                "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
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
            gpu_mem=kwargs.get("gpu_mem", 11),
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
        mem_rqmt=8,
        time_rqmt: float = 4,
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
            use_sclite=kwargs.get("use_sclite", False),
        )

    def run_lm_fusion(
        lm_type,
        exp_name,
        epoch,
        test_set_names,
        lm_scales,
        train_job,
        train_data,
        feature_net,
        bpe_size,
        args,
        beam_size=12,
        length_norm_exponent=1.0,
        prior_scales=None,
        prior_type=None,
        mini_lstm_ckpt=None,
        length_norm=True,
        prior_type_name=None,
        coverage_scale=None,
        coverage_threshold=None,
        coverage_update="sum",
        ext_lm_opts=None,
        **kwargs,
    ):
        assert lm_type in ["lstm", "trafo"], "lm type should be lstm or trafo"

        if isinstance(lm_scales, float):
            lm_scales = [lm_scales]
        if prior_scales and isinstance(prior_scales, float):
            prior_scales = [prior_scales]
        if isinstance(test_set_names, str):
            test_set_names = [test_set_names]
        assert isinstance(test_set_names, list)

        if epoch == "avg":
            search_checkpoint = train_job_avg_ckpt[exp_name]
        elif epoch == "best":
            search_checkpoint = train_job_best_epoch[exp_name]
        else:
            assert isinstance(epoch, int), "epoch must be either a defined integer or a string in {avg, best}."
            search_checkpoint = train_job.out_checkpoints[epoch]

        if ext_lm_opts is None:
            ext_lm_opts = lstm_lm_opts_map[bpe_size] if lm_type == "lstm" else trafo_lm_opts_map[bpe_size]

        time_rqmt = 1.0

        search_args = copy.deepcopy(args)

        if lm_type == "lstm":
            if beam_size > 128:
                search_args["batch_size"] = 4000 * 160

        if lm_type == "trafo":
            search_args["batch_size"] = 4000 * 160 if beam_size <= 32 else 2000 * 160
            time_rqmt = 2
            if beam_size > 50:
                time_rqmt = 3

        search_args["beam_size"] = beam_size
        if kwargs.get("batch_size", None):
            search_args["batch_size"] = kwargs["batch_size"]

        if not length_norm:
            search_args["decoder_args"].length_normalization = False
        else:
            search_args["decoder_args"].length_normalization_exponent = length_norm_exponent

        if "decoder_args" in kwargs:
            for k, v in kwargs["decoder_args"].items():
                setattr(search_args["decoder_args"], k, v)

        scales = [(e,) for e in lm_scales]

        for test_set in test_set_names:
            if prior_scales:
                import itertools

                scales = itertools.product(lm_scales, prior_scales)

            for scale in scales:
                lm_scale = scale[0]
                prior_scale = scale[1] if len(scale) == 2 else None
                # if prior_scale and prior_scale > lm_scale:
                #     continue

                # External LM opts
                ext_lm_opts["lm_scale"] = lm_scale
                search_args["ext_lm_opts"] = ext_lm_opts

                # ILM opts
                if prior_scale:
                    ilm_opts = {
                        "scale": prior_scale,
                        "type": prior_type,
                        "ctx_dim": search_args["encoder_args"].enc_key_dim,  # this is needed for mini-lstm
                    }
                    # this is needed for mini-self-att
                    if hasattr(search_args["decoder_args"], "num_layers"):
                        ilm_opts["num_dec_layers"] = search_args["decoder_args"].num_layers
                        search_args["decoder_args"].create_ilm_decoder = True
                        search_args["decoder_args"].ilm_type = prior_type

                    ilm_opts.update(kwargs.get("ilm_train_opts", {}))  # example for FFN, etc

                    search_args["prior_lm_opts"] = ilm_opts
                    search_args["preload_from_files"] = {
                        "prior_lm": {
                            "filename": search_checkpoint,  # copy ASR decoder to be used as ILM decoder
                            "prefix": "prior_",
                        }
                    }
                    if prior_type == "mini_lstm" or prior_type == "ffn":
                        assert mini_lstm_ckpt, "Mini-LSTM checkpoint not set."
                        search_args["preload_from_files"].update(
                            {
                                "mini_lstm": {
                                    "filename": mini_lstm_ckpt,
                                    "prefix": "mini_",
                                }
                            }
                        )

                    if prior_type == "train_avg_enc":
                        ilm_opts["data"] = TRAIN_AVG_ENC
                    if prior_type == "train_avg_ctx":
                        ilm_opts["data"] = TRAIN_AVG_ATT

                if prior_type_name is None:
                    prior_type_name = prior_type

                lm_desc = f"lm-scale-{lm_scale}"
                if prior_scale:
                    lm_desc += f"-prior-{prior_scale}-{prior_type_name}"
                lm_desc += f"-beam-{beam_size}"
                if length_norm is False:
                    lm_desc += "-woLenNorm"

                if coverage_scale and coverage_threshold:
                    assert isinstance(search_args["decoder_args"], RNNDecoderArgs)
                    search_args["decoder_args"].coverage_scale = coverage_scale
                    search_args["decoder_args"].coverage_threshold = coverage_threshold
                    search_args["decoder_args"].coverage_update = coverage_update
                    lm_desc += f"_coverage-thre{coverage_threshold}-scale{coverage_scale}"
                    if coverage_update != "sum":
                        lm_desc += f"-{coverage_update}"

                name = exp_name
                if kwargs.get("extra_name", None):
                    name += f"/{kwargs['extra_name']}"
                name += f"/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_set}"

                if kwargs.get("test_dataset_tuples", None):
                    test_dataset_tuples = kwargs["test_dataset_tuples"]
                else:
                    test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

                run_single_search(
                    exp_name=name,
                    train_data=train_data,
                    search_args=search_args,
                    checkpoint=search_checkpoint,
                    feature_extraction_net=feature_net,
                    recog_dataset=test_dataset_tuples[test_set][0],
                    recog_ref=test_dataset_tuples[test_set][1],
                    recog_bliss=test_dataset_tuples[test_set][2],
                    time_rqmt=kwargs.get("time_rqmt", time_rqmt),
                    use_sclite=kwargs.get("use_sclite", False),
                )

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
            enable_mail=True,
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
            if "allow_lr_scheduling" not in train_args:
                train_args["allow_lr_scheduling"] = False  # force it

        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        if train_args["encoder_args"].num_blocks > 12:
            train_args["recursion_limit"] = 10_000

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

    def train_mini_lstm(
        exp_name,
        checkpoint,
        args,
        num_epochs=20,
        lr=8e-4,
        time_rqmt=8,
        l2=1e-4,
        name="mini_lstm",
        w_drop=False,
        use_dec_state=False,
        use_ffn=False,
        ffn_opts=None,
        att_ctx_constraint_loss=None,
        att_ctx_constraint_loss_scale=0.0,
        **kwargs,
    ):
        if not w_drop:
            params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str()
        else:
            if use_ffn:
                params_freeze_str = ilm_helpers.get_ffn_params_freeze_str_w_drop(ffn_opts["num_ffn_layers"])
            else:
                params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str_w_drop()

        mini_lstm_args = copy.deepcopy(args)
        mini_lstm_args["batch_size"] = 20000 * 160
        mini_lstm_args["with_pretrain"] = False
        mini_lstm_args["lr"] = lr
        mini_lstm_args["allow_lr_scheduling"] = False
        mini_lstm_args["encoder_args"].with_ctc = False
        mini_lstm_args["keep_all_epochs"] = True  # keep everything
        mini_lstm_args["extra_str"] = params_freeze_str
        mini_lstm_args["preload_from_files"] = {
            "import": {
                "init_for_train": True,
                "ignore_missing": True,
                "filename": checkpoint,
            }
        }
        mini_lstm_args.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_lstm_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        returnn_config = create_config(
            training_datasets=mini_lstm_train_data,
            **mini_lstm_args,
            feature_extraction_net=log10_net_10ms,
        )

        inp = "s" if use_dec_state else "prev:target_embed"

        if use_ffn:
            x = inp
            activations = ffn_opts["activations"]
            for l in range(ffn_opts["num_ffn_layers"]):
                returnn_config.config["network"]["output"]["unit"]["ffn_%02i" % (l + 1)] = {
                    "class": "linear",
                    "n_out": ffn_opts["ffn_dims"][l],
                    "L2": l2,
                    "from": inp,
                    "activation": activations[l] if activations and l < len(activations) else None,
                }
                x = "ffn_%02i" % (l + 1)

            returnn_config.config["network"]["output"]["unit"]["att"] = {
                "class": "linear",
                "from": x,
                "activation": None,
                "n_out": mini_lstm_args["encoder_args"].enc_key_dim,
                "L2": l2,
            }
        else:
            # Mini-LSTM + FF

            if att_ctx_constraint_loss_scale:
                # copy original context vector
                returnn_config.config["network"]["output"]["unit"]["original_att"] = copy.deepcopy(
                    returnn_config.config["network"]["output"]["unit"]["att"]
                )
                returnn_config.config["network"]["output"]["unit"]["original_att"]["from"] = "original_att0"

                # copy original subnetwork to compute original attention vector
                layer_names = [
                    "energy",
                    "energy_in",
                    "energy_tanh",
                    "s",
                    "s_transformed",
                    "accum_att_weights",
                    "att_weights",
                    "att0",
                    "weight_feedback",
                ]
                subnet_unit = copy.deepcopy(returnn_config.config["network"]["output"]["unit"])
                for k in subnet_unit.keys():
                    if k in layer_names:
                        # special case
                        if k == "att0":
                            returnn_config.config["network"]["output"]["unit"]["original_att0"] = copy.deepcopy(
                                subnet_unit[k]
                            )
                            returnn_config.config["network"]["output"]["unit"]["original_att0"][
                                "weights"
                            ] = "original_att_weights"
                            continue

                        from_list = copy.deepcopy(subnet_unit[k]["from"])
                        if isinstance(from_list, str):
                            from_list = [from_list]
                        new_from_list = []
                        for e in from_list:
                            name = e
                            with_prev = False
                            if "prev:" in name:
                                name = name[len("prev:") :]
                                with_prev = True

                            if name in layer_names or name == "att":
                                new_from_list.append(("prev:" if with_prev else "") + "original_" + name)
                            else:
                                new_from_list.append(e)
                        l = copy.deepcopy(subnet_unit[k])
                        l["from"] = new_from_list
                        returnn_config.config["network"]["output"]["unit"]["original_{}".format(k)] = l

                if att_ctx_constraint_loss == "mse":
                    loss_name = "att_loss"
                    returnn_config.config["network"]["output"]["unit"]["se_loss"] = {
                        "class": "eval",
                        "eval": "(source(0) - source(1)) ** 2",
                        "from": ["original_att", "att"],
                    }
                    returnn_config.config["network"]["output"]["unit"]["att_loss"] = {
                        "class": "reduce",
                        "mode": "mean",
                        "axis": "F",
                        "from": "se_loss",
                        "loss": "as_is",
                        "loss_scale": att_ctx_constraint_loss_scale,
                    }
                elif att_ctx_constraint_loss.startswith("cos"):
                    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/CosineSimilarity
                    # loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
                    loss_name = "cos_loss"
                    returnn_config.config["network"]["output"]["unit"]["norm_mult"] = {
                        "class": "eval",
                        "eval": "tf.linalg.l2_normalize(source(0)) * tf.linalg.l2_normalize(source(1))",
                        "from": ["original_att", "att"],
                    }
                    returnn_config.config["network"]["output"]["unit"]["cos_loss0"] = {
                        "class": "reduce",
                        "mode": "sum",
                        "axis": "F",
                        "from": "norm_mult",
                    }
                    if att_constraints_loss == "cos":
                        eval_str = "-source(0)"
                    elif att_constraints_loss == "cos_v2":
                        eval_str = "1 - source(0)"
                    else:
                        raise ValueError(f"Unknown att_ctx_constraint_loss {att_ctx_constraint_loss}")
                    returnn_config.config["network"]["output"]["unit"]["cos_loss"] = {
                        "class": "eval",
                        "eval": eval_str,
                        "from": "cos_loss0",
                        "loss": "as_is",
                        "loss_scale": att_ctx_constraint_loss_scale,
                    }
                else:
                    raise ValueError(f"Unknown att_ctx_constraint_loss {att_ctx_constraint_loss}")

            returnn_config.config["network"]["output"]["unit"]["att_lstm"] = {
                "class": "rec",
                "unit": "nativelstm2",
                "from": inp,
                "n_out": 50,
            }

            returnn_config.config["network"]["output"]["unit"]["att"] = {
                "class": "linear",
                "from": "att_lstm",
                "activation": None,
                "n_out": mini_lstm_args["encoder_args"].enc_key_dim,
                "L2": l2,
            }

        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )
        return train_job

    def train_mini_self_att(
        exp_name,
        checkpoint,
        args,
        num_epochs=20,
        lr=8e-4,
        time_rqmt=4,
        name="mini_self_att",
        **kwargs,
    ):
        """
        Same idea as Mini-LSTM but use masked (mini-)self-attention models instead of cross attention.
        Note that each layer has its own (mini-)self-attention.

        In the case of transformer decoder, we want to replace cross-attention layers namely:
            transformer_decoder_{idx}_att_linear
        with masked self-attention models.
        """

        params_freeze_str = ilm_helpers.get_mini_self_att_params_freeze_str_w_drop(args["decoder_args"].num_layers)

        mini_self_att = copy.deepcopy(args)
        mini_self_att["batch_size"] = 20000 * 160  # TODO: does this fit now?
        mini_self_att["with_pretrain"] = False
        mini_self_att["lr"] = lr
        mini_self_att["allow_lr_scheduling"] = False
        mini_self_att["encoder_args"].with_ctc = False
        # mini_self_att['keep_all_epochs'] = True  # keep everything
        mini_self_att["extra_str"] = params_freeze_str
        mini_self_att["preload_from_files"] = {
            "import": {
                "init_for_train": True,
                "ignore_missing": True,
                "filename": checkpoint,
            }
        }
        if "decoder_args" in kwargs:
            assert isinstance(kwargs["decoder_args"], dict)
            for k, v in kwargs["decoder_args"].items():
                setattr(mini_self_att["decoder_args"], k, v)
            kwargs.pop("decoder_args")
        mini_self_att.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_self_att_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        # use masked self-att instead of cross-att with layer names having "ilm_" as prefix
        mini_self_att["decoder_args"].replace_cross_att_w_masked_self_att = True

        returnn_config = create_config(
            training_datasets=mini_self_att_train_data,
            **mini_self_att,
            feature_extraction_net=log10_net_10ms,
        )
        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )
        return train_job

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
        "cycle_ep": 915,
        "total_ep": 2035,  # 20 epochs
        "n_step": 1350,
        "learning_rates": [8e-5] * 35,
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args["encoder_args"].use_sqrd_relu = True

    # Wo LM with best: 2.28/5.63/2.48/5.71
    # Wo LM with avg:  2.28/5.60/2.48/5.75
    name = "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"
    run_exp(name, train_args=oclr_args, num_epochs=2035)

    # Att baseline with avg checkpoint: 2.27/5.39/2.41/5.51
    retrain_args = copy.deepcopy(oclr_args)
    best_global_att_avg_ckpt = train_job_avg_ckpt[name]
    retrain_args["retrain_checkpoint"] = best_global_att_avg_ckpt
    retrain_args["learning_rates_list"] = [1e-4] * 20 + list(numpy.linspace(1e-4, 1e-6, 580))
    retrain_args["lr_decay"] = 0.95
    train_j, train_data = run_exp(
        exp_name=name + f"_retrain1_const20_linDecay580_{1e-4}",
        train_args=retrain_args,
        num_epochs=600,
    )

    # Cross-domain evaluation
    # None: 16.1/16.9
    # LM: 13.9/14.6
    # LM+ILM: 12.5/13.6

    for beam_size in [10]:
        for lm_scale in [0.38]:
            run_lm_fusion(
                lm_type="lstm",
                extra_name="ted2-recogs",
                ext_lm_opts=ted2_lstm_lm_opts,
                exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                epoch="avg",
                test_set_names=["test"],
                lm_scales=[lm_scale],
                train_job=train_j,
                train_data=train_data,
                feature_net=log10_net_10ms,
                args=oclr_args,
                beam_size=beam_size,
                batch_size=10_000 * 160,
                bpe_size=BPE_10K,
                use_sclite=True,
                test_dataset_tuples=get_ted2_test_dataset_tuples(BPE_10K),
            )

    # 1.9/4.2/2.1/4.6
    run_lm_fusion(
        lm_type="trafo",
        exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
        epoch="avg",
        test_set_names=["dev-clean", "dev-other", "test-clean", "test-other"],
        lm_scales=[0.42],
        train_job=train_j,
        train_data=train_data,
        feature_net=log10_net_10ms,
        args=oclr_args,
        beam_size=32,
        batch_size=4000 * 160,
        bpe_size=BPE_10K,
        use_sclite=True,
    )

    for att_constraints_loss in ["mse"]:
        for att_constraint_scale in [0.05]:
            mini_lstm_j = train_mini_lstm(
                exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                checkpoint=best_global_att_avg_ckpt,
                args=oclr_args,
                num_epochs=40,
                w_drop=True,
                att_ctx_constraint_loss=att_constraints_loss,
                att_ctx_constraint_loss_scale=att_constraint_scale,
                name=f"mini_lstm_{att_constraints_loss}Loss{att_constraint_scale}",
            )

            # with LM
            mini_lstm_j_v2 = train_mini_lstm(
                exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                checkpoint=train_job_avg_ckpt[
                    f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
                ],
                args=oclr_args,
                num_epochs=40,
                w_drop=True,
            )

            # best recog model:
            # base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_0.0001/recog-trafo-lm/ep-avg/lm-scale-0.54-prior-0.4-mini_lstm_mseLoss0.05-beam-84_coverage-thre0.1-scale0.2
            # dev-other/test-other: 3.64/4.18
            for beam_size in [32, 70, 84]:
                run_lm_fusion(
                    lm_type="trafo",
                    exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                    epoch="avg",
                    test_set_names=["dev-clean", "dev-other", "test-clean", "test-other"],
                    lm_scales=[0.54],
                    prior_scales=[0.4],
                    prior_type="mini_lstm",
                    prior_type_name=f"mini_lstm_{att_constraints_loss}Loss{att_constraint_scale}",
                    mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key="dev_score_output/output_prob"),
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=1000 * 160,
                    bpe_size=BPE_10K,
                    use_sclite=True,
                )

                if beam_size == 32:
                    for len_norm_exp in [0.0]:
                        for k, lm, ilm in [("test-clean", 0.38, 0.3), ("test-other", 0.46, 0.34)]:
                            run_lm_fusion(
                                lm_type="trafo",
                                exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                                epoch="avg",
                                test_set_names=[k],
                                lm_scales=[lm],
                                prior_scales=[ilm],
                                prior_type="zero",
                                train_job=train_j,
                                train_data=train_data,
                                feature_net=log10_net_10ms,
                                args=oclr_args,
                                beam_size=beam_size,
                                batch_size=1000 * 160,
                                bpe_size=BPE_10K,
                                coverage_scale=0.2,
                                coverage_threshold=0.1,
                                use_sclite=True,
                                length_norm_exponent=len_norm_exp,
                            )

                run_lm_fusion(
                    lm_type="trafo",
                    exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                    epoch="avg",
                    test_set_names=["dev-other"],
                    lm_scales=[0.54],
                    prior_scales=[0.4],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j_v2, key="dev_score"),
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=1000 * 160,
                    bpe_size=BPE_10K,
                    use_sclite=True,
                )

                # TODO: with ILM
                # lm-scale-0.78-prior-0.76-mini_lstm_mseLoss0.05_ep10_lenNorm0.0-beam-12 12.1
                for length_norm_exp in [0.0]:
                    for mini_lstm_ep in [10]:
                        for beam_size in [12]:
                            for lm_scale in [0.78]:
                                for ilm_scale in [0.76]:
                                    run_lm_fusion(
                                        lm_type="lstm",
                                        extra_name="ted2-recogs",
                                        ext_lm_opts=ted2_lstm_lm_opts,
                                        exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                                        epoch="avg",
                                        test_set_names=["dev", "test"],
                                        lm_scales=[lm_scale],
                                        prior_scales=[ilm_scale],
                                        prior_type="mini_lstm",
                                        prior_type_name=f"mini_lstm_{att_constraints_loss}Loss{att_constraint_scale}_ep{mini_lstm_ep}_lenNorm{length_norm_exp}",
                                        # mini_lstm_ckpt=get_best_checkpoint(
                                        #     mini_lstm_j, key="dev_score_output/output_prob"
                                        # ),
                                        length_norm_exponent=length_norm_exp,
                                        mini_lstm_ckpt=mini_lstm_j.out_checkpoints[mini_lstm_ep],
                                        train_job=train_j,
                                        train_data=train_data,
                                        feature_net=log10_net_10ms,
                                        args=oclr_args,
                                        beam_size=beam_size,
                                        batch_size=10_000 * 160,
                                        bpe_size=BPE_10K,
                                        use_sclite=True,
                                        test_dataset_tuples=get_ted2_test_dataset_tuples(BPE_10K),
                                    )

                                for prior_type in ["avg"]:
                                    for lm_scale in [0.7, 0.72, 0.74, 0.76, 0.78]:
                                        for ilm_scale in [0.6, 0.62, 0.64, 0.66, 0.68, 0.7]:
                                            run_lm_fusion(
                                                lm_type="lstm",
                                                extra_name="ted2-recogs",
                                                ext_lm_opts=ted2_lstm_lm_opts,
                                                exp_name=f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}",
                                                epoch="avg",
                                                test_set_names=["dev", "test"],
                                                lm_scales=[lm_scale],
                                                prior_scales=[ilm_scale],
                                                prior_type=prior_type,
                                                prior_type_name={
                                                    "train_avg_enc": "avgEnc",
                                                    "train_avg_ctx": "avgAtt",
                                                    "avg": "seqAvg",
                                                }.get(prior_type),
                                                length_norm_exponent=length_norm_exp,
                                                mini_lstm_ckpt=mini_lstm_j.out_checkpoints[mini_lstm_ep],
                                                train_job=train_j,
                                                train_data=train_data,
                                                feature_net=log10_net_10ms,
                                                args={**oclr_args, "extra_prolog": ["import numpy"]},
                                                beam_size=beam_size,
                                                batch_size=10_000 * 160,
                                                bpe_size=BPE_10K,
                                                use_sclite=True,
                                                test_dataset_tuples=get_ted2_test_dataset_tuples(BPE_10K),
                                            )
