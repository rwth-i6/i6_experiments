import copy, os

import numpy

from i6_core.returnn.training import Checkpoint

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
    reset_params_init,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import (
    RETURNN_EXE,
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
        prior_scales=None,
        prior_type=None,
        mini_lstm_ckpt=None,
        length_norm=True,
        prior_type_name=None,
        coverage_scale=None,
        coverage_threshold=None,
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
        elif isinstance(epoch, Checkpoint):
            search_checkpoint = epoch
            assert "ckpt_name" in kwargs
            epoch = kwargs["ckpt_name"]
        else:
            assert isinstance(
                epoch, int
            ), "epoch must be either a defined integer or a `Checkpoint` instance or a string in {avg, best}."
            search_checkpoint = train_job.out_checkpoints[epoch]

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
                if prior_scale and prior_scale > lm_scale:
                    continue

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
                    lm_desc += f"_coverage-thre{coverage_threshold}-scale{coverage_scale}"

                name = f"{exp_name}/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_set}"

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
                    two_pass_rescore=kwargs.get("two_pass_rescore", False),
                    att_scale=kwargs.get("att_scale", 1.0),
                    ctc_scale=kwargs.get("ctc_scale", 1.0),
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

    def train_mini_lstm(
        exp_name,
        checkpoint,
        args,
        num_epochs=20,
        lr=8e-4,
        time_rqmt=4,
        l2=1e-4,
        name="mini_lstm",
        w_drop=False,
        use_dec_state=False,
        use_ffn=False,
        ffn_opts=None,
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
    train_j, train_data = run_exp(name, train_args=oclr_args, num_epochs=2035)

    # CTC greedy decoding implemented in returnn using beam search of beam size 1
    # dev-other: 6.9 without LM.
    run_decoding(
        exp_name="test_ctc_greedy",
        train_data=train_data,
        checkpoint=train_job_avg_ckpt["base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"],
        search_args={"ctc_greedy_decode": True, **oclr_args},
        feature_extraction_net=log10_net_10ms,
        bpe_size=BPE_10K,
        test_sets=["dev-other"],
        remove_label={"<s>", "<blank>"},  # blanks are removed in the network
        use_sclite=True,
    )

    # Att baseline with avg checkpoint: 2.27/5.39/2.41/5.51
    retrain_args = copy.deepcopy(oclr_args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt[name]
    retrain_args["learning_rates_list"] = [1e-4] * 20 + list(numpy.linspace(1e-4, 1e-6, 580))
    retrain_args["lr_decay"] = 0.95
    train_j, train_data = run_exp(
        exp_name=name + f"_retrain1_const20_linDecay580_{1e-4}",
        train_args=retrain_args,
        num_epochs=600,
    )

    # 2.86/6.7/3.07/6.96
    run_decoding(
        exp_name="test_ctc_greedy_best",
        train_data=train_data,
        checkpoint=train_job_avg_ckpt[
            f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
        ],
        search_args={"ctc_greedy_decode": True, **oclr_args},
        feature_extraction_net=log10_net_10ms,
        bpe_size=BPE_10K,
        test_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
        remove_label={"<s>", "<blank>"},  # blanks are removed in the network
        use_sclite=True,
    )

    def debug(name, search_bpe_path):
        from i6_core.returnn.search import SearchRemoveLabelJob
        from i6_core.returnn.search import SearchBPEtoWordsJob, ReturnnComputeWERJob
        import sisyphus.toolkit as tk

        assert isinstance(search_bpe_path, str)
        search_bpe_path = tk.Path(search_bpe_path, hash_overwrite=name)
        recognition_reference = tk.Path("/u/zeineldeen/debugging/trigg_att/refs.py")
        search_bpe = SearchRemoveLabelJob(search_bpe_path, remove_label="<s>", output_gzip=True).out_search_results
        search_words = SearchBPEtoWordsJob(search_bpe).out_word_search_results
        tk.register_output(f"ctc_att_search/debug/{name}_words", search_words)
        wer = ReturnnComputeWERJob(search_words, recognition_reference).out_wer
        tk.register_output(f"ctc_att_search/debug/{name}_wer", wer)

    debug("fixrepeat_v1", "/u/zeineldeen/debugging/trigg_att/out.txt")

    # TODO: two-pass joint decoding with CTC
    for beam_size in [12]:
        for ctc_scale in [0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]:
            att_scale = 1.0
            run_decoding(
                exp_name=f"two_pass_ctcRescore_{att_scale}_{ctc_scale}_beam{beam_size}",
                train_data=train_data,
                checkpoint=train_job_avg_ckpt[
                    f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
                ],
                search_args={"beam_size": beam_size, **oclr_args},
                feature_extraction_net=log10_net_10ms,
                bpe_size=BPE_10K,
                test_sets=["dev-clean", "dev-other", "test-clean", "test-other"],
                use_sclite=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                two_pass_rescore=True,  # two-pass rescoring
            )

    # TODO: two-pass joint decoding with CTC with LM
    for beam_size in [12, 32]:
        for ctc_scale in [0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0]:
            for lm_scale in [0.28, 0.3, 0.32, 0.35, 0.38, 0.4, 0.42]:
                att_scale = 1.0
                run_lm_fusion(
                    args=oclr_args,
                    lm_type="lstm",
                    exp_name=f"two_pass_ctcRescore_{att_scale}_{ctc_scale}_lstmLM{lm_scale}_beam{beam_size}",
                    train_data=train_data,
                    train_job=train_j,
                    feature_net=log10_net_10ms,
                    epoch=train_job_avg_ckpt[
                        f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
                    ],
                    ckpt_name="avg",
                    lm_scales=[lm_scale],
                    beam_size=beam_size,
                    bpe_size=BPE_10K,
                    test_set_names=["dev-other"],
                    use_sclite=True,
                    att_scale=att_scale,
                    ctc_scale=ctc_scale,
                    two_pass_rescore=True,  # two-pass rescoring
                )

    # TODO: one-pass joint decoding with CTC
    # for comb_score_version in [2]:
    #     for beam_size in [12]:
    #         for scale in [(1.0, 0.3)]:
    #             if isinstance(scale, tuple):
    #                 att_scale, ctc_scale = scale
    #             else:
    #                 assert isinstance(scale, float)
    #                 att_scale = scale
    #                 ctc_scale = 1.0 - scale
    #
    #             exp_name = f"joint_att_ctc_attScale{att_scale}_ctcScale{ctc_scale}_beam{beam_size}_combScoreV{comb_score_version}_fixRepeat"
    #             if only_scale_comb:
    #                 exp_name += "_onlyScaleComb"
    #             if scale_outside:
    #                 exp_name += "_scaleOutside"
    #             joint_decode_args = {
    #                 "att_scale": att_scale,
    #                 "ctc_scale": ctc_scale,
    #                 "beam_size": beam_size,
    #                 "comb_score_version": comb_score_version,
    #                 "only_scale_comb": only_scale_comb,
    #                 "scale_outside": scale_outside,
    #             }
    #             run_decoding(
    #                 exp_name=exp_name,
    #                 train_data=train_data,
    #                 checkpoint=train_job_avg_ckpt[
    #                     f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009_retrain1_const20_linDecay580_{1e-4}"
    #                 ],
    #                 search_args={
    #                     "joint_ctc_att_decode_args": joint_decode_args,
    #                     "batch_size": 10_000 * 160 if beam_size <= 128 else 15_000 * 160,
    #                     **oclr_args,
    #                 },
    #                 feature_extraction_net=log10_net_10ms,
    #                 bpe_size=BPE_10K,
    #                 test_sets=["dev-other"],
    #                 remove_label={"<s>", "<blank>"},  # blanks are removed in the network
    #                 use_sclite=True,
    #                 time_rqmt=1.0 if beam_size <= 128 else 1.5,
    #             )
