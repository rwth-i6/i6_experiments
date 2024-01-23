import copy, os

import numpy
import sisyphus.toolkit as tk

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
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import (
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
from i6_experiments.users.berger.recipe.summary import SummaryReport

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

jobs_summary_reports = {}  # dict[str, SummaryReport]

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

    def run_seq_train(
        exp_name,
        seq_train_opts,
        train_args,
        num_epochs,
        bpe_size,
        ckpt_select_score_key,
        feature_extraction_net=log10_net_10ms,
        time_rqmt=72,
    ):
        assert seq_train_opts is not None, "seq_train_opts must be set"
        exp_prefix = os.path.join(prefix_name, exp_name)
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering="laplace:.1000",
        )
        returnn_config = create_config(
            training_datasets=train_data,
            feature_extraction_net=feature_extraction_net,
            seq_train_opts=seq_train_opts,
            **train_args,
        )
        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )

        test_dataset_tuples = get_test_dataset_tuples(bpe_size)
        returnn_search_config = create_config(
            training_datasets=train_data,
            feature_extraction_net=feature_extraction_net,
            seq_train_opts=None,
            **train_args,
        )
        best_checkpoint = get_best_checkpoint(train_job, key=ckpt_select_score_key)
        train_job_best_epoch[exp_name] = best_checkpoint
        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            use_sclite=True,
            enable_mail=True,
        )
        averaged_checkpoint = get_average_checkpoint(
            train_job,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            num_average=4,
            key=ckpt_select_score_key,
        )
        train_job_avg_ckpt[exp_name] = averaged_checkpoint
        search(
            exp_prefix + f"/average_4",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            use_sclite=True,
            enable_mail=True,
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
        wer = search_single(
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
        return wer

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
        else:
            assert isinstance(epoch, int), "epoch must be either a defined integer or a string in {avg, best}."
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

                wer = run_single_search(
                    exp_name=name,
                    train_data=train_data,
                    search_args=search_args,
                    checkpoint=search_checkpoint,
                    feature_extraction_net=feature_net,
                    recog_dataset=test_dataset_tuples[test_set][0],
                    recog_ref=test_dataset_tuples[test_set][1],
                    recog_bliss=test_dataset_tuples[test_set][2],
                    time_rqmt=kwargs.get("time_rqmt", time_rqmt),
                )

                if exp_name not in jobs_summary_reports:
                    jobs_summary_reports[exp_name] = SummaryReport(
                        col_names=["test_set", "lm_scale", "prior_scale", "beam_size", "wer"], col_sort_key="wer"
                    )

                jobs_summary_reports[exp_name].add_row(
                    {
                        "test_set": test_set,
                        "lm_scale": lm_scale,
                        "prior_scale": prior_scale if prior_scale is not None else "-",
                        "beam_size": beam_size,
                        "wer": wer,
                    }
                )
                tk.register_report(f"{prefix_name}/{exp_name}/summary", jobs_summary_reports[exp_name])

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
        frontend_conv_l2=0.0001,
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

    # best: 2.28/5.63/2.48/5.71
    # Avg: 2.28/5.6/2.48/5.75
    name = "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"
    run_exp(name, train_args=oclr_args, num_epochs=2035)

    # Avg: 2.27/5.39/2.41/5.51
    retrain_args = copy.deepcopy(oclr_args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt[name]
    retrain_args["learning_rates_list"] = [1e-4] * 20 + list(numpy.linspace(1e-4, 1e-6, 580))
    retrain_args["lr_decay"] = 0.95
    retrain_exp_name = name + f"_retrain1_const20_linDecay580_{1e-4}"
    train_j, train_data = run_exp(
        exp_name=retrain_exp_name,
        train_args=retrain_args,
        num_epochs=600,
    )

    # for beam_size in [32, 40, 45, 50, 55, 60, 65, 70]:
    #     for lm_scale in [0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]:
    #         run_lm_fusion(
    #             lm_type="trafo",
    #             exp_name=retrain_exp_name,
    #             epoch="avg",
    #             test_set_names=["dev-clean", "dev-other"],
    #             lm_scales=[lm_scale],
    #             train_job=train_j,
    #             train_data=train_data,
    #             feature_net=log10_net_10ms,
    #             args=oclr_args,
    #             beam_size=beam_size,
    #             batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
    #             bpe_size=BPE_10K,
    #         )

    mini_lstm_j = train_mini_lstm(
        exp_name=retrain_exp_name,
        checkpoint=train_job_avg_ckpt[retrain_exp_name],
        args=oclr_args,
        num_epochs=40,
        w_drop=True,
    )

    for beam_size in [32]:
        for lm_scale in [0.4, 0.42, 0.44, 0.46]:
            for prior_scale in [0.3, 0.32, 0.34, 0.36]:
                run_lm_fusion(
                    lm_type="trafo",
                    exp_name=retrain_exp_name,
                    epoch="avg",
                    test_set_names=["dev-clean", "dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key="dev_score"),
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                )

    for test_set, lm, prior, beam in [("test-other", 0.54, 0.38, 70), ("test-clean", 0.48, 0.37, 32)]:
        run_lm_fusion(
            lm_type="trafo",
            exp_name=retrain_exp_name,
            epoch="avg",
            test_set_names=[test_set],
            lm_scales=[lm],
            prior_scales=[prior],
            prior_type="mini_lstm",
            mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key="dev_score"),
            train_job=train_j,
            train_data=train_data,
            feature_net=log10_net_10ms,
            args=oclr_args,
            beam_size=beam,
            batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
            bpe_size=BPE_10K,
        )

    # --------------------------- Seq. Training --------------------------- #

    # TODO: double softmax
    for total_ep, lr, const_ep in [(200, 1e-4, 20), (200, 1e-4, 40)]:
        for abs_scale, rel_scale, ce_scale in [
            (2.0, 0.4, 0.0),
            (2.0, 0.35, 0.0),
            (3.0, 0.35, 0.0),
            (2.0, 0.25, 0.0),
            (3.0, 0.35, 0.1),
        ]:
            am_scale = abs_scale
            lm_scale = rel_scale * am_scale
            seq_train_opts = {
                "type": "double_softmax",
                "loss_scale": 1.0,
                "am_scale": am_scale,
                "lm_scale": lm_scale,
                "ce_scale": ce_scale,
            }
            args = copy.deepcopy(retrain_args)
            args["learning_rates_list"] = [lr] * const_ep + list(numpy.linspace(lr, 1e-6, total_ep - const_ep))
            train_j = run_seq_train(
                exp_name=f"att_retrain1_doubleSoftmax_am{am_scale}_lm{lm_scale}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
                seq_train_opts=seq_train_opts,
                train_args=args,
                num_epochs=total_ep,
                bpe_size=BPE_10K,
                ckpt_select_score_key="dev_score_output/double_softmax_loss",
            )

            # att_retrain1_doubleSoftmax_am2.0_lm0.5_transLM_ep200_lr0.0001_const20_ce0.0
            # Avg: dev-clean: 2.49 - dev-other: 6.02 - test-clean: 2.69 - test-other: 6.16
            # if am_scale == 2 and rel_scale == 0.25 and ce_scale == 0.0:
            for search_beam in [40]:
                for search_lm_scale in [0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]:
                    run_lm_fusion(
                        lm_type="trafo",
                        exp_name=f"att_retrain1_doubleSoftmax_am{am_scale}_lm{lm_scale}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
                        epoch="avg",
                        test_set_names=["dev-other"],
                        lm_scales=[search_lm_scale],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=search_beam,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                    )

    # TODO: min_wer
    # beam size 4 and 5 seqs per batch use 9.4 GB GPU mem
    # beam size 8 and 3 seqs per batch use 8.1 GB GPU mem

    # att_retrain1_minWER_am1.0_lm0.3_beam8_transLM_ep20_lr2e-05_const20_ce0.01/recog-trafo-lm/ep-avg/lm-scale-0.4-beam-40/dev-other/wer
    # 4.14

    for total_ep, lr, const_ep in [(20, 2e-5, 20), (20, 3e-5, 20)]:
        for abs_scale, rel_scale, ce_scale, beam in [
            (1.0, 0.3, 0.01, 8),
            (1.0, 0.35, 0.01, 8),
            (1.0, 0.4, 0.01, 8),
        ]:
            am_scale = abs_scale
            lm_scale = rel_scale * am_scale
            seq_train_opts = {
                "type": "min_wer",
                "loss_scale": 1.0,
                "am_scale": am_scale,
                "lm_scale": lm_scale,
                "ce_scale": ce_scale,
                "beam_size": beam,
            }

            args = copy.deepcopy(retrain_args)
            args["accum_grad"] = 3
            args["max_seqs"] = 3

            args["learning_rates_list"] = [lr] * const_ep + list(numpy.linspace(lr, 1e-6, total_ep - const_ep))
            train_j = run_seq_train(
                exp_name=f"att_retrain1_minWER_am{am_scale}_lm{lm_scale}_beam{beam}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
                seq_train_opts=seq_train_opts,
                train_args=args,
                num_epochs=total_ep,
                bpe_size=BPE_10K,
                time_rqmt=total_ep + 4,
                ckpt_select_score_key="dev_score_min_wer",
            )

            for search_beam in [40]:
                for search_lm_scale in [0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]:
                    run_lm_fusion(
                        lm_type="trafo",
                        exp_name=f"att_retrain1_minWER_am{am_scale}_lm{lm_scale}_beam{beam}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
                        epoch="avg",
                        test_set_names=["dev-other"],
                        lm_scales=[search_lm_scale],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=search_beam,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                    )

            # att_retrain1_minWER_am1.0_lm0.3_beam8_transLM_ep20_lr2e-05_const20_ce0.01
            # if lr == 2e-5 and abs_scale == 1.0 and rel_scale == 0.3 and ce_scale == 0.01:
            #     mini_lstm_j = train_mini_lstm(
            #         exp_name=f"att_retrain1_minWER_am{am_scale}_lm{lm_scale}_beam{beam}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
            #         checkpoint=train_job_avg_ckpt[
            #             f"att_retrain1_minWER_am{am_scale}_lm{lm_scale}_beam{beam}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}"
            #         ],
            #         args=oclr_args,
            #         num_epochs=40,
            #         w_drop=True,
            #     )
            #
            #     for search_beam in [40, 50, 70]:
            #         for search_lm_scale in [0.58, 0.6, 0.62, 0.64, 0.66, 0.68]:
            #             for search_ilm_scale in [0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.45]:
            #                 run_lm_fusion(
            #                     lm_type="trafo",
            #                     exp_name=f"att_retrain1_minWER_am{am_scale}_lm{lm_scale}_beam{beam}_transLM_ep{total_ep}_lr{lr}_const{const_ep}_ce{ce_scale}",
            #                     epoch="avg",
            #                     test_set_names=["dev-other"],
            #                     lm_scales=[search_lm_scale],
            #                     train_job=train_j,
            #                     train_data=train_data,
            #                     feature_net=log10_net_10ms,
            #                     args=oclr_args,
            #                     beam_size=search_beam,
            #                     batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
            #                     bpe_size=BPE_10K,
            #                     prior_type="mini_lstm",
            #                     prior_scales=[search_ilm_scale],
            #                     mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key="dev_score"),
            #                 )

    # TODO: MMI
