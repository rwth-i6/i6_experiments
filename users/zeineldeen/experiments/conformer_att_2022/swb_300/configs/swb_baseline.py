import copy, os

from sisyphus import *
import numpy

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
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.feature_extraction_net import (
    log10_net_10ms,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.swb_300.pipeline import (
    training,
    search,
    get_average_checkpoint,
    get_best_checkpoint,
    search_single,
)
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import ilm_helpers
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import get_lm, ZeineldeenLM

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000
BPE_500 = 500

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

dev_datasets = ["hub5e00"]
test_datasets = ["hub5e01", "rt03s"]


def conformer_baseline():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def get_test_dataset_tuples(bpe_size, selected_datasets=None):
        test_dataset_tuples = {}
        for testset in ["hub5e00", "hub5e01", "rt03s"]:
            if selected_datasets and testset not in selected_datasets:
                continue
            test_dataset_tuples[testset] = build_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
        return test_dataset_tuples

    def run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )
        train_job = training(
            exp_prefix, returnn_config, RETURNN_CPU_EXE, kwargs.get("returnn_root", RETURNN_ROOT), num_epochs=num_epochs
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
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
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
        else:
            assert isinstance(epoch, int), "epoch must be either a defined integer or a string in {avg, best}."
            search_checkpoint = train_job.out_checkpoints[epoch]

        ext_lm_opts = lstm_lm_opts_map[bpe_size] if lm_type == "lstm" else trafo_lm_opts_map[bpe_size]

        time_rqmt = 1.0

        search_args = copy.deepcopy(args)

        if lm_type == "lstm":
            if beam_size > 128:
                search_args["batch_size"] = 4000 * 80

        if lm_type == "trafo":
            search_args["batch_size"] = 4000 * 80 if beam_size <= 32 else 2000 * 80
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
                            {"mini_lstm": {"filename": mini_lstm_ckpt, "prefix": "mini_"}}
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
                    time_rqmt=kwargs.get("time_rqmt", time_rqmt),
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
        selected_test_datasets,
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
            returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
            num_average=num_avg,
        )
        train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            default_recog_epochs = [40] + [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)
        selected_test_dataset_tuples = get_test_dataset_tuples(bpe_size, selected_datasets=selected_test_datasets)

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}",
                returnn_search_config,
                train_job.out_checkpoints[ep],
                selected_test_dataset_tuples,
                RETURNN_CPU_EXE,
                kwargs.get("returnn_root", RETURNN_ROOT),
            )

        search(
            exp_prefix + "/default_last",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            kwargs.get("returnn_root", RETURNN_ROOT),
        )

        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            kwargs.get("returnn_root", RETURNN_ROOT),
        )

        search(
            exp_prefix + f"/average_{num_avg}",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            kwargs.get("returnn_root", RETURNN_ROOT),
        )

    def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        bpe_size=500,
        selected_test_datasets=None,
        **kwargs,
    ):
        if train_args.get("retrain_checkpoint", None):
            assert kwargs.get("epoch_wise_filter", None) is None, "epoch_wise_filter should be disabled for retraining."
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", None),
            link_speed_perturbation=train_args.get("speed_pert", False),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        train_job = run_train(
            exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs
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
            selected_test_datasets=selected_test_datasets,
            **kwargs,
        )
        return train_job, train_data

    def compute_features_stats(
        output_dirname, feat_dim, bpe_size=500, feature_extraction_net=log10_net_10ms, model_checkpoint=None, **kwargs
    ):
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,
            seq_ordering="laplace:.1000",
            partition_epoch=1,
        )
        # Dump log-mel features into HDFDataset
        dump_features_config = {}
        dump_features_config["extern_data"] = train_data.extern_data
        dump_features_config["network"] = copy.deepcopy(feature_extraction_net)
        if model_checkpoint:
            dump_features_config["network"]["output"] = {
                "class": "hdf_dump",
                "from": "log_mel_features",
                "filename": "log_mel_features.hdf",
            }
        else:
            dump_features_config["network"]["output"] = {
                "class": "copy",
                "from": "log_mel_features",
            }
        dump_features_config["forward_batch_size"] = 20_000 * 80
        dump_features_config["eval"] = train_data.train.as_returnn_opts()
        from i6_core.returnn import ReturnnForwardJob, ReturnnConfig

        hdf_filename = "log_mel_features.hdf" if model_checkpoint else "output.hdf"

        dump_features_job = ReturnnForwardJob(
            returnn_config=ReturnnConfig(config=dump_features_config),
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
            model_checkpoint=model_checkpoint,
            hdf_outputs=[hdf_filename] if model_checkpoint else [],
            device="cpu",
            mem_rqmt=15,
            time_rqmt=72,
            eval_mode=True if model_checkpoint else False,
        )
        dump_features_job.add_alias(f"swb_stats/{output_dirname}/dump_train_log_mel_features")
        tk.register_output(
            f"swb_stats/{output_dirname}/log_mel_features.hdf", dump_features_job.out_hdf_files[hdf_filename]
        )

        # Extract features stats from HDFDataset
        extract_stats_returnn_config = ReturnnConfig(
            {
                "extern_data": {
                    "data": {"dim": feat_dim},
                },
                "train": {
                    "class": "HDFDataset",
                    "files": [dump_features_job.out_hdf_files[hdf_filename]],
                    "use_cache_manager": True,
                },
                "batch_size": 20_000 * 80,
            }
        )
        from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob

        extract_mean_stddev_job = ExtractDatasetMeanStddevJob(
            returnn_config=extract_stats_returnn_config,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
        )
        extract_mean_stddev_job.add_alias(f"swb_stats/{output_dirname}/extract_mean_stddev")

        tk.register_output(f"swb_stats/{output_dirname}/mean_var", extract_mean_stddev_job.out_mean)
        tk.register_output(f"swb_stats/{output_dirname}/std_dev_var", extract_mean_stddev_job.out_std_dev)
        tk.register_output(f"swb_stats/{output_dirname}/mean_file", extract_mean_stddev_job.out_mean_file)
        tk.register_output(f"swb_stats/{output_dirname}/std_dev_file", extract_mean_stddev_job.out_std_dev_file)

        return (
            extract_mean_stddev_job.out_mean,
            extract_mean_stddev_job.out_std_dev,
            extract_mean_stddev_job.out_mean_file,
            extract_mean_stddev_job.out_std_dev_file,
        )

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
        mini_lstm_args["batch_size"] = 20000 * 80
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
            exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs, time_rqmt=time_rqmt
        )
        return train_job

    def train_mini_self_att(
        exp_name, checkpoint, args, num_epochs=20, lr=8e-4, time_rqmt=4, name="mini_self_att", **kwargs
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
        mini_self_att["batch_size"] = 20000 * 80  # TODO: does this fit now?
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
            exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs, time_rqmt=time_rqmt
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
    trafo_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 20000 * 80}
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 12000 * 80  # frames * samples per frame

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
    lstm_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 80}
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 15000 * 80  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    # - better with global norm
    # - converge better when not reducing depthwise conv kernel size during pretraining but final WER is worse

    oclr_args = copy.deepcopy(lstm_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 9e-4,
        "final_lr": 1e-6,
        "cycle_ep": 135,
        "total_ep": 300,  # 50 epochs
        "n_step": 2085,  # without using max_seq_len and laplace:6000
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args["encoder_args"].use_sqrd_relu = True

    _, _, mean, stddev = compute_features_stats(
        output_dirname="logmel_50", feat_dim=80
    )  # I found later that feat_dim does not matter. 80 is used here to not break hashes

    base_v1_args = copy.deepcopy(oclr_args)
    base_v1_args["global_stats"] = (mean, stddev)
    base_v1_args["encoder_args"].input_layer = "conv-6"
    base_v1_args["oclr_opts"]["peak_lr"] = 1e-3
    base_v1_args["max_seq_length"] = 75
    base_v1_args["oclr_opts"]["n_step"] = 2012

    # best: 8l, 12l with 0.7 dim reduction

    def get_modified_base_args(args, num_blocks, dim_reduce_factor):
        new_args = copy.deepcopy(args)
        new_args["encoder_args"].num_blocks = num_blocks
        reduced_att_heads = int(new_args["encoder_args"].att_num_heads * dim_reduce_factor)
        enc_key_dim = (
            int(new_args["encoder_args"].enc_key_dim * dim_reduce_factor / float(reduced_att_heads)) * reduced_att_heads
        )
        new_args["encoder_args"].enc_key_dim = enc_key_dim
        new_args["encoder_args"].ff_dim = 4 * enc_key_dim
        new_args["encoder_args"].att_num_heads = reduced_att_heads
        return new_args

    def run_default_exp(
        name,
        train_args,
        num_epochs=300,
        epoch_wise_filter=None,
        seq_ordering="laplace:6000",
        feature_extraction_net=log10_net_10ms,
    ):
        return run_exp(
            name,
            train_args=train_args,
            num_epochs=num_epochs,
            epoch_wise_filter=epoch_wise_filter,
            seq_ordering=seq_ordering,
            selected_test_datasets=dev_datasets,
            feature_extraction_net=feature_extraction_net,
        )

    base_12l_reduce_0_7_args = get_modified_base_args(base_v1_args, 12, 0.7)
    run_default_exp(
        "base_conf_12l_lstm_1l_conv6_sqrdReLU_peak0.001_bpe500_maxSeqLen75_dimReduce0.7_ep300",
        train_args=base_12l_reduce_0_7_args,
    )

    args = copy.deepcopy(base_12l_reduce_0_7_args)
    args["oclr_opts"]["n_step"] = 2085
    args["max_seq_length"] = None
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_peak{1e-3}_bpe500_dimReduce{0.7}_noMaxSeqLen",
        train_args=args,
    )

    # args = copy.deepcopy(base_12l_reduce_0_7_args)
    # args["oclr_opts"]["n_step"] = 2082
    # args["max_seq_length"] = 100
    # run_default_exp(
    #     f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_peak{1e-3}_bpe500_dimReduce{0.7}_maxSeqLen100_step2082",
    #     train_args=args,
    # )

    # TODO: more layers
    # 12, 0.7 dim: 48M, full dim: 87M
    for enc_layers in [16]:
        args = copy.deepcopy(base_12l_reduce_0_7_args)
        args["encoder_args"].num_blocks = enc_layers
        if enc_layers >= 16:
            args["recursion_limit"] = 4000
        run_default_exp(
            f"base_conf_{enc_layers}l_lstm_1l_conv{6}_sqrdReLU_peak{1e-3}_bpe500_maxSeqLen75_dimReduce{0.7}",
            train_args=args,
        )

    # TODO: longer training
    for ep in [100 * 6, 150 * 6, 200 * 6]:
        for max_seq_len, n_step in [(75, 2012), (100, 2082), (None, 2085)]:
            args = copy.deepcopy(base_12l_reduce_0_7_args)
            args["oclr_opts"]["cycle_ep"] = int(0.45 * ep)
            args["oclr_opts"]["total_ep"] = ep
            args["max_seq_length"] = max_seq_len
            args["oclr_opts"]["n_step"] = n_step
            run_default_exp(
                f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_peak{1e-3}_bpe500_maxSeqLen{max_seq_len}_dimReduce{0.7}_ep{ep}",
                train_args=args,
                num_epochs=ep,
            )

    # TODO: shuff
    # for shuff in ["laplace:.40", "laplace:.42"]:
    #     run_default_exp(
    #         f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_shuff{shuff}",
    #         train_args=base_12l_reduce_0_7_args,
    #         seq_ordering=shuff,
    #     )

    # TODO: fix depthwise conv kernel size during pretraining
    # worse
    # args = copy.deepcopy(base_12l_reduce_0_7_args)
    # args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
    # run_default_exp(
    #     f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_woDepthwiseConvPre",
    #     train_args=args,
    # )
    args = copy.deepcopy(base_v1_args)
    args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{1.0}_woDepthwiseConvPre",
        train_args=args,
    )

    # TODO: feature dim. default is 50
    for dim in [40, 60, 80]:
        args = copy.deepcopy(base_12l_reduce_0_7_args)
        new_feat_net = copy.deepcopy(log10_net_10ms)
        new_feat_net["mel_filterbank"]["nr_of_filters"] = dim
        new_feat_net["mel_filterbank"]["n_out"] = dim
        _, _, mean, stddev = compute_features_stats(
            feature_extraction_net=new_feat_net, feat_dim=dim, output_dirname=f"logmel_{dim}"
        )
        args["global_stats"] = (mean, stddev)
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_featDim{dim}",
            train_args=args,
            feature_extraction_net=new_feat_net,
        )

    # TODO: avg-OCLR epoch-based
    args = copy.deepcopy(base_12l_reduce_0_7_args)
    oclr_n_step = args["oclr_opts"]["n_step"]
    args.pop("oclr_opts")
    lrs = numpy.concatenate(
        [
            numpy.linspace(1e-4, 1e-3, 135 * oclr_n_step),
            numpy.linspace(1e-3, 1e-4, 135 * oclr_n_step),
            numpy.linspace(1e-4, 1e-6, 30 * oclr_n_step),
        ]
    )  # (num_epochs * n_step,)
    args["learning_rates_list"] = list(numpy.mean(lrs.reshape(-1, oclr_n_step), axis=-1))
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_ep{300}_avgEpochOCLR",
        train_args=args,
    )

    # TODO: wo initial batch size during pretraining
    args = copy.deepcopy(base_12l_reduce_0_7_args)
    args["pretrain_opts"]["initial_batch_size"] = None
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_ep{300}_woPreInitBs",
        train_args=args,
    )

    # TODO: less reps
    for reps in [2, 3, 4]:
        args = copy.deepcopy(base_v1_args)
        args["pretrain_reps"] = reps
        args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{1.0}_woDepthwiseConvPre_preReps{reps}",
            train_args=args,
        )

        args = copy.deepcopy(base_12l_reduce_0_7_args)
        args["pretrain_reps"] = reps
        args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen{75}_dimReduce{0.7}_woDepthwiseConvPre_preReps{reps}",
            train_args=args,
        )

        args = copy.deepcopy(base_12l_reduce_0_7_args)
        args["pretrain_reps"] = reps
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen{75}_dimReduce{0.7}_preReps{reps}",
            train_args=args,
        )

    # TODO: without initial bs pre + no depthwise conv pre
    for i, tmp_args in enumerate([base_12l_reduce_0_7_args, base_v1_args]):
        args = copy.deepcopy(tmp_args)
        args["pretrain_opts"]["initial_batch_size"] = None
        args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
        reduce_dim = 0.7 if i == 0 else 1.0
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{reduce_dim}_ep{300}_woPreInitBs_woDepthwiseConvPre",
            train_args=args,
        )

    # TODO: epoch-based OCLR
    args = copy.deepcopy(base_12l_reduce_0_7_args)
    args.pop("oclr_opts")
    args["learning_rates_list"] = (
        list(numpy.linspace(1e-4, 1e-3, 135))
        + list(numpy.linspace(1e-3, 1e-4, 135))
        + list(numpy.linspace(1e-4, 1e-6, 30))
    )
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_ep{300}_epochOCLR",
        train_args=args,
    )

    for reduce_factor in [0.75, 0.8]:
        args = get_modified_base_args(base_v1_args, num_blocks=12, dim_reduce_factor=reduce_factor)
        args["max_seq_length"] = None
        args["oclr_opts"]["n_step"] = 2085
        run_default_exp(
            f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{reduce_factor}_ep{300}",
            train_args=args,
        )

    args = copy.deepcopy(base_12l_reduce_0_7_args)
    args["decoder_args"].use_zoneout_output = True
    run_default_exp(
        f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_maxSeqLen75_dimReduce{0.7}_ep{300}_fixedZoneout",
        train_args=args,
    )

    # TODO: long training
    # for peak_lr in [7e-4, 8e-4, 9e-4, 2e-3]:
    #     args = copy.deepcopy(base_12l_reduce_0_7_args)
    #     args["max_seq_length"] = None
    #     args["oclr_opts"]["cycle_ep"] = int(0.45 * 600)
    #     args["oclr_opts"]["total_ep"] = 600
    #     args["oclr_opts"]["peak_lr"] = peak_lr
    #     run_default_exp(
    #         f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_bpe500_noMaxSeqLen_dimReduce{0.7}_ep{600}_peakLR{peak_lr}",
    #         train_args=args,
    #         num_epochs=600,
    #     )

    # for reps in [1, 2, 3, 4, 5]:
    #     args = copy.deepcopy(base_12l_reduce_0_7_args)
    #     args["oclr_opts"]["n_step"] = 2085
    #     args["max_seq_length"] = None
    #     args["pretrain_reps"] = reps
    #     args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = ["conv_kernel_size"]
    #     run_default_exp(
    #         f"base_conf_{12}l_lstm_1l_conv{6}_sqrdReLU_peak{1e-3}_bpe500_dimReduce{0.7}_noMaxSeqLen_woDepthwiseConvPre_preReps{reps}",
    #         train_args=args,
    #     )
