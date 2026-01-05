from sisyphus import tk

from typing import Optional

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
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.rossenbach.experiments.domain_testing_2024_tf.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
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


def mohammad_baseline():
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
            RETURNN_EXE,
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
                returnn_exe=RETURNN_EXE,
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
                returnn_exe=RETURNN_EXE,
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
        length_norm_exponent=1.0,
        prior_type_name=None,
        coverage_scale=None,
        coverage_threshold=None,
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
        elif isinstance(epoch, Checkpoint):
            search_checkpoint = epoch
            assert "ckpt_name" in kwargs
            epoch = kwargs["ckpt_name"]
        else:
            assert isinstance(
                epoch, int
            ), "epoch must be either a defined integer or a `Checkpoint` instance or a string in {avg, best}."
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
        elif length_norm_exponent != 1.0:
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
                if length_norm_exponent != 1.0:
                    lm_desc += "-LenNorm%.2f" % length_norm_exponent

                if coverage_scale and coverage_threshold:
                    assert isinstance(search_args["decoder_args"], RNNDecoderArgs)
                    search_args["decoder_args"].coverage_scale = coverage_scale
                    search_args["decoder_args"].coverage_threshold = coverage_threshold
                    lm_desc += f"_coverage-thre{coverage_threshold}-scale{coverage_scale}"

                if ext_lm_opts:
                    name = f"{exp_name}/recog-{ext_lm_opts['name']}-lm/ep-{epoch}/{lm_desc}/{test_set}"
                else:
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
                    #att_scale=kwargs.get("att_scale", 1.0),
                    #ctc_scale=kwargs.get("ctc_scale", 1.0),
                )


    def run_lm_fusion_custom(
        lm_type,
        exp_name,
        epoch,
        test_sets: dict[str, tuple],
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
        length_norm_exponent=1.0,
        prior_type_name=None,
        coverage_scale=None,
        coverage_threshold=None,
        ext_lm_opts=None,
        **kwargs,
    ):
        assert lm_type in ["lstm", "trafo"], "lm type should be lstm or trafo"

        if isinstance(lm_scales, float):
            lm_scales = [lm_scales]
        if prior_scales and isinstance(prior_scales, float):
            prior_scales = [prior_scales]

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
        elif length_norm_exponent != 1.0:
            search_args["decoder_args"].length_normalization_exponent = length_norm_exponent

        if "decoder_args" in kwargs:
            for k, v in kwargs["decoder_args"].items():
                setattr(search_args["decoder_args"], k, v)

        scales = [(e,) for e in lm_scales]

        for test_name, test_set in test_sets.items():
            if prior_scales:
                import itertools

                scales = itertools.product(lm_scales, prior_scales)

            for scale in scales:
                lm_scale = scale[0]
                prior_scale = scale[1] if len(scale) == 2 else None
                #if prior_scale and prior_scale > lm_scale:
                #    continue

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
                else:
                    lm_desc += "-noILM"
                lm_desc += f"-beam-{beam_size}"
                if length_norm is False:
                    lm_desc += "-woLenNorm"
                if length_norm_exponent != 1.0:
                    lm_desc += "-LenNorm%.2f" % length_norm_exponent

                if coverage_scale and coverage_threshold:
                    assert isinstance(search_args["decoder_args"], RNNDecoderArgs)
                    search_args["decoder_args"].coverage_scale = coverage_scale
                    search_args["decoder_args"].coverage_threshold = coverage_threshold
                    lm_desc += f"_coverage-thre{coverage_threshold}-scale{coverage_scale}"

                if ext_lm_opts:
                    name = f"{exp_name}/recog-{ext_lm_opts['name']}-lm/ep-{epoch}/{lm_desc}/{test_name}"
                else:
                    name = f"{exp_name}/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_name}"

                run_single_search(
                    exp_name=name,
                    train_data=train_data,
                    search_args=search_args,
                    checkpoint=search_checkpoint,
                    feature_extraction_net=feature_net,
                    recog_dataset=test_set[0],
                    recog_ref=test_set[1],
                    recog_bliss=test_set[2],
                    time_rqmt=kwargs.get("time_rqmt", time_rqmt),
                    two_pass_rescore=kwargs.get("two_pass_rescore", False),
                    use_sclite=kwargs.get("use_sclite", False),
                    #att_scale=kwargs.get("att_scale", 1.0),
                    #ctc_scale=kwargs.get("ctc_scale", 1.0),
                )

    def run_decoding(
        exp_name,
        train_data,
        checkpoint,
        search_args,
        feature_extraction_net,
        test_sets: dict[str, tuple],
        time_rqmt: float = 1.0,
        remove_label=None,
        two_pass_rescore=False,
        **kwargs,
    ):
        for test_name, test_set in test_sets.items():
            run_single_search(
                exp_name=exp_name + f"/recogs/{test_name}",
                train_data=train_data,
                search_args=search_args,
                checkpoint=checkpoint,
                feature_extraction_net=feature_extraction_net,
                recog_dataset=test_set[0],
                recog_ref=test_set[1],
                recog_bliss=test_set[2],
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
            returnn_exe=RETURNN_EXE,
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
                RETURNN_EXE,
                RETURNN_ROOT,
            )

        search(
            exp_prefix + "/default_last",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + f"/average_{num_avg}",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_EXE,
            RETURNN_ROOT,
            use_sclite=True,
        )
        return averaged_checkpoint, returnn_search_config

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

        avg_checkpoint, search_config = run_search(
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
        return train_job, train_data, avg_checkpoint, search_config

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
            RETURNN_EXE,
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
            RETURNN_EXE,
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
    train_j, train_data, avg_checkpoint, search_config = run_exp(name, train_args=oclr_args, num_epochs=2035)

    dev_other_noise07 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/dev-other_sequiturg2p_glowtts460_noise07.xml.gz",
        hash_overwrite="dev-other_sequiturg2p_glowtts460_noise07.xml.gz-1"  # did a mistake here, thus a new hash
    )
    dev_other_noise055 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/dev-other_sequiturg2p_glowtts460_noise055.xml.gz",
        hash_overwrite="dev-other_sequiturg2p_glowtts460_noise055.xml.gz"
    )
    dev_other_noise03 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/dev-other_sequiturg2p_glowtts460_noise03.xml.gz",
        hash_overwrite="dev-other_sequiturg2p_glowtts460_noise03.xml.gz"
    )
    medline_wmt22_noise055 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/wmt22_medline_v1_sequiturg2p_glowtts460_noise055.xml.gz",
        hash_overwrite="wmt22_medline_v1_sequiturg2p_glowtts460_noise055.xml.gz"
    )
    medline_wmt22_noise07 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/wmt22_medline_v1_sequiturg2p_glowtts460_noise07.xml.gz",
        hash_overwrite="wmt22_medline_v1_sequiturg2p_glowtts460_noise07.xml.gz"
    )

    MTGv4_dev_noise055 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/MTG_trial4_dev_sequiturg2p_glowtts460_noise055.xml.gz",
        hash_overwrite="MTG_trial4_dev_sequiturg2p_glowtts460_noise055.xml.gz"
    )

    MTGv4_test_noise055 = tk.Path(
        "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/MTG_trial4_test_sequiturg2p_glowtts460_noise055.xml.gz",
        hash_overwrite="MTG_trial4_test_sequiturg2p_glowtts460_noise055.xml.gz"
    )

    medline_test_corpora_055 = {
        wmtyear: tk.Path(
            "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export/wmt%i_medline_v2_sequiturg2p_glowtts460_noise055.xml.gz" % wmtyear,
            hash_overwrite="wmt%i_medline_v2_sequiturg2p_glowtts460_noise055.xml.gz" % wmtyear
        )
        for wmtyear in [21, 23, 24]
    }

    from .data_helper import build_extern_test_dataset_tuple
    dev_other_noise03_tup = build_extern_test_dataset_tuple(bliss=dev_other_noise03)
    dev_other_noise055_tup = build_extern_test_dataset_tuple(bliss=dev_other_noise055)
    dev_other_noise07_tup = build_extern_test_dataset_tuple(bliss=dev_other_noise07)
    medline_wmt22_noise055_tup = build_extern_test_dataset_tuple(bliss=medline_wmt22_noise055)
    medline_wmt_test_noise055_tuples = {
        key: build_extern_test_dataset_tuple(bliss=medline_test_corpora_055[key]) for key in [21, 23, 24]
    }
    medline_wmt22_noise07_tup = build_extern_test_dataset_tuple(bliss=medline_wmt22_noise07)
    MTGv4_dev_noise055_tup = build_extern_test_dataset_tuple(bliss=MTGv4_dev_noise055)
    MTGv4_test_noise055_tup = build_extern_test_dataset_tuple(bliss=MTGv4_test_noise055)


    # Non-LM experiments
    run_decoding(
        exp_name="test_normal_decode",
        train_data=train_data,
        checkpoint=train_job_avg_ckpt["base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak0.0009"],
        search_args={**oclr_args},
        feature_extraction_net=log10_net_10ms,
        test_sets={
            "dev-other_sequiturg2p_glowtts460_noise03": dev_other_noise03_tup,
            "dev-other_sequiturg2p_glowtts460_noise055": dev_other_noise055_tup,
            "dev-other_sequiturg2p_glowtts460_noise07": dev_other_noise07_tup,
            "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
            "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
            "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
            "MTG_trial4_dev_sequiturg2p_glowtts460_noise055": MTGv4_dev_noise055_tup,
        },
        # remove_label={"<s>", "<blank>"},  # blanks are removed in the network
        use_sclite=True,
    )

    mini_lstm = train_mini_lstm(
        name,
        avg_checkpoint,
        args=oclr_args,
        num_epochs=80,
        w_drop=True,
    )
    
    for beam_size in [32]: # , 40, 45, 50]:
        for lm_scale in [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52]:
            for prior_scale in [0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44]:
                run_lm_fusion(
                    lm_type="trafo",
                    exp_name=name,
                    epoch="avg",
                    test_set_names=["dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                )
                
    for beam_size in [32]: # , 40, 45, 50]:
        for lm_scale in [0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52]:
            for prior_scale in [0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44]:
                run_lm_fusion(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="avg",
                    test_set_names=["dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                )

    from ..storage import get_lm_model_as_opts
    test_lm_opts = get_lm_model_as_opts("test_lm")
    ls_2x1k_lm_opts = get_lm_model_as_opts("ls_2x1k")
    for beam_size in [1, 32]: # , 40, 45, 50]:
        for lm_scale in [0.45, 0.50, 0.55, 0.60, 0.65, 0.7]:
            for prior_scale in [0.35, 0.40, 0.45, 0.5]:
                run_lm_fusion(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="avg",
                    test_set_names=["dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=test_lm_opts
                )

    # normal dev-other recognition with
    for beam_size in [32, 64]: # , 40, 45, 50]:
        for lm_scale in [0.45, 0.50, 0.55, 0.60, 0.65, 0.7]:
            for prior_scale in [0.35, 0.40, 0.45, 0.5]:
                exponents = [1.0]
                if lm_scale >= 0.55 and prior_scale >= 0.4:
                    exponents=[0.3, 0.5, 0.7]
                for exp in exponents:
                    run_lm_fusion(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_set_names=["dev-other"],
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=ls_2x1k_lm_opts,
                        length_norm_exponent=exp,
                    )
                    if beam_size == 64 and prior_scale == 0.35 and lm_scale == 0.55 and exp == 1.0:
                        run_lm_fusion(
                            lm_type="lstm",
                            exp_name=name,
                            epoch="best",
                            test_set_names=["test-other"],
                            lm_scales=[lm_scale],
                            prior_scales=[prior_scale],
                            prior_type="mini_lstm",
                            mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                            train_job=train_j,
                            train_data=train_data,
                            feature_net=log10_net_10ms,
                            args=oclr_args,
                            beam_size=beam_size,
                            batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                            bpe_size=BPE_10K,
                            ext_lm_opts=ls_2x1k_lm_opts,
                            length_norm_exponent=exp,
                        )




    for beam_size in [32]: # , 40, 45, 50]:
        for lm_scale in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            for prior_scale in [0.30, 0.35, 0.40, 0.45]:
                run_lm_fusion_custom(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="best",
                    test_sets={
                        "dev-other_sequiturg2p_glowtts460_noise055": dev_other_noise055_tup,
                    },
                    test_set_names=["dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=test_lm_opts
                )

    for beam_size in [32, 64]: # , 40, 45, 50]:
        for lm_scale in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            for prior_scale in [0.30, 0.35, 0.40, 0.45]:
                run_lm_fusion_custom(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="best",
                    test_sets={
                        "dev-other_sequiturg2p_glowtts460_noise055": dev_other_noise055_tup,
                    },
                    test_set_names=["dev-other"],
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=ls_2x1k_lm_opts
                )
   
   
   
    # medline but with LS LM
    for beam_size in [32]: # , 40, 45, 50]:
        for lm_scale in [0.2, 0.3, 0.4, 0.5]:
            for prior_scale in [0.1, 0.2, 0.3, 0.4]:
                for exponent in [0.0, 0.5, 1.0]:
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                            # "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=ls_2x1k_lm_opts,
                        use_sclite=True,
                        length_norm=exponent > 0.0,
                        length_norm_exponent=exponent,
                    )
   
    # medline large search space test
    test_medline_lm_opts = get_lm_model_as_opts("test_medline_lm")
    for beam_size in [1, 32, 64, 128]: # , 40, 45, 50]:
        for lm_scale in [0.60, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9]:
            for prior_scale in [0.50, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                run_lm_fusion_custom(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="best",
                    test_sets={
                        "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                        "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                    },
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=test_medline_lm_opts
                )

    test_medline_lm_opts = get_lm_model_as_opts("medline_2x1k")
    for beam_size in [1, 32, 64]:  # , 40, 45, 50]:
        for lm_scale in [0.75, 0.80, 0.85, 0.9]:
            for prior_scale in [0.65, 0.7, 0.75, 0.8, 0.85]:
                run_lm_fusion_custom(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="best",
                    test_sets={
                        "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                        "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                    },
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=test_medline_lm_opts,
                    use_sclite=True,
                )
                
    # LM length normalization
    test_medline_lm_opts = get_lm_model_as_opts("medline_2x1k")
    for beam_size in [64, 128, 256]:  # , 40, 45, 50]:
        exponents = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.9]
        if beam_size == 256:
            exponents = [0.3, 0.4, 0.5]
        for length_norm_exponent in exponents:
            for lm_scale in [0.80, 0.85, 0.9]:
                for prior_scale in [0.7, 0.75, 0.8, 0.85]:
                    if length_norm_exponent > 0.0:
                        length_norm = True
                    else:
                        length_norm = False
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                            # "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=test_medline_lm_opts,
                        use_sclite=True,
                        length_norm=length_norm,
                        length_norm_exponent=length_norm_exponent,
                    )
                    if beam_size == 64 and length_norm_exponent == 0.5 and lm_scale == 0.85 and prior_scale == 0.8:
                        run_lm_fusion_custom(
                            lm_type="lstm",
                            exp_name=name,
                            epoch="best",
                            test_sets={
                                "wmt21_medline_v2_sequiturgp2_glowtts460_noise055": medline_wmt_test_noise055_tuples[21],
                                "wmt23_medline_v2_sequiturgp2_glowtts460_noise055": medline_wmt_test_noise055_tuples[23],
                                "wmt24_medline_v2_sequiturgp2_glowtts460_noise055": medline_wmt_test_noise055_tuples[24],
                            },
                            lm_scales=[lm_scale],
                            prior_scales=[prior_scale],
                            prior_type="mini_lstm",
                            mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                            train_job=train_j,
                            train_data=train_data,
                            feature_net=log10_net_10ms,
                            args=oclr_args,
                            beam_size=beam_size,
                            batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                            bpe_size=BPE_10K,
                            ext_lm_opts=test_medline_lm_opts,
                            use_sclite=True,
                            length_norm=length_norm,
                            length_norm_exponent=length_norm_exponent,
                        )


    # Zero ILM check
    test_medline_lm_opts = get_lm_model_as_opts("medline_2x1k")
    for beam_size in [32]:  # , 40, 45, 50]:
        # exponents = [0.0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.9]
        exponents = [0.5, 1.0]
        for length_norm_exponent in exponents:
            for lm_scale in [0.6, 0.7, 0.8, 0.9]:
                for prior_scale in [0.5, 0.6, 0.7, 0.8]:
                    if length_norm_exponent > 0.0:
                        length_norm = True
                    else:
                        length_norm = False
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                            # "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="zero",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=test_medline_lm_opts,
                        use_sclite=True,
                        length_norm=length_norm,
                        length_norm_exponent=length_norm_exponent,
                    )


    # No ILM check
    test_medline_lm_opts = get_lm_model_as_opts("medline_2x1k")
    for beam_size in [32]:  # , 40, 45, 50]:
        for lm_scale in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
            for prior_scale in [0.0]:
                run_lm_fusion_custom(
                    lm_type="lstm",
                    exp_name=name,
                    epoch="best",
                    test_sets={
                        "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                        # "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                    },
                    lm_scales=[lm_scale],
                    prior_scales=[prior_scale],
                    prior_type="mini_lstm",
                    mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                    train_job=train_j,
                    train_data=train_data,
                    feature_net=log10_net_10ms,
                    args=oclr_args,
                    beam_size=beam_size,
                    batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                    bpe_size=BPE_10K,
                    ext_lm_opts=test_medline_lm_opts,
                    use_sclite=True,
                    length_norm=True,
                    length_norm_exponent=1.0,
                )
                    
                    
    test_medline_lm_opts = get_lm_model_as_opts("medline_4x2k")
    for beam_size in [64]:  # , 40, 45, 50]:
        exponents = [0.5]
        for length_norm_exponent in exponents:
            for lm_scale in [0.80, 0.85, 0.9]:
                for prior_scale in [0.7, 0.75, 0.8, 0.85]:
                    if length_norm_exponent > 0.0:
                        length_norm = True
                    else:
                        length_norm = False
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "wmt22_medline_v1_sequiturgp2_glowtts460_noise055": medline_wmt22_noise055_tup,
                            # "wmt22_medline_v1_sequiturgp2_glowtts460_noise07": medline_wmt22_noise07_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=test_medline_lm_opts,
                        use_sclite=True,
                        length_norm=length_norm,
                        length_norm_exponent=length_norm_exponent,
                    )

    # ------------------------------------------
    # MTG STUFF
    # ------------------------------------------
    
    
    # MTGv4 but with LS LM
    for beam_size in [32]: # , 40, 45, 50]:
        for lm_scale in [0.2, 0.3, 0.4, 0.5]:
            for prior_scale in [0.1, 0.2, 0.3, 0.4]:
                for exponent in [0.0, 0.5, 1.0]:
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "MTG_trial4_dev_sequiturg2p_glowtts460_noise055": MTGv4_dev_noise055_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=ls_2x1k_lm_opts,
                        use_sclite=True,
                        length_norm=exponent > 0.0,
                        length_norm_exponent=exponent,
                    )


    MTG_2x1k_lm_opts = get_lm_model_as_opts("MTGv4_2x1k")
    # LM length normalization
    test_medline_lm_opts = get_lm_model_as_opts("medline_2x1k")
    for beam_size in [64]:
        exponents = [0.0, 0.5, 1.0]
        for length_norm_exponent in exponents:
            for lm_scale in [0.4, 0.5, 0.6, 0.7, 0.8]:
                for prior_scale in [0.5, 0.6, 0.7, 0.8, 0.9]:
                    if length_norm_exponent > 0.0:
                        length_norm = True
                    else:
                        length_norm = False
                    run_lm_fusion_custom(
                        lm_type="lstm",
                        exp_name=name,
                        epoch="best",
                        test_sets={
                            "MTG_trial4_dev_sequiturg2p_glowtts460_noise055": MTGv4_dev_noise055_tup,
                        },
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type="mini_lstm",
                        mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                        train_job=train_j,
                        train_data=train_data,
                        feature_net=log10_net_10ms,
                        args=oclr_args,
                        beam_size=beam_size,
                        batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        bpe_size=BPE_10K,
                        ext_lm_opts=MTG_2x1k_lm_opts,
                        use_sclite=True,
                        length_norm=length_norm,
                        length_norm_exponent=length_norm_exponent,
                    )
                    if lm_scale == 0.7 and prior_scale == 0.7 and length_norm_exponent == 0.5:
                        run_lm_fusion_custom(
                            lm_type="lstm",
                            exp_name=name,
                            epoch="best",
                            test_sets={
                                "MTG_trial4_test_sequiturg2p_glowtts460_noise055": MTGv4_test_noise055_tup,
                            },
                            lm_scales=[lm_scale],
                            prior_scales=[prior_scale],
                            prior_type="mini_lstm",
                            mini_lstm_ckpt=mini_lstm.out_checkpoints[29],
                            train_job=train_j,
                            train_data=train_data,
                            feature_net=log10_net_10ms,
                            args=oclr_args,
                            beam_size=beam_size,
                            batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                            bpe_size=BPE_10K,
                            ext_lm_opts=MTG_2x1k_lm_opts,
                            use_sclite=True,
                            length_norm=length_norm,
                            length_norm_exponent=length_norm_exponent,
                        )

