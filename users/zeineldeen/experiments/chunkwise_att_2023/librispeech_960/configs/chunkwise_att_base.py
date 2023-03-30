"""
based on:
users/zeineldeen/experiments/conformer_att_2022/librispeech_960/configs/baseline_960h_v2.py
"""

from __future__ import annotations
from typing import Optional, Union, List
import copy
import os

import numpy

from sisyphus import tk

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.chunkwise_attention_asr_config import (
    create_config,
    ConformerEncoderArgs,
    RNNDecoderArgs,
)

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
)
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
    build_chunkwise_training_datasets,
)
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.default_tools import (
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

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023 import (
    tools_eval_funcs,
    tools_eval_funcs_old,
)

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.forward import ReturnnForwardJob

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


abs_name = os.path.abspath(__file__)
prefix_name = os.path.basename(abs_name)[: -len(".py")]


def get_test_dataset_tuples(bpe_size, selected_datasets=None):
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        if selected_datasets and testset not in selected_datasets:
            continue
        test_dataset_tuples[testset] = build_test_dataset(
            testset,
            use_raw_features=True,
            bpe_size=bpe_size,
        )
    return test_dataset_tuples


def run_train(
    prefix_name: str,
    exp_name: str,
    train_args,
    train_data,
    feature_extraction_net,
    num_epochs,
    recog_epochs,
    time_rqmt=168,
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
        time_rqmt=time_rqmt,
    )
    return train_job


def run_single_search(
    prefix_name: str,
    exp_name: str,
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
        returnn_root=RETURNN_ROOT,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
    )


def run_lm_fusion(
    lm_type,
    prefix_name: str,
    exp_name: str,
    epoch: Union[str, int],
    test_set_names: Union[str, List[str]],
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

            run_single_search(
                prefix_name=prefix_name,
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
    prefix_name: str,
    exp_name: str,
    train_args,
    train_data,
    train_job,
    feature_extraction_net,
    num_epochs,
    search_args,
    recog_epochs,
    bpe_size,
    run_all_for_best_last_avg=False,
    recog_ext_pipeline=False,
    **kwargs,
):
    exp_prefix = os.path.join(prefix_name, exp_name)

    search_args = search_args if search_args is not None else copy.deepcopy(train_args)
    search_args["search_type"] = None

    returnn_search_config = create_config(
        training_datasets=train_data,
        **search_args,
        feature_extraction_net=feature_extraction_net,
        is_recog=True,
        recog_ext_pipeline=recog_ext_pipeline,
    )

    num_avg = kwargs.get("num_avg", 4)
    averaged_checkpoint = get_average_checkpoint(
        train_job,
        returnn_exe=RETURNN_CPU_EXE,
        returnn_root=RETURNN_ROOT,
        num_average=num_avg,
        key=kwargs.get("key", "dev_score_output/output_prob"),
    )
    if num_avg == 4:  # TODO: just for now to not break hashes
        train_job_avg_ckpt[exp_name] = averaged_checkpoint

    best_checkpoint = get_best_checkpoint(train_job, key=kwargs.get("key", "dev_score_output/output_prob"))
    train_job_best_epoch[exp_name] = best_checkpoint

    if recog_epochs is None:
        default_recog_epochs = [40] + [80 * i for i in range(1, int(num_epochs / 80) + 1)]
        if num_epochs % 80 != 0:
            default_recog_epochs += [num_epochs]
    else:
        default_recog_epochs = recog_epochs

    test_dataset_tuples = get_test_dataset_tuples(
        bpe_size=bpe_size, selected_datasets=kwargs.get("selected_datasets", None)
    )

    all_test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

    remove_label = {"<s>", "</s>", "<blank>"} if recog_ext_pipeline else None

    for ep in default_recog_epochs:
        search(
            exp_prefix + f"/recogs/ep-{ep}",
            returnn_search_config,
            train_job.out_checkpoints[ep],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            use_sclite=kwargs.get("use_sclite", False),
            recog_ext_pipeline=recog_ext_pipeline,
            remove_label=remove_label,
        )

    search(
        exp_prefix + "/default_last",
        returnn_search_config,
        train_job.out_checkpoints[num_epochs],
        all_test_dataset_tuples if run_all_for_best_last_avg else test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
        use_sclite=kwargs.get("use_sclite", False),
        recog_ext_pipeline=recog_ext_pipeline,
        remove_label=remove_label,
    )

    search(
        exp_prefix + "/default_best",
        returnn_search_config,
        best_checkpoint,
        all_test_dataset_tuples if run_all_for_best_last_avg else test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
        use_sclite=kwargs.get("use_sclite", False),
        recog_ext_pipeline=recog_ext_pipeline,
        remove_label=remove_label,
    )

    search(
        exp_prefix + f"/average_{num_avg}",
        returnn_search_config,
        averaged_checkpoint,
        all_test_dataset_tuples if run_all_for_best_last_avg else test_dataset_tuples,
        RETURNN_CPU_EXE,
        RETURNN_ROOT,
        use_sclite=kwargs.get("use_sclite", False),
        recog_ext_pipeline=recog_ext_pipeline,
        remove_label=remove_label,
    )


def run_exp(
    prefix_name: str,
    exp_name: str,
    train_args,
    feature_extraction_net=log10_net_10ms,
    num_epochs=300,
    search_args=None,
    recog_epochs=None,
    bpe_size=10000,
    partition_epoch=20,
    time_rqmt=168,
    train_fixed_alignment=None,
    cv_fixed_alignment=None,
    recog_ext_pipeline=False,
    **kwargs,
):
    if train_fixed_alignment:
        assert cv_fixed_alignment, "cv alignment is not set."
        train_data = build_chunkwise_training_datasets(
            train_fixed_alignment=train_fixed_alignment,
            cv_fixed_alignment=cv_fixed_alignment,
            bpe_size=bpe_size,
            use_raw_features=True,
            partition_epoch=partition_epoch,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
    else:
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            partition_epoch=partition_epoch,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

    train_job = run_train(
        prefix_name,
        exp_name,
        train_args,
        train_data,
        feature_extraction_net,
        num_epochs,
        recog_epochs,
        time_rqmt=time_rqmt,
        **kwargs,
    )
    train_jobs_map[exp_name] = train_job

    run_search(
        prefix_name,
        exp_name,
        train_args,
        train_data,
        train_job,
        feature_extraction_net,
        num_epochs,
        search_args,
        recog_epochs,
        bpe_size=bpe_size,
        recog_ext_pipeline=recog_ext_pipeline,
        **kwargs,
    )
    return train_job, train_data


def run_forward(
    prefix_name: str,
    exp_name: str,
    train_args,
    model_ckpt,
    hdf_layers=None,
    feature_extraction_net=log10_net_10ms,
    bpe_size=10000,
    time_rqmt=12,
    mem_rqmt=15,
    override_returnn_config=None,
    seq_postfix=0,
    **kwargs,
):
    # build train, dev, and devtrain
    # - No speed pert
    # - Partition epoch 1
    # - No curr. learning

    train_data = build_training_datasets(
        bpe_size=bpe_size,
        use_raw_features=True,
        partition_epoch=1,
        epoch_wise_filter=None,
        link_speed_perturbation=False,
        seq_postfix=seq_postfix,
        seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
    )

    if train_args.get("dump_alignments_dataset", None):
        dump_dataset = train_args["dump_alignments_dataset"]
    elif train_args.get("dump_ctc_dataset", None):
        dump_dataset = train_args["dump_ctc_dataset"]
    else:
        raise Exception("No dump dataset specified.")

    assert dump_dataset in ["train", "dev"]

    exp_prefix = os.path.join(prefix_name, exp_name)

    if override_returnn_config:
        returnn_config = copy.deepcopy(override_returnn_config)
    else:
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
        )

    if isinstance(model_ckpt, str):
        model_ckpt_index_path = tk.Path(model_ckpt + ".index")
        model_ckpt = Checkpoint(index_path=model_ckpt_index_path)
    elif isinstance(model_ckpt, Checkpoint):
        pass
    else:
        raise TypeError(f"model_ckpt must be str or Checkpoint, got {type(model_ckpt)}")
    forward_j = ReturnnForwardJob(
        model_checkpoint=model_ckpt,
        hdf_outputs=hdf_layers,
        returnn_config=returnn_config,
        returnn_python_exe=RETURNN_CPU_EXE,
        returnn_root=RETURNN_ROOT,
        time_rqmt=time_rqmt,
        mem_rqmt=mem_rqmt,
        eval_mode=kwargs.get("do_eval", True),
        device=kwargs.get("device", "gpu"),
    )

    forward_j.add_alias(exp_prefix + "/forward_hdf/" + dump_dataset)

    if hdf_layers is None:
        hdf_layers = ["output.hdf"]

    for layer in hdf_layers:
        tk.register_output(
            os.path.join(exp_prefix, "hdfs", dump_dataset),
            forward_j.out_hdf_files[layer],
        )

    return forward_j.out_hdf_files


def train_mini_lstm(
    prefix_name: str,
    exp_name: str,
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
    prefix_name: str,
    exp_name: str,
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
    use_sqrd_relu=True,
)
apply_fairseq_init_to_conformer(conformer_enc_args)
conformer_enc_args.ctc_loss_scale = 1.0

rnn_dec_args = RNNDecoderArgs()

training_args = dict()
training_args["speed_pert"] = True
training_args["with_pretrain"] = False

lstm_training_args = copy.deepcopy(training_args)
lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

lstm_dec_exp_args = copy.deepcopy(
    {
        **lstm_training_args,
        "encoder_args": conformer_enc_args,
        "decoder_args": rnn_dec_args,
    }
)

# --------------------------- Experiments --------------------------- #

# Global attention baseline:
#
# dev-clean  2.28
# dev-other  5.63
# test-clean  2.48
# test-other  5.71

global_att_best_ckpt = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/models-backup/best_att_100/avg_ckpt/epoch.2029"

# from Albert:
# with task=“train” and search_type=“end-of-chunk”, it would align on-the-fly
# with task=“eval”, add a hdf-dump-layer, and search_type=“end-of-chunk”, you can dump it
# with task=“train” and search_type default (None), it would train using a fixed alignment

default_args = copy.deepcopy(lstm_dec_exp_args)
default_args["learning_rates_list"] = list(numpy.linspace(8e-4, 1e-5, 60))
default_args["retrain_checkpoint"] = global_att_best_ckpt
default_args["chunk_size"] = 20
default_args["chunk_step"] = 20 * 3 // 4
default_args["search_type"] = "end-of-chunk"  # align on-the-fly


def get_ctc_chunksyn_align_config(
    dataset_name,
    ctc_alignments,
    chunk_step,
    eoc_idx=0,
    hash_full_python_code=False,
    ignore_eoc_in_input=False,  # workaround for broken CTC/RNA alignments which include EOS (=EOC)
):
    from i6_experiments.common.setups.returnn import serialization

    config = ReturnnConfig(
        {
            "extern_data": {
                "bpe_labels": {
                    "available_for_inference": False,
                    "dim": 10026,  # from CTC so +1 for blank
                    "shape": (None,),
                    "sparse": True,
                },
            },
            "eval_datasets": {
                dataset_name: {
                    "class": "MetaDataset",
                    "data_map": {"bpe_labels": ("hdf_dataset", "data")},
                    "datasets": {
                        "hdf_dataset": {
                            "class": "HDFDataset",
                            "files": [ctc_alignments],
                        },
                    },
                    "seq_order_control_dataset": "hdf_dataset",
                },
            },
            "network": {
                "chunked_align": {
                    "class": "eval",
                    "eval": tools_eval_funcs.get_chunked_align,
                    "out_type": tools_eval_funcs.get_chunked_align_out_type,
                    "from": "data:bpe_labels",
                    "eval_locals": {"chunk_step": chunk_step, "eoc_idx": eoc_idx},
                },
                "output": {
                    "class": "hdf_dump",
                    "from": "chunked_align",
                    "filename": f"alignments-{dataset_name}.hdf",
                },
            },
            "batch_size": 5000,
        }
    )
    if ignore_eoc_in_input:
        config.config["network"]["chunked_align"]["eval_locals"].setdefault("ignore_indices", []).append(eoc_idx)
    return serialization.get_serializable_config(config, hash_full_python_code=hash_full_python_code)


def get_ctc_rna_based_chunk_alignments(
    *,
    base_model_train_args: Optional[dict] = None,
    ctc_dump_exp_name: Optional[str] = None,
    fixed_ctc_rna_align_without_eos: bool = True,
    ignore_eoc_in_input: bool = False,
    chunk_sizes: Optional[List[int]] = None,
    chunk_step_factors: Optional[List[Union[int, float]]] = None,
    model_ckpt: Optional[Union[str, Checkpoint]] = None,
):
    """
    Get CTC/RNA based chunk alignments for train/dev datasets.
    """
    # save time-sync -> chunk-sync converted alignments.
    ctc_align_wo_speed_pert = {
        "train": {},
        "dev": {},
    }

    if model_ckpt is None:
        model_ckpt = global_att_best_ckpt

    if fixed_ctc_rna_align_without_eos:
        assert not ignore_eoc_in_input  # should not be needed then

    if not ctc_dump_exp_name:
        ctc_dump_exp_name = "dump_ctc_alignment_wo_speedPert"
        if fixed_ctc_rna_align_without_eos:
            ctc_dump_exp_name += "_wo_eos"
        have_custom_exp_name = False
    else:
        have_custom_exp_name = True

    for dataset in ["train", "dev"]:
        args = copy.deepcopy(base_model_train_args or default_args)
        args["dump_ctc_dataset"] = dataset
        args["batch_size"] *= 2

        # CTC alignment with blank.
        j = run_forward(
            prefix_name=prefix_name,
            exp_name=ctc_dump_exp_name,
            train_args=args,
            model_ckpt=model_ckpt,
            hdf_layers=[f"alignments-{dataset}.hdf"],
            seq_postfix=None if fixed_ctc_rna_align_without_eos else 0,
        )

        # convert w.r.t different chunk sizes and chunk steps
        if not chunk_sizes:
            chunk_sizes = [1, 2, 5, 8] + list(range(10, 55, 5)) + [60, 70, 80, 100]
        for chunk_size in chunk_sizes:
            if not chunk_step_factors:
                chunk_step_factors = [1 / 2, 3 / 4, 0.9, 1]  # 1 = no overlap
            for chunk_step_factor in chunk_step_factors:
                chunk_step = max(1, int(chunk_size * chunk_step_factor))

                if have_custom_exp_name:
                    ctc_chunk_sync_align_exp_name = f"{ctc_dump_exp_name}_chunk{chunk_size}-{chunk_step}"
                else:
                    ctc_chunk_sync_align_exp_name = f"ctc_chunk_sync_align_wo_speedPert_{chunk_size}-{chunk_step}"
                    if fixed_ctc_rna_align_without_eos:
                        ctc_chunk_sync_align_exp_name += "_wo_eos"

                ctc_chunk_sync_align = run_forward(
                    prefix_name=prefix_name,
                    exp_name=ctc_chunk_sync_align_exp_name,
                    train_args=args,
                    model_ckpt=model_ckpt,
                    hdf_layers=[f"alignments-{dataset}.hdf"],
                    override_returnn_config=get_ctc_chunksyn_align_config(
                        dataset,
                        ctc_alignments=j[f"alignments-{dataset}.hdf"],
                        chunk_step=chunk_step,
                        ignore_eoc_in_input=ignore_eoc_in_input,
                    ),
                    device="cpu",
                )

                ctc_align_wo_speed_pert[dataset][f"{chunk_size}_{chunk_step}"] = ctc_chunk_sync_align[
                    f"alignments-{dataset}.hdf"
                ]

    return ctc_align_wo_speed_pert


def run_chunkwise_train(
    ctc_chunksync_align,
    total_epochs=None,
    chunk_sizes=None,
    chunk_step_factors=None,
    suffix="",
    enable_check_align=True,
    **kwargs,
):
    # train with ctc chunk-sync alignment

    if total_epochs is None:
        total_epochs = [40, 60, 100]
    if chunk_sizes is None:
        chunk_sizes = [1, 2, 5, 8, 10, 20, 30, 40, 50]
    if chunk_step_factors is None:
        chunk_step_factors = [1 / 2, 3 / 4, 0.9, 1]

    for total_epochs in total_epochs:
        for chunk_size in chunk_sizes:
            for chunk_step_factor in chunk_step_factors:
                for start_lr in [1e-4]:
                    for decay_pt_factor in [1 / 3]:
                        train_args = copy.deepcopy(default_args)
                        train_args["speed_pert"] = False  # no speed pert
                        train_args["search_type"] = None  # fixed alignment

                        train_args["max_seq_length"] = None  # no filtering!

                        train_args["encoder_args"].with_ctc = False  # No CTC
                        train_args["enable_check_align"] = enable_check_align  # to not break hashes

                        decay_pt = int(total_epochs * decay_pt_factor)

                        train_args["chunk_size"] = chunk_size

                        chunk_step = max(1, int(chunk_size * chunk_step_factor))
                        train_args["chunk_step"] = chunk_step

                        train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
                            numpy.linspace(start_lr, 1e-6, total_epochs - decay_pt)
                        )

                        exp_name = f"base_chunkwise_att_chunk-{chunk_size}_step-{chunk_step}_linDecay{total_epochs}_{start_lr}_decayPt{decay_pt_factor}_fixed_align"
                        if suffix:
                            exp_name += suffix

                        if ctc_chunksync_align:
                            run_exp(
                                prefix_name=prefix_name,
                                exp_name=exp_name,
                                train_args=train_args,
                                num_epochs=total_epochs,
                                train_fixed_alignment=ctc_chunksync_align["train"][f"{chunk_size}_{chunk_step}"],
                                cv_fixed_alignment=ctc_chunksync_align["dev"][f"{chunk_size}_{chunk_step}"],
                                epoch_wise_filter=None,
                                time_rqmt=72,
                                selected_datasets=["dev-other"],
                                key="dev_score",
                                use_sclite=True,
                                **kwargs,
                            )
                        else:
                            train_args["search_type"] = "end-of-chunk"  # on-the-fly alignment
                            run_exp(
                                prefix_name=prefix_name,
                                exp_name=exp_name,
                                train_args=train_args,
                                num_epochs=total_epochs,
                                epoch_wise_filter=None,
                                time_rqmt=72,
                                selected_datasets=["dev-other"],
                                key="dev_score",
                                use_sclite=True,
                                **kwargs,
                            )


def _run_exp_full_sum_simple_approx(
    *,
    enc_stream_type: Optional[str],
    chunk_size: int,
    chunk_step_factor: float,
    total_epochs: int,
    with_ctc: bool = False,
):
    start_lr = 1e-4
    decay_pt_factor = 1 / 3
    train_args = copy.deepcopy(default_args)

    train_args["speed_pert"] = False  # no speed pert
    train_args["search_type"] = None
    train_args["max_seq_length"] = None  # no filtering!

    train_args["encoder_args"].with_ctc = with_ctc

    if enc_stream_type == "causal" or enc_stream_type.startswith("causal-"):
        train_args["encoder_args"].use_causal_layers = True
        if enc_stream_type == "causal-reset-conv":
            train_args["encoder_args"].conv_alternative_name = "depthwise_conv2_causal"
            train_args.setdefault("retrain_checkpoint_opts", {}).setdefault("ignore_params_prefixes", []).extend(
                [
                    "conformer_block_%02i_conv_mod_depthwise_conv2_causal/" % (i + 1)
                    for i in range(train_args["encoder_args"].num_blocks)
                ]
            )

    decay_pt = int(total_epochs * decay_pt_factor)

    train_args["chunk_size"] = chunk_size

    chunk_step = max(1, int(chunk_size * chunk_step_factor))
    train_args["chunk_step"] = chunk_step

    chunk_level = "input" if enc_stream_type == "chunked" else "encoder"
    train_args["chunk_level"] = chunk_level
    train_args["eoc_idx"] = 0
    train_args["decoder_args"].prev_target_embed_direct = True
    train_args["decoder_args"].full_sum_simple_approx = True

    if chunk_level == "input":
        # It needs more memory because there are mini batches
        # where the chunk size is larger than the sequences,
        # thus increasing the overall memory consumption of the whole encoder.
        train_args["batch_size"] = int(train_args["batch_size"] * 0.75)
        train_args["accum_grad"] = int(train_args.get("accum_grad", 2) * 1.5)

    train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
        numpy.linspace(start_lr, 1e-6, total_epochs - decay_pt)
    )

    train_args["enable_check_align"] = False

    train_args["batch_size"] = int(0.75 * train_args["batch_size"])
    train_args["accum_grad"] = int(1.5 * train_args.get("accum_grad", 2))

    exp_name_parts = [
        "chunk_att_simpleFS",
        f"enc-{enc_stream_type}-conf",
        f"chunksize-{chunk_size}",
        f"chunkstep-{chunk_step}",
        f"linDecay{total_epochs}_{start_lr}_decayPt{decay_pt_factor}",
        f"ctc{with_ctc}",
    ]

    run_exp(
        prefix_name=prefix_name,
        exp_name="_".join(exp_name_parts),
        train_args=train_args,
        num_epochs=total_epochs,
        epoch_wise_filter=None,
        time_rqmt=72,
        selected_datasets=["dev-other"],
        key="dev_score_output/output_prob" if with_ctc else "dev_score",
        use_sclite=True,
    )


def baseline():
    ctc_align_wo_speed_pert = get_ctc_rna_based_chunk_alignments(
        fixed_ctc_rna_align_without_eos=False,
        chunk_sizes=[1, 2, 5, 8, 10, 20, 30, 40, 50],
        chunk_step_factors=[1, 1 / 2, 3 / 4],  # do not run for 0.9 for that but with fixed alignment
    )

    # imported from `/work/asr4/zeyer/setups-data/combined/2021-05-31/work`
    fixed_ctc_align_wo_speed_pert = get_ctc_rna_based_chunk_alignments(fixed_ctc_rna_align_without_eos=True)

    run_chunkwise_train(ctc_align_wo_speed_pert, chunk_step_factors=[1, 1 / 2, 3 / 4], enable_check_align=False)

    # TODO: run exps with fixed align
    run_chunkwise_train(
        fixed_ctc_align_wo_speed_pert, suffix="_wo_eos", run_all_for_best_last_avg=True, enable_check_align=False
    )

    # TODO: longer training for chunk sizes 1 and 2
    run_chunkwise_train(
        fixed_ctc_align_wo_speed_pert,
        chunk_step_factors=[1, 1 / 2],
        chunk_sizes=[1, 2],
        total_epochs=[150, 200, 250],
        run_all_for_best_last_avg=True,
        enable_check_align=False,
    )

    # TODO: alignment on-the-fly
    run_chunkwise_train(
        ctc_chunksync_align=None,
        suffix="_on_the_fly",
        run_all_for_best_last_avg=True,
        chunk_step_factors=[1, 0.75],
        total_epochs=[100],
        enable_check_align=False,
    )

    # TODO: approximated full-sum
    for chunk_size in [10, 20, 30]:
        _run_exp_full_sum_simple_approx(
            enc_stream_type="global", chunk_size=chunk_size, chunk_step_factor=1.0, total_epochs=40
        )
