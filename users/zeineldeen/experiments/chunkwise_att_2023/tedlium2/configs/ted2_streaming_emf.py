"""
based on:
users/zeineldeen/experiments/conformer_att_2022/librispeech_960/configs/baseline_960h_v2.py
"""

from __future__ import annotations
from typing import Optional, Union, List
import copy
import os

import numpy, math

from sisyphus import tk

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.chunkwise_attention_asr_config import (
    create_config,
    ConformerEncoderArgs,
    RNNDecoderArgs,
)

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
)
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.tedlium2.data import (
    build_training_datasets,
    build_test_dataset,
    build_chunkwise_training_datasets,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.default_tools import (
    RETURNN_CPU_EXE,
    SCTK_BINARY_PATH,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.default_tools import (
    RETURNN_ROOT_V2 as RETURNN_ROOT,
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

# train:
# Seq-length 'audio_features' Stats:
#   92973 seqs
#   Mean: 131396.26378626248  (8.2 sec)
#   Std dev: 70640.3150616384
#   Min/max: 4480 / 360977  (0.28 sec / 22.56 sec)
# Seq-length 'bpe_labels' Stats:
#   92973 seqs
#   Mean: 39.46015509879227
#   Std dev: 23.121978685238307
#   Min/max: 2 / 153
#
# dev:
# Seq-length 'audio_features' Stats:
#   507 seqs
#   Mean: 181567.81065088772 (11 sec)
#   Std dev: 90829.89353459886
#   Min/max: 7520 / 638720  (0.47 sec / 39.92 sec)
# Seq-length 'bpe_labels' Stats:
#   507 seqs
#   Mean: 58.13412228796851
#   Std dev: 33.32746140640928
#   Min/max: 2 / 211
#
# test:
# Seq-length 'audio_features' Stats:
#   1155 seqs
#   Mean: 130516.24675324683 (8 sec)
#   Std dev: 69077.22250260337  (4 sec)
#   Min/max: 5600 / 520768 (0.35 sec / 32.55 sec)
# Seq-length 'bpe_labels' Stats:
#   1155 seqs
#   Mean: 38.384415584415635
#   Std dev: 22.381681584409716
#   Min/max: 2 / 179


# ----------------------------------------------------------- #


abs_name = os.path.abspath(__file__)
prefix_name = os.path.basename(abs_name)[: -len(".py")]


def get_test_dataset_tuples(bpe_size, selected_datasets=None):
    test_dataset_tuples = {}
    for testset in ["dev", "test"]:
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
    time_rqmt: float = 168,
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
        gpu_mem=kwargs.get("gpu_mem", 11),
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
    recog_bliss_corpus,
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
        recognition_bliss_corpus=recog_bliss_corpus,
        returnn_exe=RETURNN_CPU_EXE,
        returnn_root=RETURNN_ROOT,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
    )


def run_concat_seq_recog(exp_name, corpus_names, num, train_data, search_args, checkpoint, mem_rqmt=8, time_rqmt=1):
    exp_prefix = os.path.join(prefix_name, exp_name)

    from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.concat_seqs import (
        ConcatDatasetSeqsJob,
        ConcatSeqsDataset,
        CreateConcatSeqsCTMAndSTMJob,
    )
    from i6_core.corpus.convert import CorpusToStmJob

    if isinstance(corpus_names, str):
        corpus_names = [corpus_names]
    assert isinstance(corpus_names, list)

    for corpus_name in corpus_names:
        test_datasets = get_test_dataset_tuples(bpe_size=BPE_1K)
        stm = CorpusToStmJob(bliss_corpus=test_datasets[corpus_name][2]).out_stm_path
        tk.register_output(f"concat_seqs/{num}/orig_{corpus_name}_stm", stm)
        concat_dataset_seqs = ConcatDatasetSeqsJob(corpus_name="TED-LIUM-realease2", stm=stm, num=num, overlap_dur=None)
        tk.register_output(f"concat_seqs/{num}/{corpus_name}_stm", concat_dataset_seqs.out_stm)
        tk.register_output(f"concat_seqs/{num}/{corpus_name}_tags", concat_dataset_seqs.out_concat_seq_tags)
        tk.register_output(f"concat_seqs/{num}/{corpus_name}_lens", concat_dataset_seqs.out_concat_seq_lens_py)

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=log10_net_10ms,
            is_recog=True,
        )

        returnn_concat_dataset = ConcatSeqsDataset(
            dataset=test_datasets[corpus_name][0].as_returnn_opts(),
            seq_tags=concat_dataset_seqs.out_concat_seq_tags,
            seq_lens_py=concat_dataset_seqs.out_orig_seq_lens_py,
        )

        _, search_words = search_single(
            os.path.join(exp_prefix, corpus_name),
            returnn_search_config,
            checkpoint,
            recognition_dataset=returnn_concat_dataset,
            recognition_reference=test_datasets[corpus_name][1],
            recognition_bliss_corpus=test_datasets[corpus_name][2],
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
            # no scoring
            use_sclite=False,
            use_returnn_compute_wer=False,
        )

        from i6_core.corpus.convert import CorpusToStmJob
        from i6_core.recognition.scoring import ScliteJob

        stm_file = concat_dataset_seqs.out_stm

        concat_ctm_and_stm_job = CreateConcatSeqsCTMAndSTMJob(
            recog_words_file=search_words, stm_py_file=concat_dataset_seqs.out_stm_py, stm_file=stm_file
        )
        tk.register_output(exp_prefix + f"/{corpus_name}/sclite/stm", concat_ctm_and_stm_job.out_stm_file)
        tk.register_output(exp_prefix + f"/{corpus_name}/sclite/ctm", concat_ctm_and_stm_job.out_ctm_file)

        sclite_job = ScliteJob(
            ref=concat_ctm_and_stm_job.out_stm_file,
            hyp=concat_ctm_and_stm_job.out_ctm_file,
            sctk_binary_path=SCTK_BINARY_PATH,
        )
        tk.register_output(exp_prefix + f"/{corpus_name}/sclite/wer", sclite_job.out_wer)
        tk.register_output(exp_prefix + f"/{corpus_name}/sclite/report", sclite_job.out_report_dir)


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
                recog_bliss_corpus=test_dataset_tuples[test_set][2],
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
        enable_mail=True,
    )


def run_exp(
    prefix_name: str,
    exp_name: str,
    train_args,
    feature_extraction_net=log10_net_10ms,
    num_epochs=200,
    search_args=None,
    recog_epochs=None,
    bpe_size=1000,
    partition_epoch=4,
    time_rqmt: float = 168,
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
            epoch_wise_filter=None,
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
    else:
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            partition_epoch=partition_epoch,
            epoch_wise_filter=None,
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
            seq_postfix=kwargs.get("seq_postfix", 0),
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

    if kwargs.get("concat_recog_opts", None):
        ckpt_ = kwargs["concat_recog_opts"]["checkpoint"]
        if isinstance(ckpt_, str):
            assert ckpt_ in ["best", "avg"]
            if ckpt_ == "best":
                concat_recog_ckpt = train_job_best_epoch[exp_name]
            else:
                concat_recog_ckpt = train_job_avg_ckpt[exp_name]
        elif isinstance(ckpt_, int):
            concat_recog_ckpt = train_job.out_checkpoints[ckpt_]
        else:
            raise TypeError(f"concat_recog_opts['checkpoint'] must be str or int, got {type(ckpt_)}")
        concat_recog_search_args = kwargs["concat_recog_opts"].get("search_args", None)
        search_args_ = copy.deepcopy(train_args)
        if concat_recog_search_args:
            search_args_.update(concat_recog_search_args)
        run_concat_seq_recog(
            exp_name=exp_name + f"_concat{kwargs['concat_recog_opts']['num']}",
            corpus_names=kwargs["concat_recog_opts"]["corpus_names"],
            num=kwargs["concat_recog_opts"]["num"],
            train_data=train_data,
            search_args=search_args_,
            checkpoint=concat_recog_ckpt,
        )

    return train_job, train_data


def run_forward(
    prefix_name: str,
    exp_name: str,
    train_args,
    model_ckpt,
    hdf_layers=None,
    feature_extraction_net=log10_net_10ms,
    bpe_size=1000,
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

    if isinstance(model_ckpt, tk.Path):
        path_as_str = model_ckpt.path
        assert isinstance(path_as_str, str)
        new_hash_overwrite = (model_ckpt.hash_overwrite[0], model_ckpt.hash_overwrite[1] + "_index")
        model_ckpt = Checkpoint(index_path=tk.Path(path_as_str + ".index", hash_overwrite=new_hash_overwrite))
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
        bpe_size=1000,
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
        bpe_size=1000,
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
    frontend_conv_l2=0.0001,
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

from .ted2_streaming import run_chunkwise_train


def baseline():
    # TODO: emformer memory
    for left_context, center_context, right_context, conv_cache_size, mem_size in [
        (0, 20, 5, 1, 2),
        (0, 10, 5, 4, 4),
    ]:
        run_chunkwise_train(
            enc_stream_type="chunked",
            run_all_for_best_last_avg=True,
            enable_check_align=False,
            chunk_sizes=[left_context + center_context + right_context],
            chunk_step_factors=[center_context / (left_context + center_context + right_context)],
            start_lrs=[2e-4],
            decay_pt_factors=[1 / 3],
            gpu_mem=24,
            total_epochs=[120],
            batch_size=15_000,
            accum_grad=2,
            time_rqmt=120,
            end_slice_start=left_context,
            end_slice_size=center_context,
            window_left_padding=left_context * 6,
            conf_mem_opts={
                "self_att_version": 1,
                "mem_size": mem_size,
                "use_cached_prev_kv": True,
                "conv_cache_size": conv_cache_size,
                "mem_slice_start": left_context,
                "mem_slice_size": center_context,
                "use_emformer_mem": True,
            },
            suffix=f"_L{left_context}_C{center_context}_R{right_context}",
        )
