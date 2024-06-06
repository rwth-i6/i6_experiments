"""
Chunked AED Sisyphus Config for TED-LIUM-v2 which contains experiments for this paper:
    Title: Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition
    Link: https://arxiv.org/abs/2309.08436
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
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.librispeech_960.default_tools import (
    # RETURNN_ROOT,
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

# Trafo LM trained by willi
# - setup dir: /u/michel/setups/language_modelling/tedlium/neurallm/trafo_kazuki19
# - config: /u/michel/setups/language_modelling/tedlium/neurallm/trafo_kazuki19/tdl1.config
trafo_lm_net = TransformerLM(
    source="prev:output",
    num_layers=30,
    vocab_size=1057,
    use_as_ext_lm=True,
    ff_dim=4096,
    att_num_heads=12,
    embed_dim=128,
    qk_dim=768,
    v_dim=768,
    out_dim=768,
    emb_cpu_lookup=False,
    embed_pos=False,  # wo abs pos encoding
    conv_act="gelu",
)
trafo_lm_net.create_network()
trafo_1k_lm_opts = {
    "lm_subnet": trafo_lm_net.network.get_net(),
    "load_on_init_opts": {
        "filename": "/work/asr4/michel/setups-data/language_modelling/tedlium/neurallm/trafo_kazuki19/net-model/network.020",
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}

trafo_lm_opts_map = {BPE_1K: trafo_1k_lm_opts}

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
        kwargs.get("returnn_root", RETURNN_ROOT),
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
        use_sclite=True,
    )


def run_concat_seq_recog(
    exp_name, corpus_names, num, train_data, search_args, checkpoint, mem_rqmt=8, time_rqmt=1, gpu_mem=11
):
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
            gpu_mem=gpu_mem,
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
        returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
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
            kwargs.get("returnn_root", RETURNN_ROOT),
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
        kwargs.get("returnn_root", RETURNN_ROOT),
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
        kwargs.get("returnn_root", RETURNN_ROOT),
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
        kwargs.get("returnn_root", RETURNN_ROOT),
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

    if kwargs.get("do_not_train", False):
        return None, train_data
    else:
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
            gpu_mem=kwargs.get("gpu_mem", 11),
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
    if kwargs.get("cpu_type", None):
        assert "sbatch_args" not in forward_j.rqmt
        forward_j.rqmt["cpu_type"] = kwargs["cpu_type"]

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


def compute_search_errors(exp_name, corpus_name, train_data, forward_args, model_checkpoint):
    returnn_root_fix = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2023-06-14--streaming-conf/returnn",
        hash_overwrite="returnn_root_fix_forward",
    )

    returnn_forward_config = create_config(
        training_datasets=train_data,
        **forward_args,
        feature_extraction_net=log10_net_10ms,
        is_recog=True,
    )
    returnn_forward_config.config["forward_batch_size"] = 15_000 * 160

    # add test dataset
    test_dataset = get_test_dataset_tuples(bpe_size=BPE_1K, selected_datasets=[corpus_name])
    returnn_forward_config.config["forward_data"] = test_dataset[corpus_name][0].as_returnn_opts()
    # remove all others if exist
    returnn_forward_config.config.pop("train", None)
    returnn_forward_config.config.pop("dev", None)
    returnn_forward_config.config.pop("devtrain", None)

    def dump_scores_and_targets(config_, targets_hdf_name, scores_hdf_name):
        config_.config["network"]["dump_targets"] = {
            "class": "hdf_dump",
            "from": "data:bpe_labels",
            "filename": f"{targets_hdf_name}.hdf",
            "is_output_layer": True,
        }
        assert "output_log_prob" not in config_.config["network"]["output"]["unit"], "output_log_prob already exists?"
        config_.config["network"]["output"]["unit"]["output_log_prob"] = {
            "class": "activation",
            "from": "output_prob",
            "activation": "safe_log",
        }
        config_.config["network"]["output"]["unit"]["dump_output"] = {
            "class": "hdf_dump",
            "from": "output_log_prob",
            "filename": f"{scores_hdf_name}.hdf",
            "is_output_layer": True,
        }
        j = ReturnnForwardJob(
            model_checkpoint=model_checkpoint,
            returnn_config=config_,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=returnn_root_fix,
            hdf_outputs=[f"{targets_hdf_name}.hdf", f"{scores_hdf_name}.hdf"],
            eval_mode=True,
        )
        return j

    j = dump_scores_and_targets(
        copy.deepcopy(returnn_forward_config),
        targets_hdf_name="ground_truth_targets",
        scores_hdf_name="ground_truth_scores",
    )
    ground_truth_targets = j.out_hdf_files["ground_truth_targets.hdf"]
    ground_truth_scores = j.out_hdf_files["ground_truth_scores.hdf"]
    tk.register_output(
        os.path.join(prefix_name, exp_name, f"{corpus_name}_search_errors/ground_truth_targets"), ground_truth_targets
    )
    tk.register_output(
        os.path.join(prefix_name, exp_name, f"{corpus_name}_search_errors/ground_truth_scores"), ground_truth_scores
    )

    # 2. forward with beam search and dump scores of 1-best
    config_ = copy.deepcopy(returnn_forward_config)
    config_.config["network"]["dump_decision"] = {
        "class": "hdf_dump",
        "from": "out_best_wo_blank",
        "filename": "search_output.hdf",
        "is_output_layer": True,
    }
    returnn_forward_config.post_config["forward_use_search"] = True  # allow search
    j = ReturnnForwardJob(
        model_checkpoint=model_checkpoint,
        returnn_config=config_,
        returnn_python_exe=RETURNN_CPU_EXE,
        returnn_root=returnn_root_fix,
        hdf_outputs=["search_output.hdf"],
        eval_mode=False,
    )
    search_output = j.out_hdf_files["search_output.hdf"]  # targets included here?
    tk.register_output(os.path.join(prefix_name, exp_name, f"{corpus_name}_search_errors/search_output"), search_output)

    j = dump_scores_and_targets(
        copy.deepcopy(returnn_forward_config), targets_hdf_name="search_targets", scores_hdf_name="search_scores"
    )
    search_targets = j.out_hdf_files["search_targets.hdf"]
    search_scores = j.out_hdf_files["search_scores.hdf"]
    tk.register_output(
        os.path.join(prefix_name, exp_name, f"{corpus_name}_search_errors/search_targets"), search_targets
    )
    tk.register_output(os.path.join(prefix_name, exp_name, f"{corpus_name}_search_errors/search_scores"), search_scores)

    from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.search_errors import ComputeSearchErrorsJob

    # search_errors = ComputeSearchErrorsJob(
    #     ground_truth_scores_hdf=ground_truth_scores,
    #     search_scores_hdf=search_scores,
    #     ground_truth_targets_hdf=ground_truth_targets,
    #     search_targets_hdf=None,  #
    #     blank_idx=0,  # EOC
    # )
    # tk.register_output(
    #     os.path.join(exp_name, f"{corpus_name}_search_errors/search_errors"), search_errors.out_search_errors
    # )


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

# base_bpe1000_peakLR0.0008_ep400_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.15_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12
# 7.4/6.9
global_att_ep400 = "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average"

# base_bpe1000_peakLR0.0008_ep200_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.1_woDepthConvPre
# 8.2/7.6
# global_att_ep200 = tk.Path(
#     "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.43W6IZbd5Arr/output/model/average",
#     hash_overwrite="global_att_ep200",
# )

# from Albert:
# with task=“train” and search_type=“end-of-chunk”, it would align on-the-fly
# with task=“eval”, add a hdf-dump-layer, and search_type=“end-of-chunk”, you can dump it
# with task=“train” and search_type default (None), it would train using a fixed alignment

default_args = copy.deepcopy(lstm_dec_exp_args)
default_args["learning_rates_list"] = list(numpy.linspace(8e-4, 1e-5, 60))
default_args["retrain_checkpoint"] = global_att_ep400
default_args["chunk_size"] = 20
default_args["chunk_step"] = 20 * 3 // 4
default_args["search_type"] = "end-of-chunk"  # align on-the-fly

default_args["decoder_args"].embed_dim = 256
default_args["encoder_args"].mhsa_weight_dropout = 0.0
default_args["encoder_args"].ff_weight_dropout = 0.1
default_args["encoder_args"].conv_weight_dropout = 0.1
default_args["encoder_args"].dropout = 0.15
default_args["encoder_args"].dropout_in = 0.15
default_args["encoder_args"].att_dropout = 0.15
default_args["decoder_args"].use_zoneout_output = True

from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.configs.ted2_att_baseline import (
    compute_features_stats,
)

_, _, global_mean, global_std = compute_features_stats(output_dirname="logmel_80", feat_dim=80)
default_args["global_stats"] = {"mean": global_mean, "stddev": global_std}


def get_base_v1_args(base_args, lr, ep, enc_drop=0.1, pretrain_reps=3):
    #  base_bpe1000_peakLR0.0008_ep200_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.1_woDepthConvPre
    # Average ckpt: 8.19/7.64 (50 epochs)
    # - Epoch-based OCLR with peak LR 8e-4
    # - EncDrop 0.1, fixed zoneout
    # - Pretrain 3, no depthwise conv pretrain
    # - Feature global normalization

    base_v1_args = copy.deepcopy(base_args)
    base_v1_args.pop("oclr_opts", None)
    cyc_ep = int(0.45 * ep)
    # Epoch-based OCLR
    base_v1_args["learning_rates_list"] = (
        list(numpy.linspace(lr / 10, lr, cyc_ep))
        + list(numpy.linspace(lr, lr / 10, cyc_ep))
        + list(numpy.linspace(lr / 10, 1e-6, ep - 2 * cyc_ep))
    )
    base_v1_args["global_stats"] = {"mean": global_mean, "stddev": global_std}
    base_v1_args["with_pretrain"] = True
    base_v1_args["pretrain_reps"] = pretrain_reps
    base_v1_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 22500 * 160,
        "ignored_keys_for_reduce_dim": ["conv_kernel_size"],
    }
    base_v1_args["encoder_args"].dropout = enc_drop
    base_v1_args["encoder_args"].dropout_in = enc_drop
    base_v1_args["encoder_args"].att_dropout = enc_drop
    base_v1_args["decoder_args"].use_zoneout_output = True
    return base_v1_args


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
                    "dim": 1058,  # from CTC so +1 for blank
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
            "batch_size": 5000,  # ctc alignment
        }
    )
    config.post_config["use_tensorflow"] = True
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
        model_ckpt = global_att_ep400

    if fixed_ctc_rna_align_without_eos:
        assert not ignore_eoc_in_input  # should not be needed then

    if not ctc_dump_exp_name:
        ctc_dump_exp_name = "dump_ctc_alignment_wo_speedPert"
        if fixed_ctc_rna_align_without_eos:
            ctc_dump_exp_name += "_wo_eos"
        have_custom_exp_name = False
    else:
        have_custom_exp_name = True

    ctc_dump_exp_name = "debug1"

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
                    time_rqmt=1.0,
                    cpu_type="cpu_short",
                )

                ctc_align_wo_speed_pert[dataset][f"{chunk_size}_{chunk_step}"] = ctc_chunk_sync_align[
                    f"alignments-{dataset}.hdf"
                ]

    return ctc_align_wo_speed_pert


def run_chunkwise_train(
    total_epochs: List[int],
    chunk_sizes: List[Optional[int]],
    chunk_step_factors: List[Optional[float]],
    enc_stream_type: str = "global",
    suffix: str = "",
    enable_check_align: bool = True,
    on_the_fly_align: bool = False,
    with_ctc: bool = False,
    ctc_self_align_delay: int = None,
    ctc_self_align_delay_scale: float = 0.5,
    batch_size: int = 15_000,
    accum_grad: int = 2,
    time_rqmt: float = 72,
    start_lrs: Union[float, List[Optional[float]]] = 1e-4,
    final_lrs: Union[float, List[Optional[float]]] = 1e-6,
    decay_pt_factors: Union[float, List[Optional[float]]] = 1 / 3,
    window_left_padding: Optional[int] = None,
    end_slice_size: Optional[int] = None,
    end_slice_start: Optional[int] = None,
    pos_enc: Optional[str] = "rel",
    conf_mem_opts: Optional[dict] = None,
    full_sum_approx: bool = False,
    retrain_ckpt: Optional[Union[tk.Path, str]] = None,
    chunked_decoder: bool = True,
    epoch_oclr_lr: Optional[float] = None,
    decoder_mask_eoc: Optional[bool] = None,
    speed_pert: bool = False,
    from_scratch_train: bool = False,
    lrs_list: Optional[List[float]] = None,
    lr_list_desc: Optional[str] = None,
    return_args: bool = False,
    return_hdf_align: bool = False,
    do_not_train=False,
    **kwargs,
):
    if isinstance(start_lrs, float):
        start_lrs = [start_lrs]
    if isinstance(final_lrs, float):
        final_lrs = [final_lrs]
    if isinstance(decay_pt_factors, float):
        decay_pt_factors = [decay_pt_factors]

    # train with ctc chunk-sync alignment
    ctc_chunksync_align = get_ctc_rna_based_chunk_alignments(
        fixed_ctc_rna_align_without_eos=True,
        chunk_sizes=chunk_sizes,
        chunk_step_factors=chunk_step_factors,
        model_ckpt=retrain_ckpt,
    )

    for total_epoch in total_epochs:
        for chunk_size in chunk_sizes:
            for chunk_step_factor in chunk_step_factors:
                for start_lr in start_lrs:
                    for final_lr in final_lrs:
                        for decay_pt_factor in decay_pt_factors:
                            train_args = copy.deepcopy(default_args)
                            train_args["speed_pert"] = speed_pert
                            train_args["search_type"] = None

                            train_args["max_seq_length"] = None  # no filtering!

                            train_args["encoder_args"].with_ctc = with_ctc
                            if ctc_self_align_delay:
                                assert with_ctc, "need CTC for self-align"
                                train_args["encoder_args"].ctc_self_align_delay = ctc_self_align_delay
                                train_args["encoder_args"].ctc_self_align_scale = ctc_self_align_delay_scale

                            if enc_stream_type == "causal" or enc_stream_type.startswith("causal-"):
                                if enc_stream_type == "causal":
                                    train_args["encoder_args"].use_causal_layers = True  # causal MHSA and conv
                                elif enc_stream_type == "causal-mhsa":
                                    train_args["encoder_args"].use_causal_layers = True
                                    train_args["encoder_args"].use_causal_conv = False  # causal MHSA only
                                elif enc_stream_type == "causal-reset-conv":
                                    train_args["encoder_args"].use_causal_layers = True
                                    train_args["encoder_args"].conv_alternative_name = "depthwise_conv2_causal"
                                    train_args.setdefault("retrain_checkpoint_opts", {}).setdefault(
                                        "ignore_params_prefixes", []
                                    ).extend(
                                        [
                                            "conformer_block_%02i_conv_mod_depthwise_conv2_causal/" % (i + 1)
                                            for i in range(train_args["encoder_args"].num_blocks)
                                        ]
                                    )

                            train_args["batch_size"] = batch_size * 160
                            train_args["accum_grad"] = accum_grad

                            train_args["enable_check_align"] = enable_check_align  # to not break hashes

                            train_args["chunk_size"] = chunk_size
                            if chunk_size is None:
                                train_args["chunk_step"] = None
                                chunk_step = None
                            else:
                                chunk_step = max(1, int(chunk_size * chunk_step_factor))
                                train_args["chunk_step"] = chunk_step

                            if lrs_list is not None:
                                train_args["learning_rates_list"] = lrs_list
                            elif epoch_oclr_lr:
                                assert start_lr is None
                                cyc_ep = int(0.45 * total_epoch)
                                train_args["learning_rates_list"] = (
                                    list(numpy.linspace(epoch_oclr_lr / 10, epoch_oclr_lr, cyc_ep))
                                    + list(numpy.linspace(epoch_oclr_lr, epoch_oclr_lr / 10, cyc_ep))
                                    + list(numpy.linspace(epoch_oclr_lr / 10, final_lr, total_epoch - 2 * cyc_ep))
                                )
                            else:
                                decay_pt = int(total_epoch * decay_pt_factor)
                                train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
                                    numpy.linspace(start_lr, final_lr, total_epoch - decay_pt)
                                )

                            chunk_level = "input" if enc_stream_type == "chunked" else "encoder"
                            train_args["chunk_level"] = chunk_level
                            train_args["eoc_idx"] = 0

                            exp_name = f"{enc_stream_type}_att_chunk"
                            if chunk_size is not None:
                                assert chunk_step is not None
                                exp_name += f"-{chunk_size}_step-{chunk_step}"
                            else:
                                exp_name += "-globalAtt"  # no chunking

                            if start_lr is not None:
                                exp_name += f"_linDecay{total_epoch}_{start_lr}_decayPt{decay_pt_factor}"
                            if final_lr is not None:
                                exp_name += f"_finalLR{final_lr}"
                            elif epoch_oclr_lr is not None:
                                exp_name += f"_epochOCLR-{epoch_oclr_lr}_ep{total_epoch}"
                            elif lrs_list:
                                assert lr_list_desc
                                exp_name += f"_{lr_list_desc}"

                            exp_name += f"_bs{batch_size}_accum{accum_grad}"

                            if window_left_padding is not None:
                                train_args["window_left_padding"] = window_left_padding
                                exp_name += f"_winLeft{window_left_padding}"

                            if end_slice_size is not None:
                                train_args["end_slice_size"] = end_slice_size
                                assert end_slice_start is not None, "need end_slice_start"
                                train_args["end_slice_start"] = end_slice_start
                                exp_name += f"_endSliceStart{end_slice_start}_endSlice{end_slice_size}"

                            if pos_enc is None:
                                train_args["encoder_args"].pos_enc = pos_enc
                                exp_name += f"_woPosEnc"
                            else:
                                assert pos_enc == "rel"

                            if conf_mem_opts is not None:
                                train_args["conf_mem_opts"] = conf_mem_opts
                                exp_name += f"_memVariant{conf_mem_opts['self_att_version']}"
                                mem_size = conf_mem_opts.get("mem_size", 1)
                                if mem_size > 1:
                                    exp_name += f"_memSize{mem_size}"
                                if conf_mem_opts.get("mask_paddings", False):
                                    exp_name += f"_memMaskPad"
                                if conf_mem_opts.get("conv_cache_size", None):
                                    exp_name += f"_convCache{conf_mem_opts['conv_cache_size']}"
                                if conf_mem_opts.get("use_cached_prev_kv", False):
                                    exp_name += f"_useCachedKV"
                                if conf_mem_opts.get("mem_slice_start", None) is not None:
                                    assert conf_mem_opts.get("mem_slice_size", None) is not None
                                    exp_name += (
                                        f"_memSlice{conf_mem_opts['mem_slice_start']}-{conf_mem_opts['mem_slice_size']}"
                                    )
                                if conf_mem_opts.get("use_emformer_mem", False):
                                    exp_name += f"_emfMem"
                                train_args["recursion_limit"] = 4000

                            if with_ctc:
                                exp_name += "_withCtc"

                            if full_sum_approx:
                                # NOTE: no need to mask EOC for the decoder since the targets do not contain EOC (just bpe labels)
                                train_args["decoder_args"].prev_target_embed_direct = True
                                train_args["decoder_args"].full_sum_simple_approx = True
                                exp_name += "_fullSumApprox"

                            if decoder_mask_eoc:
                                train_args["decoder_args"].masked_computation_blank_idx = train_args["eoc_idx"]
                                exp_name += "_maskEOC"

                            if retrain_ckpt:
                                assert suffix, "set suffix for retrain to avoid overwriting"
                                train_args["retrain_checkpoint"] = retrain_ckpt

                            train_args["chunked_decoder"] = chunked_decoder
                            if not chunked_decoder:
                                exp_name += "_noChunkedDec"

                            if from_scratch_train:
                                train_args.update(get_base_v1_args(train_args, lr=epoch_oclr_lr, ep=total_epoch))
                                train_args["with_pretrain"] = True
                                train_args["retrain_checkpoint"] = None
                                exp_name += "_fromScratch"

                            if kwargs.get("rel_pos_clipping", None):
                                train_args["encoder_args"].rel_pos_clipping = kwargs["rel_pos_clipping"]
                                exp_name += f"_relPosClip{kwargs['rel_pos_clipping']}"

                            if kwargs.get("remove_att_ctx_from_dec_state", False):
                                train_args["remove_att_ctx_from_dec_state"] = True
                                exp_name += "_woDecAtt"

                            if kwargs.get("use_curr_enc_for_dec_state", False):
                                train_args["use_curr_enc_for_dec_state"] = True
                                exp_name += "_useCurrFrame"

                            if kwargs.get("asr_ce_scale", None) is not None:
                                train_args["decoder_args"].ce_loss_scale = kwargs["asr_ce_scale"]
                                exp_name += f"_ceScale{kwargs['asr_ce_scale']}"

                            if suffix:
                                exp_name += suffix

                            # override
                            if with_ctc:
                                search_score_key = "dev_score_output/output_prob"
                            else:
                                search_score_key = "dev_score"

                            if chunk_size is None or chunked_decoder is False or from_scratch_train:
                                run_exp(
                                    prefix_name=prefix_name,
                                    exp_name=exp_name,
                                    train_args=train_args,
                                    num_epochs=total_epoch,
                                    epoch_wise_filter=None,
                                    time_rqmt=time_rqmt,
                                    key=search_score_key,
                                    use_sclite=True,
                                    speed_pert=speed_pert,
                                    **kwargs,
                                )
                            elif on_the_fly_align:
                                train_args["search_type"] = "end-of-chunk"  # on-the-fly alignment
                                exp_name += "_onTheFlyAlign"
                                _, train_data = run_exp(
                                    prefix_name=prefix_name,
                                    exp_name=exp_name,
                                    train_args=train_args,
                                    num_epochs=total_epoch,
                                    epoch_wise_filter=None,
                                    time_rqmt=time_rqmt,
                                    key=search_score_key,
                                    use_sclite=True,
                                    do_not_train=do_not_train,
                                    **kwargs,
                                )
                                if return_hdf_align:
                                    return (
                                        train_args,
                                        exp_name,
                                        train_data,
                                        ctc_chunksync_align["train"][f"{chunk_size}_{chunk_step}"],
                                        ctc_chunksync_align["dev"][f"{chunk_size}_{chunk_step}"],
                                    )
                            else:
                                if full_sum_approx:
                                    # just use original targets without EOC
                                    train_fixed_alignment = None
                                    cv_fixed_alignment = None
                                else:
                                    assert ctc_chunksync_align, "Need CTC chunk-sync alignments"
                                    train_fixed_alignment = ctc_chunksync_align["train"][f"{chunk_size}_{chunk_step}"]
                                    cv_fixed_alignment = ctc_chunksync_align["dev"][f"{chunk_size}_{chunk_step}"]
                                _, train_data = run_exp(
                                    prefix_name=prefix_name,
                                    exp_name=exp_name,
                                    train_args=train_args,
                                    num_epochs=total_epoch,
                                    train_fixed_alignment=train_fixed_alignment,
                                    cv_fixed_alignment=cv_fixed_alignment,
                                    epoch_wise_filter=None,
                                    time_rqmt=time_rqmt,
                                    key=search_score_key,
                                    use_sclite=True,
                                    seq_postfix=None if full_sum_approx else 0,
                                    **kwargs,
                                )

                                if return_args:
                                    assert len(total_epochs) == 1
                                    assert len(chunk_sizes) == 1
                                    assert len(chunk_step_factors) == 1
                                    assert len(start_lrs) == 1
                                    assert len(decay_pt_factors) == 1

                                    if return_hdf_align:
                                        return (
                                            train_args,
                                            exp_name,
                                            train_data,
                                            train_fixed_alignment,
                                            cv_fixed_alignment,
                                        )
                                    return train_args, exp_name, train_data


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
        selected_datasets=["dev"],
        key="dev_score_output/output_prob" if with_ctc else "dev_score",
        use_sclite=True,
    )


def py():
    for decoder_mask_eoc in [True, False]:
        run_chunkwise_train(
            enc_stream_type="global",
            run_all_for_best_last_avg=True,
            enable_check_align=False,
            chunk_sizes=[5, 10, 25],
            chunk_step_factors=[1],
            start_lrs=[2e-4],
            decay_pt_factors=[0.0, 0.1, 0.2, 0.25],
            final_lrs=[1e-6, 1e-5],
            gpu_mem=11,
            total_epochs=[20 * 4],
            batch_size=15_000,
            accum_grad=2,
            time_rqmt=120,
            decoder_mask_eoc=decoder_mask_eoc,
        )

    for decoder_mask_eoc in [True, False]:
        run_chunkwise_train(
            enc_stream_type="global",
            run_all_for_best_last_avg=True,
            enable_check_align=False,
            chunk_sizes=[1, 5, 10, 25],
            chunk_step_factors=[1],
            start_lrs=[2e-4],
            decay_pt_factors=[0.25],
            final_lrs=[1e-6],
            gpu_mem=11,
            total_epochs=[20 * 4, 30 * 4, 40 * 4],
            batch_size=15_000,
            accum_grad=2,
            time_rqmt=120,
            decoder_mask_eoc=decoder_mask_eoc,
        )

    # TODO: mask out EOC symbol + ctx from decoder state
    run_chunkwise_train(
        enc_stream_type="global",
        run_all_for_best_last_avg=True,
        enable_check_align=False,
        chunk_sizes=[1, 5, 10, 25],
        chunk_step_factors=[1],
        start_lrs=[2e-4],
        decay_pt_factors=[0.25, 1 / 3],
        final_lrs=[1e-6],
        gpu_mem=11,
        total_epochs=[20 * 4, 40 * 4],
        batch_size=15_000,
        accum_grad=2,
        time_rqmt=120,
        decoder_mask_eoc=True,
        remove_att_ctx_from_dec_state=True,
    )

    # TODO: exact full-sum training
    run_chunkwise_train(
        enc_stream_type="global",
        run_all_for_best_last_avg=True,
        enable_check_align=False,
        chunk_sizes=[25],
        chunk_step_factors=[1],
        start_lrs=[2e-4],
        decay_pt_factors=[0.25, 1 / 3],
        final_lrs=[1e-6],
        gpu_mem=11,
        total_epochs=[20 * 4, 40 * 4],
        batch_size=15_000,
        accum_grad=2,
        time_rqmt=120,
        decoder_mask_eoc=False,  # there are no blanks in the target seq
        remove_att_ctx_from_dec_state=True,  # remove att ctx dependency so we can do exact full sum
        full_sum_approx=True,
    )

    # TODO: exact full-sum training starting from viterbi-trained model

    # # TODO: use overlap
    # run_chunkwise_train(
    #     enc_stream_type="global",
    #     run_all_for_best_last_avg=True,
    #     enable_check_align=False,
    #     chunk_sizes=[5, 10, 25],
    #     chunk_step_factors=[0.5, 0.75],  # 50%, 25% overlaps
    #     start_lrs=[2e-4],
    #     decay_pt_factors=[0.25, 1 / 3],
    #     final_lrs=[1e-6],
    #     gpu_mem=11,
    #     total_epochs=[20 * 4, 40 * 4],
    #     batch_size=15_000,
    #     accum_grad=2,
    #     time_rqmt=120,
    #     decoder_mask_eoc=True,
    #     remove_att_ctx_from_dec_state=True,
    # )
