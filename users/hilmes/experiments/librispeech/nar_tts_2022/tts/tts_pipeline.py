"""
Pipeline file for experiments with the standard CTC TTS model
"""
from sisyphus import tk
from i6_core.returnn import ReturnnConfig, ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.corpus import (
    CorpusReplaceOrthFromReferenceCorpus,
    MergeCorporaJob,
    SegmentCorpusJob,
)
from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Import,
    Network,
    PythonEnlargeStackWorkaroundCode,
)
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict
from i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.data import (
    TTSTrainingDatasets,
    TTSForwardData,
)
from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
from i6_experiments.users.hilmes.experiments.librispeech.util.asr_evaluation import (
    asr_evaluation,
)


def get_training_config(
    returnn_common_root: tk.Path, training_datasets: TTSTrainingDatasets, **kwargs
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for the ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param training_datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN training config
    """
    post_config = {
        "cleanup_old_models": True,
        "use_tensorflow": True,
        "tf_log_memory_usage": True,
        "stop_on_nonfinite_train_score": False,
        "log_batch_size": True,
        "debug_print_layer_output_template": True,
        "cache_size": "0",
    }
    config = {
        "behavior_version": 12,
        ############
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "accum_grad_multiple_step": 2,
        "gradient_clip": 1,
        "gradient_noise": 0,
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_score_dec_output",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 18000,
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 60,
    }

    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in training_datasets.datastreams.items()
    ]
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()

    rc_recursionlimit = PythonEnlargeStackWorkaroundCode
    rc_extern_data = ExternData(extern_data=extern_data)
    rc_model = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.NARTTSModel"
    )
    rc_construction_code = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.tts_model.construct_network"
    )

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map={
            "net_module": rc_model.object_name,
            "phoneme_data": "phonemes",
            "duration_data": "duration_data",
            "label_data": "speaker_labels",
            "audio_data": "audio_features",
            "time_dim": "phonemes_time",
            "label_time_dim": "speaker_labels_time",
            "speech_time_dim": "audio_features_time",
            "duration_time_dim": "duration_data_time",
        },
        net_kwargs={"training": True, **kwargs},
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_model,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
        packages={
            "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
            "i6_experiments.users.hilmes.modules",
        },
    )

    returnn_config = ReturnnConfig(
        config=config, post_config=post_config, python_epilog=[serializer]
    )

    return returnn_config


def get_forward_config(
    returnn_common_root,
    forward_dataset: TTSForwardData,
    use_true_durations: bool = False,
    **kwargs,
):
    """
    Returns the RETURNN config serialized by :class:`ReturnnCommonSerializer` in returnn_common for forward_ctc_model
    :param returnn_common_root: returnn_common version to be used, usually output of CloneGitRepositoryJob
    :param datasets: datasets for training
    :param kwargs: arguments to be passed to the network construction
    :return: RETURNN forward config
    """

    config = {
        "behavior_version": 12,
        "forward_batch_size": 18000,
        "max_seqs": 60,
        "forward_use_search": True,
        "target": "dec_output",
    }
    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in forward_dataset.datastreams.items()
    ]
    config["eval"] = forward_dataset.dataset.as_returnn_opts()

    rc_recursionlimit = PythonEnlargeStackWorkaroundCode
    rc_extern_data = ExternData(extern_data=extern_data)
    rc_model = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.NARTTSModel"
    )
    rc_construction_code = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.construct_network"
    )

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map={
            "net_module": rc_model.object_name,
            "phoneme_data": "phonemes",
            "duration_data": "duration_data",
            "label_data": "speaker_labels",
            "audio_data": "audio_features",
            "time_dim": "phonemes_time",
            "label_time_dim": "speaker_labels_time",
            "speech_time_dim": "audio_features_time",
            "duration_time_dim": "duration_data_time",
        },
        net_kwargs={
            "training": False,
            "use_true_durations": use_true_durations,
            **kwargs,
        },
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_model,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
        packages={
            "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
            "i6_experiments.users.hilmes.modules",
        },
    )

    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

    return returnn_config


def get_extraction_config(
    returnn_common_root, forward_dataset: TTSForwardData, **kwargs
):
    """

    :param returnn_common_root:
    :param forward_dataset:
    :param kwargs:
    :return:
    """
    config = {
        "behavior_version": 12,
        "forward_batch_size": 18000,
        "max_seqs": 60,
        "forward_use_search": True,
        "target": "dec_output",
    }
    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in forward_dataset.datastreams.items()
    ]
    config["eval"] = forward_dataset.dataset.as_returnn_opts()

    rc_recursionlimit = PythonEnlargeStackWorkaroundCode
    rc_extern_data = ExternData(extern_data=extern_data)
    rc_model = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.NARTTSModel"
    )
    rc_construction_code = Import(
        "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks.construct_network"
    )

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map={
            "net_module": rc_model.object_name,
            "phoneme_data": "phonemes",
            "duration_data": "duration_data",
            "label_data": "speaker_labels",
            "audio_data": "audio_features",
            "time_dim": "phonemes_time",
            "label_time_dim": "speaker_labels_time",
            "speech_time_dim": "audio_features_time",
            "duration_time_dim": "duration_data_time",
        },
        net_kwargs={"training": False, "dump_speaker_embeddings": True, **kwargs},
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_model,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
        packages={
            "i6_experiments.users.hilmes.experiments.librispeech.nar_tts_2022.networks",
            "i6_experiments.users.hilmes.modules",
        },
    )

    returnn_config = ReturnnConfig(config=config, python_epilog=[serializer])

    return returnn_config


def tts_training(config, returnn_exe, returnn_root, prefix, num_epochs=200):
    """

    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :param num_epochs:
    :return:
    """
    train_job = ReturnnTrainingJob(
        config,
        log_verbosity=5,
        num_epochs=num_epochs,
        time_rqmt=120,
        mem_rqmt=32,
        cpu_rqmt=4,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    train_job.add_alias(prefix + "/training")
    tk.register_output(prefix + "/training.models", train_job.out_model_dir)

    return train_job


def tts_forward(checkpoint, config, returnn_exe, returnn_root, prefix):
    """

    :param checkpoint:
    :param config:
    :param returnn_exe:
    :param returnn_root:
    :param prefix:
    :return:
    """
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=config,
        hdf_outputs=[],
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    forward_job.add_alias(prefix + "/forward")

    return forward_job


def gl_swer(name, vocoder, checkpoint, config, returnn_root, returnn_exe):
    """

    :param name:
    :param vocoder:
    :param checkpoint:
    :param config:
    :param returnn_root:
    :param returnn_exe:
    :return:
    """
    forward_job = tts_forward(
        checkpoint=checkpoint,
        config=config,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        prefix=name,
    )
    forward_hdf = forward_job.out_hdf_files["output.hdf"]
    forward_vocoded, vocoder_forward_job = vocoder.vocode(
        forward_hdf, iterations=30, cleanup=True
    )
    verification = VerifyCorpus(forward_vocoded).out
    cleanup = MultiJobCleanup(
        [forward_job, vocoder_forward_job], verification, output_only=True
    )
    tk.register_output(name + "/ctc_model" + "/".join(["cleanup", name]), cleanup.out)

    corpus_object_dict = get_corpus_object_dict(
        audio_format="ogg", output_prefix="corpora"
    )
    cv_synth_corpus_job = CorpusReplaceOrthFromReferenceCorpus(
        forward_vocoded, corpus_object_dict["train-clean-100"].corpus_file
    )
    cv_synth_corpus_job.add_input(verification)
    cv_synth_corpus = cv_synth_corpus_job.out_corpus
    librispeech_trafo = tk.Path(
        "/u/rossenbach/experiments/librispeech_tts/config/evaluation/asr/pretrained_configs/trafo.specaug4.12l.ffdim4."
        "pretrain3.natctc_recognize_pretrained.config"
    )
    asr_evaluation(
        config_file=librispeech_trafo,
        corpus=cv_synth_corpus,
        output_path=name + "/ctc_model",
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
    )


def synthesize_with_splits(
    name,
    reference_corpus: tk.Path,
    corpus_name: str,
    job_splits: int,
    datasets: TTSForwardData,
    returnn_root,
    returnn_exe,
    returnn_common_root,
    checkpoint,
    vocoder,
    **tts_model_kwargs,
):
    """

    :param name:
    :param corpus: Needs to be the matching corpus for datasets
    :param corpus_name: Name of the corpus for the ReplaceOrthJob
    :param job_splits: number of splits performed
    :param datasets: datasets including datastream supposed to hold the audio data in .train
    :param returnn_root:
    :param returnn_exe:
    :param returnn_common_root:
    :param checkpoint:
    :param vocoder:
    :param tts_model_kwargs: kwargs to be passed to the tts model for synthesis
    :return:
    """
    forward_segments = SegmentCorpusJob(reference_corpus, job_splits)

    verifications = []
    output_corpora = []
    for i in range(job_splits):
        name = name + "/synth_corpus/part_%i/" % i
        forward_config = get_forward_config(
            returnn_common_root=returnn_common_root,
            forward_dataset=datasets,
            **tts_model_kwargs,
        )
        forward_config.config["eval"]["datasets"]["audio"][
            "segment_file"
        ] = forward_segments.out_single_segment_files[i + 1]

        last_forward_job = ReturnnForwardJob(
            model_checkpoint=checkpoint,
            returnn_config=forward_config,
            hdf_outputs=[],
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
        )
        last_forward_job.add_alias(name + "forward")
        forward_hdf = last_forward_job.out_hdf_files["output.hdf"]
        tk.register_output(name, forward_hdf)

        forward_vocoded, vocoder_forward_job = vocoder.vocode(
            forward_hdf, iterations=30, cleanup=True, name=name
        )
        tk.register_output(name + "synthesized_corpus.xml.gz", forward_vocoded)
        output_corpora.append(forward_vocoded)
        verification = VerifyCorpus(forward_vocoded).out
        verifications.append(verification)

        cleanup = MultiJobCleanup(
            [last_forward_job, vocoder_forward_job], verification, output_only=True
        )
        tk.register_output(
            name + "/".join(["cleanup", "synth_corpus/part_%i/" % i]), cleanup.out
        )

    from i6_core.corpus.transform import MergeStrategy

    merge_job = MergeCorporaJob(
        output_corpora, corpus_name, merge_strategy=MergeStrategy.FLAT
    )
    for verfication in verifications:
        merge_job.add_input(verfication)

    cv_synth_corpus = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=merge_job.out_merged_corpus,
        reference_bliss_corpus=reference_corpus,
    ).out_corpus

    tk.register_output(name + "synth_corpus/synthesized_corpus.xml.gz", cv_synth_corpus)
    return cv_synth_corpus
