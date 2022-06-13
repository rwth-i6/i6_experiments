import os
from sisyphus import tk

from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.rossenbach.common_setups.returnn import datasets
from i6_experiments.users.rossenbach.tts.speaker_embedding import SpeakerLabelHDFFromBliss


def _make_meta_dataset(audio_dataset, speaker_dataset):
    """

    :param datasets.OggZipDataset audio_dataset:
    :param datasets.HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    meta_dataset = datasets.MetaDataset(
        data_map={'audio_features': ('audio', 'data'),
                  'phon_labels': ('audio', 'classes'),
                  'speaker_labels': ('speaker', 'data'),
                  },
        datasets={
            'audio': audio_dataset.as_returnn_opts(),
            'speaker': speaker_dataset.as_returnn_opts()
        },
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def build_training_dataset(datagroup, train_dataset_key, cv_dataset_key, vocab_datastream, log_mel_datastream):
    """

    :param DatasetGroup datagroup:
    :param str train_dataset_key:
    :param str cv_dataset_key:
    :param VocabularyDatastream vocab_datastream:
    :param AudioFeatureDatastream log_mel_datastream:
    :return:
    :rtype: tuple(GenericDataset, GenericDataset, dict)
    """
    # check that train and cv are shared
    assert (datagroup.get_segmented_corpus_object(train_dataset_key)[0].corpus_file ==
            datagroup.get_segmented_corpus_object(cv_dataset_key)[0].corpus_file), (
        "train and cv need to share a common corpus file"
    )

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=datagroup.get_segmented_corpus_object(train_dataset_key)[0].corpus_file
    )
    train_cv_speaker_hdf = speaker_label_job.out_speaker_hdf
    num_speakers = speaker_label_job.out_num_speakers

    train_cv_hdf_dataset = datasets.HDFDataset(
        files=train_cv_speaker_hdf
    )

    train_ogg_zip, train_segments = datagroup.get_segmented_zip_dataset(train_dataset_key)
    train_ogg_dataset = datasets.OggZipDataset(
        path=train_ogg_zip,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=2,
        seq_ordering="laplace:.1000"
    )
    train_dataset = _make_meta_dataset(train_ogg_dataset, train_cv_hdf_dataset)

    cv_ogg_zip, cv_segments = datagroup.get_segmented_zip_dataset(cv_dataset_key)
    cv_ogg_dataset = datasets.OggZipDataset(
        path=cv_ogg_zip,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = _make_meta_dataset(cv_ogg_dataset, train_cv_hdf_dataset)

    extern_data = {
        "phon_labels": vocab_datastream.as_returnn_data_opts(),
        "audio_features": log_mel_datastream.as_returnn_data_opts(),
        "speaker_labels": {'shape': (None, ), 'sparse': True, 'dim': num_speakers, 'available_for_inference': True}
    }

    return train_dataset, cv_dataset, extern_data


def tts_training(returnn_config, num_epochs, returnn_gpu_exe, returnn_root, output_path, **kwargs):
    """

    :param ReturnnConfig returnn_config:
    :param int num_epochs:
    :param Path returnn_gpu_exe:
    :param Path returnn_root:
    :param str output_path:
    :param kwargs: additional parameters for ReturnnTrainingJob
    :return:
    :rtype: ReturnnTrainingJob
    """

    additional_args = {
        "time_rqmt": 120,
        "mem_rqmt": 16,
        "cpu_rqmt": 4,
        **kwargs,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        log_verbosity=5,
        num_epochs=num_epochs,
        returnn_python_exe=returnn_gpu_exe,
        returnn_root=returnn_root,
        **additional_args
    )
    train_job.add_alias(os.path.join(output_path, "tts_training"))

    tk.register_output(os.path.join(output_path, "tts_training.models"), train_job.out_model_dir)
    tk.register_output(os.path.join(output_path, "tts_training.config"), train_job.out_returnn_config_file)

    return train_job


def tts_forward(returnn_config, checkpoint, returnn_gpu_exe, returnn_root, output_path):
    """

    :param ReturnnConfig returnn_config: returnn config for the `forward` task
    :param Checkpoint checkpoint:
    :param Path returnn_gpu_exe:
    :param Path returnn_root:
    :param str output_path:
    :return: synthesized audio feature hdf
    :rtype: Path
    """
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=returnn_config,
        hdf_outputs=[],
        returnn_python_exe=returnn_gpu_exe,
        returnn_root=returnn_root
    )

    forward_job.add_alias(os.path.join(output_path, "tts_forward"))

    return forward_job.out_hdf_files["output.hdf"]