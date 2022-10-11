import os
from sisyphus import tk
from typing import List

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.forward import ReturnnForwardJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict

from i6_experiments.users.rossenbach.common_setups.returnn import datasets
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.users.rossenbach.setups.tts.preprocessing import process_corpus_text_with_extended_lexicon, extend_lexicon_with_tts_lemmas
from i6_experiments.users.rossenbach.tts.speaker_embedding import SpeakerLabelHDFFromBliss

from ..data import (
    get_tts_log_mel_datastream,
    get_ls100_silence_preprocess_ogg_zip,
    get_ls100_silence_preprocessed_bliss,
    get_vocab_datastream,
    make_meta_dataset
)

def build_training_dataset():
    """
    :return:
    :rtype: tuple(GenericDataset, GenericDataset, dict)
    """

    bliss_dataset = get_ls100_silence_preprocessed_bliss()
    zip_dataset = get_ls100_silence_preprocess_ogg_zip()

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream()
    log_mel_datastream = get_tts_log_mel_datastream()

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=bliss_dataset
    )
    train_cv_speaker_hdf = speaker_label_job.out_speaker_hdf
    num_speakers = speaker_label_job.out_num_speakers

    train_cv_hdf_dataset = datasets.HDFDataset(
        files=train_cv_speaker_hdf
    )

    train_ogg_dataset = datasets.OggZipDataset(
        path=zip_dataset,
        audio_options=log_mel_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=2,
        seq_ordering="laplace:.1000"
    )
    train_dataset = make_meta_dataset(train_ogg_dataset, train_cv_hdf_dataset)

    cv_ogg_dataset = datasets.OggZipDataset(
        path=zip_dataset,
        audio_options=log_mel_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_meta_dataset(cv_ogg_dataset, train_cv_hdf_dataset)

    extern_data = {
        "phon_labels": vocab_datastream.as_returnn_extern_data_opts(),
        "audio_features": log_mel_datastream.as_returnn_extern_data_opts(),
        "speaker_labels": {'shape': (None, ), 'sparse': True, 'dim': num_speakers, 'available_for_inference': True}
    }

    return train_dataset, cv_dataset, extern_data


def build_dev_dataset(returnn_root: tk.Path, returnn_cpu_exe: tk.Path, alias_path: str):
    pass


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