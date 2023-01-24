import os
from sisyphus import tk
from typing import List

from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.users.rossenbach.common_setups.returnn import datasets

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments
from i6_experiments.users.rossenbach.setups.tts.preprocessing import process_corpus_text_with_extended_lexicon
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

def get_tts_forward_data_legacy(librispeech_subcorpus, speaker_embedding_hdf, segment_file = None, speaker_embedding_size=256):
    vocab_datastream = get_vocab_datastream()

    bliss_corpus = get_bliss_corpus_dict(audio_format="ogg")[librispeech_subcorpus]
    bliss_corpus_tts_format = process_corpus_text_with_extended_lexicon(
        bliss_corpus=bliss_corpus,
        lexicon=get_lexicon(corpus_key="train-other-960")  # use full lexicon
    )
    speaker_bliss_corpus = get_bliss_corpus_dict()["train-clean-100"]

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_corpus_tts_format,
        no_audio=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    ).out_ogg_zip


    inference_ogg_zip = datasets.OggZipDataset(
        path=zip_dataset,
        audio_options=None,
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=segment_file,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    mapping_pkl = RandomSpeakerAssignmentJob(bliss_corpus=bliss_corpus, speaker_bliss_corpus=speaker_bliss_corpus, shuffle=True).out_mapping


    inference_dataset = _make_inference_meta_dataset(
        inference_ogg_zip, speaker_hdf_dataset, duration_dataset=None
    )

    datastreams = {
        "phon_labels": vocab_datastream,
    }
    datastreams["speaker_labels"] = FeatureDatastream(
        available_for_inference=True, feature_size=speaker_embedding_size)

    return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)