import os
from sisyphus import tk
from typing import List

from i6_core.returnn.training import ReturnnTrainingJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict

from i6_experiments.users.rossenbach.common_setups.returnn import datasets
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioFeatureDatastream, DBMelFilterbankOptions, ReturnnAudioFeatureOptions, FeatureType

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.users.rossenbach.setups.tts.preprocessing import process_corpus_text_with_extended_lexicon, extend_lexicon
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


def get_vocab_datastream(lexicon: tk.Path, alias_path: str) -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param alias_path:
    :return:
    :rtype: VocabularyDatastream
    """
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(alias_path, "returnn_vocab_from_lexicon"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_audio_datastream(
        statistics_ogg_zips: List[tk.Path],
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        alias_path: str,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param alias_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=0.97,
        feature_options=DBMelFilterbankOptions(
            fmin=60,
            fmax=7600,
            min_amp=1e-10,
            center=True,
        )
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options
    )

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        statistics_ogg_zips,
        use_scalar_only=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=alias_path,
    )
    return audio_datastream



def build_training_dataset(returnn_root: tk.Path, returnn_cpu_exe: tk.Path, alias_path: str):
    """

    :param returnn_root:
    :param returnn_cpu_exe:
    :return:
    :rtype: tuple(GenericDataset, GenericDataset, dict)
    """

    # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    # get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    librispeech_g2p_lexicon = extend_lexicon(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True)["train-clean-100"])

    # convert the corpus transcriptions into phoneme and marker representation
    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=librispeech_g2p_lexicon)

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=returnn_cpu_exe,
        returnn_root=returnn_root
    ).out_ogg_zip

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, alias_path)
    log_mel_datastream = get_audio_datastream(
        statistics_ogg_zips=[zip_dataset],
        returnn_python_exe=returnn_cpu_exe,
        returnn_root=returnn_root,
        alias_path=alias_path
    )

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=sil_pp_train_clean_100_tts
    )
    train_cv_speaker_hdf = speaker_label_job.out_speaker_hdf
    num_speakers = speaker_label_job.out_num_speakers

    train_cv_hdf_dataset = datasets.HDFDataset(
        files=train_cv_speaker_hdf
    )

    train_ogg_dataset = datasets.OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=2,
        seq_ordering="laplace:.1000"
    )
    train_dataset = _make_meta_dataset(train_ogg_dataset, train_cv_hdf_dataset)

    cv_ogg_dataset = datasets.OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = _make_meta_dataset(cv_ogg_dataset, train_cv_hdf_dataset)

    extern_data = {
        "phon_labels": vocab_datastream.as_returnn_extern_data_opts(),
        "audio_features": log_mel_datastream.as_returnn_extern_data_opts(),
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