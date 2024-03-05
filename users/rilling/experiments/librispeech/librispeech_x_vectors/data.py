"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper, BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_lexicon,
    get_bliss_corpus_dict,
    get_ogg_zip_dict,
    get_arpa_lm_dict,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments

from i6_experiments.users.rossenbach.datasets.librispeech import (
    get_librispeech_tts_segments,
    get_ls_train_clean_100_tts_silencepreprocessed,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
    AudioFeatureDatastream,
    DBMelFilterbankOptions,
    ReturnnAudioFeatureOptions,
    FeatureType,
    ReturnnAudioRawOptions,
    AudioRawDatastream,
)


from i6_experiments.users.rossenbach.setups.tts.preprocessing import (
    process_corpus_text_with_extended_lexicon,
    extend_lexicon_with_tts_lemmas,
    extend_lexicon_with_blank,
)

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from .default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

DATA_PREFIX = "experiments/librispeech/librispeech_glow_asr/data/"

# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]


@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    epoch_wise_filters: List[EpochWiseFilter]
    seq_ordering: str


# --------------------------- Helper functions  -----------------------------------


def get_librispeech_lexicon(corpus_key="train-clean-100", with_g2p: bool = True) -> tk.Path:
    """
    get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    :return:
    """
    if (with_g2p):
        return extend_lexicon_with_tts_lemmas(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)[corpus_key])
    else:
        return extend_lexicon_with_tts_lemmas(get_bliss_lexicon(use_stress_marker=False))

def get_text_lexicon(corpus_key="train-clean-100") -> tk.Path:
    lexicon = get_lexicon(with_blank=True, with_g2p=False, corpus_key=corpus_key)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(lexicon).out_lexicon
    return word_lexicon

def get_arpa_lm(lm_key="4gram") -> tk.Path:
    return get_arpa_lm_dict()[lm_key]


def get_tts_extended_bliss(ls_corpus_key, remove_unk_seqs=False) -> tk.Path:
    """
    get a modified ls corpus using the TTS processing
    :param ls_corpus_key
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    ls_bliss = get_bliss_corpus_dict(audio_format="ogg")[ls_corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        ls_bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=ls_bliss, bliss_lexicon=get_lexicon(), all_unknown=False
        ).out_corpus
    tts_ls_bliss = process_corpus_text_with_extended_lexicon(
        bliss_corpus=ls_bliss, lexicon=get_librispeech_lexicon(corpus_key="train-clean-100")
    )

    return tts_ls_bliss


def get_lexicon(with_blank: bool = False, with_g2p: bool = True, corpus_key: str ="train-clean-100") -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_lexicon(corpus_key=corpus_key, with_g2p=with_g2p)
    lexicon = extend_lexicon_with_tts_lemmas(lexicon)
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon


def get_vocab_datastream(with_blank: bool = False, corpus_key="train-clean-100") -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param with_blank: datastream for CTC training
    """
    lexicon = get_lexicon(with_blank, corpus_key=corpus_key)
    blacklist = {"[SILENCE]"}
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist=blacklist)
    name = "returnn_vocab_from_lexicon_with_blank" if with_blank else "returnn_vocab_from_lexicon"
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, name))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


@lru_cache
def get_ls100_silence_preprocessed_bliss() -> tk.Path:
    """
    Get the modified ls100 corpus for the TTS task with silence preprocessing
    :return: Bliss xml file
    """
    # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    # convert the corpus transcriptions into phoneme and marker representation
    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file, lexicon=get_librispeech_lexicon()
    )

    return sil_pp_train_clean_100_tts


def get_train_bliss_and_zip(ls_corpus_key, silence_preprocessed=True, remove_unk_seqs=False):
    """
    :param ls_corpus_key: e.g. train-clean-100, see LibriSpeech data definition
    :param for_training:
    :param silence_preprocessed:
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    if silence_preprocessed:
        if ls_corpus_key == "train-clean-100":
            bliss_dataset = get_ls100_silence_preprocessed_bliss()
        else:
            assert False, "invalid key %s" % ls_corpus_key
    else:
        bliss_dataset = get_tts_extended_bliss(ls_corpus_key=ls_corpus_key, remove_unk_seqs=remove_unk_seqs)

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    ).out_ogg_zip

    return bliss_dataset, zip_dataset


def get_test_bliss_and_zip(ls_corpus_key):
    """
    for now just return the original ogg zip

    :param ls_corpus_key: e.g. train-clean-100, see LibriSpeech data definition
    :return:
    """
    return (
        get_bliss_corpus_dict(audio_format="ogg")[ls_corpus_key],
        get_ogg_zip_dict(returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)[ls_corpus_key],
    )


def get_tts_log_mel_datastream(silence_preprocessing=False) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    Supports both centered and non-centered windowing, as we need non-centered windowing for RASR-compatible
    feature extraction, but centered windowing to support linear-features for the vocoder mel-to-linear training.

    :param silence_preprocessing: calculates statistics with silence-preprocessed data
    """
    feature_options_center = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=0.97,
        sample_rate=16000,
        feature_options=DBMelFilterbankOptions(
            fmin=60,
            fmax=7600,
            min_amp=1e-10,
            center=True,
        ),
    )
    audio_datastream = AudioFeatureDatastream(available_for_inference=False, options=feature_options_center)

    ls100_bliss, ls100_ogg_zip = get_train_bliss_and_zip("train-clean-100", silence_preprocessed=silence_preprocessing)
    train_segments, _ = get_librispeech_tts_segments()

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        [ls100_ogg_zip],
        segment_file=train_segments,
        use_scalar_only=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        alias_path=DATA_PREFIX + "ls100/",
    )
    return audio_datastream


@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True, options=ReturnnAudioRawOptions(peak_normalization=False, preemphasis=0.97)
    )
    return audio_datastream


# --------------------------- Dataset functions  -----------------------------------


def build_training_datasets(
    librispeech_key: str,
    settings: TrainingDatasetSettings,
    silence_preprocessing=False,
) -> TrainingDatasets:
    """

    :param settings:
    :param output_path:
    """

    assert silence_preprocessing is False, "Silence preprocessing not supported yet"

    train_bliss, train_ogg = get_train_bliss_and_zip("train-clean-100", silence_preprocessed=False)
    # _, dev_clean_ogg = get_train_bliss_and_zip("dev-clean", silence_preprocessed=False, remove_unk_seqs=True)
    # _, dev_other_ogg = get_train_bliss_and_zip("dev-other", silence_preprocessed=False, remove_unk_seqs=True)

    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=librispeech_key)


    train_bpe_datastream = get_vocab_datastream(corpus_key=librispeech_key, with_blank=True)
    audio_datastream = get_audio_raw_datastream()

    datastreams = {
        "raw_audio": audio_datastream,
        "phon_labels": train_bpe_datastream,
    }

    data_map = {"raw_audio": ("zip_dataset", "data"), "phon_labels": ("zip_dataset", "classes")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts["epoch_wise_filter"] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts["epoch_wise_filter"][(fr, to)] = {"max_mean_len": max_mean_len}

    def make_meta(dataset: OggZipDataset):
        return MetaDataset(
            data_map=data_map, datasets={"zip_dataset": dataset}, seq_order_control_dataset="zip_dataset"
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
        segment_file=train_segments
    )
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        seq_ordering="sorted_reverse"
    )
    cv_dataset = make_meta(cv_zip_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta(devtrain_zip_dataset)

    return TrainingDatasets(train=train_dataset, cv=cv_dataset, devtrain=devtrain_dataset, datastreams=datastreams)


@lru_cache()
def build_test_dataset(librispeech_key: str, dataset_key: str, silence_preprocessing=False):
    """

    :param librispeech_key: base librispeech training set for vocab generation
    :param dataset_key: test dataset to generate
    :param silence_preprocessing: use a setup with silence preprocessing
    """
    assert silence_preprocessing is False

    _, test_ogg = get_test_bliss_and_zip(dataset_key)
    bliss_dict = get_bliss_corpus_dict()
    audio_datastream = get_audio_raw_datastream()

    data_map = {
        "raw_audio": ("zip_dataset", "data")
    }

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_dict[dataset_key]
