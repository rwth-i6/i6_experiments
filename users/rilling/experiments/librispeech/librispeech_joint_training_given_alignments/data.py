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
from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob
from i6_core.lm.kenlm import CreateBinaryLMJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

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

from i6_experiments.users.rilling.joint_training.text_hdf.text_hdf_from_bliss import TextHDFFromBliss

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset, HDFDataset

from .default_tools import MINI_RETURNN_ROOT, RETURNN_EXE, KENLM_BINARY_PATH

DATA_PREFIX = "experiments/librispeech/data/joint"

# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]


@dataclass(frozen=True)
class TrainingDataset:
    """
    Dataclass for Alignment Datasets
    """

    train: Dataset
    cv: Dataset
    devtrain: Dataset
    cv_asr: Dataset
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


def get_librispeech_lexicon(corpus_key="train-clean-100", with_g2p: bool = True, add_silence=True) -> tk.Path:
    """
    get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    :return:
    """
    if with_g2p:
        return extend_lexicon_with_tts_lemmas(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=add_silence)[corpus_key])
    else:
        return extend_lexicon_with_tts_lemmas(get_bliss_lexicon(use_stress_marker=False, add_silence=add_silence))

def get_librispeech_eow_lexicon(corpus_key="train-clean-100", with_g2p=True) -> tk.Path:

    """
    get the g2p bliss lexicon with EOW tokens added
    :return:
    """
    if with_g2p:
        lex =  get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)[corpus_key]
    else:
        lex =  get_bliss_lexicon(use_stress_marker=False)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_text_lexicon(corpus_key="train-clean-100") -> tk.Path:
    """
    Get the lexicon of the librispeech corpus with the given key in txt format
    adds blank and tts lemmas but does not add [SILENCE] since this is used 
    to define the lexicon for CTC decoder

    :param str corpus_key: Key of the librispeech corpus, defaults to "train-clean-100"
    :return tk.Path: Path to the txt file containing the lexicon
    """
    # lexicon = get_tts_lexicon(with_blank=True, with_g2p=False, corpus_key=corpus_key, add_silence=False) # Without adding silence,
    lexicon = get_asr_lexicon(corpus_key=corpus_key, with_g2p=True)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon

    word_lexicon = BlissLexiconToWordLexicon(lexicon).out_lexicon
    return word_lexicon


def get_arpa_lm(lm_key="4gram") -> tk.Path:
    return get_arpa_lm_dict()[lm_key]

def get_binary_lm(lm_key="4gram") -> tk.Path:
    arpa_lm = get_arpa_lm(lm_key)
    lm = CreateBinaryLMJob(arpa_lm=arpa_lm, kenlm_binary_folder=KENLM_BINARY_PATH).out_lm
    return lm

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
            bliss_corpus=ls_bliss, bliss_lexicon=get_tts_lexicon(), all_unknown=False
        ).out_corpus
    tts_ls_bliss = process_corpus_text_with_extended_lexicon(
        bliss_corpus=ls_bliss, lexicon=get_librispeech_lexicon(corpus_key="train-clean-100")
    )

    return tts_ls_bliss


def get_tts_lexicon(with_blank: bool = False, with_g2p: bool = True, corpus_key: str = "train-clean-100", add_silence=True) -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_lexicon(corpus_key=corpus_key, with_g2p=with_g2p, add_silence=add_silence)
    lexicon = extend_lexicon_with_tts_lemmas(lexicon)
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon

def get_asr_lexicon(corpus_key="train-clean-100", with_g2p=True) -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_eow_lexicon(corpus_key=corpus_key, with_g2p=with_g2p)
    return lexicon


def get_vocab_datastream_from_lexicon(lexicon, with_blank: bool = False) -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param with_blank: datastream for CTC training
    """
    # lexicon = get_tts_lexicon(with_blank, corpus_key=corpus_key)
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

def filter_bliss_corpus_by_segments(corpus, segments):
    return FilterCorpusBySegmentsJob(bliss_corpus=corpus, segment_file=segments).out_corpus

def get_asr_bliss(ls_corpus_key, remove_unk_seqs=False) -> tk.Path:
    """
    get a modified ls corpus with unknown removed for cross validation

    :param ls_corpus_key
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    ls_bliss = get_bliss_corpus_dict(audio_format="ogg")[ls_corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob
        ls_bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=ls_bliss,
            bliss_lexicon=get_asr_lexicon(),
            all_unknown=False
        ).out_corpus

    # default train lexicon
    lexicon = get_librispeech_eow_lexicon(with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(ls_bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus

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


def make_meta_dataset(audio_dataset, speaker_dataset, asr_text_dataset, duration_dataset=None):
    """
    Shared function to create a metadatset with joined audio and speaker information

    :param datasets.OggZipDataset audio_dataset:
    :param datasets.HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    data_map = {
        "audio_features": ("audio", "data"),
        "phonemes": ("audio", "classes"),
        "phonemes_eow": ("asr_labels", "data"),
    }

    ds = {"audio": audio_dataset.as_returnn_opts(), "asr_labels": asr_text_dataset.as_returnn_opts()}

    if speaker_dataset is not None:
        data_map["speaker_labels"] = ("speaker", "data")
        ds["speaker"] = speaker_dataset.as_returnn_opts()

    if duration_dataset:
        data_map["durations"] = ("durations", "data")
        ds["durations"] = duration_dataset.as_returnn_opts()

    meta_dataset = MetaDataset(
        data_map=data_map,
        datasets=ds,
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def build_training_dataset(
    librispeech_key: str,
    settings: TrainingDatasetSettings,
    silence_preprocessing=False,
    use_tts_train_segments=False,
    durations_file=None,
) -> TrainingDataset:
    """

    :param settings:
    :param output_path:
    """
    assert durations_file is None or (
        durations_file is not None and use_tts_train_segments
    ), "Using TTS Train Segments is mandatory when using external durations"

    if use_tts_train_segments:
        train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=librispeech_key)
    else:
        train_segments = None

    train_bliss, train_ogg = get_train_bliss_and_zip("train-clean-100", silence_preprocessed=silence_preprocessing)
    dev_clean_bliss_tts, dev_clean_ogg = get_train_bliss_and_zip(
        "dev-clean", silence_preprocessed=silence_preprocessing, remove_unk_seqs=True
    )
    dev_other_bliss_tts, dev_other_ogg = get_train_bliss_and_zip(
        "dev-other", silence_preprocessed=silence_preprocessing, remove_unk_seqs=True
    )

    tts_lexicon = get_tts_lexicon(with_blank=True)
    train_phoneme_datastream_tts = get_vocab_datastream_from_lexicon(tts_lexicon, with_blank=True)

    train_bliss_asr = get_asr_bliss("train-clean-100") 
    asr_lexicon = get_asr_lexicon(corpus_key=librispeech_key)
    train_phoneme_datastream_asr = get_vocab_datastream_from_lexicon(asr_lexicon, with_blank=True)
    train_eow_phonemes_hdf_job = TextHDFFromBliss(train_bliss_asr, train_phoneme_datastream_asr.vocab)
    train_eow_phonemes_dataset = HDFDataset(files=[train_eow_phonemes_hdf_job.out_text_hdf], segment_file=train_segments)
    cv_eow_phonemes_dataset = HDFDataset(files=[train_eow_phonemes_hdf_job.out_text_hdf], segment_file=cv_segments)

    devclean_asr_lexicon = get_asr_lexicon("dev-clean", with_g2p=False)
    devother_asr_lexicon = get_asr_lexicon("dev-other", with_g2p=False)
    devclean_bliss_asr = get_asr_bliss("dev-clean", remove_unk_seqs=True)
    devother_bliss_asr = get_asr_bliss("dev-other", remove_unk_seqs=True)
    devclean_phoneme_datastream_asr = get_vocab_datastream_from_lexicon(devclean_asr_lexicon, with_blank=True)
    devother_phoneme_datastream_asr = get_vocab_datastream_from_lexicon(devother_asr_lexicon, with_blank=True)
    dev_eow_phonemes_hdf_job = TextHDFFromBliss([devclean_bliss_asr, devother_bliss_asr], [devclean_phoneme_datastream_asr.vocab, devother_phoneme_datastream_asr.vocab])
    dev_eow_phonemes_dataset = HDFDataset(files=[dev_eow_phonemes_hdf_job.out_text_hdf])

    audio_datastream = get_audio_raw_datastream()

    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=train_bliss,
        returnn_root=MINI_RETURNN_ROOT,
    )
    joint_speaker_hdf = speaker_label_job.out_speaker_hdf

    # dev_clean_speaker_label_job = SpeakerLabelHDFFromBlissJob(
    #     bliss_corpus=dev_clean_bliss_tts,
    #     returnn_root=MINI_RETURNN_ROOT,
    # )
    # dev_clean_speaker_hdf = dev_clean_speaker_label_job.out_speaker_hdf
    # dev_other_speaker_label_job = SpeakerLabelHDFFromBlissJob(
    #     bliss_corpus=dev_other_bliss_tts,
    #     returnn_root=MINI_RETURNN_ROOT,
    # )
    # dev_other_speaker_hdf = dev_other_speaker_label_job.out_speaker_hdf

    joint_speaker_dataset = HDFDataset(files=[joint_speaker_hdf])
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    datastreams = {
        "audio_features": audio_datastream,
        "phonemes": train_phoneme_datastream_tts,
        "phonemes_eow": train_phoneme_datastream_asr,
        "speaker_labels": speaker_datastream,
    }

    if durations_file is not None:
        duration_dataset = HDFDataset(files=[durations_file])
        duration_dataset_train = HDFDataset(files=[durations_file], segment_file=train_segments)
        duration_dataset_cv = HDFDataset(files=[durations_file], segment_file=cv_segments)
    else:
        duration_dataset, duration_dataset_train, duration_dataset_cv = (None, None, None)

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts["epoch_wise_filter"] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts["epoch_wise_filter"][(fr, to)] = {"max_mean_len": max_mean_len}

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_phoneme_datastream_tts.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        segment_file=train_segments,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta_dataset(train_zip_dataset, joint_speaker_dataset, train_eow_phonemes_dataset, duration_dataset=duration_dataset_train)

    cv_zip_dataset_asr = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_phoneme_datastream_tts.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
    )
    cv_dataset_asr = make_meta_dataset(cv_zip_dataset_asr, None, dev_eow_phonemes_dataset)

    cv_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_phoneme_datastream_tts.as_returnn_targets_opts(),
        segment_file=cv_segments,
        seq_ordering="sorted",
    )
    cv_dataset = make_meta_dataset(cv_zip_dataset, joint_speaker_dataset, cv_eow_phonemes_dataset, duration_dataset=duration_dataset_cv)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_phoneme_datastream_tts.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta_dataset(devtrain_zip_dataset, joint_speaker_dataset, train_eow_phonemes_dataset, duration_dataset=duration_dataset)

    return TrainingDataset(train=train_dataset, cv=cv_dataset, cv_asr=cv_dataset_asr, devtrain=devtrain_dataset, datastreams=datastreams)


@lru_cache()
def build_test_dataset(librispeech_key: str, dataset_key: str, silence_preprocessing=False, test_on_tts_cv=False):
    """

    :param librispeech_key: base librispeech training set for vocab generation
    :param dataset_key: test dataset to generate
    :param silence_preprocessing: use a setup with silence preprocessing
    """
    assert silence_preprocessing is False
    assert not test_on_tts_cv or (test_on_tts_cv and dataset_key == "train-clean-100")
    if test_on_tts_cv:
        _, test_ogg = get_train_bliss_and_zip(dataset_key)
    else:
        _, test_ogg = get_test_bliss_and_zip(dataset_key)
    bliss_dict = get_bliss_corpus_dict()
    audio_datastream = get_audio_raw_datastream()

    if (test_on_tts_cv):
        _, cv_segments = get_librispeech_tts_segments(ls_corpus_key=dataset_key)
        bliss_corpus = filter_bliss_corpus_by_segments(bliss_dict[dataset_key], cv_segments)
    else:
        cv_segments = None
        bliss_corpus = bliss_dict[dataset_key]

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
        segment_file=cv_segments
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_corpus
