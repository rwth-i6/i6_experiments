from dataclasses import dataclass, asdict
from functools import lru_cache
import os
from sisyphus import tk
from typing import List, Dict, Optional, Union


from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob
from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.corpus.segments import SegmentCorpusJob

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.loquacious.lexicon import get_g2p_augmented_bliss_lexicon_dict, get_bliss_lexicon
from i6_experiments.users.rossenbach.datasets.loquacious import get_loquacious_tts_segments

from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, HDFDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.base import Datastream, FeatureDatastream
from i6_experiments.common.setups.returnn.datastreams.audio import (
    AudioFeatureDatastream,
    DBMelFilterbankOptions,
    ReturnnAudioFeatureOptions,
    FeatureType,
)
from i6_experiments.common.setups.returnn import datasets as returnn_datasets



from i6_experiments.users.rossenbach.setups.tts.preprocessing import (
    process_corpus_text_with_extended_lexicon,
    extend_lexicon_with_tts_lemmas,
    extend_lexicon_with_blank
)
from ..common import get_audio_raw_datastream, TrainingDatasets
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

DATA_PREFIX = "experiments/alignment_analysis_tts/data/"


@lru_cache
def get_loquacious_extended_lexicon(with_blank: bool, corpus_key="train-small") -> tk.Path:
    """
    get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    :return:
    """
    lexicon = extend_lexicon_with_tts_lemmas(
        get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=False,
            add_silence=False
        )[corpus_key]
    )
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon

def get_text_lexicon(corpus_key="train-small") -> tk.Path:
    bliss_lex = get_loquacious_extended_lexicon(with_blank=False, corpus_key=corpus_key)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def get_tts_extended_bliss(loq_corpus_key, lexicon_loq_corpus_key: Optional[str] = None, verify_lexicon=False) -> tk.Path:
    """
    get a modified ls corpus using the TTS processing

    :param loq_corpus_key: which loquacious part to get
    :param lexicon_loq_corpus_key: which librispeech part to use as base
    :return:
    """
    ls_bliss = get_bliss_corpus_dict()[loq_corpus_key]
    lexicon = get_loquacious_extended_lexicon(with_blank=False, corpus_key=lexicon_loq_corpus_key or loq_corpus_key)
    tts_ls_bliss = process_corpus_text_with_extended_lexicon(
        bliss_corpus=ls_bliss,
        lexicon=lexicon)

    return tts_ls_bliss


def make_tts_meta_dataset(audio_dataset, speaker_dataset, duration_dataset=None, include_speakers=True):
    """
    Shared function to create a metadatset with joined audio and speaker information

    :param datasets.OggZipDataset audio_dataset:
    :param datasets.HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    data_map = {
        'raw_audio': ('audio', 'data'),
        'phonemes': ('audio', 'classes'),
    }
    if include_speakers:
        data_map["speaker_embeddings"] = ("speaker", "data")

    datasets = {
        'audio': audio_dataset.as_returnn_opts(),
    }
    if include_speakers:
        datasets["speaker"] = speaker_dataset.as_returnn_opts()

    if duration_dataset:
        data_map["durations"] = ("durations", "data")
        datasets["durations"] = duration_dataset.as_returnn_opts()

    meta_dataset = returnn_datasets.MetaDataset(
        data_map=data_map,
        datasets=datasets,
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def get_tts_bliss_and_zip(loq_corpus_key, silence_preprocessed=False):
    """
    :param ls_corpus_key: e.g. train-clean-100, see LibriSpeech data definition
    :param silence_preprocessed:
    :return:
    """
    if silence_preprocessed:
        assert "Silence preprocessed not added yet here"

    bliss_dataset = get_tts_extended_bliss(loq_corpus_key=loq_corpus_key)

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    ).out_ogg_zip

    return bliss_dataset, zip_dataset


@lru_cache
def get_extended_lexicon(with_blank: bool = False, corpus_key="train-small") -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_loquacious_extended_lexicon(corpus_key=corpus_key)
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon


@lru_cache
def get_vocab_datastream(with_blank: bool = False, corpus_key="train-small") -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param with_blank: datastream for CTC training
    """
    lexicon = get_loquacious_extended_lexicon(with_blank, corpus_key=corpus_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    name = "returnn_vocab_from_lexicon_with_blank" if with_blank else "returnn_vocab_from_lexicon"
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, name))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


@lru_cache
def get_tts_log_mel_datastream(loq_corpus_key, silence_preprocessed=False) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    Supports both centered and non-centered windowing, as we need non-centered windowing for RASR-compatible
    feature extraction, but centered windowing to support linear-features for the vocoder mel-to-linear training.

    :param center: use center for CTC and Attention alignment, but not for GMM for RASR compatibility
    """
    feature_options_center = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=None,
        sample_rate=16000,
        feature_options=DBMelFilterbankOptions(
            fmin=0,
            fmax=7600,
            min_amp=1e-10,
            center=True,
        )
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options_center
    )

    _, ls100_ogg_zip = get_tts_bliss_and_zip(loq_corpus_key=loq_corpus_key, silence_preprocessed=silence_preprocessed)
    train_segments, _ = get_loquacious_tts_segments(loq_corpus_key=loq_corpus_key)

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        [ls100_ogg_zip],
        segment_file=train_segments,
        use_scalar_only=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        alias_path=DATA_PREFIX + f"{loq_corpus_key}/",
    )
    return audio_datastream

@dataclass(frozen=True)
class FeatureModelTrainingDatasets(TrainingDatasets):
    """
    Dataclass for Alignment Datasets
    """

    # train: Dataset
    # cv: Dataset
    joint: Dataset
    # datastreams: Dict[str, Datastream]


def build_durationtts_training_dataset(
        duration_hdf,
        loq_corpus_key="train.small",
        partition_epoch=1,
        dynamic_speaker_embeddings: Optional[tk.Path] = None,
        dynamic_speaker_embedding_size: Optional[Union[int, tk.Variable]] = None,
    ) -> FeatureModelTrainingDatasets:
    """

    :param duration_hdf:
    :param ls_corpus_key:
    :param partition_epoch:
    :return:
    """
    bliss_dataset, zip_dataset = get_tts_bliss_and_zip(loq_corpus_key=loq_corpus_key, silence_preprocessed=False)

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_loquacious_tts_segments(loq_corpus_key=loq_corpus_key)

    vocab_datastream = get_vocab_datastream(with_blank=False, corpus_key=loq_corpus_key)
    audio_datastream = get_audio_raw_datastream()

    # we currently assume that train and cv share the same corpus file
    if dynamic_speaker_embeddings is None:
        # assert False, "Loquacious only supports dynamic embeddings"
        # we allow this now for vocoder training
        joint_speaker_dataset = None
    else:
        assert dynamic_speaker_embedding_size is not None
        joint_speaker_dataset = HDFDataset(
            files=[dynamic_speaker_embeddings],
        )
        speaker_datastream = FeatureDatastream(
            available_for_inference=True,
            feature_size=dynamic_speaker_embedding_size
        )


    duration_dataset = HDFDataset(
        files=[duration_hdf]
    )

    # ----- Ogg and Meta datasets

    train_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=partition_epoch,
        seq_ordering="laplace:.1000"
    )
    train_dataset = make_tts_meta_dataset(train_ogg_dataset, joint_speaker_dataset, duration_dataset=duration_dataset)

    cv_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_tts_meta_dataset(cv_ogg_dataset, joint_speaker_dataset, duration_dataset=duration_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_tts_meta_dataset(devtrain_zip_dataset, joint_speaker_dataset, duration_dataset=duration_dataset)

    joint_ogg_zip = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = make_tts_meta_dataset(joint_ogg_zip, joint_speaker_dataset)

    # ----- final outputs

    datastreams = {
        "raw_audio": audio_datastream,
        "phonemes": vocab_datastream,
        "speaker_embeddings": speaker_datastream,
    }

    durationtts_datasets = FeatureModelTrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
        prior=None,
    )

    return durationtts_datasets


@dataclass
class GeneratingDataset():
    split_datasets: List[Dataset]
    datastreams: Dict[str, Datastream]

def build_dynamic_speakers_generating_dataset(
        text_bliss: tk.Path,
        speaker_embedding_hdf: tk.Path,
        speaker_embedding_size: int,
        num_splits: int,
        distribute_speakers: bool = True,
        loq_corpus_key: str = "train.small",
        seed=None,
):
    """

    :param text_bliss:  corpus used for generation, needs to be processed for TTS
    :param num_splits: split via segments for parallel generation
    :param ls_corpus_key: base corpus to take speakers from
    :param randomize_speaker: Assign shuffled speakers from refrence corpus
    :return:
    """
    vocab_datastream = get_vocab_datastream(with_blank=True, corpus_key=loq_corpus_key)
    from i6_experiments.users.rossenbach.tts.speaker_embedding import DistributeDynamicSpeakerEmbeddingsJob

    if distribute_speakers:
        distributed_speaker_embeddings = DistributeDynamicSpeakerEmbeddingsJob(
            bliss_corpus=text_bliss,
            speaker_embedding_hdf=speaker_embedding_hdf,
            options=None
        ).out
    else:
        distributed_speaker_embeddings = speaker_embedding_hdf


    segments_dict = SegmentCorpusJob(text_bliss, num_segments=num_splits).out_single_segment_files

    text_only_zip = BlissToOggZipJob(
        bliss_corpus=text_bliss,
        no_audio=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    ).out_ogg_zip

    speaker_dataset = HDFDataset(
        files=[distributed_speaker_embeddings]
    )
    speaker_datastream = FeatureDatastream(
        available_for_inference=True,
        feature_size=speaker_embedding_size,
    )

    # ----- Ogg and Meta datasets

    datasets = []
    for i in range(num_splits):
        train_ogg_dataset = OggZipDataset(
            files=text_only_zip,
            audio_options=None,
            target_options=vocab_datastream.as_returnn_targets_opts(),
            segment_file=segments_dict[i+1],  # bullshit counting from 1 ...
            partition_epoch=1,
            seq_ordering="sorted_reverse"
        )
        train_dataset = make_tts_meta_dataset(train_ogg_dataset, speaker_dataset, include_speakers=True)
        datasets.append(train_dataset)

    datastreams = {
        "phonemes": vocab_datastream,
        "speaker_embeddings": speaker_datastream,
    }

    generating_dataset = GeneratingDataset(
        split_datasets=datasets,
        datastreams=datastreams,
    )

    return generating_dataset


