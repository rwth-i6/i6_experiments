from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper
from i6_core.corpus.convert import CorpusToTextDictJob
from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from i6_experiments.common.datasets.librispeech import (
    get_bliss_lexicon,
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_corpus_dict,
    get_subword_nmt_bpe,
    get_ogg_zip_dict,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
    AudioRawDatastream,
    ReturnnAudioRawOptions,
    AudioFeatureDatastream,
    ReturnnAudioFeatureOptions,
    FeatureType,
    DBMelFilterbankOptions,
)
from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments

from i6_experiments.users.gruev.lexicon.bpe_lexicon import CreateBPELexiconJob
from i6_experiments.users.gruev.pytorch.default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

from sisyphus import tk


# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset  # GenericDataset
    cv: Dataset  # GenericDataset
    devtrain: Dataset  # GenericDataset
    datastreams: Dict[str, Datastream]


@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    seq_ordering: str
    epoch_wise_filters: Optional[List[EpochWiseFilter]] = None


# --------------------------- Lexicon helper functions  -----------------------------------

subword_nmt_repo = tk.Path(
    "/u/atanas.gruev/setups/librispeech/2023-02-23--conformer-ctc/work"
    "/i6_core/tools/git/CloneGitRepositoryJob.rEnqw3X2cgyB/output/subword-nmt"
)


@lru_cache
def get_librispeech_lexicon(
    corpus_key="train-other-960", use_stress_marker=False, add_unknown_phoneme_and_mapping=True,
) -> tk.Path:
    return get_g2p_augmented_bliss_lexicon_dict(
        use_stress_marker=use_stress_marker, add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )[corpus_key]


def get_librispeech_bpe_lexicon(
    corpus_key="train-other-960", use_stress_marker=False, add_unknown_phoneme_and_mapping=True,
) -> tk.Path:
    bliss_lexicon = get_g2p_augmented_bliss_lexicon_dict(
        use_stress_marker=use_stress_marker, add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
    )[corpus_key]

    bpe_settings = get_subword_nmt_bpe(corpus_key=corpus_key, bpe_size=5000, unk_label="[UNK]")
    bliss_lexicon = CreateBPELexiconJob(
        base_lexicon_path=bliss_lexicon.get_path(),
        subword_nmt_repo=subword_nmt_repo,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        unk_label="[UNKNOWN]",
    ).out_lexicon

    return bliss_lexicon


def get_text_lexicon(corpus_key="train-other-960") -> tk.Path:
    bliss_lex = get_librispeech_bpe_lexicon(corpus_key)
    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


# --------------------------- Helper functions  -----------------------------------


@lru_cache()
def get_bpe_datastream(librispeech_key: str, bpe_size: int, is_recog: bool, seq_postfix=0,) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="[UNK]")
    bpe_targets = BpeDatastream(
        available_for_inference=False, bpe_settings=bpe_settings, seq_postfix=seq_postfix, use_unk_label=is_recog,
    )
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream(preemphasis: Optional[float] = None):
    """
    :param preemphasis: set the pre-emphasis filter factor
    :return:
    """
    audio_datastream = AudioRawDatastream(
        available_for_inference=True, options=ReturnnAudioRawOptions(peak_normalization=True, preemphasis=preemphasis),
    )
    return audio_datastream


@lru_cache()
def get_audio_datastream(
    statistics_ogg_zips: List[tk.Path], returnn_python_exe: tk.Path, returnn_root: tk.Path, alias_path: str,
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
    # default: mfcc-80-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=0.97,
        feature_options=DBMelFilterbankOptions(fmin=60, fmax=7600, min_amp=1e-10, center=True,),
    )
    audio_datastream = AudioFeatureDatastream(available_for_inference=False, options=feature_options)

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        statistics_ogg_zips,
        use_scalar_only=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=alias_path,
    )
    return audio_datastream


# --------------------------- Dataset functions  -----------------------------------


def build_training_datasets(
    librispeech_key: str,
    bpe_size: int,
    settings: TrainingDatasetSettings,
    preemphasis: Optional[float] = None,
    bpe_datastream_seq_postfix: Optional[int] = 0,
) -> TrainingDatasets:
    """
    :param settings:
    :param output_path:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    train_bpe_datastream = get_bpe_datastream(
        librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=False, seq_postfix=bpe_datastream_seq_postfix,
    )
    audio_datastream = get_audio_raw_datastream(preemphasis)

    datastreams = {
        "audio_features": audio_datastream,
        "bpe_labels": train_bpe_datastream,
    }

    data_map = {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    }

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
            data_map=data_map, datasets={"zip_dataset": dataset}, seq_order_control_dataset="zip_dataset",
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
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

    return TrainingDatasets(train=train_dataset, cv=cv_dataset, devtrain=devtrain_dataset, datastreams=datastreams,)


@lru_cache()
def build_test_dataset(
    librispeech_key: str,
    dataset_key: str,
    bpe_size: int,
    preemphasis: Optional[float] = None,
    bpe_datastream_seq_postfix: Optional[int] = 0,
    segment_file: Optional[tk.Path] = None,
):
    """
    :param dataset_key:
    :param bpe_size:
    :param preemphasis:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key], segment_file=segment_file).out_dictionary

    train_bpe_datastream = get_bpe_datastream(
        librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=True, seq_postfix=bpe_datastream_seq_postfix,
    )
    audio_datastream = get_audio_raw_datastream(preemphasis)

    data_map = {
        "audio_features": ("zip_dataset", "data"),
        "bpe_labels": ("zip_dataset", "classes"),
    }

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        segment_file=segment_file,
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset",
    )

    return test_dataset, test_reference_dict_file, bliss_dict[dataset_key]
