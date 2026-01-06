"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from sisyphus import tk


from i6_core.meta.system import CorpusObject

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.loquacious.lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon_dict
from .common import get_dev_short_bliss
from i6_experiments.common.setups.rasr import RasrDataInput


durations = {
    "train.small": 250.0,
    "dev.all": 16.5,
    "dev.commonvoice": 5.0,
    "dev.librispeech": 5.0,
    "dev.voxpopuli": 5.0,
    "dev.yodas": 1.5,
    "test.all": 16.5,
    "test.commonvoice": 5.0,
    "test.librispeech": 5.0,
    "test.voxpopuli": 5.0,
    "test.yodas": 1.5,
    "dev.short": 6.0,
}


concurrent = {
    "train.small": 25,
    "dev.all": 15,
    "dev.commonvoice": 5,
    "dev.librispeech": 5,
    "dev.voxpopuli": 5,
    "dev.yodas": 2,
    "test.all": 15,
    "test.commonvoice": 5,
    "test.librispeech": 5,
    "test.voxpopuli": 5,
    "test.yodas": 2,
    "dev.short": 5,
}

@lru_cache()
def get_corpus_object_dict(audio_format="flac", output_prefix="datasets"):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a dict of CorpusObjects.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str output_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
        - 'train-clean-100'
        - 'train-clean-360'
        - 'train-clean-460'
        - 'train-other-500'
        - 'train-other-960'
    :rtype: dict[str, CorpusObject]
    """
    dev_short_bliss = get_dev_short_bliss()
    bliss_corpus_dict = {**get_bliss_corpus_dict(), "dev.short": dev_short_bliss}

    corpus_object_dict = {}

    for corpus_name, bliss_corpus in bliss_corpus_dict.items():
        if corpus_name in ["train.medium", "train.medium-wo-small"]:
            continue
        corpus_object = CorpusObject()
        corpus_object.corpus_file = bliss_corpus
        corpus_object.audio_format = "ogg"
        corpus_object.audio_dir = None
        corpus_object.duration = durations[corpus_name]

        corpus_object_dict[corpus_name] = corpus_object

    return corpus_object_dict


    

@dataclass()
class CorpusData:
    """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

    train_data: Dict[str, RasrDataInput]
    dev_data: Dict[str, RasrDataInput]
    test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs(
    corpus_key: str, use_g2p_training: bool = True, use_stress_marker: bool = False, variant: int = 1,
) -> CorpusData:
    """
    Create the corpus data for any LibriSpeech RASR setup

    :param corpus_key: which LibriSpeech subset to use e.g. train-other-960, refer to common/datasets/librispeech.py
    :param use_g2p_training: If true, uses Sequitur to generate full lexicon coverage for the training data
    :param use_stress_marker: If the phoneme representation should include the ARPA stress marker
        Sometimes this is also referred to as "unfolded" lexicon.
    :return: (train_data, dev_data, test_data)
    """

    # Dictionary containing all LibriSpeech CorpusObject entries
    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    # Definition of the official 4-gram LM to be used as default LM
    lm = {
        "filename": tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned.arpa.gz"),
        "type": "ARPA",
        "scale": 10,
    }

    # This is the standard LibriSpeech lexicon
    lexicon = {
        "filename": get_bliss_lexicon(
            use_stress_marker=use_stress_marker,
            add_unknown_phoneme_and_mapping=not use_g2p_training,
            variant=variant,
        ),
        "normalize_pronunciation": False,
    }

    # In case we train with a G2P-augmented lexicon we do not use the same lexicon for training and recognition.
    # The recognition Lexion is always without G2P to ensure better comparability
    if use_g2p_training:
        train_lexicon = {
            "filename": get_g2p_augmented_bliss_lexicon_dict(
                use_stress_marker=use_stress_marker,
                add_unknown_phoneme_and_mapping=False,
                variant=variant,
            )[corpus_key],
            "normalize_pronunciation": False,
        }
    else:
        train_lexicon = lexicon

    # Here we define all corpora that are used.
    # The training corpus is dynamic based on which subset we want to use,
    # but dev and test are fixed.
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[corpus_key] = RasrDataInput(
        corpus_object=corpus_object_dict[corpus_key],
        concurrent=concurrent[corpus_key],
        lexicon=train_lexicon,
        lm=None,
    )

    for dev_key in ["dev.short"]:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=concurrent[dev_key],
            lexicon=lexicon,
            lm=lm,
        )

    for test_key in ["test.librispeech"]:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dict[test_key],
            concurrent=concurrent[test_key],
            lexicon=lexicon,
            lm=lm,
        )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )