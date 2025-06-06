__all__ = [
    "get_init_args",
    "get_data_inputs",
]

from typing import Dict, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr


from i6_experiments.common.baselines.librispeech.data import CorpusData
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
    get_arpa_lm_dict,
    get_bliss_lexicon,
    get_g2p_augmented_bliss_lexicon_dict,
    constants,
)
from i6_experiments.common.setups.rasr import (
    RasrDataInput,
    RasrInitArgs,
)

from i6_experiments.users.raissi.args.rasr.features.init_args import get_feature_extraction_args_16kHz
from i6_experiments.users.raissi.args.rasr.am.init_args import get_init_am_args


# -------------------- functions --------------------
def get_init_args(
    *,
    dc_detection: bool = False,
    am_extra_args: Optional[Dict] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_options_extra_args: Optional[Dict] = None,
):
    """
    :param dc_detection:
    :param am_extra_args:
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_options_extra_args:
    :return:
    """

    am_args = get_init_am_args()
    if am_extra_args is not None:
        am_args.update(am_extra_args)

    costa_args = {"eval_recordings": True, "eval_lm": False}

    feature_extraction_args = get_feature_extraction_args_16kHz(
        dc_detection=dc_detection,
        mfcc_cepstrum_options=mfcc_cepstrum_options,
        mfcc_args=mfcc_extra_args,
        gt_args=gt_options_extra_args,
    )
    return RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
    )


def get_corpus_data_inputs(
    corpus_key: str, use_g2p_training: bool = True, use_stress_marker: bool = False,
    audio_format: str = "wav"
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
    corpus_object_dict = get_corpus_object_dict(audio_format=audio_format, output_prefix="corpora")

    # Definition of the official 4-gram LM to be used as default LM
    lm = {
        "filename": get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    # This is the standard LibriSpeech lexicon
    lexicon = {
        "filename": get_bliss_lexicon(
            use_stress_marker=use_stress_marker,
            add_unknown_phoneme_and_mapping=True,
        ),
        "normalize_pronunciation": False,
    }

    # In case we train with a G2P-augmented lexicon we do not use the same lexicon for training and recognition.
    # The recognition Lexion is always without G2P to ensure better comparability
    if use_g2p_training:
        train_lexicon = {
            "filename": get_g2p_augmented_bliss_lexicon_dict(
                use_stress_marker=use_stress_marker,
                add_unknown_phoneme_and_mapping=True,
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
        concurrent=constants.concurrent[corpus_key],
        lexicon=train_lexicon,
        lm=None,
    )

    for dev_key in ["dev-clean", "dev-other"]:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=constants.concurrent[dev_key],
            lexicon=lexicon,
            lm=lm,
        )

    for test_key in ["test-clean", "test-other"]:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dict[test_key],
            concurrent=constants.concurrent[test_key],
            lexicon=lexicon,
            lm=lm,
        )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )


def get_number_of_segments():
    num_segments = constants.num_segments
    num_segments[f"train-other-960"] = 0
    for subset in ["clean-100", "clean-360", "other-500"]:
        num_segments[f"train-other-960"]+= num_segments[f"train-{subset}"]
    return num_segments
