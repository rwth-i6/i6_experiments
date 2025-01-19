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
from i6_core.audio import BlissChangeEncodingJob
import i6_core.meta as meta
from i6_core.meta import CorpusObject

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

prepath_data = "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export"

dev_other_noise07 = tk.Path(("/").join([prepath_data, "dev-other_sequiturg2p_glowtts460_noise07.xml.gz"]),cached=True, hash_overwrite="GLOWTTS_V1_DEV_07")
dev_other_noise03 = tk.Path(("/").join([prepath_data, "dev-other_sequiturg2p_glowtts460_noise03.xml.gz"]),cached=True, hash_overwrite="GLOWTTS_V1_DEV_03")
dev_other_noise055 = tk.Path(("/").join([prepath_data, "dev-other_sequiturg2p_glowtts460_noise055.xml.gz"]),cached=True, hash_overwrite="GLOWTTS_V1_DEV_055")
dev_other_noise0551 = tk.Path(("/").join([prepath_data, "dev-other_sequiturg2p_glowtts460_noise055_seed1.xml.gz"]),cached=True, hash_overwrite="GLOWTTS_V1_DEV_0551")
dev_other_noise0552 = tk.Path(("/").join([prepath_data, "dev-other_sequiturg2p_glowtts460_noise055_seed2.xml.gz"]),cached=True, hash_overwrite="GLOWTTS_V1_DEV_0552")

TTS_DEVOTHER = {
    0.3: dev_other_noise03, 0.7: dev_other_noise07, 0.55: dev_other_noise055, 0.551: dev_other_noise0551, 0.552: dev_other_noise0552,
}



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


def get_eval_corpus_object_dict(ogg_corpus: CorpusObject):
    conversion_job = BlissChangeEncodingJob(
        corpus_file=ogg_corpus, output_format="wav", sample_rate=16000
    )
    crp_obj = meta.CorpusObject()
    crp_obj.corpus_file = conversion_job.out_corpus
    crp_obj.audio_dir = conversion_job.out_audio_folder
    crp_obj.audio_format = "wav"
    crp_obj.duration = 1.0

    return crp_obj


def get_corpus_data_inputs(
    corpus_key: str, noise: float = 0.7, use_g2p_training: bool = True, use_stress_marker: bool = False
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

    assert noise in TTS_DEVOTHER, f"update the TTS_DEVOTHER with {noise} path"
    corpus_object_dict["dev-other"] = get_eval_corpus_object_dict(TTS_DEVOTHER[noise])

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
