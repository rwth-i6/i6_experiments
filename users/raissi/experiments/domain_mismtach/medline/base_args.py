__all__ = [
    "get_init_args",
    "get_final_output",
]

from dataclasses import dataclass
from typing import Dict, Optional

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------
import i6_core.corpus as corpus_recipes
import i6_core.meta as meta
from i6_core.audio import BlissChangeEncodingJob

from i6_experiments.common.baselines.librispeech.data import CorpusData
from i6_experiments.common.datasets.librispeech import (
    get_corpus_object_dict,
    get_arpa_lm_dict,
    get_bliss_lexicon,
    get_g2p_augmented_bliss_lexicon_dict,
    constants,
)

import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.common.setups.rasr import (
    RasrDataInput,
    RasrInitArgs,
)

from i6_experiments.users.raissi.args.rasr.features.init_args import get_feature_extraction_args_16kHz
from i6_experiments.users.raissi.args.rasr.am.init_args import get_init_am_args
from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import InputKey

@dataclass
class DATASET:
    lexicon: tk.Path
    corpus: tk.Path
    description: Optional[str]
    lm: Optional[tk.Path] = None



prepath_data = "/work/asr3/raissi/data/domain_mismatch/medline"



MEDLINE_V1_DEV_DATA = DATASET(
    lexicon=tk.Path(f"{prepath_data}/lexicon/v1/oov.lexicon.gz", cached=True, hash_overwrite="v1_nick"),
    corpus=tk.Path(f"{prepath_data}/corpus/v1/corpus_ogg.xml.gz", cached=True, hash_overwrite="v1_nick"),
    lm=tk.Path(f"{prepath_data}/lm/v1/ufal_version1_lm1.gz", cached=True, hash_overwrite="v1_nick"),
    description="first quick dirty version just to gte the pipeline ready."
)


MEDLINE_CORPORA = ["dev"]
MEDLINE_DURATIONS = {"dev": 1.0}
MEDLINE_DEV_VERSIONS={
    1: MEDLINE_V1_DEV_DATA,
}

MEDLINE_TEST_VERSIONS={}


MEDLINE_DATA = {
    "dev": MEDLINE_DEV_VERSIONS,
}




"""
conversion_job = BlissChangeEncodingJob(
    corpus_file=CORPUS_V1_OGG, output_format="wav", sample_rate=16000
)

CORPUS_V1 = conversion_job.out_corpus
MEDLINE_AUDIO_PATH = conversion_job.out_audio_folder
"""


# -------------------- functions --------------------
def _get_bliss_corpus_dict(corpus, segment_mapping, compressed=True):
    corpus_files = {}

    for key in segment_mapping.keys():
        filter_job = corpus_recipes.FilterCorpusBySegmentsJob(
            bliss_corpus=corpus,
            segment_file=segment_mapping[key],
            compressed=compressed,
            invert_match=False
        )
        corpus_files[key] = filter_job.out_corpus

    return corpus_files

def _get_eval_corpus_object_dict(name: str, version: int=1, segment_mapping: tk.Path=None):
    """
    You can either have a segment list and divide a corpus into subcopora or you call this for a specific corpus
    """
    assert version in MEDLINE_DATA[name].keys()
    assert name in MEDLINE_DATA

    corpus = MEDLINE_DATA[name][version].corpus

    if segment_mapping is not None:
        corpora = _get_bliss_corpus_dict(
            corpus=corpus,
            compressed=True,
            segment_mapping=segment_mapping,

        )
    else:
        corpora = {name: corpus}



    corpus_object_dict = {}
    for k, v in corpora.items():
        conversion_job = BlissChangeEncodingJob(
            corpus_file=v, output_format="wav", sample_rate=16000
        )
        crp_obj = meta.CorpusObject()
        crp_obj.corpus_file = conversion_job.out_corpus
        crp_obj.audio_dir =  conversion_job.out_audio_folder
        crp_obj.audio_format = "wav"
        crp_obj.duration = MEDLINE_DURATIONS[k]
        corpus_object_dict[k] = crp_obj


    return corpus_object_dict


def get_corpus_data_inputs(
    corpus_key: str, version: int = 1, segment_mapping_domain:Dict = None, use_g2p_training: bool = True, use_stress_marker: bool = False
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
    corpus_object_dict_lbs = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")
    # Definition of the official 4-gram LM to be used as default LM
    lm_lbs = {
        "filename": get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }
    # This is the standard LibriSpeech lexicon
    lexicon_lbs = {
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
        train_lexicon = lexicon_lbs

    #domain dev_data
    if segment_mapping_domain is not None:

        corpus_object_dict_medline_all = _get_eval_corpus_object_dict(name="all", version=version, segment_mapping=segment_mapping_domain)
        corpus_object_dev = corpus_object_dict_medline_all["dev"]
        corpus_object_test = corpus_object_dict_medline_all["test"]

    else:
        corpus_object_dev = _get_eval_corpus_object_dict(name="dev", version=version)["dev"]

    oov_lexicon_medline = {
        "filename": MEDLINE_DATA["dev"][version].lexicon,
        "normalize_pronunciation": False,
    }

    lm_medline = {
        "filename": MEDLINE_DATA["dev"][version].lm,
        "type": "ARPA",
        "scale": 2.0,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}


    ##standard LBS 960h
    train_data_inputs[corpus_key] = RasrDataInput(
        corpus_object=corpus_object_dict_lbs[corpus_key],
        concurrent=constants.concurrent[corpus_key],
        lexicon=train_lexicon,
        lm=None,
    )
    for test_key in ["test-other"]:
        dev_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dict_lbs[test_key],
            concurrent=constants.concurrent[test_key],
            lexicon=lexicon_lbs,
            lm=lm_lbs,
        )

    for test_key in ["dev"]:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dev,
            concurrent=4,
            lexicon=oov_lexicon_medline,
            lm=lm_medline,
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

def get_final_output(name=InputKey.BASE):
    output_args = rasr_util.OutputArgs(name)

    output_args.define_corpus_type("train-other-960", "train")
    output_args.define_corpus_type("dev", "dev")
    output_args.define_corpus_type("test-other", "test")

    output_args.add_feature_to_extract("gt")

    return output_args
