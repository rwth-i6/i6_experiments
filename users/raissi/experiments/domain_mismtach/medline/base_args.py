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
from i6_core.lexicon import MergeLexiconJob

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
    lexicon_with_unk: tk.Path
    lexicon_no_unk: tk.Path
    corpus: tk.Path
    description: Optional[str]
    lm: Optional[tk.Path] = None


PREPATH_ASR3 = "/work/asr3/raissi/data/domain_mismatch/medline"
PREPATH_CORPORA = "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export"

wmt22_medline_noise07 = tk.Path(
    ("/").join([PREPATH_CORPORA, "wmt22_medline_v1_sequiturg2p_glowtts460_noise07.xml.gz"]),
    cached=True,
    hash_overwrite="GLOWTTS_V1_DEV_MED_07",
)
wmt22_medline_noise03 = tk.Path(
    ("/").join([PREPATH_CORPORA, "wmt22_medline_v1_sequiturg2p_glowtts460_noise03.xml.gz"]),
    cached=True,
    hash_overwrite="GLOWTTS_V1_DEV_MED_03",
)
#################

MEDLINE_V1_DEV_DATA = {
    0.7: DATASET(
        lexicon_with_unk=tk.Path(f"{PREPATH_ASR3}/lexicon/v1/oov.lexicon.gz", cached=True, hash_overwrite="v1_nick"),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v1/ufal_librispeech_lexicon_rasr_without_unk.xml.gz", cached=True
        ),
        corpus=tk.Path(f"{PREPATH_ASR3}/corpus/v1/corpus_ogg.xml.gz", cached=True, hash_overwrite="v1_nick"),
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v1/ufal_version1_lm1.gz", cached=True, hash_overwrite="v1_nick"),
        description="first quick dirty version just to gte the pipeline ready. lexicon have words that appear 2 times or more"
        "mix between LBs and medline",
    )
}
#################


MEDLINE_V11_DEV_DATA = {
    0.7: DATASET(
        lexicon_with_unk=tk.Path(f"{PREPATH_ASR3}/lexicon/v1/oov.lexicon.gz", cached=True, hash_overwrite="v1_nick"),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v1/ufal_librispeech_lexicon_rasr_without_unk.xml.gz", cached=True
        ),
        corpus=wmt22_medline_noise07,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v1/ufal_version1_lm1.gz", cached=True, hash_overwrite="v1_nick"),
        description="based on version 1 using the correct data input.",
    ),
    0.3: DATASET(
        lexicon_with_unk=tk.Path(f"{PREPATH_ASR3}/lexicon/v1/oov.lexicon.gz", cached=True, hash_overwrite="v1_nick"),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v1/ufal_librispeech_lexicon_rasr_without_unk.xml.gz", cached=True
        ),
        corpus=wmt22_medline_noise03,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v1/ufal_version1_lm1.gz", cached=True, hash_overwrite="v1_nick"),
        description="based on version 1 using the correct data input.",
    ),
}
#################

MEDLINE_V21_DEV_DATA = {
    0.7: DATASET(
        lexicon_with_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/medline+LBS/ufal_v1_mixlex_v2.rasr_with_unk.xml.gz",
            cached=True,
            hash_overwrite="v21_unk",
        ),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/medline+LBS/ufal_v1_mixlex_v2.rasr_without_unk.xml.gz",
            cached=True,
            hash_overwrite="v21_nounk",
        ),
        corpus=wmt22_medline_noise07,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v2/medline+LBS/ufal_v1_mixlex_v2.lm.gz", cached=True, hash_overwrite="v21_lm"),
        description="lexicon uses both LBS and medline data with words repeating 3 or more",
    ),
    0.3: DATASET(
        lexicon_with_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/medline+LBS/ufal_v1_mixlex_v2.rasr_with_unk.xml.gz",
            cached=True,
            hash_overwrite="v21_unk",
        ),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/medline+LBS/ufal_v1_mixlex_v2.rasr_without_unk.xml.gz",
            cached=True,
            hash_overwrite="v21_nounk",
        ),
        corpus=wmt22_medline_noise03,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v2/medline+LBS/ufal_v1_mixlex_v2.lm.gz", cached=True, hash_overwrite="v21_lm"),
        description="lexicon uses both LBS and medline data with words repeating 3 or more",
    ),
}

MEDLINE_V22_DEV_DATA = {
    0.7: DATASET(
        lexicon_with_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/only_medline/ufal_v1_3more_only.rasr_with_unk.xml.gz",
            cached=True,
            hash_overwrite="v22_unk",
        ),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/only_medline/ufal_v1_3more_only.rasr_without_unk.xml.gz",
            cached=True,
            hash_overwrite="v22_nounk",
        ),
        corpus=wmt22_medline_noise07,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v2/only_medline/ufal_v1_lm_3more.gz", cached=True, hash_overwrite="v22_lm"),
        description="lexicon uses only medline data with words repeating 3 or more",
    ),
    0.3: DATASET(
        lexicon_with_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/only_medline/ufal_v1_3more_only.rasr_with_unk.xml.gz",
            cached=True,
            hash_overwrite="v22_unk",
        ),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v2/only_medline/ufal_v1_3more_only.rasr_without_unk.xml.gz",
            cached=True,
            hash_overwrite="v22_nounk",
        ),
        corpus=wmt22_medline_noise03,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v2/only_medline/ufal_v1_lm_3more.gz", cached=True, hash_overwrite="v22_lm"),
        description="lexicon uses only medline data with words repeating 3 or more",
    ),
}


MEDLINE_CORPORA = ["dev"]
MEDLINE_DURATIONS = {"dev": 1.0}
MEDLINE_DEV_VERSIONS = {1: MEDLINE_V1_DEV_DATA, 1.1: MEDLINE_V11_DEV_DATA, 2.1: MEDLINE_V21_DEV_DATA, 2.2: MEDLINE_V22_DEV_DATA}

MEDLINE_TEST_VERSIONS = {}


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
            bliss_corpus=corpus, segment_file=segment_mapping[key], compressed=compressed, invert_match=False
        )
        corpus_files[key] = filter_job.out_corpus

    return corpus_files


def _get_eval_corpus_object_dict(name: str, version: int = 1, noise: float = 0.7, segment_mapping: tk.Path = None):
    """
    You can either have a segment list and divide a corpus into subcopora or you call this for a specific corpus
    """
    assert version in MEDLINE_DATA[name].keys()
    assert name in MEDLINE_DATA

    corpus = MEDLINE_DATA[name][version][noise].corpus

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
        conversion_job = BlissChangeEncodingJob(corpus_file=v, output_format="wav", sample_rate=16000)
        crp_obj = meta.CorpusObject()
        crp_obj.corpus_file = conversion_job.out_corpus
        crp_obj.audio_dir = conversion_job.out_audio_folder
        crp_obj.audio_format = "wav"
        crp_obj.duration = MEDLINE_DURATIONS[k]
        corpus_object_dict[k] = crp_obj

    return corpus_object_dict


def get_corpus_data_inputs(
    corpus_key: str,
    version: float = 1,
    noise: float = 0.7,
    segment_mapping_domain: Dict = None,
    add_unknown_for_medline_lex: bool = True,
    use_g2p_training: bool = True,
    use_stress_marker: bool = False,
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

    # domain dev_data
    if segment_mapping_domain is not None:

        corpus_object_dict_medline_all = _get_eval_corpus_object_dict(
            name="all", version=version, segment_mapping=segment_mapping_domain
        )
        corpus_object_dev = corpus_object_dict_medline_all["dev"]
        corpus_object_test = corpus_object_dict_medline_all["test"]

    else:
        corpus_object_dev = _get_eval_corpus_object_dict(name="dev", version=version, noise=noise)["dev"]

    med_lex = (
        MEDLINE_DATA["dev"][version][noise].lexicon_with_unk
        if add_unknown_for_medline_lex
        else MEDLINE_DATA["dev"][version][noise].lexicon_no_unk
    )
    if version > 1:
        seed_lexicon = "seed_withunk.xml.gz" if add_unknown_for_medline_lex else "seed_nounk.xml.gz"
        seed_lexicon_path = tk.Path(("/").join([f"{PREPATH_ASR3}", f"lexicon/seed_lbs_lexicon_nolemmata/{seed_lexicon}"]), hash_overwrite=f"seed_{seed_lexicon}")
        med_lex = MergeLexiconJob([seed_lexicon_path, med_lex]).out_bliss_lexicon

    oov_lexicon_medline = {
        "filename": med_lex,
        "normalize_pronunciation": False,
    }

    lm_medline = {
        "filename": MEDLINE_DATA["dev"][version][noise].lm,
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
    for dev_key in ["dev"]:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dev,
            concurrent=12,
            lexicon=oov_lexicon_medline,
            lm=lm_medline,
        )
    for test_key in ["test-other"]:
        test_data_inputs[test_key] = RasrDataInput(
            corpus_object=corpus_object_dict_lbs[test_key],
            concurrent=constants.concurrent[test_key],
            lexicon=lexicon_lbs,
            lm=lm_lbs,
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
        num_segments[f"train-other-960"] += num_segments[f"train-{subset}"]
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
