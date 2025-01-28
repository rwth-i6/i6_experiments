__all__ = [
    "get_init_args",
    "get_final_output",
]

from dataclasses import dataclass
from typing import Dict, Optional, Union

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


PREPATH_ASR3 = "/work/asr3/raissi/data/domain_mismatch/mtg"
PREPATH_CORPORA = "/u/rossenbach/experiments/tts_decoder_asr/output/domain_test_tina_export"

mtg_noise055 = tk.Path(
    ("/").join([PREPATH_CORPORA, "MTG_trial3_dev_sequiturg2p_glowtts460_noise055.xml.gz"]),
    cached=True,
    hash_overwrite="GLOWTTS_TRIAL3_DEV_MTG_07",
)


#################

MTG_V1_DEV_DATA = {
    0.55: DATASET(
        lexicon_with_unk=tk.Path(f"{PREPATH_ASR3}/lexicon/v1/MTG_trial3.lsoverride.rasr_with_unk.xml.gz", cached=True, hash_overwrite="mtg_v1_unk"),
        lexicon_no_unk=tk.Path(
            f"{PREPATH_ASR3}/lexicon/v1/MTG_trial3.lsoverride.rasr_without_unk.xml.gz", cached=True, hash_overwrite="mtg_v1_nounk"
        ),
        corpus=mtg_noise055,
        lm=tk.Path(f"{PREPATH_ASR3}/lm/v1/MTG_trial3.lm.gz", cached=True, hash_overwrite="mtg_lm_v1"),
        description="first version with correct pronunciations",
    )
}
#################


MTG_CORPORA = ["dev"]
MTG_DURATIONS = {"dev": 1.0}
MTG_DEV_VERSIONS = {1: MTG_V1_DEV_DATA,
                        }

MTG_TEST_VERSIONS = {}


MTG_DATA = {
    "dev": MTG_DEV_VERSIONS,
}


# -------------------- functions --------------------
def _get_bliss_corpus_dict(corpus, segment_mapping, compressed=True):
    corpus_files = {}

    for key in segment_mapping.keys():
        filter_job = corpus_recipes.FilterCorpusBySegmentsJob(
            bliss_corpus=corpus, segment_file=segment_mapping[key], compressed=compressed, invert_match=False
        )
        corpus_files[key] = filter_job.out_corpus

    return corpus_files


def _get_eval_corpus_object_dict(name: str, version: Union[int, float] = 1, noise: float = 0.7, segment_mapping: tk.Path = None):
    """
    You can either have a segment list and divide a corpus into subcopora or you call this for a specific corpus
    """
    assert version in MTG_DATA[name].keys()
    assert name in MTG_DATA

    corpus = MTG_DATA[name][version][noise].corpus

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
        crp_obj.duration = MTG_DURATIONS[k]
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
        MTG_DATA["dev"][version][noise].lexicon_with_unk
        if add_unknown_for_medline_lex
        else MTG_DATA["dev"][version][noise].lexicon_no_unk
    )

    seed_lexicon = "seed_withunk.xml.gz" if add_unknown_for_medline_lex else "seed_nounk.xml.gz"
    seed_lexicon_path = tk.Path(("/").join([f"{PREPATH_ASR3}", f"lexicon/seed_lbs_lexicon_nolemmata/{seed_lexicon}"]), hash_overwrite=f"seed_{seed_lexicon}")
    med_lex = MergeLexiconJob([seed_lexicon_path, med_lex]).out_bliss_lexicon

    oov_lexicon_medline = {
        "filename": med_lex,
        "normalize_pronunciation": False,
    }

    lm_medline = {
        "filename": MTG_DATA["dev"][version][noise].lm,
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
