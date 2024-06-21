__all__ = [
    "get_init_args",
    "get_corpus_data_inputs",
    "get_final_output",
]

from typing import Dict
from collections import defaultdict

#----------recipes--------------------
from i6_core.features.filterbank import filter_width_from_channels

from i6_experiments.common.baselines.librispeech.default_tools import SCTK_BINARY_PATH
from i6_experiments.common.baselines.librispeech.data import CorpusData
import i6_experiments.common.datasets.tedlium2_v2 as ted_dataset
import i6_experiments.common.setups.rasr as rasr_util
from i6_experiments.common.setups.rasr.config.lex_config import (
    LexiconRasrConfig,
)
from i6_experiments.common.setups.rasr.config.lm_config import ArpaLmRasrConfig

#TED specific
from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2_v2.corpus import get_corpus_object_dict
from i6_experiments.common.datasets.tedlium2_v2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

from i6_experiments.users.raissi.setups.common.data.pipeline_helpers import (
    InputKey
)

def get_init_args():
    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global",
        "nonword_phones": "",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            6.0,
        ),  # only used when tying_type = global-and-nonword
    }

    costa_args = {"eval_recordings": True, "eval_lm": False}

    feature_extraction_args = {
        "fb": {
            "filterbank_options": {
                "warping_function": "mel",
                "filter_width": filter_width_from_channels(channels=80, warping_function="mel", f_max=8000),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": False,
                },
                "fft_options": None,
                "add_features_output": True,
                "apply_log": True,
                "add_epsilon": True,
            }
        }
    }

    scorer_args = {"sctk_binary_path": SCTK_BINARY_PATH}

    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer_args=scorer_args,
    )



def get_corpus_data_inputs(add_unknown_phoneme_and_mapping: bool = True) -> Dict[str, Dict[str, rasr_util.RasrDataInput]]:

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    train_lexicon = LexiconRasrConfig(
        get_g2p_augmented_bliss_lexicon(
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping, output_prefix="lexicon"
        ),
        False,
    )

    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    comb_lm = ArpaLmRasrConfig(lm_path=lm.ngram_lm)

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs["train"] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict["train"],
            lexicon=train_lexicon.get_dict(),
            concurrent=CONCURRENT["train"],
            lm=None,
        )
    dev_data_inputs["dev"] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict["dev"],
        lexicon=train_lexicon.get_dict(),
        concurrent=CONCURRENT["dev"],
        lm=comb_lm.get_dict(),
    )
    test_data_inputs["test"] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict["test"],
        lexicon=train_lexicon.get_dict(),
        concurrent=CONCURRENT["test"],
        lm=comb_lm.get_dict(),
    )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )




# -------------------- helpers --------------------
def get_final_output(name=InputKey.BASE):
    output_args = rasr_util.OutputArgs(name)

    output_args.define_corpus_type("train", "train")
    output_args.define_corpus_type("dev", "dev")
    output_args.define_corpus_type("test", "test")

    output_args.add_feature_to_extract("fb")

    return output_args
