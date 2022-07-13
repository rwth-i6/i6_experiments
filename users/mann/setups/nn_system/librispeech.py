from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import librispeech
from i6_experiments.users.mann.setups.legacy_corpus import librispeech as legacy
from recipe.i6_core import features

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput

PREFIX_PATH1K = "/work/asr3/luescher/setups-data/librispeech/best-model/960h_2019-04-10/"
PREFIX_PATH = "/work/asr3/luescher/setups-data/librispeech/best-model/100h_2019-04-10/"
default_feature_paths = {
    "train-clean-100": PREFIX_PATH + "FeatureExtraction.Gammatone.tp4cEAa0YLIP/output/gt.cache.",
    'dev-clean' : "/u/michel/setups/librispeech/work/features/extraction/FeatureExtraction.Gammatone.DA0TtL8MbCKI/output/gt.cache.",
    "test-clean" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.INa6z5A4JvZ5/output/gt.cache.",
    "train-other-960": PREFIX_PATH1K + "FeatureExtraction.Gammatone.de79otVcMWSK/output/gt.cache.",
    "dev-other" : PREFIX_PATH1K + "FeatureExtraction.Gammatone.qrINHi3yh3GH/output/gt.cache.",
    "test-other" : "/u/luescher/setups/librispeech/2018-12-18--960h/work/features/extraction/FeatureExtraction.Gammatone.qqN3kYqQ6QHF/output/gt.cache.",
}
default_alignment_file = Path(PREFIX_PATH + "AlignmentJob.Mg44tFDRPnuh/output/alignment.cache.bundle", cached=True)

def get_librispeech_system():
    """Under construction. Do not use for training yet!"""
    corpus_object_dict = librispeech.get_corpus_object_dict()
    lm_dict = librispeech.get_arpa_lm_dict()
    lexicon_dict = librispeech.get_g2p_augmented_bliss_lexicon_dict()

    lbs_system = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

    rasr_data_input = RasrDataInput(
        corpus_object_dict["train-clean-100"],
        {"filename": lexicon_dict["train-clean-100"], "normalize_pronunciation": False}
    )

    lbs_system.add_corpus("train", rasr_data_input, add_lm=False)

    corpus = "train"
    feature_flow = "gt"
    lbs_system.feature_flows[corpus][feature_flow] = features.basic_cache_flow(Path(default_feature_paths["train-clean-100"] + "bundle", cached=True))

    return lbs_system


def get_legacy_librispeech_system():
    corpora = legacy.LibriSpeechCorpora()
    lm_dict = librispeech.get_arpa_lm_dict()
    lexicon_dict = librispeech.get_g2p_augmented_bliss_lexicon_dict()
    lbs_system = LibriNNSystem(epochs=[12, 24, 32, 48, 80, 160], num_input=50)

    rasr_data_input = RasrDataInput(
        corpora.corpora["train"]["train-clean-100"],
        {"filename": lexicon_dict["train-clean-100"], "normalize_pronunciation": False}
    )
    lbs_system.add_corpus("train", rasr_data_input, add_lm=False)

    corpus = "train"
    feature_flow = "gt"
    lbs_system.feature_flows[corpus][feature_flow] = features.basic_cache_flow(Path(default_feature_paths["train-clean-100"] + "bundle", cached=True))

    lbs_system.alignments["train"]["init_align"] = default_alignment_file
    lbs_system._init_am()
    del lbs_system.crp["train"].acoustic_model_config.tdp
    lbs_system.init_nn(**lbs_system.init_nn_args)
    return lbs_system