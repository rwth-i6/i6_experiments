from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import librispeech
from i6_experiments.users.mann.setups.legacy_corpus import librispeech as legacy
from recipe.i6_core import features

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput

SCTK_PATH = Path('/u/beck/programs/sctk-2.4.0/bin/')

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
default_tf_native_ops = Path('/work/tools/asr/returnn_native_ops/20190919_0e23bcd20/generic/NativeLstm2/NativeLstm2.so')
default_mixture_path = Path(PREFIX_PATH + "EstimateMixturesJob.accumulate.dctnjFBP8hos/output/am.mix", cached=True)
default_mono_mixture_path = Path("/u/michel/setups/librispeech/work/mm/mixtures/EstimateMixturesJob.accumulate.rNKsxWShoABt/output/am.mix", cached=True)

init_nn_args = {
    'name': 'crnn',
    'corpus': 'train',
    'dev_size': 0.001,
    'bad_segments': None,
    'dump': True}

default_nn_training_args = {
    'feature_corpus': 'train',
    'alignment': ('train', 'init_align', -1),
    'num_classes': lambda s: {
        'monophone':   211,
        'cart'     : 12001
    }[s.csp['base'].acoustic_model_config.state_tying.type],
    'num_epochs': 160,
    'train_corpus': 'crnn_train',
    'dev_corpus'  : 'crnn_dev',
    'partition_epochs': {'train': 8, 'dev' : 1},
    'save_interval': 4,
    'time_rqmt': 120,
    'mem_rqmt' : 24,
    'log_verbosity': 5,
    'use_python_control': True,
    'feature_flow': 'gt'}

default_nn_dump_args = {
    'feature_corpus': 'train',
    'alignment': ('train', 'init_align', -1),
    'num_classes': 12001,
    'num_epochs': 1,
    'train_corpus': 'crnn_train_dump',
    'dev_corpus'  : 'crnn_dev',
    'partition_epochs': {'train': 1, 'dev' : 1},
    'save_interval': 4,
    'time_rqmt': 1,
    'mem_rqmt' : 4,
    'use_python_control': True,
    'feature_flow': 'gt'}

default_scorer_args = {
    # 'prior_mixtures': ('train', 'init_mixture'),
    'prior_mixtures': None,
    'prior_scale': 0.70,
    'feature_dimension': 50}

default_recognition_args = {
    'corpus': 'dev',
    'flow': 'gt',
    'pronunciation_scale': 3.0,
    'lm_scale': 5.0,
    'search_parameters': {'beam-pruning': 18.0, # TODO: 15
                        'beam-pruning-limit': 100000,
                        'word-end-pruning': 0.5,
                        'word-end-pruning-limit': 10000},
    'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                'best_path_algo': 'bellman-ford' },
    'rtf': 50}

default_compile_tf_bins = dict(
    crnn_python_exe  = '/u/michel/py2-theano/bin/python2.7',
    crnn_root        = '/u/michel/git_projects/returnn'
)
default_compile_tf_bins = {}

TOTAL_FRAMES = 36107903

# import paths
default_reduced_segment_path = '/work/asr3/michel/mann/misc/tmp/reduced.train.segments'

def get_librispeech_system():
    """Under construction. Do not use for training yet!"""
    corpus_object_dict = librispeech.get_corpus_object_dict()
    lm_dict = librispeech.get_arpa_lm_dict()
    lexicon_dict = librispeech.get_g2p_augmented_bliss_lexicon_dict()

    lbs_system = LibriNNSystem(
        epochs=[12, 24, 32, 48, 80, 160],
        num_input=50,
    )

    rasr_data_input = RasrDataInput(
        corpus_object_dict["train-clean-100"],
        {"filename": lexicon_dict["train-clean-100"], "normalize_pronunciation": False},
        concurrent=librispeech.constants.concurrent["train-clean-100"]
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
    lbs_system = NNSystem(
        epochs=[12, 24, 32, 48, 80, 160],
        num_input=50,
        native_ops_path=default_tf_native_ops
    )

    rasr_data_input = RasrDataInput(
        corpora.corpora["train"]["train-clean-100"],
        {"filename": lexicon_dict["train-clean-100"], "normalize_pronunciation": False},
        concurrent=librispeech.constants.concurrent["train-clean-100"]
    )
    lbs_system.add_corpus("train", rasr_data_input, add_lm=False)

    rasr_data_input = RasrDataInput(
        corpora.corpora["dev"]["dev-clean"],
        lexicon={
            "filename": lexicon_dict["train-clean-100"],
            "normalize_pronunciation": False},
        lm={
            "filename": lm_dict["4gram"],
            "type": "ARPA",
            "scale": 5.0},
        concurrent=librispeech.constants.concurrent["dev-clean"]
    )
    lbs_system.add_corpus("dev", rasr_data_input, add_lm=True)

    corpus = "train"
    feature_flow = "gt"
    lbs_system.feature_flows[corpus][feature_flow] = features.basic_cache_flow(Path(default_feature_paths["train-clean-100"] + "bundle", cached=True))
    lbs_system.feature_flows["dev"][feature_flow] = features.basic_cache_flow(Path(default_feature_paths["dev-clean"] + "bundle", cached=True))
    lbs_system.alignments["train"]["init_align"] = default_alignment_file
    lbs_system._init_am()
    del lbs_system.crp["train"].acoustic_model_config.tdp
    lbs_system.default_nn_training_args = default_nn_training_args
    lbs_system.default_recognition_args = default_recognition_args
    lbs_system.default_scorer_args      = default_scorer_args
    lbs_system.default_reduced_segment_path = default_reduced_segment_path
    lbs_system.num_frames = {"train": TOTAL_FRAMES}
    # lbs_system.mixtures["train"]["init_mixture"] = default_mono_mixture_path
    lbs_system.create_stm_from_corpus("dev")
    lbs_system.set_sclite_scorer("dev", sctk_binary_path=SCTK_PATH)
    lbs_system.init_nn(**init_nn_args)
    return lbs_system