from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import BaseSystem, NNSystem, ExpConfig
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.common.datasets import switchboard
from i6_experiments.users.mann.setups.legacy_corpus import swb1 as legacy
from recipe.i6_core import features
from recipe.i6_core import returnn
from recipe.i6_core import meta

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput

SCTK_PATH = Path('/u/beck/programs/sctk-2.4.0/bin/')

init_nn_args = {
    'name': 'crnn',
    'corpus': 'train',
    'dev_size': 0.05,
    'alignment_logs': True,
}

default_nn_training_args = {
    'feature_corpus': 'train',
    'alignment': ('train', 'init_align', -1),
    'num_epochs': 320,
    'partition_epochs': {'train': 6, 'dev' : 1},
    'save_interval': 4,
    'num_classes': lambda s: s.num_classes(),
    'time_rqmt': 120,
    'mem_rqmt' : 12,
    'use_python_control': True,
    'feature_flow': 'gt'
}

default_scorer_args = {
    'prior_mixtures': ('train', 'init_mixture'),
    'prior_scale': 0.70,
    'feature_dimension': 40
}

default_recognition_args = {
    'corpus': 'dev',
    'flow': 'gt',
    'pronunciation_scale': 1.0,
    'lm_scale': 10.,
    'search_parameters': {
        'beam-pruning': 16.0,
        'beam-pruning-limit': 100000,
        'word-end-pruning': 0.5,
        'word-end-pruning-limit': 10000},
    'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                'best_path_algo': 'bellman-ford' },
    'rtf': 50
}

default_cart_lda_args = {
    'corpus': 'train',
    'initial_flow': 'gt',
    'context_flow': 'gt',
    'context_size':  15,
    'alignment': 'init_align',
    'num_dim': 40,
    'num_iter':  2,
    'eigenvalue_args': {},
    'generalized_eigenvalue_args': {'all': {'verification_tolerance': 1e14} }
}

# import paths
default_reduced_segment_path = '/u/mann/experiments/librispeech/recipe/setups/mann/librispeech/reduced.train.segments'
PREFIX_PATH                       = "/work/asr3/michel/setups-data/SWB_sis/"
PREFIX_PATH_asr4                  = "/work/asr4/michel/setups-data/SWB_sis/"
default_allophones_file      = PREFIX_PATH + "allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones"
default_alignment_file       = Path('/work/asr3/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.cache.bundle', cached=True)
extra_alignment_file         = Path('/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.cache.bundle', cached=True) # gmm
default_alignment_logs = ['/work/asr3/michel/setups-data/SWB_sis/' + \
    'mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.log.{id}.gz' \
        .format(id=id) for id in range(1, 201)]
extra_alignment_logs = [
    f'/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.log.{id}.gz'
    for id in range(1, 201)
]
default_cart_file            = Path(PREFIX_PATH + "cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz", cached=True)

default_mixture_path  = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.Fb561bWZLwiJ/output/am.mix",cached=True)
default_mono_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.m5wLIWW876pl/output/am.mix", cached=True)
default_feature_paths = {
    'train': PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.Jlfrg2riiRX3/output/gt.cache.bundle",
    'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.dVkMNkHYPXb4/output/gt.cache.bundle",
    'eval' : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.O4lUG0y7lrKt/output/gt.cache.bundle"
}

RETURNN_PYTHON_HOME = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1')
RETURNN_PYTHON_EXE = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8')

RETURNN_REPOSITORY_URL = 'https://github.com/rwth-i6/returnn.git'

RASR_BINARY_PATH = Path('/work/tools/asr/rasr/20220603_github_default/arch/linux-x86_64-standard')

from collections import namedtuple
Binaries = namedtuple('Binaries', ['returnn', 'native_lstm', 'rasr'])


def init_binaries():
    # clone returnn
    from i6_core.tools import CloneGitRepositoryJob
    returnn_root = CloneGitRepositoryJob(
        RETURNN_REPOSITORY_URL,
    ).out_repository

    # compile lstm ops
    native_lstm = returnn.CompileNativeOpJob(
        "NativeLstm2",
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        search_numpy_blas=True
    ).out_op

    # compile rasr
    from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
    rasr_binary_path = compile_rasr_binaries_i6mode()
    # rasr_binary_path = None
    return Binaries(returnn_root, native_lstm, rasr_binary_path)


def get_legacy_switchboard_system():
    """Returns the an NNSystem for the legacy switchboard corpus setup."""
    subcorpus_mapping = { 'train': 'full', 'dev': 'dev_zoltan', 'eval': 'hub5-01'}
    train_eval_mapping = { 'train': 'train', 'dev': 'eval', 'eval': 'eval'}

    binaries = init_binaries()

    epochs = [12, 24, 32, 80, 160, 240, 320]
    system = NNSystem(
        num_input=40,
        epochs=epochs,
        rasr_binary_path=binaries.rasr,
        native_ops_path=binaries.native_lstm,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        returnn_python_home=RETURNN_PYTHON_HOME,
        returnn_root=binaries.returnn,
    )

    # Create the system
    for c, subcorpus in subcorpus_mapping.items():
        corpus = legacy.corpora[c][subcorpus]
        
        rasr_input = RasrDataInput(
            corpus_object=corpus,
            lexicon={
                "filename": legacy.lexica[train_eval_mapping[c]],
                "normalize_pronunciation": False,
            },
            lm={
                "filename": legacy.lms[train_eval_mapping[c]],
                "type": "ARPA",
                "scale": 12.0},
            concurrent=legacy.concurrent[train_eval_mapping[c]].get(subcorpus, 20),
        )

        system.add_corpus(c, rasr_input, add_lm=(c != "train"))
        system.feature_flows[c]['gt'] = features.basic_cache_flow(
            Path(default_feature_paths[c], cached=True),
        )
    system.alignments['train']['init_align'] = default_alignment_file
    system.mixtures['train']['init_mixture'] = default_mixture_path
    system._init_am()
    del system.crp['train'].acoustic_model_config.tdp

    st = system.crp["base"].acoustic_model_config.state_tying
    st.type = "cart"
    st.file = default_cart_file
    system.set_num_classes("cart", 9001)

    for args in ["default_nn_training_args", "default_scorer_args", "default_recognition_args"]:
        setattr(system, args, globals()[args])
    
    system.init_nn(**init_nn_args)
    for c in ["dev", "eval"]:
        # add glm and stm files
        system.glm_files[c] = legacy.glm_path[subcorpus_mapping[c]]
        system.stm_files[c] = legacy.stm_path[subcorpus_mapping[c]]
        system.set_hub5_scorer(corpus=c)
    return system

from collections import UserDict
class CustomDict(UserDict):
    """Custom dict that lets you map values of specific keys
    to a different value.
    """
    def map_item(self, key, func):
        d = self.copy()
        d[key] = func(d[key])
        return d
    
    def map(self, **kwargs):
        d = self.copy()
        for key, func in kwargs.items():
            d[key] = func(d[key])
        return d

# make cart questions and estimate cart on alignment
def get_cart(
    system: BaseSystem,
    hmm_partition: int=3,
):
    # create cart questions
    from i6_core.cart import PythonCartQuestions
    from i6_core.meta import CartAndLDA
    cart_questions = PythonCartQuestions(
        phonemes=legacy.cart_phonemes,
        steps=legacy.cart_steps,
        hmm_states=hmm_partition,
    )
    
    args = CustomDict(default_cart_lda_args.copy())
    corpus = args.pop("corpus") 
    context_size = args.pop("context_size")
    select_feature_flow = lambda flow: meta.select_element(system.feature_flows, corpus, flow)
    select_alignment = lambda alignment: meta.select_element(system.alignments, corpus, alignment)

    from i6_core import lda
    get_ctx_flow = lambda flow: lda.add_context_flow(
            feature_net = system.feature_flows[corpus][flow],
            max_size = context_size,
            right = int(context_size / 2)
        )
    args = args.map(
        initial_flow=select_feature_flow,
        context_flow=get_ctx_flow,
        alignment=select_alignment,
    )
    
    cart_and_lda = CartAndLDA(
        original_crp=system.crp[corpus],
        questions=cart_questions,
        **args
    )

    system.crp["base"].acoustic_model_config.state_tying.type = 'cart'
    system.crp["base"].acoustic_model_config.state_tying.file = cart_and_lda.last_cart_tree
    system.set_num_classes("cart", cart_and_lda.last_num_cart_labels)

    return None
