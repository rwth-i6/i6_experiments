from sisyphus import tk, Path

import os

from recipe.i6_experiments.users.mann.setups.nn_system import BaseSystem, NNSystem, ExpConfig, FilterAlignmentPlugin
from recipe.i6_experiments.users.mann.setups import prior
from i6_core.datasets import tedlium2 as tedlium
from .common import (
    init_segment_order_shuffle,
    init_binaries, init_env,
    RETURNN_PYTHON_EXE,
    RETURNN_PYTHON_HOME,
    init_extended_train_corpus
)
from i6_experiments.common.baselines.tedlium2.default_tools import SCTK_BINARY_PATH
from i6_experiments.common.baselines.tedlium2.data import get_corpus_data_inputs
from i6_experiments.common.datasets.tedlium2.constants import CONCURRENT
from i6_experiments.common.datasets.tedlium2.corpus import (
    get_corpus_object_dict,
    get_stm_dict,
)
from i6_experiments.common.datasets.tedlium2.lexicon import (
    get_g2p_augmented_bliss_lexicon,
)
from recipe.i6_core import (
    features,
    returnn,
    lexicon,
    meta,
)

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput, RasrInitArgs

TOTAL_FRAMES = 69_086_029

def get_init_args():
    corpus_object_dict = get_corpus_object_dict(
        audio_format="wav", output_prefix="corpora"
    )

    rasr_data_input_dict = dict()

    wei_lm_config = {
        "filename": tk.Path(
            "/work/asr4/zhou/data/ted-lium2/lm-data/model-kazuki/countLM/4-gram.lm.dev_opt3.bgd.interpolation.pruned.5e-10.gz",
            cached=True,
        ),
        "type": "ARPA",
        "scale": 3.0, # heuristic value for monophone based on Wei's value 7.0 for CART
    }

    g2p_lexicon_config = {
        "filename": get_g2p_augmented_bliss_lexicon(output_prefix="lexicon"),
        "normalize_pronunciation": False,
    }
    tk.register_output("g2p_lexicon_config", g2p_lexicon_config["filename"])

    stm_dict = get_stm_dict()

    for name, crp_obj in corpus_object_dict.items():
        rasr_data_input_dict[name] = RasrDataInput(
            corpus_object=crp_obj,
            lexicon=g2p_lexicon_config,
            concurrent=CONCURRENT[name],
            lm=wei_lm_config,
            stm=stm_dict[name],
        )
    return rasr_data_input_dict

def get_rasr_init_args():
    return RasrInitArgs(
        am_args=None,
        costa_args=None,
        feature_extraction_args=None,
        scorer="sclite",
        scorer_args={"sctk_binary_path": SCTK_BINARY_PATH},
    )
    
default_feature_paths = {
    "dev": "/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/features/extraction/FeatureExtraction.LOGMEL.QSJduoVpxd59/output/logmel.cache.bundle",
    "train": "/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/features/extraction/FeatureExtraction.LOGMEL.zX6arCkgNY8y/output/logmel.cache.bundle",
    "train_cv": "/u/zhou/asr-exps/ted-lium2/20191022_new_baseline/work/features/extraction/FeatureExtraction.LOGMEL.QSJduoVpxd59/output/logmel.cache.bundle",
}

scorer_args = {"sctk_binary_path": SCTK_BINARY_PATH}

default_recognition_args = {
    'corpus': 'dev',
    'pronunciation_scale': 0.0,
    # 'lm_scale': 10.,
    'search_parameters': {
        'beam-pruning': 16.0,
        'beam-pruning-limit': 100000,
        'word-end-pruning': 0.5,
        'word-end-pruning-limit': 25000},
    'lattice_to_ctm_kwargs' : {
        'fill_empty_segments' : True,
        'best_path_algo': 'bellman-ford'
    },
    'rtf': 50,
}

default_scorer_args = {
    'prior_mixtures': None,
    'prior_scale': 0.70,
    'feature_dimension': 80
}

default_nn_training_args = {
    'feature_corpus': 'train',
    'alignment': None,
    'num_epochs': 200,
    'partition_epochs': {'train': 5, 'dev' : 1},
    'save_interval': 5,
    'num_classes': None,
    'time_rqmt': 60,
    'mem_rqmt' : 12,
    'use_python_control': True,
    'log_verbosity': 3,
    'feature_flow': 'logmel'
}

CORPUS_NAME = "TED-LIUM-realease2"

def get_tedlium_system(
    segment_order_shuffle=True,
    full_train_corpus=True,
    mono_eow=True,
):
    init_env()
    binaries = init_binaries(
        rasr_branch="factored-hybrid-dev",
        rasr_commit="d506533"
    )
    _tmp_rasr_root = Path("/u/raissi/dev/master-rasr-fsa").join_right("arch/linux-x86_64-standard")
    binaries = binaries._replace(rasr=_tmp_rasr_root)
    binaries.rasr.hash_overwrite = "FACTORED_HYBRID_RASR_BINARY_PATH"
    init_args = get_init_args()

    epochs = [5, 10, 40, 80, 160, 200]

    system = NNSystem(
        num_input=80,
        epochs=epochs,
        rasr_binary_path=binaries.rasr,
        native_ops_path=binaries.native_lstm,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        returnn_python_home=RETURNN_PYTHON_HOME,
        returnn_root=binaries.returnn,
    )

    system.dev_corpora = ["dev"]
    system.rasr_init_args = get_rasr_init_args()

    for corpus_key, corpus_data_input in init_args.items():
        if corpus_key == "test": continue
        system.add_corpus(corpus_key, corpus_data_input, add_lm=(corpus_key != "train"))
        feature_path = Path(default_feature_paths[corpus_key], cached=True)
        feature_flow = 'logmel'
        system.feature_flows[corpus_key][feature_flow] = features.basic_cache_flow(feature_path)
        system.feature_bundles[corpus_key][feature_flow] = feature_path
        # system.set_sclite_scorer()
    
    system.prepare_scoring()

    system._init_am()

    for args in [
        "default_nn_training_args",
        "default_scorer_args",
        "default_recognition_args"
    ]:
        setattr(system, args, globals()[args])
    
    if mono_eow:
        # word end classes
        system.set_state_tying(
            value="monophone-no-tying-dense",
            use_boundary_classes=False,
            use_word_end_classes=True,
        )
        system.state_tying_mode = "dense"
    
    if full_train_corpus:
        extra_args = init_extended_train_corpus(
            system, CORPUS_NAME,
            reinit_shuffle=segment_order_shuffle
        )
        system.default_nn_training_args.update(extra_args)
    else:
        system.init_nn(
            name="returnn",
            corpus="train",
            dev_size=1e-4,
        )

    if segment_order_shuffle and not full_train_corpus:
        init_segment_order_shuffle(system, "returnn_train", 300)

    system.init_prior_system(
        total_frames=TOTAL_FRAMES,
        eps=1e-12,
        extra_num_states=True,
    )

    return system

def get():
    return get_tedlium_system(
        segment_order_shuffle=False,
        full_train_corpus=False,
        mono_eow=False,
    )
