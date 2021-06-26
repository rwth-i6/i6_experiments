__all__ = ["NnArgs", "NnSystem"]

import sys
from typing import Dict, Optional

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

# -------------------- Recipes --------------------

import i6_core.am as am
import i6_core.corpus as corpus_recipes
import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.returnn as returnn
import i6_core.util as util

from .base_system import BaseSystem

from .util import HybridInitArgs, HybridDataInput, NnArgs

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class NnSystem(BaseSystem):
    """
    - 5 corpora types: train, devtrain, cv, dev and test
    - two training data settings: defined in returnn config or not
    - 3 different types of decoding: returnn, rasr, rasr-wei
    - 3 different lm: count, lstm, trafo
    - cv is dev for returnn training
    - dev for lm param tuning
    - test corpora for final eval

    settings needed:
    - am
    - lm
    - lexicon
    - ce training
    - ce count lm recognition
    - ce lstm lm recognition
    - ce trafo lm recognition
    - ce rescoring
    - smbr training
    - smbr count lm recognition
    - smbr lstm lm recognition
    - smbr trafo lm recognition
    - smbr rescoring
    """

    def __init__(
        self,
        returnn_root: str = gs.RETURNN_ROOT,
        returnn_python_home: str = gs.RETURNN_PYTHON_HOME,
        returnn_python_exe: str = gs.RETURNN_PYTHON_EXE,
    ):
        super().__init__()

        self.returnn_root = returnn_root
        self.returnn_python_home = returnn_python_home
        self.return_python_exe = returnn_python_exe

        self.nn_args = None

        self.devtrain_corpora = []
        self.cv_corpora = []

        self.base_returnn_args = None
        self.returnn_args = []

    # -------------------- Setup --------------------
    def init_system(
        self,
        hybrid_init_args: HybridInitArgs,
        nn_args: NnArgs,
        train_data: Dict[str, HybridDataInput],
        devtrain_data: Dict[str, HybridDataInput],
        cv_data: Dict[str, HybridDataInput],
        dev_data: Dict[str, HybridDataInput],
        test_data: Dict[str, HybridDataInput],
    ):
        self.hybrid_init_args = hybrid_init_args
        self.nn_args = nn_args

        self._init_am(**self.hybrid_init_args.am_args)

        self._assert_corpus_name_unique(
            train_data, devtrain_data, cv_data, dev_data, test_data
        )

        for name, v in sorted(train_data.items()):
            self.add_corpus(name, data=v, add_lm=False)
            self.train_corpora.append(name)
            assert self.concurrent[name] == 1, "concurrency for train data should be 1"

        for name, v in sorted(devtrain_data.items()):
            self.add_corpus(name, data=v, add_lm=False)
            self.devtrain_corpora.append(name)
            assert (
                self.concurrent[name] == 1
            ), "concurrency for devtrain data should be 1"

        for name, v in sorted(cv_data.items()):
            self.add_corpus(name, data=v, add_lm=False)
            self.cv_corpora.append(name)
            assert self.concurrent[name] == 1, "concurrency for cv data should be 1"

        for name, v in sorted(dev_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.dev_corpora.append(name)

        for name, v in sorted(test_data.items()):
            self.add_corpus(name, data=v, add_lm=True)
            self.test_corpora.append(name)

    # -------------------- Training --------------------

    def nn_rasr_training(
        self,
        name: str,
        train_corpus: str,
        dev_corpus: str,
        devtrain_corpus: str,
        feature_corpus: str,
        feature_flow: str,
        alignment: str,
        num_classes: int,
        horovod_num_processes: int,
        returnn_config: returnn.ReturnnConfig,
    ):

        self.train_nn(
            name=name,
            feature_corpus=feature_corpus,
            train_corpus=train_corpus,
            dev_corpus=dev_corpus,
            feature_flow=feature_flow,
            alignment=alignment,
            num_classes=num_classes,
            returnn_config=returnn_config,
        )

    # -------------------- Count LM Recognition --------------------
    # -------------------- LSTM LM Recognition --------------------
    # -------------------- Trafo LM Recognition --------------------
    # -------------------- Rescoring  --------------------
    # -------------------- run setup  --------------------

    def run(self, steps=("all",)):
        assert len(steps) > 0
        if list(steps) == ["all"]:
            steps = ["ce", "smbr", "lstm_recog", "trafo_recog", "rescoring"]

        if "init" in steps:
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        if "extract" in steps:
            self.extract_features(feat_args=self.nn_args.feature_extraction_args)
