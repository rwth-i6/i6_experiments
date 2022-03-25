__all__ = ["NnArgs", "NnSystem"]

import copy
import itertools
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

import sisyphus.toolkit as tk
import sisyphus.global_settings as gs

from sisyphus.delayed_ops import DelayedFormat

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath, MultiOutputPath

from .rasr_system import RasrSystem

from .util import (
    RasrInitArgs,
    ReturnnRasrDataInput,
    OggZipHdfDataInput,
    NnArgs,
    NnRecogArgs,
    RasrSteps,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class NnSystem(RasrSystem):
    """
    - 5 corpora types: train, devtrain, cv, dev and test
        devtrain is a small split from the train set which is evaluated like
        the cv but not used for error calculating. Since we can have different
        datasubsets per subepoch, we do not caculate the tran score/error on
        a consistent datasubset
    - two training data settings: defined in returnn config or not
    - 3 different types of decoding: returnn, rasr, rasr-label-sync
    - 2 different lm: count, neural
    - cv is dev for returnn training
    - dev for lm param tuning
    - test corpora for final eval

    settings needed:
    - am
    - lm
    - lexicon
    - ce training
    - ce recognition
    - ce rescoring
    - smbr training
    - smbr recognition
    - smbr rescoring
    """

    def __init__(
        self,
        returnn_root: Optional[str] = None,
        returnn_python_home: Optional[str] = None,
        returnn_python_exe: Optional[str] = None,
    ):
        super().__init__()

        self.returnn_root = returnn_root or (
            gs.RETURNN_ROOT if hasattr(gs, "RETURNN_ROOT") else None
        )
        self.returnn_python_home = returnn_python_home or (
            gs.RETURNN_PYTHON_HOME if hasattr(gs, "RETURNN_PYTHON_HOME") else None
        )
        self.returnn_python_exe = returnn_python_exe or (
            gs.RETURNN_PYTHON_EXE if hasattr(gs, "RETURNN_PYTHON_EXE") else None
        )


        self.cv_corpora = []
        self.devtrain_corpora = []

        self.train_input_data = None
        self.cv_input_data = None
        self.devtrain_input_data = None
        self.dev_input_data = None
        self.test_input_data = None

        self.train_cv_pairing = None

        self.datasets = {}

        self.hdfs = {}  # TODO remove?

        self.nn_configs = {}
        self.nn_models = {}  # TODO remove?
        self.nn_checkpoints = {}


    # -------------------- Helpers --------------------


    def _add_output_alias_for_train_job(
        self,
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],
        train_corpus_key: str,
        cv_corpus_key: str,
        name: str,
    ):
        train_job.add_alias(f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_train")
        self.jobs[f"{train_corpus_key}_{cv_corpus_key}"][name] = train_job
        self.nn_models[f"{train_corpus_key}_{cv_corpus_key}"][
            name
        ] = train_job.out_models
        self.nn_checkpoints[f"{train_corpus_key}_{cv_corpus_key}"][
            name
        ] = train_job.out_checkpoints
        self.nn_configs[f"{train_corpus_key}_{cv_corpus_key}"][
            name
        ] = train_job.out_returnn_config_file
        tk.register_output(
            f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    # -------------------- Setup --------------------
    def init_system(
        self,
        hybrid_init_args: RasrInitArgs,
        train_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
        cv_data: Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]],
        devtrain_data: Optional[
            Dict[str, Union[ReturnnRasrDataInput, OggZipHdfDataInput]]
        ] = None,
        dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        train_cv_pairing: Optional[
            List[Tuple[str, ...]]
        ] = None,  # List[Tuple[trn_c, cv_c, name, dvtr_c]]
    ):
        self.hybrid_init_args = hybrid_init_args

        self._init_am(**self.hybrid_init_args.am_args)

        devtrain_data = devtrain_data if devtrain_data is not None else {}
        dev_data = dev_data if dev_data is not None else {}
        test_data = test_data if test_data is not None else {}

        self._assert_corpus_name_unique(
            train_data, cv_data, devtrain_data, dev_data, test_data
        )

        self.train_input_data = train_data
        self.cv_input_data = cv_data
        self.devtrain_input_data = devtrain_data
        self.dev_input_data = dev_data
        self.test_input_data = test_data

        self.train_corpora.extend(list(train_data.keys()))
        self.cv_corpora.extend(list(cv_data.keys()))
        self.devtrain_corpora.extend(list(devtrain_data.keys()))
        self.dev_corpora.extend(list(dev_data.keys()))
        self.test_corpora.extend(list(test_data.keys()))

        self._set_eval_data(dev_data)
        self._set_eval_data(test_data)

        self.train_cv_pairing = (
            list(itertools.product(self.train_corpora, self.cv_corpora))
            if train_cv_pairing is None
            else train_cv_pairing
        )

        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]

            self.jobs[f"{trn_c}_{cv_c}"] = {}
            self.nn_models[f"{trn_c}_{cv_c}"] = {}
            self.nn_checkpoints[f"{trn_c}_{cv_c}"] = {}
            self.nn_configs[f"{trn_c}_{cv_c}"] = {}

    def _set_eval_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.jobs[c_key] = {}
            self.ctm_files[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow

    def prepare_data(self, raw_sampling_rate: int, feature_sampling_rate: int):
        for name in self.train_corpora + self.devtrain_corpora + self.cv_corpora:
            self.jobs[name]["ogg_zip"] = j = returnn.BlissToOggZipJob(
                bliss_corpus=self.crp[name].corpus_config.corpus_file,
                segments=self.crp[name].segment_path,
                rasr_cache=self.feature_flows[name]["init"],
                raw_sample_rate=raw_sampling_rate,
                feat_sample_rate=feature_sampling_rate,
            )
            self.oggzips[name] = j.out_ogg_zip
            j.add_alias(f"oggzip/{name}")

            # TODO self.jobs[name]["hdf_full"] = j = returnn.ReturnnDumpHDFJob()

    # -------------------- Training --------------------

    def returnn_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
        devtrain_corpus_key=None,
    ):
        assert isinstance(returnn_config, returnn.ReturnnConfig)

        returnn_config.config["train"] = self.train_input_data[
            train_corpus_key
        ].get_data_dict()
        returnn_config.config["dev"] = self.cv_input_data[cv_corpus_key].get_data_dict()
        if devtrain_corpus_key is not None:
            returnn_config.config["eval_datasets"] = {
                "devtrain": self.devtrain_input_data[
                    devtrain_corpus_key
                ].get_data_dict()
            }

        train_job = returnn.ReturnnTrainingJob(
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    def returnn_rasr_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        assert train_data.feature_flow == dev_data.feature_flow
        assert train_data.features == dev_data.features
        assert train_data.alignments == dev_data.alignments

        if train_data.feature_flow is not None:
            feature_flow = train_data.feature_flow
        else:
            if isinstance(train_data.features, rasr.FlagDependentFlowAttribute):
                feature_path = train_data.features
            elif isinstance(train_data.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": train_data.features,
                    },
                )
            elif isinstance(train_data.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": train_data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(train_data.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        if isinstance(train_data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(train_data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(train_data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(train_data.alignments, tk.Path):
            alignments = train_data.alignments
        else:
            raise NotImplementedError

        assert isinstance(returnn_config, returnn.ReturnnConfig)

        train_job = returnn.ReturnnRasrTrainingJob(
            train_crp=train_crp,
            dev_crp=dev_crp,
            feature_flow=feature_flow,
            alignment=alignments,
            returnn_config=returnn_config,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
            **nn_train_args,
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    # -------------------- Recognition --------------------




    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for eval_c in self.dev_corpora + self.test_corpora:
            stm_args = (
                self.hybrid_init_args.stm_args
                if self.hybrid_init_args.stm_args is not None
                else {}
            )
            self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.hybrid_init_args.feature_extraction_args
                for all_c in (
                    self.train_corpora
                    + self.cv_corpora
                    + self.devtrain_corpora
                    + self.dev_corpora
                    + self.test_corpora
                ):
                    self.feature_caches[all_c] = {}
                    self.feature_bundles[all_c] = {}
                    self.feature_flows[all_c] = {}
                self.extract_features(step_args)

            # ---------- Prepare data ----------
            if step_name.startswith("data"):
                self.run_data_preparation_step(step_args)

            # ---------- NN Training ----------
            if step_name.startswith("nn"):
                self.run_nn_step(step_args)

            if step_name.startswith("recog"):
                # call the specific recognition on the Recognizer
                pass
                
            # ---------- Realign ----------
            if step_name.startswith("realign"):
                # call the specific function on your Aligner
                pass