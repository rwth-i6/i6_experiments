__all__ = ["HybridArgs", "HybridSystem"]

import copy
import itertools
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)
from i6_core.util import MultiPath, MultiOutputPath

from .nn_system import NnSystem

from .util import (
    RasrInitArgs,
    ReturnnRasrDataInput,
    OggZipHdfDataInput,
    HybridArgs,
    NnRecogArgs,
    RasrSteps,
    NnForcedAlignArgs,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class HybridSystem(NnSystem):
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
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
    ):
        super().__init__(
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            blas_lib=blas_lib,
        )

        self.tf_fwd_input_name = "tf-fwd-input"

        self.cv_corpora = []
        self.devtrain_corpora = []

        self.train_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]
        self.cv_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]
        self.devtrain_input_data = (
            None
        )  # type:Optional[Dict[str, ReturnnRasrDataInput]]
        self.dev_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]
        self.test_input_data = None  # type:Optional[Dict[str, ReturnnRasrDataInput]]

        self.train_cv_pairing = None

        self.datasets = {}

        self.oggzips = {}  # TODO remove?
        self.hdfs = {}  # TODO remove?
        self.extern_rasrs = {}  # TODO remove?

        self.nn_configs = {}
        self.nn_models = {}  # TODO remove?
        self.nn_checkpoints = {}
        self.tf_flows = {}

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
        rasr_init_args: RasrInitArgs,
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
        self.rasr_init_args = rasr_init_args

        self._init_am(**self.rasr_init_args.am_args)

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

            corpus_pair_name = f"{trn_c}_{cv_c}"

            self.jobs[corpus_pair_name] = {}
            self.nn_models[corpus_pair_name] = {}
            self.nn_checkpoints[corpus_pair_name] = {}
            self.nn_configs[corpus_pair_name] = {}

    def _set_eval_data(self, data_dict):
        for c_key, c_data in data_dict.items():
            self.jobs[c_key] = {}
            self.ctm_files[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow
            self.feature_scorers[c_key] = {}
            if c_data.stm is not None:
                self.stm_files[c_key] = c_data.stm
            if c_data.glm is not None:
                self.glm_files[c_key] = c_data.glm

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

    def generate_lattices(self):
        pass

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

    def _get_feature_flow(
        self, feature_flow_key: str, data_input: ReturnnRasrDataInput
    ):
        """
        Select the appropriate feature flow from the data input object.

        If no flows are defined, tries to create the flow based on the features
        cache directly

        :param feature_flow_key: key identifier, e.g. "gt" or "mfcc40" etc...
        :param data_input: Data input object containing the flows
        :return: training feature flow
        """
        if isinstance(data_input.feature_flow, Dict):
            feature_flow = data_input.feature_flow[feature_flow_key]
        elif isinstance(data_input.feature_flow, rasr.FlowNetwork):
            feature_flow = data_input.feature_flow
        else:
            if isinstance(data_input.features, rasr.FlagDependentFlowAttribute):
                feature_path = data_input.features
            elif isinstance(data_input.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": data_input.features,
                    },
                )
            elif isinstance(data_input.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": data_input.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(data_input.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        return feature_flow

    def returnn_rasr_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
        feature_flow_key: str = "gt",
    ):
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        assert train_data.feature_flow == dev_data.feature_flow
        assert train_data.features == dev_data.features
        assert train_data.alignments == dev_data.alignments

        feature_flow = self._get_feature_flow(feature_flow_key, train_data)

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

    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.Checkpoint],
        acoustic_mixture_path: tk.Path,  # TODO maybe Optional if prior file provided -> automatically construct dummy file
        prior_scales: List[float],
        pronunciation_scales: List[float],
        lm_scales: List[float],
        optimize_am_lm_scale: bool,
        recognition_corpus_key: str,
        feature_flow_key: str,
        search_parameters: Dict,
        lattice_to_ctm_kwargs: Dict,
        parallelize_conversion: bool,
        rtf: int,
        mem: int,
        epochs: Optional[List[int]] = None,
        use_epoch_for_compile=False,
        forward_output_layer="output",
        native_ops: Optional[List[str]] = None,
        **kwargs,
    ):
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            native_op_paths = self.get_native_ops(op_names=native_ops)

            tf_graph = None
            if not use_epoch_for_compile:
                tf_graph = self.nn_compile_graph(name, returnn_config)

            feature_flow = self.feature_flows[recognition_corpus_key]
            if isinstance(feature_flow, Dict):
                feature_flow = feature_flow[feature_flow_key]
            assert isinstance(
                feature_flow, rasr.FlowNetwork
            ), f"type incorrect: {recognition_corpus_key} {type(feature_flow)}"

            epochs = epochs if epochs is not None else list(checkpoints.keys())

            for pron, lm, prior, epoch in itertools.product(
                pronunciation_scales, lm_scales, prior_scales, epochs
            ):
                assert epoch in checkpoints.keys()
                assert acoustic_mixture_path is not None

                if use_epoch_for_compile:
                    tf_graph = self.nn_compile_graph(name, returnn_config, epoch=epoch)

                scorer = rasr.PrecomputedHybridFeatureScorer(
                    prior_mixtures=acoustic_mixture_path,
                    priori_scale=prior,
                )

                tf_flow = make_precomputed_hybrid_tf_feature_flow(
                    tf_checkpoint=checkpoints[epoch],
                    tf_graph=tf_graph,
                    native_ops=native_op_paths,
                    output_layer_name=forward_output_layer,
                )
                flow = add_tf_flow_to_base_flow(feature_flow, tf_flow)

                self.feature_scorers[recognition_corpus_key][
                    f"pre-nn-{name}-{prior:02.2f}"
                ] = scorer
                self.feature_flows[recognition_corpus_key][
                    f"{feature_flow_key}-tf-{epoch:03d}"
                ] = flow

                recog_name = (
                    f"e{epoch:03d}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
                )
                recog_func(
                    name=f"{name}-{recognition_corpus_key}-{recog_name}",
                    prefix=f"nn_recog/{name}/",
                    corpus=recognition_corpus_key,
                    flow=flow,
                    feature_scorer=scorer,
                    pronunciation_scale=pron,
                    lm_scale=lm,
                    search_parameters=search_parameters,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    parallelize_conversion=parallelize_conversion,
                    rtf=rtf,
                    mem=mem,
                    **kwargs,
                )

    def nn_recog(
        self,
        train_name: str,
        train_corpus_key: str,
        returnn_config: Path,
        checkpoints: Dict[int, returnn.Checkpoint],
        step_args: HybridArgs,
    ):
        for recog_name, recog_args in step_args.recognition_args.items():
            for dev_c in self.dev_corpora:
                self.nn_recognition(
                    name=f"{train_corpus_key}-{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[
                        train_corpus_key
                    ].acoustic_mixtures,
                    recognition_corpus_key=dev_c,
                    **recog_args,
                )

            for tst_c in self.test_corpora:
                r_args = copy.deepcopy(recog_args)
                if (
                    step_args.test_recognition_args is None
                    or recog_name not in step_args.test_recognition_args.keys()
                ):
                    break
                r_args.update(step_args.test_recognition_args[recog_name])
                r_args["optimize_am_lm_scale"] = False
                self.nn_recognition(
                    name=f"{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[
                        train_corpus_key
                    ].acoustic_mixtures,
                    recognition_corpus_key=tst_c,
                    **r_args,
                )

    def nn_compile_graph(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        epoch: Optional[int] = None,
    ):
        """
        graph compile helper including alias

        :param name: name for the alias
        :param returnn_config: ReturnnConfig that defines the graph
        :param epoch: optionally a specific epoch to compile when using
            e.g. `def get_network(epoch=...)` in the config
        :return: the TF graph
        """
        graph_compile_job = returnn.CompileTFGraphJob(
            returnn_config,
            epoch=epoch,
            returnn_root=self.returnn_root,
            returnn_python_exe=self.returnn_python_exe,
        )
        graph_compile_job.add_alias(f"nn_recog/graph/{name}")
        return graph_compile_job.out_graph

    # -------------------- Rescoring  --------------------

    def nn_rescoring(self):
        # TODO calls rescoring setup
        raise NotImplementedError

    # -------------------- run functions  --------------------

    def run_data_preparation_step(self, step_args):
        # TODO here be ogg zip generation for training or lattice generation for SDT
        raise NotImplementedError

    def run_nn_step(self, step_name: str, step_args: HybridArgs):
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = (
                [pairing[2]]
                if len(pairing) >= 3
                else list(step_args.returnn_training_configs.keys())
            )
            dvtr_c_list = [pairing[3]] if len(pairing) >= 4 else self.devtrain_corpora
            dvtr_c_list = [None] if len(dvtr_c_list) == 0 else dvtr_c_list

            for name, dvtr_c in itertools.product(name_list, dvtr_c_list):
                if isinstance(self.train_input_data[trn_c], ReturnnRasrDataInput):
                    returnn_train_job = self.returnn_rasr_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                    )
                else:
                    returnn_train_job = self.returnn_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                        devtrain_corpus_key=dvtr_c,
                    )

                returnn_recog_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )

                self.nn_recog(
                    train_name=name,
                    train_corpus_key=trn_c,
                    returnn_config=returnn_recog_config,
                    checkpoints=returnn_train_job.out_checkpoints,
                    step_args=step_args,
                )

    def run_nn_recog_step(self, step_args: NnRecogArgs):
        for eval_c in self.dev_corpora + self.test_corpora:
            self.nn_recognition(recognition_corpus_key=eval_c, **asdict(step_args))

    def run_rescoring_step(self, step_args):
        for dev_c in self.dev_corpora:
            raise NotImplementedError

        for tst_c in self.test_corpora:
            raise NotImplementedError

    def run_realign_step(self, step_args):
        for trn_c in self.train_corpora:
            for devtrv_c in self.devtrain_corpora[trn_c]:
                raise NotImplementedError
            for cv_c in self.cv_corpora[trn_c]:
                raise NotImplementedError

    def run_forced_align_step(self, step_args: NnForcedAlignArgs):
        for tc_key in step_args["target_corpus_keys"]:
            featurer_scorer_corpus_key = step_args["feature_scorer_corpus_key"]
            scorer_model_key = step_args["scorer_model_key"]
            epoch = step_args["epoch"]
            base_flow = self.feature_flows[tc_key][step_args["base_flow_key"]]
            tf_flow = self.tf_flows[step_args["tf_flow_key"]]
            feature_flow = self.add_tf_flow_to_base_flow(base_flow, tf_flow)

            self.forced_align(
                name=step_args["name"],
                target_corpus_key=tc_key,
                flow=feature_flow,
                feature_scorer_corpus_key=featurer_scorer_corpus_key,
                feature_scorer=scorer_model_key,
                dump_alignment=step_args["dump_alignment"],
            )

    # -------------------- run setup  --------------------

    def run(self, steps: RasrSteps):
        if "init" in steps.get_step_names_as_list():
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        self.prepare_scoring()

        for step_idx, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    corpus_list = (
                        self.train_corpora
                        + self.cv_corpora
                        + self.devtrain_corpora
                        + self.dev_corpora
                        + self.test_corpora
                    )
                    step_args = self.rasr_init_args.feature_extraction_args
                else:
                    corpus_list = step_args.pop("corpus_list")

                for all_c in corpus_list:
                    if all_c not in self.feature_caches.keys():
                        self.feature_caches[all_c] = {}
                    if all_c not in self.feature_bundles.keys():
                        self.feature_bundles[all_c] = {}
                    if all_c not in self.feature_flows.keys():
                        self.feature_flows[all_c] = {}
                self.extract_features(step_args, corpus_list=corpus_list)

            # ---------- Prepare data ----------
            if step_name.startswith("data"):
                self.run_data_preparation_step(step_args)

            # ---------- NN Training ----------
            if step_name.startswith("nn"):
                self.run_nn_step(step_name, step_args)

            if step_name.startswith("recog"):
                self.run_nn_recog_step(step_args)

            # ---------- Rescoring ----------
            if step_name.startswith("rescor"):
                self.run_rescoring_step(step_args)

            # ---------- Realign ----------
            if step_name.startswith("realign"):
                self.run_realign_step(step_args)

            # ---------- Forced Alignment ----------
            if step_name.startswith("forced") or step_name.startswith("align"):
                self.run_forced_align_step(step_args)
