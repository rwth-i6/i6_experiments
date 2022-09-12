import copy
import itertools
import sys
from typing import Dict, List, Tuple, Optional, Union
from i6_core.rasr.config import RasrConfig
from i6_core.rasr.flow import FlowNetwork

# -------------------- Sisyphus --------------------

from sisyphus import gs, tk

# -------------------- Recipes --------------------

from i6_core import features
from i6_core import rasr
from i6_core import recognition as recog
from i6_core.util import MultiPath, MultiOutputPath
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.compile import CompileTFGraphJob, CompileNativeOpJob
from i6_core.returnn.rasr_training import ReturnnRasrTrainingJob
from i6_core.returnn.extract_prior import ReturnnRasrComputePriorJob
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.search import SearchBPEtoWordsJob, SearchWordsToCTMJob
from i6_core.lexicon.allophones import DumpStateTyingJob

from i6_experiments.common.setups.rasr.nn_system import NnSystem
from i6_experiments.common.setups.rasr.util import (
    RasrInitArgs,
    ReturnnRasrDataInput,
    RasrSteps,
    HybridArgs,
)
from i6_experiments.users.berger.recipe.returnn.rasr_search import ReturnnRasrSearchJob
from i6_experiments.users.berger.recipe.lexicon.modification import (
    AddBoundaryMarkerToLexiconJob,
)
from i6_experiments.users.berger.recipe.recognition.label_sync_search import (
    LabelSyncSearchJob,
)
from i6_experiments.users.berger.recipe.mm.alignment import LabelAlignmentJob
from i6_experiments.users.berger.recipe.rasr import (
    LabelTree,
    LabelScorer,
    GenerateLabelFileFromStateTyingJob,
)
from i6_experiments.users.berger.network.helpers.ctc_loss import (
    make_ctc_rasr_loss_config,
)

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class TransducerSystem(NnSystem):
    """
    - 4 corpora types: train, cv, dev and test
    - only train and cv corpora will be aligned
    - dev corpora for tuning
    - test corpora for final eval

    to create beforehand:
    - corpora: name and i6_core.meta.system.Corpus
    - lexicon
    - lm

    settings needed:
    - am
    - lm
    - lexicon
    - feature extraction
    """

    def __init__(
        self,
        returnn_root: Optional[str] = None,
        returnn_python_home: Optional[str] = None,
        returnn_python_exe: Optional[str] = None,
        rasr_python_home: Optional[str] = None,
        rasr_python_exe: Optional[str] = None,
        blas_lib: Optional[str] = None,
    ) -> None:
        super().__init__(
            returnn_root=returnn_root,
            returnn_python_home=returnn_python_home,
            returnn_python_exe=returnn_python_exe,
            blas_lib=blas_lib,
        )
        rasr_python_home = rasr_python_home or (
            gs.RASR_PYTHON_HOME if hasattr(gs, "RASR_PYTHON_HOME") else None
        )
        rasr_python_exe = rasr_python_exe or (
            gs.RASR_PYTHON_EXE if hasattr(gs, "RASR_PYTHON_EXE") else None
        )
        # assert rasr_python_home
        assert rasr_python_exe
        self.crp["base"].python_home = rasr_python_home
        self.crp["base"].python_program_name = rasr_python_exe

        self.cv_corpora = []
        self.align_corpora = []

        self.train_input_data = None
        self.cv_input_data = None
        self.dev_input_data = None
        self.test_input_data = None

        self.train_cv_pairing = None

        self.datasets = {}

        self.tf_nn_checkpoints = {}  # type: Dict[str, Dict[int, Checkpoint]]

        self.prior_files = {}  # type: Dict[str, Dict[int, tk.Path]]

    # -------------------- Helpers ---------------------

    @property
    def input_data(self) -> List:
        result = {}
        for data in [
            self.train_input_data,
            self.cv_input_data,
            self.dev_input_data,
            self.test_input_data,
        ]:
            result.update(data if data is not None else {})
        return result

    def make_model_loader_config(
        self, tf_graph: tk.Path, tf_checkpoint: Checkpoint
    ) -> RasrConfig:
        loader_config = rasr.RasrConfig()
        # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
        native_op = (
            self.jobs["general"]
            .setdefault(
                "compile_lstm",
                CompileNativeOpJob(
                    "NativeLstm2",
                    returnn_python_exe=self.returnn_python_exe,
                    returnn_root=self.returnn_root,
                    blas_lib=self.blas_lib,
                ),
            )
            .out_op
        )

        loader_config.type = "meta"
        loader_config.meta_graph_file = tf_graph
        loader_config.saved_model_file = tf_checkpoint
        loader_config.required_libraries = native_op
        return loader_config

    def get_prior_file(
        self,
        name: str,
        epoch: int,
        label_scorer_args: Dict,
        train_corpus_key: Optional[str] = None,
        cv_corpus_key: Optional[str] = None,
        train_returnn_config: Optional[ReturnnConfig] = None,
        checkpoint: Optional[Checkpoint] = None,
        prior_args: Optional[Dict] = None,
    ) -> tk.Path:
        if name not in self.prior_files:
            self.prior_files[name] = {}

        if label_scorer_args.get("prior_file", None):
            prior_file = label_scorer_args["prior_file"]
            self.prior_files[name][epoch] = prior_file
        elif epoch in self.prior_files[name]:
            prior_file = self.prior_files[name][epoch]
        else:  # Compute prior
            assert (
                train_corpus_key
                and cv_corpus_key
                and train_returnn_config
                and checkpoint
            ), "Must provide arguments for prior computation"
            prior_file = self.returnn_rasr_compute_priors(
                name=f"{name}-ep{epoch}",
                returnn_config=train_returnn_config,
                train_corpus_key=train_corpus_key,
                cv_corpus_key=cv_corpus_key,
                checkpoint=checkpoint,
                nn_prior_args=prior_args or {},
            )

        self.prior_files[name][epoch] = prior_file
        return prior_file

    @staticmethod
    def autoregressive_decoding(label_scorer_type: str) -> bool:
        return label_scorer_type != "precomputed-log-posterior"

    def make_tf_graph(
        self, name: str, returnn_config: ReturnnConfig, label_scorer_type: str
    ) -> tk.Path:
        rec_step_by_step = (
            "output" if self.autoregressive_decoding(label_scorer_type) else None
        )
        rec_json_info = True if rec_step_by_step else None
        graph_compile_job = self.jobs["general"].setdefault(
            f"{name}_compile",
            CompileTFGraphJob(
                returnn_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                rec_step_by_step=rec_step_by_step,
                rec_json_info=rec_json_info,
            ),
        )
        return graph_compile_job.out_graph

    def make_tf_feature_flow(
        self,
        feature_flow: FlowNetwork,
        tf_graph: tk.Path,
        tf_checkpoint: Checkpoint,
        output_tensor_name: str = "output/output_batch_major",
        append=False,
    ) -> FlowNetwork:

        # tf flow (model scoring done in tf flow node) #
        tf_flow = rasr.FlowNetwork()
        tf_flow.add_input("input-features")
        tf_flow.add_output("features")
        tf_flow.add_param("id")

        tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link("network:input-features", tf_fwd + ":features")
        tf_flow.link(tf_fwd + ":log-posteriors", "network:features")

        tf_flow.config = rasr.RasrConfig()
        tf_flow.config[tf_fwd].input_map.info_0.param_name = "features"
        tf_flow.config[
            tf_fwd
        ].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
        tf_flow.config[
            tf_fwd
        ].input_map.info_0.seq_length_tensor_name = (
            "extern_data/placeholders/data/data_dim0_size"
        )

        tf_flow.config[tf_fwd].output_map.info_0.param_name = "log-posteriors"
        tf_flow.config[tf_fwd].output_map.info_0.tensor_name = output_tensor_name

        tf_flow.config[tf_fwd].loader.type = "meta"
        tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph
        tf_flow.config[tf_fwd].loader.saved_model_file = tf_checkpoint

        native_op = self.jobs["general"].setdefault(
            "compile_lstm",
            CompileNativeOpJob(
                "NativeLstm2",
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                blas_lib=self.blas_lib,
            ).out_op,
        )

        tf_flow.config[tf_fwd].loader.required_libraries = native_op

        # interconnect flows #
        tf_feature_flow = rasr.FlowNetwork()
        base_mapping = tf_feature_flow.add_net(feature_flow)
        tf_mapping = tf_feature_flow.add_net(tf_flow)
        tf_feature_flow.interconnect_inputs(feature_flow, base_mapping)
        tf_feature_flow.interconnect(
            feature_flow,
            base_mapping,
            tf_flow,
            tf_mapping,
            {"features": "input-features"},
        )

        if append:
            concat = tf_feature_flow.add_node(
                "generic-vector-f32-concat",
                "concat",
                attr={"timestamp-port": "features"},
            )
            tf_feature_flow.link(
                tf_mapping[tf_flow.get_output_links("features").pop()], concat + ":tf"
            )
            tf_feature_flow.link(
                base_mapping[feature_flow.get_output_links("features").pop()],
                concat + ":features",
            )
            tf_feature_flow.add_output("features")
            tf_feature_flow.link(concat, "network:features")
        else:
            tf_feature_flow.interconnect_outputs(tf_flow, tf_mapping)
        # ensure cache_mode as base feature net
        tf_feature_flow.add_flags(feature_flow.flags)
        return tf_feature_flow

    def get_feature_flow_for_corpus(self, corpus_key: str) -> FlowNetwork:
        if corpus_key in self.feature_flows:
            return self.feature_flows[corpus_key]
        data = self.input_data[corpus_key]

        feature_flow = self.make_feature_flow_for_data(data)
        self.feature_flows[corpus_key] = feature_flow
        return feature_flow

    @staticmethod
    def make_feature_flow_for_data(data: ReturnnRasrDataInput) -> FlowNetwork:
        if data.feature_flow is not None:
            feature_flow = data.feature_flow
        else:
            if isinstance(data.features, rasr.FlagDependentFlowAttribute):
                feature_path = data.features
            elif isinstance(data.features, (MultiPath, MultiOutputPath)):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": data.features,
                    },
                )
            elif isinstance(data.features, tk.Path):
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "bundle": data.features,
                    },
                )
            else:
                raise NotImplementedError

            feature_flow = features.basic_cache_flow(feature_path)
            if isinstance(data.features, tk.Path):
                feature_flow.flags = {"cache_mode": "bundle"}

        return feature_flow

    def get_alignments_for_corpus(self, corpus_key: str) -> FlowNetwork:
        if corpus_key in self.alignments:
            return self.alignments[corpus_key]
        data = self.input_data[corpus_key]

        alignments = self.make_alignments_for_data(data)
        self.alignments[corpus_key] = alignments
        return alignments

    @staticmethod
    def make_alignments_for_data(data: ReturnnRasrDataInput) -> Union[tk.Path, None]:
        if isinstance(data.alignments, rasr.FlagDependentFlowAttribute):
            alignments = copy.deepcopy(data.alignments)
            net = rasr.FlowNetwork()
            net.flags = {"cache_mode": "bundle"}
            alignments = alignments.get(net)
        elif isinstance(data.alignments, (MultiPath, MultiOutputPath)):
            raise NotImplementedError
        elif isinstance(data.alignments, tk.Path):
            alignments = data.alignments
        else:
            return None

        return alignments

    def _add_output_alias_for_train_job(
        self,
        train_job: ReturnnRasrTrainingJob,
        train_corpus_key: str,
        cv_corpus_key: str,
        name: str,
    ) -> None:
        self.tf_nn_checkpoints[
            f"{train_corpus_key}_{cv_corpus_key}"
        ] = train_job.out_checkpoints
        train_job.add_alias(f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_train")
        tk.register_output(
            f"train_nn/{train_corpus_key}_{cv_corpus_key}/{name}_learning_rate.png",
            train_job.out_plot_lr,
        )

    # -------------------- Setup --------------------
    def init_system(
        self,
        hybrid_init_args: RasrInitArgs,
        train_data: Dict[str, ReturnnRasrDataInput],
        cv_data: Dict[str, ReturnnRasrDataInput],
        dev_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        test_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        align_data: Optional[Dict[str, ReturnnRasrDataInput]] = None,
        train_cv_pairing: Optional[List[Tuple[str, ...]]] = None,
        train_cv_align_pairing: Optional[List[Tuple[str, ...]]] = None,
    ) -> None:
        self.hybrid_init_args = hybrid_init_args

        self._init_am(**self.hybrid_init_args.am_args)

        dev_data = dev_data or {}
        test_data = test_data or {}
        align_data = align_data or {}

        self._assert_corpus_name_unique(train_data, cv_data, dev_data, test_data)

        self.train_input_data = train_data
        self.cv_input_data = cv_data
        self.dev_input_data = dev_data
        self.test_input_data = test_data
        self.align_input_data = align_data

        self.train_corpora.extend(list(train_data.keys()))
        self.cv_corpora.extend(list(cv_data.keys()))
        self.dev_corpora.extend(list(dev_data.keys()))
        self.test_corpora.extend(list(test_data.keys()))
        self.align_corpora.extend(list(align_data.keys()))

        self.jobs["general"] = {}

        self._set_data(train_data)
        self._set_data(cv_data)
        self._set_data(align_data)
        self._set_eval_data(dev_data)
        self._set_eval_data(test_data)

        self.train_cv_pairing = (
            list(itertools.product(self.train_corpora, self.cv_corpora))
            if train_cv_pairing is None
            else train_cv_pairing
        )

        self.train_cv_align_pairing = train_cv_align_pairing or [
            (trn_cv[0], trn_cv[1], al_c)
            for trn_cv, al_c in itertools.product(
                self.train_cv_pairing, self.align_corpora
            )
        ]

        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]

            self.jobs[f"{trn_c}_{cv_c}"] = {}

        for pairing in self.train_cv_align_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            al_c = pairing[1]

            self.jobs[f"{trn_c}_{cv_c}_{al_c}"] = {}

    def _set_data(self, data_dict: Dict) -> None:
        for c_key, c_data in data_dict.items():
            self.jobs[c_key] = {}
            self.crp[c_key] = c_data.get_crp() if c_data.crp is None else c_data.crp
            self.feature_flows[c_key] = c_data.feature_flow

    def _set_eval_data(self, data_dict: Dict) -> None:
        self._set_data(data_dict)
        for c_key in data_dict:
            self.ctm_files[c_key] = {}

    # -------------------- Training --------------------

    def returnn_rasr_training(
        self,
        name: str,
        returnn_config: ReturnnConfig,
        train_corpus_key: str,
        cv_corpus_key: str,
        use_rasr_ctc_loss: bool = False,
        rasr_ctc_loss_args: Optional[Dict] = None,
        **kwargs,
    ) -> ReturnnRasrTrainingJob:
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        feature_flow = self.get_feature_flow_for_corpus(train_corpus_key)
        alignments = self.get_alignments_for_corpus(train_corpus_key)
        dev_feature_flow = None
        dev_alignments = None
        if (
            dev_data.feature_flow != train_data.feature_flow
            or dev_data.features != train_data.features
            or dev_data.alignments != train_data.alignments
        ):
            dev_feature_flow = self.get_feature_flow_for_corpus(cv_corpus_key)
            dev_alignments = self.get_alignments_for_corpus(cv_corpus_key)

        assert isinstance(returnn_config, ReturnnConfig)

        if use_rasr_ctc_loss:
            loss_config, loss_post_config = make_ctc_rasr_loss_config(
                train_corpus_path=train_crp.corpus_config.file.get_path(),
                dev_corpus_path=dev_crp.corpus_config.file.get_path(),
                base_crp=dev_crp,
                **(rasr_ctc_loss_args or {}),
            )

            kwargs.update(
                {
                    "use_python_control": True,
                    "additional_rasr_config_files": {"rasr.loss": loss_config},
                    "additional_rasr_post_config_files": {
                        "rasr.loss": loss_post_config
                    },
                }
            )

        train_job = self.jobs[f"{train_corpus_key}_{cv_corpus_key}"].setdefault(
            name,
            ReturnnRasrTrainingJob(
                train_crp=train_crp,
                dev_crp=dev_crp,
                feature_flow=feature_flow,
                alignment=alignments,
                dev_feature_flow=dev_feature_flow,
                dev_alignment=dev_alignments,
                returnn_config=returnn_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                **kwargs,
            ),
        )
        self._add_output_alias_for_train_job(
            train_job=train_job,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            name=name,
        )

        return train_job

    # -------------------- Priors --------------------

    def returnn_rasr_compute_priors(
        self,
        name: str,
        returnn_config: ReturnnConfig,
        train_corpus_key: str,
        cv_corpus_key: str,
        checkpoint: Checkpoint,
        nn_prior_args: Optional[Dict] = None,
    ) -> tk.Path:
        nn_prior_args = nn_prior_args or {}
        train_data = self.train_input_data[train_corpus_key]
        dev_data = self.cv_input_data[cv_corpus_key]

        train_crp = train_data.get_crp()
        dev_crp = dev_data.get_crp()

        feature_flow = self.get_feature_flow_for_corpus(train_corpus_key)
        alignments = self.get_alignments_for_corpus(train_corpus_key)
        dev_feature_flow = None
        dev_alignments = None
        if (
            dev_data.feature_flow != train_data.feature_flow
            or dev_data.features != train_data.features
            or dev_data.alignments != train_data.alignments
        ):
            dev_feature_flow = self.get_feature_flow_for_corpus(cv_corpus_key)
            dev_alignments = self.get_alignments_for_corpus(cv_corpus_key)

        assert isinstance(returnn_config, ReturnnConfig)

        prior_job = self.jobs[f"{train_corpus_key}_{cv_corpus_key}"].setdefault(
            name,
            ReturnnRasrComputePriorJob(
                train_crp=train_crp,
                dev_crp=dev_crp,
                feature_flow=feature_flow,
                model_checkpoint=checkpoint,
                returnn_config=returnn_config,
                alignment=alignments,
                dev_feature_flow=dev_feature_flow,
                dev_alignment=dev_alignments,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                **nn_prior_args,
            ),
        )

        return prior_job.out_prior_xml_file

    # -------------------- Recognition --------------------
    def returnn_bpe_recognition(
        self,
        name: str,
        recognition_corpus_key: str,
        recog_returnn_config: ReturnnConfig,
        checkpoints: Dict[int, Checkpoint],
        epochs: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        with tk.block(name):
            recog_crp = self.crp[recognition_corpus_key]
            feature_flow = self.get_feature_flow_for_corpus(recognition_corpus_key)

            epochs = epochs if epochs is not None else list(checkpoints.values())
            for epoch in epochs:
                tf_checkpoint = checkpoints[epoch]

                exp_name = f"{recognition_corpus_key}-e{epoch:03d}"
                rec = self.jobs[recognition_corpus_key].setdefault(
                    f"recog_{exp_name}",
                    ReturnnRasrSearchJob(
                        crp=recog_crp,
                        feature_flow=feature_flow,
                        model_checkpoint=tf_checkpoint,
                        returnn_config=recog_returnn_config,
                        **kwargs,
                    ),
                )

                prefix = f"nn_recog/{name}/"
                rec.set_vis_name("Recog %s%s" % (prefix, exp_name))
                rec.add_alias("%srecog_%s" % (prefix, exp_name))

                word_output = self.jobs[recognition_corpus_key].setdefault(
                    f"word_{exp_name}", SearchBPEtoWordsJob(rec.out_search_file)
                )
                word2ctm = self.jobs[recognition_corpus_key].setdefault(
                    f"word2ctm_{exp_name}",
                    SearchWordsToCTMJob(
                        word_output.out_word_search_results,
                        recog_crp.corpus_config.file,
                    ),
                )

                scorer_kwargs = copy.deepcopy(self.scorer_args[recognition_corpus_key])
                scorer_kwargs[
                    self.scorer_hyp_arg[recognition_corpus_key]
                ] = word2ctm.out_ctm_file
                scorer = self.jobs[recognition_corpus_key].setdefault(
                    f"scorer_{exp_name}",
                    self.scorers[recognition_corpus_key](**scorer_kwargs),
                )
                tk.register_output(
                    "%srecog_%s.reports" % (prefix, exp_name), scorer.out_report_dir
                )

    def nn_recognition(
        self,
        name: str,
        base_key: str,
        recognition_corpus_key: str,
        train_returnn_config: ReturnnConfig,
        recog_returnn_config: ReturnnConfig,
        checkpoints: Dict[int, Checkpoint],
        lookahead_options: Dict,
        lm_scales: Optional[List[float]] = None,
        prior_scales: Optional[List[float]] = None,
        train_corpus_key: Optional[str] = None,
        cv_corpus_key: Optional[str] = None,
        prior_args: Optional[Dict] = None,
        lattice_to_ctm_kwargs: Dict = {},
        epochs: Optional[List[int]] = None,
        label_unit: str = "phoneme",
        label_file_blank: bool = True,
        add_eow: bool = True,
        add_sow: bool = False,
        recog_lexicon: Optional[tk.Path] = None,
        label_tree_args: Dict = {},
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        **kwargs,
    ) -> None:
        modified_label_scorer_args = copy.deepcopy(label_scorer_args)
        modified_label_tree_args = copy.deepcopy(label_tree_args)
        with tk.block(name):
            recog_crp = self.crp[recognition_corpus_key]

            tf_graph = self.make_tf_graph(
                base_key, recog_returnn_config, label_scorer_type
            )

            if "lexicon_config" not in label_tree_args:
                if not recog_lexicon:
                    recog_lexicon = self.crp[recognition_corpus_key].lexicon_config.file
                    if label_unit == "phoneme" and add_eow or add_sow:
                        recog_lexicon = (
                            self.jobs[recognition_corpus_key]
                            .setdefault(
                                "boundary_lex",
                                AddBoundaryMarkerToLexiconJob(
                                    recog_lexicon,
                                    add_eow,
                                    add_sow,
                                ),
                            )
                            .out_lexicon
                        )

                modified_label_tree_args["lexicon_config"] = {
                    "filename": recog_lexicon,
                    "normalize_pronunciation": self.crp[
                        recognition_corpus_key
                    ].lexicon_config.normalize_pronunciation,
                }

            label_tree = LabelTree(label_unit, **modified_label_tree_args)

            # add vocab file
            if label_unit == "phoneme" and "label_file" not in label_scorer_args:
                state_tying_file = (
                    self.jobs[recognition_corpus_key]
                    .setdefault("dump_state_tying", DumpStateTyingJob(recog_crp))
                    .out_state_tying
                )
                modified_label_scorer_args["label_file"] = (
                    self.jobs[recognition_corpus_key]
                    .setdefault(
                        "gen_label_file",
                        GenerateLabelFileFromStateTyingJob(
                            state_tying_file,
                            use_blank=label_file_blank,
                            add_eow=add_eow,
                            add_sow=add_sow,
                        ),
                    )
                    .out_label_file
                )

            base_feature_flow = self.feature_flows[recognition_corpus_key]
            assert isinstance(
                base_feature_flow, rasr.FlowNetwork
            ), f"type incorrect: {recognition_corpus_key} {type(base_feature_flow)}"

            epochs = epochs if epochs is not None else list(checkpoints.keys())

            if not prior_scales or not label_scorer_args.get("use_prior", False):
                prior_scales = [0]

            if not lm_scales:
                lm_scales = [0]

            for lm, prior, epoch in itertools.product(lm_scales, prior_scales, epochs):
                recog_crp.language_model_config.scale = lm
                assert epoch in checkpoints.keys()

                tf_checkpoint = checkpoints[epoch]

                if label_scorer_args.get("use_prior", False) and prior != 0:
                    modified_label_scorer_args["prior_file"] = self.get_prior_file(
                        base_key,
                        epoch,
                        label_scorer_args,
                        train_corpus_key,
                        cv_corpus_key,
                        train_returnn_config,
                        tf_checkpoint,
                        prior_args,
                    )

                modified_label_scorer_args["prior_scale"] = prior

                label_scorer = LabelScorer(
                    label_scorer_type, **modified_label_scorer_args
                )

                if label_scorer.need_tf_flow(label_scorer_type):
                    feature_flow = self.make_tf_feature_flow(
                        base_feature_flow, tf_graph, tf_checkpoint
                    )
                else:
                    feature_flow = base_feature_flow
                    label_scorer.set_input_config()
                    label_scorer.set_loader_config(
                        self.make_model_loader_config(tf_graph, tf_checkpoint)
                    )

                lookahead_options["scale"] = lm

                exp_name = f"{recognition_corpus_key}-e{epoch:03d}-prior{prior:02.2f}-lm{lm:02.2f}"
                rec = self.jobs[recognition_corpus_key].setdefault(
                    f"recog_{exp_name}",
                    LabelSyncSearchJob(
                        crp=recog_crp,
                        feature_flow=feature_flow,
                        label_scorer=label_scorer,
                        label_tree=label_tree,
                        lookahead_options=lookahead_options,
                        **kwargs,
                    ),
                )

                prefix = f"nn_recog/{name}/"
                rec.set_vis_name("Recog %s%s" % (prefix, exp_name))
                rec.add_alias("%srecog_%s" % (prefix, exp_name))

                lat2ctm = self.jobs[recognition_corpus_key].setdefault(
                    f"lat2ctm_{exp_name}",
                    recog.LatticeToCtmJob(
                        crp=recog_crp,
                        lattice_cache=rec.out_lattice_bundle,
                        **lattice_to_ctm_kwargs,
                    ),
                )
                self.ctm_files[recognition_corpus_key][
                    f"recog_{exp_name}"
                ] = lat2ctm.out_ctm_file

                scorer_kwargs = copy.deepcopy(self.scorer_args[recognition_corpus_key])
                scorer_kwargs[
                    self.scorer_hyp_arg[recognition_corpus_key]
                ] = lat2ctm.out_ctm_file
                scorer = self.jobs[recognition_corpus_key].setdefault(
                    f"scorer_{exp_name}",
                    self.scorers[recognition_corpus_key](**scorer_kwargs),
                )
                tk.register_output(
                    "%srecog_%s.reports" % (prefix, exp_name), scorer.out_report_dir
                )

    def nn_alignment(
        self,
        name: str,
        base_key: str,
        corpus_key: str,
        returnn_config: ReturnnConfig,
        checkpoints: Dict[int, Checkpoint],
        prior_scales: Optional[List[float]] = None,
        epochs: Optional[List[int]] = None,
        label_unit: str = "phoneme",
        label_file_blank: bool = True,
        add_eow: bool = True,
        add_sow: bool = False,
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        alignment_options: Dict = {},
        **kwargs,
    ) -> None:
        modified_label_scorer_args = copy.deepcopy(label_scorer_args)
        with tk.block(name):
            crp = self.crp[corpus_key]
            tf_graph = self.make_tf_graph(base_key, returnn_config, label_scorer_type)

            # add vocab file
            if label_unit == "phoneme" and "label_file" not in label_scorer_args:
                state_tying_file = (
                    self.jobs[corpus_key]
                    .setdefault("dump_state_tying", DumpStateTyingJob(crp))
                    .out_state_tying
                )
                modified_label_scorer_args["label_file"] = (
                    self.jobs[corpus_key]
                    .setdefault(
                        "gen_label_file",
                        GenerateLabelFileFromStateTyingJob(
                            state_tying_file,
                            use_blank=label_file_blank,
                            add_eow=add_eow,
                            add_sow=add_sow,
                        ),
                    )
                    .out_label_file
                )

            base_feature_flow = self.get_feature_flow_for_corpus(corpus_key)

            epochs = epochs if epochs is not None else list(checkpoints.keys())

            if not prior_scales or not label_scorer_args.get("use_prior", False):
                prior_scales = [0]

            for prior, epoch in itertools.product(prior_scales, epochs):
                assert epoch in checkpoints.keys()

                tf_checkpoint = checkpoints[epoch]

                if label_scorer_args.get("use_prior", False) and prior != 0:
                    modified_label_scorer_args["prior_file"] = self.get_prior_file(
                        base_key,
                        epoch,
                        label_scorer_args,
                    )
                modified_label_scorer_args["prior_scale"] = prior

                label_scorer = LabelScorer(
                    label_scorer_type, **modified_label_scorer_args
                )

                if label_scorer.need_tf_flow(label_scorer_type):
                    feature_flow = self.make_tf_feature_flow(
                        base_feature_flow, tf_graph, tf_checkpoint
                    )
                else:
                    feature_flow = base_feature_flow
                    label_scorer.set_input_config()
                    label_scorer.set_loader_config(
                        self.make_model_loader_config(tf_graph, tf_checkpoint)
                    )

                exp_name = f"{corpus_key}-e{epoch:03d}"
                align_job = self.jobs[corpus_key].setdefault(
                    f"align_{exp_name}",
                    LabelAlignmentJob(
                        crp=crp,
                        feature_flow=feature_flow,
                        label_scorer=label_scorer,
                        alignment_options=alignment_options,
                        **kwargs,
                    ),
                )
                prefix = f"nn_align/{name}/"

                align_job.set_vis_name("Alignment %s%s" % (prefix, exp_name))
                align_job.add_alias("%salign_%s" % (prefix, exp_name))
                tk.register_output(
                    "%salign_%s.cache.bundle" % (prefix, exp_name),
                    align_job.out_alignment_bundle,
                )
                self.alignments[corpus_key] = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": align_job.out_alignment_path,
                        "bundle": align_job.out_alignment_bundle,
                    },
                )

    # -------------------- run functions  --------------------

    def run_bpe_nn_step(self, step_args: HybridArgs) -> None:
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = (
                [pairing[2]]
                if len(pairing) >= 3
                else list(step_args.returnn_training_configs.keys())
            )

            for name in name_list:
                returnn_train_job = self.returnn_rasr_training(
                    name=name,
                    returnn_config=step_args.returnn_training_configs[name],
                    train_corpus_key=trn_c,
                    cv_corpus_key=cv_c,
                    **(step_args.training_args or {}),
                )

                recog_returnn_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                for recog_name, recog_args in step_args.recognition_args.items():
                    for dev_c in self.dev_corpora:
                        self.returnn_bpe_recognition(
                            name=f"{trn_c}-{name}-{recog_name}",
                            recognition_corpus_key=dev_c,
                            recog_returnn_config=recog_returnn_config,
                            checkpoints=returnn_train_job.out_checkpoints,
                            **recog_args,
                        )

    def run_bpe_recog_step(self, step_args: HybridArgs) -> None:
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = (
                [pairing[2]]
                if len(pairing) >= 3
                else list(step_args.returnn_training_configs.keys())
            )

            for name in name_list:
                recog_returnn_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                for recog_name, recog_args in step_args.test_recognition_args.items():
                    for test_c in self.test_corpora:
                        self.returnn_bpe_recognition(
                            name=f"{trn_c}-{name}-{recog_name}",
                            recognition_corpus_key=test_c,
                            recog_returnn_config=recog_returnn_config,
                            checkpoints=self.tf_nn_checkpoints[f"{trn_c}_{cv_c}"],
                            **recog_args,
                        )

    def run_nn_step(self, step_args: HybridArgs) -> None:
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = (
                [pairing[2]]
                if len(pairing) >= 3
                else list(step_args.returnn_training_configs.keys())
            )

            for name in name_list:
                returnn_train_job = self.returnn_rasr_training(
                    name=name,
                    returnn_config=step_args.returnn_training_configs[name],
                    train_corpus_key=trn_c,
                    cv_corpus_key=cv_c,
                    **(step_args.training_args or {}),
                )

                recog_returnn_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                for recog_name, recog_args in step_args.recognition_args.items():
                    for dev_c in self.dev_corpora:
                        self.nn_recognition(
                            name=f"{trn_c}-{name}-{recog_name}",
                            base_key=f"{trn_c}-{name}",
                            train_corpus_key=trn_c,
                            cv_corpus_key=cv_c,
                            recognition_corpus_key=dev_c,
                            train_returnn_config=step_args.returnn_training_configs[
                                name
                            ],
                            recog_returnn_config=recog_returnn_config,
                            checkpoints=returnn_train_job.out_checkpoints,
                            prior_args=step_args.prior_args,
                            **recog_args,
                        )

    def run_nn_recog_step(self, step_args: HybridArgs) -> None:
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = (
                [pairing[2]]
                if len(pairing) >= 3
                else list(step_args.returnn_training_configs.keys())
            )

            for name in name_list:
                recog_returnn_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                for recog_name, recog_args in step_args.test_recognition_args.items():
                    for test_c in self.test_corpora:
                        self.nn_recognition(
                            name=f"{trn_c}-{name}-{recog_name}",
                            base_key=f"{trn_c}-{name}",
                            recognition_corpus_key=test_c,
                            train_returnn_config=step_args.returnn_training_configs[
                                name
                            ],
                            recog_returnn_config=recog_returnn_config,
                            checkpoints=self.tf_nn_checkpoints[f"{trn_c}_{cv_c}"],
                            prior_args=step_args.prior_args,
                            **recog_args,
                        )

    def run_realign_step(self, step_args: HybridArgs) -> None:
        for pairing in self.train_cv_align_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            al_c = pairing[2]
            name_list = (
                [pairing[3]]
                if len(pairing) >= 4
                else list(step_args.returnn_training_configs.keys())
            )

            for name in name_list:
                recog_returnn_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )
                for align_name, align_args in step_args.alignment_args.items():
                    self.nn_alignment(
                        name=f"{trn_c}-{name}-{align_name}",
                        base_key=f"{trn_c}-{name}",
                        corpus_key=al_c,
                        returnn_config=recog_returnn_config,
                        checkpoints=self.tf_nn_checkpoints[f"{trn_c}_{cv_c}"],
                        **align_args,
                    )

    def run(self, steps: RasrSteps) -> None:
        if "init" in steps.get_step_names_as_list():
            print(
                "init needs to be run manually. provide: gmm_args, {train,dev,test}_inputs"
            )
            sys.exit(-1)

        for trn_c in self.train_corpora:
            self.store_allophones(trn_c)
            tk.register_output(
                f"allophones/{trn_c}/allophones", self.allophone_files["base"]
            )

        for eval_c in self.dev_corpora + self.test_corpora:
            if eval_c not in self.stm_files:
                stm_args = (
                    self.hybrid_init_args.stm_args
                    if self.hybrid_init_args.stm_args is not None
                    else {}
                )
                self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

        for _, (step_name, step_args) in enumerate(steps.get_step_iter()):
            # ---------- Feature Extraction ----------
            if step_name.startswith("extract"):
                if step_args is None:
                    step_args = self.hybrid_init_args.feature_extraction_args
                for all_c in (
                    self.train_corpora
                    + self.cv_corpora
                    + self.dev_corpora
                    + self.test_corpora
                    + self.align_corpora
                ):
                    self.feature_caches[all_c] = {}
                    self.feature_bundles[all_c] = {}
                    self.feature_flows[all_c] = {}
                self.extract_features(step_args)

            # ---------- NN Training ----------
            if step_name.startswith("nn"):
                self.run_nn_step(step_args)
            if step_name.startswith("nn_recog"):
                self.run_nn_recog_step(step_args)
            if step_name.startswith("bpe_nn"):
                self.run_bpe_nn_step(step_args)
            if step_name.startswith("bpe_recog"):
                self.run_bpe_recog_step(step_args)
            if step_name.startswith("realign"):
                self.run_realign_step(step_args)
