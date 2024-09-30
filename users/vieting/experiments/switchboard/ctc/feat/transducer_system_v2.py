# -------------------- General --------------------

import copy
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type, Union, Any

# -------------------- Recipes --------------------

from i6_core import am, corpus, features, lexicon, mm, rasr, recognition, returnn
from i6_experiments.common.setups.rasr.util.rasr import RasrDataInput
from i6_experiments.users.berger.recipe import mm as custom_mm
from i6_experiments.users.berger.recipe.rasr import GenerateLabelFileFromStateTyingJobV2
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.recipe import rasr as custom_rasr
from i6_experiments.users.berger.recipe import recognition as custom_recognition
from i6_experiments.users.berger.util import lru_cache_with_signature
from i6_experiments.users.vieting.tools.report import Report
from .recipe import returnn as custom_returnn


# -------------------- Sisyphus --------------------

from sisyphus import Job, tk


# -------------------- Init ------------------------

Path = tk.setup_path(__package__)


# ----------------- Helper classes -----------------
# Either returnn config or a function that creates a config out of an optuna trial object
ReturnnConfigType = Union[returnn.ReturnnConfig, custom_returnn.OptunaReturnnConfig]
TrainJobType = Union[returnn.ReturnnTrainingJob, custom_returnn.OptunaReturnnTrainingJob]
RecogJobType = Union[recognition.AdvancedTreeSearchJob, custom_recognition.GenericSeq2SeqSearchJob]
ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]


class ConfigType(Enum):
    TRAIN = auto()
    PRIOR = auto()
    ALIGN = auto()
    RECOG = auto()


@dataclass
class ReturnnConfigs:
    train_config: ReturnnConfigType
    prior_config: ReturnnConfigType = None  # type: ignore
    align_config: ReturnnConfigType = None  # type: ignore
    recog_configs: Dict[str, ReturnnConfigType] = None  # type: ignore

    def __post_init__(self):
        if self.prior_config is None:
            self.prior_config = copy.deepcopy(self.train_config)
        if self.recog_configs is None:
            self.recog_configs = {"recog": copy.deepcopy(self.prior_config)}
        if self.align_config is None:
            self.align_config = copy.deepcopy(next(iter(self.recog_configs.values())))


@dataclass
class ScorerInfo:
    ref_file: Optional[tk.Path] = None
    job_type: ScoreJobType = recognition.ScliteJob
    score_kwargs: Dict = field(default_factory=dict)

    def get_score_job(self, ctm: tk.Path) -> ScoreJob:
        assert self.ref_file is not None
        return self.job_type(hyp=ctm, ref=self.ref_file, **self.score_kwargs)


class SearchTypes(Enum):
    AdvancedTreeSearch = auto()
    GenericSeq2SeqSearchJob = auto()
    ReturnnSearch = auto()


# -------------------- System --------------------


class TransducerSystem:
    """
    - 3 corpora types: dev, test and align
    - dev corpora will be recognized during "train" or "tune" step
    - test corpora will be recognized during "recog" step
    - align corpora will be aligned during "align"

    - train and cv corpora do not have to be specified, they should be contained in the returnn config

    to create beforehand:
    - returnn configs
    - corpora: name and RasrDataInput (including lexicon, lm)
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        returnn_root: tk.Path,
        returnn_python_exe: tk.Path,
        rasr_python_home: Optional[tk.Path] = None,
        rasr_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
        require_native_lstm: bool = True,
    ) -> None:
        self.rasr_binary_path = rasr_binary_path
        self.returnn_root = returnn_root
        self.returnn_python_exe = returnn_python_exe
        self.blas_lib = blas_lib
        self._require_native_lstm = require_native_lstm

        # Build base crp
        self.base_crp = rasr.CommonRasrParameters()
        self.base_crp.python_home = rasr_python_home  # type: ignore
        self.base_crp.python_program_name = rasr_python_exe or returnn_python_exe  # type: ignore
        rasr.crp_add_default_output(self.base_crp)
        self.base_crp.set_executables(rasr_binary_path=self.rasr_binary_path)

        # exp-name mapped to ReturnnConfigs collection
        self.returnn_configs: Dict[str, ReturnnConfigs] = {}

        # exp-name mapped to train/recog job
        self.train_jobs: Dict[str, TrainJobType] = {}
        self.recog_jobs: Dict[str, RecogJobType] = {}

        # exp-name mapped to Checkpoint object
        self.best_checkpoints: Dict[str, returnn.Checkpoint] = {}

        # lists of corpus keys
        self.align_corpora: List[str] = []
        self.dev_corpora: List[str] = []
        self.test_corpora: List[str] = []

        # corpus-key mapped to RasrDataInput object
        self.corpus_data: Dict[str, RasrDataInput] = {}

        # corpus-key to CommonRasrParameters object
        self.crp: Dict[str, rasr.CommonRasrParameters] = {}

        # corpus-key to ScorerInfo
        self.scorers: Dict[str, ScorerInfo] = {}

        self._report: Optional[Report] = None

        self.alignments: Dict[str, rasr.FlagDependentFlowAttribute] = {}

    # -------------------- Setup --------------------
    def init_system(
        self,
        returnn_configs: Dict[str, ReturnnConfigs],
        align_keys: List[str] = [],
        dev_keys: List[str] = [],
        test_keys: List[str] = [],
        corpus_data: Dict[str, RasrDataInput] = {},
        am_args: Dict = {},
        scorer_info: Optional[ScorerInfo] = None,
        report: Optional[Report] = None,
    ) -> None:

        self.returnn_configs = returnn_configs

        all_keys = set(align_keys + dev_keys + test_keys)
        assert all(key in corpus_data for key in all_keys)

        self.align_corpora = align_keys
        self.dev_corpora = dev_keys
        self.test_corpora = test_keys

        self.corpus_data = corpus_data

        for key in all_keys:
            self.crp[key] = self.get_crp(corpus_data[key], am_args)

        for key in set(dev_keys + test_keys):
            self._set_scorer(key, scorer_info or ScorerInfo())

        self._report = report

    @property
    def report(self) -> Optional[Report]:
        return self._report

    # -------------------- Helpers ---------------------

    def get_crp(self, data: RasrDataInput, am_args: Dict = {}) -> rasr.CommonRasrParameters:
        crp = rasr.CommonRasrParameters(self.base_crp)

        rasr.crp_set_corpus(crp, data.corpus_object)
        crp.concurrent = data.concurrent
        crp.segment_path = corpus.SegmentCorpusJob(  # type: ignore
            data.corpus_object.corpus_file, data.concurrent
        ).out_segment_path

        crp.language_model_config = rasr.RasrConfig()  # type: ignore
        crp.language_model_config.type = data.lm["type"]  # type: ignore
        crp.language_model_config.file = data.lm["filename"]  # type: ignore

        crp.lexicon_config = rasr.RasrConfig()  # type: ignore
        crp.lexicon_config.file = data.lexicon["filename"]
        crp.lexicon_config.normalize_pronunciation = data.lexicon["normalize_pronunciation"]

        crp.acoustic_model_config = am.acoustic_model_config(**am_args)  # type: ignore
        crp.acoustic_model_config.allophones.add_all = data.lexicon["add_all"]  # type: ignore
        crp.acoustic_model_config.allophones.add_from_lexicon = data.lexicon["add_from_lexicon"]  # type: ignore

        return crp

    @lru_cache_with_signature
    def _generate_stm_for_corpus(self, key: str, **kwargs) -> None:
        stm_path = corpus.CorpusToStmJob(self.corpus_data[key].corpus_object.corpus_file, **kwargs).out_stm_path

        self.scorers[key].ref_file = stm_path

    def _set_scorer(self, key: str, scorer_info: ScorerInfo) -> None:
        self.scorers[key] = scorer_info

        if self.scorers[key].ref_file is None:
            self._generate_stm_for_corpus(key)

    @lru_cache_with_signature
    def _get_native_op(self) -> Optional[tk.Path]:
        if not self._require_native_lstm:
            return None

        # DO NOT USE BLAS ON I6, THIS WILL SLOW DOWN RECOGNITION ON OPTERON MACHNIES BY FACTOR 4
        compile_job = returnn.CompileNativeOpJob(
            "NativeLstm2",
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            blas_lib=self.blas_lib,
        )

        return compile_job.out_op

    def _make_model_loader_config(self, tf_graph: tk.Path, tf_checkpoint: returnn.Checkpoint) -> rasr.RasrConfig:
        loader_config = rasr.RasrConfig()
        loader_config.type = "meta"
        loader_config.meta_graph_file = tf_graph
        loader_config.saved_model_file = tf_checkpoint
        loader_config.required_libraries = self._get_native_op()
        return loader_config

    def _get_prior_file(
        self,
        train_exp_name: str,
        epoch: EpochType,
        prior_args: Dict = {},
        trial_num: Optional[int] = None,
        use_txt_prior: bool = False,
    ) -> tk.Path:
        checkpoint = self._get_checkpoint(train_exp_name, epoch, trial_num)

        prior_config = self.returnn_configs[train_exp_name].prior_config
        if isinstance(prior_config, returnn.ReturnnConfig):
            prior_job = returnn.ReturnnComputePriorJobV2(
                model_checkpoint=checkpoint,
                returnn_config=prior_config,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                **prior_args,
            )
        else:
            train_job = self.train_jobs[train_exp_name]
            assert isinstance(train_job, custom_returnn.OptunaReturnnTrainingJob)
            if trial_num is None:
                trial = train_job.out_best_trial
            else:
                trial = train_job.out_trials[trial_num]
            prior_job = custom_returnn.OptunaReturnnComputePriorJob(
                optuna_returnn_config=prior_config,
                trial=trial,
                model_checkpoint=checkpoint,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                **prior_args,
            )

        if use_txt_prior:
            return prior_job.out_prior_txt_file
        else:
            return prior_job.out_prior_xml_file

    @staticmethod
    def _is_autoregressive_decoding(label_scorer_type: str) -> bool:
        return label_scorer_type != "precomputed-log-posterior"

    def _make_tf_graph(
        self,
        train_exp_name: str,
        returnn_config: ReturnnConfigType,
        epoch: EpochType,
        label_scorer_type: str = "precomputed-log-posterior",
        trial_num: Optional[int] = None,
    ) -> tk.Path:
        train_job = self.train_jobs[train_exp_name]
        rec_step_by_step = "output" if self._is_autoregressive_decoding(label_scorer_type) else None
        if isinstance(returnn_config, returnn.ReturnnConfig):
            assert isinstance(train_job, returnn.ReturnnTrainingJob)
            graph_compile_job = returnn.CompileTFGraphJob(
                returnn_config,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                epoch=self._get_epoch_value(train_exp_name, epoch, trial_num),
                rec_step_by_step=rec_step_by_step,
                rec_json_info=bool(rec_step_by_step),
            )
        else:
            assert isinstance(train_job, custom_returnn.OptunaReturnnTrainingJob)
            if trial_num is None:
                trial = train_job.out_best_trial
            else:
                trial = train_job.out_trials[trial_num]
            graph_compile_job = custom_returnn.OptunaCompileTFGraphJob(
                optuna_returnn_config=returnn_config,
                trial=trial,
                returnn_root=self.returnn_root,
                returnn_python_exe=self.returnn_python_exe,
                epoch=self._get_epoch_value(train_exp_name, epoch, trial_num),
                rec_step_by_step=rec_step_by_step,
                rec_json_info=bool(rec_step_by_step),
            )
        return graph_compile_job.out_graph

    def _make_base_feature_flow(self, corpus_key: str, **kwargs):
        if kwargs.pop("type", "") == "gammatone":
            return features.gammatone_flow(**kwargs)
        audio_format = self.corpus_data[corpus_key].corpus_object.audio_format
        args = {
            "audio_format": audio_format,
            "dc_detection": False,
            "input_options": {"block-size": 1},
            "scale_input": 2**-15,
        }
        args.update(kwargs)
        return features.samples_flow(**args)

    def _make_tf_feature_flow(
        self,
        base_flow: rasr.FlowNetwork,
        tf_graph: tk.Path,
        tf_checkpoint: returnn.Checkpoint,
        output_layer_name: str = "output",
    ) -> rasr.FlowNetwork:
        # tf flow (model scoring done in tf flow node) #
        input_name = "tf-fwd_input"

        tf_flow = rasr.FlowNetwork()
        tf_flow.add_input(input_name)
        tf_flow.add_output("features")
        tf_flow.add_param("id")

        tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link(f"network:{input_name}", f"{tf_fwd}:input")
        tf_flow.link(f"{tf_fwd}:log-posteriors", "network:features")

        tf_flow.config = rasr.RasrConfig()  # type: ignore
        tf_flow.config[tf_fwd].input_map.info_0.param_name = "input"  # type: ignore
        tf_flow.config[tf_fwd].input_map.info_0.tensor_name = "extern_data/placeholders/data/data"  # type: ignore
        tf_flow.config[
            tf_fwd
        ].input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"  # type: ignore

        tf_flow.config[tf_fwd].output_map.info_0.param_name = "log-posteriors"  # type: ignore
        tf_flow.config[tf_fwd].output_map.info_0.tensor_name = f"{output_layer_name}/output_batch_major"  # type: ignore

        tf_flow.config[tf_fwd].loader.type = "meta"  # type: ignore
        tf_flow.config[tf_fwd].loader.meta_graph_file = tf_graph  # type: ignore
        tf_flow.config[tf_fwd].loader.saved_model_file = tf_checkpoint  # type: ignore

        tf_flow.config[tf_fwd].loader.required_libraries = self._get_native_op()  # type: ignore

        # interconnect flows #
        ext_flow = rasr.FlowNetwork()
        base_mapping = ext_flow.add_net(base_flow)
        tf_mapping = ext_flow.add_net(tf_flow)
        ext_flow.interconnect_inputs(base_flow, base_mapping)
        ext_flow.interconnect(
            base_flow,
            base_mapping,
            tf_flow,
            tf_mapping,
            {list(base_flow.outputs)[0]: input_name},
        )

        ext_flow.interconnect_outputs(tf_flow, tf_mapping)
        # ensure cache_mode as base feature net
        ext_flow.add_flags(base_flow.flags)
        return ext_flow

    @lru_cache_with_signature
    def _get_epoch_value(
        self, train_exp_name: str, epoch: EpochType, trial_num: Optional[int] = None
    ) -> Union[float, tk.Variable]:
        if epoch != "best":
            return epoch

        train_job = self.train_jobs[train_exp_name]
        if isinstance(train_job, returnn.ReturnnTrainingJob) or trial_num is None:
            lr = train_job.out_learning_rates
        else:
            lr = train_job.out_trial_learning_rates[trial_num]
        return custom_returnn.GetBestEpochJob(lr).out_epoch

    @lru_cache_with_signature
    def _get_checkpoint(
        self,
        train_exp_name: str,
        epoch: EpochType,
        trial_num: Optional[int] = None,
    ) -> returnn.Checkpoint:
        train_job = self.train_jobs[train_exp_name]

        if isinstance(train_job, returnn.ReturnnTrainingJob) or trial_num is None:
            if epoch == "best":
                return custom_returnn.GetBestCheckpointJob(
                    train_job.out_model_dir, train_job.out_learning_rates
                ).out_checkpoint
            return train_job.out_checkpoints[epoch]
        else:
            if epoch == "best":
                return custom_returnn.GetBestCheckpointJob(
                    train_job.out_trial_model_dir[trial_num],
                    train_job.out_trial_learning_rates[trial_num],
                ).out_checkpoint
            return train_job.out_trial_checkpoints[trial_num][epoch]

    @lru_cache_with_signature
    def _get_trial_value(self, train_exp_name: str, trial_num: Optional[int] = None) -> Union[int, tk.Variable]:
        if trial_num is not None:
            return trial_num

        train_job = self.train_jobs[train_exp_name]
        if isinstance(train_job, returnn.ReturnnTrainingJob):
            return 0
        return train_job.out_best_trial

    # -------------------- Main behaviour functions --------------------

    def returnn_training(
        self,
        train_exp_name: str,
        **kwargs,
    ) -> TrainJobType:

        returnn_config = self.returnn_configs[train_exp_name].train_config
        if isinstance(returnn_config, returnn.ReturnnConfig):
            train_job = returnn.ReturnnTrainingJob(
                returnn_config=returnn_config,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                **kwargs,
            )
        else:
            assert isinstance(returnn_config, tuple)
            train_job = custom_returnn.OptunaReturnnTrainingJob(
                optuna_returnn_config=returnn_config,
                study_name=train_exp_name,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                **kwargs,
            )
        self._register_train_job_output(train_exp_name=train_exp_name, train_job=train_job)

        return train_job

    def _register_train_job_output(
        self,
        train_exp_name: str,
        train_job: TrainJobType,
    ) -> None:
        self.train_jobs[train_exp_name] = train_job

        train_job.add_alias(f"train_nn/{train_exp_name}")
        tk.register_output(f"train_nn/{train_exp_name}/learning_rate.png", train_job.out_plot_lr)

    # -------------------- Recognition --------------------

    def _lattice_scoring(
        self,
        filename: str,
        recognition_corpus_key: str,
        recognition_job: Union[recognition.AdvancedTreeSearchJob, custom_recognition.GenericSeq2SeqSearchJob],
        **kwargs,
    ) -> ScoreJob:

        lat2ctm = recognition.LatticeToCtmJob(
            crp=self.crp[recognition_corpus_key],
            lattice_cache=recognition_job.out_lattice_bundle,
            **kwargs,
        )

        score_job = self.scorers[recognition_corpus_key].get_score_job(lat2ctm.out_ctm_file)
        tk.register_output(
            f"{filename}.reports",
            score_job.out_report_dir,
        )

        return score_job

    def _bpe_scoring(
        self,
        filename: str,
        recognition_corpus_key: str,
        recognition_job: returnn.ReturnnSearchJobV2,
    ) -> Job:
        bpe2word = returnn.SearchBPEtoWordsJob(recognition_job.out_search_file)
        word2ctm = returnn.SearchWordsToCTMJob(
            bpe2word.out_word_search_results,
            self.corpus_data[recognition_corpus_key].corpus_object.corpus_file,
        )

        score_job = self.scorers[recognition_corpus_key].get_score_job(word2ctm.out_ctm_file)
        tk.register_output(
            f"{filename}.reports",
            score_job.out_report_dir,
        )

        return score_job

    def _nn_recognition(self, search_type: SearchTypes, **kwargs) -> None:
        return {
            SearchTypes.GenericSeq2SeqSearchJob: self._seq2seq_nn_recognition,
            SearchTypes.AdvancedTreeSearch: self._atr_nn_recognition,
        }[search_type](**kwargs)

    def requires_label_file(self, label_unit: str) -> bool:
        return label_unit != "hmm"

    @lru_cache_with_signature
    def _get_label_file(self, key: str) -> tk.Path:
        state_tying_file = lexicon.DumpStateTyingJob(self.crp[key]).out_state_tying
        return GenerateLabelFileFromStateTyingJobV2(
            state_tying_file,
        ).out_label_file

    def _get_feature_flow_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        base_feature_flow: rasr.FlowNetwork,
        tf_graph: tk.Path,
        checkpoint: returnn.Checkpoint,
        **kwargs,
    ) -> rasr.FlowNetwork:
        if custom_rasr.LabelScorer.need_tf_flow(label_scorer.scorer_type):
            feature_flow = self._make_tf_feature_flow(base_feature_flow, tf_graph, checkpoint, **kwargs)
        else:
            feature_flow = base_feature_flow
            label_scorer.set_input_config()
            label_scorer.set_loader_config(self._make_model_loader_config(tf_graph, checkpoint))

        return feature_flow

    def _seq2seq_nn_recognition(
        self,
        train_exp_name: str,
        recog_exp_name: str,
        recognition_corpus_key: str,
        lookahead_options: Dict,
        trial_num: Optional[int] = None,
        epochs: List[EpochType] = [],
        lm_scales: List[float] = [0],
        prior_scales: List[float] = [0],
        prior_args: Dict = {},
        lattice_to_ctm_kwargs: Dict = {},
        label_unit: str = "phoneme",
        label_tree_args: Dict = {},
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        flow_args: Dict = {},
        tf_flow_args: Dict = {},
        extra_name: str = "",
        report_args: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        crp = self.crp[recognition_corpus_key]

        label_tree = custom_rasr.LabelTree(
            label_unit,
            lexicon_config=self.corpus_data[recognition_corpus_key].lexicon,
            **label_tree_args,
        )

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(recognition_corpus_key)

        base_feature_flow = self._make_base_feature_flow(recognition_corpus_key, **flow_args)

        for lm_scale, prior_scale, epoch in itertools.product(lm_scales, prior_scales, epochs):
            tf_graph = self._make_tf_graph(
                train_exp_name=train_exp_name,
                returnn_config=self.returnn_configs[train_exp_name].recog_configs[recog_exp_name],
                label_scorer_type=label_scorer_type,
                epoch=epoch,
                trial_num=trial_num,
            )

            checkpoint = self._get_checkpoint(train_exp_name, epoch, trial_num)

            crp.language_model_config.scale = lm_scale  # type: ignore

            if label_scorer_args.get("use_prior", False) and prior_scale:
                prior_file = self._get_prior_file(
                    train_exp_name=train_exp_name,
                    epoch=epoch,
                    prior_args=prior_args,
                    trial_num=trial_num,
                )
                mod_label_scorer_args["prior_file"] = prior_file
            else:
                mod_label_scorer_args.pop("prior_file", None)
            mod_label_scorer_args["prior_scale"] = prior_scale

            label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

            feature_flow = self._get_feature_flow_for_label_scorer(
                label_scorer=label_scorer,
                base_feature_flow=base_feature_flow,
                tf_graph=tf_graph,
                checkpoint=checkpoint,
                **tf_flow_args,
            )

            rec = custom_recognition.GenericSeq2SeqSearchJob(
                crp=crp,
                feature_flow=feature_flow,
                label_scorer=label_scorer,
                label_tree=label_tree,
                lookahead_options={"scale": lm_scale, **lookahead_options},
                **kwargs,
            )

            if isinstance(epoch, int):
                epoch_str = f"{epoch:03d}"
            else:
                epoch_str = epoch
            exp_full = f"{recog_exp_name}_e-{epoch_str}_prior-{prior_scale:02.2f}_lm-{lm_scale:02.2f}{extra_name}"

            if trial_num is None:
                path = f"nn_recog/{recognition_corpus_key}/{train_exp_name}/{exp_full}"
            else:
                path = f"nn_recog/{recognition_corpus_key}/{train_exp_name}/trial-{trial_num:03d}/{exp_full}"

            rec.set_vis_name(f"Recog {path}")
            rec.add_alias(path)
            self.recog_jobs[path] = rec

            if recognition_corpus_key in self.scorers:
                scorer_job = self._lattice_scoring(
                    filename=path,
                    recognition_corpus_key=recognition_corpus_key,
                    recognition_job=rec,
                    **lattice_to_ctm_kwargs,
                )

                if self._report:
                    report_args = report_args or {}
                    self._report.add(
                        {
                            "train_name": train_exp_name,
                            "recog_name": recog_exp_name,
                            "corpus": recognition_corpus_key,
                            "trial": self._get_trial_value(train_exp_name, trial_num),
                            "epoch": self._get_epoch_value(train_exp_name, epoch, trial_num),
                            "prior_scale": prior_scale,
                            "lm_scale": lm_scale,
                            "wer": scorer_job.out_wer,
                            "sub": scorer_job.out_percent_substitution,
                            "del": scorer_job.out_percent_deletions,
                            "ins": scorer_job.out_percent_insertions,
                            **report_args.get(train_exp_name, {}),
                        }
                    )

    def _atr_nn_recognition(
        self,
        train_exp_name: str,
        recog_exp_name: str,
        recognition_corpus_key: str,
        num_inputs: int,
        num_classes: int,
        trial_num: Optional[int] = None,
        epochs: List[EpochType] = [],
        lm_scales: List[float] = [0],
        prior_scales: List[float] = [0],
        pronunciation_scales: List[float] = [0],
        prior_args: Dict = {},
        lattice_to_ctm_kwargs: Dict = {},
        flow_args: Dict = {},
        **kwargs,
    ) -> None:
        crp = self.crp[recognition_corpus_key]

        acoustic_mixture_path = mm.CreateDummyMixturesJob(num_classes, num_inputs).out_mixtures

        base_feature_flow = self._make_base_feature_flow(recognition_corpus_key, **flow_args)

        for lm_scale, prior_scale, pronunciation_scale, epoch in itertools.product(
            lm_scales, prior_scales, pronunciation_scales, epochs
        ):
            tf_graph = self._make_tf_graph(
                train_exp_name=train_exp_name,
                returnn_config=self.returnn_configs[train_exp_name].recog_configs[recog_exp_name],
                epoch=epoch,
                trial_num=trial_num,
            )
            checkpoint = self._get_checkpoint(train_exp_name, epoch, trial_num)
            prior_file = self._get_prior_file(
                train_exp_name=train_exp_name,
                epoch=epoch,
                prior_args=prior_args,
                trial_num=trial_num,
            )

            crp.language_model_config.scale = lm_scale  # type: ignore

            feature_scorer = rasr.PrecomputedHybridFeatureScorer(
                prior_mixtures=acoustic_mixture_path,
                priori_scale=prior_scale,
                prior_file=prior_file,
            )

            model_combination_config = rasr.RasrConfig()
            model_combination_config.pronunciation_scale = pronunciation_scale

            feature_flow = self._make_tf_feature_flow(
                base_feature_flow,
                tf_graph,
                checkpoint,
            )

            rec = recognition.AdvancedTreeSearchJob(
                crp=crp,
                feature_flow=feature_flow,
                feature_scorer=feature_scorer,
                model_combination_config=model_combination_config,
                **kwargs,
            )

            if isinstance(epoch, int):
                epoch_str = f"{epoch:03d}"
            else:
                epoch_str = epoch
            exp_full = f"{recog_exp_name}_e-{epoch_str}_pron-{pronunciation_scale:02.2f}_prior-{prior_scale:02.2f}_lm-{lm_scale:02.2f}"

            if trial_num is None:
                path = f"nn_recog/{recognition_corpus_key}/{train_exp_name}/{exp_full}"
            else:
                path = f"nn_recog/{recognition_corpus_key}/{train_exp_name}/trial-{trial_num:03d}/{exp_full}"

            rec.set_vis_name(f"Recog {path}")
            rec.add_alias(path)

            scorer_job = self._lattice_scoring(
                filename=path,
                recognition_corpus_key=recognition_corpus_key,
                recognition_job=rec,
                **lattice_to_ctm_kwargs,
            )

            if self._summary_report:
                self._summary_report.add_row(
                    {
                        SummaryKey.TRAIN_NAME.value: train_exp_name,
                        SummaryKey.RECOG_NAME.value: recog_exp_name,
                        SummaryKey.CORPUS.value: recognition_corpus_key,
                        SummaryKey.TRIAL.value: self._get_trial_value(train_exp_name, trial_num),
                        SummaryKey.EPOCH.value: self._get_epoch_value(train_exp_name, epoch, trial_num),
                        SummaryKey.PRIOR.value: prior_scale,
                        SummaryKey.LM.value: lm_scale,
                        SummaryKey.WER.value: scorer_job.out_wer,
                        SummaryKey.SUB.value: scorer_job.out_percent_substitution,
                        SummaryKey.DEL.value: scorer_job.out_percent_deletions,
                        SummaryKey.INS.value: scorer_job.out_percent_insertions,
                        SummaryKey.ERR.value: scorer_job.out_num_errors,
                    }
                )

    def _nn_alignment(
        self,
        train_exp_name: str,
        align_corpus_key: str,
        trial_num: Optional[int] = None,
        epochs: List[EpochType] = [],
        lm_scales: List[float] = [],
        prior_scales: List[float] = [0],
        prior_args: Dict = {},
        label_unit: str = "phoneme",
        label_scorer_type: str = "precomputed-log-posterior",
        label_scorer_args: Dict = {},
        flow_args: Dict = {},
        **kwargs,
    ) -> None:
        crp = self.crp[align_corpus_key]

        mod_label_scorer_args = copy.deepcopy(label_scorer_args)
        if self.requires_label_file(label_unit):
            mod_label_scorer_args["label_file"] = self._get_label_file(align_corpus_key)

        base_feature_flow = self._make_base_feature_flow(align_corpus_key, **flow_args)

        for lm_scale, prior_scale, epoch in itertools.product(lm_scales, prior_scales, epochs):
            tf_graph = self._make_tf_graph(
                train_exp_name=train_exp_name,
                returnn_config=self.returnn_configs[train_exp_name].align_config,
                epoch=epoch,
                label_scorer_type=label_scorer_type,
                trial_num=trial_num,
            )

            checkpoint = self._get_checkpoint(train_exp_name, epoch, trial_num)

            crp.language_model_config.scale = lm_scale  # type: ignore

            if label_scorer_args.get("use_prior", False) and prior_scale:
                prior_file = self._get_prior_file(
                    train_exp_name=train_exp_name,
                    epoch=epoch,
                    prior_args=prior_args,
                    trial_num=trial_num,
                )
                mod_label_scorer_args["prior_file"] = prior_file
            else:
                mod_label_scorer_args.pop("prior_file", None)
            mod_label_scorer_args["prior_scale"] = prior_scale

            label_scorer = custom_rasr.LabelScorer(label_scorer_type, **mod_label_scorer_args)

            feature_flow = self._get_feature_flow_for_label_scorer(
                label_scorer=label_scorer,
                base_feature_flow=base_feature_flow,
                tf_graph=tf_graph,
                checkpoint=checkpoint,
            )

            align = custom_mm.Seq2SeqAlignmentJob(
                crp=crp,
                feature_flow=feature_flow,
                label_scorer=label_scorer,
                **kwargs,
            )

            if isinstance(epoch, int):
                epoch_str = f"{epoch:03d}"
            else:
                epoch_str = epoch
            exp_full = f"align_e-{epoch_str}_prior-{prior_scale:02.2f}"

            if trial_num is None:
                path = f"nn_recog/{align_corpus_key}/{train_exp_name}/{exp_full}"
            else:
                path = f"nn_recog/{align_corpus_key}/{train_exp_name}/trial-{trial_num:03d}/{exp_full}"

            align.set_vis_name(f"Alignment {path}")
            align.add_alias(path)

            tk.register_output(
                f"{path}.alignment.cache.bundle",
                align.out_alignment_bundle,
            )
            if align_corpus_key not in self.alignments:
                self.alignments[align_corpus_key] = {}
            self.alignments[align_corpus_key][train_exp_name] = rasr.FlagDependentFlowAttribute(
                "cache_mode",
                {
                    "task_dependent": align.out_alignment_path,
                    "bundle": align.out_alignment_bundle,
                },
            )

    # -------------------- run functions  --------------------
    def run_recogs_for_corpora(
        self,
        corpora: List[str],
        train_exp_name: str,
        search_type: SearchTypes,
        **kwargs,
    ):
        for c_key in corpora:
            for recog_exp_name in self.returnn_configs[train_exp_name].recog_configs.keys():
                self._nn_recognition(
                    search_type=search_type,
                    train_exp_name=train_exp_name,
                    recog_exp_name=recog_exp_name,
                    recognition_corpus_key=c_key,
                    **kwargs,
                )

    def run_train_step(
        self,
        train_args: Optional[Dict] = None,
    ) -> None:
        for train_exp_name in self.returnn_configs.keys():
            self.returnn_training(train_exp_name, **(train_args or {}))

    def run_dev_recog_step(
        self,
        recog_args: Optional[Dict] = None,
        search_type: SearchTypes = SearchTypes.GenericSeq2SeqSearchJob,
        **kwargs,
    ) -> None:
        for train_exp_name in self.returnn_configs.keys():
            self.run_recogs_for_corpora(
                self.dev_corpora,
                train_exp_name,
                search_type,
                **(recog_args or {}),
                **kwargs,
            )

    def run_test_recog_step(
        self,
        recog_args: Optional[Dict] = None,
        search_type: SearchTypes = SearchTypes.GenericSeq2SeqSearchJob,
    ) -> None:
        for train_exp_name in self.returnn_configs.keys():
            self.run_recogs_for_corpora(
                self.test_corpora,
                train_exp_name,
                search_type,
                **(recog_args or {}),
            )

    def run_align_step(self, align_args: Optional[Dict] = None) -> None:
        for al_c in self.align_corpora:
            self._nn_alignment(
                align_corpus_key=al_c,
                **(align_args or {}),
            )