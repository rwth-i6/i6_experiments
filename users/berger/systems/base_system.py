# -------------------- Imports --------------------

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional

from i6_core import corpus, rasr, recognition, am
from i6_experiments.common.setups.rasr.util.rasr import RasrDataInput
from i6_experiments.users.berger.recipe import summary as custom_summary
from i6_experiments.users.berger.util import ToolPaths
from sisyphus import tk

from . import types
from .functors import Functors


Path = tk.setup_path(__package__)


class BaseSystem(ABC, Generic[types.TrainJobType, types.ConfigType]):
    def __init__(
        self,
        tool_paths: ToolPaths,
        summary_keys: Optional[List[types.SummaryKey]] = None,
    ) -> None:
        self._tool_paths = tool_paths

        # Build base crp
        self._base_crp = rasr.CommonRasrParameters()
        self._base_crp.python_program_name = tool_paths.rasr_python_exe  # type: ignore
        rasr.crp_add_default_output(self._base_crp)
        self._base_crp.set_executables(rasr_binary_path=tool_paths.rasr_binary_path)

        # exp-name mapped to ReturnnConfigs collection
        self._returnn_configs: Dict[str, types.ReturnnConfigs[types.ConfigType]] = {}

        # exp-name mapped to ReturnnConfigs collection
        self._train_jobs: Dict[str, types.TrainJobType] = {}

        # lists of corpus keys
        self._align_corpora: List[str] = []
        self._dev_corpora: List[str] = []
        self._test_corpora: List[str] = []

        # corpus-key mapped to CorpusInfo object
        self._corpus_info: Dict[str, types.CorpusInfo] = {}

        self.summary_report = custom_summary.SummaryReport(
            [key.value for key in (summary_keys or types.SummaryKey)],
            types.SummaryKey.WER.value,
        )

        self._functors = self._initialize_functors()

    @abstractmethod
    def _initialize_functors(self) -> Functors[types.TrainJobType, types.ConfigType]:
        ...

    def add_experiment_configs(
        self, returnn_configs: Dict[str, types.ReturnnConfigs[types.ConfigType]]
    ) -> None:
        self._returnn_configs.update(returnn_configs)

    def init_corpora(
        self,
        align_keys: List[str] = [],
        dev_keys: List[str] = [],
        test_keys: List[str] = [],
        corpus_data: Dict[str, RasrDataInput] = {},
        am_args: Dict = {},
        scorer_type: types.ScoreJobType = recognition.ScliteJob,
        stm_kwargs: Dict = {},
        score_kwargs: Dict = {},
    ) -> None:
        all_keys = align_keys + dev_keys + test_keys
        assert all(key in corpus_data for key in all_keys)

        self._align_corpora = align_keys
        self._dev_corpora = dev_keys
        self._test_corpora = test_keys

        for key, data in corpus_data.items():
            assert data.corpus_object.corpus_file is not None
            crp = self._get_crp_for_data_input(data, am_args, base_crp=self._base_crp)
            corpus_info = types.CorpusInfo(data, crp)
            if key in dev_keys + test_keys:
                corpus_info.scorer = self._get_scorer(
                    data.corpus_object.corpus_file,
                    stm_kwargs,
                    scorer_type,
                    score_kwargs,
                )
            self._corpus_info[key] = corpus_info

    @staticmethod
    def _get_crp_for_data_input(
        data: RasrDataInput,
        am_args: Dict = {},
        base_crp: Optional[rasr.CommonRasrParameters] = None,
    ) -> rasr.CommonRasrParameters:
        crp = rasr.CommonRasrParameters(base_crp)

        rasr.crp_set_corpus(crp, data.corpus_object)
        crp.concurrent = data.concurrent
        crp.segment_path = corpus.SegmentCorpusJob(  # type: ignore
            data.corpus_object.corpus_file, data.concurrent
        ).out_segment_path

        if data.lm is not None:
            crp.language_model_config = rasr.RasrConfig()  # type: ignore
            crp.language_model_config.type = data.lm["type"]  # type: ignore
            crp.language_model_config.file = data.lm["filename"]  # type: ignore

        crp.lexicon_config = rasr.RasrConfig()  # type: ignore
        crp.lexicon_config.file = data.lexicon["filename"]
        crp.lexicon_config.normalize_pronunciation = data.lexicon[
            "normalize_pronunciation"
        ]

        crp.acoustic_model_config = am.acoustic_model_config(**am_args)  # type: ignore
        crp.acoustic_model_config.allophones.add_all = data.lexicon["add_all"]  # type: ignore
        crp.acoustic_model_config.allophones.add_from_lexicon = data.lexicon["add_from_lexicon"]  # type: ignore

        return crp

    def _get_scorer(
        self,
        corpus_file: tk.Path,
        stm_kwargs: Dict,
        scorer_type: types.ScoreJobType,
        score_kwargs: Dict,
    ) -> types.ScorerInfo:
        stm_path = corpus.CorpusToStmJob(corpus_file, **stm_kwargs).out_stm_path
        return types.ScorerInfo(
            ref_file=stm_path, job_type=scorer_type, score_kwargs=score_kwargs
        )

    # -------------------- run functions  --------------------

    def run_train_step(self, **kwargs) -> None:
        for train_exp_name, configs in self._returnn_configs.items():
            named_train_config = types.NamedConfig(train_exp_name, configs.train_config)
            self._train_jobs[train_exp_name] = self._functors.train(
                train_config=named_train_config, **kwargs
            )

    def run_recogs_for_corpora(
        self,
        corpora: List[str],
        train_exp_name: str,
        **kwargs,
    ) -> None:
        returnn_configs = self._returnn_configs[train_exp_name]
        train_job = self._train_jobs[train_exp_name]
        named_train_job = types.NamedTrainJob(train_exp_name, train_job)
        for c_key in corpora:
            named_corpus = types.NamedCorpusInfo(c_key, self._corpus_info[c_key])
            for recog_exp_name, recog_config in returnn_configs.recog_configs.items():
                named_recog_config = types.NamedConfig(recog_exp_name, recog_config)
                recog_results = self._functors.recognize(
                    train_job=named_train_job,
                    prior_config=returnn_configs.prior_config,
                    recog_config=named_recog_config,
                    recog_corpus=named_corpus,
                    **kwargs,
                )
                self.summary_report.add_rows(recog_results)

    def run_dev_recog_step(self, **kwargs) -> None:
        for train_exp_name in self._returnn_configs.keys():
            self.run_recogs_for_corpora(
                self._dev_corpora,
                train_exp_name,
                **kwargs,
            )

    def run_test_recog_step(self, **kwargs) -> None:
        for train_exp_name in self._returnn_configs.keys():
            self.run_recogs_for_corpora(self._test_corpora, train_exp_name, **kwargs)

    def run_align_step(self, **kwargs) -> None:
        for train_exp_name in self._returnn_configs.keys():
            train_job = self._train_jobs[train_exp_name]
            named_train_job = types.NamedTrainJob(train_exp_name, train_job)
            for c_key in self._align_corpora:
                named_corpus = types.NamedCorpusInfo(c_key, self._corpus_info[c_key])
                prior_config = self._returnn_configs[train_exp_name].prior_config
                align_config = self._returnn_configs[train_exp_name].align_config
                self._functors.align(
                    train_job=named_train_job,
                    prior_config=prior_config,
                    align_config=align_config,
                    align_corpus=named_corpus,
                    **kwargs,
                )
