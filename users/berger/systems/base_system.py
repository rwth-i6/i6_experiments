# -------------------- Imports --------------------

from abc import ABC, abstractmethod
import copy
from typing import Dict, Generic, List, Optional

from i6_core import corpus, rasr, recognition
from i6_experiments.users.berger.recipe import summary as custom_summary
from i6_experiments.users.berger.util import ToolPaths
from i6_experiments.users.berger import helpers as helpers
from sisyphus import tk

from . import types
from . import dataclasses
from .functors import Functors

Path = tk.setup_path(__package__)


class BaseSystem(ABC, Generic[types.TrainJobType, types.ConfigType]):
    def __init__(
        self,
        tool_paths: ToolPaths,
        summary_keys: Optional[List[dataclasses.SummaryKey]] = None,
    ) -> None:
        self._tool_paths = tool_paths

        # Build base crp
        self._base_crp = rasr.CommonRasrParameters()
        self._base_crp.python_program_name = tool_paths.rasr_python_exe  # type: ignore
        rasr.crp_add_default_output(self._base_crp)
        self._base_crp.set_executables(rasr_binary_path=tool_paths.rasr_binary_path)

        # exp-name mapped to ReturnnConfigs collection
        self._returnn_configs: Dict[str, dataclasses.ReturnnConfigs[types.ConfigType]] = {}

        # exp-name mapped to CustomStepKwargs collection
        self._custom_step_kwargs: Dict[str, dataclasses.CustomStepKwargs] = {}

        # exp-name mapped to ReturnnConfigs collection
        self._train_jobs: Dict[str, types.TrainJobType] = {}

        # lists of corpus keys
        self._align_corpora: List[str] = []
        self._dev_corpora: List[str] = []
        self._test_corpora: List[str] = []

        # corpus-key mapped to CorpusInfo object
        self._corpus_info: Dict[str, dataclasses.CorpusInfo] = {}

        self.summary_report = custom_summary.SummaryReport(
            [key.value for key in (summary_keys or dataclasses.SummaryKey)],
            dataclasses.SummaryKey.WER.value,
        )

        self._functors = self._initialize_functors()

    @abstractmethod
    def _initialize_functors(self) -> Functors[types.TrainJobType, types.ConfigType]:
        ...

    def get_train_job(self, exp_name: Optional[str] = None) -> types.TrainJobType:
        if exp_name is not None:
            assert exp_name in self._train_jobs
            return self._train_jobs[exp_name]
        else:
            assert len(self._train_jobs) == 1
            return next(iter(self._train_jobs.values()))

    def cleanup_experiments(self) -> None:
        self._returnn_configs.clear()
        self._custom_step_kwargs.clear()

    def add_experiment_configs(
        self,
        name: str,
        returnn_configs: dataclasses.ReturnnConfigs[types.ConfigType],
        custom_step_kwargs: Optional[dataclasses.CustomStepKwargs] = None,
    ) -> None:
        self._returnn_configs[name] = returnn_configs
        if custom_step_kwargs is not None:
            self._custom_step_kwargs[name] = custom_step_kwargs
        else:
            self._custom_step_kwargs[name] = dataclasses.CustomStepKwargs()

    def init_corpora(
        self,
        align_keys: List[str] = [],
        dev_keys: List[str] = [],
        test_keys: List[str] = [],
        corpus_data: Dict[str, helpers.RasrDataInput] = {},
        am_args: Dict = {},
    ) -> None:
        all_keys = align_keys + dev_keys + test_keys
        assert all(key in corpus_data for key in all_keys)

        self._align_corpora = align_keys
        self._dev_corpora = dev_keys
        self._test_corpora = test_keys

        for key, data in corpus_data.items():
            assert data.corpus_object.corpus_file is not None
            crp = helpers.get_crp_for_data_input(
                data=data,
                am_args=am_args,
                base_crp=self._base_crp,
                tool_paths=self._tool_paths,
            )
            corpus_info = dataclasses.CorpusInfo(data, crp)
            self._corpus_info[key] = corpus_info

    @staticmethod
    def _get_scorer(
        corpus_file: tk.Path,
        stm_kwargs: Dict,
        scorer_type: types.ScoreJobType,
        score_kwargs: Dict,
        stm_path: Optional[tk.Path] = None,
    ) -> dataclasses.ScorerInfo:
        if stm_path is None:
            stm_path = corpus.CorpusToStmJob(corpus_file, **stm_kwargs).out_stm_path
        return dataclasses.ScorerInfo(ref_file=stm_path, job_type=scorer_type, score_kwargs=score_kwargs)

    def setup_scoring(
        self,
        scoring_corpora: Optional[Dict[str, tk.Path]] = None,
        scorer_type: types.ScoreJobType = recognition.ScliteJob,
        stm_kwargs: Dict = {},
        score_kwargs: Dict = {},
        stm_paths: Optional[Dict[str, tk.Path]] = None,
    ) -> None:
        if scoring_corpora is None:
            scoring_corpora = {}
            for key in self._dev_corpora + self._test_corpora:
                corpus_file = self._corpus_info[key].data.corpus_object.corpus_file
                assert corpus_file is not None
                scoring_corpora[key] = corpus_file
        if stm_paths is None:
            stm_paths = {}
        for key, corpus_file in scoring_corpora.items():
            if key not in self._corpus_info:
                continue
            scorer = self._get_scorer(
                corpus_file=corpus_file,
                stm_kwargs=stm_kwargs,
                scorer_type=scorer_type,
                score_kwargs=score_kwargs,
                stm_path=stm_paths.get(key, None),
            )
            self._corpus_info[key].scorer = scorer

    # -------------------- run functions  --------------------

    def run_train_step(self, **kwargs) -> None:
        for train_exp_name, configs in self._returnn_configs.items():
            mod_kwargs = copy.deepcopy(kwargs)
            mod_kwargs.update(self._custom_step_kwargs[train_exp_name].train_step_kwargs)
            named_train_config = dataclasses.NamedConfig(train_exp_name, configs.train_config)
            self._train_jobs[train_exp_name] = self._functors.train(train_config=named_train_config, **mod_kwargs)

    def run_recogs_for_corpora(
        self,
        corpora: List[str],
        train_exp_name: str,
        **kwargs,
    ) -> None:
        returnn_configs = self._returnn_configs[train_exp_name]
        train_job = self._train_jobs[train_exp_name]
        named_train_job = dataclasses.NamedTrainJob(train_exp_name, train_job)
        for c_key in corpora:
            named_corpus = dataclasses.NamedCorpusInfo(c_key, self._corpus_info[c_key])
            for recog_exp_name, recog_config in returnn_configs.recog_configs.items():
                named_recog_config = dataclasses.NamedConfig(recog_exp_name, recog_config)
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
            mod_kwargs = copy.deepcopy(kwargs)
            mod_kwargs.update(self._custom_step_kwargs[train_exp_name].dev_recog_step_kwargs)
            self.run_recogs_for_corpora(self._dev_corpora, train_exp_name, **mod_kwargs)

    def run_test_recog_step(self, **kwargs) -> None:
        for train_exp_name in self._returnn_configs.keys():
            mod_kwargs = copy.deepcopy(kwargs)
            mod_kwargs.update(self._custom_step_kwargs[train_exp_name].test_recog_step_kwargs)
            self.run_recogs_for_corpora(self._test_corpora, train_exp_name, **mod_kwargs)

    def run_align_step(self, **kwargs) -> Dict:
        results = {}
        for train_exp_name in self._returnn_configs.keys():
            mod_kwargs = copy.deepcopy(kwargs)
            mod_kwargs.update(self._custom_step_kwargs[train_exp_name].align_step_kwargs)
            train_job = self._train_jobs[train_exp_name]
            named_train_job = dataclasses.NamedTrainJob(train_exp_name, train_job)
            exp_results = {}
            for c_key in self._align_corpora:
                named_corpus = dataclasses.NamedCorpusInfo(c_key, self._corpus_info[c_key])
                prior_config = self._returnn_configs[train_exp_name].prior_config
                align_config = self._returnn_configs[train_exp_name].align_config
                exp_results[c_key] = self._functors.align(
                    train_job=named_train_job,
                    prior_config=prior_config,
                    align_config=align_config,
                    align_corpus=named_corpus,
                    **mod_kwargs,
                )
            if len(exp_results) == 1:
                exp_results = next(iter(exp_results.values()))
            results[train_exp_name] = exp_results
        if len(results) == 1:
            results = next(iter(results.values()))
        return results
