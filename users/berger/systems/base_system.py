# -------------------- Imports --------------------

from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional

from i6_core import corpus, rasr, recognition
from i6_experiments.users.berger.recipe import summary as custom_summary
from i6_experiments.users.berger.util import ToolPaths
from i6_experiments.users.berger import helpers as helpers
from sisyphus import tk

from . import types
from .types import TrainJobType, ConfigType
from . import dataclasses
from .functors import Functors

Path = tk.setup_path(__package__)


class BaseSystem(ABC, Generic[TrainJobType, ConfigType]):
    def __init__(
        self,
        tool_paths: ToolPaths,
        summary_keys: Optional[List[dataclasses.SummaryKey]] = None,
        summary_sort_keys: Optional[List[dataclasses.SummaryKey]] = None,
    ) -> None:
        self._tool_paths = tool_paths

        # Build base crp
        self._base_crp = rasr.CommonRasrParameters()
        self._base_crp.python_program_name = tool_paths.rasr_python_exe  # type: ignore
        rasr.crp_add_default_output(self._base_crp)
        self._base_crp.set_executables(rasr_binary_path=tool_paths.rasr_binary_path)

        # exp-name mapped to ReturnnConfigs collection
        self._returnn_configs: Dict[str, dataclasses.ReturnnConfigs[types.ConfigType]] = {}

        # exp-name mapped to ReturnnConfigs collection
        self._train_jobs: Dict[str, types.TrainJobType] = {}

        # lists of corpus keys
        self._align_corpora: List[str] = []
        self._dev_corpora: List[str] = []
        self._test_corpora: List[str] = []

        # corpus-key mapped to CorpusInfo object
        self._corpus_info: Dict[str, dataclasses.CorpusInfo] = {}

        if summary_keys is None:
            summary_keys = list(dataclasses.SummaryKey)
        if summary_sort_keys is None:
            summary_sort_keys = [dataclasses.SummaryKey.ERR, dataclasses.SummaryKey.WER]
        self.summary_report = custom_summary.SummaryReport(
            [key.value for key in summary_keys],
            [key.value for key in summary_sort_keys],
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

    def add_experiment_configs(
        self,
        name: str,
        returnn_configs: dataclasses.ReturnnConfigs[types.ConfigType],
    ) -> None:
        self._returnn_configs[name] = returnn_configs

    def init_corpora(
        self,
        align_keys: Optional[List[str]] = None,
        dev_keys: Optional[List[str]] = None,
        test_keys: Optional[List[str]] = None,
        corpus_data: Optional[Dict[str, helpers.RasrDataInput]] = None,
        am_args: Optional[Dict] = None,
    ) -> None:
        if align_keys is None:
            align_keys = []
        if dev_keys is None:
            dev_keys = []
        if test_keys is None:
            test_keys = []
        if corpus_data is None:
            corpus_data = {}
        if am_args is None:
            am_args = {}

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
        stm_kwargs: Optional[Dict] = None,
        score_kwargs: Optional[Dict] = None,
        stm_paths: Optional[Dict[str, tk.Path]] = None,
    ) -> None:
        if stm_kwargs is None:
            stm_kwargs = {}
        if score_kwargs is None:
            score_kwargs = {}
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

    def run_train_step(
        self, exp_names: Optional[List[str]] = None, train_descriptor: Optional[str] = None, **kwargs
    ) -> None:
        if exp_names is None:
            exp_names = list(self._returnn_configs.keys())
        for exp_name in exp_names:
            configs = self._returnn_configs[exp_name]
            if train_descriptor is not None:
                exp_name = f"{exp_name}_{train_descriptor}"
            named_train_config = dataclasses.NamedConfig(exp_name, configs.train_config)
            self._train_jobs[exp_name] = self._functors.train(train_config=named_train_config, **kwargs)

    def run_recog_step_for_corpora(
        self,
        exp_names: Optional[List[str]] = None,
        corpora: Optional[List[str]] = None,
        recog_descriptor: Optional[str] = None,
        **kwargs,
    ) -> None:
        if exp_names is None:
            exp_names = list(self._returnn_configs.keys())
        if corpora is None:
            corpora = self._dev_corpora + self._test_corpora
        else:
            assert all(c in self._dev_corpora + self._test_corpora for c in corpora)

        for exp_name in exp_names:
            assert exp_name in self._returnn_configs, f"Experiment {exp_name} has not been registered"
            returnn_configs = self._returnn_configs[exp_name]

            assert exp_name in self._train_jobs, f"Must first run training before recognition for experiment {exp_name}"
            train_job = self._train_jobs[exp_name]
            named_train_job = dataclasses.NamedTrainJob(exp_name, train_job)

            for c_key in corpora:
                named_corpus = dataclasses.NamedCorpusInfo(c_key, self._corpus_info[c_key])
                for recog_exp_name, recog_config in returnn_configs.recog_configs.items():
                    if recog_descriptor is not None:
                        recog_exp_name = f"{recog_exp_name}_{recog_descriptor}"
                    named_recog_config = dataclasses.NamedConfig(recog_exp_name, recog_config)
                    recog_results = self._functors.recognize(
                        train_job=named_train_job,
                        prior_config=returnn_configs.prior_config,
                        recog_config=named_recog_config,
                        recog_corpus=named_corpus,
                        **kwargs,
                    )
                    self.summary_report.add_rows(recog_results)

    def run_dev_recog_step(self, exp_names: Optional[List[str]] = None, **kwargs) -> None:
        self.run_recog_step_for_corpora(exp_names, self._dev_corpora, **kwargs)

    def run_test_recog_step(self, exp_names: Optional[List[str]] = None, **kwargs) -> None:
        self.run_recog_step_for_corpora(exp_names, self._test_corpora, **kwargs)

    def run_align_step(
        self,
        exp_names: Optional[List[str]] = None,
        corpora: Optional[List[str]] = None,
        align_descriptor: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, dataclasses.AlignmentData]]:
        if exp_names is None:
            exp_names = list(self._returnn_configs.keys())
        if corpora is None:
            corpora = self._align_corpora
        else:
            assert all(c in self._align_corpora for c in corpora)

        results = {}
        for exp_name in exp_names:
            train_job = self._train_jobs[exp_name]
            named_train_job = dataclasses.NamedTrainJob(exp_name, train_job)
            exp_results = {}
            for c_key in corpora:
                named_corpus = dataclasses.NamedCorpusInfo(c_key, self._corpus_info[c_key])
                prior_config = self._returnn_configs[exp_name].prior_config
                align_config = self._returnn_configs[exp_name].align_config

                if align_descriptor is not None:
                    exp_name = f"{exp_name}_{align_descriptor}"
                exp_results[c_key] = self._functors.align(
                    train_job=named_train_job,
                    prior_config=prior_config,
                    align_config=align_config,
                    align_corpus=named_corpus,
                    **kwargs,
                )
            results[exp_name] = exp_results
        return results
