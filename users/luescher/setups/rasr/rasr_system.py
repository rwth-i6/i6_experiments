__all__ = ["RasrSystem"]

import copy
from typing import List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

import sisyphus.global_settings as gs
import sisyphus.toolkit as tk

# -------------------- Recipes --------------------

import i6_core.am as am
import i6_core.corpus as corpus_recipes
import i6_core.meta as meta
import i6_core.mm as mm
import i6_core.rasr as rasr

from .util import RasrDataInput

# -------------------- Init --------------------

Path = tk.setup_path(__package__)

# -------------------- System --------------------


class RasrSystem(meta.System):
    """
    very limited, so TODO:
    - 3 corpora types: train, dev and test
    - only train corpora will be aligned
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

    corpus_key -> str (basically corpus identifier, corpus_name would also be an option)
    corpus_object -> CorpusObject
    corpus_file -> str (would be filepath)
    corpus -> Path
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
    ):
        """

        :param rasr_binary_path: path to the rasr binary folder
        :param rasr_arch: RASR compile architecture suffix
        """
        super().__init__()

        self.rasr_binary_path = rasr_binary_path
        self.rasr_arch = rasr_arch

        if hasattr(gs, "RASR_PYTHON_HOME") and gs.RASR_PYTHON_HOME is not None:
            self.crp["base"].python_home = gs.RASR_PYTHON_HOME
        if hasattr(gs, "RASR_PYTHON_EXE") and gs.RASR_PYTHON_EXE is not None:
            self.crp["base"].python_program_name = gs.RASR_PYTHON_EXE

        rasr.FlowNetwork.default_flags["cache_mode"] = "task_dependent"

        self.rasr_init_args = None

        self.train_corpora = []
        self.dev_corpora = []
        self.test_corpora = []

        self.corpora = {}
        self.concurrent = {}

    # -------------------- base functions --------------------
    @tk.block()
    def _init_am(self, **kwargs):
        """
        TODO: docstring

        :param kwargs:
        :return:
        """
        self.crp["base"].acoustic_model_config = am.acoustic_model_config(**kwargs)
        allow_zero_weights = kwargs.get("allow_zero_weights", False)
        if allow_zero_weights:
            self.crp["base"].acoustic_model_config.mixture_set.allow_zero_weights = True
            self.crp[
                "base"
            ].acoustic_model_config.old_mixture_set.allow_zero_weights = True

    @tk.block()
    def _init_corpus(self, corpus_key: str):
        """
        TODO: docstring

        :param corpus_key:
        :return:
        """
        segm_corpus_job = corpus_recipes.SegmentCorpusJob(
            self.corpora[corpus_key].corpus_file, self.concurrent[corpus_key]
        )
        self.set_corpus(
            name=corpus_key,
            corpus=self.corpora[corpus_key],
            concurrent=self.concurrent[corpus_key],
            segment_path=segm_corpus_job.out_segment_path,
        )

        self.crp[corpus_key].set_executables(
            rasr_binary_path=self.rasr_binary_path, rasr_arch=self.rasr_arch
        )
        self.jobs[corpus_key]["segment_corpus"] = segm_corpus_job

    @tk.block()
    def _init_lm(
        self, corpus_key: str, filename: Path, type: str, scale: int, **kwargs
    ):
        """
        TODO: docstring

        :param corpus_key:
        :param filename:
        :param type:
        :param scale:
        :param kwargs:
        :return:
        """
        self.crp[corpus_key].language_model_config = rasr.RasrConfig()
        self.crp[corpus_key].language_model_config.type = type
        self.crp[corpus_key].language_model_config.file = filename
        self.crp[corpus_key].language_model_config.scale = scale

    @tk.block()
    def _init_lexicon(
        self, corpus_key: str, filename: Path, normalize_pronunciation: bool, **kwargs
    ):
        """
        TODO: docstring

        :param corpus_key:
        :param filename:
        :param normalize_pronunciation:
        :param kwargs:
        :return:
        """
        self.crp[corpus_key].lexicon_config = rasr.RasrConfig()
        self.crp[corpus_key].lexicon_config.file = filename
        self.crp[
            corpus_key
        ].lexicon_config.normalize_pronunciation = normalize_pronunciation

    def _set_scorer_for_corpus(self, eval_corpus_key: str):
        """
        TODO: docstring

        :param eval_corpus_key:
        :return:
        """
        scorer_args = copy.deepcopy(self.rasr_init_args.scorer_args)
        if self.rasr_init_args.scorer == "kaldi":
            scorer_args = (
                scorer_args
                if scorer_args is not None
                else dict(mapping={"[SILENCE]": ""})
            )
            self.set_kaldi_scorer(
                corpus=eval_corpus_key,
                **scorer_args,
            )
        elif self.rasr_init_args.scorer == "hub5":
            scorer_args = scorer_args if scorer_args is not None else {}
            self.set_hub5_scorer(corpus=eval_corpus_key, **scorer_args)
        else:
            scorer_args = (
                scorer_args if scorer_args is not None else dict(sort_files=False)
            )
            self.set_sclite_scorer(
                corpus=eval_corpus_key,
                **scorer_args,
            )

    def prepare_scoring(self):
        """
        Initializes the scorer for each dev and test corpus, and create stm files if not already given
        """
        for eval_c in self.dev_corpora + self.test_corpora:
            if eval_c not in self.stm_files:
                stm_args = (
                    self.rasr_init_args.stm_args
                    if self.rasr_init_args.stm_args is not None
                    else {}
                )
                self.create_stm_from_corpus(eval_c, **stm_args)
            self._set_scorer_for_corpus(eval_c)

    @staticmethod
    def _assert_corpus_name_unique(*args):
        """
        TODO: docstring

        :param args:
        :return:
        """
        name_list = []
        for i in args:
            name_list.extend(list(i.keys()))
        assert len(name_list) == len(set(name_list)), "corpus names are not unique"

    # -------------------- base functions --------------------

    def set_binaries_for_crp(
        self,
        crp_key: str,
        rasr_binary_path: tk.Path,
        rasr_arch: str = "linux-x86_64-standard",
        python_home: Optional[tk.Path] = None,
        python_exe: Optional[tk.Path] = None,
    ):
        """
        Set explicit binaries and python for RASR with respect to a specific crp entry.

        If this is done for the `base` crp, this function should be called before any other call to `system`,
        especially before `init_system` or `add_corpus`.

        :param binary_path: path to a RASR binary folder
        :param rasr_arch: RASR architecture suffix
        :param python_home: path to the python virtual environment base directory
            in case of None nothing will be set
        :param python_exe: path to the python binary that should be executed
            in case of None nothing will be set
        """
        self.crp[crp_key].set_executables(
            rasr_binary_path=rasr_binary_path, rasr_arch=rasr_arch
        )

        if python_home is not None:
            self.crp[crp_key].python_home = python_home
        if python_exe is not None:
            self.crp[crp_key].python_program_name = python_exe

    def add_corpus(self, corpus_key: str, data: RasrDataInput, add_lm: bool):
        """
        TODO: docstring

        :param corpus_key:
        :param data:
        :param add_lm:
        :return:
        """
        self.corpora[corpus_key] = data.corpus_object
        self.concurrent[corpus_key] = data.concurrent
        self._init_corpus(corpus_key)
        self._init_lexicon(corpus_key, **data.lexicon)
        if add_lm:
            self._init_lm(corpus_key, **data.lm)
        if data.stm is not None:
            self.stm_files[corpus_key] = data.stm
        if data.glm is not None:
            self.glm_files[corpus_key] = data.glm
        tk.register_output(
            f"corpora/{corpus_key}.xml.gz", data.corpus_object.corpus_file
        )

    def extract_features_for_corpus(self, corpus: str, feat_args: dict):
        """
        TODO: docstring

        :param corpus:
        :param feat_args:
        :return:
        """
        for k, v in feat_args.items():
            if k == "mfcc":
                self.mfcc_features(corpus, **v)
            if k == "gt":
                self.gt_features(corpus, **v)
            if k == "fb":
                self.fb_features(corpus, **v)
            if k == "energy":
                self.energy_features(corpus, **v)
            if k == "voiced":
                self.voiced_features(corpus, **v)
            if k == "plp":
                self.plp_features(corpus, **v)
            if k == "tone":
                self.tone_features(corpus, **v)
            if k not in ("mfcc", "gt", "fb", "energy", "voiced", "tone", "plp"):
                self.generic_features(corpus, k, **v)

    @tk.block()
    def extract_features(
        self, feat_args: dict, corpus_list: Optional[List[str]] = None, **kwargs
    ):
        """
        TODO: docstring
        TODO: add more generic flow dependencies

        :param feat_args: see RasrInitArgs.feature_extraction_args
        :param corpus_list:
        :param kwargs:
        :return:
        """
        if corpus_list is None:
            corpus_list = self.train_corpora + self.dev_corpora + self.test_corpora

        for c in corpus_list:
            self.extract_features_for_corpus(c, feat_args)

        for tc in self.train_corpora:
            for fk in feat_args.keys():
                if fk == "mfcc":
                    self.normalize(tc, "mfcc+deriv", corpus_list)
                if "energy" in feat_args.keys():
                    if fk == "mfcc":
                        self.add_energy_to_features(tc, "mfcc+deriv")
                        self.add_energy_to_features(tc, "mfcc+deriv+norm")
                    if fk == "gt":
                        self.add_energy_to_features(tc, "gt")
                    if fk == "fb":
                        self.add_energy_to_features(tc, "fb")

    # -------------------- Single Density Mixtures --------------------

    def single_density_mixtures(
        self, name: str, corpus_key: str, feature_flow_key: str, alignment: str
    ):
        """
        TODO: docstring

        :param name:
        :param corpus_key:
        :param feature_flow_key:
        :param alignment:
        :return:
        """
        self.estimate_mixtures(
            name=name,
            corpus=corpus_key,
            flow=feature_flow_key,
            alignment=meta.select_element(
                self.alignments, corpus_key, (corpus_key, alignment, -1)
            ),
            split_first=False,
        )
        tk.register_output(
            f"train/sdm/{corpus_key}.{name}.mix",
            self.mixtures[corpus_key][f"estimate_mixtures_{name}"],
        )

    # -------------------- Forced Alignment --------------------

    def forced_align(
        self,
        name: str,
        *,
        target_corpus_key: str,
        flow: Union[
            str,
            List[str],
            Tuple[str],
            rasr.FlagDependentFlowAttribute,
            rasr.FlowNetwork,
        ],
        feature_scorer_corpus_key: str = None,
        feature_scorer: Union[str, List[str], Tuple[str], rasr.FeatureScorer],
        scorer_index: int = -1,
        dump_alignment: bool = False,
        **kwargs,
    ):
        """
        TODO: docstring

        :param name:
        :param target_corpus_key:
        :param flow:
        :param feature_scorer_corpus_key:
        :param feature_scorer:
        :param scorer_index:
        :param dump_alignment:
        :param kwargs:
        :return:
        """
        selected_feature_scorer = meta.select_element(
            self.feature_scorers,
            feature_scorer_corpus_key,
            feature_scorer,
            scorer_index,
        )
        self.align(
            name=name,
            corpus=target_corpus_key,
            flow=flow,
            feature_scorer=selected_feature_scorer,
            **kwargs,
        )

        align_job: mm.AlignmentJob = self.jobs[target_corpus_key]["alignment_%s" % name]
        align_job.add_alias(
            "forced_alignment/alignment_%s/%s" % (name, target_corpus_key)
        )
        tk.register_output(
            "forced_alignment/alignment_%s_%s.bundle" % (name, target_corpus_key),
            align_job.out_alignment_bundle,
        )

        if dump_alignment:
            dump_job = mm.DumpAlignmentJob(
                crp=self.crp[target_corpus_key],
                feature_flow=meta.select_element(
                    self.feature_flows, target_corpus_key, flow
                ),
                original_alignment=meta.select_element(
                    self.alignments, target_corpus_key, name
                ),
            )
            self.jobs[target_corpus_key]["alignment_dump_%s" % name] = dump_job
            dump_job.add_alias(
                "forced_alignment/alignment_dump_%s/%s" % (name, target_corpus_key)
            )
            tk.register_output(
                "forced_alignment/alignment_dump_%s_%s.bundle"
                % (name, target_corpus_key),
                dump_job.out_alignment_bundle,
            )
