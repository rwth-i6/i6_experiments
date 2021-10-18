__all__ = ["RasrSystem"]

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
    """

    def __init__(self):
        super().__init__()

        if hasattr(gs, "RASR_PYTHON_HOME") and gs.RASR_PYTHON_HOME is not None:
            self.crp["base"].python_home = gs.RASR_PYTHON_HOME
        if hasattr(gs, "RASR_PYTHON_EXE") and gs.RASR_PYTHON_EXE is not None:
            self.crp["base"].python_program_name = gs.RASR_PYTHON_EXE

        rasr.FlowNetwork.default_flags["cache_mode"] = "task_dependent"

        self.hybrid_init_args = None

        self.train_corpora = []
        self.dev_corpora = []
        self.test_corpora = []

        self.corpora = {}
        self.concurrent = {}

    # -------------------- base functions --------------------
    @tk.block()
    def _init_am(self, **kwargs):
        self.crp["base"].acoustic_model_config = am.acoustic_model_config(**kwargs)
        allow_zero_weights = kwargs.get("allow_zero_weights", False)
        if allow_zero_weights:
            self.crp["base"].acoustic_model_config.mixture_set.allow_zero_weights = True
            self.crp[
                "base"
            ].acoustic_model_config.old_mixture_set.allow_zero_weights = True

    @tk.block()
    def _init_corpus(self, name: str):
        segm_corpus_job = corpus_recipes.SegmentCorpusJob(
            self.corpora[name].corpus_file, self.concurrent[name]
        )
        self.set_corpus(
            name=name,
            corpus=self.corpora[name],
            concurrent=self.concurrent[name],
            segment_path=segm_corpus_job.out_segment_path,
        )
        self.jobs[name]["segment_corpus"] = segm_corpus_job

    @tk.block()
    def _init_lm(self, name: str, filename: Path, type: str, scale: int, **kwargs):
        self.crp[name].language_model_config = rasr.RasrConfig()
        self.crp[name].language_model_config.type = type
        self.crp[name].language_model_config.file = filename
        self.crp[name].language_model_config.scale = scale

    @tk.block()
    def _init_lexicon(
        self, name: str, filename: Path, normalize_pronunciation: bool, **kwargs
    ):
        self.crp[name].lexicon_config = rasr.RasrConfig()
        self.crp[name].lexicon_config.file = filename
        self.crp[name].lexicon_config.normalize_pronunciation = normalize_pronunciation

    @staticmethod
    def _assert_corpus_name_unique(*args):
        name_list = []
        for i in args:
            name_list.extend(list(i.keys()))
        assert len(name_list) == len(set(name_list)), "corpus names are not unique"

    def add_corpus(self, name: str, data: RasrDataInput, add_lm: bool):
        self.corpora[name] = data.corpus_object
        self.concurrent[name] = data.concurrent
        self._init_corpus(name)
        self._init_lexicon(name, **data.lexicon)
        if add_lm:
            self._init_lm(name, **data.lm)

    @tk.block()
    def extract_features(self, feat_args: dict, **kwargs):
        corpus_list = self.train_corpora + self.dev_corpora + self.test_corpora

        for k, v in feat_args.items():
            for c in corpus_list:
                if k == "mfcc":
                    self.mfcc_features(c, **v)
                if k == "gt":
                    self.gt_features(c, **v)
                if k == "fb":
                    self.fb_features(c, **v)
                if k == "energy":
                    self.energy_features(c, **v)
                if k == "voiced":
                    self.voiced_features(c, **v)
                if k == "plp":
                    self.plp_features(c, **v)
                if k == "tone":
                    self.tone_features(c, **v)
                if k not in ("mfcc", "gt", "fb", "energy", "voiced", "tone", "plp"):
                    self.generic_features(c, k, **v)
            if k == "mfcc":
                for trn_c in self.train_corpora:
                    self.normalize(trn_c, "mfcc+deriv", corpus_list)

        for kk in feat_args.keys():
            if "energy" in feat_args.keys():
                for t in self.train_corpora:
                    if kk == "mfcc":
                        self.add_energy_to_features(t, "mfcc+deriv")
                        self.add_energy_to_features(t, "mfcc+deriv+norm")
                    if kk == "gt":
                        self.add_energy_to_features(t, "gt")
                    if kk == "fb":
                        self.add_energy_to_features(t, "fb")

    # -------------------- Single Density Mixtures --------------------

    def single_density_mixtures(
        self, name: str, corpus: str, feature_flow: str, alignment: str
    ):
        self.estimate_mixtures(
            name=name,
            corpus=corpus,
            flow=feature_flow,
            alignment=meta.select_element(
                self.alignments, corpus, (corpus, alignment, -1)
            ),
            split_first=False,
        )

    # -------------------- Forced Alignment --------------------

    def forced_align(self, name, corpus, target_corpus, flow, feature_scorer, **kwargs):
        align_job = mm.AlignmentJob(
            crp=self.crp[target_corpus],
            feature_flow=meta.select_element(self.feature_flows, target_corpus, flow),
            feature_scorer=meta.select_element(
                self.feature_scorers, corpus, feature_scorer
            ),
            **kwargs,
        )

        self.jobs[target_corpus]["alignment_%s" % name] = align_job
        align_job.add_alias("forced_alignment/alignment_%s" % name)
        tk.register_output(
            "forced_alignment/alignment_%s.bundle" % name,
            align_job.out_alignment_bundle,
        )
        self.alignments[target_corpus][name] = [
            rasr.FlagDependentFlowAttribute(
                "cache_mode",
                {
                    "task_dependent": align_job.out_alignment_path,
                    "bundle": align_job.out_alignment_bundle,
                },
            )
        ]

        dump_job = mm.DumpAlignmentJob(
            crp=self.crp[target_corpus],
            feature_flow=meta.select_element(self.feature_flows, target_corpus, flow),
            original_alignment=meta.select_element(
                self.alignments, target_corpus, name
            ),
        )
        self.jobs[target_corpus]["alignment_dump_%s" % name] = dump_job
        dump_job.add_alias("forced_alignment/alignment_dump_%s" % name)
        tk.register_output(
            "forced_alignment/alignment_dump_%s.bundle" % name,
            dump_job.out_alignment_bundle,
        )
