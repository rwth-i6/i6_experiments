__all__ = ["BuildGenericSeq2SeqGlobalCacheJob", "GenericSeq2SeqSearchJobV2"]

from typing import List, Optional, Tuple
from sisyphus import *

assert __package__ is not None
Path = setup_path(__package__)

import shutil

from i6_core import rasr, util
from i6_experiments.users.berger.recipe.rasr.label_tree_and_scorer import LabelTree, LabelScorer


class BuildGenericSeq2SeqGlobalCacheJob(rasr.RasrCommand, Job):
    """
    Standalone job to create the global-cache for generic-seq2seq-tree-search
    """

    def __init__(
        self,
        crp: rasr.CommonRasrParameters,
        label_tree: LabelTree,
        label_scorer: LabelScorer,
        search_parameters: Optional[dict] = None,
        extra_config: Optional[rasr.RasrConfig] = None,
        extra_post_config: Optional[rasr.RasrConfig] = None,
    ):
        """
        :param crp: common RASR params (required: lexicon, acoustic_model, language_model, recognizer)
        :param label_tree: label tree object for structuring the search tree
        :param label_scorer: label scorer object for score computation
        :param extra_config: overlay config that influences the Job's hash
        :param extra_post_config: overlay config that does not influences the Job's hash
        """
        self.set_vis_name("Build Global Cache")

        self.config, self.post_config = BuildGenericSeq2SeqGlobalCacheJob.create_config(
            crp=crp,
            label_tree=label_tree,
            label_scorer=label_scorer,
            search_parameters=search_parameters,
            extra_config=extra_config,
            extra_post_config=extra_post_config,
        )

        self.exe = self.select_exe(crp.flf_tool_exe, "flf-tool")

        self.out_log_file = self.log_file_output_path("build_global_cache", crp, False)
        self.out_global_cache = self.output_path("global.cache", cached=True)

        self.rqmt = {"time": 1, "cpu": 1, "mem": 4}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def create_files(self):
        self.write_config(self.config, self.post_config, "build_global_cache.config")
        with open("dummy.corpus", "wt") as f:
            f.write('<?xml version="1.0" encoding="utf-8" ?>\n<corpus name="dummy"></corpus>')
        with open("dummy.flow", "wt") as f:
            f.write(f'<?xml version="1.0" encoding="utf-8" ?>\n<network><out name="features" /></network>')
        extra_code = (
            ":${THEANO_FLAGS:="
            '}\nexport THEANO_FLAGS="$THEANO_FLAGS,device=cpu,force_device=True"\nexport TF_DEVICE="cpu"'
        )
        self.write_run_script(self.exe, "build_global_cache.config", extra_code=extra_code)

    def run(self):
        self.run_script(1, self.out_log_file)
        shutil.move("global.cache", self.out_global_cache.get_path())

    @classmethod
    def create_config(
        cls,
        crp: rasr.CommonRasrParameters,
        label_tree: LabelTree,
        label_scorer: LabelScorer,
        search_parameters: Optional[dict],
        extra_config: Optional[rasr.RasrConfig],
        extra_post_config: Optional[rasr.RasrConfig],
    ):
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
                "lookahead_language_model": "flf-lattice-tool.network.recognizer.recognizer.lookahead-lm",
            },
        )

        # Apply config from label tree
        label_tree.apply_config(
            "flf-lattice-tool.network.recognizer.recognizer.label-tree",
            config,
            post_config,
        )

        # Optional lexicon overwrite
        if label_tree.lexicon_config is not None:
            config["flf-lattice-tool.lexicon"]._update(label_tree.lexicon_config)

        # Apply config from label scorer and eliminate unnecessary arguments that don't affect the search space (scale, prior)
        label_scorer_reduced = LabelScorer(
            scorer_type=label_scorer.scorer_type,
            scale=1.0,
            label_file=label_scorer.label_file,
            num_classes=label_scorer.num_classes,
            use_prior=False,
            extra_args={key: val for key, val in label_scorer.extra_args.items() if key != "first-order"},
        )

        label_scorer_reduced.apply_config("flf-lattice-tool.network.recognizer.label-scorer", config, post_config)

        # search settings #
        search_config = rasr.RasrConfig()
        if search_parameters is not None:
            for key, val in search_parameters.items():
                search_config[key.replace("_", "-")] = val

        config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # flf network #
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = "1->recognizer:1"
        config.flf_lattice_tool.corpus.file = "dummy.corpus"
        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.links = "sink"
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False
        config.flf_lattice_tool.network.recognizer.search_type = "generic-seq2seq-tree-search"
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = "dummy.flow"
        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        # skip conventional AM or load it without GMM #
        if crp.acoustic_model_config is None:
            config.flf_lattice_tool.network.recognizer.use_acoustic_model = False
        else:
            config.flf_lattice_tool.network.recognizer.use_mixture = False
            if config.flf_lattice_tool.network.recognizer.acoustic_model._get("length") is not None:
                del config.flf_lattice_tool.network.recognizer.acoustic_model["length"]

        # disable scaling
        if config.flf_lattice_tool.network.recognizer.lm._get("scale") is not None:
            del config.flf_lattice_tool.network.recognizer.lm["scale"]

        if config.flf_lattice_tool.network.recognizer.recognizer._get("lookahead-lm") is not None:
            del config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm["scale"]

        config.flf_lattice_tool.network.recognition_mode = "init-only"
        config.flf_lattice_tool.network.search_type = "generic-seq2seq-tree-search"

        config.flf_lattice_tool.global_cache.file = "global.cache"
        config.flf_lattice_tool.global_cache.read_only = False

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, _ = cls.create_config(**kwargs)
        return super().hash({"config": config, "exe": kwargs["crp"].speech_recognizer_exe})


class GenericSeq2SeqSearchJobV2(rasr.RasrCommand, Job):
    __sis_hash_exclude__ = {"num_threads": None}

    def __init__(
        self,
        crp: rasr.CommonRasrParameters,
        feature_flow: rasr.FlowNetwork,
        label_tree: LabelTree,
        label_scorer: LabelScorer,
        rasr_exe: Optional[tk.Path] = None,
        search_parameters: Optional[dict] = None,
        lm_lookahead: bool = True,
        lookahead_options: Optional[dict] = None,
        eval_single_best: bool = True,
        eval_best_in_lattice: bool = True,
        use_gpu: bool = False,
        global_cache: Optional[tk.Path] = None,
        rtf: float = 2,
        mem: float = 8,
        extra_config: Optional[rasr.RasrConfig] = None,
        extra_post_config: Optional[rasr.RasrConfig] = None,
        num_threads: int = 2,
    ):
        self.set_vis_name("Generic Seq2Seq Search")

        self.config, self.post_config = GenericSeq2SeqSearchJobV2.create_config(
            crp=crp,
            feature_flow=feature_flow,
            label_tree=label_tree,
            label_scorer=label_scorer,
            search_parameters=search_parameters,
            lm_lookahead=lm_lookahead,
            lookahead_options=lookahead_options,
            eval_single_best=eval_single_best,
            eval_best_in_lattice=eval_best_in_lattice,
            extra_config=extra_config,
            extra_post_config=extra_post_config,
            global_cache=global_cache,
        )
        self.feature_flow = feature_flow
        if rasr_exe is not None:
            self.rasr_exe = rasr_exe
        else:
            self.rasr_exe = crp.flf_tool_exe
        assert self.rasr_exe is not None

        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu
        self.num_threads = num_threads

        self.out_log_file = self.log_file_output_path("search", crp, True)

        self.out_single_lattice_caches = dict(
            (task_id, self.output_path("lattice.cache.%d" % task_id, cached=True))
            for task_id in range(1, crp.concurrent + 1)
        )
        self.out_lattice_bundle = self.output_path("lattice.bundle", cached=True)
        self.out_lattice_path = util.MultiOutputPath(
            self, "lattice.cache.$(TASK)", self.out_single_lattice_caches, cached=True
        )

        self.rqmt = {
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 24),
            "cpu": num_threads,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))

    def create_files(self):
        self.write_config(self.config, self.post_config, "recognition.config")
        self.feature_flow.write_to_file("feature.flow")
        util.write_paths_to_file(self.out_lattice_bundle, self.out_single_lattice_caches.values())
        extra_code = 'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        # sometimes crash without this
        if not self.use_gpu:
            extra_code += "\nexport CUDA_VISIBLE_DEVICES="

        extra_code += f"\nexport OMP_NUM_THREADS={self.num_threads}"
        extra_code += f"\nexport MKL_NUM_THREADS={self.num_threads}"
        self.write_run_script(self.rasr_exe, "recognition.config", extra_code=extra_code)

    def stop_run(self, task_id):
        print(f"run job {task_id} exceeds specified rqmt and stopped")

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "lattice.cache.%d" % task_id,
            self.out_single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists(f"recognition.log.{task_id}")
        util.delete_if_exists(f"lattice.cache.{task_id}")

    @classmethod
    def create_config(
        cls,
        crp: rasr.CommonRasrParameters,
        feature_flow: rasr.FlowNetwork,
        label_tree: LabelTree,
        label_scorer: LabelScorer,
        search_parameters: Optional[dict] = None,
        lm_lookahead: bool = True,
        lookahead_options: Optional[dict] = None,
        eval_single_best: bool = True,
        eval_best_in_lattice: bool = True,
        extra_config: Optional[rasr.RasrConfig] = None,
        extra_post_config: Optional[rasr.RasrConfig] = None,
        global_cache: Optional[tk.Path] = None,
        **_,
    ):
        # get config from csp #
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
                "lookahead_language_model": "flf-lattice-tool.network.recognizer.recognizer.lookahead-lm",
            },
            parallelize=True,
        )

        # acoustic model maybe used for allophones and state-tying, but no mixture is needed
        # skip conventional AM or load it without GMM
        if crp.acoustic_model_config is None:
            config.flf_lattice_tool.network.recognizer.use_acoustic_model = False
        else:
            config.flf_lattice_tool.network.recognizer.use_mixture = False

        # feature flow #
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = "feature.flow"
        if feature_flow.outputs != {"features"}:
            assert len(feature_flow.outputs) == 1, "not implemented otherwise"
            config.flf_lattice_tool.network.recognizer.feature_extraction.main_port_name = next(
                iter(feature_flow.outputs)
            )

        feature_flow.apply_config(
            "flf-lattice-tool.network.recognizer.feature-extraction",
            config,
            post_config,
        )

        # label tree and optional lexicon overwrite
        label_tree.apply_config(
            "flf-lattice-tool.network.recognizer.recognizer.label-tree",
            config,
            post_config,
        )
        if label_tree.lexicon_config is not None:
            config["flf-lattice-tool.lexicon"]._update(label_tree.lexicon_config)

        # label scorer
        label_scorer.apply_config("flf-lattice-tool.network.recognizer.label-scorer", config, post_config)

        # search settings #
        search_config = rasr.RasrConfig()
        if search_parameters is not None:
            for key, val in search_parameters.items():
                search_config[key.replace("_", "-")] = val

        config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # lookahead settings #
        la_opts = {
            "history_limit": 1,
            "cache_low": 2000,
            "cache_high": 3000,
        }
        if lookahead_options is not None:
            la_opts.update(lookahead_options)

        config.flf_lattice_tool.network.recognizer.recognizer.optimize_lattice = True

        la_config = rasr.RasrConfig()
        la_config._value = lm_lookahead

        if "laziness" in la_opts:
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead_laziness = la_opts["laziness"]

        if lm_lookahead:
            if "history_limit" in la_opts:
                la_config.history_limit = la_opts["history_limit"]
            if "tree_cutoff" in la_opts:
                la_config.tree_cutoff = la_opts["tree_cutoff"]
            if "minimum_representation" in la_opts:
                la_config.minimum_representation = la_opts["minimum_representation"]
            if "lm_lookahead_scale" in la_opts:
                la_config.lm_lookahead_scale = la_opts["lm_lookahead_scale"]
            if "cache_low" in la_opts:
                post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_low = la_opts[
                    "cache_low"
                ]
            if "cache_high" in la_opts:
                post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_high = la_opts[
                    "cache_high"
                ]

        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead = la_config

        # flf network #
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = "1->recognizer:1 0->archive-writer:1 0->evaluator:1"

        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.search_type = "generic-seq2seq-tree-search"
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False

        if label_scorer.config.label_unit == "hmm":
            config.flf_lattice_tool.network.recognizer.links = "expand"
            config.flf_lattice_tool.network.expand.type = "expand-transits"
            config.flf_lattice_tool.network.expand.links = "evaluator archive-writer"
        else:
            config.flf_lattice_tool.network.recognizer.links = "evaluator archive-writer"

        config.flf_lattice_tool.network.evaluator.type = "evaluator"
        config.flf_lattice_tool.network.evaluator.links = "sink:0"
        config.flf_lattice_tool.network.evaluator.word_errors = True
        config.flf_lattice_tool.network.evaluator.single_best = eval_single_best
        config.flf_lattice_tool.network.evaluator.best_in_lattice = eval_best_in_lattice
        config.flf_lattice_tool.network.evaluator.edit_distance.format = "bliss"
        config.flf_lattice_tool.network.evaluator.edit_distance.allow_broken_words = False

        config.flf_lattice_tool.network.archive_writer.type = "archive-writer"
        config.flf_lattice_tool.network.archive_writer.links = "sink:1"
        config.flf_lattice_tool.network.archive_writer.format = "flf"
        config.flf_lattice_tool.network.archive_writer.path = "lattice.cache.$(TASK)"
        post_config.flf_lattice_tool.network.archive_writer.info = True

        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False
        post_config["*"].output_channel.unbuffered = True

        if global_cache is None:
            global_cache = BuildGenericSeq2SeqGlobalCacheJob(
                crp=crp, label_tree=label_tree, label_scorer=label_scorer
            ).out_global_cache

        post_config.flf_lattice_tool.global_cache.read_only = True
        post_config.flf_lattice_tool.global_cache.file = global_cache

        # update parameters #
        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, _ = cls.create_config(**kwargs)
        if kwargs["rasr_exe"] is not None:
            rasr_exe = kwargs["rasr_exe"]
        else:
            rasr_exe = kwargs["crp"].flf_tool_exe
        return super().hash(
            {
                "config": config,
                "feature_flow": kwargs["feature_flow"],
                "exe": rasr_exe,
            }
        )
