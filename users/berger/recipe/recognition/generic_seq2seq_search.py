__all__ = ["GenericSeq2SeqLmImageAndGlobalCacheJob", "GenericSeq2SeqSearchJob"]

from sisyphus import *

Path = setup_path(__package__)

import shutil
import copy

from i6_core import rasr, util


class GenericSeq2SeqLmImageAndGlobalCacheJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        label_tree,
        label_scorer,
        extra_config=None,
        extra_post_config=None,
        default_search=False,
        mem=4,
        local_job=False,
        sprint_exe=None,
    ):
        self.set_vis_name("LabelSyncSearch Precomptue LM Image/Global Cache")
        kwargs = locals()
        del kwargs["self"]

        (
            self.config,
            self.post_config,
            self.num_images,
        ) = GenericSeq2SeqLmImageAndGlobalCacheJob.create_config(**kwargs)
        if sprint_exe is None:
            sprint_exe = crp.flf_tool_exe
        self.exe = self.select_exe(sprint_exe, "flf-tool")
        self.log_file = self.log_file_output_path("image-cache", crp, False)
        self.lm_images = {
            i: self.output_path("lm-%d.image" % i, cached=True)
            for i in range(1, self.num_images + 1)
        }
        self.global_cache = self.output_path("global.cache", cached=True)

        self.local_job = local_job
        self.rqmt = {"time": 1, "cpu": 1, "mem": mem}

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=self.local_job)

    def create_files(self):
        self.write_config(self.config, self.post_config, "image-cache.config")
        with open("dummy.corpus", "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8" ?>\n<corpus name="dummy"></corpus>'
            )
        with open("dummy.flow", "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="utf-8" ?>\n<network><out name="features" /></network>'
            )
        extra_code = (
            ":${THEANO_FLAGS:="
            '}\nexport THEANO_FLAGS="$THEANO_FLAGS,device=cpu,force_device=True"\nexport TF_DEVICE="cpu"'
        )
        self.write_run_script(self.exe, "image-cache.config", extra_code=extra_code)

    def run(self):
        self.run_script(1, self.log_file)
        for i in range(1, self.num_images + 1):
            shutil.move("lm-%d.image" % i, self.lm_images[i].get_path())
        shutil.move("global.cache", self.global_cache.get_path())

    def cleanup_before_run(self, cmd, retry, *args):
        util.backup_if_exists("image-cache.log")

    @classmethod
    def find_arpa_lms(cls, config):
        result = []
        # scoring lm #
        lm_config = config.flf_lattice_tool.network.recognizer.lm
        if lm_config.type == "ARPA" and lm_config._get("image") is None:
            result.append(lm_config)
        elif lm_config.type == "combine":
            for i in range(1, lm_config.num_lms + 1):
                sub_lm_config = lm_config["lm-%d" % i]
                if sub_lm_config.type == "ARPA" and sub_lm_config._get("image") is None:
                    result.append(sub_lm_config)
        # lookahead lm #
        separate_lookahead_lm = (
            config.flf_lattice_tool.network.recognizer.recognizer.separate_lookahead_lm
        )
        lookahead_lm_config = (
            config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm
        )
        if separate_lookahead_lm:
            if (
                lookahead_lm_config.type == "ARPA"
                and lookahead_lm_config._get("image") is None
            ):
                result.append(lookahead_lm_config)
        # recombination lm #
        separate_recombination_lm = (
            config.flf_lattice_tool.network.recognizer.recognizer.separate_recombination_lm
        )
        recombination_lm_config = (
            config.flf_lattice_tool.network.recognizer.recognizer.recombination_lm
        )
        if separate_recombination_lm:
            if (
                recombination_lm_config.type == "ARPA"
                and recombination_lm_config._get("image") is None
            ):
                result.append(recombination_lm_config)
        return result

    @classmethod
    def create_config(
        cls,
        crp,
        label_tree,
        label_scorer,
        extra_config=None,
        extra_post_config=None,
        default_search=False,
        **kwargs
    ):
        # get config from csp #
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
            },
        )

        # label tree and optional lexicon overwrite #
        label_tree.apply_config(
            "flf-lattice-tool.network.recognizer.recognizer.label-tree",
            config,
            post_config,
        )
        if label_tree.lexicon_config is not None:
            config["flf-lattice-tool.lexicon"]._update(label_tree.lexicon_config)
        # label scorer #
        label_scorer.apply_config(
            "flf-lattice-tool.network.recognizer.label-scorer", config, post_config
        )

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
        config.flf_lattice_tool.network.recognizer.search_type = (
            "generic-seq2seq-tree-search"
        )
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = (
            "dummy.flow"
        )
        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False

        # skip conventional AM or load it without GMM #
        if crp.acoustic_model_config is None:
            config.flf_lattice_tool.network.recognizer.use_acoustic_model = False
        else:
            config.flf_lattice_tool.network.recognizer.use_mixture = False
            del config.flf_lattice_tool.network.recognizer.acoustic_model["length"]

        # disable scaling #
        del config.flf_lattice_tool.network.recognizer.label_scorer["scale"]
        del config.flf_lattice_tool.network.recognizer.label_scorer["priori-scale"]
        del config.flf_lattice_tool.network.recognizer.lm["scale"]

        # unify search/pruning (maybe lm-scale dependent) #
        if default_search:
            search_config = GenericSeq2SeqSearchJob.get_default_search_config()
            config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # update extra params #
        if extra_config and cls.find_arpa_lms(copy.deepcopy(extra_config)):
            config._update(extra_config)
        else:
            post_config._update(
                extra_config
            )  # mainly run-time params: image/cache independent
        post_config._update(extra_post_config)

        # lm images #
        arpa_lms = cls.find_arpa_lms(config)
        for i, lm_config in enumerate(arpa_lms):
            lm_config.image = "lm-%d.image" % (i + 1)

        # global cache #
        config.flf_lattice_tool.global_cache.file = "global.cache"

        return config, post_config, len(arpa_lms)

    @classmethod
    def hash(cls, kwargs):
        config, post_config, num_images = cls.create_config(**kwargs)
        sprint_exe = kwargs["sprint_exe"]
        if sprint_exe is None:
            sprint_exe = kwargs["crp"].flf_tool_exe
        return super().hash({"config": config, "exe": sprint_exe})


class GenericSeq2SeqSearchJob(rasr.RasrCommand, Job):
    def __init__(
        self,
        crp,
        feature_flow,
        label_tree,
        label_scorer,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        eval_single_best=True,
        eval_best_in_lattice=True,
        use_gpu=False,
        rtf=2,
        mem=8,
        hard_rqmt=False,
        extra_config=None,
        extra_post_config=None,
        sprint_exe=None,  # allow separat executable than default settings
        lm_gc_job=None,
        lm_gc_job_local=False,
        lm_gc_job_mem=8,
        lm_gc_job_default_search=False,
    ):  # TODO set this to true later
        self.set_vis_name("Generic Seq2Seq Search")
        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = GenericSeq2SeqSearchJob.create_config(**kwargs)
        self.feature_flow = feature_flow
        if sprint_exe is None:
            sprint_exe = crp.flf_tool_exe
        self.exe = self.select_exe(sprint_exe, "flf-tool")
        self.concurrent = crp.concurrent
        self.use_gpu = use_gpu

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
            "time": max(crp.corpus_duration * rtf / crp.concurrent, 4.5),
            "cpu": 3,
            "gpu": 1 if self.use_gpu else 0,
            "mem": mem,
        }
        # no automatic resume with doubled rqmt
        self.hard_rqmt = hard_rqmt

    def tasks(self):
        yield Task("create_files", mini_task=True)
        if self.hard_rqmt:  # TODO
            resume = None
        else:
            resume = "run"
        yield Task(
            "run", resume=resume, rqmt=self.rqmt, args=range(1, self.concurrent + 1)
        )

    def create_files(self):
        self.write_config(self.config, self.post_config, "recognition.config")
        self.feature_flow.write_to_file("feature.flow")
        util.write_paths_to_file(
            self.out_lattice_bundle, self.out_single_lattice_caches.values()
        )
        extra_code = 'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        # sometimes crash without this
        if not self.use_gpu:
            extra_code += "\nexport CUDA_VISIBLE_DEVICES="
        extra_code += "\nexport OMP_NUM_THREADS=%i" % self.rqmt["cpu"]
        self.write_run_script(self.exe, "recognition.config", extra_code=extra_code)

    # TODO maybe not needed
    def stop_run(self, task_id):
        print("run job %d exceeds specified rqmt and stoped" % task_id)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "lattice.cache.%d" % task_id,
            self.out_single_lattice_caches[task_id].get_path(),
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("recognition.log.%d" % task_id)
        util.delete_if_exists("lattice.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        label_tree,
        label_scorer,
        search_parameters=None,
        lm_lookahead=True,
        lookahead_options=None,
        eval_single_best=True,
        eval_best_in_lattice=True,
        extra_config=None,
        extra_post_config=None,
        sprint_exe=None,
        lm_gc_job=None,
        lm_gc_job_local=True,
        lm_gc_job_mem=6,
        lm_gc_job_default_search=False,
        **kwargs
    ):

        # optional individual lm-image and global-cache job #
        if lm_gc_job is None:
            lm_gc_job = GenericSeq2SeqLmImageAndGlobalCacheJob(
                crp,
                label_tree,
                label_scorer,
                extra_config,
                extra_post_config,
                mem=lm_gc_job_mem,
                local_job=lm_gc_job_local,
                sprint_exe=sprint_exe,
                default_search=lm_gc_job_default_search,
            )

        # get config from csp #
        config, post_config = rasr.build_config_from_mapping(
            crp,
            {
                "corpus": "flf-lattice-tool.corpus",
                "lexicon": "flf-lattice-tool.lexicon",
                "acoustic_model": "flf-lattice-tool.network.recognizer.acoustic-model",
                "language_model": "flf-lattice-tool.network.recognizer.lm",
            },
            parallelize=True,
        )

        # acoustic model maybe used for allophones and state-tying, but no mixture is needed #
        # skip conventional AM or load it without GMM #
        if crp.acoustic_model_config is None:
            config.flf_lattice_tool.network.recognizer.use_acoustic_model = False
        else:
            config.flf_lattice_tool.network.recognizer.use_mixture = False

        # feature flow #
        config.flf_lattice_tool.network.recognizer.feature_extraction.file = (
            "feature.flow"
        )
        feature_flow.apply_config(
            "flf-lattice-tool.network.recognizer.feature-extraction",
            config,
            post_config,
        )

        # label tree and optional lexicon overwrite #
        label_tree.apply_config(
            "flf-lattice-tool.network.recognizer.recognizer.label-tree",
            config,
            post_config,
        )
        if label_tree.lexicon_config is not None:
            config["flf-lattice-tool.lexicon"]._update(label_tree.lexicon_config)

        # label scorer #
        label_scorer.apply_config(
            "flf-lattice-tool.network.recognizer.label-scorer", config, post_config
        )

        # search settings #
        search_config = rasr.RasrConfig()
        if search_parameters is not None:
            for key in search_parameters.keys():
                search_config[key] = search_parameters[key]
        config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # lookahead settings #
        la_opts = {
            "history_limit": 1,
            "cache_low": 2000,
            "cache_high": 3000,
        }
        if lookahead_options is not None:
            la_opts.update(lookahead_options)

        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead = (
            rasr.RasrConfig()
        )
        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead._value = (
            lm_lookahead
        )
        if "laziness" in la_opts:
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead_laziness = la_opts[
                "laziness"
            ]
        config.flf_lattice_tool.network.recognizer.recognizer.optimize_lattice = True
        if lm_lookahead:
            if "history_limit" in la_opts:
                config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.history_limit = la_opts[
                    "history_limit"
                ]
            if "tree_cutoff" in la_opts:
                config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.tree_cutoff = la_opts[
                    "tree_cutoff"
                ]
            if "minimum_representation" in la_opts:
                config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.minimum_representation = la_opts[
                    "minimum_representation"
                ]
            if "lm_lookahead_scale" in la_opts:
                config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.lm_lookahead_scale = la_opts[
                    "lm_lookahead_scale"
                ]
            if "cache_low" in la_opts:
                post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_low = la_opts[
                    "cache_low"
                ]
            if "cache_high" in la_opts:
                post_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.cache_size_high = la_opts[
                    "cache_high"
                ]

        # flf network #
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "1->recognizer:1 0->archive-writer:1 0->evaluator:1"
        )

        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.search_type = (
            "generic-seq2seq-tree-search"
        )
        config.flf_lattice_tool.network.recognizer.apply_non_word_closure_filter = False
        config.flf_lattice_tool.network.recognizer.add_confidence_score = False
        config.flf_lattice_tool.network.recognizer.apply_posterior_pruning = False

        if label_scorer.config.label_unit == "hmm":
            config.flf_lattice_tool.network.recognizer.links = "expand"
            config.flf_lattice_tool.network.expand.type = "expand-transits"
            config.flf_lattice_tool.network.expand.links = "evaluator archive-writer"
        else:
            config.flf_lattice_tool.network.recognizer.links = (
                "evaluator archive-writer"
            )

        config.flf_lattice_tool.network.evaluator.type = "evaluator"
        config.flf_lattice_tool.network.evaluator.links = "sink:0"
        config.flf_lattice_tool.network.evaluator.word_errors = True
        config.flf_lattice_tool.network.evaluator.single_best = eval_single_best
        config.flf_lattice_tool.network.evaluator.best_in_lattice = eval_best_in_lattice
        config.flf_lattice_tool.network.evaluator.edit_distance.format = "bliss"
        config.flf_lattice_tool.network.evaluator.edit_distance.allow_broken_words = (
            False
        )

        config.flf_lattice_tool.network.archive_writer.type = "archive-writer"
        config.flf_lattice_tool.network.archive_writer.links = "sink:1"
        config.flf_lattice_tool.network.archive_writer.format = "flf"
        config.flf_lattice_tool.network.archive_writer.path = "lattice.cache.$(TASK)"
        post_config.flf_lattice_tool.network.archive_writer.info = True

        config.flf_lattice_tool.network.sink.type = "sink"
        post_config.flf_lattice_tool.network.sink.warn_on_empty_lattice = True
        post_config.flf_lattice_tool.network.sink.error_on_empty_lattice = False
        post_config["*"].output_channel.unbuffered = True

        # update parameters #
        config._update(extra_config)
        post_config._update(extra_post_config)

        # image and cache #
        arpa_lms = GenericSeq2SeqLmImageAndGlobalCacheJob.find_arpa_lms(config)
        assert (
            len(arpa_lms) == lm_gc_job.num_images
        ), "mismatch between image-cache config and recognition config"
        for i, lm_config in enumerate(arpa_lms):
            lm_config.image = lm_gc_job.lm_images[i + 1]

        if post_config.flf_lattice_tool.global_cache._get("file") is None:
            post_config.flf_lattice_tool.global_cache.read_only = True
            post_config.flf_lattice_tool.global_cache.file = lm_gc_job.global_cache

        return config, post_config

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        sprint_exe = kwargs["sprint_exe"]
        if sprint_exe is None:
            sprint_exe = kwargs["crp"].flf_tool_exe
        return super().hash(
            {
                "config": config,
                "feature_flow": kwargs["feature_flow"],
                "exe": sprint_exe,
            }
        )
