__all__ = ["LabelSyncSearchLmImageAndGlobalCacheJob", "LabelSyncSearchJob"]

from sisyphus import *

Path = setup_path(__package__)

import os
import shutil
import copy

from i6_core import lm, rasr, util
from i6_core.returnn.extract_prior import ReturnnComputePriorJob


class LabelSyncSearchLmImageAndGlobalCacheJob(rasr.RasrCommand, Job):
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
        ) = LabelSyncSearchLmImageAndGlobalCacheJob.create_config(**kwargs)
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
        self.rqmt = {"time": 4.5, "cpu": 1, "mem": mem, "qsub_args": "-q 40C*"}

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
        config.flf_lattice_tool.network.recognizer.search_type = "label-sync-search"
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
            search_config = LabelSyncSearchJob.get_default_search_config()
            config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # update extra params #
        if cls.find_arpa_lms(copy.deepcopy(extra_config)):
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


class LabelSyncSearchJob(rasr.RasrCommand, Job):
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
        lm_gc_job_mem=2,
        lm_gc_job_default_search=False,
    ):  # TODO set this to true later
        self.set_vis_name("Label Synchronized Search")
        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = LabelSyncSearchJob.create_config(**kwargs)
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
            "cpu": 2,
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
        extra_code = (
            ":${{THEANO_FLAGS:="
            "}}\n"
            'export THEANO_FLAGS="$THEANO_FLAGS,device={0},force_device=True"\n'
            'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        )
        # sometimes crash without this
        if not self.use_gpu:
            extra_code += "\nexport CUDA_VISIBLE_DEVICES="
        self.write_run_script(self.exe, "recognition.config", extra_code=extra_code)

    # TODO maybe not needed
    def stop_run(self, task_id):
        print("run job %d exceeds specified rqmt and stoped" % task_id)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "lattice.cache.%d" % task_id, self.out_single_lattice_caches[task_id].get_path()
        )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("recognition.log.%d" % task_id)
        util.delete_if_exists("lattice.cache.%d" % task_id)

    # other hidden params set in extra_config #
    @classmethod
    def get_default_search_config(cls, lm_scale=1.0, **kwargs):
        search_config = rasr.RasrConfig()
        # TODO relation not clear yet, so far like this
        search_config.label_pruning = 10.0
        search_config.label_pruning_limit = 20000
        search_config.word_end_pruning = 0.5
        search_config.word_end_pruning_limit = 5000

        search_config.create_lattice = True
        search_config.optimize_lattice = True
        return search_config

    @classmethod
    def get_default_lookahead_config(cls, scale=1.0, **kwargs):
        lookahead_config = rasr.RasrConfig()
        lookahead_config.history_limit = -1
        lookahead_config.cache_size_low = 2000
        lookahead_config.cache_size_high = 3000
        # lookahead scale has to be explicitly set now (independent of lm.scale) #
        lookahead_config.scale = scale
        return lookahead_config

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        label_tree,
        label_scorer,
        search_options=None,
        lm_lookahead=True,
        lookahead_options=None,
        eval_single_best=True,
        eval_best_in_lattice=True,
        extra_config=None,
        extra_post_config=None,
        sprint_exe=None,
        lm_gc_job=None,
        lm_gc_job_local=False,
        lm_gc_job_mem=16,
        lm_gc_job_default_search=False,
        **kwargs
    ):

        # optional individual lm-image and global-cache job #
        if lm_gc_job is None:
            lm_gc_job = LabelSyncSearchLmImageAndGlobalCacheJob(
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
        lm_scale = config.flf_lattice_tool.network.recognizer.lm.scale
        search_config = cls.get_default_search_config(lm_scale)
        if search_options is not None:
            for key in search_options.keys():
                search_config[key] = search_options[key]
        config.flf_lattice_tool.network.recognizer.recognizer._update(search_config)

        # lookahead settings #
        config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead._value = (
            lm_lookahead
        )
        if lm_lookahead:
            lookahead_config = cls.get_default_lookahead_config(lm_scale)
            if lookahead_options is not None:
                for key in lookahead_options.keys():
                    lookahead_config[key] = lookahead_options[key]
            config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead._update(
                lookahead_config
            )

        # flf network #
        config.flf_lattice_tool.network.initial_nodes = "segment"
        config.flf_lattice_tool.network.segment.type = "speech-segment"
        config.flf_lattice_tool.network.segment.links = (
            "1->recognizer:1 0->archive-writer:1 0->evaluator:1"
        )

        config.flf_lattice_tool.network.recognizer.type = "recognizer"
        config.flf_lattice_tool.network.recognizer.search_type = "label-sync-search"
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

        # update parameters #
        config._update(extra_config)
        post_config._update(extra_post_config)

        # image and cache #
        arpa_lms = LabelSyncSearchLmImageAndGlobalCacheJob.find_arpa_lms(config)
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


# prior computation using exact settings from training (default using cpu only) #
class LabelSyncComputePriorJob(ReturnnComputePriorJob):
    def __init__(self,
                 model_checkpoint,
                 returnn_config,
                 train_path, num_classes=None,
                 context_size=0, valid_context=[], prior_config=None, plot_prior=True,
                 *,  # args below are keyword only
                 log_verbosity=3, device='cpu',
                 time_rqmt=4, mem_rqmt=12, cpu_rqmt=3, qsub_rqmt=None,
                 returnn_python_exe=None, returnn_root=None):
        """

        :param Checkpoint model_checkpoint:
        :param train_path:
        :param num_classes:
        :param context_size:
        :param valid_context:
        :param prior_config:
        :param plot_prior:
        :param log_verbosity:
        :param device:
        :param time_rqmt:
        :param mem_rqmt:
        :param cpu_rqmt:
        :param qsub_rqmt:
        :param returnn_python_exe:
        :param returnn_root:
        """

        super().__init__(model_checkpoint=model_checkpoint,
                         returnn_config=returnn_config,
                         log_verbosity=log_verbosity, device=device,
                         time_rqmt=time_rqmt, mem_rqmt=mem_rqmt, cpu_rqmt=cpu_rqmt,
                         returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)

        self.train_path = train_path

        # possible context-dependent prior
        self.context_size = context_size
        if self.context_size > 0:
            if not valid_context:
                assert num_classes is not None
                import itertools
                self.valid_context = list(itertools.permutations(range(num_classes), self.context_size))
            else: self.valid_context = valid_context
            if isinstance(prior_config, dict):
                prior_config = CRNNConfig(prior_config)
            assert isinstance(prior_config, (str, tk.Path, CRNNConfig))
            self.prior_config = prior_config
            self.prior = self.output_path('prior') # overwrite original
            self.plot_prior=plot_prior

        if qsub_rqmt is not None:
            self.rqmt['qsub_args'] = qsub_rqmt
        self.device = device
        if device != 'gpu':
            self.rqmt['gpu'] = 0
            self.rqmt['time'] = 168
            self.rqmt['qsub_args'] = '-q 40C*' # somehow crashes with older cpus

    def tasks(self):
        yield Task('create_files', mini_task=True)
        if self.context_size > 0:
            yield Task('run_prior', resume='run_prior', rqmt=self.rqmt, args=range(1, len(self.valid_context)+1))
            yield Task('finalize', resume='finalize', mini_task=True)
        else:
            yield Task('run_prior', resume='run_prior', rqmt=self.rqmt)
            yield Task('plot', resume='plot', mini_task=True)

    def create_run_file(self):
        msg = script_header(tk.uncached_path(self.crnn_python_exe))

        if self.context_size > 0:
            if isinstance(self.prior_config, CRNNConfig):
                config_path = 'crnn.config'
                self.prior_config.write(config_path)
            else:
                config_path = self.prior_config
            name = ''
            post_msg = ''
            for ctx in range(1, self.context_size+1):
                name += '$%d-' %ctx
                post_msg += ' ++label_context%d $%d' %(ctx, ctx)
            output_path = self.prior.get_path() + '.' + name[:-1] + '.txt'
        else:
            config_path = self.crnn_config_file.get_path()
            output_path = self.prior_txt.get_path()
            post_msg = ''

        msg += ' '.join(['$PY', os.path.join(tk.uncached_path(self.crnn_root), 'rnn.py'), config_path, '--task', 'compute_priors', '--output_file', output_path, '++load_epoch', str(self.epoch), '++tf_log_dir', 'None', '++device', self.device])
        msg += post_msg
        with open('rnn.sh', 'wt') as f:
            f.write(msg)

    def create_files(self):
        self.create_run_file()
        os.chmod('rnn.sh', stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        # copy sprint config and flow (same as used in training) #
        for f in os.listdir(self.train_path):
            if f.endswith('.flow') or f.endswith('.config'):
                shutil.copy(os.path.join(self.train_path, f), '.')

    def run_prior(self, task_id=None):
        if self.context_size > 0:
            assert task_id is not None
            args = list( map(str, self.valid_context[task_id-1]) )
            sp.check_call(['./rnn.sh'] + args)
        else:
            sp.check_call(['./rnn.sh'])

    def finalize(self):
        assert self.context_size > 0
        import numpy as np
        for ctx in self.valid_context:
            # fixed format to match RASR
            output = self.prior.get_path() + '.' + '-'.join(list(map(str, ctx))) + '.txt'
            with open(output, 'rt') as f:
                merged_scores = np.loadtxt(f, delimiter=' ')
            with open(output.replace('txt', 'xml'), 'wt') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n' % len(merged_scores))
                f.write(' '.join('%.20e' % s for s in merged_scores) + '\n')
                f.write('</vector-f32>')

            if self.plot_prior:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.clf()
                xdata = range(len(merged_scores))
                plt.semilogy(xdata, np.exp(merged_scores))
                plt.xlabel('emission idx')
                plt.ylabel('prior')
                plt.grid(True)
                plt.savefig(output.replace('txt', 'png'))

        # just for sisyphus dependency
        open(self.prior.get_path(), 'a').close()

    @classmethod
    def hash(cls, kwargs):
        d = { 'model'           : kwargs['model'],
              'train_path'      : kwargs['train_path'],
              'crnn_python_exe' : kwargs['crnn_python_exe'],
              'crnn_root'       : kwargs['crnn_root']
              }
        if kwargs['context_size'] > 0:
            d.update({
                'context_size' : kwargs['context_size'],
                'valid_context': kwargs['valid_context'],
                'prior_config' : kwargs['prior_config']
            })
            if not kwargs['valid_context']:
                d['num_classes'] = kwargs['num_classes']
        return Job.hash(d)
