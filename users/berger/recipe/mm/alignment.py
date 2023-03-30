__all__ = ["Seq2SeqAlignmentJob"]

import shutil

from sisyphus import *

Path = setup_path(__package__)

from .flow import label_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util


class Seq2SeqAlignmentJob(rasr.RasrCommand, Job):
    """
    Modified alignment job for Weis LabelSyncDecoder RASR branch

    """

    def __init__(
        self,
        crp,
        feature_flow,
        label_scorer,
        alignment_options,
        word_boundaries=False,
        align_node_options={},
        use_gpu=False,
        rtf=1.0,
        rasr_exe=None,
        extra_config=None,
        extra_post_config=None,
    ):
        """
        :param recipe.rasr.csp.CommonSprintParameters crp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param bool label_aligner:
        :param recipe.rasr.LabelScorer label_scorer:
        :param dict[str] align_node_options:
        :param bool use_gpu:
        :param float rtf:
        :param extra_config:
        :param extra_post_config:
        """

        assert label_scorer is not None, "need label scorer for label aligner"
        self.set_vis_name("Alignment")

        kwargs = locals()
        del kwargs["self"]

        self.config, self.post_config = Seq2SeqAlignmentJob.create_config(**kwargs)
        self.alignment_flow = Seq2SeqAlignmentJob.create_flow(**kwargs)
        self.concurrent = crp.concurrent
        if rasr_exe is None:
            rasr_exe = crp.acoustic_model_trainer_exe
        self.exe = self.select_exe(rasr_exe, "acoustic-model-trainer")
        self.use_gpu = use_gpu
        self.word_boundaries = word_boundaries

        self.out_log_file = self.log_file_output_path("alignment", crp, True)
        self.out_single_alignment_caches = dict(
            (i, self.output_path("alignment.cache.%d" % i, cached=True))
            for i in range(1, self.concurrent + 1)
        )
        self.out_alignment_path = util.MultiOutputPath(
            self,
            "alignment.cache.$(TASK)",
            self.out_single_alignment_caches,
            cached=True,
        )
        self.out_alignment_bundle = self.output_path(
            "alignment.cache.bundle", cached=True
        )

        if self.word_boundaries:
            self.single_word_boundary_caches = dict(
                (i, self.output_path("word_boundary.cache.%d" % i, cached=True))
                for i in range(1, self.concurrent + 1)
            )
            self.word_boundary_path = util.MultiOutputPath(
                self,
                "word_boundary.cache.$(TASK)",
                self.single_word_boundary_caches,
                cached=True,
            )
            self.word_boundary_bundle = self.output_path(
                "word_boundary.cache.bundle", cached=True
            )

        self.rqmt = {
            "time": max(rtf * crp.corpus_duration / crp.concurrent, 0.5),
            "cpu": 1,
            "gpu": 1 if self.use_gpu else 0,
            "mem": 2,
        }

    def tasks(self):
        rqmt = self.rqmt.copy()
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=rqmt, args=range(1, self.concurrent + 1))

    def create_files(self):
        self.write_config(self.config, self.post_config, "alignment.config")
        self.alignment_flow.write_to_file("alignment.flow")
        util.write_paths_to_file(
            self.out_alignment_bundle, self.out_single_alignment_caches.values()
        )
        if self.word_boundaries:
            util.write_paths_to_file(
                self.word_boundary_bundle, self.single_word_boundary_caches.values()
            )
        extra_code = 'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
        self.write_run_script(self.exe, "alignment.config", extra_code=extra_code)

    def run(self, task_id):
        self.run_script(task_id, self.out_log_file[task_id])
        shutil.move(
            "alignment.cache.%d" % task_id,
            self.out_single_alignment_caches[task_id].get_path(),
        )
        if self.word_boundaries:
            shutil.move(
                "word_boundary.cache.%d" % task_id,
                self.single_word_boundary_caches[task_id].get_path(),
            )

    def cleanup_before_run(self, cmd, retry, task_id, *args):
        util.backup_if_exists("alignment.log.%d" % task_id)
        util.delete_if_exists("alignment.cache.%d" % task_id)
        if self.word_boundaries:
            util.delete_if_zero("word_boundary.cache.%d" % task_id)

    @classmethod
    def create_config(
        cls,
        crp,
        feature_flow,
        alignment_options,
        word_boundaries,
        label_scorer,
        align_node_options,
        extra_config,
        extra_post_config,
        **kwargs
    ):
        """
        :param recipe.rasr.csp.CommonSprintParameters csp:
        :param feature_flow:
        :param rasr.FeatureScorer feature_scorer:
        :param dict[str] alignment_options:
        :param bool word_boundaries:
        :param recipe.rasr.LabelScorer label_scorer:
        :param dict[str] align_node_options:
        :param extra_config:
        :param extra_post_config:
        :return: config, post_config
        :rtype: (rasr.SprintConfig, rasr.SprintConfig)
        """

        alignment_flow = cls.create_flow(feature_flow)
        align_node = "speech-seq2seq-alignment"
        assert label_scorer is not None, "need label scorer for seq2seq aligner"

        # acoustic model + lexicon for the flow nodes
        mapping = {
            "corpus": "acoustic-model-trainer.corpus",
            "lexicon": [],
            "acoustic_model": [],
        }
        for node in alignment_flow.get_node_names_by_filter(align_node):
            mapping["lexicon"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.lexicon"
                % node
            )
            mapping["acoustic_model"].append(
                "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.%s.model-combination.acoustic-model"
                % node
            )

        config, post_config = rasr.build_config_from_mapping(
            crp, mapping, parallelize=True
        )

        # alignment options for the flow nodes
        alignopt = {}
        if alignment_options is not None:
            alignopt.update(alignment_options)
        for node in alignment_flow.get_node_names_by_filter(align_node):
            node_config = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[
                node
            ]
            # alignment node option
            for k, v in align_node_options.items():
                node_config[k] = v
            # alinger search option
            node_config.aligner = rasr.RasrConfig()
            for k, v in alignopt.items():
                node_config.aligner[k] = v
            # scorer
            label_scorer.apply_config("label-scorer", node_config, node_config)

        alignment_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = (
            "alignment.flow"
        )
        post_config["*"].allow_overwrite = True

        config._update(extra_config)
        post_config._update(extra_post_config)

        return config, post_config

    @classmethod
    def create_flow(cls, feature_flow, **kwargs):
        return label_alignment_flow(feature_flow, "alignment.cache.$(TASK)")

    @classmethod
    def hash(cls, kwargs):
        config, post_config = cls.create_config(**kwargs)
        alignment_flow = cls.create_flow(**kwargs)
        rasr_exe = kwargs["rasr_exe"]
        if rasr_exe is None:
            rasr_exe = kwargs["crp"].acoustic_model_trainer_exe
        return super().hash(
            {"config": config, "alignment_flow": alignment_flow, "exe": rasr_exe}
        )
