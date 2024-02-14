__all__ = ["Seq2SeqAlignmentJob", "ComputeTSEJob"]

import shutil
import statistics

from sisyphus import *

Path = setup_path(__package__)

from .flow import label_alignment_flow
import i6_core.rasr as rasr
import i6_core.util as util
import i6_core.lib.rasr_cache as rasr_cache
from typing import Any, Callable, Dict, Optional, List, Tuple, Union
from collections import Counter


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
            (i, self.output_path("alignment.cache.%d" % i, cached=True)) for i in range(1, self.concurrent + 1)
        )
        self.out_alignment_path = util.MultiOutputPath(
            self,
            "alignment.cache.$(TASK)",
            self.out_single_alignment_caches,
            cached=True,
        )
        self.out_alignment_bundle = self.output_path("alignment.cache.bundle", cached=True)

        if self.word_boundaries:
            self.single_word_boundary_caches = dict(
                (i, self.output_path("word_boundary.cache.%d" % i, cached=True)) for i in range(1, self.concurrent + 1)
            )
            self.word_boundary_path = util.MultiOutputPath(
                self,
                "word_boundary.cache.$(TASK)",
                self.single_word_boundary_caches,
                cached=True,
            )
            self.word_boundary_bundle = self.output_path("word_boundary.cache.bundle", cached=True)

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
        util.write_paths_to_file(self.out_alignment_bundle, self.out_single_alignment_caches.values())
        if self.word_boundaries:
            util.write_paths_to_file(self.word_boundary_bundle, self.single_word_boundary_caches.values())
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
        **kwargs,
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

        config, post_config = rasr.build_config_from_mapping(crp, mapping, parallelize=True)

        # alignment options for the flow nodes
        alignopt = {}
        if alignment_options is not None:
            alignopt.update(alignment_options)
        for node in alignment_flow.get_node_names_by_filter(align_node):
            node_config = config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction[node]
            # alignment node option
            for k, v in align_node_options.items():
                node_config[k] = v
            # alinger search option
            node_config.aligner = rasr.RasrConfig()
            for k, v in alignopt.items():
                node_config.aligner[k] = v
            # scorer
            label_scorer.apply_config("label-scorer", node_config.model_combination, node_config.model_combination)

        alignment_flow.apply_config(
            "acoustic-model-trainer.aligning-feature-extractor.feature-extraction",
            config,
            post_config,
        )

        config.action = "dry"
        config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.file = "alignment.flow"
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
        return super().hash({"config": config, "alignment_flow": alignment_flow, "exe": rasr_exe})


class ComputeTSEJob(Job):
    """
    Compute TSE of some alignment compared to a reference
    """

    def __init__(
        self,
        alignment_cache: tk.Path,
        ref_alignment_cache: tk.Path,
        allophone_file: tk.Path,
        ref_allophone_file: tk.Path,
        silence_phone: str = "[SILENCE]",
        ref_silence_phone: str = "[SILENCE]",
        upsample_factor: int = 1,
        ref_upsample_factor: int = 1,
        seq_tag_transform: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        :param alignment_cache: RASR alignment cache file or bundle for which to compute TSEs
        :param ref_alignment_cache: Reference RASR alignment cache file to compare word boundaries to
        :param allophone_file: Allophone file corresponding to `alignment_cache`
        :param ref_allophone_file: Allophone file corresponding to `ref_alignment_cache`
        :param silence_phone: Silence phoneme string in lexicon corresponding to `allophone_file`
        :param ref_silence_phone: Silence phoneme string in lexicon corresponding to `ref_allophone_file`
        :param upsample_factor: Factor to upsample alignment if it was generated by a model with subsampling
        :param ref_upsample_factor: Factor to upsample reference alignment if it was generated by a model with subsampling
        :param seq_tag_transform: Function that transforms seq tag in alignment cache such that it matches the seq tags in the reference
        """
        self.alignment_cache = alignment_cache
        self.allophone_file = allophone_file
        self.silence_phone = silence_phone
        self.upsample_factor = upsample_factor

        self.ref_alignment_cache = ref_alignment_cache
        self.ref_allophone_file = ref_allophone_file
        self.ref_silence_phone = ref_silence_phone
        self.ref_upsample_factor = ref_upsample_factor

        self.seq_tag_transform = seq_tag_transform

        self.out_tse_frames = self.output_var("tse_frames")
        self.out_word_start_frame_differences = self.output_var("start_frame_differences")
        self.out_plot_word_start_frame_differences = self.output_path("start_frame_differences.png")
        self.out_word_end_frame_differences = self.output_var("end_frame_differences")
        self.out_plot_word_end_frame_differences = self.output_path("end_frame_differences.png")
        self.out_boundary_frame_differences = self.output_var("boundary_frame_differences")
        self.out_plot_boundary_frame_differences = self.output_path("boundary_frame_differences.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)
        yield Task("plot", resume="plot", mini_task=True)

    @staticmethod
    def _compute_word_boundaries(
        alignments: Union[rasr_cache.FileArchive, rasr_cache.FileArchiveBundle],
        allophone_map: List[str],
        seq_tag: str,
        silence_phone: str,
        upsample_factor: int,
    ) -> Tuple[List[int], List[int]]:
        word_starts = []
        word_ends = []

        align_seq = alignments.read(seq_tag, "align")
        assert align_seq is not None

        seq_allophones = [allophone_map[item[1]] for item in align_seq]
        if upsample_factor > 1:
            seq_allophones = sum([[allo] * upsample_factor for allo in seq_allophones], [])

        for t, allophone in enumerate(seq_allophones):
            is_sil = silence_phone in allophone

            if not is_sil and "@i" in allophone and (t == 0 or seq_allophones[t - 1] != allophone):
                word_starts.append(t)

            if (
                not is_sil
                and "@f" in allophone
                and (t == len(seq_allophones) - 1 or seq_allophones[t + 1] != allophone)
            ):
                word_ends.append(t)

        return word_starts, word_ends

    def run(self) -> None:
        start_differences = Counter()
        end_differences = Counter()
        differences = Counter()

        alignments = rasr_cache.open_file_archive(self.alignment_cache.get())
        alignments.setAllophones(self.allophone_file.get())
        if isinstance(alignments, rasr_cache.FileArchiveBundle):
            allophone_map = next(iter(alignments.archives.values())).allophones
        else:
            allophone_map = alignments.allophones

        ref_alignments = rasr_cache.open_file_archive(self.ref_alignment_cache.get())
        ref_alignments.setAllophones(self.ref_allophone_file.get())
        if isinstance(ref_alignments, rasr_cache.FileArchiveBundle):
            ref_allophone_map = next(iter(ref_alignments.archives.values())).allophones
        else:
            ref_allophone_map = ref_alignments.allophones

        file_list = [tag for tag in alignments.file_list() if not tag.endswith(".attribs")]

        for idx, seq_tag in enumerate(file_list, start=1):
            word_starts, word_ends = self._compute_word_boundaries(
                alignments, allophone_map, seq_tag, self.silence_phone, self.upsample_factor
            )

            if self.seq_tag_transform is not None:
                ref_seq_tag = self.seq_tag_transform(seq_tag)
            else:
                ref_seq_tag = seq_tag

            ref_word_starts, ref_word_ends = self._compute_word_boundaries(
                ref_alignments, ref_allophone_map, ref_seq_tag, self.ref_silence_phone, self.ref_upsample_factor
            )

            seq_word_start_diffs = [start - ref_start for start, ref_start in zip(word_starts, ref_word_starts)]
            seq_word_end_diffs = [end - ref_end for end, ref_end in zip(word_ends, ref_word_ends)]
            seq_differences = seq_word_start_diffs + seq_word_end_diffs

            start_differences.update(seq_word_start_diffs)
            end_differences.update(seq_word_end_diffs)
            differences.update(seq_differences)

            seq_tse = statistics.mean(abs(diff) for diff in seq_differences)

            print(
                f"Sequence {seq_tag} ({idx} / {len(file_list)}):\n    Word start distances are {seq_word_start_diffs}\n    Word end distances are {seq_word_end_diffs}\n    Sequence TSE is {seq_tse} frames"
            )

        self.out_word_start_frame_differences.set(
            {key: start_differences[key] for key in sorted(start_differences.keys())}
        )
        self.out_word_end_frame_differences.set({key: end_differences[key] for key in sorted(end_differences.keys())})
        self.out_boundary_frame_differences.set({key: differences[key] for key in sorted(differences.keys())})
        self.out_tse_frames.set(statistics.mean(abs(diff) for diff in differences.elements()))

    def plot(self):
        for descr, dict_file, plot_file in [
            (
                "start",
                self.out_word_start_frame_differences.get_path(),
                self.out_plot_word_start_frame_differences.get_path(),
            ),
            (
                "end",
                self.out_word_end_frame_differences.get_path(),
                self.out_plot_word_end_frame_differences.get_path(),
            ),
            (
                "boundary",
                self.out_boundary_frame_differences.get_path(),
                self.out_plot_boundary_frame_differences.get_path(),
            ),
        ]:
            with open(dict_file, "r") as f:
                diff_dict = eval(f.read())

            ranges = [-30, -20, -15, -10, -5, -1, 2, 6, 11, 16, 21, 31]

            range_strings = []
            range_strings.append(f"<{ranges[0]}")
            for idx in range(1, len(ranges)):
                range_strings.append(f"{ranges[idx - 1]} - {ranges[idx] - 1}")
            range_strings.append(f">{ranges[-1] - 1}")

            range_counts = [0] * (len(ranges) + 1)

            for key, count in diff_dict.items():
                idx = 0
                while idx < len(ranges) and ranges[idx] <= key:
                    idx += 1

                range_counts[idx] += count

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(range_strings, range_counts, color="skyblue")
            plt.xlabel(f"Word {descr} shift (frames)")
            plt.ylabel("Counts")
            plt.title(f"Word {descr} shift counts")
            plt.xticks(rotation=45)

            plt.savefig(plot_file)
