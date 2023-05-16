__all__ = ["FilterAlignmentPlugin"]
from sisyphus import tk

from collections import ChainMap

from i6_core.meta import select_element
import i6_experiments.users.mann.experimental.extractors as extr
import i6_core.corpus as corpus_recipes

class FilterAlignmentPlugin:
    DEFAULT_FEATURE_CORPUS = "train"
    def __init__(self, system, dev_size, prefix="crnn", **_ignored):
        self.system = system
        self.dev_size = dev_size
        self.prefix = prefix
    
    def get_all_segments(self, alignment):
        import i6_experiments.users.mann.experimental.extractors as extr
        import i6_core.corpus as corpus_recipes
        all_segments = corpus_recipes.SegmentCorpusJob(
            self.system.corpora[self.DEFAULT_FEATURE_CORPUS].corpus_file,
            num_segments = 1
        )
        alignment_logs = (
            select_element(
                self.system.alignments,
                self.DEFAULT_FEATURE_CORPUS,
                alignment)
                .alternatives["bundle"]
                .creator
                .out_log_file
        )
        filtered_segments = extr.FilterSegmentsByAlignmentFailures(
            {1: all_segments.single_segment_files[1]},
            alignment_logs
        )
        return filtered_segments.single_segment_files[1]
    
    def filter_segments(self, corpus, alignment, alignment_logs=None):
        all_segments = self.system.all_segments[corpus]
        
        assert isinstance(all_segments, (tk.Path, str))

        if not isinstance(all_segments, tk.Path):
            all_segments = tk.Path(all_segments)

        if alignment_logs is None:
            alignment_logs = (
                select_element(
                    self.system.alignments,
                    "train",
                    alignment)
                    .alternatives["bundle"]
                    .creator
                    .out_log_file
            )

        alignment_failure_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
        filtered_segments = corpus_recipes.FilterSegmentsByListJob({1: all_segments}, alignment_failure_list)

        return filtered_segments.out_single_segment_files[1]

    
    def apply(self, training_args, alignment=None, alignment_logs=None, **_ignored):
        import i6_experiments.users.mann.experimental.extractors as extr
        import i6_core.corpus as corpus_recipes
        dev_size = self.dev_size
        if alignment is None:
            alignment = training_args["alignment"]
        full_train_args = ChainMap(training_args, self.system.default_nn_training_args)
        if isinstance(alignment, dict):
            assert alignment_logs is None or isinstance(alignment_logs, dict)
            for key, sub_alignment in alignment.items():
                assert key in ["train", "dev"]
                sub_alignment_logs = alignment_logs.get(key, None) if alignment_logs is not None else None
                feature_corpus = full_train_args[key + "_corpus"]         
                filt_segments = self.filter_segments(feature_corpus, sub_alignment, sub_alignment_logs)
                self.system.crp[feature_corpus].segment_path = filt_segments
            return
        feature_corpus = full_train_args["feature_corpus"]
        filt_segments = self.filter_segments(feature_corpus, alignment, alignment_logs)
        new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
            segment_file = filt_segments,
            split = { 'train': 1.0 - dev_size, 'dev': dev_size }
        )
        for ds in ["train", "dev"]:
            self.system.csp[self.prefix + "_" + ds].segment_path = new_segments.out_segments[ds]
