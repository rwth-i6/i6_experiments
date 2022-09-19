__all__ = ["FilterAlignmentPlugin"]
from i6_core.meta import select_element

class FilterAlignmentPlugin:
    DEFAULT_FEATURE_CORPUS = "train"
    def __init__(self, system, dev_size, **_ignored):
        self.system = system
        self.dev_size = dev_size
    
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
    
    def apply(self, training_args, alignment=None, **_ignored):
        import i6_experiments.users.mann.experimental.extractors as extr
        import i6_core.corpus as corpus_recipes
        dev_size = self.dev_size
        all_segments = corpus_recipes.SegmentCorpusJob(
            self.system.corpora[self.DEFAULT_FEATURE_CORPUS].corpus_file,
            num_segments = 1
        ).out_single_segment_files
        if alignment is None:
            alignment = training_args["alignment"]
        alignment_logs = (
            select_element(
                self.system.alignments,
                self.DEFAULT_FEATURE_CORPUS,
                alignment)
                .alternatives["bundle"]
                .creator
                .out_log_file
        )
        # filtered_segments = extr.FilterSegmentsByAlignmentFailures(
        #     {1: all_segments.single_segment_files[1]},
        #     alignment_logs
        # )
        alignment_failure_list = extr.ExtractAlignmentFailuresJob(alignment_logs).out_filter_list
        filtered_segments = corpus_recipes.FilterSegmentsByListJob(all_segments, alignment_failure_list)
        new_segments = corpus_recipes.ShuffleAndSplitSegmentsJob(
            segment_file = filtered_segments.out_single_segment_files[1],
            split = { 'train': 1.0 - dev_size, 'dev': dev_size }
        )
        for ds in ["train", "dev"]:
            self.system.csp["crnn_" + ds].segment_path = new_segments.out_segments[ds]
