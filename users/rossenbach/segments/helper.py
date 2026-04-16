from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob

def shuffle_and_head(segment_file, num_lines):
    # only shuffle, this is deterministic
    shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
        segment_file=segment_file,
        split={"shuffle": 1.0},
        shuffle=True
    )
    segment_file = shuffle_segment_file_job.out_segments["shuffle"]
    return HeadJob(segment_file, num_lines=num_lines).out