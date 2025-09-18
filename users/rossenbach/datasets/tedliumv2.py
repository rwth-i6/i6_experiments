import copy
from collections import defaultdict
import random
from functools import lru_cache
import os

from sisyphus import Job, Task, tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.meta import CorpusObject
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.bpe import get_bpe_settings, get_returnn_subword_nmt
from i6_experiments.users.rossenbach.audio.silence_removal import ffmpeg_silence_remove

from .librispeech import GenerateBalancedSpeakerDevSegmentFileJob

def get_tedliumv2_tts_segments():
    """
    Generate the fixed train and dev segments for fixed speaker TTS training

    :return:
    """
    bliss_corpus_dict = get_bliss_corpus_dict()
    segments = SegmentCorpusJob(bliss_corpus_dict["train"], 1).out_single_segment_files[1]
    generate_tts_segments_job = GenerateBalancedSpeakerDevSegmentFileJob(
        segment_file=segments,
        dev_segments_per_speaker=4
    )
    return generate_tts_segments_job.out_train_segments, generate_tts_segments_job.out_dev_segments