import copy
from collections import defaultdict
import random
from functools import lru_cache
import os

from sisyphus import Job, Task, tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.meta import CorpusObject
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict, get_corpus_object_dict
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.bpe import get_bpe_settings, get_returnn_subword_nmt
from i6_experiments.users.rossenbach.audio.silence_removal import ffmpeg_silence_remove


@lru_cache()
def get_librispeech_bpe(corpus_key, bpe_size, unk_label="<unk>", output_prefix=""):
    """
    Get the BPE tokens via the subword-nmt fork for a librispeech setup.
    When using the default settings this will give 100% compatible BPE settings to
    Albert Zeyers and Kazuki Iries setups.

    :param str corpus_key:
    :param int bpe_size:
    :param str output_prefix
    :return:
    :rtype: BPESettings
    """

    output_prefix = os.path.join(output_prefix, "librispeech_%s_bpe_%i" % (corpus_key, bpe_size))

    subword_nmt_commit_hash = "6ba4515d684393496502b79188be13af9cad66e2"
    subword_nmt_repo = get_returnn_subword_nmt(commit_hash=subword_nmt_commit_hash, output_prefix=output_prefix)
    train_other_960 = get_bliss_corpus_dict("flac", "corpora")[corpus_key]
    bpe_settings = get_bpe_settings(
        train_other_960,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=output_prefix,
        subword_nmt_repo_path=subword_nmt_repo)
    return bpe_settings


@lru_cache()
def get_mixed_cv_segments(output_prefix="datasets"):
    """
    Create a mixed crossvalidation set containing
    1500 lines of dev-clean and 1500 lines of dev-other

    :return: line based segment file
    :rtype: Path
    """
    bliss_corpus_dict = get_bliss_corpus_dict(output_prefix=output_prefix)
    dev_clean = bliss_corpus_dict['dev-clean']
    dev_other = bliss_corpus_dict['dev-other']

    dev_clean_segments = SegmentCorpusJob(dev_clean, 1).out_single_segment_files[1]
    dev_other_segments = SegmentCorpusJob(dev_other, 1).out_single_segment_files[1]

    def shuffle_and_head(segment_file, num_lines):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file,
            split={"shuffle": 1.0},
            shuffle=True
        )
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_clean_subset = shuffle_and_head(dev_clean_segments, 1500)
    dev_other_subset = shuffle_and_head(dev_other_segments, 1500)

    dev_cv_segments = PipelineJob([dev_clean_subset, dev_other_subset], [], mini_task=True).out

    return dev_cv_segments


class GenerateBalancedSpeakerDevSegmentFileJob(Job):
    """

    Generates specific train and dev segment files for TTS training with fixed speaker embeddings.

    To guarantee an even distribution, a fixed number of dev segments is chosen per speaker/book combination

    :return:
    """
    def __init__(self, segment_file, dev_segments_per_speaker):
        """

        :param tk.Path segment_file:
        :param int dev_segments_per_speaker:
        """
        self.segment_file = segment_file
        self.dev_segments_per_speaker = dev_segments_per_speaker

        self.out_dev_segments = self.output_path("dev.segments")
        self.out_train_segments = self.output_path("train.segments")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        random.seed(42)
        segment_file = open(self.segment_file.get_path(), "rt")

        segments = []
        speaker_segments = defaultdict(lambda :list())
        for line in segment_file.readlines():
            segment = line.strip()
            segments.append(segment)
            speaker = segment.split("/")[-1].split("-")[0]
            speaker_segments[speaker].append(segment)

        dev_segments = []

        for sb in speaker_segments.keys():
            segs = speaker_segments[sb]
            random.shuffle(segs)
            dev_segments.extend(segs[:self.dev_segments_per_speaker])

        with open(self.out_dev_segments.get_path(), "wt") as f:
            for dev_segment in dev_segments:
                f.write(dev_segment + "\n")

        dev_segments = set(dev_segments)

        with open(self.out_train_segments.get_path(), "wt") as f:
            for segment in segments:
                if segment not in dev_segments:
                    f.write(segment + "\n")


@lru_cache()
def get_librispeech_tts_segments(ls_corpus_key="train-clean-100"):
    """
    Generate the fixed train and dev segments for fixed speaker TTS training

    :return:
    """
    bliss_corpus_dict = get_bliss_corpus_dict()
    segments = SegmentCorpusJob(bliss_corpus_dict[ls_corpus_key], 1).out_single_segment_files[1]
    generate_tts_segments_job = GenerateBalancedSpeakerDevSegmentFileJob(
        segment_file=segments,
        dev_segments_per_speaker=4
    )
    return generate_tts_segments_job.out_train_segments, generate_tts_segments_job.out_dev_segments


def get_speaker_extraction_segments(subcorpus_name):
    """

    :param subcorpus_name:
    :return:
    """
    bliss_corpus_dict = get_bliss_corpus_dict()
    segments = SegmentCorpusJob(bliss_corpus_dict[subcorpus_name], 1).out_single_segment_files[1]
    generate_tts_segments_job = GenerateBalancedSpeakerDevSegmentFileJob(
        segment_file=segments,
        dev_segments_per_speaker=1
    )
    return generate_tts_segments_job.out_dev_segments


def get_ls_train_clean_100_tts_silencepreprocessed(alias_path=""):
    """
    This returns the silence-preprocessed version of LibriSpeech train-clean-100 with
    FFmpeg silence preprocessing using a threshold of -50dB for silence
    :return:
    """
    corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")
    train_100_corpus = corpus_object_dict['train-clean-100']

    processed_corpus = CorpusObject()
    processed_corpus.audio_format = "ogg"
    # this is not the true duration, but because it is unknown we just copy
    # this as no further implications to the pipeline except for the RTF settings to estimate SGE usage
    processed_corpus.duration = train_100_corpus.duration
    processed_corpus.audio_dir = train_100_corpus.audio_dir
    processed_corpus.corpus_file = ffmpeg_silence_remove(
        train_100_corpus.corpus_file,
        stop_threshold = -50,
        stop_duration = 0,
        force_output_format = 'ogg',
        # the pipeline uses n4.1.4, but we assume that it is safe to user other versions of FFMPEG as well
        # hash overwrite is no longer needed, as the ffmpeg binary is not hashed unless specifically requested
        ffmpeg_binary=tk.Path("/u/rossenbach/bin/ffmpeg", hash_overwrite="FFMPEG"))

    return copy.deepcopy(processed_corpus)


def get_ls_train_clean_360_tts_silencepreprocessed(alias_path=""):
    """
    This returns the silence-preprocessed version of LibriSpeech train-clean-100 with
    FFmpeg silence preprocessing using a threshold of -50dB for silence
    :return:
    """
    corpus_object_dict = get_corpus_object_dict(audio_format="ogg", output_prefix="corpora")
    train_360_corpus = corpus_object_dict['train-clean-360']

    processed_corpus = CorpusObject()
    processed_corpus.audio_format = "ogg"
    # this is not the true duration, but because it is unknown we just copy
    # this as no further implications to the pipeline except for the RTF settings to estimate SGE usage
    processed_corpus.duration = train_360_corpus.duration
    processed_corpus.audio_dir = train_360_corpus.audio_dir
    processed_corpus.corpus_file = ffmpeg_silence_remove(
        train_360_corpus.corpus_file,
        stop_threshold = -50,
        stop_duration = 0,
        force_output_format = 'ogg',
        # the pipeline uses n4.1.4, but we assume that it is safe to user other versions of FFMPEG as well
        # hash overwrite is no longer needed, as the ffmpeg binary is not hashed unless specifically requested
        ffmpeg_binary=tk.Path("/u/rossenbach/bin/ffmpeg", hash_overwrite="FFMPEG"))

    return copy.deepcopy(processed_corpus)
