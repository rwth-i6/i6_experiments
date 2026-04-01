__all__ = [
    "DownloadLibriSpeechTTSCorpusJob",
    "DownloadLibriSpeechTTSMetadataJob",
    "LibriSpeechTTSCreateBlissCorpusJob",
]

import shutil
import subprocess
import csv
from sisyphus import *

from i6_core.lib import corpus
from i6_core.util import uopen

from i6_core.audio.encoding import BlissChangeEncodingJob
from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
from i6_core.datasets.librispeech import *

from collections import defaultdict
import random
from functools import lru_cache
import os

from sisyphus import Job, Task, tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict, get_corpus_object_dict


class DownloadLibriSpeechTTSCorpusJob(Job):
    """
    Download a part of the LibriSpeech corpus from
    https://www.openslr.org/resources/12
    and checks for file integrity via md5sum

    (see also: https://www.openslr.org/12/)

    To get the corpus metadata, use
    DownloadLibriSpeechMetadataJob

    self.out_corpus_folder links to the root of the speaker_id/chapter/*
    folder structure
    """

    def __init__(self, corpus_key):
        """
        :param str corpus_key: corpus identifier, e.g. "train-clean-100"
        """
        self.corpus_key = corpus_key
        self.out_speakers = self.output_path("SPEAKERS.txt")
        self.out_chapters = self.output_path("CHAPTERS.txt")
        self.out_books = self.output_path("BOOKS.txt")

        assert corpus_key in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

        self.out_corpus_folder = self.output_path("%s" % self.corpus_key)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        subprocess.check_call(["wget", "https://www.openslr.org/resources/60/checksum.md5"])
        subprocess.check_call(["wget", "https://www.openslr.org/resources/60/%s.tar.gz" % self.corpus_key])

        with open("checksum.md5", "rt") as md5_in, open("md5sum-%s.txt" % self.corpus_key, "wt") as md5_out:
            for line in md5_in:
                split = line.strip().split(" ")
                if split[-1].split(".")[0] == self.corpus_key:
                    md5_out.write(line)
                    break

        subprocess.check_call(["md5sum", "--status", "-c", "md5sum-%s.txt" % self.corpus_key])
        subprocess.check_call(
            [
                "tar",
                "-xf",
                "%s.tar.gz" % self.corpus_key,
                "-C",
                ".",
            ]
        )
        self._move_files()
        os.unlink("%s.tar.gz" % self.corpus_key)
        shutil.rmtree("LibriTTS")

    def _move_files(self):
        shutil.move("LibriTTS/%s" % self.corpus_key, self.out_corpus_folder.get_path())
        shutil.move("LibriTTS/SPEAKERS.txt", self.out_speakers.get_path())
        shutil.move("LibriTTS/CHAPTERS.txt", self.out_chapters.get_path())
        shutil.move("LibriTTS/BOOKS.txt", self.out_books.get_path())

class LibriSpeechTTSCreateBlissCorpusJob(Job):
    """
    Creates a Bliss corpus from a LibriSpeech corpus folder using the speaker information in addition

    Outputs a single bliss .xml.gz file
    """

    def __init__(self, corpus_folder):
        """
        :param Path corpus_folder: Path to a LibriSpeech corpus folder
        :param Path speaker_metadata: Path to SPEAKER.TXT file from the MetdataJob (out_speakers)
        """
        self.corpus_folder = corpus_folder
        self.speaker_metadata = speaker_metadata

        self.out_corpus = self.output_path("corpus.xml.gz")

        self._speakers = {}  # dict(key: id, value: [sex, subset, min, name]
        self._transcripts = []  # [dict(name, chapter, segment, orth, path)]

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        self._get_speakers()
        self._get_transcripts()

        c = corpus.Corpus()
        c.name = os.path.basename(self.corpus_folder.get_path())

        used_speaker_ids = set()  # store which speakers are used

        for transcript in self._transcripts:
            name = transcript["segment"]
            recording = corpus.Recording()
            recording.name = name
            recording.speaker_name = transcript["speaker_id"]
            recording.audio = "{}/{}.wav".format(transcript["path"], name)

            used_speaker_ids.add(transcript["speaker_id"])

            segment = corpus.Segment()
            segment.name = name
            segment.start = 0
            segment.end = float("inf")
            segment.orth = transcript["orth"].strip()

            recording.segments.append(segment)
            c.recordings.append(recording)

        for speaker_id, speaker_info in sorted(self._speakers.items()):
            if speaker_id not in used_speaker_ids:
                continue
            speaker = corpus.Speaker()
            speaker.name = speaker_id
            speaker.attribs["gender"] = "male" if speaker_info[0] == "M" else "female"
            c.add_speaker(speaker)

        c.dump(self.out_corpus.get_path())

    def _get_speakers(self):
        """
        Extract the speakers from the SPEAKERS.TXT file
        """
        with uopen(self.speaker_metadata, "r") as speakersfile:
            for line in speakersfile:
                if line[0] == ";":
                    continue
                procline = list(map(str.strip, line.split("|")))
                self._speakers[int(procline[0])] = [
                    procline[1],
                    procline[2],
                    float(procline[3]),
                    procline[4],
                ]

    def _get_transcripts(self):
        """
        Traverse the folder structure and search for the *.trans.txt files and read the content
        """
        for dirpath, dirs, files in sorted(os.walk(self.corpus_folder.get_path(), followlinks=True)):
            for file in files:
                if not file.endswith(".trans.tsv"):
                    continue
                with open(os.path.join(dirpath, file), 'r', newline='', encoding='utf-8') as tsvfile:
                    for line in tsvfile:
                        row = line.strip().split("\t")
                        orth = row[2]
                        procline = row[0].split("_")

                        transcript = {
                            "speaker_id": int(procline[0]),
                            "chapter": int(procline[1]),
                            "segment": row[0],
                            "orth": orth,
                            "path": dirpath,
                        }
                        self._transcripts.append(transcript)

@lru_cache()
def get_tts_bliss_corpus_dict(audio_format="flac", output_prefix="datasets"):
    """
    Download and create a bliss corpus for each of the LibriSpeech training corpora and test sets,
    and return all corpora as a single corpus dict.

    No outputs will be registered.

    :param str audio_format: flac (no re-encoding), wav or ogg
    :param str output_prefix:
    :return: A corpus dict with the following entries:
        - 'dev-clean'
        - 'dev-other'
        - 'test-clean'
        - 'test-other'
        - 'train-clean-100'
        - 'train-clean-360'
        - 'train-clean-460'
        - 'train-other-500'
        - 'train-other-960'
    :rtype: dict[str, tk.Path]
    """
    assert audio_format in ["flac", "ogg", "wav"]

    output_prefix = os.path.join(output_prefix, "LibriSpeech")

    #download_metadata_job = DownloadLibriSpeechTTSMetadataJob()
    #download_metadata_job.add_alias(os.path.join(output_prefix, "download", "metadata_job"))

    def _get_corpus(corpus_name):
        download_corpus_job = DownloadLibriSpeechTTSCorpusJob(corpus_key=corpus_name)
        create_bliss_corpus_job = LibriSpeechTTSCreateBlissCorpusJob(
            corpus_folder=download_corpus_job.out_corpus_folder,
            speaker_metadata=download_corpus_job.out_speakers,
        )
        download_corpus_job.add_alias(os.path.join(output_prefix, "download", corpus_name))
        create_bliss_corpus_job.add_alias(os.path.join(output_prefix, "create_bliss", corpus_name))
        return create_bliss_corpus_job.out_corpus

    corpus_names = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    bliss_corpus_dict = {corpus_name: _get_corpus(corpus_name) for corpus_name in corpus_names}

    audio_format_options = {
        "wav": {
            "output_format": "wav",
            "codec": "pcm_s16le",
        },
        "ogg": {"output_format": "ogg", "codec": "libvorbis"},
    }

    if audio_format != "flac":
        converted_bliss_corpus_dict = {}
        for corpus_name, flac_corpus in bliss_corpus_dict.items():
            bliss_change_encoding_job = BlissChangeEncodingJob(
                corpus_file=flac_corpus,
                sample_rate=16000,
                **audio_format_options[audio_format],
            )
            bliss_change_encoding_job.add_alias(
                os.path.join(
                    output_prefix,
                    "%s_conversion" % audio_format,
                    corpus_name,
                )
            )
            converted_bliss_corpus_dict[corpus_name] = bliss_change_encoding_job.out_corpus
    else:
        converted_bliss_corpus_dict = bliss_corpus_dict

    def _merge_corpora(corpora, name):
        merge_job = MergeCorporaJob(bliss_corpora=corpora, name=name, merge_strategy=MergeStrategy.FLAT)
        merge_job.add_alias(os.path.join(output_prefix, "%s_merge" % audio_format, name))
        return merge_job.out_merged_corpus

    converted_bliss_corpus_dict["train-clean-460"] = _merge_corpora(
        corpora=[
            converted_bliss_corpus_dict["train-clean-100"],
            converted_bliss_corpus_dict["train-clean-360"],
        ],
        name="train-clean-460",
    )

    converted_bliss_corpus_dict["train-other-960"] = _merge_corpora(
        corpora=[
            converted_bliss_corpus_dict["train-clean-100"],
            converted_bliss_corpus_dict["train-clean-360"],
            converted_bliss_corpus_dict["train-other-500"],
        ],
        name="train-other-960",
    )

    return converted_bliss_corpus_dict

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
            speaker = segment.split("_")[0]
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
def get_libritts_tts_segments(bliss_corpus):
    """
    Generate the fixed train and dev segments for fixed speaker TTS training

    :return:
    """
    segments = SegmentCorpusJob(bliss_corpus, 1).out_single_segment_files[1]
    generate_tts_segments_job = GenerateBalancedSpeakerDevSegmentFileJob(
        segment_file=segments,
        dev_segments_per_speaker=4
    )
    return generate_tts_segments_job.out_train_segments, generate_tts_segments_job.out_dev_segments
