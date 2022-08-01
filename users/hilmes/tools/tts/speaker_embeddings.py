import subprocess

from sisyphus import *
import h5py
import numpy
import pickle
import random

from i6_private.users.rossenbach.lib.hdf import SimpleHDFWriter
from i6_core.lib import corpus


class DistributeSpeakerEmbeddings(Job):
    """
    distribute speaker embeddings contained in an hdf file to a new hdf file with mappings to the given bliss corpus
    """

    def __init__(
        self,
        speaker_embedding_hdf,
        bliss_corpus=None,
        text_file=None,
        options=None,
        use_full_seq_name=True,
    ):
        """

        :param tk.Path speaker_embedding_hdf:
        :param tk.Path bliss_corpus:
        :param tk.Path text_file: WARNING: no .gz support so far!
        :param options:
        :param use_full_seq_name:
        """
        self.speaker_embedding_hdf = speaker_embedding_hdf
        self.bliss_corpus = bliss_corpus
        self.text_file = text_file
        assert (
            self.bliss_corpus
            or self.text_file
            and not (self.bliss_corpus and self.text_file)
        )
        self.use_full_seq_name = use_full_seq_name
        self.options = options
        if self.options is None:
            self.options = {"mode": "random"}

        assert self.options["mode"] in ["random", "length_buckets"], (
            "invalid mode %s" % options["mode"]
        )

        self.out = self.output_path("speaker_embeddings.hdf")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _random(self, seq_tags):
        if "seed" in self.options:
            random.seed(self.options["seed"])

        random.shuffle(self.speaker_embedding_features)
        embedding_index = 0
        for seq_tag in seq_tags:
            self.hdf_writer.insert_batch(
                numpy.asarray([self.speaker_embedding_features[embedding_index]]),
                [1],
                [seq_tag],
            )
            embedding_index += 1
            if embedding_index >= len(self.speaker_embedding_features):
                embedding_index = 0

    def _random_matching_length(self):

        text_corpus = corpus.Corpus()
        assert len(text_corpus.subcorpora) == 0
        text_corpus.load(tk.uncached_path(self.options["corpus"]))

        text_durations = {}

        max_duration = 0
        for recording in text_corpus.recordings:
            assert len(recording.segments) == 1
            segment = recording.segments[0]  # type:corpus.Segment
            segment_name = "/".join([self.c.name, recording.name, segment.name])
            if not self.use_full_seq_name:
                segment_name = segment.name
            seg_len = len(segment.orth)
            text_durations[segment_name] = seg_len
            if seg_len > max_duration:
                max_duration = seg_len

        bucket_size = int(self.options["bucket_size"])
        buckets = [[] for i in range(0, max_duration + bucket_size, bucket_size)]
        bucket_indices = [0] * len(buckets)

        # fill buckets
        for tag, feature in zip(
            self.speaker_embedding_tags, self.speaker_embedding_features
        ):
            buckets[text_durations[tag] // bucket_size].append(feature)

        # shuffle buckets
        for bucket in buckets:
            random.shuffle(bucket)

        for recording in self.c.recordings:
            assert len(recording.segments) == 1
            segment = recording.segments[0]  # type:corpus.Segment
            segment_name = "/".join([self.c.name, recording.name, segment.name])
            if not self.use_full_seq_name:
                segment_name = segment.name

            # search for nearest target bucket
            target_bucket = len(segment.orth) // bucket_size
            for i in range(1000):
                if (
                    0 <= target_bucket + i < len(buckets)
                    and len(buckets[target_bucket + i]) > 0
                ):
                    target_bucket = target_bucket + i
                    break
                if (
                    0 <= target_bucket - i < len(buckets)
                    and len(buckets[target_bucket - i]) > 0
                ):
                    target_bucket = target_bucket - i
                    break

            speaker_embedding = buckets[target_bucket][bucket_indices[target_bucket]]
            self.hdf_writer.insert_batch(
                numpy.asarray([speaker_embedding]), [1], [segment_name]
            )
            bucket_indices[target_bucket] += 1
            if bucket_indices[target_bucket] >= len(buckets[target_bucket]):
                bucket_indices[target_bucket] = 0

    def run(self):

        speaker_embedding_data = h5py.File(
            tk.uncached_path(self.speaker_embedding_hdf), "r"
        )
        speaker_embedding_inputs = speaker_embedding_data["inputs"]
        speaker_embedding_raw_tags = speaker_embedding_data["seqTags"]
        speaker_embedding_lengths = speaker_embedding_data["seqLengths"]

        self.speaker_embedding_features = []
        self.speaker_embedding_tags = []
        offset = 0
        for tag, length in zip(speaker_embedding_raw_tags, speaker_embedding_lengths):
            self.speaker_embedding_features.append(
                speaker_embedding_inputs[offset : offset + length[0]]
            )
            self.speaker_embedding_tags.append(tag)
            offset += length[0]

        self.hdf_writer = SimpleHDFWriter(
            tk.uncached_path(self.out), dim=self.speaker_embedding_features[0].shape[-1]
        )

        seq_tags = []
        if self.bliss_corpus:
            self.c = corpus.Corpus()
            self.c.load(tk.uncached_path(self.bliss_corpus))
            for segment in self.c.segments():
                seq_tags.append(segment.fullname())
        elif self.text_file:
            pipe = subprocess.Popen(
                ["zcat", "-f", self.text_file.get_path()], stdout=subprocess.PIPE
            )
            output = subprocess.check_output(["wc", "-l"], stdin=pipe.stdout)
            length = int(output.decode())
            seq_tags = ["line-%i" % i for i in range(length)]

        mode = self.options.get("mode")
        if mode == "random":
            self._random(seq_tags)
        elif mode == "length_buckets" and self.bliss_corpus:
            self._random_matching_length()
        else:
            assert False

        self.hdf_writer.close()


class SpeakerLabelHDFFromBliss(Job):
    def __init__(self, bliss_corpus):
        """

        :param bliss_corpus:
        """
        self.bliss_corpus = bliss_corpus
        self.out = self.output_path("speaker_labels.hdf")
        self.num_speakers = self.output_var("num_speakers")
        self.speaker_dict = self.output_path("speaker_dict.pkl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        bliss = corpus.Corpus()
        bliss.load(tk.uncached_path(self.bliss_corpus))
        speaker_by_index = {}
        index_by_speaker = {}
        num_speakers = len(bliss.speakers)
        self.num_speakers.set(num_speakers)
        for i, speaker in enumerate(bliss.speakers):
            speaker_by_index[i] = speaker
            index_by_speaker[speaker] = i

        pickle.dump(speaker_by_index, open(tk.uncached_path(self.speaker_dict), "wb"))

        hdf_writer = SimpleHDFWriter(
            tk.uncached_path(self.out), dim=num_speakers, ndim=1
        )

        for recording in bliss.all_recordings():
            for segment in recording.segments:
                speaker_name = segment.speaker_name or recording.speaker_name
                speaker_index = index_by_speaker[speaker_name]
                segment_name = "/".join([bliss.name, recording.name, segment.name])
                hdf_writer.insert_batch(
                    numpy.asarray([[speaker_index]], dtype="int32"), [1], [segment_name]
                )

        hdf_writer.close()
