import subprocess

from sisyphus import *
import h5py
import numpy
import pickle
import random
from typing import Iterator, Optional, Dict

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


class RandomSpeakerAssignmentJob(Job):
    """
    Depending on input either randomizes speaker indices within corpus or uses speaker names from different corpus
    """
    def __init__(self, bliss_corpus: tk.Path, speaker_bliss_corpus: Optional[tk.Path] = None, keep_ratio: bool = True):
        """

        :param bliss_corpus: Corpus to assign speakers to
        :param speaker_bliss_corpus: If not None assign speakers from this corpus
        :param keep_ratio: keep the original ratio of speaker amounts equal
        """
        self.bliss_corpus = bliss_corpus
        self.speaker_bliss = speaker_bliss_corpus if speaker_bliss_corpus is not None else bliss_corpus
        self.keep_ratio = keep_ratio

        self.out_mapping = self.output_path("out_mapping.pkl")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):

        bliss = corpus.Corpus()
        bliss.load(self.bliss_corpus.get_path())
        speakers = []
        speaker_bliss = corpus.Corpus()
        speaker_bliss.load(self.speaker_bliss.get_path())
        for recording in speaker_bliss.all_recordings():
            for segment in recording.segments:
                speakers.append(segment.speaker_name or recording.speaker_name)
        if not self.keep_ratio:
            speakers = list(set(speakers))
        random.shuffle(speakers)

        mapping = {}
        idx = 0
        for recording in bliss.all_recordings():
            for segment in recording.segments:
                mapping[segment.fullname()] = speakers[idx]
                idx += 1
                if idx >= len(speakers):
                    idx = 0
        with open(self.out_mapping, "wb") as f:
            pickle.dump(mapping, file=f)


class CalculateSpeakerPriorJob(Job):
    """
    Calculates the average Speaker Prior from a given speaker prior hdf file
    """
    def __init__(
        self,
        vae_hdf: tk.Path,
        corpus_file: tk.Path,
        prior_dim: int = 8
    ):
        self.vae_hdf = vae_hdf
        self.corpus_file = corpus_file
        self.prior_dim = prior_dim

        self.out_prior = self.output_path("speaker_prior.hdf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):

        vae_data = h5py.File(
            self.vae_hdf.get_path(), "r"
        )
        vae_inputs = vae_data["inputs"]
        vae_raw_tags = list(vae_data["seqTags"])

        bliss = corpus.Corpus()
        bliss.load(self.corpus_file.get_path())

        speaker_sums = {}
        speaker_counts = {}
        speaker_by_index = {}
        index_by_speaker = {}
        for i, speaker in enumerate(bliss.speakers):
            speaker_by_index[i] = speaker
            index_by_speaker[speaker] = i
            speaker_sums[i] = numpy.array((len(vae_inputs[0]),))
            speaker_counts[i] = 0
        for recording in bliss.all_recordings():
            for segment in recording.segments:
                idx = vae_raw_tags.index(segment.name)
                speaker_idx = index_by_speaker[segment.speaker_name or recording.speaker_name]
                speaker_sums[speaker_idx] += vae_inputs[idx]
                speaker_counts[speaker_idx] += 1

        hdf_writer = SimpleHDFWriter(
            self.out_prior.get_path(), dim=len(vae_inputs[0]), ndim=1
        )
        for speaker_idx in range(len(bliss.speakers)):
            prior = speaker_sums[speaker_idx] / speaker_counts[speaker_idx]
            hdf_writer.insert_batch(
                numpy.asarray([[prior]], dtype="float32"), [1], [str(speaker_by_index[speaker_idx])])
        hdf_writer.close()


class SingularizeHDFPerSpeakerJob(Job):
    """
    Removes duplicates from a HDF, e.g. a speaker embedding hdf which contains multiples (of the same) speaker embedding
    This job assumes that all speakers are in the given bliss. This means that if you give a corpus with less/different
    speakers, than in the HDF this will cause a mismatch.

    """
    def __init__(self, hdf_file: tk.Path, speaker_bliss: tk.Path):

        self.hdf_file = hdf_file
        self.speaker_bliss = speaker_bliss

        self.out_hdf = self.output_path("out.hdf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):

        hdf_data = h5py.File(
            self.hdf_file.get_path(), "r"
        )

        inputs = hdf_data["inputs"]
        raw_tags = hdf_data["seqTags"]
        lengths = hdf_data["seqLengths"]

        bliss = corpus.Corpus()
        bliss.load(self.speaker_bliss.get_path())

        num_speakers = len(bliss.speakers)
        tag_to_value = {}

        offset = 0
        dim = inputs[0: lengths[0][0]][0].shape[-1]
        for tag, length in zip(raw_tags, lengths):
            tag_to_value[tag] = inputs[offset: offset + length[0]]
            offset += length[0]

        index_to_value = {}
        for recording in bliss.all_recordings():
            for segment in recording.segments:
                speaker_name = segment.speaker_name or recording.speaker_name
                # not only check that we already have the speaker but also that we handle a bigger corpus (e.g.
                # embeddings of dev but got the full corpus
                if speaker_name not in index_to_value.keys() and segment.fullname() in tag_to_value.keys():
                    index_to_value[speaker_name] = tag_to_value[segment.fullname()]
            if len(index_to_value) == num_speakers:
                break

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=dim
        )

        for index in index_to_value:
            hdf_writer.insert_batch(
                numpy.asarray([index_to_value[index]]),
                [1],
                [index],
            )
        hdf_writer.close()


class DistributeHDFByMappingJob(Job):
    """
    Applies a given mapping (segment -> any index) onto the given HDF with internal mapping (any index -> vector) to
    produce an HDF with mapping (segment -> vector)
    """
    def __init__(self, hdf_file: tk.Path, mapping: tk.Path):
        """

        :param hdf_file:
        :param mapping: currently only supports .pkl dictionaries
        """
        self.hdf_file = hdf_file
        self.mapping = mapping

        self.out_hdf = self.output_path("out.hdf")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        hdf_data = h5py.File(
            self.hdf_file.get_path(), "r"
        )

        inputs = hdf_data["inputs"]
        raw_tags = list(hdf_data["seqTags"])
        lengths = hdf_data["seqLengths"]

        with open(self.mapping.get_path(), "rb") as f:
            mapping = pickle.load(f)  # type: Dict

        tag_to_value = {}
        tag_to_length = {}
        offset = 0
        dim = inputs[0: lengths[0][0]][0].shape[-1]
        # reconstruct hdf dict to distribute better in case length is not 1
        for tag, length in zip(raw_tags, lengths):
            tag_to_value[tag] = inputs[offset: offset + length[0]]
            tag_to_length[tag] = length[0]
            offset += length[0]

        hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=dim, ndim=2
        )
        for segment_tag, index in mapping.items():
            hdf_writer.insert_batch(
                numpy.asarray([tag_to_value[index]]),
                [tag_to_length[index]],
                [segment_tag],
            )
        hdf_writer.close()
