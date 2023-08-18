import copy
import numpy as np
import os.path
from typing import Callable, Optional

from sisyphus import Job, Task, Path

from i6_core.lib import corpus
from i6_core.lib.hdf import get_returnn_simple_hdf_writer
from i6_core.lib.rasr_cache import FileArchiveBundle
from lazy_dataset.database import JsonDatabase


def zip_strict(*args):
    assert len(set([len(a) for a in args])) == 1, [len(a) for a in args]
    yield from zip(*args)


class EnhancedMeetingDataToBlissCorpusJob(Job):
    """
    Convert Paderborn"s json-based lazy dataset to a bliss corpus
    """

    __sis_hash_exclude__ = {"hash_audio_path_mapping": True}

    def __init__(
        self,
        json_database: Path,
        audio_path_mapping: Callable,
        hash_audio_path_mapping: bool = True,
        dataset_name: str = "corpus",
        sample_rate: int = 16000,
    ):
        """
        :param json_database: json database for Paderborn"s enhanced meeting data
        :param audio_path_mapping: callable to map audio paths, e.g. from /paderborn/path/example.wav to /rwth/path/...
        :param hash_audio_path_mapping: if False, audio_path_mapping will not be hashed
        :param dataset_name: name of dataset in database
        :param sample_rate: sample rate of audio data
        """
        self.json_database = json_database
        self.audio_path_mapping = audio_path_mapping
        self.hash_audio_path_mapping = hash_audio_path_mapping
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate

        self.out_bliss_corpus = self.output_path("corpus.xml.gz")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        db = JsonDatabase(self.json_database.get())
        ds = db.get_dataset(self.dataset_name)

        c = corpus.Corpus()
        c.name = self.dataset_name

        for ex in ds:
            for channel, file in enumerate(ex["audio_path"]["enhanced"]):
                r = corpus.Recording()
                r.name = f"{ex['example_id']}_{channel}"
                r.audio = self.audio_path_mapping(file)

                assert (
                    int(os.path.basename(r.audio).split("_")[-1].strip(".wav"))
                    == channel
                ), f"wav files should end on '_<channel>.wav'. For channel {channel}, got {r.audio}"

                for (
                    segment_id,
                    speaker_id,
                    transcription,
                    num_samples,
                    offset,
                    channel_alignment,
                ) in zip_strict(
                    range(len(ex["speaker_id"])),
                    ex["speaker_id"],
                    ex.get("kaldi_transcription") or ex["transcription"],
                    ex["num_samples"]["original_source"],
                    ex["offset"]["original_source"],
                    ex["channel_alignment"],
                ):
                    if channel == channel_alignment:
                        s = corpus.Segment()
                        s.name = f"{segment_id:04d}"
                        s.speaker_name = speaker_id
                        s.orth = transcription
                        s.start = offset / self.sample_rate
                        s.end = (offset + num_samples) / self.sample_rate
                        r.add_segment(s)

                        speaker = corpus.Speaker()
                        speaker.name = speaker_id
                        if speaker not in c.speakers:
                            c.add_speaker(speaker)
                c.add_recording(r)

        c.dump(self.out_bliss_corpus.get())

    @classmethod
    def hash(cls, kwargs):
        d = copy.copy(kwargs)
        if not kwargs["hash_audio_path_mapping"]:
            d.pop("audio_path_mapping")
        return super().hash(d)


class EnhancedMeetingDataToSplitBlissCorporaJob(Job):
    """
    Convert Paderborn"s json-based lazy dataset to three bliss corpora: one containing the mixed observation
    and two containing the separated audios
    """

    __sis_hash_exclude__ = {"hash_audio_path_mapping": True}

    def __init__(
        self,
        json_database: Path,
        enhanced_audio_path_mapping: Callable,
        mix_audio_path_mapping: Callable,
        hash_audio_path_mapping: bool = True,
        dataset_name: str = "corpus",
        sample_rate: int = 16000,
    ):
        """
        :param json_database: json database for Paderborn"s enhanced meeting data
        :param audio_path_mapping: callable to map audio paths, e.g. from /paderborn/path/example.wav to /rwth/path/...
        :param hash_audio_path_mapping: if False, audio_path_mapping will not be hashed
        :param dataset_name: name of dataset in database
        :param sample_rate: sample rate of audio data
        """
        self.json_database = json_database
        self.enhanced_audio_path_mapping = enhanced_audio_path_mapping
        self.mix_audio_path_mapping = mix_audio_path_mapping
        self.hash_audio_path_mapping = hash_audio_path_mapping
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate

        self.out_bliss_corpus_primary = self.output_path("corpus_primary.xml.gz")
        self.out_bliss_corpus_secondary = self.output_path("corpus_secondary.xml.gz")
        self.out_bliss_corpus_mix = self.output_path("corpus_mix.xml.gz")

        self.rqmt = {"time": 6, "mem": 8, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        db = JsonDatabase(self.json_database.get())
        ds = db.get_dataset(self.dataset_name)

        c_prim = corpus.Corpus()
        c_prim.name = self.dataset_name

        c_sec = corpus.Corpus()
        c_sec.name = self.dataset_name

        c_mix = corpus.Corpus()
        c_mix.name = self.dataset_name

        for ex in ds:
            files = list(ex["audio_path"]["enhanced"])
            mix_file = ex["audio_path"]["observation"]
            assert len(files) == 2

            r_prim = []
            r_sec = []
            r_mix = []

            for channel in range(len(files)):
                for rec, audio in [
                    (r_prim, self.enhanced_audio_path_mapping(files[channel])),
                    (r_sec, self.enhanced_audio_path_mapping(files[1-channel])),
                    (r_mix, self.mix_audio_path_mapping(mix_file))
                ]:
                    r = corpus.Recording()
                    r.name = f"{ex['example_id']}_{channel}"
                    r.audio = audio
                    rec.append(r)

            for (
                segment_id,
                speaker_id,
                transcription,
                num_samples,
                offset,
                channel_alignment,
            ) in zip_strict(
                range(len(ex["speaker_id"])),
                ex["speaker_id"],
                ex.get("kaldi_transcription") or ex["transcription"],
                ex["num_samples"]["original_source"],
                ex["offset"]["original_source"],
                ex["channel_alignment"],
            ):
                for rec in [r_prim, r_sec, r_mix]:
                    s = corpus.Segment()
                    s.name = f"{segment_id:04d}"
                    s.speaker_name = speaker_id
                    s.orth = transcription
                    s.start = offset / self.sample_rate
                    s.end = (offset + num_samples) / self.sample_rate

                    rec[channel_alignment].add_segment(s)

                    speaker = corpus.Speaker()
                    speaker.name = speaker_id
                    if speaker not in c_prim.speakers:
                        c_prim.add_speaker(speaker)
                        c_sec.add_speaker(speaker)
                        c_mix.add_speaker(speaker)
            c_prim.add_recording(r_prim[0])
            c_prim.add_recording(r_prim[1])
            c_sec.add_recording(r_sec[0])
            c_sec.add_recording(r_sec[1])
            c_mix.add_recording(r_mix[0])
            c_mix.add_recording(r_mix[1])

        c_prim.dump(self.out_bliss_corpus_primary.get())
        c_sec.dump(self.out_bliss_corpus_secondary.get())
        c_mix.dump(self.out_bliss_corpus_mix.get())

    @classmethod
    def hash(cls, kwargs):
        d = copy.copy(kwargs)
        if not kwargs["hash_audio_path_mapping"]:
            d.pop("enhanced_audio_path_mapping")
            d.pop("mix_audio_path_mapping")
        return super().hash(d)


class EnhancedEvalDataToBlissCorpusJob(EnhancedMeetingDataToSplitBlissCorporaJob):
    """
    Similar to EnhancedMeetingDataToBlissCorpusJob, but here we do not use meta information about channel assignment,
    segment start and end times, etc.

    The output corpus will just have the full separated audio as recording and on segment per recording.
    """

    def run(self):
        db = JsonDatabase(self.json_database.get())
        ds = db.get_dataset(self.dataset_name)

        c_prim = corpus.Corpus()
        c_prim.name = self.dataset_name

        c_sec = corpus.Corpus()
        c_sec.name = self.dataset_name

        c_mix = corpus.Corpus()
        c_mix.name = self.dataset_name

        for ex in ds:
            files = list(ex["audio_path"]["enhanced"])
            assert len(files) == 2
            mix_file = ex["audio_path"]["observation"]

            for channel in range(len(files)):
                for c, file in [
                    (c_prim, self.enhanced_audio_path_mapping(files[channel])),
                    (c_sec, self.enhanced_audio_path_mapping(files[1 - channel])),
                    (c_mix, self.mix_audio_path_mapping(mix_file))
                ]:
                    r = corpus.Recording()
                    r.name = f"{ex['example_id']}_{channel}"
                    r.audio = file

                    s = corpus.Segment()
                    s.name = "0001"
                    s.orth = " /// ".join(
                        " /// ".join(t for t in ts)
                        for ts in ex["transcription"].values()
                    )
                    s.start = 0.0
                    s.end = np.inf
                    r.add_segment(s)
                    c.add_recording(r)

        c_prim.dump(self.out_bliss_corpus_primary.get())
        c_sec.dump(self.out_bliss_corpus_secondary.get())
        c_mix.dump(self.out_bliss_corpus_mix.get())


class EnhancedSegmentedEvalDataToBlissCorpusJob(EnhancedMeetingDataToBlissCorpusJob):
    """
    Similar to EnhancedMeetingDataToBlissCorpusJob, but here we do not use meta information about channel assignment,
    segment start and end times, etc.

    The output corpus will just have the full separated audio as recording and on segment per recording.
    """

    def run(self):
        db = JsonDatabase(self.json_database.get())
        ds = db.get_dataset(self.dataset_name)

        c = corpus.Corpus()
        c.name = self.dataset_name

        for ex in ds:
            for channel, file_full in enumerate(ex["audio_path"]["enhanced"]):
                # TODO: this is very specific about the existing v0 libricss tfgridnet data.
                # It should be handled via the json itself.
                folder = (
                    self.audio_path_mapping(file_full)
                    .replace("/audio/", "/audio_segmented_vad/")
                    .replace(".wav", "/")
                )
                for idx, file in enumerate(sorted(os.listdir(folder))):
                    r = corpus.Recording()
                    r.name = f"{ex['example_id']}_{channel}_{idx}"
                    r.audio = self.audio_path_mapping(os.path.join(folder, file))

                    s = corpus.Segment()
                    s.name = "0001"
                    s.orth = " /// ".join(
                        " /// ".join(t for t in ts)
                        for ts in ex["transcription"].values()
                    )
                    s.start = 0.0
                    s.end = np.inf
                    r.add_segment(s)
                    c.add_recording(r)

        c.dump(self.out_bliss_corpus.get())


class EnhancedMeetingDataRasrAlignmentPadAndDumpHDFJob(Job):
    """
    This Job reads Rasr alignment caches, pads the alignment given meta information and dumps them in hdf files.

    Based on RasrAlignmentDumpHDFJob
    """

    def __init__(
        self,
        json_database: Path,
        dataset_name: str,
        feature_hdf: Path,
        alignment_cache: Path,
        allophone_file: Path,
        state_tying_file: Path,
        data_type: type = np.uint16,
        returnn_root: Optional[Path] = None,
    ):
        """
        :param json_database: json database for Paderborn"s enhanced meeting data
        :param dataset_name: name of dataset in database
        :param feature_hdf: path to hdf with dumped features for enhanced meeting data
        :param alignment_cache: bundle file, e.g. output of an AlignmentJob
        :param allophone_file: e.g. output of a StoreAllophonesJob
        :param state_tying_file: e.g. output of a DumpStateTyingJob
        :param data_type: type that is used to store the data
        :param returnn_root: file path to the RETURNN repository root folder
        """
        self.json_database = json_database
        self.dataset_name = dataset_name
        self.feature_hdf = feature_hdf
        self.alignment_cache = alignment_cache
        self.allophone_file = allophone_file
        self.state_tying_file = state_tying_file
        self.out_hdf_file = self.output_path(f"alignment.hdf")
        self.returnn_root = returnn_root
        self.data_type = data_type
        self.rqmt = {"cpu": 1, "mem": 8, "time": 0.5}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        state_tying = dict(
            (k, int(v))
            for l in open(self.state_tying_file.get_path())
            for k, v in [l.strip().split()[0:2]]
        )
        silence_keys = [key for key in state_tying if "silence" in key.lower()]
        assert len(silence_keys) == 1, "could not automatically infer silence key"
        silence_key = silence_keys[0]

        alignment_cache = FileArchiveBundle(self.alignment_cache.get_path())
        alignment_cache.setAllophones(self.allophone_file.get_path())
        allophones = list(alignment_cache.archives.values())[0].allophones

        returnn_root = (
            None if self.returnn_root is None else self.returnn_root.get_path()
        )
        SimpleHDFWriter = get_returnn_simple_hdf_writer(returnn_root)
        out_hdf = SimpleHDFWriter(filename=self.out_hdf_file, dim=1)
        from returnn.datasets.hdf import HDFDataset

        feature_hdf = HDFDataset([self.feature_hdf.get()])

        db = JsonDatabase(self.json_database.get())
        ds = db.get_dataset(self.dataset_name)

        for ex in ds:
            for channel, wav_file in enumerate(ex["audio_path"]["enhanced"]):
                for (
                    segment_id,
                    source_id,
                    num_samples,
                    offset,
                    channel_alignment,
                ) in zip_strict(
                    range(len(ex["source_id"])),
                    ex["source_id"],
                    ex["num_samples"]["original_source"],
                    ex["offset"]["original_source"],
                    ex["channel_alignment"],
                ):
                    if channel == channel_alignment:
                        seq_name = f"{self.dataset_name}/{ex['example_id']}_{channel}/{segment_id:04d}"
                        features = feature_hdf.get_data_by_seq_tag(seq_name, "data")

                        # alignment
                        targets = []
                        alignment = alignment_cache.read(source_id, "align")
                        alignment_states = [
                            "%s.%d" % (allophones[t[1]], t[2]) for t in alignment
                        ]
                        for allophone in alignment_states:
                            targets.append(state_tying[allophone])

                        # synchronize lengths of alignments and features
                        for _ in range(features.shape[0] - len(targets)):
                            print(f"appending silence frame for segment {seq_name}.")
                            targets.append(state_tying[silence_key])
                        for _ in range(len(targets) - features.shape[0]):
                            print(f"removing last frame for segment {seq_name}.")
                            del targets[-1]

                        data = np.array(targets).astype(np.int32)
                        assert features.shape[0] == data.shape[0], (
                            f"mismatch between features and alignment for seq {seq_name}:"
                            f"{features.shape} vs. {data.shape}"
                        )
                        out_hdf.insert_batch(
                            inputs=data.reshape(1, -1, 1),
                            seq_len=[data.shape[0]],
                            seq_tag=[seq_name],
                        )

        out_hdf.close()
