from sisyphus import Job, Task, tk

import h5py
import numpy
import pickle
import random
import sys

from typing import Any, Dict, Optional

from i6_core.lib import corpus
from i6_core.util import uopen

from i6_experiments.users.rossenbach.lib.hdf import SimpleHDFWriter
from i6_experiments.users.rossenbach.tools.venv import CreatePythonVEnvV2Job



class SpeakerLabelHDFFromBliss(Job):
    """
    Extract speakers from a bliss corpus and create an HDF file with a speaker index
    matching the speaker entry in the corpus speakers for each segment
    """

    def __init__(self, bliss_corpus: tk.Path):
        """
        :param bliss_corpus:
        """
        self.bliss_corpus = bliss_corpus
        self.out_speaker_hdf = self.output_path("speaker_labels.hdf")
        self.out_num_speakers = self.output_var("num_speakers")
        self.out_speaker_dict = self.output_path("speaker_dict.pkl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):

        bliss = corpus.Corpus()
        bliss.load(self.bliss_corpus.get_path())
        speaker_by_index = {}
        index_by_speaker = {}
        num_speakers = len(bliss.speakers)
        self.out_num_speakers.set(num_speakers)
        for i, speaker in enumerate(bliss.all_speakers()):
            speaker_by_index[i] = speaker.name
            index_by_speaker[speaker.name] = i

        pickle.dump(speaker_by_index, uopen(self.out_speaker_dict, "wb"))

        hdf_writer = SimpleHDFWriter(self.out_speaker_hdf.get_path(), dim=num_speakers, ndim=1)

        for recording in bliss.all_recordings():
            for segment in recording.segments:
                speaker_name = segment.speaker_name or recording.speaker_name
                speaker_index = index_by_speaker[speaker_name]
                segment_name = "/".join([bliss.name, recording.name, segment.name])
                hdf_writer.insert_batch(numpy.asarray([[speaker_index]], dtype="int32"), [1], [segment_name])

        hdf_writer.close()
        
        
class ResemblyzerEmbeddingHDFFromBliss(Job):
    """
    Extract speakers from a bliss corpus and create an HDF file with a speaker embedding from Resemblyzer
    """

    def __init__(self, bliss_corpus: tk.Path):
        """
        :param bliss_corpus:
        """
        self.bliss_corpus = bliss_corpus
        self.out_speaker_hdf = self.output_path("resemblyzer_embeddings.hdf")

        self.resemblyzer_site_packages = CreatePythonVEnvV2Job([["resemblyzer"]], venv_extra_args=["--system-site-packages"]).out_python_site_pkg

        self.rqmt = {"cpu": 8, "mem": 24, "time": 24}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        sys.path.insert(0, self.resemblyzer_site_packages.get())
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()

        bliss = corpus.Corpus()
        bliss.load(self.bliss_corpus.get_path())

        # resemblyzer is fixed to 256 dims
        hdf_writer = SimpleHDFWriter(self.out_speaker_hdf.get_path(), dim=256, ndim=2)

        for recording in bliss.all_recordings():
            # support only single segment for now
            assert len(recording.segments) == 1
            wav = preprocess_wav(recording.audio)
            for segment in recording.segments:
                embedding = encoder.embed_utterance(wav)
                segment_name = segment.fullname()
                # insert always as B,T,F, here 1,1,F
                hdf_writer.insert_batch(
                    numpy.asarray([[embedding]], dtype="float32"), [1], [segment_name]
                )
        hdf_writer.close()


class DistributeDynamicSpeakerEmbeddingsJob(Job):
    """
    distribute speaker embeddings contained in an hdf file to a new hdf file with mappings to the given bliss corpus
    """

    def __init__(
            self,
            bliss_corpus: tk.Path,
            speaker_embedding_hdf: tk.Path,
            options: Optional[Dict[str, Any]] = None,
        ):
        """

        :param bliss_corpus:
        :param speaker_embedding_hdf:
        :param options:
        """
        self.bliss_corpus = bliss_corpus
        self.speaker_embedding_hdf = speaker_embedding_hdf
        self.options = options
        if self.options is None:
            self.options = {'mode': 'random'}
        else:
            for key in self.options.keys():
                assert key in ["mode", "seed"], f"invalid option key: {key}"

        assert self.options['mode'] in ['random'], "invalid mode %s" % options['mode']

        self.out = self.output_path("speaker_embeddings.hdf")

        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task('run', mini_task=True)

    def _random(
            self,
            c,
            hdf_writer,
            speaker_embedding_features,
    ):
        if 'seed' in self.options:
            random.seed(self.options['seed'])

        random.shuffle(speaker_embedding_features)

        # shuffle once and do a rotating distribution in order to achieve a rather equal distribution of available embeddings
        embedding_index = 0
        for recording in c.recordings:
            assert len(recording.segments) == 1
            segment = recording.segments[0]  # type:corpus.Segment

            segment_name = segment.fullname()

            hdf_writer.insert_batch(numpy.asarray([speaker_embedding_features[embedding_index]]),
                                         [1],
                                         [segment_name])
            embedding_index += 1
            if embedding_index >= len(speaker_embedding_features):
                embedding_index = 0

    def run(self):

        speaker_embedding_data = h5py.File(self.speaker_embedding_hdf.get(), 'r')
        speaker_embedding_inputs = speaker_embedding_data['inputs']
        speaker_embedding_raw_tags = speaker_embedding_data['seqTags']
        speaker_embedding_lengths = speaker_embedding_data['seqLengths']

        speaker_embedding_features = []
        speaker_embedding_tags = []
        offset = 0
        for tag, length in zip(speaker_embedding_raw_tags, speaker_embedding_lengths):
            speaker_embedding_features.append(speaker_embedding_inputs[offset:offset + length[0]])
            speaker_embedding_tags.append(tag)
            offset += length[0]

        hdf_writer = SimpleHDFWriter(self.out.get(), dim=speaker_embedding_features[0].shape[-1], ndim=2)

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get())

        mode = self.options.get('mode')
        if mode == "random":
            self._random(c, hdf_writer, speaker_embedding_features)
        #elif mode == "length_buckets":
        #    self._random_matching_length()
        else:
            assert False

        hdf_writer.close()
