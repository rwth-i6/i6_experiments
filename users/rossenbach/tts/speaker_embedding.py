from sisyphus import *
import numpy
import pickle

from i6_experiments.users.rossenbach.lib.hdf import SimpleHDFWriter
from i6_core.lib import corpus
from i6_core.util import uopen


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