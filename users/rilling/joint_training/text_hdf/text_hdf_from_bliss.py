from sisyphus import *
import numpy as np
from typing import Union, List

from i6_experiments.users.rossenbach.lib.hdf import SimpleHDFWriter
from i6_core.lib import corpus
from i6_core.util import uopen

from returnn.datasets.util.vocabulary import Vocabulary


class TextHDFFromBliss(Job):
    """
    Extract text from a bliss corpus and save to HDF format
    E.g. used for joint training, where additional ASR corpus is necessary.
    """

    def __init__(
        self,
        bliss_corpora: Union[tk.Path, List[tk.Path]],
        vocab_files: Union[tk.Path, List[tk.Path]],
        unknown_label="[UNKNOWN]",
    ):
        if isinstance(bliss_corpora, list):
            assert isinstance(vocab_files, list)
            self.bliss_corpora = bliss_corpora
            self.vocab_files = vocab_files
        else:
            self.bliss_corpora = [bliss_corpora]
            self.vocab_files = [vocab_files]
        self.unknown_label = unknown_label
        self.out_text_hdf = self.output_path("corpus_text.hdf")

    def task(self):
        yield Task("run", mini_task=True)

    def run(self):
        vocabs = []
        vocab_num_labels = None
        for v in self.vocab_files:
            vocabs.append(Vocabulary(v.get_path(), unknown_label=self.unknown_label))
            if vocab_num_labels is None:
                vocab_num_labels=vocabs[-1].num_labels
            else:
                assert vocabs[-1].num_labels == vocab_num_labels

        hdf_writer = SimpleHDFWriter(self.out_text_hdf.get_path(), dim=vocab_num_labels, ndim=1)

        for b, v in zip(self.bliss_corpora, vocabs):
            bliss = corpus.Corpus()
            bliss.load(b.get_path())

            for recording in bliss.all_recordings():
                for segment in recording.segments:
                    transformed = v.get_seq(segment.orth)
                    segment_name = "/".join([bliss.name, recording.name, segment.name])
                    hdf_writer.insert_batch(
                        np.asarray([transformed], dtype="int32"), [len(transformed)], [segment_name]
                    )
        hdf_writer.close()
