from typing import Any, Dict, Iterator, Optional, Union
import copy
import os
import shutil
import subprocess as sp
import ast

import numpy as np

from returnn.datasets.hdf import SimpleHDFWriter

from i6_core.lib import corpus

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase


class DumpCorpusTextAsUtf8ToHdfJob(Job):
    def __init__(
        self,
        bliss_corpus: tk.Path,
        concurrent: int = 1,
    ):
        """

        Args:
            text_file: each line contains a sequence of phonemes separated by space
            phoneme_vocab:
            concurrent: number of concurrent hdf files to dump
            fixed_random_subset: if given, only use a fixed random subset of the data
        """
        self.bliss_corpus = bliss_corpus
        self.concurrent = concurrent

        self.out_hdfs = {i: self.output_path(f"data_{i}.hdf") for i in range(self.concurrent)}

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 4}, args=range(1, self.concurrent + 1))

    def run(self, task_id):
        import random
        import tempfile

        corpus_ = corpus.Corpus()
        corpus_.load(self.bliss_corpus.get_path())

        segments = [segment for recording in corpus_.recordings for segment in recording.segments]
        random.Random(42).shuffle(segments)
        num_segments = len(segments)
        segments = [segments[i] for i in range(num_segments) if (i % self.concurrent) == (task_id - 1)]
        num_segments = len(segments)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_hdf = os.path.join(tmp_dir, "out.hdf")
            hdf_writer = SimpleHDFWriter(filename=tmp_hdf, dim=256, ndim=1)

            for i, segment in enumerate(segments):
                text = segment.orth
                seq_tag = segment.fullname()

                data = np.array([list(text.encode("utf8"))], dtype=np.uint8)  # (1, T)

                seq_len = data.shape[1]
                seq_lens = {0: np.array([seq_len])}
                batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

                hdf_writer.insert_batch(
                    data,
                    seq_len=seq_lens,
                    seq_tag=[seq_tag],
                    extra={"seq_sizes": batch_seq_sizes},
                )

                if i % 10_000 == 0:
                    print(f"Processed sequence {i}/{num_segments} ({i / num_segments * 100}%)")

            hdf_writer.close()
            shutil.move(tmp_hdf, self.out_hdfs[task_id - 1].get_path())
