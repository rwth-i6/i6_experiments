import h5py
from sisyphus import Job, Task, tk
import numpy as np
import torch
from i6_core.lib.hdf import get_returnn_simple_hdf_writer


class AverageXVectorSpeakerEmbeddings(Job):
    def __init__(self, x_vector_hdf: tk.Path, returnn_root: tk.Path):
        self.x_vector_hdf = x_vector_hdf
        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("output.hdf")

    def task(self):
        yield Task("run", mini_task=True)

    def run(self):
        x_vectors, seq_tags, speaker_labels = self.load_xvector_data(self.x_vector_hdf)

        speaker_labels, indices = torch.sort(torch.Tensor(speaker_labels))
        x_vectors = x_vectors[indices, :]
        seq_tags = list(np.array(seq_tags)[indices.detach().cpu().numpy()])

        length = 0
        current_label = speaker_labels[0]
        label_lengths = []
        for l in speaker_labels:
            if l != current_label:
                label_lengths.append(length)
                length = 0
                current_label = l
            length += 1
        label_lengths.append(length)

        offset = 0
        pooled_x_vectors = []
        for l in label_lengths:
            speaker_x_vector = x_vectors[offset:offset + l]
            pooled_x_vectors.append(torch.mean(speaker_x_vector))
            offset += l

        # TODO: Write to HDF


def load_xvector_data(hdf_filename):
    input_data = h5py.File(hdf_filename, "r")

    inputs = input_data["inputs"]
    seq_tags = input_data["seqTags"]
    lengths = input_data["seqLengths"]

    assert "targets" in input_data.keys()
    assert "speaker_labels" in input_data["targets"]["data"]
    speaker_labels = input_data["targets"]["data"]["speaker_labels"]

    data_seqs = []
    data_tags = []
    data_speaker_label = []
    offset = 0

    for tag, length, speaker_label in zip(seq_tags, lengths, speaker_labels):
        tag = tag if isinstance(tag, str) else tag.decode()
        in_data = inputs[offset : offset + length[0]]
        data_seqs.append(in_data)
        offset += length[0]
        data_tags.append(tag)
        data_speaker_label.append(speaker_label)

    return torch.Tensor(data_seqs), data_tags, data_speaker_label
