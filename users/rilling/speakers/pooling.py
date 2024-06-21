import h5py
from sisyphus import Job, Task, tk
import numpy as np
import torch
from i6_core.lib.hdf import get_returnn_simple_hdf_writer


class AverageXVectorSpeakerEmbeddingsJob(Job):
    """
    Job to compute the average x-vector for each speaker in a corpus
    Takes an HDF containing an x-vector for each sequence in a corpus in combination with the speaker id, 
    collects all x-vectors for every speaker id and pools them using the mean vector. 

    :param tk.Path x_vector_hdf: Path to HDF from x-vector forward job
    :param tk.Path returnn_root: Path to Returnn Root
    """    
    def __init__(self, x_vector_hdf: tk.Path, returnn_root: tk.Path):
        self.x_vector_hdf = x_vector_hdf
        self.returnn_root = returnn_root

        self.out_hdf = self.output_path("output.hdf")

    def task(self):
        yield Task("run", mini_task=True, rqmt={"sbatch_args": ["-p", "cpu_slow"]})

    def run(self):
        x_vectors, seq_tags, speaker_labels = self.load_xvector_data(self.x_vector_hdf.get_path())

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
            pooled_x_vectors.append(torch.mean(speaker_x_vector, 0))
            offset += l

        HDFWriter = get_returnn_simple_hdf_writer(self.returnn_root)
        hdf_writer = HDFWriter(self.out_hdf.get_path(), dim=pooled_x_vectors[0].shape, ndim=1)

        for i, s in enumerate(speaker_labels):
            hdf_writer.insert_batch(
                pooled_x_vectors[int(s)].unsqueeze(0).numpy(), [512], [seq_tags[i]], extra={"speaker_labels": np.array([[int(s)]])}
            )

        hdf_writer.close()

    def load_xvector_data(self, hdf_filename):
        input_data = h5py.File(hdf_filename, "r")

        inputs = input_data["inputs"]
        seq_tags = input_data["seqTags"]
        lengths = input_data["seqLengths"]

        # break if targets are not given hence no speaker_labels are in the HDF
        assert "targets" in input_data.keys()
        assert "speaker_labels" in input_data["targets"]["data"]
        speaker_labels = input_data["targets"]["data"]["speaker_labels"]

        x_vectors = np.array(inputs)
        x_vectors = x_vectors.reshape((lengths.shape[0], 512))

        return torch.Tensor(x_vectors), seq_tags, speaker_labels
