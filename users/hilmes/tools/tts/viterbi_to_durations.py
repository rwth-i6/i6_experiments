import numpy
from sisyphus import tk, Job, Task
from i6_private.users.rossenbach.lib.hdf import SimpleHDFWriter
import h5py
import sys


class ViterbiToDurationsJob(Job):
  def __init__(
    self,
    viterbi_alignment,
    skip_token=78,
    dataset_to_check=None,
    time_rqmt=2,
    mem_rqmt=4,
  ):
    """
    :param Path viterbi_alignment: Path to the alignment HDF produced by CTC/Viterbi
    :param skip_token: Value of the blank token in CTC. This is the last value in the vocabulary.
    """
    self.skip_token = skip_token
    self.align = viterbi_alignment
    self.check = dataset_to_check
    self.out_durations_hdf = self.output_path("durations.hdf")
    self.rqmt = {"time": time_rqmt, "mem": mem_rqmt}

  def tasks(self):
    yield Task("run", rqmt=self.rqmt)

  def run(self):
    # Load HDF data
    input_dur_data = h5py.File(self.align.get_path(), "r")
    num_seqs = -1
    inputs = input_dur_data["inputs"]
    seq_tags = input_dur_data["seqTags"]
    lengths = input_dur_data["seqLengths"]
    alignment, tags = self.load_normal_data(inputs, seq_tags, lengths, num_seqs)

    durations_total = []

    # sort based on tags
    if self.check is not None:
      check, check_tag = self.load_spect_data(self.check)

    # Alignemnt to duration conversion
    for alignment_idx, s in enumerate(alignment):
      durations = []
      for idx, p in enumerate(s):
        # Skip_token only appears now if 2 labels following each other are the same.
        #   Example [1,skip_token,1] -> [1,1]
        if p == self.skip_token:
          durations[-1] += 1
          continue
        if idx != 0 and s[idx] != s[idx - 1]:
          durations.append(1)
        elif idx != 0 and s[idx] == s[idx - 1]:
          durations[-1] += 1
        else:
          durations.append(1)
      # Check if lengths match if dataset is provided
      if self.check is not None:
        assert sum(durations) == len(check[alignment_idx]), (
          f"durations {sum(durations)} and spectrogram length {len(check[alignment_idx])}"
          f"do not match in length "
        )
      durations_total.append(durations)

    # Dump into HDF
    new_lengths = []
    for seq in durations_total:
      new_lengths.append([len(seq), 2, 2])
    duration_sequence = numpy.hstack(durations_total).astype(numpy.int32)
    dim = 1
    writer = SimpleHDFWriter(self.out_durations_hdf.get_path(), dim=dim, ndim=2)
    offset = 0
    for tag, length in zip(tags, new_lengths):
      in_data = duration_sequence[offset : offset + length[0]]
      in_data = numpy.expand_dims(in_data, axis=1)
      offset += length[0]
      writer.insert_batch(numpy.asarray([in_data]), [in_data.shape[0]], [tag])
    print(f"Succesfully converted durations into {(self.out_durations_hdf.get_path())}")
    writer.close()

  def load_normal_data(self, inputs, seq_tags, lengths, num_seqs):
    sequences = []
    tags = []
    offset = 0
    for tag, length in zip(seq_tags, lengths):
      tag = tag if isinstance(tag, str) else tag.decode()
      in_data = inputs[offset : offset + length[0]]
      sequences.append(in_data)
      offset += length[0]
      tags.append(tag)
      if len(sequences) == num_seqs:
        break
    return numpy.array(sequences), tags

  def load_spect_data(self, input_path):
    input_data = h5py.File(input_path.get_path(), "r")
    num_seqs = -1
    offset = 0
    inputs = input_data["inputs"]
    seq_tags = input_data["seqTags"]
    lengths = input_data["seqLengths"]

    # Load spectrogram data
    data_seq = []
    data_tag = []
    for tag, length in zip(seq_tags, lengths):
      tag = tag if isinstance(tag, str) else tag.decode()
      in_data = inputs[offset : offset + length[0]]
      data_seq.append(in_data)
      offset += length[0]
      data_tag.append(tag)
      if len(data_seq) == num_seqs:
        break
    return numpy.array(data_seq), data_tag
