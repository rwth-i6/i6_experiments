from sisyphus import *
import h5py
import numpy as np
from typing import Union, List

from i6_private.users.rossenbach.lib.hdf import SimpleHDFWriter

class ApplyLogOnHDFJob(Job):

  def __init__(self, hdf_file: tk.Path, normalize: bool = True):

    self.normalize = normalize
    self.hdf_file = hdf_file

    self.out_hdf = self.output_path("out.hdf")
    self.out_mean = self.output_var("mean")
    self.out_std = self.output_var("std")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):

    hdf_data = h5py.File(
      self.hdf_file.get_path(), "r"
    )
    inputs = hdf_data["inputs"]
    raw_tags = hdf_data["seqTags"]
    lengths = hdf_data["seqLengths"]

    seqs = {}
    offset = 0
    for idx in range(len(raw_tags)):
      data = inputs[offset:offset+lengths[idx][0]]  # 0 is the frame length not the text length (frame level energy)
      data += 1e-10
      assert all(data > 0), ("Cant apply log to 0", data, np.sum(data == 0))
      seqs[raw_tags[idx]] = np.log(data)
      offset += lengths[idx][0]

    values = np.concatenate(list(seqs.values()))
    mean = np.mean(values)
    std = np.std(values)
    self.out_mean.set(mean)
    self.out_std.set(std)
    hdf_writer = SimpleHDFWriter(
            self.out_hdf.get_path(),
            dim=1
    )
    for segment_tag, logs in seqs.items():
      data = np.asarray([logs])
      if self.normalize:
        data = (data - mean) / std
      hdf_writer.insert_batch(
        np.float32(data),
        [data.shape[1]],
        [segment_tag],
      )

    hdf_writer.close()
