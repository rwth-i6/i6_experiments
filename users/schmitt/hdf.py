import h5py
from typing import List
import numpy as np

from sisyphus import Path, tk

from i6_core.returnn import ReturnnDumpHDFJob
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.tensor import Dim, Tensor


def load_hdf_data(hdf_path: Path, num_dims: int = 1, segment_list: List = None):
  """
    Load data from an hdf file. Returns dict of form {seq_tag: data}
    :param hdf_path:
    :return:
  """
  assert num_dims in (1, 2), "Currently only 1d and 2d data is supported for reading shape data from hdf"
  with h5py.File(hdf_path.get_path(), "r") as f:
    seq_tags = f["seqTags"][()]
    seq_tags = [tag.decode("utf8") if isinstance(tag, bytes) else tag for tag in
                seq_tags]  # convert from byte to string
    data_dict = {}

    # the data, shapes and seq lengths
    hdf_data = f["inputs"][()]
    shape_data = f["targets"]["data"]["sizes"][()] if num_dims == 2 else None
    seq_lens = f["seqLengths"][()][:, 0]

    # cut out each seq from the flattened 1d tensor according to its seq len and store it in the dict
    # indexed by its seq tag
    for i, seq_len in enumerate(seq_lens):
      if segment_list is None or seq_tags[i] in segment_list:
        data_dict[seq_tags[i]] = hdf_data[:seq_len]
        if shape_data is not None:
          # shape data for 1d: x_1, x_2, x_3, ...
          # shape data for 2d: x_1, y_1, x_2, y_2, x_3, y_3, ...
          shape = shape_data[i * num_dims:(i + 1) * num_dims]
          data_dict[seq_tags[i]] = data_dict[seq_tags[i]].reshape(shape)

      # cut off the data that was just read
      hdf_data = hdf_data[seq_len:]
  return data_dict


def build_hdf_from_alignment(
        alignment_cache: tk.Path,
        allophone_file: tk.Path,
        state_tying_file: tk.Path,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        silence_phone: str = "[SILENCE]",
):
  dataset_config = {
    "class": "SprintCacheDataset",
    "data": {
      "data": {
        "filename": alignment_cache,
        "data_type": "align",
        "allophone_labeling": {
          "silence_phone": silence_phone,
          "allophone_file": allophone_file,
          "state_tying_file": state_tying_file,
        },
      }
    },
  }

  hdf_file = ReturnnDumpHDFJob(
    dataset_config, returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
  ).out_hdf

  return hdf_file


def dump_hdf_numpy(
        hdf_dataset: SimpleHDFWriter,
        data: np.array,
        seq_lens: np.array,
        seq_tags: List[str],
):
  """
  Dump data to an hdf file.

  :param data: torch.Tensor of shape (batch, spatial) with sparse_dim=dimension
  :param seq_lens: torch.Tensor of shape (batch,)
  :param seq_tags: torch.Tensor of shape (batch,)
  :param dimension: int, the sparse dimension of the data
  """
  assert len(data.shape) == 2
  assert len(seq_lens.shape) == 1
  assert data.shape[0] == seq_lens.shape[0]

  seq_lens = {0: seq_lens}
  batch_seq_sizes = np.expand_dims(seq_lens[0], 1)

  hdf_dataset.insert_batch(
    data,
    seq_len=seq_lens,
    seq_tag=list(seq_tags),
    extra={"seq_sizes": batch_seq_sizes}
  )


def dump_hdf_rf(
        hdf_dataset: SimpleHDFWriter,
        data: Tensor,
        batch_dim: Dim,
        seq_tags: Tensor,
):
  """
  Dump data to an hdf file.

  :param data: torch.Tensor of shape (batch, spatial) with sparse_dim=dimension
  :param seq_lens: torch.Tensor of shape (batch,)
  :param seq_tags: torch.Tensor of shape (batch,)
  :param dimension: int, the sparse dimension of the data
  """
  assert len(data.batch_shape) <= 2

  spatial_dims = data.remaining_dims(batch_dim)
  data_raw = data.copy_transpose(
    [batch_dim] + spatial_dims
  ).raw_tensor

  if len(spatial_dims) == 1:
    seq_lens = {0: spatial_dims[0].get_size_tensor().raw_tensor.numpy()}
    batch_seq_sizes = np.expand_dims(seq_lens[0], 1)
  else:
    seq_lens = {}
    batch_seq_sizes = np.zeros((batch_dim.get_dim_value(), 1))

  hdf_dataset.insert_batch(
    data_raw.to(device="cpu").numpy(),
    seq_len=seq_lens,
    seq_tag=list(seq_tags.raw_tensor),
    extra={"seq_sizes": batch_seq_sizes}
  )
  hdf_dataset.close()
