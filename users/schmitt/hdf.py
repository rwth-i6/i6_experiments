import h5py
from typing import List, Optional, Union
import numpy as np
import torch

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
        batch_dim: Optional[Dim],
        seq_tags: Union[Tensor, List[str]],
):
  """
  Dump data to an hdf file.

  :param data: torch.Tensor of shape (batch, spatial) with sparse_dim=dimension
  :param seq_lens: torch.Tensor of shape (batch,)
  :param seq_tags: torch.Tensor of shape (batch,)
  :param dimension: int, the sparse dimension of the data
  """
  if batch_dim is None:
    spatial_dims = data.dims
    data_raw = data.raw_tensor[None, :]
  else:
    spatial_dims = data.remaining_dims(batch_dim)
    data_raw = data.copy_transpose(
      [batch_dim] + spatial_dims
    ).raw_tensor

  n_batch = data_raw.shape[0]

  if len(spatial_dims) > 0:
    seq_lens = {}
    for i, dim in enumerate(spatial_dims):
      if dim.is_dynamic():
        size_tensor = dim.get_size_tensor().raw_tensor
        if isinstance(size_tensor, torch.Tensor):
          size_tensor = size_tensor.numpy()
        else:
          assert isinstance(size_tensor, np.ndarray)
          size_tensor = np.array([size_tensor.item()])
        seq_lens[i] = size_tensor

    batch_seq_sizes = np.zeros((n_batch, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
  else:
    seq_lens = {}
    batch_seq_sizes = np.zeros((n_batch, 1))

  seq_tags = list(seq_tags.raw_tensor) if isinstance(seq_tags, Tensor) else seq_tags

  if isinstance(data_raw, torch.Tensor):
    if data_raw.requires_grad:
      data_raw = data_raw.detach()  # cannot call .numpy() on a tensor that requires grad
    data_raw = data_raw.cpu().numpy()


  hdf_dataset.insert_batch(
    data_raw,
    seq_len=seq_lens,
    seq_tag=seq_tags,
    extra={"seq_sizes": batch_seq_sizes}
  )
  # hdf_dataset.close()
