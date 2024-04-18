import h5py
from typing import List

from sisyphus import Path, tk

from i6_core.returnn import ReturnnDumpHDFJob


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
