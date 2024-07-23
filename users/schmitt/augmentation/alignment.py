import numpy as np
from matplotlib import pyplot as plt
import ast
from collections import Counter
import torch
from torch.utils.data import DataLoader
from typing import Sequence

from i6_experiments.users.schmitt.hdf import load_hdf_data
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils

import returnn.frontend as rf
from returnn.tensor import Dim
from returnn.datasets.hdf import HDFDataset
from returnn.datasets.basic import Dataset
from returnn.torch.data import extern_data as extern_data_util

from sisyphus import Path


def shift_alignment_boundaries_single_seq(alignment: np.ndarray, blank_idx: int, max_shift: int):
  import tensorflow as tf
  alignment = tf.convert_to_tensor(alignment)
  labels = alignment[tf.not_equal(alignment, blank_idx)]
  label_positions = tf.cast(tf.where(tf.not_equal(alignment, blank_idx)), tf.int32)[:, 0]  # [S]
  # prepend -1 and append seq_len of alignment to the positions to calculate the distances to the boundaries
  label_positions_ext = tf.concat(([-1], label_positions, [alignment.shape[0] - 1]), axis=0)  # [S+2]
  # for each label, the distance to the next label on the right and left (or to the sequence boundaries)
  distance_right = label_positions_ext[2:] - label_positions - 1  # [S]
  distance_left = label_positions - label_positions_ext[:-2] - 1  # [S]
  # find the uneven distances
  distance_right_uneven = distance_right[:-1] % 2 != 0  # [S-1]
  distance_left_uneven = distance_left[1:] % 2 != 0  # [S-1]
  num_uneven = tf.reduce_sum(tf.cast(distance_right_uneven, tf.int32))
  assert tf.math.reduce_all(distance_right_uneven == distance_left_uneven), "Uneven distances are not equal."
  # randomly choose whether to ceil or floor the uneven distances
  random_ceil = tf.random.uniform((num_uneven,), minval=0, maxval=2, dtype=tf.int32)
  random_ceil_right = tf.tensor_scatter_nd_update(
    tf.zeros_like(distance_right[1:], dtype=tf.int32), tf.where(distance_right_uneven), random_ceil)
  random_ceil_left = tf.tensor_scatter_nd_update(
    tf.ones_like(distance_left[:-1], dtype=tf.int32), tf.where(distance_left_uneven), random_ceil)
  # can at most use half of the space to the right except for the last (or first) label
  # use floor division in both cases and then apply random ceil
  distance_right = tf.concat((distance_right[:-1] // 2, [distance_right[-1]]), axis=0)  # [S]
  distance_left = tf.concat(([distance_left[0]], distance_left[1:] // 2), axis=0)  # [S]
  distance_right += tf.concat((random_ceil_right, [0]), axis=0)
  distance_left += tf.concat(([0], 1 - random_ceil_left), axis=0)
  # random shift is either max_shift or the available space to the right or left
  random_shift = tf.random.uniform(
    (tf.shape(label_positions)[0],), minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)  # [S]
  random_shift = tf.clip_by_value(random_shift, -distance_left, distance_right)  # TODO bias problem to boundaries
  # new positions are the old positions plus the random shift
  new_positions = label_positions + random_shift
  assert tf.reduce_all(new_positions[1:] > new_positions[:-1]), "New positions are not sorted anymore."
  # scatter the labels into the new alignment according to the new positions
  new_alignment = tf.fill(alignment.shape, blank_idx)
  new_alignment = tf.tensor_scatter_nd_update(new_alignment, tf.expand_dims(new_positions, axis=1), labels)
  return new_alignment.numpy()


def shift_alignment_boundaries_batched(
        alignment: rf.Tensor,
        alignment_spatial_dim: Dim,
        batch_dims: Sequence[Dim],
        blank_idx: int,
        max_shift: int,
):
  non_blank_mask = rf.logical_and(
      alignment != rf.convert_to_tensor(blank_idx),
      rf.sequence_mask(alignment.dims)
  )
  labels, labels_spatial_dim = utils.get_masked(alignment, non_blank_mask, alignment_spatial_dim, batch_dims)

  label_positions = rf.where(
    non_blank_mask, rf.range_over_dim(alignment_spatial_dim), rf.convert_to_tensor(-1))
  label_positions, _ = utils.get_masked(
    label_positions,
    non_blank_mask,
    alignment_spatial_dim,
    batch_dims,
    labels_spatial_dim
  )

  labels_spatial_dim.dyn_size_ext.raw_tensor = rf.copy_to_device(labels_spatial_dim.dyn_size_ext, alignment.device).raw_tensor
  singleton_dim = Dim(description="singleton", dimension=1, kind=Dim.Types.Spatial)
  label_positions_ext, label_positions_ext_spatial_dim = rf.concat(
    (rf.expand_dim(rf.copy_to_device(alignment_spatial_dim.dyn_size_ext, alignment.device), singleton_dim), singleton_dim),
    (rf.reverse_sequence(label_positions, axis=labels_spatial_dim), labels_spatial_dim),
    allow_broadcast=True,
  )
  label_positions_ext, label_positions_ext_spatial_dim = rf.concat(
    (rf.expand_dim(rf.convert_to_tensor(-1, device=alignment.device), singleton_dim), singleton_dim),
    (rf.reverse_sequence(label_positions_ext, axis=label_positions_ext_spatial_dim), label_positions_ext_spatial_dim),
    allow_broadcast=True,
  )

  # for each label, the distance to the next label on the right and left (or to the sequence boundaries)
  distance_right = rf.gather(
    label_positions_ext,
    indices=rf.range_over_dim(labels_spatial_dim) + 2,
    axis=label_positions_ext_spatial_dim
  ) - label_positions - 1  # [S]
  distance_left = label_positions - rf.gather(
    label_positions_ext,
    indices=rf.range_over_dim(labels_spatial_dim),
    axis=label_positions_ext_spatial_dim
  ) - 1  # [S]

  # find the (common) uneven distances
  # ignore the last right distance and the first left distance as they are not shared
  labels_spatial_dim_minus_one = labels_spatial_dim - 1
  labels_spatial_dim_minus_one_range = rf.range_over_dim(labels_spatial_dim_minus_one)
  distance_right_uneven = rf.gather(
    rf.convert_to_tensor(distance_right % 2 != 0),
    indices=labels_spatial_dim_minus_one_range,
    axis=labels_spatial_dim
  )  # [S-1]
  distance_left_uneven = rf.gather(
    rf.convert_to_tensor(distance_left % 2 != 0),
    indices=labels_spatial_dim_minus_one_range + 1,
    axis=labels_spatial_dim
  )  # [S-1]

  assert rf.reduce_all(
    distance_right_uneven == distance_left_uneven,
    axis=labels_spatial_dim_minus_one,
    use_mask=False,  # not implemented otherwise
  ), "Uneven distances are not equal."

  # randomly choose whether to ceil or floor the uneven distances
  random_ceil = rf.random_uniform(dims=distance_right_uneven.dims, dtype="int32", minval=0, maxval=2)

  singleton_zero_tensor = rf.zeros(batch_dims + [singleton_dim], dtype="int32")
  random_ceil_right = rf.where(distance_right_uneven, random_ceil, rf.zeros_like(random_ceil))
  random_ceil_right, _ = rf.concat(
    (random_ceil_right, labels_spatial_dim_minus_one),
    (singleton_zero_tensor, singleton_dim),
    out_dim=labels_spatial_dim,
  )
  random_ceil_left = rf.where(distance_left_uneven, random_ceil, rf.ones_like(random_ceil))
  random_ceil_left, _ = rf.concat(
    (singleton_zero_tensor + 1, singleton_dim),
    (random_ceil_left, labels_spatial_dim_minus_one),
    out_dim=labels_spatial_dim,
  )

  # can at most use half of the space to the right except for the last (or first) label
  # use floor division in both cases and then apply random ceil
  distance_right, _ = rf.concat(
    (rf.gather(distance_right, indices=labels_spatial_dim_minus_one_range, axis=labels_spatial_dim) // 2, labels_spatial_dim_minus_one),
    (rf.gather(distance_right, indices=singleton_zero_tensor + labels_spatial_dim.dyn_size_ext - 1, axis=labels_spatial_dim), singleton_dim),
    out_dim=labels_spatial_dim,
  )
  distance_left, _ = rf.concat(
    (rf.gather(distance_left, indices=singleton_zero_tensor, axis=labels_spatial_dim), singleton_dim),
    (rf.gather(distance_left, indices=labels_spatial_dim_minus_one_range + 1, axis=labels_spatial_dim) // 2, labels_spatial_dim_minus_one),
    out_dim=labels_spatial_dim,
  )

  distance_right += random_ceil_right
  distance_left += 1 - random_ceil_left

  # random shift is either max_shift or the available space to the right or left
  random_shift = rf.random_uniform(
    dims=label_positions.dims, dtype="int32", minval=-max_shift, maxval=max_shift + 1)
  random_shift = rf.clip_by_value(random_shift, -distance_left, distance_right)

  # new positions are the old positions plus the random shift
  new_positions = label_positions + random_shift
  assert rf.reduce_all(
    rf.gather(
      new_positions, indices=labels_spatial_dim_minus_one_range + 1, axis=labels_spatial_dim
    ) > rf.gather(
      new_positions, indices=labels_spatial_dim_minus_one_range, axis=labels_spatial_dim
    ),
    axis=labels_spatial_dim_minus_one,
    use_mask=False,  # not implemented otherwise
  ), "New positions are not sorted anymore."

  # set new positions in the padded area to the (last + 1) position of the new alignment and cut that position off later
  alignment_spatial_dim_plus_one = alignment_spatial_dim + 1
  new_positions = rf.cast(new_positions, "int64")
  new_positions = rf.where(
    rf.sequence_mask(new_positions.dims),
    new_positions,
    rf.copy_to_device(
      rf.reduce_max(
        alignment_spatial_dim_plus_one.dyn_size_ext, axis=alignment_spatial_dim_plus_one.dyn_size_ext.dims) - 1,
      alignment.device
    )
  )

  # extend the alignment by one in the spatial dimension to store the labels in the padded area
  new_alignment_ext = alignment.copy_template_replace_dim_tag(
    axis=alignment.get_axis_from_description(alignment_spatial_dim),
    new_dim_tag=alignment_spatial_dim_plus_one
  )
  new_alignment_ext = new_alignment_ext.copy_transpose(batch_dims + [alignment_spatial_dim_plus_one])

  # scatter the labels into the new alignment according to the new positions
  new_alignment_ext.raw_tensor = torch.full(
    rf.zeros_like(new_alignment_ext).raw_tensor.shape,
    blank_idx,
    dtype=torch.int32,
    device=alignment.device
  ).scatter(
    dim=1,
    index=new_positions.raw_tensor.long(),
    src=labels.copy_transpose(batch_dims + [labels_spatial_dim]).raw_tensor,
  )

  new_alignment = alignment.copy()
  # cut off the last position in the spatial dimension
  new_alignment.raw_tensor = new_alignment_ext.raw_tensor[:, :-1]

  return new_alignment


def test_alignment_augmentation_single_seq(hdf_path: str):
  def calculate_shift_statistics_over_corpus(max_shift, num_iterations, alignment_data, blank_idx):
    label_position_diff_counter = Counter()
    for i, (seq_tag, alignment_old) in enumerate(alignment_data.items()):
      if i % 100 == 0:
        print(i)

      alignment_new = alignment_old.copy()
      for _ in range(num_iterations):
        alignment_new = shift_alignment_boundaries_single_seq(alignment_new, blank_idx=blank_idx, max_shift=max_shift)

      label_positions_old = np.argwhere(alignment_old != blank_idx)[:, 0]
      label_positions_new = np.argwhere(alignment_new != blank_idx)[:, 0]
      label_position_diffs = label_positions_new - label_positions_old
      label_position_diff_counter.update(label_position_diffs)

    plt.bar(label_position_diff_counter.keys(), label_position_diff_counter.values())
    plt.title(f"Label position differences for \nmax_shift={max_shift} and {num_iterations} iterations.")
    plt.show()
    plt.close()

  def compare_alignments(seq_tag, alignment_data, vocab, blank_idx, max_shift, num_iterations):
    from i6_experiments.users.schmitt.visualization.visualization import PlotAlignmentJob
    alignment_old = alignment_data[seq_tag]
    alignment_new = alignment_old.copy()
    for _ in range(num_iterations):
      alignment_new = shift_alignment_boundaries_single_seq(alignment_new, blank_idx=blank_idx, max_shift=max_shift)

    labels_old = alignment_old[alignment_old != blank_idx]
    labels_new = alignment_new[alignment_new != blank_idx]

    fig, ax_old = PlotAlignmentJob._get_fig_ax(alignment_old)
    PlotAlignmentJob._set_ticks(ax_old, alignment_old, labels_old, vocab, blank_idx, ymin=0.5)
    ax_new = ax_old.twiny()
    PlotAlignmentJob._set_ticks(ax_new, alignment_new, labels_new, vocab, blank_idx, ymax=0.5, color="g")
    plt.title(f"Comparison of alignment for {seq_tag} \nwith max_shift={max_shift} and {num_iterations} iterations.")
    plt.show()
    plt.close()

  # load vocabulary as dictionary
  with open("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab", "r") as f:
    json_data = f.read()
    vocab = ast.literal_eval(json_data)  # label -> idx
    vocab = {v: k for k, v in vocab.items()}  # idx -> label
  blank_idx = 10025
  alignment_data = load_hdf_data(Path(hdf_path))

  max_shift = 5
  num_iterations = 5
  calculate_shift_statistics_over_corpus(
    max_shift=max_shift, num_iterations=num_iterations, alignment_data=alignment_data, blank_idx=blank_idx)
  compare_alignments(
    seq_tag="dev-other/4570-56594-0002/4570-56594-0002",
    alignment_data=alignment_data,
    vocab=vocab,
    blank_idx=blank_idx,
    max_shift=max_shift,
    num_iterations=num_iterations
  )


def test_alignment_augmentation_batched(hdf_path: str):
  from returnn.torch.data import pipeline as data_pipeline
  from returnn.torch.data import returnn_dataset_wrapper
  from returnn.tensor import batch_dim

  # from test_torch_dataset.py
  def get_dataloader(
          dataset: Dataset, mp_manager: torch.multiprocessing.Manager, *, batch_size: int = 5, max_seqs: int = 2
  ) -> DataLoader:
    # Follow mostly similar logic as in the PT engine.

    epoch_mp_shared = mp_manager.Value("i", 0)
    epoch_mp_shared.value = 1
    reset_callback = returnn_dataset_wrapper.ReturnnDatasetResetMpSharedEpochCallback(
      dataset=dataset, epoch_mp_shared=epoch_mp_shared
    )

    wrapped_dataset = returnn_dataset_wrapper.ReturnnDatasetIterDataPipe(dataset, reset_callback=reset_callback)

    batches_dataset = data_pipeline.BatchingIterDataPipe(wrapped_dataset, batch_size=batch_size, max_seqs=max_seqs)

    # Test different ways to deepcopy/serialize the dataset.
    # This is what DataLoader2 also would do, although DataLoader2 also uses dill as a fallback,
    # if it is available.
    # Dill is not always available though,
    # so it is important that we make sure that it also works without dill.

    from copy import deepcopy

    deepcopy(batches_dataset)

    import pickle

    pickle.loads(pickle.dumps(batches_dataset))

    return data_pipeline.create_data_loader_from_batches(batches_dataset, {
      "num_workers": 1
    })

  blank_idx = 10025
  max_shift = 2
  num_iterations = 4

  hdf_dataset = HDFDataset([hdf_path])
  hdf_dataset.initialize()
  hdf_dataset.init_seq_order(epoch=1)

  out_spatial_dim = Dim(description="out_spatial", dimension=None, kind=Dim.Types.Spatial)
  vocab_dim = Dim(description="vocab", dimension=10026, kind=Dim.Types.Spatial)

  extern_data_dict = {
    "data": {
      "dim_tags": [batch_dim, out_spatial_dim],
      "sparse_dim": vocab_dim
    },
  }
  extern_data = extern_data_util.extern_data_template_from_config_opts(extern_data_dict)

  mp_manager = torch.multiprocessing.Manager()
  dataloader = get_dataloader(hdf_dataset, mp_manager, batch_size=10_000, max_seqs=10)
  data_iter = iter(dataloader)

  label_position_diff_counter = Counter()
  rf.select_backend_torch()

  seq_idx = 0
  while True:
    try:
      extern_data_raw = next(data_iter)
    except StopIteration:
      break

    extern_data_tensor_dict = extern_data_util.raw_dict_to_extern_data(
      extern_data_raw, extern_data_template=extern_data, device="cpu"
    )
    alignment_old = extern_data_tensor_dict["data"]

    batch_axis = alignment_old.get_axis_from_description(batch_dim)
    seq_idx += alignment_old.dims[batch_axis].dyn_size_ext.raw_tensor.item()
    print(f"Processing sequence {seq_idx}/{hdf_dataset.num_seqs}")

    alignment_new = alignment_old.copy()
    for _ in range(num_iterations):
      alignment_new = shift_alignment_boundaries_batched(
        alignment_new,
        alignment_spatial_dim=out_spatial_dim,
        batch_dims=[batch_dim],
        blank_idx=blank_idx,
        max_shift=max_shift
      )

    alignment_old_raw = alignment_old.copy_transpose([batch_dim, out_spatial_dim]).raw_tensor
    alignment_new_raw = alignment_new.copy_transpose([batch_dim, out_spatial_dim]).raw_tensor
    seq_mask_raw = rf.sequence_mask(alignment_old.dims).copy_transpose([batch_dim, out_spatial_dim]).raw_tensor

    label_positions_old = torch.where(
      torch.logical_and(
        alignment_old_raw != blank_idx,
        seq_mask_raw
      )
    )[1]  # [S]
    label_positions_new = torch.where(
      torch.logical_and(
        alignment_new_raw != blank_idx,
        seq_mask_raw
      )
    )[1]  # [S]

    label_position_diffs = label_positions_new - label_positions_old
    label_position_diff_counter.update(label_position_diffs.numpy())

  plt.bar(label_position_diff_counter.keys(), label_position_diff_counter.values())
  plt.title(f"Label position differences for \nmax_shift={max_shift} and {num_iterations} iterations.")
  plt.show()
  plt.close()


if __name__ == "__main__":
  test_alignment_augmentation_batched("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/forward/ReturnnForwardJob.1fohfY7LLczN/output/alignments.hdf")
