#!returnn.py

import sys
from typing import Optional, Tuple, Dict, Sequence
import os
from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim
from recipe.returnn_common import nn
from recipe.returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder

sys.path.insert(0, ...)  # make sure returnn_common can be imported...

demo_name, _ = os.path.splitext(__file__)
print("Hello, experiment: %s" % demo_name)

use_tensorflow = True

task = "train"
train = {"class": "Task12AXDataset", "num_seqs": 1000}
dev = {"class": "Task12AXDataset", "num_seqs": 100, "fixed_random_seed": 1}

time_dim = SpatialDim("time")
label_spatial_dim = SpatialDim("labels_spatial")
align_spatial_dim = SpatialDim("alignment_spatial")
feature_dim = FeatureDim("input", 40)
align_dim = FeatureDim("alignment", 4)
label_dim = FeatureDim("labels", 3)
emit_blank_dim = FeatureDim("emit_blank", 2)
extern_data = {
  "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
  "alignment": {"dim_tags": [batch_dim, align_spatial_dim], "sparse_dim": align_dim},
  "labels": {"dim_tags": [batch_dim, label_spatial_dim], "sparse_dim": label_dim},
  "emit_ground_truth": {"dim_tags": [batch_dim, align_spatial_dim], "sparse_dim": emit_blank_dim}
}


class SegmentalAttentionModel(nn.Module):
  def __init__(self):
    super().__init__()
    hidden_dim = nn.FeatureDim("hidden", 10)
    self.label_spatial_dim = nn.SpatialDim("labels")

    self.blank_idx = 1030

    self.encoder = BlstmCnnSpecAugEncoder(feature_dim, num_layers=6, l2=0.0001, dropout=0.3)
    self.label_sync_decoder = LabelSyncDecoder(enc_dim=self.encoder.out_dim, label_dim=label_dim)
    self.time_sync_decoder = TimeSyncDecoder(
      label_dim=label_dim, enc_dim=self.encoder.out_dim, win_size=4)

  def get_label_ground_truth(self, alignment_data) -> nn.MaskedComputation:
    is_label = alignment_data != self.blank_idx
    return nn.MaskedComputation(mask=is_label)

  def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim):
    enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim)

    return enc, enc_spatial_dim

  def decode_length_model(
    self,
    *,
    enc,
    enc_spatial_dim,
    prev_output,
    state,
    loop_step
  ):
    emit_blank_probs, state, segment_starts, segment_lens = self.time_sync_decoder(
      enc=enc,
      enc_spatial_dim=enc_spatial_dim,
      prev_output=prev_output,
      state=state,
      loop_step=loop_step
    )

    return emit_blank_probs, state, segment_starts, segment_lens

  def decode_label_model(
    self,
    *,
    enc: nn.Tensor,
    enc_spatial_dim: nn.Dim,
    label_spatial_dim: nn.Dim,
    state: Optional[nn.LayerState],
    prev_out_non_blank,
  ):
    if state is None:
      assert enc_spatial_dim != nn.single_step_dim
      batch_dims = enc.batch_dims_ordered(remove=(enc.feature_dim,))
      state = nn.LayerState(
        lm=self.label_sync_decoder.default_initial_state(batch_dims=batch_dims),
        att=nn.zeros(enc.shape_ordered)
      )

    label_prob, state = self.label_sync_decoder(
      enc=enc,
      enc_spatial_dim=enc_spatial_dim,
      segment_lens=None,
      segment_starts=0,
      prev_out_non_blank=prev_out_non_blank,
      state=state,
      label_spatial_dim=label_spatial_dim
    )

    return label_prob, state


class TimeSyncDecoder(nn.Module):
  def __init__(self, label_dim, enc_dim, win_size):
    super().__init__()

    self.blank_idx = 1030
    assert win_size % 2 == 0, "Window size needs to be divisible by 2"
    self.win_size_half = win_size // 2

    lstm_dim = nn.Dim(kind=nn.Dim.Types.Feature, description="lstm_dim", dimension=128)
    emit_dim = nn.Dim(kind=nn.Dim.Types.Feature, description="emit_out", dimension=1)

    self.emit_prob = nn.Linear(lstm_dim, emit_dim)
    self.embed = nn.Linear(label_dim, lstm_dim)
    self.lstm = nn.LSTM(enc_dim + lstm_dim, lstm_dim)

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
    return self.lstm.default_initial_state(batch_dims=batch_dims)

  def __call__(
    self,
    *,
    enc,
    enc_spatial_dim,
    prev_output,
    state,
    loop_step,
    **kwargs):

    # Neural part
    state_ = nn.LayerState()
    embed = self.embed(prev_output)
    lstm, state_.lstm = self.lstm(nn.concat_features(enc, embed), state=state.lstm, spatial_dim=enc_spatial_dim)
    emit_prob = self.emit_prob(lstm)
    blank_log_prob = nn.log_sigmoid(-emit_prob)
    emit_log_prob = nn.log_sigmoid(emit_prob)
    emit_blank_prob = nn.exp(nn.concat_features(blank_log_prob, emit_log_prob))

    # Segment calculation
    output_emit = prev_output != self.blank_idx
    state_.segment_starts0 = nn.where(output_emit, true_=loop_step, false_=state.segment_starts0)
    segment_lens0 = loop_step - state_.segment_starts0 + 1
    segment_starts1 = state_.segment_starts0 + segment_lens0 - self.win_size_half
    segment_ends0 = segment_starts1 + segment_lens0 + self.win_size_half
    seq_lens = nn.length(enc_spatial_dim)
    seq_end_too_far = segment_ends0 > seq_lens
    seq_start_too_far = segment_starts1 < 0
    segment_starts = nn.where(seq_start_too_far, true_=0, false_=segment_starts1)
    segment_ends = nn.where(seq_end_too_far, true_=seq_lens, false_=segment_ends0)
    segment_lens = segment_ends - segment_starts

    return emit_blank_prob, state_, segment_starts, segment_lens


class LabelSyncDecoder(nn.Module):
  def __init__(self, enc_dim, label_dim):
    super().__init__()

    embed_dim = nn.Dim(dimension=621)
    # vocab_size = nn.Dim(dimension=label_dim.dimension)
    lm_dim = nn.Dim(description="lm_dim", dimension=1024)
    self.key_dim = nn.Dim(dimension=1024)
    readout_dim = nn.Dim(dimension=1000)
    # enc_dim = nn.Dim(dimension=2048)

    self.embed = nn.Linear(in_dim=label_dim, out_dim=embed_dim, with_bias=False)
    self.lm = nn.LSTM(enc_dim + self.embed.out_dim, lm_dim)
    self.att_query = nn.Linear(lm_dim, lm_dim, with_bias=False)
    self.att_ctx = nn.Linear(enc_dim, lm_dim, with_bias=False)
    self.att_energy = nn.Linear(lm_dim, nn.FeatureDim("att_heads", 1))
    self.readout = nn.Linear(lm_dim + enc_dim, readout_dim)
    self.label_log_prob = nn.Linear(readout_dim / 2, label_dim)

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
    return self.lm.default_initial_state(batch_dims=batch_dims)

  def __call__(
    self,
    *,
    enc,
    enc_spatial_dim,
    segment_lens,
    segment_starts,
    prev_out_non_blank,
    state,
    label_spatial_dim
  ):
    att_head_dim = nn.Dim(dimension=1, description="att_heads")

    state_ = nn.LayerState()

    embed = self.embed(prev_out_non_blank)
    print(state.lm)
    lm, state_.lm = self.lm(nn.concat_features(state.att, embed), state=state.lm, spatial_dim=label_spatial_dim)
    print(state_.lm)

    att_query = self.att_query(lm)
    # dirty fix in nn.naming.py, line 1003
    att_values, slice_dim = nn.slice_nd(
      enc, size=segment_lens, start=segment_starts, axis=enc_spatial_dim,
      out_spatial_dim=nn.Dim(kind=nn.Dim.Types.Spatial, description="segments")
    )
    att_values_split = nn.split_dims(
      att_values, axis=att_values.feature_dim, dims=(att_head_dim, att_values.feature_dim))
    att_ctx = self.att_ctx(att_values)
    att_energy_add = nn.combine(att_ctx, att_query, kind="add")
    att_energy_tanh = nn.tanh(att_energy_add)
    att_energy = self.att_energy(att_energy_tanh)
    att_weights = nn.softmax(att_energy, axis=slice_dim, energy_factor=self.key_dim.dimension ** -0.5)
    state_.att = nn.dot(att_values_split, att_weights, reduce=slice_dim)
    state_.att, _ = nn.merge_dims(
      state_.att, axes=state_.att.batch_dims_ordered(remove=state_.att.batch_dim), out_dim=att_values.feature_dim
    )

    readout = self.readout(nn.concat_features(lm, state_.att))
    readout = nn.reduce_out(readout, mode="max", num_pieces=2)
    readout = nn.dropout(readout, 0.3, axis=readout.feature_dim)

    label_log_prob = self.label_log_prob(readout)
    label_log_prob = nn.log_softmax(label_log_prob, axis=label_log_prob.feature_dim)
    label_prob = nn.exp(label_log_prob)

    return label_prob, state_


def training():
  nn.reset_default_root_name_ctx()
  data = nn.Data(name="data", **extern_data["data"])
  data_spatial_dim = data.get_time_dim_tag()
  labels = nn.Data(name="labels", **extern_data["labels"])
  alignment = nn.Data(name="alignment", **extern_data["alignment"])
  emit_ground_truth = nn.Data(name="emit_ground_truth", **extern_data["emit_ground_truth"])
  data = nn.get_extern_data(data)
  labels = nn.get_extern_data(labels)
  alignment = nn.get_extern_data(alignment)
  emit_ground_truth = nn.get_extern_data(emit_ground_truth)

  model = SegmentalAttentionModel()
  enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
  prev_labels, prev_labels_spatial_dim = nn.prev_target_seq(
    labels, spatial_dim=label_spatial_dim, bos_idx=0, out_one_longer=False
  )
  prev_align, prev_align_spatial_dim = nn.prev_target_seq(
    alignment, spatial_dim=align_spatial_dim, bos_idx=0, out_one_longer=False
  )

  label_sync_loop = nn.Loop(axis=label_spatial_dim)
  label_sync_loop.state.label_model = nn.LayerState(
    lm=model.label_sync_decoder.default_initial_state(batch_dims=[labels.batch_dim]),
    att=nn.zeros(shape=[labels.batch_dim, enc.feature_dim])
  )

  with label_sync_loop:
    prev_label = label_sync_loop.unstack(prev_labels)
    label_probs, label_sync_loop.state.label_model = model.decode_label_model(
      enc=enc,
      enc_spatial_dim=enc_spatial_dim,
      label_spatial_dim=nn.single_step_dim,
      state=label_sync_loop.state.label_model,
      prev_out_non_blank=prev_label
    )

    loss = nn.cross_entropy(
      target=labels, estimated=label_probs, estimated_type="probs"
    )
    loss.mark_as_loss("ce_label_model")

    res = label_sync_loop.stack(label_probs)

  time_sync_loop = nn.Loop(axis=align_spatial_dim)
  time_sync_loop.state.length_model = nn.LayerState(
    lstm=model.time_sync_decoder.default_initial_state(batch_dims=[enc.batch_dim]),
    segment_starts0=nn.zeros(shape=[enc.batch_dim]))

  print(prev_align.shape)
  with time_sync_loop:
    prev_output = time_sync_loop.unstack(prev_align)
    emit_blank_probs, time_sync_loop.state.length_model, segment_starts, segment_lens = model.decode_length_model(
      enc=enc,
      enc_spatial_dim=enc_spatial_dim,
      prev_output=prev_output,
      state=time_sync_loop.state.length_model,
      loop_step=time_sync_loop.iter_idx
    )

    loss = nn.cross_entropy(
      target=emit_ground_truth, estimated=emit_blank_probs, estimated_type="probs"
    )
    loss.mark_as_loss("ce_length_model")

    res = label_sync_loop.unstack(emit_blank_probs)

  print(nn.get_returnn_config().get_net_dict_raw_dict(root_module=model))


# batching
batching = "random"
batch_size = 5000
max_seqs = 10
chunking = "200:200"

# training
optimizer = {"class": "adam"}
learning_rate = 0.01
num_epochs = 5

# log
log_verbosity = 3

if __name__ == "__main__":
  training()
