from typing import Optional, Dict, Any, Sequence, Tuple, List

from returnn.returnn.tensor import Tensor, Dim, single_step_dim
import returnn.returnn.frontend as rf
from returnn.returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample


class BaseModel(rf.Module):
  def __init__(
          self,
          in_dim: Dim,
          *,
          num_enc_layers: int = 12,
          target_dim: Dim,
          blank_idx: int,
          eos_idx: int,
          bos_idx: int,
          enc_model_dim: Dim = Dim(name="enc", dimension=512),
          enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
          enc_att_num_heads: int = 4,
          enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          att_dropout: float = 0.1,
          enc_dropout: float = 0.1,
          enc_att_dropout: float = 0.1,
          l2: float = 0.0001,
  ):
    super(BaseModel, self).__init__()

    self.in_dim = in_dim
    self.encoder = ConformerEncoder(
      in_dim,
      enc_model_dim,
      ff_dim=enc_ff_dim,
      input_layer=ConformerConvSubsample(
        in_dim,
        out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
        filter_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_sizes=[(1, 2)],
        strides=[(1, 1), (3, 1), (2, 1)],
      ),
      encoder_layer_opts=enc_conformer_layer_opts,
      num_layers=num_enc_layers,
      num_heads=enc_att_num_heads,
      dropout=enc_dropout,
      att_dropout=enc_att_dropout,
    )

    self.target_dim = target_dim
    self.blank_idx = blank_idx
    self.eos_idx = eos_idx
    self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

    self.enc_key_total_dim = enc_key_total_dim
    self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
    self.att_num_heads = att_num_heads
    self.att_dropout = att_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()

    self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
    self.enc_ctx_dropout = 0.2
    self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

    self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

    self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))

    self.s = rf.ZoneoutLSTM(
      self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
      Dim(name="lstm", dimension=1024),
      zoneout_factor_cell=0.15,
      zoneout_factor_output=0.05,
      use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
      # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
      # parts_order="ifco",
      parts_order="jifo",  # NativeLSTM (the code above converts it...)
      forget_bias=0.0,  # the code above already adds it during conversion
    )

    self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
    self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
    self.readout_in = rf.Linear(
      self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
      Dim(name="readout", dimension=1024),
    )
    self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

    for p in self.parameters():
      p.weight_decay = l2

  def encode(
    self,
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Dict[str, Tensor], Dim]:
    """encode, and extend the encoder output for things we need in the decoder"""
    # log mel filterbank features
    source, in_spatial_dim, in_dim_ = rf.stft(
      source, in_spatial_dim=in_spatial_dim, frame_step=160, frame_length=400, fft_length=512
    )
    source = rf.abs(source) ** 2.0
    source = rf.audio.mel_filterbank(source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000)
    source = rf.safe_log(source, eps=1e-10) / 2.3026
    # TODO specaug
    # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
    enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
    enc_ctx = self.enc_ctx(enc)
    inv_fertility = rf.sigmoid(self.inv_fertility(enc))
    return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

  @staticmethod
  def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
    """
    prepare the encoder output for the loop (full-sum or time-sync)
    """
    # We might improve or generalize the interface later...
    # https://github.com/rwth-i6/returnn_common/issues/202
    loop = rf.inner_loop()
    return {k: loop.unstack(v) for k, v in ext.items()}

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s": Tensor(
        "s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [self.att_num_heads * self.encoder.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }
