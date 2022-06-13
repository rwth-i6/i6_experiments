from ..network import ReturnnNetwork
from .abs_module import AbsModule


class ConvLocAwareness(AbsModule):
  """
  Attention convolution location awareness
  """

  def __init__(self, enc_key_dim, filter_size, num_channels, l2):
    super().__init__()
    self.enc_key_dim = enc_key_dim
    self.filter_size = filter_size
    self.num_channels = num_channels
    self.l2 = l2

  def create(self):
    out_net = ReturnnNetwork()

    pad_left = out_net.add_pad_layer(
      'feedback_pad_left', 'prev:att_weights', axes='s:0', padding=((self.filter_size - 1) // 2, 0),
      value=0)

    pad_right = out_net.add_pad_layer(
      'feedback_pad_right', pad_left, axes='s:0', padding=(0, (self.filter_size - 1) // 2), value=0)

    loc_att_conv = out_net.add_conv_layer(
      'loc_att_conv', pad_right, activation=None, with_bias=False, filter_size=(self.filter_size,),
      padding='valid', n_out=self.num_channels, l2=self.l2)

    self.name = out_net.add_linear_layer(
      'weight_feedback', loc_att_conv, activation=None, with_bias=False, n_out=self.enc_key_dim)

    return out_net.get_net()


class AdditiveLocAwareness(AbsModule):
  """
  Attention additive location awareness
  """

  def __init__(self, enc_key_dim, att_num_heads):
    super().__init__()
    self.enc_key_dim = enc_key_dim
    self.att_num_heads = att_num_heads

  def create(self):
    out_net = ReturnnNetwork()

    out_net.add_eval_layer('accum_att_weights', ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                       eval='source(0) + source(1) * source(2) * 0.5',
                       out_type={"dim": self.att_num_heads, "shape": (None, self.att_num_heads)})

    self.name = out_net.add_linear_layer(
      'weight_feedback', 'prev:accum_att_weights', n_out=self.enc_key_dim, with_bias=False)

    return out_net.get_net()


class AttentionMechanism(AbsModule):
  """
  Single-head or Multi-head attention mechanism
  """

  def __init__(self, enc_key_dim, att_num_heads, att_dropout, l2, loc_filter_size, loc_num_channels):
    super().__init__()
    self.enc_key_dim = enc_key_dim
    self.att_num_heads = att_num_heads

    self.att_dropout = att_dropout
    self.l2 = l2

    self.loc_filter_size = loc_filter_size
    self.loc_num_channels = loc_num_channels

  def create(self):
    out_net = ReturnnNetwork()

    out_net.add_linear_layer('s_transformed', 's', n_out=self.enc_key_dim, with_bias=False, l2=self.l2)  # project query

    if self.loc_num_channels is not None:
      assert self.loc_filter_size is not None
      weight_feedback = ConvLocAwareness(
        enc_key_dim=self.enc_key_dim, filter_size=self.loc_filter_size, num_channels=self.loc_num_channels, l2=self.l2)
    else:
      # additive
      weight_feedback = AdditiveLocAwareness(enc_key_dim=self.enc_key_dim, att_num_heads=self.att_num_heads)

    out_net.update(weight_feedback.create())  # add att weight feedback

    out_net.add_combine_layer(
      'energy_in', ['base:enc_ctx', weight_feedback.name, 's_transformed'], kind='add', n_out=self.enc_key_dim)

    # compute energies
    out_net.add_activation_layer('energy_tanh', 'energy_in', activation='tanh')
    energy = out_net.add_linear_layer('energy', 'energy_tanh', n_out=self.att_num_heads, with_bias=False, l2=self.l2)

    if self.att_dropout:
      att_weights0 = out_net.add_softmax_over_spatial_layer('att_weights0', energy)
      att_weights = out_net.add_dropout_layer(
        'att_weights', att_weights0, dropout=self.att_dropout, dropout_noise_shape={'*': None})
    else:
      att_weights = out_net.add_softmax_over_spatial_layer('att_weights', energy)

    att0 = out_net.add_generic_att_layer('att0', weights=att_weights, base='base:enc_value')
    self.name = out_net.add_merge_dims_layer('att', att0, axes='except_batch')

    return out_net.get_net()
