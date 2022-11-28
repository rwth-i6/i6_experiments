from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class RNNEncoder:
  """
  Represents RNN LSTM Attention-based Encoder
  """

  def __init__(self, input='data', enc_layers=6, bidirectional=True, residual_lstm=False, residual_proj_dim=None,
               specaug=True, with_conv=True, dropout=0.3, pool_sizes='3_2', lstm_dim=None, enc_key_dim=1024,
               enc_value_dim=2048, att_num_heads=1, target='bpe', l2=None, rec_weight_dropout=None, with_ctc=False,
               ctc_dropout=0., ctc_l2=0., ctc_opts=None, enc_proj_dim=None, ctc_loss_scale=None,
               conv_time_pooling=None):
    """
    :param str input: (layer) name of the network input
    :param int enc_layers: the number of encoder layers
    :param bool bidirectional: If set, bidirectional LSTMs are used
    :param bool specaug: If True, SpecAugment is used
    :param bool with_conv: if True, conv layers are applied initially
    :param float dropout: Dropout applied on the input of multiple layers
    :param str|int|List[int]|None pool_sizes: a list of pool sizes between LSTM layers
    :param int enc_key_dim: attention key dimension
    :param int enc_value_dim: attention value dimension
    :param int att_num_heads: number of attention heads
    :param str target: target data key name
    :param float|None l2: weight decay with l2 norm
    :param float|None rec_weight_dropout: dropout applied to the hidden-to-hidden LSTM weight matrices
    :param bool with_ctc: if set, CTC is used
    :param float ctc_dropout: dropout applied on input to ctc
    :param float ctc_l2: L2 applied to the weight matrix of CTC softmax
    :param dict[str] ctc_opts: options for CTC
    """

    self.input = input
    self.enc_layers = enc_layers

    if pool_sizes is not None:
      if isinstance(pool_sizes, str):
        pool_sizes = list(map(int, pool_sizes.split('_'))) + [1] * (enc_layers - 3)
      elif isinstance(pool_sizes, int):
        pool_sizes = [pool_sizes] * (self.enc_layers - 1)

      assert isinstance(pool_sizes, list), 'pool_sizes must be a list'
      assert all([isinstance(e, int) for e in pool_sizes]), 'pool_sizes must only contains integers'
      assert len(pool_sizes) < enc_layers

    self.pool_sizes = pool_sizes

    if conv_time_pooling is None:
      self.conv_time_pooling = [1, 1]
    else:
      self.conv_time_pooling = list(map(int, conv_time_pooling.split('_')))


    self.bidirectional = bidirectional

    self.residual_lstm = residual_lstm
    self.residual_proj_dim = residual_proj_dim

    self.specaug = specaug
    self.with_conv = with_conv
    self.dropout = dropout

    self.enc_key_dim = enc_key_dim
    self.enc_value_dim = enc_value_dim
    self.att_num_heads = att_num_heads
    self.enc_key_per_head_dim = enc_key_dim // att_num_heads
    self.enc_val_per_head_dim = enc_value_dim // att_num_heads
    self.lstm_dim = lstm_dim
    if lstm_dim is None:
      self.lstm_dim = enc_value_dim // 2

    self.target = target

    self.l2 = l2
    self.rec_weight_dropout = rec_weight_dropout

    self.with_ctc = with_ctc
    self.ctc_dropout = ctc_dropout
    self.ctc_l2 = ctc_l2
    self.ctc_loss_scale = ctc_loss_scale
    self.ctc_opts = ctc_opts
    if self.ctc_opts is None:
      self.ctc_opts = {}

    self.enc_proj_dim = enc_proj_dim

    self.network = ReturnnNetwork()

  def create_network(self):
    data = self.input
    if self.specaug:
      data = self.network.add_eval_layer(
        'source', data,
        eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")

    lstm_input = data
    if self.with_conv:
      lstm_input = self.network.add_conv_block(
        'conv_merged', data,
        hwpc_sizes=[((3, 3), (self.conv_time_pooling[0], 2), 32), ((3, 3), (self.conv_time_pooling[1], 2), 32)],
        l2=self.l2, activation=None)

    if self.residual_lstm:
      last_lstm_layer = self.network.add_residual_lstm_layers(
        lstm_input, self.enc_layers, self.lstm_dim, self.dropout, self.l2, self.rec_weight_dropout, self.pool_sizes,
        residual_proj_dim=self.residual_proj_dim, batch_norm=True)
    else:
      last_lstm_layer = self.network.add_lstm_layers(
        lstm_input, self.enc_layers, self.lstm_dim, self.dropout, self.l2, self.rec_weight_dropout, self.pool_sizes,
        self.bidirectional)

    encoder = self.network.add_copy_layer('encoder', last_lstm_layer)
    if self.enc_proj_dim:
      encoder = self.network.add_linear_layer(
        'encoder_proj', encoder, n_out=self.enc_proj_dim, l2=self.l2, dropout=self.dropout)

    if self.with_ctc:
      default_ctc_loss_opts = {"beam_width": 1, "ctc_opts": {"ignore_longer_outputs_than_inputs": True}}
      default_ctc_loss_opts.update(self.ctc_opts)
      if self.ctc_loss_scale:
        default_ctc_loss_opts['scale'] = self.ctc_loss_scale
      self.network.add_softmax_layer(
        'ctc', encoder, l2=self.ctc_l2, target=self.target, loss='ctc', dropout=self.ctc_dropout,
        loss_opts=default_ctc_loss_opts)

    return encoder
