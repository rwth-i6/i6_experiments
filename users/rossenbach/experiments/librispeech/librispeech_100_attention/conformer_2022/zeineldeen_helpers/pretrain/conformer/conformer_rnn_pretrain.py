def custom_construction_algo(idx, net_dict):
  # For debugging, use: python3 ./crnn/Pretrain.py config... Maybe set repetitions=1 below.

  import copy

  class ReturnnNetwork:
    """
    Represents a generic RETURNN network
    see docs: https://returnn.readthedocs.io/en/latest/
    """

    def __init__(self):
      self._net = {}

    def get_net(self):
      return self._net

    def add_copy_layer(self, name, source, **kwargs):
      self._net[name] = {'class': 'copy', 'from': source}
      self._net[name].update(kwargs)
      return name

    def add_eval_layer(self, name, source, eval, **kwargs):
      self._net[name] = {'class': 'eval', 'eval': eval, 'from': source}
      self._net[name].update(kwargs)
      return name

    def add_split_dim_layer(self, name, source, axis='F', dims=(-1, 1), **kwargs):
      self._net[name] = {'class': 'split_dims', 'axis': axis, 'dims': dims, 'from': source}
      self._net[name].update(kwargs)
      return name

    def add_conv_layer(self, name, source, filter_size, n_out, l2, padding='same', activation=None, with_bias=True,
                       forward_weights_init=None, **kwargs):
      d = {
        'class': 'conv', 'from': source, 'padding': padding, 'filter_size': filter_size, 'n_out': n_out,
        'activation': activation, 'with_bias': with_bias
      }
      if l2:
        d['L2'] = l2
      if forward_weights_init:
        d['forward_weights_init'] = forward_weights_init
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_linear_layer(self, name, source, n_out, activation=None, with_bias=True, dropout=0., l2=0.,
                         forward_weights_init=None, **kwargs):
      d = {
        'class': 'linear', 'activation': activation, 'with_bias': with_bias, 'from': source, 'n_out': n_out
      }
      if dropout:
        d['dropout'] = dropout
      if l2:
        d['L2'] = l2
      if forward_weights_init:
        d['forward_weights_init'] = forward_weights_init
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_pool_layer(self, name, source, pool_size, mode='max', **kwargs):
      self._net[name] = {'class': 'pool', 'from': source, 'pool_size': pool_size, 'mode': mode, 'trainable': False}
      self._net[name].update(kwargs)
      return name

    def add_merge_dims_layer(self, name, source, axes='static', **kwargs):
      self._net[name] = {'class': 'merge_dims', 'from': source, 'axes': axes}
      self._net[name].update(kwargs)
      return name

    def add_rec_layer(self, name, source, n_out, l2, rec_weight_dropout, direction=1, unit='nativelstm2', **kwargs):
      d = {'class': 'rec', 'unit': unit, 'n_out': n_out, 'direction': direction, 'from': source}
      if l2:
        d['L2'] = l2
      if rec_weight_dropout:
        if 'unit_opts' not in d:
          d['unit_opts'] = {}
        d['unit_opts'].update({'rec_weight_dropout': rec_weight_dropout})
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_choice_layer(self, name, source, target, beam_size=12, initial_output=0, **kwargs):
      self._net[name] = {'class': 'choice', 'target': target, 'beam_size': beam_size, 'from': source,
                         'initial_output': initial_output}
      self._net[name].update(kwargs)
      return name

    def add_compare_layer(self, name, source, value, kind='equal', **kwargs):
      self._net[name] = {'class': 'compare', 'kind': kind, 'from': source, 'value': value}
      self._net[name].update(kwargs)
      return name

    def add_combine_layer(self, name, source, kind, n_out, **kwargs):
      self._net[name] = {'class': 'combine', 'kind': kind, 'from': source, 'n_out': n_out}
      self._net[name].update(kwargs)
      return name

    def add_activation_layer(self, name, source, activation, **kwargs):
      self._net[name] = {'class': 'activation', 'activation': activation, 'from': source}
      self._net[name].update(kwargs)
      return name

    def add_softmax_over_spatial_layer(self, name, source, **kwargs):
      self._net[name] = {'class': 'softmax_over_spatial', 'from': source}
      self._net[name].update(kwargs)
      return name

    def add_generic_att_layer(self, name, weights, base, **kwargs):
      self._net[name] = {'class': 'generic_attention', 'weights': weights, 'base': base}
      self._net[name].update(kwargs)
      return name

    def add_rnn_cell_layer(self, name, source, n_out, unit='LSTMBlock', l2=0., **kwargs):
      d = {'class': 'rnn_cell', 'unit': unit, 'n_out': n_out, 'from': source}
      if l2:
        d['L2'] = l2
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_softmax_layer(self, name, source, l2=None, loss=None, target=None, dropout=0., loss_opts=None,
                          forward_weights_init=None, **kwargs):
      d = {'class': 'softmax', 'from': source}
      if dropout:
        d['dropout'] = dropout
      if target:
        d['target'] = target
      if loss:
        d['loss'] = loss
        if loss_opts:
          d['loss_opts'] = loss_opts
      if l2:
        d['L2'] = l2
      if forward_weights_init:
        d['forward_weights_init'] = forward_weights_init
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_dropout_layer(self, name, source, dropout, dropout_noise_shape=None, **kwargs):
      self._net[name] = {'class': 'dropout', 'from': source, 'dropout': dropout}
      if dropout_noise_shape:
        self._net[name]['dropout_noise_shape'] = dropout_noise_shape
      self._net[name].update(kwargs)
      return name

    def add_reduceout_layer(self, name, source, num_pieces=2, mode='max', **kwargs):
      self._net[name] = {'class': 'reduce_out', 'from': source, 'num_pieces': num_pieces, 'mode': mode}
      self._net[name].update(kwargs)
      return name

    def add_subnet_rec_layer(self, name, unit, target, source=None, **kwargs):
      if source is None:
        source = []
      self._net[name] = {
        'class': 'rec', 'from': source, 'unit': unit, 'target': target, 'max_seq_len': "max_len_from('base:encoder')"}
      self._net[name].update(kwargs)
      return name

    def add_decide_layer(self, name, source, target, loss='edit_distance', **kwargs):
      self._net[name] = {'class': 'decide', 'from': source, 'loss': loss, 'target': target}
      self._net[name].update(kwargs)
      return name

    def add_slice_layer(self, name, source, axis, **kwargs):
      self._net[name] = {'class': 'slice', 'from': source, 'axis': axis, **kwargs}
      return name

    def add_subnetwork(self, name, source, subnetwork_net, **kwargs):
      self._net[name] = {'class': 'subnetwork', 'from': source, 'subnetwork': subnetwork_net, **kwargs}
      return name

    def add_layer_norm_layer(self, name, source, **kwargs):
      self._net[name] = {'class': 'layer_norm', 'from': source, **kwargs}
      return name

    def add_batch_norm_layer(self, name, source, **kwargs):
      self._net[name] = {'class': 'batch_norm', 'from': source, **kwargs}
      return name

    def add_self_att_layer(self, name, source, n_out, num_heads, total_key_dim, att_dropout=0., key_shift=None,
                           forward_weights_init=None, **kwargs):
      d = {
        'class': 'self_attention', 'from': source, 'n_out': n_out, 'num_heads': num_heads,
        'total_key_dim': total_key_dim
      }
      if att_dropout:
        d['attention_dropout'] = att_dropout
      if key_shift:
        d['key_shift'] = key_shift
      if forward_weights_init:
        d['forward_weights_init'] = forward_weights_init
      d.update(kwargs)
      self._net[name] = d
      return name

    def add_pos_encoding_layer(self, name, source, add_to_input=True, **kwargs):
      self._net[name] = {'class': 'positional_encoding', 'from': source, 'add_to_input': add_to_input}
      self._net[name].update(kwargs)
      return name

    def add_relative_pos_encoding_layer(self, name, source, n_out, forward_weights_init=None, **kwargs):
      self._net[name] = {'class': 'relative_positional_encoding', 'from': source, 'n_out': n_out}
      if forward_weights_init:
        self._net[name]['forward_weights_init'] = forward_weights_init
      self._net[name].update(kwargs)
      return name

    def add_constant_layer(self, name, value, **kwargs):
      self._net[name] = {'class': 'constant', 'value': value}
      self._net[name].update(kwargs)
      return name

    def add_gating_layer(self, name, source, activation='identity', **kwargs):
      """
      out = activation(a) * gate_activation(b)  (gate_activation is sigmoid by default)
      In case of one source input, it will split by 2 over the feature dimension
      """
      self._net[name] = {'class': 'gating', 'from': source, 'activation': activation}
      self._net[name].update(kwargs)
      return name

    def add_pad_layer(self, name, source, axes, padding, **kwargs):
      self._net[name] = {'class': 'pad', 'from': source, 'axes': axes, 'padding': padding}
      self._net[name].update(**kwargs)
      return name

    def add_reduce_layer(self, name, source, mode, axes, keep_dims=False, **kwargs):
      self._net[name] = {'class': 'reduce', 'from': source, 'mode': mode, 'axes': axes, 'keep_dims': keep_dims}
      self._net[name].update(**kwargs)
      return name

    def add_variable_layer(self, name, shape, **kwargs):
      self._net[name] = {'class': 'variable', 'shape': shape}
      self._net[name].update(kwargs)
      return name

    def add_switch_layer(self, name, condition, true_from, false_from, **kwargs):
      self._net[name] = {'class': 'switch', 'condition': condition, 'true_from': true_from, 'false_from': false_from}
      self._net[name].update(kwargs)
      return name

    def add_conv_block(self, name, source, hwpc_sizes, l2, activation, dropout=0.0, init=None):
      src = self.add_split_dim_layer('source0', source)
      for idx, hwpc in enumerate(hwpc_sizes):
        filter_size, pool_size, n_out = hwpc
        src = self.add_conv_layer(
          'conv%i' % idx, src, filter_size=filter_size, n_out=n_out, l2=l2, activation=activation,
          forward_weights_init=init)
        if pool_size:
          src = self.add_pool_layer('conv%ip' % idx, src, pool_size=pool_size, padding='same')
      if dropout:
        src = self.add_dropout_layer('conv_dropout', src, dropout=dropout)
      return self.add_merge_dims_layer(name, src)

    def add_lstm_layers(self, input, num_layers, lstm_dim, dropout, l2, rec_weight_dropout, pool_sizes, bidirectional):
      src = input
      pool_idx = 0
      for layer in range(num_layers):
        lstm_fw_name = self.add_rec_layer(
          name='lstm%i_fw' % layer, source=src, n_out=lstm_dim, direction=1, dropout=dropout, l2=l2,
          rec_weight_dropout=rec_weight_dropout)
        if bidirectional:
          lstm_bw_name = self.add_rec_layer(
            name='lstm%i_bw' % layer, source=src, n_out=lstm_dim, direction=-1, dropout=dropout, l2=l2,
            rec_weight_dropout=rec_weight_dropout)
          src = [lstm_fw_name, lstm_bw_name]
        else:
          src = lstm_fw_name
        if pool_sizes and pool_idx < len(pool_sizes):
          lstm_pool_name = 'lstm%i_pool' % layer
          src = self.add_pool_layer(
            name=lstm_pool_name, source=src, pool_size=(pool_sizes[pool_idx],), padding='same')
          pool_idx += 1
      return src

    def add_residual_lstm_layers(self, input, num_layers, lstm_dim, dropout, l2, rec_weight_dropout, pool_sizes,
                                 residual_proj_dim=None, batch_norm=True):
      src = input
      pool_idx = 0

      for layer in range(num_layers):

        # Forward LSTM
        lstm_fw_name = self.add_rec_layer(
          name='lstm%i_fw' % layer, source=src, n_out=lstm_dim, direction=1, l2=l2, dropout=dropout,
          rec_weight_dropout=rec_weight_dropout)

        # Backward LSTM
        lstm_bw_name = self.add_rec_layer(
          name='lstm%i_bw' % layer, source=src, n_out=lstm_dim, direction=-1, l2=l2, dropout=dropout,
          rec_weight_dropout=rec_weight_dropout)

        # Concat LSTM outputs
        new_src = [lstm_fw_name, lstm_bw_name]

        # If given, project both LSTM output and LSTM input
        residual_lstm_out = new_src
        residual_lstm_in = src
        if residual_proj_dim:
          residual_lstm_out = self.add_linear_layer('lstm%i_lin_proj' % layer, new_src, n_out=residual_proj_dim, l2=l2)
          residual_lstm_in = self.add_linear_layer('lstm%i_inp_lin_proj' % layer, src, n_out=residual_proj_dim, l2=l2)

        # residual connection
        lstm_combine = self.add_combine_layer(
          'lstm%i_combine' % layer, [residual_lstm_in, residual_lstm_out], kind='add', n_out=residual_proj_dim)

        # apply batch norm if enabled
        if batch_norm:
          lstm_combine = self.add_batch_norm_layer(lstm_combine + '_bn', lstm_combine)

        if pool_sizes and pool_idx < len(pool_sizes):
          lstm_pool_name = 'lstm%i_pool' % layer
          src = self.add_pool_layer(
            name=lstm_pool_name, source=lstm_combine, pool_size=(pool_sizes[pool_idx],), padding='same')
          pool_idx += 1
        else:
          src = lstm_combine

      return src

    def add_dot_layer(self, name, source, **kwargs):
      self._net[name] = {'class': 'dot', 'from': source}
      self._net[name].update(kwargs)
      return name

    def __setitem__(self, key, value):
      self._net[key] = value

    def __getitem__(self, item):
      return self._net[item]

    def update(self, d: dict):
      self._net.update(d)

    def __str__(self):
      """
      Only for debugging
      """
      res = 'network = {\n'
      for k, v in self._net.items():
        res += '%s: %r\n' % (k, v)
      return res + '}'

  class ConformerEncoder:
    """
    Represents Conformer Encoder Architecture

    * Conformer: Convolution-augmented Transformer for Speech Recognition
    * Ref: https://arxiv.org/abs/2005.08100
    """

    def __init__(self, input='data', input_layer='conv', num_blocks=16, conv_kernel_size=32, specaug=True,
                 pos_enc='rel',
                 activation='swish', block_final_norm=True, ff_dim=512, ff_bias=True,
                 dropout=0.1, att_dropout=0.1, enc_key_dim=256, att_num_heads=4, target='bpe', l2=0.0, lstm_dropout=0.1,
                 rec_weight_dropout=0., with_ctc=False, native_ctc=False, ctc_dropout=0., ctc_l2=0., ctc_opts=None,
                 subsample=None, start_conv_init=None, conv_module_init=None, mhsa_init=None, mhsa_out_init=None,
                 ff_init=None, rel_pos_clipping=16, dropout_in=0.1):
      """
      :param str input: input layer name
      :param str input_layer: type of input layer which does subsampling
      :param int num_blocks: number of Conformer blocks
      :param int conv_kernel_size: kernel size for conv layers in Convolution module
      :param bool|None specaug: If true, then SpecAug is appliedi wi
      :param str|None activation: activation used to sandwich modules
      :param bool block_final_norm: if True, apply layer norm at the end of each conformer block
      :param bool final_norm: if True, apply layer norm to the output of the encoder
      :param int|None ff_dim: dimension of the first linear layer in FF module
      :param str|None ff_init: FF layers initialization
      :param bool|None ff_bias: If true, then bias is used for the FF layers
      :param float embed_dropout: dropout applied to the source embedding
      :param float dropout: general dropout
      :param float att_dropout: dropout applied to attention weights
      :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
      :param int att_num_heads: the number of attention heads
      :param str target: target labels key name
      :param float l2: add L2 regularization for trainable weights parameters
      :param float lstm_dropout: dropout applied to the input of the LSTMs in case they are used
      :param float rec_weight_dropout: dropout applied to the hidden-to-hidden weight matrices of the LSTM in case used
      :param bool with_ctc: if true, CTC loss is used
      :param bool native_ctc: if true, use returnn native ctc implementation instead of TF implementation
      :param float ctc_dropout: dropout applied on input to ctc
      :param float ctc_l2: L2 applied to the weight matrix of CTC softmax
      :param dict[str] ctc_opts: options for CTC
      """

      self.input = input
      self.input_layer = input_layer

      self.num_blocks = num_blocks
      self.conv_kernel_size = conv_kernel_size

      self.pos_enc = pos_enc
      self.rel_pos_clipping = rel_pos_clipping

      self.ff_bias = ff_bias

      self.specaug = specaug

      self.activation = activation

      self.block_final_norm = block_final_norm

      self.dropout = dropout
      self.att_dropout = att_dropout
      self.lstm_dropout = lstm_dropout

      self.dropout_in = dropout_in

      # key and value dimensions are the same
      self.enc_key_dim = enc_key_dim
      self.enc_value_dim = enc_key_dim
      self.att_num_heads = att_num_heads
      self.enc_key_per_head_dim = enc_key_dim // att_num_heads
      self.enc_val_per_head_dim = enc_key_dim // att_num_heads

      self.ff_dim = ff_dim
      if self.ff_dim is None:
        self.ff_dim = 2 * self.enc_key_dim

      self.target = target

      self.l2 = l2
      self.rec_weight_dropout = rec_weight_dropout

      self.with_ctc = with_ctc
      self.native_ctc = native_ctc
      self.ctc_dropout = ctc_dropout
      self.ctc_l2 = ctc_l2
      self.ctc_opts = ctc_opts
      if not self.ctc_opts:
        self.ctc_opts = {}

      self.start_conv_init = start_conv_init
      self.conv_module_init = conv_module_init
      self.mhsa_init = mhsa_init
      self.mhsa_out_init = mhsa_out_init
      self.ff_init = ff_init

      # add maxpooling layers
      self.subsample = subsample
      self.subsample_list = [1] * num_blocks
      if subsample:
        for idx, s in enumerate(map(int, subsample.split('_')[:num_blocks])):
          self.subsample_list[idx] = s

      self.network = ReturnnNetwork()

    def _create_ff_module(self, prefix_name, i, source):
      """
      Add Feed Forward Module:
        LN -> FFN -> Swish -> Dropout -> FFN -> Dropout

      :param str prefix_name: some prefix name
      :param int i: FF module index
      :param str source: name of source layer
      :return: last layer name of this module
      :rtype: str
      """
      prefix_name = prefix_name + '_ffmod_{}'.format(i)

      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

      ff1 = self.network.add_linear_layer(
        '{}_ff1'.format(prefix_name), ln, n_out=self.ff_dim, l2=self.l2, forward_weights_init=self.ff_init,
        with_bias=self.ff_bias)

      swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), ff1, activation=self.activation)

      drop1 = self.network.add_dropout_layer('{}_drop1'.format(prefix_name), swish_act, dropout=self.dropout)

      ff2 = self.network.add_linear_layer(
        '{}_ff2'.format(prefix_name), drop1, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
        with_bias=self.ff_bias)

      drop2 = self.network.add_dropout_layer('{}_drop2'.format(prefix_name), ff2, dropout=self.dropout)

      half_step_ff = self.network.add_eval_layer('{}_half_step'.format(prefix_name), drop2, eval='0.5 * source(0)')

      ff_module_res = self.network.add_combine_layer(
        '{}_res'.format(prefix_name), kind='add', source=[half_step_ff, source], n_out=self.enc_key_dim)

      return ff_module_res

    def _create_mhsa_module(self, prefix_name, source):
      """
      Add Multi-Headed Selft-Attention Module:
        LN + MHSA + Dropout

      :param str prefix: some prefix name
      :param str source: name of source layer
      :return: last layer name of this module
      :rtype: str
      """
      prefix_name = '{}_self_att'.format(prefix_name)
      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)
      ln_rel_pos_enc = None

      if self.pos_enc == 'rel':
        ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
          '{}_ln_rel_pos_enc'.format(prefix_name), ln, n_out=self.enc_key_per_head_dim,
          forward_weights_init=self.ff_init,
          clipping=self.rel_pos_clipping)

      mhsa = self.network.add_self_att_layer(
        '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=self.att_num_heads,
        total_key_dim=self.enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.mhsa_init,
        key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)

      mhsa_linear = self.network.add_linear_layer(
        '{}_linear'.format(prefix_name), mhsa, n_out=self.enc_key_dim, l2=self.l2,
        forward_weights_init=self.mhsa_out_init,
        with_bias=False)

      drop = self.network.add_dropout_layer('{}_dropout'.format(prefix_name), mhsa_linear, dropout=self.dropout)

      mhsa_res = self.network.add_combine_layer(
        '{}_res'.format(prefix_name), kind='add', source=[drop, source], n_out=self.enc_value_dim)
      return mhsa_res

    def _create_convolution_module(self, prefix_name, source):
      """
      Add Convolution Module:
        LN + point-wise-conv + GLU + depth-wise-conv + BN + Swish + point-wise-conv + Dropout

      :param str prefix_name: some prefix name
      :param int i: conformer module index
      :param str source: name of source layer
      :return: last layer name of this module
      :rtype: str
      """
      prefix_name = '{}_conv_mod'.format(prefix_name)

      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

      pointwise_conv1 = self.network.add_linear_layer(
        '{}_pointwise_conv1'.format(prefix_name), ln, n_out=2 * self.enc_key_dim, activation=None, l2=self.l2,
        with_bias=self.ff_bias, forward_weights_init=self.conv_module_init)

      glu_act = self.network.add_gating_layer('{}_glu'.format(prefix_name), pointwise_conv1)

      depthwise_conv = self.network.add_conv_layer(
        '{}_depthwise_conv2'.format(prefix_name), glu_act, n_out=self.enc_key_dim,
        filter_size=(self.conv_kernel_size,), groups=self.enc_key_dim, l2=self.l2,
        forward_weights_init=self.conv_module_init)

      bn = self.network.add_batch_norm_layer('{}_bn'.format(prefix_name), depthwise_conv)

      swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), bn, activation='swish')

      pointwise_conv2 = self.network.add_linear_layer(
        '{}_pointwise_conv2'.format(prefix_name), swish_act, n_out=self.enc_key_dim, activation=None, l2=self.l2,
        with_bias=self.ff_bias, forward_weights_init=self.conv_module_init)

      drop = self.network.add_dropout_layer('{}_drop'.format(prefix_name), pointwise_conv2, dropout=self.dropout)

      res = self.network.add_combine_layer(
        '{}_res'.format(prefix_name), kind='add', source=[drop, source], n_out=self.enc_key_dim)
      return res

    def _create_conformer_block(self, i, source):
      """
      Add the whole Conformer block:
        x1 = x0 + 1/2 * FFN(x0)             (FFN module 1)
        x2 = x1 + MHSA(x1)                  (MHSA)
        x3 = x2 + Conv(x2)                  (Conv module)
        x4 = LayerNorm(x3 + 1/2 * FFN(x3))  (FFN module 2)

      :param int i:
      :param str source: name of source layer
      :return: last layer name of this module
      :rtype: str
      """
      prefix_name = 'conformer_block_%02i' % i
      ff_module1 = self._create_ff_module(prefix_name, 1, source)
      mhsa = self._create_mhsa_module(prefix_name, ff_module1)
      conv_module = self._create_convolution_module(prefix_name, mhsa)
      ff_module2 = self._create_ff_module(prefix_name, 2, conv_module)
      res = ff_module2
      if self.block_final_norm:
        res = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), res)
      if self.subsample:
        assert 0 <= i - 1 < len(self.subsample)
        subsample_factor = self.subsample_list[i - 1]
        res = self.network.add_pool_layer(res + '_pool{}'.format(i), res, pool_size=(subsample_factor,))
      res = self.network.add_copy_layer(prefix_name, res)
      return res

    def create_network(self):
      """
      ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
      """
      data = self.input
      if self.specaug:
        data = self.network.add_eval_layer(
          'source', data,
          eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")

      subsampled_input = None
      if self.input_layer is None:
        subsampled_input = data
      elif 'lstm' in self.input_layer:
        sample_factor = int(self.input_layer.split('-')[1])
        pool_sizes = None
        if sample_factor == 4:
          pool_sizes = [2, 2]
        elif sample_factor == 6:
          pool_sizes = [3, 2]
        # add 2 LSTM layers with max pooling to subsample and encode positional information
        subsampled_input = self.network.add_lstm_layers(
          data, num_layers=2, lstm_dim=self.enc_key_dim, dropout=self.lstm_dropout, bidirectional=True,
          rec_weight_dropout=self.rec_weight_dropout, l2=self.l2, pool_sizes=pool_sizes)
      elif self.input_layer == 'conv':
        # subsample by 4
        subsampled_input = self.network.add_conv_block(
          'conv_merged', data, hwpc_sizes=[((3, 3), (2, 2), self.enc_key_dim), ((3, 3), (2, 2), self.enc_key_dim)],
          l2=self.l2, activation='relu')
      elif self.input_layer == 'vgg':
        subsampled_input = self.network.add_conv_block(
          'vgg_conv_merged', data, hwpc_sizes=[((3, 3), (2, 2), 32), ((3, 3), (2, 2), 64)], l2=self.l2,
          activation='relu')
      elif self.input_layer == 'neural_sp_conv':
        subsampled_input = self.network.add_conv_block(
          'conv_merged', data, hwpc_sizes=[((3, 3), (1, 1), 32), ((3, 3), (2, 2), 32)], l2=self.l2, activation='relu',
          init=self.start_conv_init)

      assert subsampled_input is not None

      source_linear = self.network.add_linear_layer(
        'source_linear', subsampled_input, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
        with_bias=False)

      # add positional encoding
      if self.pos_enc == 'abs':
        source_linear = self.network.add_pos_encoding_layer('{}_abs_pos_enc'.format(subsampled_input), source_linear)

      if self.dropout_in:
        source_linear = self.network.add_dropout_layer('source_dropout', source_linear, dropout=self.dropout_in)

      conformer_block_src = source_linear
      for i in range(1, self.num_blocks + 1):
        conformer_block_src = self._create_conformer_block(i, conformer_block_src)

      encoder = self.network.add_copy_layer('encoder', conformer_block_src)

      if self.with_ctc:
        default_ctc_loss_opts = {'beam_width': 1}
        if self.native_ctc:
          default_ctc_loss_opts['use_native'] = True
        else:
          self.ctc_opts.update({"ignore_longer_outputs_than_inputs": True})  # always enable
        if self.ctc_opts:
          default_ctc_loss_opts['ctc_opts'] = self.ctc_opts
        self.network.add_softmax_layer(
          'ctc', encoder, l2=self.ctc_l2, target=self.target, loss='ctc', dropout=self.ctc_dropout,
          loss_opts=default_ctc_loss_opts)
      return encoder

  class RNNDecoder:
    """
    Represents RNN LSTM Attention-based decoder

    Related:
      * Single headed attention based sequence-to-sequence model for state-of-the-art results on Switchboard
        ref: https://arxiv.org/abs/2001.07263
    """

    def __init__(self, base_model, source=None, dropout=0.3, label_smoothing=0.1, target='bpe',
                 length_norm=True, beam_size=12, embed_dim=621, embed_dropout=0., dec_lstm_num_units=1000,
                 dec_output_num_units=1000, l2=None, att_dropout=None, rec_weight_dropout=None, dec_zoneout=False,
                 ext_lm_opts=None, prior_lm_opts=None, local_fusion_opts=None, ff_init=None, add_lstm_lm=False,
                 lstm_lm_dim=1000, loc_conv_att_filter_size=None, loc_conv_att_num_channels=None,
                 density_ratio_opts=None, mwer=False, reduceout=True, att_num_heads=1):
      """
      :param base_model: base/encoder model instance
      :param str source: input to decoder subnetwork
      :param float dropout: Dropout applied to the softmax input
      :param float label_smoothing: label smoothing value applied to softmax
      :param List[int]|None pool_sizes: a list of pool sizes between LSTM layers
      :param int enc_key_dim: attention key dimension
      :param int att_num_heads: number of attention heads
      :param str target: target data key name
      :param int beam_size: value of the beam size
      :param int embed_dim: target embedding dimension
      :param float|None embed_dropout: dropout to be applied on the target embedding
      :param int dec_lstm_num_units: the number of hidden units for the decoder LSTM
      :param int dec_output_num_units: the number of hidden dimensions for the last layer before softmax
      :param float|None l2: weight decay with l2 norm
      :param float|None att_dropout: dropout applied to attention weights
      :param float|None rec_weight_dropout: dropout applied to weight paramters
      :param bool dec_zoneout: if set, zoneout LSTM cell is used in the decoder instead of nativelstm2
      :param dict[str]|None ext_lm_opts: external LM opts such as subnetwork, lm_model, scale, etc
      :param float|None prior_lm_scale: prior LM scale
      :param dict[str]|None local_fusion_opts: dict containing LM subnetwork, AM scale, and LM scale
        paper: https://arxiv.org/abs/2005.10049
      :param str|None ff_init: feed-forward weights initialization
      :param bool add_lstm_lm: add separate LSTM layer that acts as LM-like model
        same as here: https://arxiv.org/abs/2001.07263
      """

      self.base_model = base_model

      self.source = source
      self.dropout = dropout
      self.label_smoothing = label_smoothing

      self.enc_key_dim = base_model.enc_key_dim
      self.enc_value_dim = base_model.enc_value_dim
      self.att_num_heads = att_num_heads

      self.target = target
      self.length_norm = length_norm
      self.beam_size = beam_size

      self.embed_dim = embed_dim
      self.embed_dropout = embed_dropout
      self.dec_lstm_num_units = dec_lstm_num_units
      self.dec_output_num_units = dec_output_num_units

      self.ff_init = ff_init

      self.decision_layer_name = None  # this is set in the end-point config

      self.l2 = l2
      self.att_dropout = att_dropout
      self.rec_weight_dropout = rec_weight_dropout
      self.dec_zoneout = dec_zoneout

      self.ext_lm_opts = ext_lm_opts
      self.prior_lm_opts = prior_lm_opts

      self.local_fusion_opts = local_fusion_opts

      self.density_ratio_opts = density_ratio_opts

      self.add_lstm_lm = add_lstm_lm
      self.lstm_lm_dim = lstm_lm_dim

      self.loc_conv_att_filter_size = loc_conv_att_filter_size
      self.loc_conv_att_num_channels = loc_conv_att_num_channels

      self.mwer = mwer

      self.reduceout = reduceout

      self.network = ReturnnNetwork()

    def _create_prior_net(self, subnet_unit: ReturnnNetwork):
      prior_type = self.prior_lm_opts.get('type', 'zero')

      if prior_type == 'zero':  # set att context vector to zero
        prior_att_input = subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
      elif prior_type == 'avg':  # during search per utterance
        self.base_model.network.add_reduce_layer('encoder_mean', 'encoder', mode='mean', axes=['t'])  # [B, enc-dim]
        prior_att_input = 'base:encoder_mean'
      elif prior_type == 'train_avg_ctx':  # average all context vectors over training data
        prior_att_input = subnet_unit.add_constant_layer(
          'train_avg_ctx', value=self.prior_lm_opts['data'], with_batch_dim=True, dtype='float32')
      elif prior_type == 'train_avg_enc':  # average all encoder states over training data
        prior_att_input = subnet_unit.add_constant_layer(
          'train_avg_enc', value=self.prior_lm_opts['data'], with_batch_dim=True, dtype='float32')
      elif prior_type == 'mini_lstm':  # train a mini LM-like LSTM and use that as prior
        # example: lstmdim_100-l2_5e-05-recwd_0.0
        n_out = 50
        l2 = 0.0
        recwd = 0.0
        if self.prior_lm_opts.get('prefix_name', None):
          segs = self.prior_lm_opts['prefix_name'].split('-')
          for arg in segs:
            name, val = arg.split('_', 1)
            if name == 'lstmdim':
              n_out = int(val)
            elif name == 'l2':
              l2 = float(val)
            elif name == 'recwd':
              recwd = float(val)
        subnet_unit.add_rec_layer('mini_att_lstm', 'prev:target_embed', n_out=n_out, l2=l2, rec_weight_dropout=recwd)
        prior_att_input = subnet_unit.add_linear_layer('mini_att', 'mini_att_lstm', activation=None, n_out=2048,
                                                       l2=0.0001)
      elif prior_type == 'trained_vec':
        prior_att_input = subnet_unit.add_variable_layer('trained_vec_att_var', shape=[2048], L2=0.0001)
      elif prior_type == 'avg_zero':
        self.base_model.network.add_reduce_layer('encoder_mean', 'encoder', mode='mean', axes=['t'])  # [B, enc-dim]
        subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
        return
      elif prior_type == 'density_ratio':
        assert 'lm_subnet' in self.prior_lm_opts and 'lm_model' in self.prior_lm_opts
        return self._add_density_ratio(
          subnet_unit, lm_subnet=self.prior_lm_opts['lm_subnet'], lm_model=self.prior_lm_opts['lm_model'])
      else:
        raise ValueError('{} prior type is not supported'.format(prior_type))

      # for the first frame in decoding, don't use average but zero always
      keep_zero_frame = False if prior_type == 'zero' else self.prior_lm_opts.get('keep_zero_frame', True)
      prev_att = None
      if keep_zero_frame:
        is_first_frame = subnet_unit.add_compare_layer('is_first_frame', source=':i', kind='equal', value=0)
        zero_att = subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
        prev_att = subnet_unit.add_switch_layer(
          'prev_att', condition=is_first_frame, true_from=zero_att, false_from=prior_att_input)

      key_names = ['s', 'readout_in', 'readout', 'output_prob']
      for key_name in key_names:
        d = copy.deepcopy(subnet_unit[key_name])
        # update attention input
        new_sources = []
        from_list = d['from']
        if isinstance(from_list, str):
          from_list = [from_list]
        assert isinstance(from_list, list)
        for src in from_list:
          if 'att' in src:
            if src.split(':')[0] == 'prev':
              if keep_zero_frame:
                assert prev_att not in new_sources
                new_sources += [prev_att]  # switched based on decoder index
              else:
                new_sources += [prior_att_input]
            else:
              new_sources += [prior_att_input]
          elif src in key_names:
            new_sources += [('prev:' if 'prev' in src else '') + 'prior_{}'.format(src.split(':')[-1])]
          else:
            new_sources += [src]
        d['from'] = new_sources
        subnet_unit['prior_{}'.format(key_name)] = d
      return 'prior_output_prob'

    def _add_external_LM(self, subnet_unit: ReturnnNetwork, am_output_prob, prior_output_prob=None):
      ext_lm_subnet = self.ext_lm_opts['lm_subnet']
      ext_lm_scale = self.ext_lm_opts['lm_scale']

      assert isinstance(ext_lm_subnet, dict)
      is_recurrent = self.ext_lm_opts.get('is_recurrent', False)
      if is_recurrent:
        lm_output_prob = self.ext_lm_opts['lm_output_prob_name']
        ext_lm_subnet[lm_output_prob]['target'] = self.target
        ext_lm_subnet[lm_output_prob]['loss'] = None  # TODO: is this needed?
        subnet_unit.update(ext_lm_subnet)  # just append
      else:
        ext_lm_model = self.ext_lm_opts['lm_model']
        subnet_unit.add_subnetwork(
          'lm_output', 'prev:output', subnetwork_net=ext_lm_subnet, load_on_init=ext_lm_model)
        lm_output_prob = subnet_unit.add_activation_layer(
          'lm_output_prob', 'lm_output', activation='softmax', target=self.target)

      fusion_str = 'safe_log(source(0)) + {} * safe_log(source(1))'.format(ext_lm_scale)
      fusion_source = [am_output_prob, lm_output_prob]
      if prior_output_prob:
        fusion_source += [prior_output_prob]
        fusion_str += ' - {} * safe_log(source(2))'.format(self.prior_lm_opts['scale'])

      subnet_unit.add_eval_layer('combo_output_prob', source=fusion_source, eval=fusion_str)
      subnet_unit.add_choice_layer(
        'output', 'combo_output_prob', target=self.target, beam_size=self.beam_size, initial_output=0,
        input_type='log_prob')

    def _add_density_ratio(self, subnet_unit: ReturnnNetwork, lm_subnet, lm_model):
      subnet_unit.add_subnetwork(
        'density_ratio_output', 'prev:output', subnetwork_net=lm_subnet, load_on_init=lm_model)
      lm_output_prob = subnet_unit.add_activation_layer(
        'density_ratio_output_prob', 'density_ratio_output', activation='softmax', target=self.target)
      return lm_output_prob

    def _add_local_fusion(self, subnet: ReturnnNetwork, am_output_prob):
      lm_subnet = self.local_fusion_opts['lm_subnet']
      lm_model = self.local_fusion_opts['lm_model']
      vocab_size = self.local_fusion_opts['vocab_size']
      prefix_name = self.local_fusion_opts.get('prefix', 'local_fusion')
      with_label_smoothing = self.local_fusion_opts.get('with_label_smoothing', False)

      # make sure all layers in LM subnet are not trainable
      def make_non_trainable(d):
        for v in d.values():  # layers
          assert isinstance(v, dict)
          v.update({'trainable': False})

      # Add LM subnetwork.
      lm_subnet_copy = copy.deepcopy(lm_subnet)
      make_non_trainable(lm_subnet_copy)
      lm_subnet_name = '{}_lm_output'.format(prefix_name)
      subnet.add_subnetwork(
        lm_subnet_name, ['prev:output'], subnetwork_net=lm_subnet_copy, load_on_init=lm_model, trainable=False,
        n_out=vocab_size)
      lm_output_prob = subnet.add_activation_layer(
        '{}_lm_output_prob'.format(prefix_name), lm_subnet_name, activation='softmax',
        target=self.target)  # not in log-space

      # define new loss criteria
      combo_output_log_prob = subnet.add_eval_layer(
        'combo_output_log_prob', [am_output_prob, lm_output_prob],
        eval="self.network.get_config().typed_value('fusion_eval0_norm')(safe_log(source(0)), safe_log(source(1)))")

      # local fusion criteria. Eq. (8) in the paper
      if with_label_smoothing:
        subnet.add_eval_layer(
          'combo_output_prob', combo_output_log_prob, eval="tf.exp(source(0))", target=self.target, loss='ce',
          loss_opts={'label_smoothing': self.label_smoothing})
      else:
        subnet.add_eval_layer(
          'combo_output_prob', combo_output_log_prob, eval="tf.exp(source(0))", target=self.target, loss='ce')

      subnet.add_choice_layer(
        'output', combo_output_log_prob, target=self.target, beam_size=self.beam_size, initial_output=0,
        input_type='log_prob')

    def add_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):

      # target embedding
      if self.embed_dropout:
        subnet_unit.add_linear_layer('target_embed0', 'output', n_out=self.embed_dim, initial_output=0, with_bias=False)
        subnet_unit.add_dropout_layer(
          'target_embed', 'target_embed0', dropout=self.embed_dropout, dropout_noise_shape={'*': None})
      else:
        subnet_unit.add_linear_layer('target_embed', 'output', n_out=self.embed_dim, initial_output=0, with_bias=False)

      subnet_unit.add_compare_layer('end', source='output', value=0)  # sentence end token

      # ------ attention location-awareness ------ #

      # conv-based
      if self.loc_conv_att_filter_size:
        assert self.loc_conv_att_num_channels
        pad_left = subnet_unit.add_pad_layer(
          'feedback_pad_left', 'prev:att_weights', axes='s:0', padding=((self.loc_conv_att_filter_size - 1) // 2, 0),
          value=0)
        pad_right = subnet_unit.add_pad_layer(
          'feedback_pad_right', pad_left, axes='s:0', padding=(0, (self.loc_conv_att_filter_size - 1) // 2), value=0, )
        loc_att_conv = subnet_unit.add_conv_layer(
          'loc_att_conv', pad_right, activation=None, with_bias=False, filter_size=(self.loc_conv_att_filter_size,),
          padding='valid', n_out=self.loc_conv_att_num_channels, l2=self.l2)
        subnet_unit.add_linear_layer(
          'weight_feedback', loc_att_conv, activation=None, with_bias=False, n_out=self.enc_key_dim)
      else:
        # additive
        subnet_unit.add_eval_layer('accum_att_weights', ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
                                   eval='source(0) + source(1) * source(2) * 0.5',
                                   out_type={"dim": self.att_num_heads, "shape": (None, self.att_num_heads)})
        subnet_unit.add_linear_layer('weight_feedback', 'prev:accum_att_weights', n_out=self.enc_key_dim,
                                     with_bias=False)

      subnet_unit.add_linear_layer('s_transformed', 's', n_out=self.enc_key_dim, with_bias=False)
      subnet_unit.add_combine_layer(
        'energy_in', ['base:enc_ctx', 'weight_feedback', 's_transformed'], kind='add', n_out=self.enc_key_dim)
      subnet_unit.add_activation_layer('energy_tanh', 'energy_in', activation='tanh')
      subnet_unit.add_linear_layer('energy', 'energy_tanh', n_out=self.att_num_heads, with_bias=False)
      if self.att_dropout:
        subnet_unit.add_softmax_over_spatial_layer('att_weights0', 'energy')
        subnet_unit.add_dropout_layer(
          'att_weights', 'att_weights0', dropout=self.att_dropout, dropout_noise_shape={'*': None})
      else:
        subnet_unit.add_softmax_over_spatial_layer('att_weights', 'energy')
      subnet_unit.add_generic_att_layer('att0', weights='att_weights', base='base:enc_value')
      subnet_unit.add_merge_dims_layer('att', 'att0', axes='except_batch')

      # LM-like component same as here https://arxiv.org/pdf/2001.07263.pdf
      lstm_lm_component = None
      if self.add_lstm_lm:
        lstm_lm_component = subnet_unit.add_rnn_cell_layer(
          'lm_like_s', 'prev:target_embed', n_out=self.lstm_lm_dim, l2=self.l2)

      lstm_inputs = []
      if lstm_lm_component:
        lstm_inputs += [lstm_lm_component]
      else:
        lstm_inputs += ['prev:target_embed']
      lstm_inputs += ['prev:att']

      # LSTM decoder
      if self.dec_zoneout:
        subnet_unit.add_rnn_cell_layer(
          's', lstm_inputs, n_out=self.dec_lstm_num_units,
          unit='zoneoutlstm', unit_opts={'zoneout_factor_cell': 0.15, 'zoneout_factor_output': 0.05})
      else:
        if self.rec_weight_dropout:
          # a rec layer with unit nativelstm2 is required to use rec_weight_dropout
          subnet_unit.add_rec_layer(
            's', lstm_inputs, n_out=self.dec_lstm_num_units, l2=self.l2,
            rec_weight_dropout=self.rec_weight_dropout, unit='NativeLSTM2')
        else:
          subnet_unit.add_rnn_cell_layer(
            's', lstm_inputs, n_out=self.dec_lstm_num_units, l2=self.l2)

      # AM softmax output layer
      subnet_unit.add_linear_layer('readout_in', ["s", "prev:target_embed", "att"], n_out=self.dec_output_num_units)

      if self.reduceout:
        subnet_unit.add_reduceout_layer('readout', 'readout_in')
      else:
        subnet_unit.add_copy_layer('readout', 'readout_in')

      if self.local_fusion_opts:
        output_prob = subnet_unit.add_softmax_layer(
          'output_prob', 'readout', l2=self.l2, target=self.target, dropout=self.dropout)
        self._add_local_fusion(subnet_unit, am_output_prob=output_prob)
      elif self.mwer:
        # only MWER so CE is disabled
        output_prob = subnet_unit.add_softmax_layer(
          'output_prob', 'readout', l2=self.l2, target=self.target, dropout=self.dropout)
      else:
        output_prob = subnet_unit.add_softmax_layer(
          'output_prob', 'readout', l2=self.l2, loss='ce', loss_opts={'label_smoothing': self.label_smoothing},
          target=self.target, dropout=self.dropout)

      # for prior LM estimation
      prior_output_prob = None
      if self.prior_lm_opts:
        prior_output_prob = self._create_prior_net(subnet_unit)  # this require preload_from_files in config

      # Beam search
      # only support shallow fusion for now
      if self.ext_lm_opts:
        self._add_external_LM(subnet_unit, output_prob, prior_output_prob)
      else:
        if self.length_norm:
          subnet_unit.add_choice_layer(
            'output', 'output_prob', target=self.target, beam_size=self.beam_size, initial_output=0)
        else:
          subnet_unit.add_choice_layer(
            'output', 'output_prob', target=self.target, beam_size=self.beam_size, initial_output=0,
            length_normalization=False)

      # recurrent subnetwork
      dec_output = self.network.add_subnet_rec_layer(
        'output', unit=subnet_unit.get_net(), target=self.target, source=self.source)

      return dec_output

    def create_network(self):
      subnet_unit = ReturnnNetwork()

      dec_output = self.add_decoder_subnetwork(subnet_unit)

      # Add to Encoder network

      if hasattr(self.base_model, 'enc_proj_dim') and self.base_model.enc_proj_dim:
        self.base_model.network.add_copy_layer('enc_ctx', 'encoder_proj')
        self.base_model.network.add_split_dim_layer(
          'enc_value', 'encoder_proj', dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads))
      else:
        self.base_model.network.add_linear_layer(
          'enc_ctx', 'encoder', with_bias=True, n_out=self.enc_key_dim, l2=self.base_model.l2)
        self.base_model.network.add_split_dim_layer(
          'enc_value', 'encoder', dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads))

      self.base_model.network.add_linear_layer(
        'inv_fertility', 'encoder', activation='sigmoid', n_out=self.att_num_heads, with_bias=False)

      decision_layer_name = self.base_model.network.add_decide_layer('decision', dec_output, target=self.target)
      self.decision_layer_name = decision_layer_name

      return dec_output

  StartNumLayers = 1
  InitialDimFactor = 0.5

  encoder_keys = {'enc_key_dim', 'ff_dim', 'conv_kernel_size', 'lstm_dropout'}

  encoder_args_copy = copy.deepcopy(encoder_args)
  decoder_args_copy = copy.deepcopy(decoder_args)

  final_num_blocks = encoder_args_copy['num_blocks']
  assert final_num_blocks >= StartNumLayers

  net_dict["#config"] = {}
  if idx < 4:
    net_dict["#config"]["batch_size"] = 20000

  idx = max(idx - 1, 0)
  num_blocks = 2 ** idx

  if num_blocks > final_num_blocks:
    return None

  encoder_args_copy['num_blocks'] = num_blocks
  decoder_args_copy['label_smoothing'] = 0

  grow_frac = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
  dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac

  for key in encoder_keys:
    encoder_args_copy[key] = int(encoder_args_copy[key] * dim_frac / float(AttNumHeads)) * AttNumHeads

  for k in encoder_args_copy.keys():
    if 'dropout' in k and encoder_args_copy[k] is not None:
      if idx <= 1:
        encoder_args_copy[k] = 0.0
      else:
        encoder_args_copy[k] *= dim_frac
    if 'l2' in k and encoder_args_copy[k] is not None:
      if idx <= 1:
        encoder_args_copy[k] = 0.0
      else:
        encoder_args_copy[k] *= dim_frac

  for k in decoder_args_copy.keys():
    if 'dropout' in k and decoder_args_copy[k] is not None:
      if idx <= 1:
        decoder_args_copy[k] = 0.0
      else:
        decoder_args_copy[k] *= dim_frac
    if 'l2' in k and decoder_args_copy[k] is not None:
      if idx <= 1:
        decoder_args_copy[k] = 0.0
      else:
        decoder_args_copy[k] *= dim_frac

  conformer_encoder = ConformerEncoder(**encoder_args_copy)
  conformer_encoder.create_network()

  rnn_decoder = RNNDecoder(base_model=conformer_encoder, **decoder_args_copy)
  rnn_decoder.create_network()

  network = conformer_encoder.network.get_net()
  network.update(rnn_decoder.network.get_net())

  return network


def get_funcs():
  funcs = []
  for k, v in list(globals().items()):
    if callable(v):
      if k == 'get_funcs':
        continue
      funcs.append(v)
  return funcs
