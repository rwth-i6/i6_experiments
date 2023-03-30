__all__ = ['conformer_network']

"""
add groups in depth-wise conv layer
"""

from .returnn_network import _NetworkMakerHelper

def conformer_network(**kwargs):
  conformer_enc = ConformerEncoder(**kwargs)
  conformer_enc.create_network()
  return conformer_enc.network.get_net()


class ConformerEncoder:
  """
  Represents Conformer Encoder Architecture
  * Conformer: Convolution-augmented Transformer for Speech Recognition
  * Ref: https://arxiv.org/abs/2005.08100
  """

  def __init__(self, source='data',
               use_specaugment=True,
               input_layer='conv',
               num_blocks=6, conv_kernel_size=32, pos_enc='rel',
               activation='swish', glu_gate_activation='sigmoid', block_final_norm=True, ff_dim=512, ff_init=None, ff_bias=True,
               embed_dropout=0.1, dropout=0.1, att_dropout=0.1,
               enc_key_dim=256, att_num_heads=4, l2=None,
               lstm_dropout=0.1, rec_weight_dropout=0., with_ctc=False, native_ctc=False, ctc_dropout=0., ctc_l2=0.,
               ctc_opts=None, init_depth_scale=False, use_talking_heads=False, ce_loss_ops={}, iterated_loss_layers=None,iterated_loss_scale={'4': 0.3, '8': 0.3, '12': 0.3},
               use_rms_norm=False, reuse_upsample_params=True, reuse_softmax_params = False, tconv_l2 = 0.01, tconv_filter_size=3,
               tconv_act='relu', use_long_skip=False, blstm_before_vgg=False, output_loss_scale=1, MLP_on_output=False,
               share_MLP = False, layernorm_replace_batchnorm = False, thsa_init=None, ln_on_source_linear=False, conv_after_mhsa=True,
               macron_on_conv=False, use_updated_bn=False, bn_delay_sample_update=False, bn_as_gn_size=0, 
               conv_kernel_size_2=32, downsample_with_blstm=False, ws_on_conv=False):
    """
    :param str source: input layer name
    :param bool use_specaugment:
    :param str input_layer: type of input layer which does subsampling
    :param int num_blocks: number of Conformer blocks
    :param int conv_kernel_size: kernel size for conv layers in Convolution module
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

    self.source = source
    self.input_layer = input_layer

    self.num_blocks = num_blocks
    self.conv_kernel_size = conv_kernel_size

    self.pos_enc = pos_enc

    self.ff_init = ff_init
    self.ff_bias = ff_bias

    self.use_specaugment = use_specaugment

    self.activation = activation

    self.block_final_norm = block_final_norm

    self.embed_dropout = embed_dropout
    self.dropout = dropout
    self.att_dropout = att_dropout
    self.lstm_dropout = lstm_dropout

    # key and value dimensions are the same
    self.enc_key_dim = enc_key_dim
    self.enc_value_dim = enc_key_dim
    self.att_num_heads = att_num_heads
    self.enc_key_per_head_dim = enc_key_dim // att_num_heads
    self.enc_val_per_head_dim = enc_key_dim // att_num_heads

    self.ff_dim = ff_dim
    if self.ff_dim is None:
      self.ff_dim = 2 * self.enc_key_dim

    self.l2 = l2
    self.rec_weight_dropout = rec_weight_dropout

    self.with_ctc = with_ctc
    self.native_ctc = native_ctc
    self.ctc_dropout = ctc_dropout
    self.ctc_l2 = ctc_l2
    self.ctc_opts = ctc_opts
    if not self.ctc_opts:
      self.ctc_opts = {}
    self.init_depth_scale = init_depth_scale
    self.use_talking_heads = use_talking_heads
    self.ce_loss_ops = ce_loss_ops
    self.iterated_loss_layers = iterated_loss_layers
    self.use_rms_norm = use_rms_norm
    self.glu_gate_activation = glu_gate_activation
    self.iterated_loss_scale = iterated_loss_scale
    self.reuse_upsample_params = reuse_upsample_params
    self.reuse_softmax_params = reuse_softmax_params
    self.tconv_l2 = tconv_l2
    self.tconv_filter_size = tconv_filter_size
    self.tconv_act = tconv_act
    self.use_long_skip = use_long_skip
    self.blstm_before_vgg=blstm_before_vgg
    self.output_loss_scale = output_loss_scale
    self.MLP_on_output = MLP_on_output
    self.share_MLP = share_MLP
    self.layernorm_replace_batchnorm = layernorm_replace_batchnorm
    self.thsa_init = thsa_init
    self.ln_on_source_linear = ln_on_source_linear
    self.conv_after_mhsa = conv_after_mhsa
    self.macron_on_conv = macron_on_conv
    self.use_updated_bn = use_updated_bn
    self.bn_delay_sample_update = bn_delay_sample_update
    self.bn_as_gn_size = bn_as_gn_size
    self.conv_kernel_size_2 = conv_kernel_size_2
    self.downsample_with_blstm = downsample_with_blstm
    self.ws_on_conv = ws_on_conv

    self.network = _NetworkMakerHelper()

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

    if self.use_rms_norm:
      ln = self.network.add_rms_norm_layer('{}_ln'.format(prefix_name), source)
    else:
      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    ff1 = self.network.add_linear_layer(
      '{}_ff1'.format(prefix_name), ln, n_out=self.ff_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=self.ff_bias)

    if self.activation == 'swish':
      swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), ff1, activation=self.activation)
    elif 'glu' in self.activation:
      swish_act = self.network.add_gating_layer('{}_swish_glu'.format(prefix_name), ff1, activation='identity', gate_activation=self.activation[:-4])

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
    if self.use_rms_norm:
      ln = self.network.add_rms_norm_layer('{}_ln'.format(prefix_name), source)
    else:
      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)
    ln_rel_pos_enc = None
    if self.pos_enc == 'rel':
      ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
        '{}_ln_rel_pos_enc'.format(prefix_name), ln, n_out=self.enc_key_per_head_dim, forward_weights_init=self.ff_init)
    
    if not self.use_talking_heads:
      mhsa = self.network.add_self_att_layer(
      '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=self.att_num_heads,
      total_key_dim=self.enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.ff_init,
      key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)
    else:
      mhsa = self.network.add_talking_heads_self_layer(
        '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=self.att_num_heads,
        total_key_dim=self.enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.ff_init, 
        projection_weights_init = self.thsa_init,
        key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)
    
    mhsa_linear = self.network.add_linear_layer(
      '{}_linear'.format(prefix_name), mhsa, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
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

    if self.use_rms_norm:
      ln = self.network.add_rms_norm_layer('{}_ln'.format(prefix_name), source)
    else:
      ln = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    pointwise_conv1 = self.network.add_linear_layer(
      '{}_pointwise_conv1'.format(prefix_name), ln, n_out=2 * self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.ff_init)

    glu_act = self.network.add_gating_layer('{}_glu'.format(prefix_name), pointwise_conv1, activation='identity', gate_activation=self.glu_gate_activation)

    conv_kernel_size = self.conv_kernel_size
    if self.macron_on_conv and '_covmod_2' in prefix_name:
      conv_kernel_size = self.conv_kernel_size_2
      
    depthwise_conv = self.network.add_conv_layer(
      '{}_depthwise_conv2'.format(prefix_name), glu_act, n_out=self.enc_key_dim,
      filter_size=(conv_kernel_size,), groups=self.enc_key_dim, l2=self.l2)
    
    if self.ws_on_conv:
      filter_shape = list((conv_kernel_size,)) + [self.enc_key_dim // self.enc_key_dim, self.enc_key_dim] # (kernel_size, group, n_out)
      self.network['{}_depthwise_conv2_w'.format(prefix_name)] = {'class': 'variable', 'shape': filter_shape, 'init': "glorot_uniform", 'add_batch_axis': False}
      # weight standardization
      # self.network['{}_depthwise_conv2_w_mean'.format(prefix_name)] = {'class':'reduce', 'mode':'mean', 'axes': [0, 1], 'keep_dims':True, 'from': ['{}_depthwise_conv2_w'.format(prefix_name)]}
      self.network['{}_depthwise_conv2_w_subtract_mean'.format(prefix_name)] = {'class': 'eval', 'from': ['{}_depthwise_conv2_w'.format(prefix_name)], 'eval': 'source(0) - tf.math.reduce_mean(source(0), axis=[0, 1], keepdims=True)'}
      # self.network['{}_depthwise_conv2_w_std'.format(prefix_name)] = {'class': 'eval', 'from': ['{}_depthwise_conv2_w'.format(prefix_name)], 'eval': 'tf.math.reduce_std(source(0), axis=[0, 1], keepdims=True)'}
      self.network['{}_depthwise_conv2_w_ws'.format(prefix_name)] = {'class': 'eval', 'from': ['{}_depthwise_conv2_w_subtract_mean'.format(prefix_name), '{}_depthwise_conv2_w'.format(prefix_name)], 'eval': 'source(0)/(tf.math.reduce_std(source(1), axis=[0, 1], keepdims=True)+1e-8)'}
      depthwise_conv = self.network.add_conv_layer(
      '{}_depthwise_conv2'.format(prefix_name), glu_act, n_out=self.enc_key_dim,
      filter_size=(conv_kernel_size,), groups=self.enc_key_dim, l2=self.l2, filter='{}_depthwise_conv2_w_ws'.format(prefix_name))

    if self.layernorm_replace_batchnorm:
      # only on F dimensions
      bn = self.network.add_layer_norm_layer('{}_ln_replace_bn'.format(prefix_name), depthwise_conv)
    elif self.bn_as_gn_size>0:
      # on T and F_ dimensions
      self.network['{}_gn_split_F'.format(prefix_name)] = {'class': 'split_dims', 'from': depthwise_conv, 'axis':'F', 'dims':(self.bn_as_gn_size,-1)}
      self.network['{}_gn_reinterpret_1'.format(prefix_name)] = {'class': 'reinterpret_data', 'enforce_batch_major':True, 'from': '{}_gn_split_F'.format(prefix_name), 'set_axes': {'F': 3}}
      self.network['{}_gn'.format(prefix_name)] = {'class': 'norm', 'axes': 'TF', 'from': '{}_gn_reinterpret_1'.format(prefix_name)}
      self.network['{}_gn_merge'.format(prefix_name)] = {'class': 'merge_dims', 'axes' : (2,3), 'from': '{}_gn'.format(prefix_name)}
      self.network['{}_gn_reinterpret_2'.format(prefix_name)] = {'class': 'reinterpret_data', 'enforce_batch_major':True, 'from': '{}_gn_merge'.format(prefix_name), 'set_axes': {'F': 2}}
      bn = '{}_gn_reinterpret_2'.format(prefix_name)
    else:
      # on B and T dimensions, but update default params
      if not self.use_updated_bn:
        bn = self.network.add_batch_norm_layer('{}_bn'.format(prefix_name), depthwise_conv)
      else:
        bn = self.network.add_batch_norm_layer('{}_bn'.format(prefix_name), depthwise_conv, delay_sample_update=self.bn_delay_sample_update, epsilon=0.001, momentum=0.1, update_sample_only_in_training=True)

    if self.activation == 'swish':
      swish_act = self.network.add_activation_layer('{}_swish'.format(prefix_name), bn, activation=self.activation)
    elif 'glu' in self.activation:
      swish_act = self.network.add_gating_layer('{}_swish_glu'.format(prefix_name), bn, activation='identity', gate_activation=self.activation[:-4])

    pointwise_conv2 = self.network.add_linear_layer(
      '{}_pointwise_conv2'.format(prefix_name), swish_act, n_out=self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.ff_init)

    drop = self.network.add_dropout_layer('{}_drop'.format(prefix_name), pointwise_conv2, dropout=self.dropout)

    if self.macron_on_conv:
      half_step_drop = self.network.add_eval_layer('{}_half_step'.format(prefix_name), drop, eval='0.5 * source(0)')
      res = self.network.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=[half_step_drop, source], n_out=self.enc_key_dim)
    else:
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
    if self.macron_on_conv:
      ff_module1 = self._create_ff_module(prefix_name, 1, source)
      conv_module_1 = self._create_convolution_module(prefix_name + '_covmod_1', ff_module1)
      mhsa = self._create_mhsa_module(prefix_name, conv_module_1)
      conv_module_2 = self._create_convolution_module(prefix_name+ '_covmod_2', mhsa)
      ff_module2 = self._create_ff_module(prefix_name, 2, conv_module_2)
    else:
      ff_module1 = self._create_ff_module(prefix_name, 1, source)
      if self.conv_after_mhsa:
        mhsa = self._create_mhsa_module(prefix_name, ff_module1)
        conv_module = self._create_convolution_module(prefix_name, mhsa)
        ff_module2 = self._create_ff_module(prefix_name, 2, conv_module)
      else:
        conv_module = self._create_convolution_module(prefix_name, ff_module1)
        mhsa = self._create_mhsa_module(prefix_name, conv_module)
        ff_module2 = self._create_ff_module(prefix_name, 2, mhsa)

    res = ff_module2
    if self.block_final_norm:
      if self.use_rms_norm:
        res = self.network.add_rms_norm_layer('{}_ln'.format(prefix_name), res)
      else:
        res = self.network.add_layer_norm_layer('{}_ln'.format(prefix_name), res)
    res = self.network.add_copy_layer(prefix_name, res)
    return res

  def create_network(self):
    """
    ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
    """
    data = self.source
    if self.use_specaugment:
      data = self.network.add_eval_layer('source', data,eval="self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)")
    
    if self.blstm_before_vgg:
      data = self.network.add_lstm_layers(
          data, num_layers=1, lstm_dim=self.enc_key_dim, dropout=self.lstm_dropout, bidirectional=True,
          rec_weight_dropout=self.rec_weight_dropout, l2=0.01, pool_sizes=[1])
    
    source_split = self.network.add_split_dim_layer('source0', data)

    if self.downsample_with_blstm:
      pool_sizes = [1, 3]
      # add 2 LSTM layers with max pooling to subsample and encode positional information
      subsampled_input = self.network.add_lstm_layers(
            data, num_layers=2, lstm_dim=self.enc_key_dim, dropout=self.lstm_dropout, bidirectional=True,
            rec_weight_dropout=self.rec_weight_dropout, l2=0.01, pool_sizes=pool_sizes)
      source_linear = self.network.add_linear_layer(
      'source_linear',subsampled_input, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=False)
    else:
      self.network["c1"] = {"class" : "conv", "n_out" : 32, "filter_size": (3,3), "padding": "same", "with_bias": True, "from": source_split}
      self.network["y1"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c1"}
      self.network["p1"] = {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "y1"}
      self.network["c3"] = {"class" : "conv", "n_out" : 64, "filter_size": (3,3), "padding": "same",  "with_bias": True,"from" : "p1" }
      self.network["y3"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c3"}
      self.network["c4"] = {"class" : "conv", "n_out" : 64, "filter_size": (3,3), "padding": "same",  "with_bias": True, "from" : "y3" }
      self.network["y4"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c4"}
      self.network["p2"] = {"class": "pool", "mode": "max", "padding": "same", "pool_size": (1, 2), "from": "y4"}
      self.network["c2"] = {"class" : "conv", "n_out" : 32, "filter_size": (self.tconv_filter_size,self.tconv_filter_size), "strides": (self.tconv_filter_size, 1), "padding": "same","with_bias": True,"from" : "y4" }  # downsample time
      self.network["y2"] = {"class": "activation", "activation": self.tconv_act, "batch_norm": False, "from": "c2"}
      self.network["vgg_conv_merged"] = {"class": "merge_dims", "from": "y2", "axes": "static"}

      source_linear = self.network.add_linear_layer(
        'source_linear', "vgg_conv_merged", n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
        with_bias=False)

    # add positional encoding
    if self.pos_enc == 'abs':
      source_linear = self.network.add_pos_encoding_layer('{}_abs_pos_enc'.format('vgg'), source_linear)
    
    source_dropout_input = source_linear
    if self.ln_on_source_linear:
      source_dropout_input = self.network.add_layer_norm_layer('source_linear_ln', source_linear)

    source_dropout = self.network.add_dropout_layer('source_dropout', source_dropout_input, dropout=self.embed_dropout)

    conformer_block_src = source_dropout
    for i in range(1, self.num_blocks + 1):
      if self.init_depth_scale != False:
        self.ff_init = {'class': 'VarianceScaling','scale': self.init_depth_scale/i**0.5,'distribution':'uniform','mode': 'fan_in'}
      conformer_block_src = self._create_conformer_block(i, conformer_block_src)
      if self.use_long_skip:
        conformer_block_src = self.network.add_combine_layer('transposed_conv_{}_res'.format(i), kind='add', source=['source_linear', conformer_block_src], n_out=self.enc_key_dim)

      if self.iterated_loss_layers != None and i in self.iterated_loss_layers:
        self.network["transposedconv_%s"%i] = { "class" :"transposed_conv", "n_out" : 512, "filter_size" : [self.tconv_filter_size], "strides": [self.tconv_filter_size],
                                                  "activation": self.tconv_act, "dropout": 0.1, "L2": self.tconv_l2, "from" : conformer_block_src}
        if self.reuse_upsample_params:
          self.network["transposedconv_%s"%i]["reuse_params"] = "transposedconv"          

        self.network["masked_tconv_%s"%i] = {"class": "reinterpret_data", "from":"transposedconv_%s"%i, "size_base": "data:classes"}
        aux_loss_layer = self.network.add_copy_layer('aux_output_block_%s'%i, "masked_tconv_%s"%i)
        aux_MLP = self.network.add_linear_layer('aux_MLP_block_%s'%i, aux_loss_layer, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,with_bias=self.ff_bias)
        if self.share_MLP and i == 4:
           self.network['aux_MLP_block_%s'%i]['reuse_params'] = 'aux_MLP_block_8'

        self.network['aux_output_block_%s_ce'%i] = {'class': "softmax", 'dropout': 0.0, 'from': aux_MLP, 'loss_opts':self.ce_loss_ops, 'loss':'ce', 'target': 'classes','loss_scale': self.iterated_loss_scale[str(i)]}
        if self.reuse_softmax_params:
          self.network['aux_output_block_%s_ce'%i]["reuse_params"] = "output"

    encoder = self.network.add_copy_layer('encoder', conformer_block_src)
    self.network["transposedconv"] = { "class" :"transposed_conv", "n_out" : 512, "filter_size" : [self.tconv_filter_size], "strides": [self.tconv_filter_size], 
                                       "activation": self.tconv_act, "dropout": 0.1, "L2": self.tconv_l2, "from" :  encoder}
    # reinterpret data to match alignment dim
    self.network["masked_tconv"] = {"class": "reinterpret_data", "from": "transposedconv", "size_base": "data:classes"}
    if self.MLP_on_output:
      output_MLP = self.network.add_linear_layer('MLP_output', "masked_tconv", n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,with_bias=self.ff_bias)
    else:
      output_MLP = "masked_tconv"
    # add outout layer
    self.network['output'] = {
      'class': "softmax", 'dropout': 0.0, 'from': output_MLP,
      'loss': "ce", 'loss_opts': self.ce_loss_ops, 'target': 'classes', 'loss_scale': self.output_loss_scale,
    }

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


