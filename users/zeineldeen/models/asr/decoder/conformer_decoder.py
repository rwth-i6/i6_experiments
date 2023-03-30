from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class ConformerDecoder:
  """
  Represents Conformer Decoder with causal convolution modules and masked self-attention
  """

  def __init__(self,
               base_model, target='bpe', num_layers=6, beam_size=12, ff_init=None, ff_dim=2048,
               ff_bias=True, activation='swish', use_sqrd_relu=False, conv_kernel_size=32, conv_module_init=None,
               att_num_heads=8, dropout=0.1, att_dropout=0.1, softmax_dropout=0.0, embed_dropout=0.1, l2=0.0,
               self_att_l2=0.0, apply_embed_weight=False, label_smoothing=0.1, mhsa_init=None, half_step=True,
               mhsa_out_init=None, pos_enc=None, rel_pos_clipping=16, length_normalization=True,
               replace_cross_att_w_masked_self_att=False, create_ilm_decoder=False, ilm_type=None, ilm_args=None):

    self.base_model = base_model
    self.enc_value_dim = base_model.enc_value_dim
    self.enc_key_dim = base_model.enc_key_dim
    self.enc_att_num_heads = base_model.att_num_heads
    self.enc_key_per_head_dim = base_model.enc_key_per_head_dim
    self.enc_val_per_head_dim = base_model.enc_val_per_head_dim

    self.att_num_heads = att_num_heads

    self.target = target
    self.num_layers = num_layers
    self.beam_size = beam_size

    self.ff_init = ff_init
    self.ff_dim = ff_dim
    self.ff_bias = ff_bias

    self.conv_kernel_size = conv_kernel_size
    self.conv_module_init = conv_module_init

    self.activation = activation
    self.use_sqrd_relu = use_sqrd_relu

    self.mhsa_init = mhsa_init
    self.mhsa_out_init = mhsa_out_init

    self.pos_enc = pos_enc
    self.rel_pos_clipping = rel_pos_clipping
    self.half_step = half_step

    self.dropout = dropout
    self.softmax_dropout = softmax_dropout
    self.att_dropout = att_dropout
    self.label_smoothing = label_smoothing

    self.l2 = l2
    self.self_att_l2 = self_att_l2

    self.embed_dropout = embed_dropout
    self.embed_weight = None

    if apply_embed_weight:
      self.embed_weight = self.enc_value_dim ** 0.5

    self.decision_layer_name = None
    self.length_normalization = length_normalization

    self.replace_cross_att_w_masked_self_att = replace_cross_att_w_masked_self_att  # used to train ILM

    # used for recognition with ILM
    self.create_ilm_decoder = create_ilm_decoder
    self.ilm_type = ilm_type
    self.ilm_args = ilm_args or {}
    if self.create_ilm_decoder:
      self.replace_cross_att_w_masked_self_att = False  # keep original decoder as-is

    self.network = ReturnnNetwork()
    self.subnet_unit = ReturnnNetwork()
    self.output_prob = None

  def _create_masked_mhsa(self, prefix, source, **kwargs):
    prefix_name = '{}_self_att'.format(prefix)

    # for tuning mini-self-att ILM
    att_num_heads = kwargs.get('att_num_heads', self.att_num_heads)
    enc_key_dim = kwargs.get('enc_key_dim', self.enc_key_dim)
    enc_key_per_head_dim = enc_key_dim // att_num_heads

    ln = self.subnet_unit.add_layer_norm_layer('{}_ln'.format(prefix_name), source)
    ln_rel_pos_enc = None

    if self.pos_enc == 'rel':
      ln_rel_pos_enc = self.subnet_unit.add_relative_pos_encoding_layer(
        '{}_ln_rel_pos_enc'.format(prefix_name), ln, n_out=enc_key_per_head_dim, forward_weights_init=self.ff_init,
        clipping=self.rel_pos_clipping)

    mhsa = self.subnet_unit.add_self_att_layer(
      '{}'.format(prefix_name), ln, n_out=self.enc_value_dim, num_heads=att_num_heads, attention_left_only=True,
      total_key_dim=enc_key_dim, att_dropout=self.att_dropout, forward_weights_init=self.mhsa_init,
      key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None, l2=self.self_att_l2)

    mhsa_linear = self.subnet_unit.add_linear_layer(
      '{}_linear'.format(prefix_name), mhsa, n_out=enc_key_dim, l2=self.l2,
      forward_weights_init=self.mhsa_out_init,
      with_bias=False)

    drop = self.subnet_unit.add_dropout_layer('{}_dropout'.format(prefix_name), mhsa_linear, dropout=self.dropout)

    res_inputs = [drop, source]

    mhsa_res = self.subnet_unit.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=res_inputs, n_out=self.enc_value_dim)
    return mhsa_res


  def _create_mhsa(self, prefix, source):
    ln = self.subnet_unit.add_layer_norm_layer('{}_att_ln'.format(prefix), source)

    att_query0 = self.subnet_unit.add_linear_layer(
      '{}_att_query0'.format(prefix), ln, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_init, l2=self.l2)

    # (B, H, D/H)
    att_query = self.subnet_unit.add_split_dim_layer(
      '{}_att_query'.format(prefix), att_query0, axis='F', dims=(self.enc_att_num_heads, self.enc_key_per_head_dim))

    # --------------- Add to the encoder network --------------- #
    att_key0 = self.base_model.network.add_linear_layer(
      '{}_att_key0'.format(prefix), 'encoder', with_bias=False, n_out=self.enc_key_dim,
      forward_weights_init=self.mhsa_init, l2=self.l2)

    # (B, enc-T, H, D/H)
    att_key = self.base_model.network.add_split_dim_layer(
      '{}_att_key'.format(prefix), att_key0, axis='F', dims=(self.enc_att_num_heads, self.enc_key_per_head_dim))

    att_value0 = self.base_model.network.add_linear_layer(
      '{}_att_value0'.format(prefix), 'encoder', with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_init, l2=self.l2)

    # (B, enc-T, H, D'/H)
    att_value = self.base_model.network.add_split_dim_layer(
      '{}_att_value'.format(prefix), att_value0, axis='F', dims=(self.enc_att_num_heads, self.enc_val_per_head_dim))
    # ----------------------------------------------------------- #

    # (B, H, enc-T, 1)
    att_energy = self.subnet_unit.add_dot_layer(
      '{}_att_energy'.format(prefix), source=['base:' + att_key, att_query], red1=-1, red2=-1, var1='T', var2='T?')

    att_weights = self.subnet_unit.add_softmax_over_spatial_layer(
      '{}_att_weights'.format(prefix), att_energy, energy_factor=self.enc_key_per_head_dim ** -0.5)

    att_weights_drop = self.subnet_unit.add_dropout_layer(
      '{}_att_weights_drop'.format(prefix), att_weights, dropout=self.att_dropout, dropout_noise_shape={"*": None})

    # (B, H, V)
    att0 = self.subnet_unit.add_generic_att_layer(
      '{}_att0'.format(prefix), weights=att_weights_drop, base='base:' + att_value)

    att = self.subnet_unit.add_merge_dims_layer('{}_att'.format(prefix), att0, axes='static')  # (B, H*V) except_batch

    # output projection
    att_linear = self.subnet_unit.add_linear_layer(
      '{}_att_linear'.format(prefix), att, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_out_init, l2=self.l2)

    att_drop = self.subnet_unit.add_dropout_layer('{}_att_drop'.format(prefix), att_linear, dropout=self.dropout)

    out = self.subnet_unit.add_combine_layer(
      '{}_att_out'.format(prefix), [att_drop, source], kind='add', n_out=self.enc_value_dim)
    return out

  def _create_convolution_module(self, prefix_name, source):
    """
    Add Convolution Module:
      LN + point-wise-conv + GLU + depth-wise-conv + Swish + point-wise-conv + Dropout
    Note that BN is disabled here because it uses full sequence.

    :param str prefix_name: some prefix name
    :param str source: name of source layer
    :return: last layer name of this module
    :rtype: str
    """
    prefix_name = '{}_conv_mod'.format(prefix_name)

    ln = self.subnet_unit.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    pointwise_conv1 = self.subnet_unit.add_linear_layer(
      '{}_pointwise_conv1'.format(prefix_name), ln, n_out=2 * self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.conv_module_init)

    glu_act = self.subnet_unit.add_gating_layer('{}_glu'.format(prefix_name), pointwise_conv1)

    # Pad to make causal conv
    # TODO: This currently does not work inside a recurrent subnetwork. Need to be fixed.
    depthwise_conv_input_padded = self.subnet_unit.add_pad_layer(
      '{}_depthwise_conv_input_padded'.format(prefix_name),
      glu_act, axes='T', padding=(self.conv_kernel_size - 1, 0)
    )

    depthwise_conv = self.subnet_unit.add_conv_layer(
      '{}_depthwise_conv2'.format(prefix_name), depthwise_conv_input_padded, n_out=self.enc_key_dim,
      filter_size=(self.conv_kernel_size,), groups=self.enc_key_dim, l2=self.l2,
      forward_weights_init=self.conv_module_init, padding='valid')

    swish_act = self.subnet_unit.add_activation_layer(
      '{}_swish'.format(prefix_name), depthwise_conv, activation='swish')

    pointwise_conv2 = self.subnet_unit.add_linear_layer(
      '{}_pointwise_conv2'.format(prefix_name), swish_act, n_out=self.enc_key_dim, activation=None, l2=self.l2,
      with_bias=self.ff_bias, forward_weights_init=self.conv_module_init)

    drop = self.subnet_unit.add_dropout_layer('{}_drop'.format(prefix_name), pointwise_conv2, dropout=self.dropout)

    res_inputs = [drop, source]

    res = self.subnet_unit.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=res_inputs, n_out=self.enc_key_dim)
    return res

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

    ln = self.subnet_unit.add_layer_norm_layer('{}_ln'.format(prefix_name), source)

    ff1 = self.subnet_unit.add_linear_layer(
      '{}_ff1'.format(prefix_name), ln, n_out=self.ff_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=self.ff_bias)

    if self.use_sqrd_relu:
      swish_act = self.subnet_unit.add_activation_layer('{}_relu'.format(prefix_name), ff1, activation='relu')
      swish_act = self.subnet_unit.add_eval_layer('{}_square_relu'.format(prefix_name), swish_act, eval='source(0) ** 2')
    else:
      swish_act = self.subnet_unit.add_activation_layer('{}_swish'.format(prefix_name), ff1, activation=self.activation)

    drop1 = self.subnet_unit.add_dropout_layer('{}_drop1'.format(prefix_name), swish_act, dropout=self.dropout)

    ff2 = self.subnet_unit.add_linear_layer(
      '{}_ff2'.format(prefix_name), drop1, n_out=self.enc_key_dim, l2=self.l2, forward_weights_init=self.ff_init,
      with_bias=self.ff_bias)

    drop2 = self.subnet_unit.add_dropout_layer('{}_drop2'.format(prefix_name), ff2, dropout=self.dropout)

    if self.half_step:
      drop2 = self.subnet_unit.add_eval_layer('{}_half_step'.format(prefix_name), drop2, eval='0.5 * source(0)')

    res_inputs = [drop2, source]

    ff_module_res = self.subnet_unit.add_combine_layer(
      '{}_res'.format(prefix_name), kind='add', source=res_inputs, n_out=self.enc_key_dim)

    return ff_module_res

  def _create_decoder_block(self, source, i):
    """FF + Masked-MHSA + Causal-Conv + Cross-MHA + FF"""

    prefix = 'conformer_decoder_%02i' % i

    ff1 = self._create_ff_module(prefix, 1, source)
    masked_mhsa = self._create_masked_mhsa(prefix, ff1)
    conv_module = self._create_convolution_module(prefix, masked_mhsa)

    if self.replace_cross_att_w_masked_self_att:
      mhsa = self._create_masked_mhsa('ilm_' + prefix, conv_module, **self.ilm_args)
    else:
      mhsa = self._create_mhsa(prefix, conv_module)

    ff2 = self._create_ff_module(prefix, 2, mhsa)
    ff2_norm = self.subnet_unit.add_layer_norm_layer('{}_ln'.format(prefix), ff2)
    out = self.subnet_unit.add_copy_layer(prefix, ff2_norm)
    return out

  def _create_ilm_decoder_block(self, source, i):
    prefix = 'conformer_decoder_%02i' % i

    ff1 = self._create_ff_module('prior_' + prefix, 1, source)
    masked_mhsa = self._create_masked_mhsa('prior_' + prefix, ff1)
    conv_module = self._create_convolution_module('prior_' + prefix, masked_mhsa)

    if self.ilm_type == 'mini_lstm':
      mhsa = self._create_masked_mhsa('mini_ilm_' + prefix, conv_module, **self.ilm_args)
    else:
      assert self.ilm_type == 'zero'
      mhsa = self.subnet_unit.add_eval_layer('zero_att_%02i' % i, conv_module, eval='tf.zeros_like(source(0))')

    ff2 = self._create_ff_module('prior_' + prefix, 2, mhsa)
    ff2_norm = self.subnet_unit.add_layer_norm_layer('{}_ln'.format('prior_' + prefix), ff2)
    out = self.subnet_unit.add_copy_layer('prior_' + prefix, ff2_norm)
    return out

  def _create_decoder(self):

    self.output_prob = self.subnet_unit.add_softmax_layer(
      'output_prob', 'decoder', loss='ce',
      loss_opts={'label_smoothing': self.label_smoothing}, target=self.target, dropout=self.softmax_dropout,
      forward_weights_init=self.ff_init, l2=self.l2)

    if self.length_normalization:
      output = self.subnet_unit.add_choice_layer(
        'output', self.output_prob, target=self.target, beam_size=self.beam_size, initial_output=0)
    else:
      output = self.subnet_unit.add_choice_layer(
        'output', self.output_prob, target=self.target, beam_size=self.beam_size, initial_output=0,
        length_normalization=self.length_normalization)

      self.subnet_unit.add_compare_layer('end', output, value=0)

    target_embed_raw = self.subnet_unit.add_linear_layer(
      'target_embed_raw', 'prev:' + output, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.ff_init, l2=self.l2)

    if self.embed_weight:
      target_embed_raw = self.subnet_unit.add_eval_layer(
        'target_embed_weighted', target_embed_raw, eval='source(0) * %f' % self.embed_weight)

    target_embed = self.subnet_unit.add_dropout_layer(
      'target_embed', target_embed_raw, dropout=self.embed_dropout, dropout_noise_shape={"*": None})

    x = target_embed
    for i in range(1, self.num_layers + 1):
      x = self._create_decoder_block(x, i)
    self.subnet_unit.add_copy_layer('decoder', x)

    if self.create_ilm_decoder:
      x = target_embed
      for i in range(1, self.num_layers + 1):
        x = self._create_ilm_decoder_block( x, i)
      self.subnet_unit.add_copy_layer('prior_decoder', x)

      self.subnet_unit.add_softmax_layer(
        'prior_output_prob', 'prior_decoder', loss='ce',
        loss_opts={'label_smoothing': self.label_smoothing}, target=self.target, dropout=self.softmax_dropout,
        forward_weights_init=self.ff_init, l2=self.l2
      )

    dec_output = self.network.add_subnet_rec_layer('output', unit=self.subnet_unit.get_net(), target=self.target)

    return dec_output

  def create_network(self):
    dec_output = self._create_decoder()

    # recurrent subnetwork
    decision_layer_name = self.base_model.network.add_decide_layer('decision', dec_output, target=self.target)
    self.decision_layer_name = decision_layer_name

    return dec_output
