from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class TransformerDecoder:
  """
  Represents standard Transformer decoder

  * Attention Is All You Need
  * Ref: https://arxiv.org/abs/1706.03762
  """

  def __init__(self,
               base_model, target='bpe', num_layers=6, beam_size=12, ff_init=None, ff_dim=2048, ff_act='relu', att_num_heads=8,
               dropout=0.1, att_dropout=0.0, softmax_dropout=0.0, embed_dropout=0.1, l2=0.0, embed_pos_enc=False,
               apply_embed_weight=False, label_smoothing=0.1, mhsa_init=None, mhsa_out_init=None,
               pos_enc=None, rel_pos_clipping=16, length_normalization=True,
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
    self.ff_act = ff_act

    self.mhsa_init = mhsa_init
    self.mhsa_init_out = mhsa_out_init

    self.pos_enc = pos_enc
    self.rel_pos_clipping = rel_pos_clipping

    self.dropout = dropout
    self.softmax_dropout = softmax_dropout
    self.att_dropout = att_dropout
    self.label_smoothing = label_smoothing

    self.l2 = l2

    self.embed_dropout = embed_dropout
    self.embed_pos_enc = embed_pos_enc

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

  def _create_masked_mhsa(self, subnet_unit: ReturnnNetwork, prefix, source, **kwargs):
    prefix = '{}_self_att'.format(prefix)

    # for tuning mini-self-att ILM
    att_num_heads = kwargs.get('att_num_heads', self.att_num_heads)
    enc_key_dim = kwargs.get('enc_key_dim', self.enc_key_dim)
    enc_key_per_head_dim = enc_key_dim // att_num_heads

    ln = subnet_unit.add_layer_norm_layer('{}_ln'.format(prefix), source)

    ln_rel_pos_enc = None
    if self.pos_enc == 'rel':
      ln_rel_pos_enc = self.subnet_unit.add_relative_pos_encoding_layer(
        '{}_ln_rel_pos_enc'.format(prefix), ln, n_out=enc_key_per_head_dim, forward_weights_init=self.ff_init,
        clipping=self.rel_pos_clipping)

    att = subnet_unit.add_self_att_layer(
      '{}_att'.format(prefix), ln, num_heads=att_num_heads, total_key_dim=enc_key_dim,
      n_out=self.enc_value_dim, attention_left_only=True, att_dropout=self.att_dropout,
      forward_weights_init=self.mhsa_init, l2=self.l2, key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None)

    linear = subnet_unit.add_linear_layer(
      '{}_linear'.format(prefix), att, activation=None, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_init_out, l2=self.l2)

    drop = subnet_unit.add_dropout_layer('{}_drop'.format(prefix), linear, dropout=self.dropout)

    out = subnet_unit.add_combine_layer('{}_out'.format(prefix), [drop, source], kind='add', n_out=self.enc_value_dim)

    return out

  def _create_mhsa(self, subnet_unit: ReturnnNetwork, prefix, source):
    ln = subnet_unit.add_layer_norm_layer('{}_att_ln'.format(prefix), source)

    att_query0 = subnet_unit.add_linear_layer(
      '{}_att_query0'.format(prefix), ln, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_init, l2=self.l2)

    # (B, H, D/H)
    att_query = subnet_unit.add_split_dim_layer(
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
    att_energy = subnet_unit.add_dot_layer(
      '{}_att_energy'.format(prefix), source=['base:' + att_key, att_query], red1=-1, red2=-1, var1='T', var2='T?')

    att_weights = subnet_unit.add_softmax_over_spatial_layer(
      '{}_att_weights'.format(prefix), att_energy, energy_factor=self.enc_key_per_head_dim ** -0.5)

    att_weights_drop = subnet_unit.add_dropout_layer(
      '{}_att_weights_drop'.format(prefix), att_weights, dropout=self.att_dropout, dropout_noise_shape={"*": None})

    # (B, H, V)
    att0 = subnet_unit.add_generic_att_layer(
      '{}_att0'.format(prefix), weights=att_weights_drop, base='base:' + att_value)

    att = subnet_unit.add_merge_dims_layer('{}_att'.format(prefix), att0, axes='static')  # (B, H*V) except_batch

    # output projection
    att_linear = subnet_unit.add_linear_layer(
      '{}_att_linear'.format(prefix), att, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.mhsa_init_out, l2=self.l2)

    att_drop = subnet_unit.add_dropout_layer('{}_att_drop'.format(prefix), att_linear, dropout=self.dropout)

    out = subnet_unit.add_combine_layer(
      '{}_att_out'.format(prefix), [att_drop, source], kind='add', n_out=self.enc_value_dim)
    return out

  def _create_ff_module(self, subnet_unit: ReturnnNetwork, prefix, source):
    ff_ln = subnet_unit.add_layer_norm_layer('{}_ff_ln'.format(prefix), source)

    ff1 = subnet_unit.add_linear_layer(
      '{}_ff_conv1'.format(prefix), ff_ln, activation=self.ff_act, forward_weights_init=self.ff_init, n_out=self.ff_dim,
      with_bias=True, l2=self.l2)

    ff2 = subnet_unit.add_linear_layer(
      '{}_ff_conv2'.format(prefix), ff1, activation=None, forward_weights_init=self.ff_init, n_out=self.enc_value_dim,
      dropout=self.dropout, with_bias=True, l2=self.l2)

    drop = subnet_unit.add_dropout_layer('{}_ff_drop'.format(prefix), ff2, dropout=self.dropout)

    out = subnet_unit.add_combine_layer(
      '{}_ff_out'.format(prefix), [drop, source], kind='add', n_out=self.enc_value_dim)
    return out

  def _create_decoder_block(self, subnet_unit: ReturnnNetwork, source, i):
    prefix = 'transformer_decoder_%02i' % i
    masked_mhsa = self._create_masked_mhsa(subnet_unit, prefix, source)
    if self.replace_cross_att_w_masked_self_att:
      mhsa = self._create_masked_mhsa(subnet_unit, 'ilm_' + prefix, masked_mhsa, **self.ilm_args)
    else:
      mhsa = self._create_mhsa(subnet_unit, prefix, masked_mhsa)
    ff = self._create_ff_module(subnet_unit, prefix, mhsa)
    out = subnet_unit.add_copy_layer(prefix, ff)
    return out

  def _create_ilm_decoder_block(self, subnet_unit: ReturnnNetwork, source, i):
    prefix = 'transformer_decoder_%02i' % i
    masked_mhsa = self._create_masked_mhsa(subnet_unit, 'prior_' + prefix, source)
    if self.ilm_type == 'mini_lstm':
      mhsa = self._create_masked_mhsa(subnet_unit, 'mini_ilm_' + prefix, masked_mhsa, **self.ilm_args)
    else:
      assert self.ilm_type == 'zero'
      mhsa = subnet_unit.add_eval_layer('zero_att_%02i' % i, masked_mhsa, eval='tf.zeros_like(source(0))')
    ff = self._create_ff_module(subnet_unit, 'prior_' + prefix, mhsa)
    out = subnet_unit.add_copy_layer('prior_' + prefix, ff)
    return out

  def _create_decoder(self, subnet_unit: ReturnnNetwork):

    self.output_prob = subnet_unit.add_softmax_layer(
      'output_prob', 'decoder', loss='ce',
      loss_opts={'label_smoothing': self.label_smoothing}, target=self.target, dropout=self.softmax_dropout,
      forward_weights_init=self.ff_init, l2=self.l2)

    if self.length_normalization:
      output = subnet_unit.add_choice_layer(
        'output', self.output_prob, target=self.target, beam_size=self.beam_size, initial_output=0)
    else:
      output = subnet_unit.add_choice_layer(
        'output', self.output_prob, target=self.target, beam_size=self.beam_size, initial_output=0,
        length_normalization=self.length_normalization)

    subnet_unit.add_compare_layer('end', output, value=0)

    target_embed_raw = subnet_unit.add_linear_layer(
      'target_embed_raw', 'prev:' + output, with_bias=False, n_out=self.enc_value_dim,
      forward_weights_init=self.ff_init, l2=self.l2)

    if self.embed_weight:
      target_embed_raw = subnet_unit.add_eval_layer(
        'target_embed_weighted', target_embed_raw, eval='source(0) * %f' % self.embed_weight)

    if self.embed_pos_enc:
      target_embed_raw = subnet_unit.add_pos_encoding_layer('target_embed_pos_enc', target_embed_raw)

    target_embed = subnet_unit.add_dropout_layer(
      'target_embed', target_embed_raw, dropout=self.embed_dropout, dropout_noise_shape={"*": None})

    x = target_embed
    for i in range(1, self.num_layers + 1):
      x = self._create_decoder_block(subnet_unit, x, i)
    subnet_unit.add_layer_norm_layer('decoder', x)

    if self.create_ilm_decoder:
      x = target_embed
      for i in range(1, self.num_layers + 1):
        x = self._create_ilm_decoder_block(subnet_unit, x, i)
      subnet_unit.add_layer_norm_layer('prior_decoder', x)

      subnet_unit.add_softmax_layer(
        'prior_output_prob', 'prior_decoder', loss='ce',
        loss_opts={'label_smoothing': self.label_smoothing}, target=self.target, dropout=self.softmax_dropout,
        forward_weights_init=self.ff_init, l2=self.l2
      )

    dec_output = self.network.add_subnet_rec_layer('output', unit=subnet_unit.get_net(), target=self.target)

    return dec_output

  def create_network(self):
    dec_output = self._create_decoder(self.subnet_unit)

    # recurrent subnetwork
    decision_layer_name = self.base_model.network.add_decide_layer('decision', dec_output, target=self.target)
    self.decision_layer_name = decision_layer_name

    return dec_output
