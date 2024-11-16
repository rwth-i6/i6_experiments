from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork
import copy


class TransformerLM:

  def __init__(self, source='data:delayed', target='data', num_layers=6, ff_dim=4096, att_num_heads=8, out_dim=1024,
               qk_dim=1024, v_dim=1024, dropout=0.0, att_dropout=0.0, embed_dropout=0.0, embed_dim=128,
               emb_cpu_lookup=True, forward_weights_init=None, prefix_name=None, use_as_ext_lm=False, vocab_size=None):

    self.source = source
    self.target = target
    self.num_layers = num_layers

    self.ff_dim = ff_dim
    self.att_num_heads = att_num_heads
    self.out_dim = out_dim
    self.qk_dim = qk_dim
    self.v_dim = v_dim
    self.dropout = dropout
    self.embed_dropout = embed_dropout
    self.att_dropout = att_dropout
    self.embed_dim = embed_dim
    self.emb_cpu_lookup = emb_cpu_lookup

    # use this as default for now
    if forward_weights_init is None:
      forward_weights_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=1.0)"
    self.forward_weights_init = forward_weights_init

    self.use_as_ext_lm = use_as_ext_lm
    self.vocab_size = vocab_size
    if not prefix_name:
      prefix_name = ''
    self.prefix_name = prefix_name

    self.network = ReturnnNetwork()

  def _create_ff_block(self, subnet_unit: ReturnnNetwork, source, prefix):
    prefix = '{}_ff'.format(prefix)
    ln = subnet_unit.add_layer_norm_layer('{}_laynorm'.format(prefix), source)
    conv1 = subnet_unit.add_linear_layer('{}_conv1'.format(prefix), ln, with_bias=True, activation='relu',
                                         forward_weights_init=self.forward_weights_init, n_out=self.ff_dim)
    conv2 = subnet_unit.add_linear_layer('{}_conv2'.format(prefix), conv1, with_bias=True, activation=None,
                                         forward_weights_init=self.forward_weights_init, n_out=self.out_dim)
    drop = subnet_unit.add_dropout_layer('{}_drop'.format(prefix), conv2, dropout=self.dropout)
    out = subnet_unit.add_combine_layer('{}_out'.format(prefix), [drop, source], kind='add', n_out=self.out_dim)
    return out

  def _create_masked_mhsa(self, subnet_unit: ReturnnNetwork, source, prefix):
    prefix = '{}_self_att'.format(prefix)
    ln = subnet_unit.add_layer_norm_layer('{}_laynorm'.format(prefix), source)
    att = subnet_unit.add_self_att_layer(
      '{}_att'.format(prefix), ln, forward_weights_init=self.forward_weights_init, att_dropout=self.att_dropout,
      attention_left_only=True, n_out=self.v_dim, num_heads=self.att_num_heads, total_key_dim=self.qk_dim)
    lin = subnet_unit.add_linear_layer(
      '{}_lin'.format(prefix), att, n_out=self.out_dim, with_bias=False, forward_weights_init=self.forward_weights_init)
    drop = subnet_unit.add_dropout_layer('{}_drop'.format(prefix), lin, dropout=self.dropout)
    out = subnet_unit.add_combine_layer('{}_out'.format(prefix), [drop, source], kind='add', n_out=self.out_dim)
    return out

  def _create_decoder_block(self, subnet_unit: ReturnnNetwork, source, i):
    prefix = self.prefix_name + ('dec_%i' % i)
    masked_mhsa = self._create_masked_mhsa(subnet_unit, source, prefix)
    ff = self._create_ff_block(subnet_unit, masked_mhsa, prefix)
    out = subnet_unit.add_copy_layer(prefix, ff)
    return out

  def create_network(self):
    subnet_unit = ReturnnNetwork()
    target_embed_raw = subnet_unit.add_linear_layer(
      '{}target_embed_raw'.format(self.prefix_name), self.source, forward_weights_init=self.forward_weights_init,
      n_out=self.embed_dim, with_bias=False, param_device='CPU' if self.emb_cpu_lookup else None)

    target_embed_with_pos = subnet_unit.add_pos_encoding_layer(
      '{}target_embed_with_pos'.format(self.prefix_name), target_embed_raw)

    target_embed = subnet_unit.add_dropout_layer(
      '{}target_embed'.format(self.prefix_name), target_embed_with_pos, dropout=self.embed_dropout)

    target_embed_lin = subnet_unit.add_linear_layer(
      '{}target_embed_lin'.format(self.prefix_name), target_embed, with_bias=False,
      forward_weights_init=self.forward_weights_init, n_out=self.out_dim)

    x = target_embed_lin
    for i in range(self.num_layers):
      x = self._create_decoder_block(subnet_unit, x, i)

    # final LN
    decoder = subnet_unit.add_layer_norm_layer('{}decoder'.format(self.prefix_name), x)

    if self.use_as_ext_lm:
      subnet_unit.add_linear_layer('output', decoder, n_out=self.vocab_size)
      subnet_unit.get_net()['target_embed_raw'].pop('from')
      self.network = copy.deepcopy(subnet_unit)
    else:
      subnet_unit.add_softmax_layer(
        '{}output'.format(self.prefix_name), decoder, forward_weights_init=self.forward_weights_init, loss='ce',
        target=self.target, with_bias=True, dropout=self.dropout)

      self.network.add_subnet_rec_layer(
          'output', unit=subnet_unit.get_net(), target=self.target, source=self.source)

    return 'output'