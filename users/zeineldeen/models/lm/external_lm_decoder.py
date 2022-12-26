import copy
from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork


class ILMDecoder:

  def __init__(self, asr_decoder):
    self.asr_decoder = asr_decoder
    self.output_prob_name = None

  def _create_prior_net(self, subnet_unit: ReturnnNetwork):
    raise NotImplementedError

  def create_network(self):
    self.output_prob_name = self._create_prior_net(self.asr_decoder.subnet_unit)


class LSTMILMDecoder(ILMDecoder):
  """
  Add ILM decoder for RNN LSTM attention-based decoder. This currently assumes that there is 1 LSTM layer.
  """

  def __init__(self, asr_decoder, prior_lm_opts):
    super(LSTMILMDecoder, self).__init__(asr_decoder)
    self.asr_decoder = asr_decoder
    self.prior_lm_opts = prior_lm_opts

  def _add_prior_input(self, subnet_unit):
    # TODO: currently train_avg_ctx and train_avg_enc won't work since RETURNN does not support loading numpy vectors
    # with constant layer. A change in RETURNN was done to make this work but we need a better solution

    prior_type = self.prior_lm_opts.get('type', 'zero')

    if prior_type == 'zero':  # set att context vector to zero
      prior_att_input = subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
    elif prior_type == 'train_avg_ctx':  # average all context vectors over training data
      prior_att_input = subnet_unit.add_constant_layer(
        'train_avg_ctx', value=self.prior_lm_opts['data'], with_batch_dim=True, dtype='float32')
    elif prior_type == 'train_avg_enc':  # average all encoder states over training data
      prior_att_input = subnet_unit.add_constant_layer(
        'train_avg_enc', value=self.prior_lm_opts['data'], with_batch_dim=True, dtype='float32')
    elif prior_type == 'mini_lstm':  # train a mini LM-like LSTM and use that as prior
      mini_lstm_dim = self.prior_lm_opts.get('mini_lstm_dim', 50)
      ctx_dim = self.prior_lm_opts['ctx_dim']
      subnet_unit.add_rec_layer('mini_att_lstm', 'prev:target_embed', n_out=mini_lstm_dim, l2=0.0)
      prior_att_input = subnet_unit.add_linear_layer(
        'mini_att', 'mini_att_lstm', activation=None, n_out=ctx_dim)
    elif prior_type == 'trained_vec':
      prior_att_input = subnet_unit.add_variable_layer('trained_vec_att_var', shape=[2048], L2=0.0001)
    else:
      raise ValueError('{} prior type is not supported'.format(prior_type))

    return prior_att_input

  def _create_prior_net(self, subnet_unit: ReturnnNetwork):
    prior_att_input = self._add_prior_input(subnet_unit)

    # for the first frame in decoding, don't use average but zero always
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
            assert prev_att not in new_sources
            new_sources += [prev_att]  # switched based on decoder index
          else:
            new_sources += [prior_att_input]
        elif src in key_names:
          new_sources += [('prev:' if 'prev' in src else '') + 'prior_{}'.format(src.split(':')[-1])]
        else:
          new_sources += [src]
      d['from'] = new_sources
      subnet_unit['prior_{}'.format(key_name)] = d
    return 'prior_output_prob'


class TransformerILMDecoder(ILMDecoder):
  """
  Add ILM decoder for transformer attention-based decoder. It is possible to have multiple layers here.
  Layers: Masked MHSA - Cross MHSA - FF
  """

  def __init__(self, asr_decoder, prior_lm_opts):
    super(TransformerILMDecoder, self).__init__(asr_decoder)
    self.asr_decoder = asr_decoder
    self.prior_lm_opts = prior_lm_opts  # type: dict[str]

  def _add_prior_input(self, subnet_unit: ReturnnNetwork):
    prior_type = self.prior_lm_opts.get('type', None)
    assert prior_type is not None, 'prior_type not defined'

    if prior_type == 'mini_lstm':
      # add mini lstm layers
      subnet_unit.add_rec_layer(
        'mini_att_lstm', 'prev:' + self.prior_lm_opts.get('target_embed_name', 'target_embed'),
        n_out=self.prior_lm_opts.get('mini_lstm_dim', 50), l2=self.prior_lm_opts.get('l2', 0.0))
      prior_att_input = subnet_unit.add_linear_layer(
        'mini_att', 'mini_att_lstm', activation=None, n_out=512, l2=0.0001)
    elif prior_type == 'zero':
      prior_att_input = subnet_unit.add_eval_layer(
        'zero_att', 'transformer_decoder_01_att', eval='tf.zeros_like(source(0))')
    else:
      raise ValueError()

    return prior_att_input

  def _create_prior_net(self, subnet_unit):
    prior_att_input = self._add_prior_input(subnet_unit)

    ignored_layers = {
      'end', 'mini_att_lstm', 'mini_att', 'output', 'target_embed', 'target_embed_raw', 'target_embed_weighted',
      '_att_query0', '_att_query', '_att_energy', '_att_weights', '_att_weights_drop'}

    linear_variant = self.prior_lm_opts.get('variant', None)
    if self.prior_lm_opts['type'] == 'zero':
      assert linear_variant == 'no-linear', 'for zero approach, linear projection is not needed'

    for layer in list(self.asr_decoder.subnet_unit.get_net().keys()):
      if any([layer.endswith(l) for l in ignored_layers]):
        continue

      if linear_variant == 'no-linear' and layer.endswith('_att_linear'):
        continue  # just ignore

      d = copy.deepcopy(subnet_unit[layer])

      from_list = d.get('from', None)
      if from_list is None:
        continue
      if isinstance(from_list, str):
        from_list = [from_list]
      assert isinstance(from_list, list)
      new_from_list = []
      for e in from_list:
        if e.endswith('_att') and not e.endswith('_self_att_att'):  # a workaround to fix bug
          new_from_list.append(prior_att_input)
        elif linear_variant == 'no-linear' and e.endswith('_att_linear'):
          # use estimated context vector directly instead of att_linear. att_linear layer ignored above
          new_from_list.append(prior_att_input)
        elif (linear_variant == 'import-linear' or linear_variant == 'scratch-linear') and e.endswith('_att_linear'):
          # to be loaded from mini_lstm checkpoint
          new_from_list.append('mini_%s' % e)
        elif any([e.endswith(l) for l in ignored_layers]):
          new_from_list.append(e)
        else:
          new_from_list.append(('prev:' if 'prev' in e else '') + 'prior_%s' % e)

      d['from'] = new_from_list

      if (linear_variant == 'import-linear' or linear_variant == 'scratch-linear') and layer.endswith('_att_linear'):
        # to be loaded from mini_lstm checkpoint
        subnet_unit['mini_%s' % layer] = d
      else:
        subnet_unit['prior_%s' % layer] = d

    return 'prior_output_prob'


class TransformerMiniLSTMDecoder(ILMDecoder):
  """
  Add ILM decoder for transformer attention-based decoder
  Layers: Masked MHSA - Cross MHSA - FF

  Basically, for each transformer layer, we try to estimate a context vector. Let M be the number of decoder layers.
  This can be done either by training M Mini-LSTMs and M linear projections where each will estimate the context vector
  for layer m (many). Or, we can train only 1 Mini-LSTM and do M linear projections (single).
  """

  def __init__(self, asr_decoder, prior_lm_opts):
    super(TransformerMiniLSTMDecoder, self).__init__(asr_decoder)
    self.asr_decoder = asr_decoder
    self.prior_lm_opts = prior_lm_opts  # type: dict[str]

  def _add_prior_input(self, subnet_unit: ReturnnNetwork):
    prior_type = self.prior_lm_opts['type']
    assert prior_type == 'mini_lstm'

    num_layers = self.prior_lm_opts['dec_layers']
    assert num_layers > 0

    variant = self.prior_lm_opts['mini_lstm_variant']
    assert variant in ['single', 'many']

    single_att_proj = self.prior_lm_opts.get('single_att_proj', False)  # by default we have dec_N att projections

    if variant == 'single':
      subnet_unit.add_rec_layer(
        'mini_att_lstm', 'prev:' + self.prior_lm_opts.get('target_embed_name', 'target_embed'),
        n_out=self.prior_lm_opts.get('mini_lstm_dim', 50), l2=self.prior_lm_opts.get('l2', 0.0))
    else:
      for i in range(1, num_layers + 1):
        subnet_unit.add_rec_layer(
          'mini_att_lstm_%02i' % i, 'prev:' + self.prior_lm_opts.get('target_embed_name', 'target_embed'),
          n_out=self.prior_lm_opts.get('mini_lstm_dim', 50), l2=self.prior_lm_opts.get('l2', 0.0))

    if single_att_proj:
      assert variant == 'single', 'single_att_proj requires only one mini-lstm'
      subnet_unit.add_linear_layer('mini_att', 'mini_att_lstm', activation=None, n_out=512, l2=0.0001)
    else:
      for i in range(1, num_layers + 1):
        subnet_unit.add_linear_layer(
          'mini_att_%02i' % i, 'mini_att_lstm_%02i' % i if variant == 'many' else 'mini_att_lstm', activation=None,
          n_out=512, l2=0.0001)

  def _create_prior_net(self, subnet_unit):
    self._add_prior_input(subnet_unit)

    linear_variant = self.prior_lm_opts['linear_variant']

    # used to make sure these inputs are not modified when modifying the "from" inputs list
    ignored_layers = {
      'end', 'output', 'target_embed', 'target_embed_raw', 'target_embed_weighted',
      '_att_query0', '_att_query', '_att_energy', '_att_weights', '_att_weights_drop'}

    for i in range(1, self.prior_lm_opts['dec_layers'] + 1):
      ignored_layers.add('mini_att_lstm_%02i' % i)
      ignored_layers.add('mini_att_%02i' % i)

    # do not change mini_att as input
    if self.prior_lm_opts.get('single_att_proj', False):
      ignored_layers.add('mini_att')

    for layer in list(self.asr_decoder.subnet_unit.get_net().keys()):
      if any([layer.endswith(l) for l in ignored_layers]):
        continue

      if linear_variant == 'no-linear' and layer.endswith('_att_linear'):
        continue  # just ignore

      d = copy.deepcopy(subnet_unit[layer])

      from_list = d.get('from', None)
      if from_list is None:
        continue
      if isinstance(from_list, str):
        from_list = [from_list]
      assert isinstance(from_list, list)
      new_from_list = []
      for e in from_list:
        if e.endswith('_att') and not e.endswith('_self_att_att'):  # a workaround to fix bug
          new_from_list.append('mini_att_%02i' % int(e.split('_')[2]))
        elif linear_variant == 'no-linear' and e.endswith('_att_linear'):
          # use estimated context vector directly instead of att_linear. att_linear layer ignored above
          new_from_list.append('mini_att_%02i' % int(e.split('_')[2]))
        elif (linear_variant == 'import-linear' or linear_variant == 'scratch-linear') and e.endswith('_att_linear'):
          # to be loaded from mini_lstm checkpoint
          new_from_list.append('mini_%s' % e)
        elif any([e.endswith(l) for l in ignored_layers]):
          new_from_list.append(e)
        else:
          new_from_list.append(('prev:' if 'prev' in e else '') + 'prior_%s' % e)

      d['from'] = new_from_list

      if (linear_variant == 'import-linear' or linear_variant == 'scratch-linear') and layer.endswith('_att_linear'):
        # to be loaded from mini_lstm checkpoint
        subnet_unit['mini_%s' % layer] = d
      else:
        subnet_unit['prior_%s' % layer] = d

    return 'prior_output_prob'


class TransformerMiniSelfAttDecoder(ILMDecoder):
  """
  Add (mini-)self-attention ILM for Transformer decoder.
  """

  def __init__(self, asr_decoder, prior_lm_opts):
    super(TransformerMiniSelfAttDecoder, self).__init__(asr_decoder)
    self.asr_decoder = asr_decoder
    self.prior_lm_opts = prior_lm_opts  # type: dict[str]

  def _create_prior_net(self, subnet_unit):
      # handled inside `TransformerDecoder` class.
      return 'prior_output_prob'


class ExternalLMDecoder:
  """
  Integrates an external LM decoder into an ASR decoder
  """

  def __init__(self, asr_decoder, ext_lm_opts, beam_size, dec_type, prior_lm_opts=None, length_normalization=True):
    self.asr_decoder = copy.deepcopy(asr_decoder)
    self.am_output_prob = self.asr_decoder.output_prob
    self.target = self.asr_decoder.target
    self.ext_lm_opts = ext_lm_opts
    self.beam_size = beam_size
    self.prior_lm_opts = prior_lm_opts
    self.dec_type = dec_type
    self.length_normalization = length_normalization

    self.network = None

  def _create_external_lm_net(self) -> dict:
    lm_net_out = ReturnnNetwork()

    ext_lm_subnet = self.ext_lm_opts['lm_subnet']
    ext_lm_scale = self.ext_lm_opts['lm_scale']

    assert isinstance(ext_lm_subnet, dict)
    is_recurrent = self.ext_lm_opts.get('is_recurrent', False)  # TODO: is this needed? we can always use subnet maybe
    if is_recurrent:
      lm_output_prob = self.ext_lm_opts['lm_output_prob_name']
      ext_lm_subnet[lm_output_prob]['target'] = self.target
      lm_net_out.update(ext_lm_subnet)  # just append
    else:
      ext_lm_model = self.ext_lm_opts.get('lm_model', None)
      if ext_lm_model:
        load_on_init = ext_lm_model
      else:
        assert 'load_on_init_opts' in self.ext_lm_opts, 'load_on_init opts or lm_model are missing for loading subnet.'
        assert 'filename' in self.ext_lm_opts['load_on_init_opts'], 'Checkpoint missing for loading subnet.'
        load_on_init = self.ext_lm_opts['load_on_init_opts']
      lm_net_out.add_subnetwork(
        'lm_output', 'prev:output', subnetwork_net=ext_lm_subnet, load_on_init=load_on_init)
      lm_output_prob = lm_net_out.add_activation_layer(
        'lm_output_prob', 'lm_output', activation='softmax', target=self.target)

    fusion_str = 'safe_log(source(0)) + {} * safe_log(source(1))'.format(ext_lm_scale)  # shallow fusion
    fusion_source = [self.am_output_prob, lm_output_prob]

    if self.prior_lm_opts:

      if self.dec_type == 'lstm':
        ilm_decoder = LSTMILMDecoder(self.asr_decoder, self.prior_lm_opts)
      elif self.dec_type == 'transformer':
        ilm_decoder = TransformerMiniSelfAttDecoder(self.asr_decoder, self.prior_lm_opts)
      else:
        raise ValueError('dec type: {} is not valid'.format(self.dec_type))

      ilm_decoder.create_network()  # add ILM
      fusion_str += ' - {} * safe_log(source(2))'.format(self.prior_lm_opts['scale'])
      fusion_source += [ilm_decoder.output_prob_name]

    lm_net_out.add_eval_layer('combo_output_prob', source=fusion_source, eval=fusion_str)
    if self.length_normalization:
      lm_net_out.add_choice_layer(
        'output', 'combo_output_prob', target=self.target, beam_size=self.beam_size, initial_output=0,
        input_type='log_prob')
    else:
      lm_net_out.add_choice_layer(
        'output', 'combo_output_prob', target=self.target, beam_size=self.beam_size, initial_output=0,
        input_type='log_prob', length_normalization=self.length_normalization)

    return lm_net_out.get_net()

  def create_network(self):
    lm_net = self._create_external_lm_net()
    self.asr_decoder.subnet_unit.update(lm_net)
    self.network = copy.deepcopy(self.asr_decoder.network)
