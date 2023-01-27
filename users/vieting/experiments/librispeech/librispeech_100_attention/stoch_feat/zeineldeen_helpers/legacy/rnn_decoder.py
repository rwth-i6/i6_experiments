from recipe.crnn.helpers.zeineldeen.network import ReturnnNetwork
import copy


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
               lstm_lm_dim=1000, loc_conv_att_filter_size=None, loc_conv_att_num_channels=None, density_ratio_opts=None,
               mwer=False, reduceout=True, att_num_heads=1, embed_weight=False, coverage_term_scale=None,
               ilmt_opts=None, trained_scales=False, remove_softmax_bias=False, relax_att_scale=None,
               ce_loss_scale=None, dec_state_no_label_ctx=False, add_no_label_ctx_s_to_output=False):
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

    self.embed_weight = embed_weight

    self.coverage_term_scale = coverage_term_scale

    self.ilmt_opts = ilmt_opts

    self.trained_scales = trained_scales

    self.remove_softmax_bias = remove_softmax_bias

    self.relax_att_scale = relax_att_scale

    self.ce_loss_scale = ce_loss_scale

    self.dec_state_no_label_ctx = dec_state_no_label_ctx
    self.add_no_label_ctx_s_to_output = add_no_label_ctx_s_to_output

    self.network = ReturnnNetwork()

  def _create_prior_net(self, subnet_unit: ReturnnNetwork, opts):
    prior_type = opts.get('type', 'zero')

    # fixed_ctx_vec_variants = ['zero', 'avg', 'train_avg_ctx', 'train_avg_enc', 'avg_zero', 'trained_vec']

    if prior_type == 'zero':  # set att context vector to zero
      prior_att_input = subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
    elif prior_type == 'avg':  # during search per utterance
      self.base_model.network.add_reduce_layer('encoder_mean', 'encoder', mode='mean', axes=['t'])  # [B, enc-dim]
      prior_att_input = 'base:encoder_mean'
    elif prior_type == 'train_avg_ctx':  # average all context vectors over training data
      prior_att_input = subnet_unit.add_constant_layer(
        'train_avg_ctx', value=opts['data'], with_batch_dim=True, dtype='float32')
    elif prior_type == 'train_avg_enc':  # average all encoder states over training data
      prior_att_input = subnet_unit.add_constant_layer(
        'train_avg_enc', value=opts['data'], with_batch_dim=True, dtype='float32')
    elif prior_type == 'mini_lstm':  # train a mini LM-like LSTM and use that as prior
      # example: lstmdim_100-l2_5e-05-recwd_0.0
      n_out = 50
      l2 = 0.0
      recwd = 0.0
      if opts.get('prefix_name', None):
        segs = opts['prefix_name'].split('-')
        for arg in segs:
          name, val = arg.split('_', 1)
          if name == 'lstmdim':
            n_out = int(val)
          elif name == 'l2':
            l2 = float(val)
          elif name == 'recwd':
            recwd = float(val)

      mini_lstm_inputs = opts.get('mini_lstm_inp', 'prev:target_embed').split('+')
      if len(mini_lstm_inputs) == 1:
        mini_lstm_inputs = mini_lstm_inputs[0]

      subnet_unit.add_rec_layer('mini_att_lstm', mini_lstm_inputs, n_out=n_out, l2=l2, rec_weight_dropout=recwd)
      prior_att_input = subnet_unit.add_linear_layer('mini_att', 'mini_att_lstm', activation=None, n_out=2048, l2=0.0001)
    elif prior_type == 'adaptive_ctx_vec':  # \hat{c}_i = FF(h_i)
      num_layers = opts.get('num_layers', 3)
      dim = opts.get('dim', 512)
      act = opts.get('act', 'relu')
      x = 's'
      for i in range(num_layers):
        x = subnet_unit.add_linear_layer('adaptive_att_%d' % i, x, n_out=dim, **opts.get('att_opts', {}))
        x = subnet_unit.add_activation_layer('adaptive_att_%d_%s' % (i, act), x, activation=act)
      prior_att_input = subnet_unit.add_linear_layer('adaptive_att', x, n_out=2048, **opts.get('att_opts', {}))
    elif prior_type == 'trained_vec':
      prior_att_input = subnet_unit.add_variable_layer('trained_vec_att_var', shape=[2048], L2=0.0001)
    elif prior_type == 'avg_zero':
      self.base_model.network.add_reduce_layer('encoder_mean', 'encoder', mode='mean', axes=['t'])  # [B, enc-dim]
      subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
      return
    elif prior_type == 'density_ratio':
      assert 'lm_subnet' in opts and 'lm_model' in opts
      return self._add_density_ratio(
        subnet_unit, lm_subnet=opts['lm_subnet'], lm_model=opts['lm_model'])
    else:
      raise ValueError('{} prior type is not supported'.format(prior_type))

    if prior_type != 'mini_lstm':
      is_first_frame = subnet_unit.add_compare_layer('is_first_frame', source=':i', kind='equal', value=0)
      zero_att = subnet_unit.add_eval_layer('zero_att', 'att', eval='tf.zeros_like(source(0))')
      prev_att = subnet_unit.add_switch_layer(
        'prev_att', condition=is_first_frame, true_from=zero_att, false_from=prior_att_input)
    else:
      prev_att = 'prev:' + prior_att_input

    assert prev_att is not None

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
            new_sources += [prev_att]
          else:
            new_sources += [prior_att_input]
        elif src in key_names:
          new_sources += [('prev:' if 'prev' in src else '') + 'prior_{}'.format(src.split(':')[-1])]
        else:
          new_sources += [src]
      d['from'] = new_sources
      subnet_unit['prior_{}'.format(key_name)] = d
    return 'prior_output_prob'

  def _create_ilmt_net(self, subnet_unit: ReturnnNetwork):
    self._create_prior_net(subnet_unit, self.ilmt_opts)  # add prior layers
    subnet_unit['prior_output_prob']['loss_opts'].update({'scale': self.ilmt_opts['scale']})

    # remove label smoothing
    if 'label_smoothing' in subnet_unit['prior_output_prob']['loss_opts'] and self.ilmt_opts.get('no_ilmt_lbs', False):
      subnet_unit['prior_output_prob']['loss_opts']['label_smoothing'] = 0

    if 'label_smoothing' in subnet_unit['prior_output_prob']['loss_opts'] and self.ilmt_opts.get('no_asr_lbs', False):
      subnet_unit['output_prob']['loss_opts']['label_smoothing'] = 0

    reuse_params_mapping = {
      'prior_s': {
        'lstm_cell/kernel': 'output/rec/s/rec/lstm_cell/kernel',
        'lstm_cell/bias': 'output/rec/s/rec/lstm_cell/bias'
      }
    }
    for name in ['prior_readout_in', 'prior_readout', 'prior_output_prob']:
      reuse_params_mapping[name] = {
        'W': 'output/rec/{}/W'.format(name[len('prior_'):]),
        'b': 'output/rec/{}/b'.format(name[len('prior_'):])
      }

    if self.ilmt_opts.get('share_params', False):
      from recipe.crnn.config import CodeWrapper
      layer_names = ['prior_s', 'prior_readout_in', 'prior_readout', 'prior_output_prob']
      for layer in layer_names:
        value = copy.deepcopy(subnet_unit[layer])
        map = reuse_params_mapping[layer]
        value['reuse_params'] = {'map': {}}
        for k, v in map.items():
          value['reuse_params']['map'][k] = {
            'custom': CodeWrapper("lambda **_kwargs: get_var('{}', _kwargs['shape'])".format(v))
          }
        if layer == 'prior_s':
          #value['reuse_params'] = {'auto_create_missing': True, 'reuse_layer': 's'}
          value['reuse_params']['auto_create_missing'] = True
        subnet_unit[layer] = value


  def _add_external_LM(self, subnet_unit: ReturnnNetwork, am_output_prob, prior_output_prob=None):
    ext_lm_scale = self.ext_lm_opts['lm_scale'] if not self.trained_scales else 'lm_scale'

    is_recurrent = self.ext_lm_opts.get('is_recurrent', False)

    log_lm_prob = False  # if lm_prob is already in log-space or not

    if 'gram_lm' in self.ext_lm_opts['name']:
      log_lm_prob = True  # already in log-space
      lm_output_prob = subnet_unit.add_kenlm_layer('lm_output_prob', **self.ext_lm_opts['kenlm_opts'])
    elif is_recurrent:
      ext_lm_subnet = self.ext_lm_opts['lm_subnet']
      assert isinstance(ext_lm_subnet, dict)

      lm_output_prob = self.ext_lm_opts['lm_output_prob_name']
      ext_lm_subnet[lm_output_prob]['target'] = self.target
      ext_lm_subnet[lm_output_prob]['loss'] = None  # TODO: is this needed?
      subnet_unit.update(ext_lm_subnet)  # just append
    else:
      ext_lm_subnet = self.ext_lm_opts['lm_subnet']
      assert isinstance(ext_lm_subnet, dict)

      ext_lm_model = self.ext_lm_opts['lm_model']
      subnet_unit.add_subnetwork(
        'lm_output', 'prev:output', subnetwork_net=ext_lm_subnet, load_on_init=ext_lm_model)
      lm_output_prob = subnet_unit.add_activation_layer(
        'lm_output_prob', 'lm_output', activation='softmax', target=self.target)

    fusion_str = 'safe_log(source(0)) + {} * '.format(ext_lm_scale)
    if log_lm_prob:
      fusion_str += 'source(1)'
    else:
      fusion_str += 'safe_log(source(1))'

    fusion_source = [am_output_prob, lm_output_prob]
    if prior_output_prob:
      fusion_source += [prior_output_prob]
      prior_scale = self.prior_lm_opts['scale'] if not self.trained_scales else 'prior_scale'
      fusion_str += ' - {} * safe_log(source(2))'.format(prior_scale)

    if self.coverage_term_scale:
      fusion_str += ' + {} * source({})'.format(self.coverage_term_scale, len(fusion_source))
      fusion_source += ['accum_coverage']

    if self.trained_scales:
      fusion_str = 'source(0) * safe_log(source(1)) + source(2) * safe_log(source(3))'
      fusion_source = ['am_scale', am_output_prob, 'lm_scale', lm_output_prob]
      if prior_output_prob:
        fusion_str += ' - source(4) * safe_log(source(5))'
        fusion_source += ['prior_scale', prior_output_prob]

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
    prefix_name = self.local_fusion_opts.get('prefix', 'local_fusion')
    with_label_smoothing = self.local_fusion_opts.get('with_label_smoothing', False)

    if self.local_fusion_opts['lm_type'] == 'n_gram':
      lm_output_prob = subnet.add_kenlm_layer(
        '{}_lm_output_prob'.format(prefix_name), **self.local_fusion_opts['kenlm_opts'])
    else:
      lm_subnet = self.local_fusion_opts['lm_subnet']
      lm_model = self.local_fusion_opts['lm_model']
      vocab_size = self.local_fusion_opts['vocab_size']

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
        '{}_lm_output_prob'.format(prefix_name), lm_subnet_name, activation='softmax', target=self.target)  # not in log-space

    # define new loss criteria
    eval_str = "self.network.get_config().typed_value('fusion_eval0_norm')(safe_log(source(0)), safe_log(source(1)))"
    if self.local_fusion_opts['lm_type'] == 'n_gram':
      eval_str = "self.network.get_config().typed_value('fusion_eval0_norm')(safe_log(source(0)), source(1))"
    combo_output_log_prob = subnet.add_eval_layer(
      'combo_output_log_prob', [am_output_prob, lm_output_prob], eval=eval_str)

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
      # TODO: this is not a good approach. if i want to load a checkpoint from a trained model without embed dropout,
      # i would need to remap variable name target_embed to target_embed0 to load target_embed0/W 
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
        'feedback_pad_right', pad_left, axes='s:0', padding=(0, (self.loc_conv_att_filter_size - 1) // 2), value=0)
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
      subnet_unit.add_linear_layer('weight_feedback', 'prev:accum_att_weights', n_out=self.enc_key_dim, with_bias=False)

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
      if self.relax_att_scale:
        subnet_unit.add_softmax_over_spatial_layer('att_weights0', 'energy')
        subnet_unit.add_length_layer('encoder_len', 'base:encoder', dtype='float32')  # [B]
        subnet_unit.add_eval_layer(
          'scaled_encoder_len', source=['encoder_len'], eval='{} / source(0)'.format(self.relax_att_scale))
        subnet_unit.add_eval_layer(
          'att_weights', source=['att_weights0', 'scaled_encoder_len'],
          eval='{} * source(0) + source(1)'.format(1 - self.relax_att_scale))
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

    if self.dec_state_no_label_ctx:
      lstm_inputs = ['prev:att']  # no label feedback

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
    if self.dec_state_no_label_ctx and self.add_lstm_lm:
      subnet_unit.add_linear_layer(
        'readout_in', ["lm_like_s", "prev:target_embed", "att"], n_out=self.dec_output_num_units)
      if self.add_no_label_ctx_s_to_output:
        subnet_unit.add_linear_layer(
          'readout_in', ["lm_like_s", "s", "prev:target_embed", "att"], n_out=self.dec_output_num_units)
    else:
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
      ce_loss_opts = {'label_smoothing': self.label_smoothing}
      if self.ce_loss_scale:
        ce_loss_opts['scale'] = self.ce_loss_scale
      output_prob = subnet_unit.add_softmax_layer(
        'output_prob', 'readout', l2=self.l2, loss='ce', loss_opts=ce_loss_opts,
        target=self.target, dropout=self.dropout)

    # do not load the bias
    if self.remove_softmax_bias:
      subnet_unit['output_prob']['with_bias'] = False

    # for prior LM estimation
    prior_output_prob = None
    if self.prior_lm_opts:
      prior_output_prob = self._create_prior_net(subnet_unit, self.prior_lm_opts)  # this require preload_from_files in config

    # Beam search
    # only support shallow fusion for now
    if self.ext_lm_opts:
      self._add_external_LM(subnet_unit, output_prob, prior_output_prob)
    else:
      if self.coverage_term_scale:
        output_prob = subnet_unit.add_eval_layer(
          'combo_output_prob', eval='safe_log(source(0)) + {} * source(1)'.format(self.coverage_term_scale),
          source=['output_prob', 'accum_coverage'])
        input_type = 'log_prob'
      else:
        output_prob = 'output_prob'
        input_type = None

      if self.length_norm:
        subnet_unit.add_choice_layer(
          'output', output_prob, target=self.target, beam_size=self.beam_size, initial_output=0, input_type=input_type)
      else:
        subnet_unit.add_choice_layer(
          'output', output_prob, target=self.target, beam_size=self.beam_size, initial_output=0,
          length_normalization=self.length_norm, input_type=input_type)

    if self.ilmt_opts:
      self._create_ilmt_net(subnet_unit)

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
