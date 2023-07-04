## lm_config
import os
import copy

# ------------------------------ Recipes ------------------------------
from sisyphus import tk
Path = tk.Path

# only used for seq-training/full_sum training
# import recipe.crnn        as crnn

### construct RETURNN network layers on demand of training ###

def make_network():
  network = dict()
  fromList = ['data']
  return network, fromList

def add_loss_to_layer(network, name, loss, loss_opts=None, target=None, **kwargs):
  assert loss is not None
  network[name]['loss'] = loss
  if loss_opts:
    network[name]['loss_opts'] = loss_opts
  if target is not None:
    network[name]['target'] = target
  return network

def add_specaug_source_layer(network, name='source', nextLayers=['fwd_lstm_1', 'bwd_lstm_1']):
  network2 = copy.deepcopy(network)
  network2[name] = { 'class': 'eval',
                     'eval' : "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)"
                   }
  for layer in nextLayers:
    if not layer in network2: continue
    network2[layer]['from'] = [name]
  return network2, name 

def add_linear_layer(network: object, name: object, fromList: object, size: object, l2: object = 0.01, dropout: object = None, bias: object = None, activation: object = None, **kwargs) -> object:
  network[name] = { 'class'     : 'linear',
                    'n_out'     : size,
                    'from'      : fromList,
                    'activation': activation
                  }
  if l2 is not None:
    network[name]['L2'] = l2
  if dropout is not None:
    network[name]['dropout'] = dropout
  # bias is default true in RETURNN
  if bias is not None:
    network[name]['with_bias'] = bias
  if kwargs.get('random_norm_init', False):
    network[name]['forward_weights_init'] = 'random_normal_initializer(mean=0.0, stddev=0.1)'
  if kwargs.get('initial', None) is not None:
    network[name]['initial_output'] = kwargs.get('initial', None)
  if kwargs.get('loss', None) is not None:
    network = add_loss_to_layer(network, name, **kwargs)
  if kwargs.get('reuse_params', None) is not None:
    network[name]['reuse_params'] = kwargs.get('reuse_params', None)
  if not kwargs.get('trainable', True):
    network[name]['trainable'] = False
  if kwargs.get('out_type', None) is not None:
    network[name]['out_type'] = kwargs.get('out_type', None)
  # Note: this is not in the master RETURNN branch
  if kwargs.get('safe_embedding', False):
    network[name]['safe_embedding'] = True # 0-vectors for out-of-range ids (only for embedding)
  if kwargs.get('validate_indices', False):
    network[name]['validate_indices'] = True # round out-of-range ids to 0 (only for embedding)
  return network, name

def add_activation_layer(network, name, fromList, activation, **kwargs):
  network[name] = { 'class'     : 'activation',
                    'from'      : fromList,
                    'activation': activation
                  }
  if kwargs.get('loss', None) is not None:
    network = add_loss_to_layer(network, name, **kwargs)
  return network, name

def add_lstm_layer(network, name, fromList, size, l2=0.01, dropout=0.1, bidirectional=True, unit='nativelstm2', **kwargs):
  if bidirectional:
    layers = [('fwd_'+name,1), ('bwd_'+name,-1)]
  else: layers = [(name,1)]

  names = []
  for n,d in layers:
    network[n] = { 'class'     : 'rec',
                   'unit'      : unit,
                   'n_out'     : size,
                   'from'      : fromList,
                   'direction' : d,
                   'dropout'   : dropout,
                   'L2'        : l2
                 }
    if kwargs.get('drop_connect', None) is not None:
      network[n]['unit_opts'] = {'rec_weight_dropout': kwargs.get('drop_connect', None)}
    if kwargs.get('random_norm_init', False):
      network[n]['forward_weights_init'] = 'random_normal_initializer(mean=0.0, stddev=0.1)'
      network[n]['recurrent_weights_init'] = 'random_normal_initializer(mean=0.0, stddev=0.1)'
      network[n]['bias_init'] = 'random_normal_initializer(mean=0.0, stddev=0.1)'
    if not kwargs.get('trainable', True):
      network[n]['trainable'] = False
    names.append(n)

  if len(names) == 1: names = names[0]
  return network, names

def add_constant_layer(network, name, value, dtype='int32', with_batch_dim=True, **kwargs):
  network[name] = { 'class' : 'constant',
                    'value' : value,
                    'dtype' : dtype,
                    'with_batch_dim': with_batch_dim
                  }
  if kwargs.get('out_type', {}):
    network[name]['out_type'] = kwargs.get('out_type', {})
  if kwargs.get('initial', None) is not None:
    network[name]['initial_output'] = kwargs.get('initial', None)
  return network, name

def add_cast_layer(network, name, fromList, dtype='float32'):
  network[name] = { 'class' : 'cast',
                    'from'  : fromList,
                    'dtype' : dtype
                  }
  return network, name

def add_expand_dim_layer(network, name, fromList, axis, out_type=None):
  network[name] = { 'class' : 'expand_dims',
                    'from'  : fromList,
                    'axis'  : 2 # if int, then automatically batch major
                  }
  if out_type is not None:
    network[name]['out_type'] = out_type
  return network, name

def add_copy_layer(network, name, fromList, initial=None, loss=None, **kwargs):
  network[name] = { 'class' : 'copy',
                    'from'  : fromList
                  }
  if initial is not None:
    network[name]['initial_output'] = initial
  if loss is not None:
    network = add_loss_to_layer(network, name, loss, **kwargs)
  if kwargs.get('is_output', False):
    network[name]['is_output_layer'] = True
  if kwargs.get('dropout', None) is not None:
    network[name]['dropout'] = kwargs.get('dropout', None)
  return network, name

def add_compare_layer(network, name, fromList, value=None, kind='not_equal', initial=None):
  network[name] = { 'class' : 'compare',
                    'from'  : fromList,
                    'kind'  : kind
                  }
  if value is not None:
    network[name]['value'] = value
  if initial is not None:
    network[name]['initial_output'] = initial
  return network, name

def make_subnet(fromList, net):
  subnet = { 'class': 'subnetwork', 
             'from' : fromList,
             'subnetwork' : net
           } 
  return subnet

# masked computation 
def add_mask_layer(network: object, name: object, fromList: object, mask: object, unit: object = {'class': 'copy'}, **kwargs: object) -> object:
  network[name] = { 'class' : 'masked_computation',
                    'from'  : fromList,
                    'mask'  : mask,
                    'unit'  : unit,
                  }
  # more likely to be used in training where input is already masked elsewhere: directly use
  if kwargs.get('masked_from', None) is not None:
    network[name]['masked_from'] = kwargs.get('masked_from', None) 
  # heuristics likely not needed anymore, use pad layer to achieve the same
  if kwargs.get('initial', None) is not None:
    network[name]['unit']['initial_output'] = kwargs.get('initial', None)
  if kwargs.get('keep_last_for_prev', False):
    network[name]['keep_last_for_prev'] = True
  if kwargs.get('is_output', False):
    network[name]['is_output_layer'] = True
  return network, name

def add_unmask_layer(network, name, fromList, mask, **kwargs):
  network[name] = { 'class' : 'unmask', 
                    'from'  : fromList,
                    'mask'  : mask
                  }
  # do not use initial_output but directly the 1st frame of input for the first Fs
  if kwargs.get('skip_initial', True):
    network[name]['skip_initial'] = True
  return network, name

def add_padding_layer(network, name, fromList, axes='T', padding=(0,1), value=0, mode='constant', n_out=None, **kwargs):
  network[name] = { 'class'  : 'pad',
                    'from'   : fromList,
                    'axes'   : axes,
                    'padding': padding,
                    'value'  : value,
                    'mode'   : mode
                  }
  if n_out is not None:
    network[name]['n_out'] = n_out
  if kwargs.get('is_output', False):
    network[name]['is_output_layer'] = True
  if kwargs.get('initial', None) is not None:
    network[name]['initial_output'] = kwargs.get('initial', None)
  if kwargs.get('out_type', None) is not None:
    network[name]['out_type'] = kwargs.get('out_type', None)
  return network, name

def add_time_postfix_layer(network, name, fromList, postfix, repeat=1):
  network[name] = { 'class'   : 'postfix_in_time',
                    'from'    : fromList,
                    'postfix' : postfix,
                    'repeat'  : repeat
                  }
  return network, name

def add_axis_range_layer(network, name, fromList, axis='T', unbroadcast=True):
  network[name] = { 'class': 'range_in_axis',
                    'from' : fromList,
                    'axis' : axis, 
                    'unbroadcast' : unbroadcast
                  }
  return network, name

def add_shift_layer(network, name, fromList, axis='T', amount=1, pad=True, **kwargs):
  network[name] = { 'class'  : 'shift_axis',
                    'from'   : fromList,
                    'axis'   : axis,
                    'amount' : amount,
                    'pad'    : pad
                  }
  if kwargs.get('adjust_size', None) is not None:
    network[name]['adjust_size_info'] = kwargs.get('adjust_size', None)
  if kwargs.get('initial', None) is not None:
    network[name]['initial_output'] = kwargs.get('initial', None)
  return network, name

def add_seq_len_mask_layer(network, name, fromList, axis="T", mask_value=0):
  network[name] = { 'class': 'seq_len_mask',
                    'from' : fromList,
                    'axis' : axis,
                    'mask_value' : mask_value
                  }
  return network, name

def add_pool_layer(network, name, fromList, mode='max', pool_size=(2,), padding='same', **kwargs):
  network[name] = { 'class'     : 'pool',
                    'mode'      : mode,
                    'padding'   : padding,
                    'pool_size' : pool_size,
                    'from'      : fromList,
                    'trainable' : False
                  }
  return network, name

def add_reinterpret_data_layer(network, name, fromList, size_base=None, **kwargs):
  network[name] = { 'class'     : 'reinterpret_data',
                    'from'      : fromList
                  }
  if kwargs.get('loss', None) is not None:
    network = add_loss_to_layer(network, name, **kwargs)
  if size_base is not None:
    network[name]['size_base'] = size_base
  if kwargs.get('enforce_time_major', False):
    network[name]['enforce_time_major'] = True
  if kwargs.get('set_sparse', None) is not None:
    network[name]['set_sparse'] = kwargs.get('set_sparse', None)
  if kwargs.get('set_sparse_dim', None) is not None:
    network[name]['set_sparse_dim'] = kwargs.get('set_sparse_dim', None)
  if kwargs.get('is_output', False):
    network[name]['is_output_layer'] = True
  return network, name

def add_window_layer(network, name, fromList, winSize, winLeft, **kwargs):
  network[name] = { 'class'       : 'window',
                    'from'        : fromList,
                    'window_size' : winSize,
                    'window_left' : winLeft
                    # default along time axis and 0 padding (also works inside rec loop)
                  }
  return network, name

def add_merge_dim_layer(network, name, fromList, axes='except_time', **kwargs):
  network[name] = { 'class' : 'merge_dims',
                    'from'  : fromList,
                    'axes'  : axes
                  }
  return network, name

def add_split_dim_layer(network, name, fromList, axis, dims, **kwargs):
  network[name] = { 'class' : 'split_dims',
                    'from'  : fromList,
                    'axis'  : axis,
                    'dims'  : dims
                  }
  return network, name

def add_slice_layer(network, name, fromList, axis='F', start=None, end=None, step=None):
  network[name] = { 'class' : 'slice',
                    'from'  : fromList,
                    'axis'  : axis, 
                    'slice_start' : start,
                    'slice_end'   : end,
                    'slice_step'  : step
                  }
  return network, name

def add_squeeze_layer(network, name, fromList, axis, enforce_batch_dim_axis=None):
  network[name] = { 'class' : 'squeeze',
                    'from'  : fromList,
                    'axis'  : axis
                  }
  if enforce_batch_dim_axis is not None:
    network[name]['enforce_batch_dim_axis'] = enforce_batch_dim_axis
  return network, name

def add_layer_norm_layer(network, name, fromList):
  network[name] = { 'class' : 'layer_norm',
                    'from'  : fromList
                  }
  return network, name

def add_batch_norm_layer(network, name, fromList, **kwargs):
  network[name] = { 'class' : 'batch_norm',
                    'from'  : fromList
                  }
  # RETURNN defaults wrong
  if kwargs.get('fix_settings', False):
    network[name].update({
      'momentum': 0.1,
      'epsilon' : 1e-5,
      # otherwise eval may be batch-size and utterance-order dependent !
      'update_sample_only_in_training': True,
      'delay_sample_update': True
    })
  # freeze batch norm running average in training: consistent with testing
  if kwargs.get('freeze_average', False):
    network[name]['momentum'] = 0.0
    network[name]['use_sample'] = 1.0
  return network, name

# eval layer is a also special case of combine layer, but we distinguish them explicitly here
# and only restricted to the 'kind' usage
def add_combine_layer(network, name, fromList, kind='add', **kwargs):
  network[name] = { 'class' : 'combine',
                    'from'  : fromList,
                    'kind'  : kind
                  }
  if kwargs.get('activation', None) is not None:
    network[name]['activation'] = kwargs.get('activation', None)
  if kwargs.get('with_bias', None) is not None:
    network[name]['with_bias'] = kwargs.get('with_bias', None)
  if kwargs.get('n_out', None) is not None:
    network[name]['n_out'] = kwargs.get('n_out', None)
  if kwargs.get('is_output', False):
    network[name]['is_output_layer'] = True
  return network, name

# Note: RETURNN source(i, auto_convert=True, enforce_batch_major=False, as_data=False)
def add_eval_layer(network, name, fromList, eval_str, **kwargs):
  network[name] = { 'class' : 'eval',
                    'from'  : fromList,
                    'eval'  : eval_str
                  }
  if kwargs.get('loss', None) is not None:
    network = add_loss_to_layer(network, name, **kwargs)
  if kwargs.get('initial', None) is not None:
    network[name]['initial_output'] = kwargs.get('initial', None)
  if kwargs.get('n_out', None) is not None:
    network[name]['n_out'] = kwargs.get('n_out', None)
  if kwargs.get('out_type', None) is not None:
    network[name]['out_type'] = kwargs.get('out_type', None)
  return network, name

def add_variable_layer(network, name, shape, **kwargs):
  network[name] = { 'class' : 'variable',
                    'shape' : shape
                  }
  return network, name

# generic attention
def add_attention_layer(network, name, base, weights, **kwargs):
  network[name] = { 'class'  : 'generic_attention',
                    'base'   : base,
                    'weights': weights
                  }
  return network, name

def add_spatial_softmax_layer(network, name, fromList, **kwargs):
  network[name] = { 'class' : 'softmax_over_spatial',
                    'from'  : fromList
                  }
  return network, name

def add_rel_pos_encoding_layer(network, name, fromList, n_out, clipping=64, **kwargs):
  network[name] = { 'class'   : 'relative_positional_encoding',
                    'from'    : fromList,
                    'n_out'   : n_out,
                    'clipping': clipping
                  }
  return network, name

def add_self_attention_layer(network, name, fromList, n_out, num_heads, total_key_dim, key_shift=None, attention_dropout=None, **kwargs): 
  network[name] = { 'class'        : 'self_attention',
                    'from'         : fromList,
                    'n_out'        : n_out,
                    'num_heads'    : num_heads,
                    'total_key_dim': total_key_dim
                  }
  if key_shift is not None:
    network[name]['key_shift'] = key_shift
  if attention_dropout is not None:
    network[name]['attention_dropout'] = attention_dropout
  return network, name

def add_conv_layer(network, name, fromList, n_out, filter_size, padding='VALID', l2=0.01, bias=True, activation=None, **kwargs):
  network[name] = { 'class'      : 'conv',
                    'from'       : fromList,
                    'n_out'      : n_out, 
                    'filter_size': filter_size, 
                    'padding'    : padding,
                    'with_bias'  : bias,
                    'activation' : activation
                  }
  if l2 is not None:
    network[name]['L2'] = l2
  if kwargs.get('strides', None) is not None:
    network[name]['strides'] = kwargs.get('strides', None)
  if kwargs.get('groups', None) is not None:
    network[name]['groups'] = kwargs.get('groups', None)
  if not kwargs.get('trainable', True):
    network[name]['trainable'] = False
  return network, name

def add_gating_layer(network, name, fromList, activation=None, gate_activation='sigmoid', **kwargs):
  network[name] = { 'class'      : 'gating',
                    'from'       : fromList,
                    'activation' : activation,
                    'gate_activation': gate_activation
                  }
  return network, name

def add_reduce_layer(network, name, fromList, mode='mean', axes='T', keep_dims=False, **kwargs):
  network[name] = { 'class'    : 'reduce',
                    'from'     : fromList,
                    'mode'     : mode,
                    'axes'     : axes,
                    'keep_dims': keep_dims
                  }
  return network, name

def add_reduce_out_layer(network, name, fromList, mode='max', num_pieces=2, **kwargs):
  network[name] = { 'class'     : 'reduce_out',
                    'from'      : fromList,
                    'mode'      : mode,
                    'num_pieces': num_pieces
                  }
  return network, name

# Convolution block
def add_conv_block(network, fromList, conv_layers, conv_filter, conv_size, pool_size=None, name_prefix='conv', **kwargs):
  network, fromList = add_split_dim_layer(network, 'conv_source', fromList, axis='F', dims=(-1,1))
  for idx in range(conv_layers):
    name = name_prefix + '_' + str(idx+1)
    network, fromList = add_conv_layer(network, name, fromList, conv_size, conv_filter, padding='same', **kwargs)
    if pool_size is not None:
      name += '_pool'
      if isinstance(pool_size, list):
        assert idx < len(pool_size)
        pool = pool_size[idx]
      else: pool = pool_size
      assert isinstance(pool, tuple)
      if any([p > 1 for p in pool]):
        network, fromList = add_pool_layer(network, name, fromList, pool_size=pool)
  network, fromList = add_merge_dim_layer(network, 'conv_merged', fromList, axes='static')
  return network, fromList

# BLSTM encoder with optional max-pool subsampling
def build_encoder_network(num_layers=6, size=512, max_pool=[], **kwargs):
  network, fromList = make_network()
  # Convolution layers (no subsampling) 
  if kwargs.pop('initial_convolution', False):
    # TODO no pooling on feature dim ? (correlation is already low)
    conv_layers, conv_filter, conv_size, pool = kwargs.pop('convolution_layers', (2, (3, 3), 32, (1, 2)))
    network, fromList = add_conv_block(network, fromList, conv_layers, conv_filter, conv_size, pool_size=pool, **kwargs)
  # BLSTM layers
  for idx in range(num_layers):
    name = 'lstm_'+str(idx+1)
    network, fromList = add_lstm_layer(network, name, fromList, size, **kwargs)
    if max_pool and idx < len(max_pool) and max_pool[idx] > 1:
      name = 'max_pool_'+str(idx+1)
      network, fromList = add_pool_layer(network, name, fromList, pool_size=(max_pool[idx],))
  return network, fromList

# Conformer encoder TODO freeze encoder: pass trainable False
def add_conformer_block(network, name, fromList, size, dropout, l2, **kwargs):
  # feed-forward module
  def add_ff_module(net, n, fin):
    net, fout = add_layer_norm_layer(net, n+'_ln', fin)
    net, fout = add_linear_layer(net, n+'_linear_swish', fout, size*4, l2=l2, activation='swish')
    net, fout = add_linear_layer(net, n+'_dropout_linear', fout, size, l2=l2, dropout=dropout)
    net, fout = add_copy_layer(net, n+'_dropout', fout, dropout=dropout)
    net, fout = add_eval_layer(net, n+'_half_res_add', [fout, fin], "0.5 * source(0) + source(1)")
    return net, fout

  # multi-head self-attention module
  def add_mhsa_module(net, n, fin, heads, posEncSize, posEncClip, posEnc=True):
    net, fout = add_layer_norm_layer(net, n+'_ln', fin)
    if posEnc:
      net, fpos = add_rel_pos_encoding_layer(net, n+'_relpos_encoding', fout, posEncSize, clipping=posEncClip)
    else: fpos = None 
    net, fout = add_self_attention_layer(net, n+'_self_attention', fout, size, heads, size, key_shift=fpos, attention_dropout=dropout)
    net, fout = add_linear_layer(net, n+'_att_linear', fout, size, l2=l2, bias=False)
    net, fout = add_copy_layer(net, n+'_dropout', fout, dropout=dropout)
    net, fout = add_combine_layer(net, n+'_res_add', [fout, fin])
    return net, fout

  # convolution module
  def add_conv_module(net, n, fin, filterSize, bnFix, bnFreeze, bn2ln):
    net, fout = add_layer_norm_layer(net, n+'_ln', fin)
    # glu weights merged into pointwise conv, i.e. linear layer
    net, fout = add_linear_layer(net, n+'_pointwise_conv_1', fout, size*2, l2=l2)
    net, fout = add_gating_layer(net, n+'_glu', fout)
    net, fout = add_conv_layer(net, n+'_depthwise_conv', fout, size, filterSize, padding='same', l2=l2, groups=size)
    if bn2ln:
      net, fout = add_layer_norm_layer(net, n+'_bn2ln', fout)
    else:
      net, fout = add_batch_norm_layer(net, n+'_bn', fout, fix_settings=bnFix, freeze_average=bnFreeze)
    net, fout = add_activation_layer(net, n+'_swish', fout, 'swish')
    net, fout = add_linear_layer(net, n+'_pointwise_conv_2', fout, size, l2=l2)
    net, fout = add_copy_layer(net, n+'_dropout', fout, dropout=dropout)
    net, fout = add_combine_layer(net, n+'_res_add', [fout, fin])
    return net, fout

  network, fList = add_ff_module(network, name+'_ffmod_1', fromList)

  mhsa_args = {
    'heads'     : kwargs.get('num_att_heads', 8),
    'posEncSize': kwargs.get('pos_enc_size', 64), 
    'posEncClip': kwargs.get('pos_enc_clip', 64), # default clipping 16 in RETURNN
    'posEnc'    : kwargs.get('pos_encoding',True)
  }
  conv_args = {
    'filterSize': kwargs.get('conv_filter_size', (32,)),
    'bnFix'     : kwargs.get('batch_norm_fix', False),
    'bnFreeze'  : kwargs.get('batch_norm_freeze', False),
    'bn2ln'     : kwargs.get('batch_norm_to_layer_norm', False)
  }
  if kwargs.get('switch_conv_mhsa_module', False):
    network, fList = add_conv_module(network, name+'_conv_mod', fList, **conv_args)
    network, fList = add_mhsa_module(network, name+'_mhsa_mod', fList, **mhsa_args)
  else: 
    network, fList = add_mhsa_module(network, name+'_mhsa_mod', fList, **mhsa_args) 
    network, fList = add_conv_module(network, name+'_conv_mod', fList, **conv_args)

  network, fList = add_ff_module(network, name+'_ffmod_2', fList)
  network, fList = add_layer_norm_layer(network, name+'_output', fList)
  return network, fList 

def build_conformer_encoder(num_blocks=12, size=512, dropout=0.1, l2=0.0001, max_pool=[], **kwargs):
  network, fromList = make_network()
  # Input block
  if kwargs.get('initial_convolution', True): 
    # vgg conv with subsampling 4
    if kwargs.get('vgg_conv', True):
      network, fromList = add_conv_block(network, fromList, 1, (3, 3), 32, pool_size=(1, 2), activation='swish', **kwargs)
      stride1, stride2 = kwargs.get('vgg_conv_strides', (2,2))
      network, fList = add_conv_layer(network, 'conv_2', network[fromList]['from'], 64, (3, 3), padding='same', strides=(stride1,1), activation='swish', **kwargs)
      network, fList = add_conv_layer(network, 'conv_3', fList, 64, (3, 3), padding='same', strides=(stride2,1), activation='swish', **kwargs)
      network[fromList]['from'] = fList
    elif kwargs.get('stride_subsampling', False):
      conv_layers, conv_filter, conv_size, strides = kwargs.pop('convolution_layers', (2, (3, 3), 32, [2, 2]))
      network, fromList = add_conv_block(network, fromList, conv_layers, conv_filter, conv_size, strides=strides, **kwargs)
    else: # max_pool subsampling
      conv_layers, conv_filter, conv_size, pool = kwargs.pop('convolution_layers', (2, (3, 3), 32, (1, 2)))
      network, fromList = add_conv_block(network, fromList, conv_layers, conv_filter, conv_size, pool_size=pool, **kwargs)
    assert not max_pool
  elif kwargs.get('initial_blstm', False): # BLSTM with subsampling 4
    layers, uniSize, pool = kwargs.pop('blstm_layers', (2, 512, [2,2]))
    network, fromList = build_encoder_network(num_layers=layers, size=uniSize, max_pool=pool, dropout=dropout, l2=l2, **kwargs)
    assert not max_pool
  network, fromList = add_linear_layer(network, 'input_linear', fromList, size, l2=l2, bias=False)
  network, fromList = add_copy_layer(network, 'input_dropout', fromList, dropout=dropout)

  # Conformer blocks 
  for idx in range(num_blocks):
    name = 'conformer_'+str(idx+1)
    network, fromList = add_conformer_block(network, name, fromList, size, dropout, l2, **kwargs)
    # also allow subsampling between conformer blocks
    if max_pool and idx < len(max_pool) and max_pool[idx] > 1:
      name += '_max_pool'
      network, fromList = add_pool_layer(network, name, fromList, pool_size=(max_pool[idx],))
  return network, fromList

# -- output and loss --
def add_loss_layer(network, name, fromList, loss='ce', **kwargs):
  network[name] = { 'class'  : 'loss',
                    'from'   : fromList,
                    'loss_'  : loss
                  }
  if kwargs.get('target', None) is not None:
    network[name]['target_'] = kwargs.get('target', None)
  if kwargs.get('loss_opts', None) is not None:
    network[name]['loss_opts_'] = kwargs.get('loss_opts', None)
  return network, name

def add_output_layer(network, fromList, name='output', loss='ce', loss_opts=None, cls='softmax', **kwargs):
  network[name] = { 'class' : cls,
                    'from'  : fromList 
                  }
  if loss is not None:
    network = add_loss_to_layer(network, name, loss, loss_opts=loss_opts, **kwargs)
  else:
    n_out = kwargs.get('n_out', None)
    assert n_out is not None, 'either loss or n_out need to be given'
    network[name]['n_out'] = n_out
    network[name]['is_output_layer'] = True

  if kwargs.get('random_norm_init', False):
    network[name]['forward_weights_init'] = 'random_normal_initializer(mean=0.0, stddev=0.1)'
  if kwargs.get('dropout', None) is not None:
    network[name]['dropout'] = kwargs.get('dropout', None)
  if kwargs.get('loss_scale', None) is not None:
    network[name]['loss_scale'] = kwargs.get('loss_scale', None)
  if kwargs.get('activation', None) is not None:
    network[name]['class'] = 'linear'
    network[name]['activation'] = kwargs.get('activation', None)
  if kwargs.get('reuse_params', None) is not None:
    network[name]['reuse_params'] = kwargs.get('reuse_params', None)
  network[name].update(kwargs.get('extra_args',{}))
  return network

def add_sMBR_output(inNetwork, name='output_ac', output='output', ce_smooth=0.1, **kwargs):
  network = copy.deepcopy(inNetwork)
  network[output]['loss_scale'] = ce_smooth
  network[name] = { 'class' : 'copy',
                    'from'  : output,
                    'loss'  : 'sprint',
                    'loss_scale': 1 - ce_smooth,
                    'loss_opts' : { 'sprint_opts' : crnn.CustomCRNNSprintTrainingJob.create_sprint_loss_opts(loss_mode='sMBR', num_sprint_instance=1) }
                  }
  return network                 

# full-sum training using sprint FSA (so far only fast_bw loss)
def add_full_sum_output_layer(network, fromList, num_classes, loss='fast_bw', name='output', **kwargs):
  output_args = { 'name'      : name,
                  'loss'      : loss,
                  'loss_opts' : { 'sprint_opts' : crnn.CustomCRNNSprintTrainingJob.create_sprint_loss_opts(**kwargs),
                                  'tdp_scale'   : kwargs.get('tdp_scale', 0.0)
                                },
                  'extra_args': { 'target': None,
                                  'n_out' : num_classes # no target to infer output size
                                }
                }
  return add_output_layer(network, fromList, **output_args)

# decoder output layer using rec-layer unit (including prediction and joint network)
def add_decoder_output_rec_layer(network, fromList, recUnit, optimize_move_layers_out=None, **kwargs):
  network = copy.deepcopy(network)
  network['output'] = { 'class'    : 'rec',
                        'from'     : fromList,
                        # only relevant for beam_search: e.g. determine length by targets
                        'cheating' : False,
                        'target'   : kwargs.get('target', 'classes'),
                        'unit'     : recUnit
                      }
  if optimize_move_layers_out is not None:
    network['output']['optimize_move_layers_out'] = optimize_move_layers_out
  if kwargs.get('max_seq_len', None) is not None:
    network['output']['max_seq_len'] = kwargs.get('max_seq_len', None)
  return network

def add_choice_layer(network, name='output_choice', fromList=['output'], initial=0, beam=1, **kwargs):
  network[name] = { 'class'          : 'choice',
                    'target'         : kwargs.get('target', 'classes'),
                    'from'           : fromList,
                    'initial_output' : initial,
                    # only relevant for beam_search: e.g. task='search'
                    'cheating'       : 'False', # include targets in the beam
                    'beam_size'      : beam 
                  }
  if kwargs.get('scheduled_sampling', False):
    network[name]['scheduled_sampling'] = kwargs.get('scheduled_sampling', False)
  if kwargs.get('input_type', None) is not None:
    network[name]['input_type'] = kwargs.get('input_type', None)
  # Note: either/none of the following is needed for recognition
  # old compile_tf_graph
  if kwargs.get('is_stochastic_var', None) is not None:
    network[name]['is_stochastic_var'] = kwargs.get('is_stochastic_var', None)
  # new compile_tf_graph
  if kwargs.get('score_dependent', None) is not None:
    network[name]['score_dependent'] = kwargs.get('score_dependent', None)
  return network

def make_recog_rec_network(trainRecNetwork, removeList=[], update={}, recRemoveList=[], recUpdate={}):
  # Note: can not add new layers
  def modify(net, toRemove, toUpdate):
    network = copy.deepcopy(net)
    for lname in net.keys():
      # name pattern match: removal
      removed = False
      for rk in toRemove:
        if rk in lname:
          del network[lname]
          removed = True
          break
      if removed: continue
      # name match: dict update
      if lname in toUpdate:
        network[lname].update( toUpdate[lname] )
    return network
  # apply change
  recogRecNetwork = modify(trainRecNetwork, removeList, update)
  if recRemoveList or recUpdate:
    assert recogRecNetwork['output']['class'] == 'rec'
    recUnit = modify(recogRecNetwork['output']['unit'], recRemoveList, recUpdate)
    recogRecNetwork['output']['unit'] = recUnit
  return recogRecNetwork

# simple zero-encoder estimated internal LM TODO add more
def make_internal_LM_rec_network(recUnit, name, scale, lm_output, num_classes, blankIndex=0, posterior='output'):
  assert blankIndex == 0, 'assume blank index 0'
  assert posterior in recUnit
  recUnit[posterior].update({
    'class'     : 'linear',
    'activation': 'log_softmax'
  })
  # TODO exclude bias ?
  recUnit, fList = add_linear_layer(recUnit, 'intLM_logits', lm_output, num_classes, reuse_params=posterior)
  recUnit, fList = add_slice_layer(recUnit, 'intLM_logits_noBlank', fList, start=1)
  recUnit, fList = add_activation_layer(recUnit, 'intLM_softmax', fList, 'log_softmax')
  recUnit, fList = add_padding_layer(recUnit, 'intLM_prior', fList, axes='F', value=0, padding=(1,0), n_out=num_classes)
  # log(posterior) - alpha * log(prior)
  recUnit, fList = add_eval_layer(recUnit, name, [posterior, fList], 'source(0) - %s * source(1)' %(str(scale)) )
  return recUnit


## ----------------------- extra python code ----------------------- ##
# SpecAugment #
def get_spec_augment_mask_python(codeFile=None, max_time_num=6, max_time=5, max_feature_num=4, max_feature=5, conservatvie_step=2000, feature_limit=None, customRep={}):
  path = os.path.dirname(os.path.abspath(__file__))
  if codeFile is None:
    if feature_limit is not None:
      codeFile = os.path.join(path, 'spec_augment_mask_flimit.py')
    else:
      codeFile = os.path.join(path, 'spec_augment_mask.py')
  elif codeFile in os.listdir(path):
    codeFile = os.path.join(path, codeFile)
  with open(codeFile,'r') as f:
    python_code = f.read()

  python_code = python_code.replace('max_time_num = 6', "max_time_num = %d" %max_time_num)
  python_code = python_code.replace('max_time = 5', "max_time = %d" %max_time)
  python_code = python_code.replace('max_feature_num = 4', "max_feature_num = %d" %max_feature_num)
  python_code = python_code.replace('max_feature = 5', "max_feature = %d" %max_feature)
  python_code = python_code.replace('conservatvie_step = 2000', "conservatvie_step = %d" %conservatvie_step)

  if feature_limit is not None:
    assert isinstance(feature_limit, int)
    python_code = python_code.replace('feature_limit = 80', "feature_limit = %d" %feature_limit)

  for old, new in customRep.items():
    python_code = python_code.replace("%s" %old, "%s" %new)
  return python_code

def get_extern_data_python(codeFile=None, nInput=50, nOutput=40):
  if codeFile is None:
    path = os.path.dirname(os.path.abspath(__file__))
    codeFile = os.path.join(path, 'extern_data.py')
  with open(codeFile,'r') as f:
    python_code = f.read()

  python_code = python_code.replace("nInput", str(nInput))
  python_code = python_code.replace("nOutput", str(nOutput))
  return python_code

# custom pretrain construction with down-sampling #
def get_pretrain_python(codeFile=None, repetitions='1', customRep={}):
  path = os.path.dirname(os.path.abspath(__file__))
  if codeFile is None:
    codeFile = os.path.join(path, 'pretrain.py')
  elif codeFile in os.listdir(path):
    codeFile = os.path.join(path, codeFile)
  with open(codeFile,'r') as f:
    python_code = f.read()

  if not isinstance(repetitions, str):
    repetitions = str(repetitions)
  if not repetitions == '1':
    python_code = python_code.replace("'repetitions': 1", "'repetitions': %s" %repetitions)
  for old, new in customRep.items():
    python_code = python_code.replace("%s" %old, "%s" %new)
  return python_code

def get_segmental_loss_python(codeFile=None, time_axis=None):
  path = os.path.dirname(os.path.abspath(__file__))
  if codeFile is None:
    codeFile = os.path.join(path, 'segmental_loss.py')
  elif codeFile in os.listdir(path):
    codeFile = os.path.join(path, codeFile)
  with open(codeFile,'r') as f:
    python_code = f.read()
  if time_axis is not None:
    python_code = python_code.replace('axis=0', 'axis=%d' %time_axis)
  return python_code

def get_extra_python(codeFile, customRep={}):
  assert codeFile is not None
  path = os.path.dirname(os.path.abspath(__file__))
  codeFile = os.path.join(path, codeFile)
  with open(codeFile,'r') as f:
    python_code = f.read()
  for old, new in customRep.items():
    python_code = python_code.replace("%s" %old, "%s" %new)
  return python_code









