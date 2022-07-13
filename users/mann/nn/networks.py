__all__ = ['mlp_network', 'blstm_network', 'tdnn_network']

import copy

def mlp_network(layers=None, activation='sigmoid', dropout=0.1, l2=0.0, feature_window=1):
  if layers is None:
    layers = [1000]
  num_layers = len(layers)
  assert num_layers > 0

  result = { 'output' : { 'class' : 'softmax', 'from' : ['hidden%d' % num_layers] } }

  input_layer = 'data'
  if feature_window > 1:
    result['feature_window'] = { 'class': 'window', 'window': feature_window, 'from': ['data'] }
    input_layer = 'feature_window'

  for l, size in enumerate(layers):
    l += 1  # start counting from 1
    result['hidden%d' % l] = { 'class'      : 'hidden',
                               'n_out'      : size,
                               'activation' : activation,
                               'dropout'    : dropout,
                               'from'       : [('hidden%d' % (l - 1)) if l > 1 else input_layer] }
    if l2 > 0.0:
      result['hidden%d' % l]['L2'] = l2

  return result

def blstm_network(layers=None, dropout=0.1, l2=0.0):
  if layers is None:
    layers = [1000]
  num_layers = len(layers)
  assert num_layers > 0

  result = { 'output' : { 'class' : 'softmax', 'from' : ['fwd_%d' % num_layers, 'bwd_%d' % num_layers] } }

  for l, size in enumerate(layers):
    l += 1  # start counting from 1
    for direction, name in [(1, 'fwd'), (-1, 'bwd')]:
      if l == 1:
        from_layers = ['data']
      else:
        from_layers = ['fwd_%d' % (l - 1), 'bwd_%d' % (l - 1)]
      result['%s_%d' % (name, l)] = { 'class'     : 'rec',
                                      'unit'      : 'lstmp',
                                      'direction' : direction,
                                      'n_out'     : size,
                                      'dropout'   : dropout,
                                      'L2'        : l2,
                                      'from'      : from_layers }

  return result

def tdnn_network(layers, filters, dilation=None, padding="same", activation='relu', dropout=0.1, l2=0.01, batch_norm=True):
    # These days we would have a larger number of layers.
    # All filter_size should be 3.
    # Also, regarding dilation:
    #   If the ultimate frame rate is 30ms, you should never use dilation.  Instead have a few (e.g. 3) layers, then a layer
    # with subsampling/stride = 3, then a bunch of layers (e.g. 7).  No dilation.
    #   If the ultimate frame rate is 10ms, you can start with 3 layers with no dilation, then have e.g. 7 layers with dilation=3.
    assert len(layers) == len(filters)
    modes = ["same", "valid", "causal"]
    # modes = ["SAME", "VALID", "CAUSAL"]
    # padding = padding.upper()
    # print(padding)
    assert  padding in modes \
        or (len(padding) == len(layers) and all(p in modes for p in padding))
    num_layers = len(filters)
    assert num_layers > 0

    if dilation is None:
        dilation = [3] * num_layers
        dilation[0] = 1
        dilation[1] = 2
    
    if isinstance(padding, str):
        padding = [padding] * len(layers)


    result = {'output': {'class': 'softmax', 'loss': 'ce', 'from': ['tdnn_%d' % num_layers]}}

    input_layer = 'data'

    for l, (width, size, pad) in enumerate(zip(layers, filters, padding), 1):
    # l += 1  # start counting from 1
        result["tdnn_%s" % l] = {
            "class": "conv",
            "n_out": width,
            "activation": activation,
            "with_bias": True,
            "filter_size": (size,),
            "padding": pad,
            "strides": 1,
            "dilation_rate": dilation[l-1],
            "batch_norm": batch_norm,
            "dropout": dropout,
            'from'      : [('tdnn_%d' % (l - 1)) if l > 1 else input_layer]}
    return result


