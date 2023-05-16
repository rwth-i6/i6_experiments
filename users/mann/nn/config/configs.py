__all__ = ['feed_forward_config', 'blstm_config']

from i6_core.returnn.config import ReturnnConfig

def feed_forward_config(num_features, network, learning_rate=1e-3, momentum=0.1, batch_size=500, use_tensorflow=False, **kwargs):
  result = { 'num_outputs' : { 'data': [num_features, 2] },
             'chunking'    : '50:50',
             'batch_size'  : batch_size,
             'start_batch' : 'auto',
             'start_epoch' : 'auto',

             'learning_rate'         : learning_rate,
             'learning_rate_control' : 'newbob_relative',

             'network' : network,
           }
  if not use_tensorflow:
    result['window'] = 1,
    result['loss']   = 'ce'
  else:
    result['use_tensorflow'] = True

  result.update(**kwargs)

  return result

def blstm_config(num_outputs, network, learning_rate=1e-3, batch_size=5000, max_seqs=40, **kwargs):
  result = {
    'num_outputs'    : { 'data': [num_outputs, 2] },
    'batch_size'     : batch_size,
    'max_seqs'       : max_seqs,
    'max_seq_length' : batch_size,
    'start_epoch'    : 'auto',
    'start_batch'    : 'auto',
    'window'         : 1,

    'learning_rate'         : learning_rate,
    'learning_rate_control' : 'newbob_relative',
    'gradient_clip'         : 10,

    'network' : network
  }
  result.update(**kwargs)

  return ReturnnConfig(result)
