#!returnn/rnn.py
# kate: syntax python;

import os
from returnn.util.basic import get_login_username
from returnn.tf.util.data import DimensionTag

demo_name, _ = os.path.splitext(__file__)
print("Hello, experiment: %s" % demo_name)

# task
use_tensorflow = True
task = "train"

# data
train = {
  "class": "DummyDatasetMultipleSequenceLength",
  "num_seqs": 10,
  "input_dim": 20,
  "output_dim": 5,
  "seq_len": {
    "data": 20,
    "classes": 20
    }
  }
# num_inputs = 1
# num_outputs = 4

extern_data = {
  "data": {
    "dim": 20,
    "same_dim_tags_as": {
      "t": DimensionTag(kind=DimensionTag.Types.Spatial, description='time', dimension=None)
      }
    },
  "classes": {
    "dim": 5,
    "sparse": True
    }
  }

dev = train.copy()
dev.update({
             "num_seqs": train["num_seqs"] // 10,
             "fixed_random_seed": 42
             })

att_t_tag = DimensionTag(kind=DimensionTag.Types.Spatial, description="att_t", dimension=None)

# network
# (also defined by num_inputs & num_outputs)
network = {
  # 'source': {'class': 'eval',
  #                     'eval': "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
  #                     'from': 'log_mel_features'},
  'source0': {
    'class': 'split_dims',
    'axis': 'F',
    'dims': (-1, 1),
    'from': 'data:data'
    },
  'conv0': {
    'class': 'conv',
    'from': 'source0',
    'padding': 'same',
    'filter_size': (3, 3),
    'n_out': 32,
    'activation': 'relu',
    'with_bias': True
    },
  'conv0p': {
    'class': 'pool',
    'from': 'conv0',
    'pool_size': (1, 2),
    'mode': 'max',
    'trainable': False,
    'padding': 'same'
    },
  'conv_out': {
    'class': 'copy',
    'from': 'conv0p'
    },
  'subsample_conv0': {
    'class': 'conv',
    'from': 'conv_out',
    'padding': 'same',
    'filter_size': (3, 3),
    'n_out': 64,
    'activation': 'relu',
    'with_bias': True,
    'strides': (3, 1)
    },
  'subsample_conv1': {
    'class': 'conv',
    'from': 'subsample_conv0',
    'padding': 'same',
    'filter_size': (3, 3),
    'n_out': 64,
    'activation': 'relu',
    'with_bias': True,
    'strides': (2, 1)
    },
  'conv_merged': {
    'class': 'merge_dims',
    'from': 'subsample_conv1',
    'axes': 'static'
    },
  'source_linear': {
    'class': 'linear',
    'activation': None,
    'with_bias': False,
    'from': 'conv_merged',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_ffmod_1_ln': {
    'class': 'layer_norm',
    'from': 'source_linear'
    },
  'conformer_block_01_ffmod_1_ff1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_ffmod_1_ln',
    'n_out': 1024,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_ffmod_1_relu': {
    'class': 'activation',
    'activation': 'relu',
    'from': 'conformer_block_01_ffmod_1_ff1'
    },
  'conformer_block_01_ffmod_1_square_relu': {
    'class': 'eval',
    'eval': 'source(0) ** 2',
    'from': 'conformer_block_01_ffmod_1_relu'
    },
  'conformer_block_01_ffmod_1_drop1': {
    'class': 'dropout',
    'from': 'conformer_block_01_ffmod_1_square_relu',
    'dropout': 0.0
    },
  'conformer_block_01_ffmod_1_ff2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_ffmod_1_drop1',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_ffmod_1_drop2': {
    'class': 'dropout',
    'from': 'conformer_block_01_ffmod_1_ff2',
    'dropout': 0.0
    },
  'conformer_block_01_ffmod_1_half_step': {
    'class': 'eval',
    'eval': '0.5 * source(0)',
    'from': 'conformer_block_01_ffmod_1_drop2'
    },
  'conformer_block_01_ffmod_1_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_01_ffmod_1_half_step', 'source_linear'],
    'n_out': 256
    },
  'conformer_block_01_self_att_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_01_ffmod_1_res'
    },
  'conformer_block_01_self_att_ln_rel_pos_enc': {
    'class': 'relative_positional_encoding',
    'from': 'conformer_block_01_self_att_ln',
    'n_out': 32,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    'clipping': 16
    },
  'conformer_block_01_self_att': {
    'class': 'self_attention',
    'from': 'conformer_block_01_self_att_ln',
    'n_out': 256,
    'num_heads': 8,
    'total_key_dim': 256,
    'key_shift': 'conformer_block_01_self_att_ln_rel_pos_enc',
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"
    },
  'conformer_block_01_self_att_linear': {
    'class': 'linear',
    'activation': None,
    'with_bias': False,
    'from': 'conformer_block_01_self_att',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_self_att_dropout': {
    'class': 'dropout',
    'from': 'conformer_block_01_self_att_linear',
    'dropout': 0.0
    },
  'conformer_block_01_self_att_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_01_self_att_dropout', 'conformer_block_01_ffmod_1_res'],
    'n_out': 256
    },
  'conformer_block_01_conv_mod_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_01_self_att_res'
    },
  'conformer_block_01_conv_mod_pointwise_conv1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_conv_mod_ln',
    'n_out': 512,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_conv_mod_glu': {
    'class': 'gating',
    'from': 'conformer_block_01_conv_mod_pointwise_conv1',
    'activation': 'identity'
    },
  'conformer_block_01_conv_mod_depthwise_conv2': {
    'class': 'conv',
    'from': 'conformer_block_01_conv_mod_glu',
    'padding': 'same',
    'filter_size': (16,),
    'n_out': 256,
    'activation': None,
    'with_bias': True,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    'groups': 256
    },
  'conformer_block_01_conv_mod_bn': {
    'class': 'batch_norm',
    'from': 'conformer_block_01_conv_mod_depthwise_conv2',
    'momentum': 0.1,
    'epsilon': 0.001,
    'update_sample_only_in_training': True,
    'delay_sample_update': True
    },
  'conformer_block_01_conv_mod_swish': {
    'class': 'activation',
    'activation': 'swish',
    'from': 'conformer_block_01_conv_mod_bn'
    },
  'conformer_block_01_conv_mod_pointwise_conv2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_conv_mod_swish',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_conv_mod_drop': {
    'class': 'dropout',
    'from': 'conformer_block_01_conv_mod_pointwise_conv2',
    'dropout': 0.0
    },
  'conformer_block_01_conv_mod_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_01_conv_mod_drop', 'conformer_block_01_self_att_res'],
    'n_out': 256
    },
  'conformer_block_01_ffmod_2_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_01_conv_mod_res'
    },
  'conformer_block_01_ffmod_2_ff1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_ffmod_2_ln',
    'n_out': 1024,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_ffmod_2_relu': {
    'class': 'activation',
    'activation': 'relu',
    'from': 'conformer_block_01_ffmod_2_ff1'
    },
  'conformer_block_01_ffmod_2_square_relu': {
    'class': 'eval',
    'eval': 'source(0) ** 2',
    'from': 'conformer_block_01_ffmod_2_relu'
    },
  'conformer_block_01_ffmod_2_drop1': {
    'class': 'dropout',
    'from': 'conformer_block_01_ffmod_2_square_relu',
    'dropout': 0.0
    },
  'conformer_block_01_ffmod_2_ff2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_01_ffmod_2_drop1',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_01_ffmod_2_drop2': {
    'class': 'dropout',
    'from': 'conformer_block_01_ffmod_2_ff2',
    'dropout': 0.0
    },
  'conformer_block_01_ffmod_2_half_step': {
    'class': 'eval',
    'eval': '0.5 * source(0)',
    'from': 'conformer_block_01_ffmod_2_drop2'
    },
  'conformer_block_01_ffmod_2_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_01_ffmod_2_half_step', 'conformer_block_01_conv_mod_res'],
    'n_out': 256
    },
  'conformer_block_01_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_01_ffmod_2_res'
    },
  'conformer_block_01': {
    'class': 'copy',
    'from': 'conformer_block_01_ln'
    },
  'conformer_block_02_ffmod_1_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_01'
    },
  'conformer_block_02_ffmod_1_ff1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_ffmod_1_ln',
    'n_out': 1024,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_ffmod_1_relu': {
    'class': 'activation',
    'activation': 'relu',
    'from': 'conformer_block_02_ffmod_1_ff1'
    },
  'conformer_block_02_ffmod_1_square_relu': {
    'class': 'eval',
    'eval': 'source(0) ** 2',
    'from': 'conformer_block_02_ffmod_1_relu'
    },
  'conformer_block_02_ffmod_1_drop1': {
    'class': 'dropout',
    'from': 'conformer_block_02_ffmod_1_square_relu',
    'dropout': 0.0
    },
  'conformer_block_02_ffmod_1_ff2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_ffmod_1_drop1',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_ffmod_1_drop2': {
    'class': 'dropout',
    'from': 'conformer_block_02_ffmod_1_ff2',
    'dropout': 0.0
    },
  'conformer_block_02_ffmod_1_half_step': {
    'class': 'eval',
    'eval': '0.5 * source(0)',
    'from': 'conformer_block_02_ffmod_1_drop2'
    },
  'conformer_block_02_ffmod_1_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_02_ffmod_1_half_step', 'conformer_block_01'],
    'n_out': 256
    },
  'conformer_block_02_self_att_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_02_ffmod_1_res'
    },
  'conformer_block_02_self_att_ln_rel_pos_enc': {
    'class': 'relative_positional_encoding',
    'from': 'conformer_block_02_self_att_ln',
    'n_out': 32,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    'clipping': 16
    },
  'conformer_block_02_self_att': {
    'class': 'self_attention',
    'from': 'conformer_block_02_self_att_ln',
    'n_out': 256,
    'num_heads': 8,
    'total_key_dim': 256,
    'key_shift': 'conformer_block_02_self_att_ln_rel_pos_enc',
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"
    },
  'conformer_block_02_self_att_linear': {
    'class': 'linear',
    'activation': None,
    'with_bias': False,
    'from': 'conformer_block_02_self_att',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_self_att_dropout': {
    'class': 'dropout',
    'from': 'conformer_block_02_self_att_linear',
    'dropout': 0.0
    },
  'conformer_block_02_self_att_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_02_self_att_dropout', 'conformer_block_02_ffmod_1_res'],
    'n_out': 256
    },
  'conformer_block_02_conv_mod_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_02_self_att_res'
    },
  'conformer_block_02_conv_mod_pointwise_conv1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_conv_mod_ln',
    'n_out': 512,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_conv_mod_glu': {
    'class': 'gating',
    'from': 'conformer_block_02_conv_mod_pointwise_conv1',
    'activation': 'identity'
    },
  'conformer_block_02_conv_mod_depthwise_conv2': {
    'class': 'conv',
    'from': 'conformer_block_02_conv_mod_glu',
    'padding': 'same',
    'filter_size': (16,),
    'n_out': 256,
    'activation': None,
    'with_bias': True,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)",
    'groups': 256
    },
  'conformer_block_02_conv_mod_bn': {
    'class': 'batch_norm',
    'from': 'conformer_block_02_conv_mod_depthwise_conv2',
    'momentum': 0.1,
    'epsilon': 0.001,
    'update_sample_only_in_training': True,
    'delay_sample_update': True
    },
  'conformer_block_02_conv_mod_swish': {
    'class': 'activation',
    'activation': 'swish',
    'from': 'conformer_block_02_conv_mod_bn'
    },
  'conformer_block_02_conv_mod_pointwise_conv2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_conv_mod_swish',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_conv_mod_drop': {
    'class': 'dropout',
    'from': 'conformer_block_02_conv_mod_pointwise_conv2',
    'dropout': 0.0
    },
  'conformer_block_02_conv_mod_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_02_conv_mod_drop', 'conformer_block_02_self_att_res'],
    'n_out': 256
    },
  'conformer_block_02_ffmod_2_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_02_conv_mod_res'
    },
  'conformer_block_02_ffmod_2_ff1': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_ffmod_2_ln',
    'n_out': 1024,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_ffmod_2_relu': {
    'class': 'activation',
    'activation': 'relu',
    'from': 'conformer_block_02_ffmod_2_ff1'
    },
  'conformer_block_02_ffmod_2_square_relu': {
    'class': 'eval',
    'eval': 'source(0) ** 2',
    'from': 'conformer_block_02_ffmod_2_relu'
    },
  'conformer_block_02_ffmod_2_drop1': {
    'class': 'dropout',
    'from': 'conformer_block_02_ffmod_2_square_relu',
    'dropout': 0.0
    },
  'conformer_block_02_ffmod_2_ff2': {
    'class': 'linear',
    'activation': None,
    'with_bias': True,
    'from': 'conformer_block_02_ffmod_2_drop1',
    'n_out': 256,
    'forward_weights_init': "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"
    },
  'conformer_block_02_ffmod_2_drop2': {
    'class': 'dropout',
    'from': 'conformer_block_02_ffmod_2_ff2',
    'dropout': 0.0
    },
  'conformer_block_02_ffmod_2_half_step': {
    'class': 'eval',
    'eval': '0.5 * source(0)',
    'from': 'conformer_block_02_ffmod_2_drop2'
    },
  'conformer_block_02_ffmod_2_res': {
    'class': 'combine',
    'kind': 'add',
    'from': ['conformer_block_02_ffmod_2_half_step', 'conformer_block_02_conv_mod_res'],
    'n_out': 256
    },
  'conformer_block_02_ln': {
    'class': 'layer_norm',
    'from': 'conformer_block_02_ffmod_2_res'
    },
  'conformer_block_02': {
    'class': 'copy',
    'from': 'conformer_block_02_ln'
    },
  'encoder': {
    'class': 'copy',
    'from': 'data:data'
    }, # 'ctc': {
  #   'class': 'softmax',
  #   'from': 'encoder',
  #   'target': 'label_ground_truth',
  #   'loss': 'ctc',
  #   'loss_opts': {
  #     'beam_width': 1,
  #     'use_native': False
  #     },
  #   'loss_scale': 1.0
  #   },
  'inv_fertility': {
    'class': 'linear',
    'activation': 'sigmoid',
    'with_bias': False,
    'from': 'encoder',
    'n_out': 1
    },
  'output': {
    'class': 'rec',
    'from': 'encoder',
    'unit': {
      'emit_blank_log_prob': {
        'class': 'copy',
        'from': ['blank_log_prob', 'emit_log_prob']
        },
      'emit_blank_prob': {
        'activation': 'exp',
        'class': 'activation',
        'from': 'emit_blank_log_prob',
        'loss': 'ce',
        'loss_opts': {
          'focal_loss_factor': 0.0
          },
        'target': 'emit_ground_truth'
        },
      'output': {
        'beam_size': 4,
        'cheating': 'exclusive',
        'class': 'choice',
        'from': 'data',
        'initial_output': 0,
        'input_type': 'log_prob',
        'target': 'targetb'
        },
      'am': {
        'class': 'copy',
        'from': 'data:source'
        },
      'blank_log_prob': {
        'class': 'eval',
        'eval': 'tf.math.log_sigmoid(-source(0))',
        'from': 'emit_prob0'
        },
      'const1': {
        'class': 'constant',
        'value': 1
        },
      'emit_log_prob': {
        'activation': 'log_sigmoid',
        'class': 'activation',
        'from': 'emit_prob0'
        },
      'emit_prob0': {
        'activation': None,
        'class': 'linear',
        'from': 's',
        'is_output_layer': True,
        'n_out': 1
        },
      'output_emit': {
        'class': 'compare',
        'from': 'output',
        'initial_output': True,
        'kind': 'not_equal',
        'value': 4
        },
      'prev_out_embed': {
        'activation': None,
        'class': 'linear',
        'from': 'prev:output',
        'n_out': 128
        }, # 's': {
      #   'L2': 0.0001,
      #   'class': 'rec',
      #   'dropout': 0.3,
      #   'from': ['am', 'prev_out_embed'],
      #   'n_out': 128,
      #   'unit': 'lstm',
      #   'unit_opts': {
      #     'rec_weight_dropout': 0.3
      #     }
      #   },
      's': {
        'class': 'rnn_cell',
        'unit': 'zoneoutlstm',
        'n_out': 128,
        'from': ['am', 'prev_out_embed'],
        'unit_opts': {
          'zoneout_factor_cell': 0.15,
          'zoneout_factor_output': 0.05
          }
        },
      'segment_ends': {
        'class': 'switch',
        'condition': 'seq_end_too_far',
        'false_from': 'segment_ends1',
        'true_from': 'seq_lens'
        },
      'segment_ends1': {
        'class': 'eval',
        'eval': 'source(0) + source(1) + 2',
        'from': ['segment_starts1', 'segment_lens1']
        },
      'segment_lens': {
        'class': 'eval',
        'eval': 'source(0) - source(1)',
        'from': ['segment_ends', 'segment_starts'],
        'is_output_layer': True
        },
      'segment_lens0': {
        'class': 'combine',
        'from': [':i', 'segment_starts1'],
        'kind': 'sub'
        },
      'segment_lens1': {
        'class': 'combine',
        'from': ['segment_lens0', 'const1'],
        'is_output_layer': True,
        'kind': 'add'
        },
      'segment_starts': {
        'class': 'switch',
        'condition': 'seq_start_too_far',
        'false_from': 'segment_starts2',
        'true_from': 0
        },
      'segment_starts1': {
        'class': 'switch',
        'condition': 'prev:output_emit',
        'false_from': 'prev:segment_starts1',
        'initial_output': 0,
        'is_output_layer': True,
        'true_from': ':i'
        },
      'segment_starts2': {
        'class': 'eval',
        'eval': 'source(0) + source(1) - 2',
        'from': ['segment_starts1', 'segment_lens1'],
        'is_output_layer': True
        },
      'seq_end_too_far': {
        'class': 'compare',
        'from': ['segment_ends1', 'seq_lens'],
        'kind': 'greater'
        },
      'seq_lens': {
        'class': 'length',
        'from': 'base:encoder'
        },
      'seq_start_too_far': {
        'class': 'compare',
        'from': ['segment_starts2'],
        'kind': 'less',
        'value': 0
        }
      },
    'target': 'targetb',
    'back_prop': True,
    'include_eos': True,
    'size_target': 'targetb'
    },
  '#config': {
    'batch_size': 3600000
    },
  '#copy_param_mode': 'subset',
  'stft': {
    'class': 'stft',
    'frame_shift': 160,
    'frame_size': 400,
    'fft_size': 512,
    'from': 'data:data'
    },
  'abs': {
    'class': 'activation',
    'from': 'stft',
    'activation': 'abs'
    },
  'power': {
    'class': 'eval',
    'from': 'abs',
    'eval': 'source(0) ** 2'
    },
  'mel_filterbank': {
    'class': 'mel_filterbank',
    'from': 'power',
    'fft_size': 512,
    'nr_of_filters': 80,
    'n_out': 80
    },
  'log': {
    'from': 'mel_filterbank',
    'class': 'activation',
    'activation': 'safe_log',
    'opts': {
      'eps': 1e-10
      }
    },
  'log10': {
    'from': 'log',
    'class': 'eval',
    'eval': 'source(0) / 2.3026'
    },
  'log_mel_features': {
    'class': 'copy',
    'from': 'log10'
    },
  'existing_alignment': {
    'class': 'reinterpret_data',
    'from': 'data:classes',
    'set_sparse': True,
    'set_sparse_dim': 5,
    'size_base': 'encoder'
    },
  'is_label': {
    'class': 'compare',
    'from': 'existing_alignment',
    'kind': 'not_equal',
    'value': 4
    },
  'label_ground_truth_masked': {
    'class': 'reinterpret_data',
    'enforce_batch_major': True,
    'from': 'label_ground_truth_masked0',
    'register_as_extern_data': 'label_ground_truth',
    'set_sparse_dim': 4
    },
  'label_ground_truth_masked0': {
    'class': 'masked_computation',
    'from': 'existing_alignment',
    'mask': 'is_label',
    'unit': {
      'class': 'copy',
      'from': 'data'
      }
    },
  'emit_ground_truth': {
    'class': 'reinterpret_data',
    'from': 'emit_ground_truth0',
    'is_output_layer': True,
    'register_as_extern_data': 'emit_ground_truth',
    'set_sparse': True,
    'set_sparse_dim': 2
    },
  'emit_ground_truth0': {
    'class': 'switch',
    'condition': 'is_label',
    'false_from': 'const0',
    'true_from': 'const1'
    },
  'const0': {
    'class': 'constant',
    'value': 0,
    'with_batch_dim': True
    },
  'const1': {
    'class': 'constant',
    'value': 1,
    'with_batch_dim': True
    },
  'labels_with_blank_ground_truth': {
    'class': 'copy',
    'from': 'existing_alignment',
    'register_as_extern_data': 'targetb'
    },
  'segment_lens_masked': {
    'class': 'masked_computation',
    'from': 'output/segment_lens',
    'mask': 'is_label',
    'register_as_extern_data': 'segment_lens_masked',
    'unit': {
      'class': 'copy',
      'from': 'data'
      }
    },
  'segment_starts_masked': {
    'class': 'masked_computation',
    'from': 'output/segment_starts',
    'mask': 'is_label',
    'register_as_extern_data': 'segment_starts_masked',
    'unit': {
      'class': 'copy',
      'from': 'data'
      }
    },
  'label_model': {
    'class': 'rec',
    'from': [],
    'unit': {
      'target_embed0': {
        'class': 'linear',
        'activation': None,
        'with_bias': False,
        'from': 'output',
        'n_out': 640,
        'initial_output': 0
        },
      'target_embed': {
        'class': 'dropout',
        'from': 'target_embed0',
        'dropout': 0.0,
        'dropout_noise_shape': {
          '*': None
          }
        }, # "accum_weights": {
      #   "class": "eval",
      #   "from": ["segment_range", ""]
      # },
      "max_segment_len": {
        "class": "reduce",
        "from": "segment_lens",
        "mode": "max"
        },
      "segment_range0": {
        "class": "range_in_axis",
        "from": "att_weights",
        "axis": "stag:att_t"
        },
      # "segment_range0": {
      #   "class": "range_from_length",
      #   "from": "max_segment_len",
      #   },
      "segment_range1": {
        "class": "combine",
        "kind": "add",
        "from": ["segment_range0", "segment_starts"]
        },
      "enc_lens": {
        "class": "length", "from": "base:encoder"
        },
      "enc_max_idx": {
        "class": "eval", "from": "enc_lens", "eval": "source(0) - 1"
        },
      "segment_range_mask": {
        "class": "compare", "from": ["segment_range1", "enc_lens"], "kind": "greater_equal"
        },
      "segment_range": {
        "class": "switch",
        "condition": "segment_range_mask",
        "true_from": "enc_max_idx",
        "false_from": "segment_range1"
        },
      "att_weights_squeezed": {
        "class": "squeeze",
        "from": "att_weights",
        "axis": "f"
        },
      "att_weights_scattered": {
        "class": "scatter_nd",
        "from": "att_weights_squeezed",
        "position": "segment_range",
        "position_axis": "stag:att_t",
        "output_dim_via_time_from": "base:encoder",
        # "is_output_layer": True
        },
      'accum_weights': {
        'class': 'eval',
        'eval': 'source(0) + source(1) * source(2) * 0.5',
        'from': ['prev:accum_weights', 'att_weights_scattered', 'base:inv_fertility'],
        'out_type': {
          'dim': 1,
          'shape': (None, 1)
          }
        },
      "print_att_weights": {
        "class": "print",
        "from": "att_weights",
        "is_output_layer": True
        },
      "print_segment_range": {
        "class": "print",
        "from": "segment_range",
        "is_output_layer": True
        },
      "print_accum_weights": {
        "class": "print",
        "from": "accum_weights",
        "is_output_layer": True
        },
      "print_targetb": {
        "class": "print",
        "from": "data:targetb",
        "is_output_layer": True
        },
      "print_segment_starts": {
        "class": "print",
        "from": "segment_starts",
        "is_output_layer": True
        },
      "print_segment_lens": {
        "class": "print",
        "from": "segment_lens",
        "is_output_layer": True
        },
      'lm_transformed': {
        'class': 'linear',
        'activation': None,
        'with_bias': False,
        'from': 'lm',
        'n_out': 1024
        },
      # 'accum_att_weights': {
      #   'class': 'eval',
      #   'eval': 'source(0) + source(1) * source(2) * 0.5',
      #   'from': ['prev:accum_att_weights', 'att_weights', 'base:inv_fertility'],
      #   'out_type': {
      #     'dim': 1,
      #     'shape': (None, 1)
      #     }
      #   },
      # 'weight_feedback': {
      #   'class': 'constant',
      #   'value': 0,
      #   'dtype': 'float32',
      #   'with_batch_dim': True
      #   },
      "prev_accum_weights_sliced0": {
        'class': 'slice_nd',
        'from': 'prev:accum_weights',
        'size': 'segment_lens',
        'start': 'segment_starts'
        },
      "prev_accum_weights_sliced": {
        "class": "reinterpret_data",
        "from": "prev_accum_weights_sliced0",
        "set_dim_tags": {
          "stag:sliced-time:prev_accum_weights_sliced": att_t_tag
        },
      },
      "weight_feedback": {
          "class": "linear",
          "activation": None,
          "with_bias": False,
          "from": "prev_accum_weights_sliced",
          "n_out": 1024,
      },
      'energy_in': {
        'class': 'combine',
        'kind': 'add',
        'from': ['att_ctx', 'weight_feedback', 'lm_transformed'],
        'n_out': 1024
        },
      'energy_tanh': {
        'class': 'activation',
        'activation': 'tanh',
        'from': 'energy_in'
        },
      'energy': {
        'class': 'linear',
        'activation': None,
        'with_bias': False,
        'from': 'energy_tanh',
        'n_out': 1
        },
      'att_weights': {
        'class': 'softmax_over_spatial',
        'from': 'energy',
        'axis': 'stag:att_t'
        },
      'att0': {
        'class': 'generic_attention',
        'weights': 'att_weights',
        'base': 'att_val'
        },
      'att': {
        'class': 'merge_dims',
        'from': 'att0',
        'axes': 'except_batch'
        },
      'lm': {
        'class': 'rnn_cell',
        'unit': 'zoneoutlstm',
        'n_out': 1024,
        'from': ['prev:target_embed', 'prev:att'],
        'unit_opts': {
          'zoneout_factor_cell': 0.15,
          'zoneout_factor_output': 0.05
          }
        },
      'readout_in': {
        'class': 'linear',
        'activation': None,
        'with_bias': True,
        'from': ['lm', 'prev:target_embed', 'att'],
        'n_out': 1024
        },
      'readout': {
        'class': 'reduce_out',
        'from': 'readout_in',
        'num_pieces': 2,
        'mode': 'max'
        },
      'output_prob': {
        'class': 'softmax',
        'from': 'readout',
        'target': 'label_ground_truth',
        'loss': 'ce',
        'loss_opts': {
          'label_smoothing': 0
          }
        },
      'output': {
        'class': 'choice',
        'target': 'label_ground_truth',
        'beam_size': 12,
        'from': 'output_prob',
        'initial_output': 0
        },
      'segment_lens': {
        'axis': 't',
        'class': 'gather',
        'from': 'base:data:segment_lens_masked',
        'position': ':i'
        },
      'segment_starts': {
        'axis': 't',
        'class': 'gather',
        'from': 'base:data:segment_starts_masked',
        'position': ':i'
        },
      'segments0': {
        'class': 'slice_nd',
        'from': 'base:encoder',
        'size': 'segment_lens',
        'start': 'segment_starts'
        },
      "segments": {
        "class": "reinterpret_data",
        "from": "segments0",
        "set_dim_tags": {
          "stag:sliced-time:segments": att_t_tag
        },
      },
      'att_ctx': {
        'activation': None,
        'class': 'linear',
        'name_scope': '/enc_ctx',
        'from': 'segments',
        'n_out': 1024,
        'with_bias': True
        },
      'att_val': {
        'class': 'copy',
        'from': 'segments'
        }
      },
    'target': 'label_ground_truth',
    'name_scope': 'output/rec',
    'is_output_layer': True
    }
  }

# debug_print_layer_output_template = True

# trainer
batching = "random"
batch_size = 5000
max_seqs = 1
chunking = "0"
optimizer = {
  "class": "adam"
  }
gradient_noise = 0.3
learning_rate = 0.01
learning_rate_control = "newbob"
learning_rate_control_relative_error_relative_lr = True
model = "/tmp/%s/returnn/%s/model" % (
get_login_username(), demo_name)  # https://github.com/tensorflow/tensorflow/issues/6537
num_epochs = 100
save_interval = 20

# log
log_verbosity = 5
