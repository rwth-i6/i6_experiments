import numpy
import copy

from .zeineldeen_helpers.models.asr.encoder.conformer_encoder import ConformerEncoder
from .zeineldeen_helpers.models.asr.decoder.rnn_decoder import RNNDecoder
from .zeineldeen_helpers.models.lm.external_lm_decoder import ExternalLMDecoder

from .zeineldeen_helpers import specaugment
from .zeineldeen_helpers import data_aug

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

# ------------------------------- Default config ------------------------------- #

from .base_config import config, post_config


# -------------------------- LR Scheduling -------------------------- #

# Noam LR
noam_lr_str = """
def noam(n, model_d={model_d}, warmup_n={warmup_n}):
  from returnn.tf.compat import v1 as tf
  model_d = tf.cast(model_d, tf.float32)
  n = tf.cast(n, tf.float32)
  warmup_n = tf.cast(warmup_n, tf.float32)
  return tf.pow(model_d, -0.5) * tf.minimum(tf.pow(n, -0.5), n * tf.pow(warmup_n, -1.5))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
  return learning_rate * noam(n=global_train_step)
"""

warmup_lr_str = """
def warmup_lr(step, warmup_steps={warmup_steps}):
    from returnn.tf.compat import v1 as tf
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.cast(warmup_steps, tf.float32)
    return tf.pow(warmup_steps, 0.5) * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
  return learning_rate * warmup_lr(step=global_train_step)
"""

cyclic_lr = """
def cyclic_lr(step, decay={decay}, interval={interval}):
    from returnn.tf.compat import v1 as tf
    return tf.pow(decay, tf.cast(tf.math.floormod(step,interval), dtype=float))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    return learning_rate * cyclic_lr(step=global_train_step)
"""

# -------------------------- Pretraining -------------------------- #

# By default:
# - Start with 1 layer
# - Grow layers and dimensions (subset copy mode is used)
# - Use exponential grow for layers


def pretrain_layers_and_dims(idx, net_dict: dict, encoder_args, decoder_args, reduce_dims=True, initial_dim_factor=0.5,
                             exp_step=True, start_num_layers=1, repeat_first=True):

  StartNumLayers = start_num_layers
  InitialDimFactor = initial_dim_factor

  encoder_args_copy = copy.deepcopy(encoder_args)
  decoder_args_copy = copy.deepcopy(decoder_args)

  final_num_blocks = encoder_args_copy['num_blocks']
  assert final_num_blocks >= StartNumLayers

  extra_net_dict = dict()
  extra_net_dict["#config"] = {}
  if idx < 4:
    extra_net_dict["#config"]["batch_size"] = 20000

  if repeat_first:
    idx = max(idx - 1, 0)

  if exp_step:
    num_blocks = max(2 ** idx, StartNumLayers)
  else:
    num_blocks = max(2 * idx, StartNumLayers)

  if num_blocks > final_num_blocks:
    return None

  encoder_args_copy['num_blocks'] = num_blocks
  decoder_args_copy['label_smoothing'] = 0
  AttNumHeads = encoder_args_copy['att_num_heads']

  if reduce_dims:
    grow_frac = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
    dim_frac = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac
  else:
    dim_frac = 1

  encoder_keys = {'enc_key_dim', 'ff_dim', 'conv_kernel_size'}

  if reduce_dims:
    for key in encoder_keys:
      encoder_args_copy[key] = int(encoder_args_copy[key] * dim_frac / float(AttNumHeads)) * AttNumHeads

  for k in encoder_args_copy.keys():
    if 'dropout' in k and encoder_args_copy[k] is not None:
      if idx <= 1:
        encoder_args_copy[k] = 0.0
      else:
        encoder_args_copy[k] *= dim_frac
    if 'l2' in k and encoder_args_copy[k] is not None:
      if idx <= 1:
        encoder_args_copy[k] = 0.0
      else:
        encoder_args_copy[k] *= dim_frac

  for k in decoder_args_copy.keys():
    if 'dropout' in k and decoder_args_copy[k] is not None:
      if idx <= 1:
        decoder_args_copy[k] = 0.0
      else:
        decoder_args_copy[k] *= dim_frac
    if 'l2' in k and decoder_args_copy[k] is not None:
      if idx <= 1:
        decoder_args_copy[k] = 0.0
      else:
        decoder_args_copy[k] *= dim_frac

  conformer_encoder = ConformerEncoder(**encoder_args_copy)
  conformer_encoder.create_network()

  rnn_decoder = RNNDecoder(base_model=conformer_encoder, **decoder_args_copy)
  rnn_decoder.create_network()

  net_dict = conformer_encoder.network.get_net()
  net_dict.update(rnn_decoder.network.get_net())
  net_dict.update(extra_net_dict)

  return net_dict

# -------------------------------------------------------------------- #


def create_config(
  name, training_datasets, is_recog=False, input_key="audio_features", retrain_checkpoint=None, enc_layers=12, enc_key_dim=512, enc_att_num_heads=8,
  dec_att_num_heads=1, dropout=0.1, input_layer='lstm-6', conv_kernel_size=32, ff_dim=2048, ff_init=None, att_dropout=0.0,
  lr=0.0008, wup_start_lr=0.0003, lr_decay=0.9, const_lr=0, wup=10, epoch_split=20, batch_size=10000,
  accum_grad=2, with_ctc=True, embed_dim=512, l2=0.0001, pretrain_reps=5, max_seq_length=75,
  rec_weight_dropout=0.0, dec_lstm_num_units=512, dec_output_num_units=1024, noam_opts=None,
  warmup_lr_opts=None, with_pretrain=True, pretrain_opts=None, native_ctc=True, pos_enc='rel',
  loc_conv_att_filter_size=None, loc_conv_att_num_channels=None, start_conv_init=None,
  mhsa_init=None, conv_module_init=None, mhsa_out_init=None, embed_dropout=0.1, speed_pert=False,
  gradient_clip_global_norm=0.0, rel_pos_clipping=16, subsample=None, dropout_in=0.1, embed_weight_init=None,
  lstm_weights_init=None, reduceout=True, ctc_loss_scale=None, batch_norm_opts=None, stoc_layers_prob=0.0, use_ln=False,
  self_att_l2=0.0, sandwich_conv=False
):

  exp_config = copy.deepcopy(config)  # type: dict

  exp_config["extern_data"] = training_datasets.extern_data
  exp_config["train"] = training_datasets.train.as_returnn_opts()
  exp_config["dev"] = training_datasets.cv.as_returnn_opts()
  exp_config["eval_datasets"] = {'devtrain': training_datasets.devtrain.as_returnn_opts()}

  target = 'bpe_labels'

  # add default hyperparameters
  # model and learning_rates paths are added in the CRNNTraining job
  hyperparams = {
    'gradient_clip': 0,
    'adam': True,
    'optimizer_epsilon': 1e-8,
    'accum_grad_multiple_step': accum_grad,
    'gradient_noise': 0.0,
    'batch_size': batch_size,
    'max_seqs': 200,
    'truncation': -1
  }
  if max_seq_length:
    hyperparams['max_seq_length'] = {target: max_seq_length}  # char-bpe
  if gradient_clip_global_norm:
    hyperparams['gradient_clip_global_norm'] = gradient_clip_global_norm

  extra_python_code = '\n'.join(['import sys', 'sys.setrecursionlimit(3000)'])  # for network construction

  # LR scheduling
  if noam_opts:
    noam_opts['model_d'] = enc_key_dim
    exp_config['learning_rate'] = noam_opts['lr']
    exp_config['learning_rate_control'] = 'constant'
    extra_python_code += '\n' + noam_lr_str.format(**noam_opts)
  elif warmup_lr_opts:
    exp_config['learning_rate'] = warmup_lr_opts['peak_lr']
    exp_config['learning_rate_control'] = 'constant'
    extra_python_code += '\n' + warmup_lr_str.format(**warmup_lr_opts)
  else:
    learning_rates = [wup_start_lr] * const_lr +  list(numpy.linspace(wup_start_lr, lr, num=wup))
    exp_config['learning_rate'] = lr
    exp_config['learning_rates'] = learning_rates
    exp_config['min_learning_rate'] = lr / 50
    exp_config['learning_rate_control'] = "newbob_multi_epoch"
    exp_config['learning_rate_control_relative_error_relative_lr'] = True
    exp_config['learning_rate_control_min_num_epochs_per_new_lr'] = 3
    exp_config['use_learning_rate_control_always'] = True
    exp_config['newbob_multi_num_epochs'] = epoch_split
    exp_config['newbob_multi_update_interval'] = 1
    exp_config['newbob_learning_rate_decay'] = lr_decay

  # -------------------------- network -------------------------- #

  encoder_args = dict(
    target=target, input="data:" + input_key, num_blocks=enc_layers, enc_key_dim=enc_key_dim, att_num_heads=enc_att_num_heads, dropout=dropout,
    att_dropout=att_dropout, ff_init=ff_init, ff_dim=ff_dim, input_layer=input_layer, conv_kernel_size=conv_kernel_size,
    with_ctc=with_ctc, pos_enc=pos_enc, native_ctc=native_ctc, l2=l2, lstm_dropout=dropout, start_conv_init=start_conv_init,
    rel_pos_clipping=rel_pos_clipping, subsample=subsample, dropout_in=dropout_in, mhsa_init=mhsa_init,
    conv_module_init=conv_module_init, mhsa_out_init=mhsa_out_init, ctc_loss_scale=ctc_loss_scale,
    stoc_layers_prob=stoc_layers_prob, batch_norm_opts=batch_norm_opts, use_ln=use_ln,
    self_att_l2=self_att_l2, sandwich_conv=sandwich_conv)

  conformer_encoder = ConformerEncoder(**encoder_args)
  conformer_encoder.create_network()

  decoder_args = dict(target=target, rec_weight_dropout=rec_weight_dropout, l2=l2, dec_zoneout=False,
    att_dropout=att_dropout, embed_dropout=embed_dropout, dec_lstm_num_units=dec_lstm_num_units,
    embed_dim=embed_dim, dec_output_num_units=dec_output_num_units, dropout=dropout, att_num_heads=dec_att_num_heads,
    loc_conv_att_num_channels=loc_conv_att_num_channels, loc_conv_att_filter_size=loc_conv_att_filter_size,
    embed_weight_init=embed_weight_init, lstm_weights_init=lstm_weights_init, reduceout=reduceout)

  rnn_decoder = RNNDecoder(base_model=conformer_encoder, **decoder_args)
  rnn_decoder.create_network()

  if ext_lm_opts:
    rnn_decoder = ExternalLMDecoder(
      rnn_decoder, ext_lm_opts, prior_lm_opts=prior_lm_opts, beam_size=beam_size, dec_type='transformer')
    transformer_decoder.create_network()

  # add full network
  exp_config['network'] = conformer_encoder.network.get_net()  # type: dict
  exp_config['network'].update(rnn_decoder.network.get_net())

  decision_layer_name = rnn_decoder.decision_layer_name
  exp_config['search_output_layer'] = decision_layer_name

  # -------------------------- end network -------------------------- #

  # add hyperparmas
  exp_config.update(hyperparams)

  python_prolog = specaugment.specaug_tf2.get_funcs()  # type: list

  if speed_pert:
    python_prolog += [data_aug.speed_pert]

  if retrain_checkpoint is not None:
    exp_config['import_model_train_epoch1'] = retrain_checkpoint

  # add pretraining
  if with_pretrain and ext_lm_opts is None and retrain_checkpoint is None:

    if pretrain_opts is None:
      pretrain_opts = {}

    pretrain_networks = []
    idx = 0
    while True:
      net = pretrain_layers_and_dims(idx, exp_config['network'], encoder_args, decoder_args, **pretrain_opts)
      if not net:
        break
      pretrain_networks.append(net)
      idx += 1

    exp_config['pretrain_nets_lookup'] = {k: v for k, v in enumerate(pretrain_networks)}

    exp_config['pretrain'] = {
      "repetitions": pretrain_reps,
      "copy_param_mode": "subset",
      "construction_algo": CodeWrapper('custom_construction_algo')
    }

    pretrain_algo_str = 'def custom_construction_algo(idx, net_dict):\n\treturn pretrain_nets_lookup.get(idx, None)'
    python_prolog += [pretrain_algo_str]

  returnn_config = ReturnnConfig(
    exp_config, post_config=post_config, python_prolog=python_prolog, python_epilog=extra_python_code, hash_full_python_code=True,
    pprint_kwargs={'sort_dicts': False})

  return returnn_config

