from sisyphus import *
import sys
import os
import numpy
import copy
import math

from . import zeineldeen_helpers as helpers

from .zeineldeen_helpers.models.asr.encoder.conformer_encoder import ConformerEncoder
from .zeineldeen_helpers.models.asr.decoder.transformer_decoder import TransformerDecoder
from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM
from .zeineldeen_helpers.models.lm.external_lm_decoder import ExternalLMDecoder
from .zeineldeen_helpers.models.lm import generic_lm

from .zeineldeen_helpers import specaugment
from .zeineldeen_helpers import data_aug

from i6_core.returnn.config import ReturnnConfig, CodeWrapper


config = {

}

# changing these does not change the hash
post_config = {
    'use_tensorflow': True,
    'tf_log_memory_usage': True,
    'cleanup_old_models': True,
    'log_batch_size': True,
    'debug_print_layer_output_template': True,
    'debug_mode': False,
    'batching': 'random'
}

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

cycle_lr_str = """
def cyclic_lr(step, decay={decay}, interval={interval}):
    from returnn.tf.compat import v1 as tf
    return tf.pow(decay, tf.cast(tf.math.floormod(step, interval), dtype=float))

def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
    from returnn.tf.compat import v1 as tf
    return learning_rate * cyclic_lr(step=global_train_step)
"""

# -------------------------- SpecAugment -------------------------- #

specaug_transform_func = """
def transform(data, network, max_time_dim={max_time_dim}, freq_dim_factor={freq_dim_factor}):
  x = data.placeholder
  from returnn.tf.compat import v1 as tf
  # summary("features", x)
  step = network.global_train_step
  step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
  step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
  def get_masked():
      x_masked = x
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
        min_num=step1 + step2, max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
        max_dims=max_time_dim)
      x_masked = random_mask(
        x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
        min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
        max_dims=data.dim // freq_dim_factor)
      #summary("features_mask", x_masked)
      return x_masked
  x = network.cond_on_train(get_masked, lambda: x)
  return x
"""

# -------------------------- Pretraining -------------------------- #


def pretrain_layers_and_dims(
        idx, net_dict: dict, encoder_args, decoder_args, variant, reduce_dims=True, initial_dim_factor=0.5,
        initial_batch_size=20000, initial_batch_size_idx=3, second_bs=None, second_bs_idx=None, enc_dec_share_grow_frac=True):

    InitialDimFactor = initial_dim_factor

    encoder_keys = ['ff_dim', 'enc_key_dim', 'conv_kernel_size']
    decoder_keys = ['ff_dim']
    encoder_args_copy = copy.deepcopy(encoder_args)
    decoder_args_copy = copy.deepcopy(decoder_args)

    final_num_blocks = encoder_args['num_blocks']

    assert final_num_blocks >= 2

    extra_net_dict = dict()
    extra_net_dict["#config"] = {}

    if idx < initial_batch_size_idx:
        extra_net_dict["#config"]["batch_size"] = initial_batch_size
    elif second_bs:
        assert second_bs_idx is not None
        if idx < second_bs_idx:
            extra_net_dict["#config"]["batch_size"] = second_bs

    idx = max(idx - 1, 0)  # repeat first 0, 0, 1, 2, ...

    if variant == 1:
        num_blocks = max(2 * idx, 1)  # 1/1/2/4/6/8/10/12 -> 8
        StartNumLayers = 1
    elif variant == 2:
        num_blocks = 2 ** idx  # 1/1/2/4/8/12 -> 6
        StartNumLayers = 1
    elif variant == 3:
        idx += 1
        num_blocks = 2 * idx  # 2/2/4/6/8/10/12 -> 7
        StartNumLayers = 2
    elif variant == 4:
        idx += 1
        num_blocks = 2 ** idx  # 2/2/4/8/12 -> 5
        StartNumLayers = 2
    elif variant == 5:
        idx += 2
        num_blocks = 2 ** idx  # 4/4/8/12 -> 4
        StartNumLayers = 4
    elif variant == 6:
        idx += 1  # 1 1 2 3
        num_blocks = 4 * idx  # 4 4 8 12 16
        StartNumLayers = 4
    else:
        raise ValueError("variant {} is not defined".format(variant))

    if num_blocks > final_num_blocks:
        return None

    transf_dec_layers = decoder_args_copy['dec_layers']
    num_transf_layers = min(num_blocks, transf_dec_layers)

    encoder_args_copy['num_blocks'] = num_blocks
    decoder_args_copy['dec_layers'] = num_transf_layers
    decoder_args_copy['label_smoothing'] = 0
    AttNumHeads = encoder_args_copy['att_num_heads']

    if reduce_dims:
        grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
        dim_frac_enc = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_enc

        if enc_dec_share_grow_frac:
            grow_frac_dec = grow_frac_enc
        else:
            grow_frac_dec = 1.0 - float(transf_dec_layers - num_transf_layers) / (transf_dec_layers - StartNumLayers)

        dim_frac_dec = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_dec
    else:
        dim_frac_enc = 1
        dim_frac_dec = 1

    if reduce_dims:
        for key in encoder_keys:
            encoder_args_copy[key] = int(encoder_args[key] * dim_frac_enc / float(AttNumHeads)) * AttNumHeads

        for key in decoder_keys:
            decoder_args_copy[key] = int(decoder_args[key] * dim_frac_dec / float(AttNumHeads)) * AttNumHeads

    # do not enable regulizations in the first pretraining step to make it more stable
    for k in encoder_args_copy.keys():
        if 'dropout' in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc
        if 'l2' in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc

    for k in decoder_args_copy.keys():
        if 'dropout' in k and decoder_args_copy[k] is not None:
            if idx <= 1:
                decoder_args_copy[k] = 0.0
            else:
                decoder_args_copy[k] *= dim_frac_dec
        if 'l2' in k and decoder_args_copy[k] is not None:
            if idx <= 1:
                decoder_args_copy[k] = 0.0
            else:
                decoder_args_copy[k] *= dim_frac_dec

    conformer_encoder = ConformerEncoder(**encoder_args_copy)
    conformer_encoder.create_network()

    transformer_decoder = TransformerDecoder(base_model=conformer_encoder, **decoder_args_copy)
    transformer_decoder.create_network()

    net_dict = conformer_encoder.network.get_net()
    net_dict.update(transformer_decoder.network.get_net())
    net_dict.update(extra_net_dict)

    return net_dict

# -------------------------------------------------------------------- #


def create_config(
        name, training_datasets, is_recog=False, input_key="audio_features", enc_layers=12, dec_layers=6, enc_key_dim=512,
        att_num_heads=8, dropout=0.1, input_layer='lstm-6', conv_kernel_size=32, ff_dim=2048, ff_init=None,
        att_dropout=0.1, lr=0.0008, wup_start_lr=0.0003, lr_decay=0.9, const_lr=0, wup=10, epoch_split=20, batch_size=10000,
        accum_grad=2, with_ctc=True, l2=0.0001, pretrain_reps=5, max_seq_length=75, noam_opts=None,
        warmup_lr_opts=None, with_pretrain=True, pretrain_opts=None, native_ctc=True, pos_enc='rel', embed_pos_enc=False,
        start_conv_init=None, mhsa_init=None, conv_module_init=None, mhsa_out_init=None, embed_dropout=0.1, speed_pert=True,
        gradient_clip_global_norm=0.0, rel_pos_clipping=16, subsample=None, dropout_in=0.1, label_smoothing=0.1,
        apply_embed_weight=False, audio_feature_opts=None, ctc_loss_scale=None, ext_lm_opts=None, beam_size=12, dec_l2=0.0,
        softmax_dropout=0.0, stoc_layers_prob=0.0, prior_lm_opts=None, gradient_noise=0.0, adamw=False, retrain_opts=None,
        decouple_constraints_factor=0.025, batch_norm_opts=None, extra_str=None, preload_from_files=None, min_lr_factor=50,
        use_ln=False, gradient_clip=0.0, specaug_str_func_opts=None, dec_pos_enc=None, self_att_l2=0.0,
        sandwich_conv=False, recursion_limit=3000):

    exp_config = copy.deepcopy(config)  # type: dict

    exp_config["extern_data"] = training_datasets.extern_data
    exp_config["train"] = training_datasets.train.as_returnn_opts()
    exp_config["dev"] = training_datasets.cv.as_returnn_opts()
    exp_config["eval_datasets"] = {'devtrain': training_datasets.devtrain.as_returnn_opts()}

    target = 'bpe_labels'

    # add default hyperparameters
    # model and learning_rates paths are added in the CRNNTraining job
    hyperparams = {
        'gradient_clip': gradient_clip,
        'accum_grad_multiple_step': accum_grad,
        'gradient_noise': gradient_noise,
        'batch_size': batch_size,
        'max_seqs': 200,
        'truncation': -1
    }
    # default: Adam optimizer
    hyperparams['adam'] = True
    hyperparams['optimizer_epsilon'] = 1e-8

    if adamw:
        hyperparams['decouple_constraints'] = True
        hyperparams['decouple_constraints_factor'] = decouple_constraints_factor

    if max_seq_length:
        hyperparams['max_seq_length'] = {target: max_seq_length}  # char-bpe
    if gradient_clip_global_norm:
        hyperparams['gradient_clip_global_norm'] = gradient_clip_global_norm

    extra_python_code = '\n'.join(['import sys', 'sys.setrecursionlimit({})'.format(recursion_limit)])  # for network construction

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
    else:  # newbob
        if retrain_opts is not None:
            learning_rates = None
        elif isinstance(const_lr, int):
            learning_rates = [wup_start_lr] * const_lr + list(numpy.linspace(wup_start_lr, lr, num=wup))
        elif isinstance(const_lr, list):
            assert len(const_lr) == 2
            learning_rates = \
                [wup_start_lr] * const_lr[0] + list(numpy.linspace(wup_start_lr, lr, num=wup)) + [lr] * const_lr[1]
        else:
            raise ValueError('unknown const_lr format')

        exp_config['learning_rate'] = lr
        exp_config['learning_rates'] = learning_rates
        exp_config['min_learning_rate'] = lr / min_lr_factor
        exp_config['learning_rate_control'] = "newbob_multi_epoch"
        exp_config['learning_rate_control_relative_error_relative_lr'] = True
        exp_config['learning_rate_control_min_num_epochs_per_new_lr'] = 3
        exp_config['use_learning_rate_control_always'] = True
        exp_config['newbob_multi_num_epochs'] = epoch_split
        exp_config['newbob_multi_update_interval'] = 1
        exp_config['newbob_learning_rate_decay'] = lr_decay

    # -------------------------- network -------------------------- #

    encoder_args = dict(
        target=target, input="data:" + input_key, num_blocks=enc_layers, enc_key_dim=enc_key_dim, att_num_heads=att_num_heads, dropout=dropout,
        att_dropout=att_dropout, ff_init=ff_init, ff_dim=ff_dim, input_layer=input_layer, conv_kernel_size=conv_kernel_size,
        with_ctc=with_ctc, pos_enc=pos_enc, native_ctc=native_ctc, l2=l2, lstm_dropout=dropout, start_conv_init=start_conv_init,
        rel_pos_clipping=rel_pos_clipping, subsample=subsample, dropout_in=dropout_in, mhsa_init=mhsa_init,
        conv_module_init=conv_module_init, mhsa_out_init=mhsa_out_init, ctc_loss_scale=ctc_loss_scale,
        stoc_layers_prob=stoc_layers_prob, batch_norm_opts=batch_norm_opts, use_ln=use_ln,
        self_att_l2=self_att_l2, sandwich_conv=sandwich_conv)

    conformer_encoder = ConformerEncoder(**encoder_args)
    conformer_encoder.create_network()

    decoder_args = dict(
        target=target, dec_layers=dec_layers, att_dropout=att_dropout, embed_dropout=embed_dropout,
        dropout=dropout, att_num_heads=att_num_heads, label_smoothing=label_smoothing, embed_pos_enc=embed_pos_enc,
        ff_dim=ff_dim, ff_init=ff_init, apply_embed_weight=apply_embed_weight, mhsa_init=mhsa_init,
        mhsa_out_init=mhsa_out_init, l2=dec_l2, softmax_dropout=softmax_dropout, beam_size=beam_size,
        pos_enc=dec_pos_enc, rel_pos_clipping=rel_pos_clipping)

    transformer_decoder = TransformerDecoder(base_model=conformer_encoder, **decoder_args)
    transformer_decoder.create_network()

    decision_layer_name = transformer_decoder.decision_layer_name
    exp_config['search_output_layer'] = decision_layer_name

    if ext_lm_opts:
        transformer_decoder = ExternalLMDecoder(
            transformer_decoder, ext_lm_opts, prior_lm_opts=prior_lm_opts, beam_size=beam_size, dec_type='transformer')
        transformer_decoder.create_network()

    # add full network
    exp_config['network'] = conformer_encoder.network.get_net()  # type: dict
    exp_config['network'].update(transformer_decoder.network.get_net())

    # -------------------------- end network -------------------------- #

    # add hyperparmas
    exp_config.update(hyperparams)

    if retrain_opts is not None:
        retrain_model_name = retrain_opts.get('model_name', name)
        idx = retrain_opts['idx']
        if 'model' in retrain_opts:
            model = retrain_opts['model']
            retrain_epoch = model.split('.')[-1]
        else:
            assert 'epoch' in retrain_opts
            retrain_epoch = retrain_opts['epoch']
            model = name_to_train_job[retrain_model_name].checkpoints[retrain_epoch].ckpt_path
        exp_config['import_model_train_epoch1'] = model
        name += '-retrain{}_ep{}'.format(idx, retrain_epoch)

    if ext_lm_opts and ext_lm_opts.get('preload_from_files'):
        assert 'preload_from_files' not in exp_config
        exp_config['preload_from_files'] = copy.deepcopy(ext_lm_opts['preload_from_files'])

    if preload_from_files:
        if 'preload_from_files' not in exp_config:
            exp_config['preload_from_files'] = {}
        exp_config['preload_from_files'].update(preload_from_files)

    if specaug_str_func_opts:
        python_prolog = specaugment.specaug_wo_transform.get_funcs()
        extra_python_code += '\n' + specaug_transform_func.format(**specaug_str_func_opts)
    else:
        python_prolog = specaugment.specaug_tf2.get_funcs()  # type: list

    if speed_pert:
        python_prolog += [data_aug.speed_pert]

    # add pretraining
    if with_pretrain and ext_lm_opts is None and retrain_opts is None:

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

    if extra_str:
        extra_python_code += '\n' + extra_str

    returnn_config = ReturnnConfig(
        exp_config, post_config=post_config, python_prolog=python_prolog, python_epilog=extra_python_code, hash_full_python_code=True,
        pprint_kwargs={'sort_dicts': False})

    return returnn_config