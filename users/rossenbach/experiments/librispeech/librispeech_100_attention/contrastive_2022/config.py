import copy
import dataclasses
import numpy

from i6_core.returnn.config import CodeWrapper, ReturnnConfig

from . import zeineldeen_helpers as helpers

from .zeineldeen_helpers.models.asr.encoder.rnn_encoder import RNNEncoder
from .zeineldeen_helpers.legacy.rnn_decoder import RNNDecoder
#from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM
#from .zeineldeen_helpers.models.lm import generic_lm
from .zeineldeen_helpers.pretrain.blstm import blstm_pretrain3, blstm_pretrain_contrastive_loss
from .zeineldeen_helpers import specaugment

from .cl_variant import create_cl_variant

# changing these does not change the hash
post_config = {
    'use_tensorflow': True,
    'tf_log_memory_usage': True,
    'cleanup_old_models': True,
    'log_batch_size': True,
    'debug_print_layer_output_template': True,
    'cache_size': '0',
}

wup_start_lr = 0.0003
initial_lr = 0.0008

learning_rates = [wup_start_lr] * 10 + list(numpy.linspace(wup_start_lr, initial_lr, num=10))

config_template = {
    'gradient_clip': 0,
    'optimizer': {'class': 'Adam', 'epsilon': 1e-8},
    'accum_grad_multiple_step': 2,
    'gradient_noise': 0.0,
    'learning_rates': learning_rates,
    'min_learning_rate': 0.00001,
    'learning_rate_control': "newbob_multi_epoch",
    'learning_rate_control_relative_error_relative_lr': True,
    'learning_rate_control_min_num_epochs_per_new_lr': 3,
    'use_learning_rate_control_always': True,
    'newbob_multi_num_epochs': 3,
    'newbob_multi_update_interval': 1,
    'newbob_learning_rate_decay': 0.9,
    'batch_size': 10000,
    'max_seqs': 200,
    # 'truncation': -1
}

target = "bpe_labels"
input = "data:audio_features"


@dataclasses.dataclass()
class NetworkOptions:
    rec_weight_dropout: float = 0.3
    l2: float = 0.001
    dec_zoneout: bool = True
    embed_dropout: float = 0.3
    att_dropout: float = 0.3
    lstm_dropout: float = 0.3
    softmax_dropout: float = 0.3
    label_smoothing: float = 0.1
    with_ctc: bool = True
    dec_lstm_num_units: int = 1000
    enc_lstm_dim: int = 1024
    enc_key_dim: int = 1024
    enc_value_dim: int = 2048
    pool_sizes: str = '3_2'
    beam_size: int = 12
    with_conv: bool = True
    prior_lm_opts: dict = None
    remove_softmax_bias: bool = False,
    relax_att_scale: float = None
    ctc_loss_scale: float = None
    ce_loss_scale: float = None
    conv_time_pooling: str = None

    ext_lm_opts=None
    coverage_term_scale=None
    local_fusion_opts=None



def create_network(options: NetworkOptions):
    rnn_encoder = RNNEncoder(
        input=input,
        target=target, rec_weight_dropout=options.rec_weight_dropout, l2=options.l2, with_ctc=options.with_ctc, dropout=options.lstm_dropout,
        lstm_dim=options.enc_lstm_dim, enc_key_dim=options.enc_key_dim, enc_value_dim=options.enc_value_dim, pool_sizes=options.pool_sizes,
        with_conv=options.with_conv, ctc_loss_scale=options.ctc_loss_scale, conv_time_pooling=options.conv_time_pooling)
    rnn_encoder.create_network()

    rnn_decoder = RNNDecoder(
        base_model=rnn_encoder, target=target, l2=options.l2, dec_zoneout=options.dec_zoneout,
        att_dropout=options.att_dropout, embed_dropout=options.embed_dropout, dropout=options.softmax_dropout, label_smoothing=options.label_smoothing,
        dec_lstm_num_units=options.dec_lstm_num_units, beam_size=options.beam_size, ext_lm_opts=options.ext_lm_opts, prior_lm_opts=options.prior_lm_opts,
        coverage_term_scale=options.coverage_term_scale, remove_softmax_bias=options.remove_softmax_bias,
        local_fusion_opts=options.local_fusion_opts, relax_att_scale=options.relax_att_scale, ce_loss_scale=options.ce_loss_scale)
    rnn_decoder.create_network()

    # add full network
    network = rnn_encoder.network.get_net()
    network.update(rnn_decoder.network.get_net())

    return network, rnn_encoder, rnn_decoder


def create_config(
        training_datasets,
        network_options: NetworkOptions,
        pretrain_reps=5,
        retrain_opts=None,
        preload_from_files=None, with_pretrain=True, extra_str=None, coverage_term_scale=None, coverage_term_thre=None,
        local_fusion_opts=None,
        contrastive_loss_opts=None,
        **kwargs):

    config = copy.deepcopy(config_template)

    config["extern_data"] = training_datasets.extern_data
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()

    # -------------------------- network -------------------------- #
    network, rnn_encoder, rnn_decoder = create_network(network_options)
    config['network'] = network

    # this is needed for pretrain custom construction function
    config['AttNumHeads'] = rnn_encoder.att_num_heads
    config['EncValuePerHeadDim'] = rnn_encoder.enc_val_per_head_dim

    decision_layer_name = rnn_decoder.decision_layer_name
    config['search_output_layer'] = decision_layer_name

    create_cl_variant(contrastive_loss_opts, config)

    # -------------------------- end network -------------------------- #

    if network_options.ext_lm_opts and network_options.ext_lm_opts.get('preload_from_files'):
        assert 'preload_from_files' not in config
        config['preload_from_files'] = copy.deepcopy(network_options.ext_lm_opts['preload_from_files'])

    if preload_from_files:
        if 'preload_from_files' not in config:
            config['preload_from_files'] = {}
        config['preload_from_files'].update(preload_from_files)

    if retrain_opts is not None:
        model = retrain_opts['model']
        config['import_model_train_epoch1'] = model
        config['learning_rates'] = [initial_lr]

    if contrastive_loss_opts:
        python_prolog = [
            "from returnn.tf.util.data import batch_dim",
            "from returnn.tf.util.data import SpatialDim",
            "from returnn.tf.util.data import FeatureDim"
        ]
        if contrastive_loss_opts.get('masking_method', None) == 'specaug':
            python_prolog += specaugment.contrastive_loss_as_specaug.get_funcs()
        else:
            python_prolog += specaugment.specaug_and_contrastive_loss.get_funcs()
    else:
        python_prolog = specaugment.simple_specaug.get_funcs()

    # add pretraining
    if retrain_opts is None and with_pretrain:
        config['pretrain'] = {
            "repetitions": pretrain_reps,
            "copy_param_mode": "subset",
            "construction_algo": CodeWrapper("custom_construction_algo")  # add to config as-is
        }
        if contrastive_loss_opts:
            python_prolog += blstm_pretrain_contrastive_loss.get_funcs()
        else:
            python_prolog += blstm_pretrain3.get_funcs()

    extra_python_code = ''

    #if local_fusion_opts:
    #    lm_scale = local_fusion_opts['lm_scale']
    #    am_scale = local_fusion_opts['am_scale']
    #    extra_python_code += '\n' + local_fusion_norm.format(lm_scale=lm_scale, am_scale=am_scale)

    if kwargs.get('recursion_limit', None):
        extra_python_code += '\n'.join(['import sys', 'sys.setrecursionlimit({})'.format(kwargs['recursion_limit'])])

    if extra_str:
        extra_python_code += '\n' + extra_str

    if coverage_term_scale:
        layers = {
            'accum_w': {
                'class': 'eval', 'from': ['prev:att_weights', 'att_weights'], 'eval': 'source(0) + source(1)'
            },  # [B, enc-T, 1]
            'merge_accum_w': {'class': 'merge_dims', 'from': 'accum_w', 'axes': 'except_batch'},  # [B, enc-T]
            'coverage_mask': {'class': 'compare', 'from': 'merge_accum_w', 'kind': 'greater', 'value': coverage_term_thre},
            'float_coverage_mask': {'class': 'cast', 'from': 'coverage_mask', 'dtype': 'float32'},
            'accum_coverage': {'class': 'reduce', 'mode': 'sum', 'from': 'float_coverage_mask', 'axis': -1}  # [B]
        }

        # add layers to subnetwork
        for k, v in layers.items():
            config['network']['output']['unit'][k] = v

    if kwargs.get('all_layers_in_loop', False):
        config['optimize_move_layers_out'] = False

    returnn_config = ReturnnConfig(
        config, post_config=post_config, python_prolog=python_prolog,
        python_epilog=extra_python_code, pprint_kwargs={'sort_dicts': False})

    return returnn_config