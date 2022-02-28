import copy
import dataclasses
import numpy

from i6_core.returnn.config import CodeWrapper, ReturnnConfig

from . import zeineldeen_helpers as helpers

from .zeineldeen_helpers.models.asr.encoder.rnn_encoder import RNNEncoder
from .zeineldeen_helpers.models.asr.encoder.conformer_encoder import ConformerEncoder
from .zeineldeen_helpers.legacy.rnn_decoder import RNNDecoder
from .zeineldeen_helpers.models.asr.decoder.transformer_decoder import TransformerDecoder
#from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM
#from .zeineldeen_helpers.models.lm import generic_lm
from .zeineldeen_helpers.pretrain.blstm import blstm_pretrain3, blstm_pretrain_contrastive_loss
from .zeineldeen_helpers.pretrain.conformer import conformer_transformer_pretrain
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
    "load_ignore_missing_vars": True,
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
    l2: float = 0.001
    embed_dropout: float = 0.3
    att_dropout: float = 0.3
    softmax_dropout: float = 0.3
    label_smoothing: float = 0.1
    with_ctc: bool = True
    enc_lstm_dim: int = 1024
    enc_key_dim: int = 1024
    enc_value_dim: int = 2048
    pool_sizes: str = '3_2'
    with_conv: bool = True
    prior_lm_opts: dict = None
    remove_softmax_bias: bool = False,
    relax_att_scale: float = None
    ctc_loss_scale: float = None
    ce_loss_scale: float = None
    conv_time_pooling: str = None

    beam_size: int = 12

    ext_lm_opts=None
    coverage_term_scale=None
    local_fusion_opts=None


# -------------------------- Pretraining -------------------------- #


def pretrain_layers_and_dims(
        idx, net_dict: dict, encoder_args, decoder_args, variant, reduce_dims=True, initial_dim_factor=0.5,
        initial_batch_size=20000, enc_dec_share_grow_frac=True):

    InitialDimFactor = initial_dim_factor

    encoder_keys = ['ff_dim', 'enc_key_dim', 'conv_kernel_size']
    decoder_keys = ['ff_dim']
    encoder_args_copy = copy.deepcopy(encoder_args)
    decoder_args_copy = copy.deepcopy(decoder_args)

    final_num_blocks = encoder_args['num_blocks']

    assert final_num_blocks >= 2

    extra_net_dict = dict()
    extra_net_dict["#config"] = {}

    if idx < 3:
        extra_net_dict["#config"]["batch_size"] = initial_batch_size

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

@dataclasses.dataclass()
class EncoderOptions():
    input: str = "data:audio_features"
    num_blocks: int = 12
    enc_key_dim: int = 512
    att_num_heads: int = 8
    dropout: float = 0.1
    input_layer: str = 'lstm-6'
    conv_kernel_size: int = 32
    ff_dim: int = 2048
    ff_init: str = None
    att_dropout: float = 0.1
    with_ctc: bool = True
    l2: float = 0.0001
    native_ctc: bool = True
    pos_enc: str = 'rel'
    start_conv_init: str = None
    mhsa_init: str = None
    conv_module_init: str = None
    mhsa_out_init: str = None
    rel_pos_clipping: int = 16
    subsample: str = None
    dropout_in: float = 0.1
    ctc_loss_scale: float = None
    stoc_layers_prob:float = 0.0
    batch_norm_opts: dict = None
    use_ln: bool = False
    self_att_l2=0.0,
    sandwich_conv=False

    target: str = "bpe_labels"

@dataclasses.dataclass()
class DecoderOptions():
    target: str = "bpe_labels"
    dec_layers: int = 6
    att_num_heads: int = 8
    dropout: float = 0.1
    ff_dim: int = 2048
    ff_init: str = None
    att_dropout: float = 0.1
    embed_dropout: float = 0.1
    l2: float = 0.0001
    mhsa_init: str = None
    mhsa_out_init: str = None
    rel_pos_clipping: int = 16
    apply_embed_weight: bool = False
    softmax_dropout: float = 0.0
    beam_size: int = 12
    pos_enc: str = None


def create_network(
        encoder_options: EncoderOptions,
        decoder_options: DecoderOptions
    ):

    rnn_encoder = ConformerEncoder(
        **dataclasses.asdict(encoder_options)
    )

    rnn_decoder = TransformerDecoder(base_model=rnn_encoder, **dataclasses.asdict(decoder_options))

    rnn_encoder.create_network()
    rnn_decoder.create_network()
    # add full network
    network = rnn_encoder.network.get_net()
    network.update(rnn_decoder.network.get_net())

    return network, rnn_encoder, rnn_decoder


def create_config(
        training_datasets,
        encoder_options: EncoderOptions,
        decoder_options: DecoderOptions,
        pretrain_reps=6,
        retrain_opts=None,
        ext_lm_opts: dict = None,
        preload_from_files=None, with_pretrain=True, extra_str=None, coverage_term_scale=None, coverage_term_thre=None,
        local_fusion_opts=None,
        contrastive_loss_opts=None,
        behavior_version=None,
        specaug_str_func_opts=None,
        **kwargs):

    config = copy.deepcopy(config_template)

    if behavior_version:
        config['behavior_version'] = behavior_version

    config["extern_data"] = training_datasets.extern_data
    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()

    # -------------------------- network -------------------------- #
    network, rnn_encoder, rnn_decoder = create_network(encoder_options=encoder_options, decoder_options=decoder_options)
    config['network'] = network

    # this is needed for pretrain custom construction function
    config['AttNumHeads'] = rnn_encoder.att_num_heads
    config['EncValuePerHeadDim'] = rnn_encoder.enc_val_per_head_dim

    decision_layer_name = rnn_decoder.decision_layer_name
    config['search_output_layer'] = decision_layer_name

    create_cl_variant(contrastive_loss_opts, config)


    # update trafo
    config['att_kv_feat_dim'] = CodeWrapper("FeatureDim(\"att_kv_feat_dim\", 64)")  # 512/8
    for n in range(1, decoder_options.dec_layers + 1):

        # update keys
        config['network']['transformer_decoder_%02i_att_key_' % n] = \
            copy.deepcopy(config['network']['transformer_decoder_%02i_att_key' % n])
        config['network']['transformer_decoder_%02i_att_key' % n] = {
            'class': 'reinterpret_data', 'from': 'transformer_decoder_%02i_att_key_' % n,
            'set_dim_tags': {'F': CodeWrapper('att_kv_feat_dim')}
        }

        # # update queries
        config['network']['output']['unit']['transformer_decoder_%02i_att_query_' % n] = \
            copy.deepcopy(config['network']['output']['unit']['transformer_decoder_%02i_att_query' % n])
        config['network']['output']['unit']['transformer_decoder_%02i_att_query' % n] = {
            'class': 'reinterpret_data', 'from': 'transformer_decoder_%02i_att_query_' % n,
            'set_dim_tags': {'F': CodeWrapper('att_kv_feat_dim')}
        }

        # update dot layers
        config['network']['output']['unit']['transformer_decoder_%02i_att_energy' % n] = {
            'class': 'dot', 'from': ["base:transformer_decoder_%02i_att_key" % n, 'transformer_decoder_%02i_att_query' % n],
            'reduce': CodeWrapper('att_kv_feat_dim')
        }


    # -------------------------- end network -------------------------- #

    if ext_lm_opts and ext_lm_opts.get('preload_from_files'):
        assert 'preload_from_files' not in config
        config['preload_from_files'] = copy.deepcopy(ext_lm_opts['preload_from_files'])

    if preload_from_files:
        if 'preload_from_files' not in config:
            config['preload_from_files'] = {}
        config['preload_from_files'].update(preload_from_files)

    if retrain_opts is not None:
        model = retrain_opts['model']
        config['import_model_train_epoch1'] = model
        config['learning_rates'] = [initial_lr]

    python_prolog = ["from returnn.tf.util.data import FeatureDim"]
    if contrastive_loss_opts:
        python_prolog += [
            "from returnn.tf.util.data import batch_dim",
            "from returnn.tf.util.data import SpatialDim",
        ]
        python_prolog += specaugment.conformer_specaug_wo_time.get_funcs()
        python_prolog += [get_contrastive_loss_mask.format(**contrastive_loss_opts)]
    else:
        if specaug_str_func_opts:
            python_prolog += specaugment.specaug_wo_transform.get_funcs()
            extra_python_code += '\n' + specaug_transform_func.format(**specaug_str_func_opts)
        else:
            python_prolog += specaugment.specaug_tf2.get_funcs()  # type: list

    # add pretraining
    if with_pretrain and ext_lm_opts is None and retrain_opts is None:

        pretrain_opts = {'variant': 4}

        pretrain_networks = []
        idx = 0
        while True:
            net = pretrain_layers_and_dims(idx, config['network'], dataclasses.asdict(encoder_options), dataclasses.asdict(decoder_options), **pretrain_opts)
            if not net:
                break
            pretrain_networks.append(net)
            idx += 1

        config['pretrain_nets_lookup'] = {k: v for k, v in enumerate(pretrain_networks)}

        config['pretrain'] = {
            "repetitions": pretrain_reps,
            "copy_param_mode": "subset",
            "construction_algo": CodeWrapper('custom_construction_algo')
        }

        pretrain_algo_str = 'def custom_construction_algo(idx, net_dict):\n\treturn pretrain_nets_lookup.get(idx, None)'
        python_prolog += [pretrain_algo_str]



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
