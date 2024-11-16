import numpy
import copy
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder import (
    ConformerEncoder,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.transformer_decoder import (
    TransformerDecoder,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.rnn_decoder import RNNDecoder
from i6_experiments.users.zeineldeen.models.lm.external_lm_decoder import (
    ExternalLMDecoder,
)

from i6_experiments.users.zeineldeen import data_aug
from i6_experiments.users.zeineldeen.data_aug import specaugment

from i6_core.returnn.config import ReturnnConfig, CodeWrapper


# -------------------------- Base Config -------------------------- #

config = {}

# changing these does not change the hash
post_config = {
    "use_tensorflow": True,
    "tf_log_memory_usage": True,
    "cleanup_old_models": True,
    "log_batch_size": True,
    "debug_print_layer_output_template": True,
    "debug_mode": False,
    "batching": "random",
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

oclr_str = """
def dynamic_learning_rate(*, network, global_train_step, learning_rate, **kwargs):
      initialLR  = {initial_lr}
      peakLR     = {peak_lr}
      finalLR    = {final_lr}
      cycleEpoch = {cycle_ep}
      totalEpoch = {total_ep}
      nStep      = {n_step}
    
      steps     = cycleEpoch * nStep
      stepSize  = (peakLR - initialLR) / steps
      steps2    = (totalEpoch - 2 * cycleEpoch) * nStep
      stepSize2 = (initialLR - finalLR) / steps2
    
      import tensorflow as tf
      n = tf.cast(global_train_step, tf.float32)
      return tf.where(global_train_step <= steps, initialLR + stepSize * n,
                 tf.where(global_train_step <= 2*steps, peakLR - stepSize * (n - steps), 
                     tf.maximum(initialLR - stepSize2 * (n - 2*steps), finalLR)))
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
    idx,
    net_dict: dict,
    encoder_type,
    decoder_type,
    encoder_args,
    decoder_args,
    variant,
    reduce_dims=True,
    initial_dim_factor=0.5,
    initial_batch_size=20000 * 160,
    initial_batch_size_idx=3,
    second_bs=None,
    second_bs_idx=None,
    enc_dec_share_grow_frac=True,
    repeat_first=True,
):
    """
    Pretraining implementation that works for multiple encoder/decoder combinations

    :param idx:
    :param net_dict:
    :param encoder_type:
    :param decoder_type:
    :param encoder_args:
    :param decoder_args:
    :param variant:
    :param reduce_dims:
    :param initial_dim_factor:
    :param initial_batch_size:
    :param initial_batch_size_idx:
    :param second_bs:
    :param second_bs_idx:
    :param enc_dec_share_grow_frac:
    :return:
    """

    InitialDimFactor = initial_dim_factor

    encoder_keys = ["ff_dim", "enc_key_dim", "conv_kernel_size"]
    decoder_keys = ["ff_dim"]
    encoder_args_copy = copy.deepcopy(encoder_args)
    decoder_args_copy = copy.deepcopy(decoder_args)

    final_num_blocks = encoder_args["num_blocks"]

    assert final_num_blocks >= 2

    extra_net_dict = dict()
    extra_net_dict["#config"] = {}

    if idx < initial_batch_size_idx:
        extra_net_dict["#config"]["batch_size"] = initial_batch_size
    elif second_bs:
        assert second_bs_idx is not None
        if idx < second_bs_idx:
            extra_net_dict["#config"]["batch_size"] = second_bs

    if repeat_first:
        idx = max(idx - 1, 0)  # repeat first 0, 0, 1, 2, ...

    if variant == 1:
        num_blocks = max(2 * idx, 1)  # 1/1/2/4/6/8/10/12 -> 8
        StartNumLayers = 1
    elif variant == 2:
        num_blocks = 2**idx  # 1/1/2/4/8/12 -> 6
        StartNumLayers = 1
    elif variant == 3:
        idx += 1
        num_blocks = 2 * idx  # 2/2/4/6/8/10/12 -> 7
        StartNumLayers = 2
    elif variant == 4:
        idx += 1
        num_blocks = 2**idx  # 2/2/4/8/12 -> 5
        StartNumLayers = 2
    elif variant == 5:
        idx += 2
        num_blocks = 2**idx  # 4/4/8/12 -> 4
        StartNumLayers = 4
    elif variant == 6:
        idx += 1  # 1 1 2 3
        num_blocks = 4 * idx  # 4 4 8 12 16
        StartNumLayers = 4
    else:
        raise ValueError("variant {} is not defined".format(variant))

    if num_blocks > final_num_blocks:
        return None

    encoder_args_copy["num_blocks"] = num_blocks
    decoder_args_copy["label_smoothing"] = 0
    AttNumHeads = encoder_args_copy["att_num_heads"]

    if reduce_dims:
        grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (
            final_num_blocks - StartNumLayers
        )
        dim_frac_enc = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_enc

        for key in encoder_keys:
            encoder_args_copy[key] = (
                int(encoder_args[key] * dim_frac_enc / float(AttNumHeads)) * AttNumHeads
            )

        if decoder_type == TransformerDecoder:
            transf_dec_layers = decoder_args_copy["num_layers"]
            num_transf_layers = min(num_blocks, transf_dec_layers)
            decoder_args_copy["num_layers"] = num_transf_layers

            if enc_dec_share_grow_frac:
                grow_frac_dec = grow_frac_enc
            else:
                grow_frac_dec = 1.0 - float(transf_dec_layers - num_transf_layers) / (
                    transf_dec_layers - StartNumLayers
                )

            dim_frac_dec = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_dec

            for key in decoder_keys:
                decoder_args_copy[key] = (
                    int(decoder_args[key] * dim_frac_dec / float(AttNumHeads))
                    * AttNumHeads
                )
        else:
            dim_frac_dec = 1
    else:
        dim_frac_enc = 1
        dim_frac_dec = 1

    # do not enable regulizations in the first pretraining step to make it more stable
    for k in encoder_args_copy.keys():
        if "dropout" in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc
        if "l2" in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc

    for k in decoder_args_copy.keys():
        if "dropout" in k and decoder_args_copy[k] is not None:
            if idx <= 1:
                decoder_args_copy[k] = 0.0
            else:
                decoder_args_copy[k] *= dim_frac_dec
        if "l2" in k and decoder_args_copy[k] is not None:
            if idx <= 1:
                decoder_args_copy[k] = 0.0
            else:
                decoder_args_copy[k] *= dim_frac_dec

    conformer_encoder = encoder_type(**encoder_args_copy)
    conformer_encoder.create_network()

    transformer_decoder = decoder_type(
        base_model=conformer_encoder, **decoder_args_copy
    )
    transformer_decoder.create_network()

    net_dict = conformer_encoder.network.get_net()
    net_dict.update(transformer_decoder.network.get_net())
    net_dict.update(extra_net_dict)

    return net_dict


# -------------------------------------------------------------------- #


class EncoderArgs:
    pass


@dataclass
class ConformerEncoderArgs(EncoderArgs):
    num_blocks: int = 12
    enc_key_dim: int = 512
    att_num_heads: int = 8
    ff_dim: int = 2048
    conv_kernel_size: int = 32
    input_layer: str = "lstm-6"
    input_layer_conv_act: str = "relu"
    pos_enc: str = "rel"

    sandwich_conv: bool = False
    subsample: Optional[str] = None

    # ctc
    with_ctc: bool = True
    native_ctc: bool = True
    ctc_loss_scale: Optional[float] = None

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None
    conv_module_init: Optional[str] = None
    start_conv_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    dropout_in: float = 0.1
    att_dropout: float = 0.1
    lstm_dropout: float = 0.1

    # norms
    batch_norm_opts: Optional[Dict[str, Any]] = None
    use_ln: bool = False

    # other regularization
    l2: float = 0.0001
    self_att_l2: float = 0.0
    rel_pos_clipping: int = 16

    use_sqrd_relu: bool = False


class DecoderArgs:
    pass


@dataclass
class TransformerDecoderArgs(DecoderArgs):
    num_layers: int = 6
    att_num_heads: int = 8
    ff_dim: int = 2048
    ff_act: str = 'relu'
    pos_enc: Optional[str] = None
    embed_pos_enc: bool = False

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    att_dropout: float = 0.1
    embed_dropout: float = 0.1
    softmax_dropout: float = 0.0

    # other regularization
    l2: float = 0.0
    rel_pos_clipping: int = 16
    label_smoothing: float = 0.1
    apply_embed_weight: bool = False

    length_normalization: bool = True

    replace_cross_att_w_masked_self_att: bool = False
    create_ilm_decoder: bool = False
    ilm_type: bool = None
    ilm_args: Optional[dict] = None

@dataclass
class RNNDecoderArgs(DecoderArgs):
    att_num_heads: int = 1
    lstm_num_units: int = 1024
    output_num_units: int = 1024
    embed_dim: int = 640
    enc_key_dim: int = 1024  # also attention dim  # also attention dim

    # location feedback
    loc_conv_att_num_channels: Optional[int] = None
    loc_conv_att_filter_size: Optional[int] = None

    # param init
    lstm_weights_init: Optional[str] = None
    embed_weight_init: Optional[str] = None

    # dropout
    dropout: float = 0.0
    softmax_dropout: float = 0.3
    att_dropout: float = 0.0
    embed_dropout: float = 0.1
    rec_weight_dropout: float = 0.0

    # other regularization
    l2: float = 0.0001
    zoneout: bool = True
    reduceout: bool = True

    # lstm lm
    lstm_lm_proj_dim: int = 1024
    lstm_lm_dim: int = 1024
    add_lstm_lm: bool = False

    length_normalization: bool = True

    coverage_scale: float = None
    coverage_threshold: float = None

def create_config(
    training_datasets,
    encoder_args: EncoderArgs,
    decoder_args: DecoderArgs,
    with_staged_network=False,
    is_recog=False,
    input_key="audio_features",
    lr=0.0008,
    wup_start_lr=0.0003,
    lr_decay=0.9,
    const_lr=0,
    wup=10,
    epoch_split=20,
    batch_size=10000,
    accum_grad=2,
    pretrain_reps=5,
    max_seq_length=75,
    noam_opts=None,
    warmup_lr_opts=None,
    with_pretrain=True,
    pretrain_opts=None,
    speed_pert=True,
    oclr_opts=None,
    gradient_clip_global_norm=0.0,
    gradient_clip=0.0,
    ext_lm_opts=None,
    beam_size=12,
    recog_epochs=None,
    prior_lm_opts=None,
    gradient_noise=0.0,
    adamw=False,
    retrain_checkpoint=None,
    decouple_constraints_factor=0.025,
    extra_str=None,
    preload_from_files=None,
    min_lr_factor=50,
    specaug_str_func_opts=None,
    recursion_limit=3000,
    feature_extraction_net=None,
    config_override=None,
    feature_extraction_net_global_norm=False,
    freeze_bn=False,
    keep_all_epochs=False,
    allow_lr_scheduling=True,
    learning_rates_list=None,
    min_lr=None,
):

    exp_config = copy.deepcopy(config)  # type: dict
    exp_post_config = copy.deepcopy(post_config)

    exp_config["extern_data"] = training_datasets.extern_data

    if not is_recog:
        exp_config["train"] = training_datasets.train.as_returnn_opts()
        exp_config["dev"] = training_datasets.cv.as_returnn_opts()
        exp_config["eval_datasets"] = {
            "devtrain": training_datasets.devtrain.as_returnn_opts()
        }

    target = "bpe_labels"

    # add default hyperparameters
    # model and learning_rates paths are added in the CRNNTraining job
    hyperparams = {
        "gradient_clip": gradient_clip,
        "accum_grad_multiple_step": accum_grad,
        "gradient_noise": gradient_noise,
        "batch_size": batch_size,
        "max_seqs": 200,
        "truncation": -1,
    }
    # default: Adam optimizer
    hyperparams["adam"] = True
    hyperparams["optimizer_epsilon"] = 1e-8

    if adamw:
        hyperparams["decouple_constraints"] = True
        hyperparams["decouple_constraints_factor"] = decouple_constraints_factor

    if max_seq_length:
        hyperparams["max_seq_length"] = {target: max_seq_length}  # char-bpe
    if gradient_clip_global_norm:
        hyperparams["gradient_clip_global_norm"] = gradient_clip_global_norm

    extra_python_code = "\n".join(
        ["import sys", "sys.setrecursionlimit({})".format(recursion_limit)]
    )  # for network construction

    # LR scheduling
    if noam_opts and retrain_checkpoint is None and allow_lr_scheduling:
        noam_opts["model_d"] = encoder_args.enc_key_dim
        exp_config["learning_rate"] = noam_opts["lr"]
        exp_config["learning_rate_control"] = "constant"
        extra_python_code += "\n" + noam_lr_str.format(**noam_opts)
    elif warmup_lr_opts and retrain_checkpoint is None and allow_lr_scheduling:
        if warmup_lr_opts.get("learning_rates", None):
            exp_config["learning_rates"] = warmup_lr_opts["learning_rates"]
        exp_config["learning_rate"] = warmup_lr_opts["peak_lr"]
        exp_config["learning_rate_control"] = "constant"
        extra_python_code += "\n" + warmup_lr_str.format(**warmup_lr_opts)
    elif oclr_opts and retrain_checkpoint is None and allow_lr_scheduling:
        if oclr_opts.get("learning_rates", None):
            exp_config["learning_rates"] = oclr_opts["learning_rates"]
        exp_config["learning_rate"] = oclr_opts["peak_lr"]
        exp_config["learning_rate_control"] = "constant"
        oclr_peak_lr = oclr_opts["peak_lr"]
        oclr_initial_lr = oclr_peak_lr / 10
        extra_python_code += "\n" + oclr_str.format(
            **oclr_opts, initial_lr=oclr_initial_lr
        )
    else:  # newbob
        if learning_rates_list:
            learning_rates = learning_rates_list
        else:
            if const_lr is None:
                const_lr = 0
            if retrain_checkpoint is not None:
                learning_rates = None
            elif not allow_lr_scheduling:
                learning_rates = None
            elif isinstance(const_lr, int):
                learning_rates = [wup_start_lr] * const_lr + list(
                    numpy.linspace(wup_start_lr, lr, num=wup)
                )
            elif isinstance(const_lr, list):
                assert len(const_lr) == 2
                learning_rates = (
                    [wup_start_lr] * const_lr[0]
                    + list(numpy.linspace(wup_start_lr, lr, num=wup))
                    + [lr] * const_lr[1]
                )
            else:
                raise ValueError("unknown const_lr format")

        exp_config["learning_rate"] = lr
        exp_config["learning_rates"] = learning_rates
        exp_config["min_learning_rate"] = lr / min_lr_factor if min_lr is None else min_lr
        exp_config["learning_rate_control"] = "newbob_multi_epoch"
        exp_config["learning_rate_control_relative_error_relative_lr"] = True
        exp_config["learning_rate_control_min_num_epochs_per_new_lr"] = 3
        exp_config["use_learning_rate_control_always"] = True
        exp_config["newbob_multi_num_epochs"] = epoch_split
        exp_config["newbob_multi_update_interval"] = 1
        exp_config["newbob_learning_rate_decay"] = lr_decay

    # -------------------------- network -------------------------- #
    encoder_type = None
    if isinstance(encoder_args, ConformerEncoderArgs):
        encoder_type = ConformerEncoder

    if isinstance(decoder_args, TransformerDecoderArgs):
        decoder_type = TransformerDecoder
        dec_type = 'transformer'
    elif isinstance(decoder_args, RNNDecoderArgs):
        decoder_type = RNNDecoder
        dec_type = 'lstm'
    else:
        assert False, "invalid decoder_args type"

    encoder_args = asdict(encoder_args)
    if feature_extraction_net:
        encoder_args.update({"target": target, "input": "log_mel_features"})
    else:
        encoder_args.update({"target": target, "input": "data:" + input_key})

    if freeze_bn:
        # freeze BN during training (e.g when retraining.)
        encoder_args['batch_norm_opts'] = {'momentum': 0.0, 'use_sample': 1.0}

    conformer_encoder = encoder_type(**encoder_args)
    conformer_encoder.create_network()

    decoder_args = asdict(decoder_args)
    decoder_args.update({"target": target, "beam_size": beam_size})

    transformer_decoder = decoder_type(base_model=conformer_encoder, **decoder_args)
    transformer_decoder.create_network()

    decision_layer_name = transformer_decoder.decision_layer_name
    exp_config["search_output_layer"] = decision_layer_name

    if ext_lm_opts:
        transformer_decoder = ExternalLMDecoder(
            transformer_decoder,
            ext_lm_opts,
            prior_lm_opts=prior_lm_opts,
            beam_size=beam_size,
            dec_type=dec_type,
            length_normalization=decoder_args['length_normalization'],
        )
        transformer_decoder.create_network()

    # add full network
    exp_config["network"] = conformer_encoder.network.get_net()  # type: dict
    exp_config["network"].update(transformer_decoder.network.get_net())

    if feature_extraction_net:
        if feature_extraction_net_global_norm:
            # TODO: just for experimenting!
            exp_config[
                "a_global_mean_logmel80"
            ] = "/u/zeineldeen/setups/librispeech/2020-08-31--att-phon/feat-stats/stats-logmelfb.mean.txt"
            exp_config[
                "a_global_stddev_logmel80"
            ] = "/u/zeineldeen/setups/librispeech/2020-08-31--att-phon/feat-stats/stats-logmelfb.std_dev.txt"

            exp_config["a_global_mean_var"] = CodeWrapper(
                "numpy.loadtxt(a_global_mean_logmel80)"
            )
            exp_config["a_global_stddev_var"] = CodeWrapper(
                "numpy.loadtxt(a_global_stddev_logmel80)"
            )

            feature_extraction_net = copy.deepcopy(feature_extraction_net)

            feature_extraction_net["a_global_mean"] = {
                "class": "constant",
                "value": CodeWrapper("a_global_mean_var"),
            }
            feature_extraction_net["a_global_stddev"] = {
                "class": "constant",
                "value": CodeWrapper("a_global_stddev_var"),
            }

            # TODO: does it broadcast automatically?
            feature_extraction_net["log10_norm"] = {
                "class": "eval",
                "from": ["log10", "a_global_mean", "a_global_stddev"],
                "eval": "(source(0) - source(1)) / source(2)",
            }

            feature_extraction_net["log_mel_features"]["from"] = "log10_norm"

        exp_config["network"].update(feature_extraction_net)

    # -------------------------- end network -------------------------- #

    # add hyperparmas
    exp_config.update(hyperparams)

    if retrain_checkpoint is not None:
        exp_config["import_model_train_epoch1"] = retrain_checkpoint

    if ext_lm_opts and ext_lm_opts.get("preload_from_files"):
        if "preload_from_files" not in exp_config:
            exp_config["preload_from_files"] = {}
        exp_config["preload_from_files"].update(
            copy.deepcopy(ext_lm_opts["preload_from_files"])
        )

    if preload_from_files:
        if "preload_from_files" not in exp_config:
            exp_config["preload_from_files"] = {}
        exp_config["preload_from_files"].update(preload_from_files)

    if specaug_str_func_opts:
        python_prolog = specaugment.specaug_helpers.get_funcs()
        extra_python_code += "\n" + specaug_transform_func.format(
            **specaug_str_func_opts
        )
    else:
        python_prolog = specaugment.specaug_tf2.get_funcs()  # type: list

    if speed_pert:
        python_prolog += [data_aug.speed_pert]

    staged_network_dict = None

    # add pretraining
    if (
        with_pretrain
        and ext_lm_opts is None
        and retrain_checkpoint is None
        and is_recog is False
    ):
        if with_staged_network:
            staged_network_dict = {}
            idx = 0
            while True:
                net = pretrain_layers_and_dims(
                    idx,
                    exp_config["network"],
                    encoder_type,
                    decoder_type,
                    encoder_args,
                    decoder_args,
                    **pretrain_opts
                )
                if not net:
                    break
                net["#copy_param_mode"] = "subset"
                if feature_extraction_net:
                    net.update(feature_extraction_net)
                staged_network_dict[(idx * pretrain_reps) + 1] = net
                idx += 1
            staged_network_dict[(idx * pretrain_reps) + 1] = exp_config["network"]
            exp_config.pop("network")
        else:
            if pretrain_opts is None:
                pretrain_opts = {}

            pretrain_networks = []
            idx = 0
            while True:
                net = pretrain_layers_and_dims(
                    idx,
                    exp_config["network"],
                    encoder_type,
                    decoder_type,
                    encoder_args,
                    decoder_args,
                    **pretrain_opts
                )
                if not net:
                    break
                pretrain_networks.append(net)
                idx += 1

            exp_config["pretrain_nets_lookup"] = {
                k: v for k, v in enumerate(pretrain_networks)
            }

            exp_config["pretrain"] = {
                "repetitions": pretrain_reps,
                "copy_param_mode": "subset",
                "construction_algo": CodeWrapper("custom_construction_algo"),
            }

            pretrain_algo_str = "def custom_construction_algo(idx, net_dict):\n\treturn pretrain_nets_lookup.get(idx, None)"
            python_prolog += [pretrain_algo_str]

    if recog_epochs:
        assert isinstance(recog_epochs, list)
        exp_post_config["cleanup_old_models"] = {"keep": recog_epochs}

    if keep_all_epochs:
        exp_post_config['cleanup_old_models'] = False

    if extra_str:
        extra_python_code += "\n" + extra_str

    if config_override:
        exp_config.update(config_override)

    if feature_extraction_net_global_norm:
        python_prolog += ["import numpy"]

    returnn_config = ReturnnConfig(
        exp_config,
        staged_network_dict=staged_network_dict,
        post_config=exp_post_config,
        python_prolog=python_prolog,
        python_epilog=extra_python_code,
        hash_full_python_code=True,
        pprint_kwargs={"sort_dicts": False},
    )

    return returnn_config
