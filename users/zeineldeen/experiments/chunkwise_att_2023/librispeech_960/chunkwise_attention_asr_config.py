"""
Builds Chunkwise AED Config
"""

from __future__ import annotations

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
from i6_experiments.users.zeineldeen.models.asr.decoder.conformer_decoder import (
    ConformerDecoder,
)
from i6_experiments.users.zeineldeen.models.lm.external_lm_decoder import (
    ExternalLMDecoder,
)

from i6_experiments.users.zeyer.experiments.exp2023_02_16_chunked_attention.model import (
    RNNDecoder as ChunkwiseRNNDecoder,
    _check_alignment,
)

from i6_experiments.users.zeineldeen import data_aug
from i6_experiments.users.zeineldeen.data_aug import specaugment

from i6_core.returnn.config import ReturnnConfig, CodeWrapper

# The code here does not need the user to use returnn_common.
# However, we internally make use of some helper code from returnn_common.
from i6_experiments.common.setups.returnn.serialization import get_serializable_config

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
    initial_batch_size=None,
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

    encoder_keys = [
        "ff_dim",
        "enc_key_dim",
        "conv_kernel_size",
    ]  # TODO: effect of pretraining conv font-end?
    decoder_keys = ["ff_dim"]
    encoder_args_copy = copy.deepcopy(encoder_args)
    decoder_args_copy = copy.deepcopy(decoder_args)

    final_num_blocks = encoder_args["num_blocks"]

    assert final_num_blocks >= 2

    extra_net_dict = dict()
    extra_net_dict["#config"] = {}

    if initial_batch_size:
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
    EncoderAttNumHeads = encoder_args_copy["att_num_heads"]
    DecoderAttNumHeads = decoder_args_copy["att_num_heads"]

    if reduce_dims:
        grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
        dim_frac_enc = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_enc

        for key in encoder_keys:
            encoder_args_copy[key] = (
                int(encoder_args[key] * dim_frac_enc / float(EncoderAttNumHeads)) * EncoderAttNumHeads
            )

        if decoder_type == TransformerDecoder or decoder_type == ConformerDecoder:
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

            if decoder_type == ConformerDecoder:
                decoder_keys += ["conv_kernel_size"]

            for key in decoder_keys:
                decoder_args_copy[key] = (
                    int(decoder_args[key] * dim_frac_dec / float(DecoderAttNumHeads)) * DecoderAttNumHeads
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

    encoder_model = encoder_type(**encoder_args_copy)
    encoder_model.create_network()

    decoder_model = decoder_type(base_model=encoder_model, **decoder_args_copy)
    decoder_model.create_network()

    net_dict = encoder_model.network.get_net()
    net_dict.update(decoder_model.network.get_net())
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
    conv_alternative_name: Optional[str] = None
    use_causal_layers: bool = False

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

    output_layer_name: str = "encoder"


class DecoderArgs:
    pass


@dataclass
class TransformerDecoderArgs(DecoderArgs):
    num_layers: int = 6
    att_num_heads: int = 8
    ff_dim: int = 2048
    ff_act: str = "relu"
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

    # ILM
    replace_cross_att_w_masked_self_att: bool = False
    create_ilm_decoder: bool = False
    ilm_type: bool = None
    ilm_args: Optional[dict] = None


@dataclass
class ConformerDecoderArgs(DecoderArgs):
    num_layers: int = 6
    att_num_heads: int = 8
    ff_dim: int = 2048
    pos_enc: Optional[str] = "rel"

    # conv module
    conv_kernel_size: int = 32

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None
    conv_module_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    att_dropout: float = 0.1
    embed_dropout: float = 0.1
    softmax_dropout: float = 0.1

    # other regularization
    l2: float = 0.0001
    rel_pos_clipping: int = 16
    label_smoothing: float = 0.1
    apply_embed_weight: bool = False

    length_normalization: bool = True

    use_sqrd_relu: bool = False

    # ILM
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
    masked_computation_blank_idx: Optional[int] = None
    prev_target_embed_direct: bool = False
    full_sum_simple_approx: bool = False

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
    retrain_checkpoint_opts=None,
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
    chunk_size=20,
    chunk_step=None,
    chunk_level="encoder",  # or "input"
    eoc_idx=0,
    search_type=None,
    dump_alignments_dataset=None,  # train, dev, etc
    dump_ctc=False,
    dump_ctc_dataset=None,  # train, dev, etc
    enable_check_align=True,
    recog_ext_pipeline=False,
):
    exp_config = copy.deepcopy(config)  # type: dict
    exp_post_config = copy.deepcopy(post_config)

    exp_config["extern_data"] = training_datasets.extern_data

    assert (
        dump_alignments_dataset is None or dump_ctc_dataset is None
    ), "dump_alignments_dataset and dump_ctc_dataset are mutually exclusive"

    if not is_recog:
        if not dump_alignments_dataset and not dump_ctc and not dump_ctc_dataset:
            exp_config["train"] = training_datasets.train.as_returnn_opts()
            exp_config["dev"] = training_datasets.cv.as_returnn_opts()
            if training_datasets.devtrain:
                exp_config["eval_datasets"] = {"devtrain": training_datasets.devtrain.as_returnn_opts()}
        else:
            if dump_ctc:
                exp_config["eval_datasets"] = {
                    "train": training_datasets.train.as_returnn_opts(),
                    "dev": training_datasets.cv.as_returnn_opts(),
                }
            if dump_alignments_dataset or dump_ctc_dataset:
                dump_data = dump_alignments_dataset if dump_alignments_dataset else dump_ctc_dataset
                if dump_data == "train":
                    exp_config["eval_datasets"] = {
                        "train": training_datasets.train.as_returnn_opts(),
                    }
                elif dump_data == "dev":
                    exp_config["eval_datasets"] = {
                        "dev": training_datasets.cv.as_returnn_opts(),
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
        extra_python_code += "\n" + oclr_str.format(**oclr_opts, initial_lr=oclr_initial_lr)
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
                learning_rates = [wup_start_lr] * const_lr + list(numpy.linspace(wup_start_lr, lr, num=wup))
            elif isinstance(const_lr, list):
                assert len(const_lr) == 2
                learning_rates = (
                    [wup_start_lr] * const_lr[0] + list(numpy.linspace(wup_start_lr, lr, num=wup)) + [lr] * const_lr[1]
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
    assert isinstance(encoder_args, ConformerEncoderArgs)
    encoder_type = ConformerEncoder

    assert isinstance(decoder_args, RNNDecoderArgs)
    decoder_type = ChunkwiseRNNDecoder
    dec_type = "lstm"

    encoder_args = asdict(encoder_args)
    if feature_extraction_net:
        encoder_args.update({"target": target, "input": "log_mel_features"})
    else:
        encoder_args.update({"target": target, "input": "data:" + input_key})

    if freeze_bn:
        # freeze BN during training (e.g when retraining.)
        encoder_args["batch_norm_opts"] = {"momentum": 0.0, "use_sample": 1.0}

    encoder_args["output_layer_name"] = "encoder_full_seq"

    if chunk_size and chunk_size > 0:
        from returnn.tf.util.data import SpatialDim

        chunked_time_dim = SpatialDim("chunked-time")
        chunk_size_dim = SpatialDim("chunk-size", chunk_size)

        input_ = encoder_args["input"]
        specaug_ = encoder_args.get("specaug", True)
        input_chunk_size_dim = None
        in_chunk_size = None
        in_chunk_step = None
        if chunk_level == "input":
            encoder_args["input"] = "input_chunked"
            assert encoder_args["input_layer"] in ["lstm-6", "conv-6"]  # hardcoded factor 6 below
            in_chunk_size = chunk_size * 6
            in_chunk_step = chunk_step * 6
            input_chunk_size_dim = SpatialDim("input-chunk-size", in_chunk_size)
            encoder_args["specaug"] = False  # need to do it before
            encoder_args["fix_merge_dims"] = True  # broken otherwise

        conformer_encoder = encoder_type(**encoder_args)
        conformer_encoder.create_network()

        if chunk_level == "encoder":

            conformer_encoder.network["encoder"] = {
                "class": "window",
                "from": "encoder_full_seq",
                "window_dim": chunk_size_dim,
                "stride": chunk_step,
                "out_spatial_dim": chunked_time_dim,
            }

            if not dump_ctc_dataset:  # to not break hashes for CTC dumping
                if chunk_size == chunk_step:
                    conformer_encoder.network["encoder"]["window_left"] = 0
                    conformer_encoder.network["encoder"]["window_right"] = chunk_size - 1
                else:
                    # formula works also for chunk_size == chunk_step
                    # but used separate to not break hashes because window_right is set
                    conformer_encoder.network["encoder"]["window_left"] = (
                        ((chunk_size // 2 - 1) * (chunk_size - chunk_step) // (chunk_size - 1)) if chunk_size > 1 else 0
                    )

        elif chunk_level == "input":

            if specaug_:
                input_ = conformer_encoder.network.add_eval_layer(
                    "source",
                    input_,
                    eval="self.network.get_config().typed_value('transform')"
                    "(source(0, as_data=True), network=self.network)",
                )

            conformer_encoder.network["_input_chunked"] = {
                "class": "window",
                "from": input_,
                "window_dim": input_chunk_size_dim,
                "stride": in_chunk_step,
                "out_spatial_dim": chunked_time_dim,
                "window_left": ((in_chunk_size // 2 - 1) * (in_chunk_size - in_chunk_step) // (in_chunk_size - 1))
                if in_chunk_size > 1
                else 0,
            }
            conformer_encoder.network["__input_chunked"] = {
                "class": "merge_dims",
                "from": "_input_chunked",
                "axes": ["B", chunked_time_dim],
                "keep_order": True,
            }
            conformer_encoder.network["input_chunked"] = {
                "class": "reinterpret_data",
                "from": "__input_chunked",
                "set_axes": {"T": input_chunk_size_dim},
            }

            conformer_encoder.network["_encoder"] = {
                "class": "reinterpret_data",
                "from": "encoder_full_seq",
                "set_dim_tags": {"T": chunk_size_dim},
            }
            conformer_encoder.network["__encoder"] = {
                "class": "split_batch_time",
                "from": "_encoder",
                "base": "_input_chunked",
            }
            conformer_encoder.network["encoder"] = {
                "class": "reinterpret_data",
                "from": "__encoder",
                "set_axes": {"T": chunked_time_dim},
            }

        else:
            raise ValueError(f"invalid chunk_level: {chunk_level!r}")

    else:
        conformer_encoder = encoder_type(**encoder_args)
        conformer_encoder.create_network()

        chunk_size_dim = None
        chunked_time_dim = None
        conformer_encoder.network.add_copy_layer("encoder", "encoder_full_seq")

    decoder_args = asdict(decoder_args)
    decoder_args.update({"target": target, "beam_size": beam_size})

    decoder_args["enc_chunks_dim"] = chunked_time_dim
    decoder_args["enc_time_dim"] = chunk_size_dim
    decoder_args["eos_id"] = eoc_idx
    decoder_args["search_type"] = search_type
    decoder_args["enable_check_align"] = enable_check_align  # just here to keep some old changes

    if decoder_args["full_sum_simple_approx"] and is_recog:
        decoder_args["full_sum_simple_approx"] = False
        decoder_args["masked_computation_blank_idx"] = eoc_idx

    transformer_decoder = decoder_type(base_model=conformer_encoder, **decoder_args)
    if not dump_ctc_dataset:
        transformer_decoder.create_network()

    if recog_ext_pipeline:
        # Unfiltered, with full beam.
        exp_config["search_output_layer"] = transformer_decoder.dec_output
    else:
        exp_config["search_output_layer"] = transformer_decoder.decision_layer_name

    if ext_lm_opts:
        transformer_decoder = ExternalLMDecoder(
            transformer_decoder,
            ext_lm_opts,
            prior_lm_opts=prior_lm_opts,
            beam_size=beam_size,
            dec_type=dec_type,
            length_normalization=decoder_args["length_normalization"],
        )
        transformer_decoder.create_network()

    # add full network
    exp_config["network"] = conformer_encoder.network.get_net()  # type: dict

    # do not add decoder when dumping with CTC
    if not dump_ctc_dataset:
        exp_config["network"].update(transformer_decoder.network.get_net())

    if feature_extraction_net:
        exp_config["network"].update(feature_extraction_net)

    # if chunked_time_dim:
    #   exp_config['network']["_check_alignment"] = {
    #     "class": "eval",
    #     "from": 'out_best_wo_blank',
    #     "eval": CodeWrapper('_check_alignment'),
    #     "is_output_layer": True,
    #   }

    if dump_alignments_dataset:
        exp_config["network"]["dump_out_best"] = {
            "class": "hdf_dump",
            "from": "out_best",
            "filename": f"alignments-{dump_alignments_dataset}.hdf",
            "is_output_layer": True,
        }
        # exp_config['load'] = retrain_checkpoint

    if dump_ctc:
        exp_config["network"]["ctc_forced_align"] = {
            "class": "forced_align",
            "from": "ctc",
            "input_type": "prob",
            "align_target": "data:bpe_labels",
            "topology": "rna",
        }
        exp_config["network"]["ctc_forced_align_dump"] = {
            "class": "hdf_dump",
            "from": "ctc_forced_align",
            "filename": "alignments-{dataset_name}.hdf",
            "dump_per_run": True,
            "is_output_layer": True,
        }
        exp_config["load"] = retrain_checkpoint

    if dump_ctc_dataset:
        exp_config["network"]["ctc_forced_align"] = {
            "class": "forced_align",
            "from": "ctc",
            "input_type": "prob",
            "align_target": "data:bpe_labels",
            "topology": "rna",
        }
        exp_config["network"]["ctc_forced_align_dump"] = {
            "class": "hdf_dump",
            "from": "ctc_forced_align",
            f"filename": f"alignments-{dump_ctc_dataset}.hdf",
            "is_output_layer": True,
        }
        hyperparams["max_seq_length"] = None  # remove max seq len filtering

    # TODO: fix search bug
    if is_recog:
        exp_config["network"]["out_best_wo_blank"]["target"] = "bpe_labels"

        tmp = copy.deepcopy(exp_config["network"]["output"]["max_seq_len"])
        exp_config["network"]["output"]["max_seq_len"] = tmp + " * 2"

    # -------------------------- end network -------------------------- #

    # add hyperparmas
    exp_config.update(hyperparams)

    if retrain_checkpoint is not None:
        if retrain_checkpoint_opts:
            # Use the preload_from_files mechanism, which can do the same as import_model_train_epoch1,
            # but also allows for further options.
            exp_config.setdefault("preload_from_files", {})["_01_retrain_checkpoint"] = {
                "filename": retrain_checkpoint,
                "init_for_train": True,  # like import_model_train_epoch1
                **retrain_checkpoint_opts,  # for example ignore_missing, ignore_params, var_name_mapping, ...
            }
        else:
            exp_config["import_model_train_epoch1"] = retrain_checkpoint

    if ext_lm_opts and ext_lm_opts.get("preload_from_files"):
        exp_config.setdefault("preload_from_files", {}).update(copy.deepcopy(ext_lm_opts["preload_from_files"]))

    if preload_from_files:
        exp_config.setdefault("preload_from_files", {}).update(preload_from_files)

    if specaug_str_func_opts:
        python_prolog = specaugment.specaug_helpers.get_funcs()
        extra_python_code += "\n" + specaug_transform_func.format(**specaug_str_func_opts)
    else:
        python_prolog = specaugment.specaug_tf2.get_funcs()  # type: list

    if speed_pert:
        python_prolog += [data_aug.speed_pert]

    # python_prolog += [_check_alignment]

    staged_network_dict = None

    # add pretraining
    if with_pretrain and ext_lm_opts is None and retrain_checkpoint is None and is_recog is False:
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
                    **pretrain_opts,
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
                    **pretrain_opts,
                )
                if not net:
                    break
                pretrain_networks.append(net)
                idx += 1

            exp_config["pretrain_nets_lookup"] = {k: v for k, v in enumerate(pretrain_networks)}

            exp_config["pretrain"] = {
                "repetitions": pretrain_reps,
                "copy_param_mode": "subset",
                "construction_algo": CodeWrapper("custom_construction_algo"),
            }

            pretrain_algo_str = (
                "def custom_construction_algo(idx, net_dict):\n" "\treturn pretrain_nets_lookup.get(idx, None)"
            )
            python_prolog += [pretrain_algo_str]

    if recog_epochs:
        assert isinstance(recog_epochs, list)
        exp_post_config["cleanup_old_models"] = {"keep": recog_epochs}

    if keep_all_epochs:
        exp_post_config["cleanup_old_models"] = False

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
        pprint_kwargs={"sort_dicts": False},
        hash_full_python_code=True if dump_ctc_dataset else False,  # to avoid hash changes for CTC alignment dump
    )

    serialized_config = get_serializable_config(returnn_config)

    # from pprint import pprint
    # pprint(serialized_config.config)

    return serialized_config
