"""
Builds Chunkwise AED Config
"""

from __future__ import annotations

import numpy
import copy
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


from i6_experiments.users.zeineldeen.models.asr.encoder.args import (
    ConformerEncoderCommonArgs,
    EncoderArgs,
    ConformerEncoderArgs,
    ConformerEncoderV2Args,
    EBranchformerEncoderArgs,
)

from i6_experiments.users.zeineldeen.models.asr.decoder.args import (
    DecoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ChunkwiseRNNDecoderArgs,
    ConformerDecoderArgs,
)

from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder import (
    ConformerEncoder,
    ConformerMemoryVariantOpts,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.transformer_decoder import (
    TransformerDecoder,
)
from i6_experiments.users.zeineldeen.models.asr.decoder.conformer_decoder import (
    ConformerDecoder,
)
from i6_experiments.users.zeineldeen.models.lm.external_lm_decoder_v2 import (
    ExternalLMDecoder,
)

from i6_experiments.users.zeineldeen.models.asr.decoder.chunked_rnn_decoder import (
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


def load_qkv_mats(name, shape, reader):
    if not name.startswith("conformer_block_"):
        return None

    idx = name.split("_")[2]
    qkv_tensor = reader.get_tensor("conformer_block_%s_self_att/QKV" % idx)
    num_heads = enc_att_num_heads_dim.dimension
    model_dim_per_head = enc_dim_per_head_dim.dimension
    model_dim = num_heads * model_dim_per_head
    assert qkv_tensor.shape == (model_dim, 3 * model_dim)
    import numpy

    qkv_tensor_ = qkv_tensor.reshape((model_dim, num_heads, 3, model_dim_per_head))
    q = qkv_tensor_[:, :, 0, :].reshape((model_dim, model_dim))
    k = qkv_tensor_[:, :, 1, :].reshape((model_dim, model_dim))
    v = qkv_tensor_[:, :, 2, :].reshape((model_dim, model_dim))

    if name == "conformer_block_%s_self_att_ln_K/W" % idx:
        return k
    elif name == "conformer_block_%s_self_att_ln_Q/W" % idx:
        return q
    elif name == "conformer_block_%s_self_att_ln_V/W" % idx:
        return v
    return None


def load_params_v2(name, shape, reader):
    import numpy

    model_dim = val_dim.dimension

    if name.startswith("conformer_block_"):
        idx = name.split("_")[2]
        qkv_tensor = reader.get_tensor("conformer_block_%s_self_att/QKV" % idx)
        assert qkv_tensor.shape == (model_dim, 3 * model_dim)

        qkv_tensor_ = qkv_tensor.reshape((model_dim, num_heads, 3, model_dim_per_head))
        q = qkv_tensor_[:, :, 0, :].reshape((model_dim, model_dim))
        k = qkv_tensor_[:, :, 1, :].reshape((model_dim, model_dim))
        v = qkv_tensor_[:, :, 2, :].reshape((model_dim, model_dim))

        if name == "conformer_block_%s_self_att_ln_K/W" % idx:
            return k
        elif name == "conformer_block_%s_self_att_ln_Q/W" % idx:
            return q
        elif name == "conformer_block_%s_self_att_ln_V/W" % idx:
            return v
    else:
        # input is [y_{i-1}, c_{i-1}, h_{i-1}]
        s_kernel = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")  # (input_dim, 4 * lstm_dim)
        s_bias = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")  # (4 * lstm_dim,)
        target_embed = reader.get_tensor("output/rec/target_embed0/W")  # (V, embed_dim)
        embed_dim = target_embed.shape[1]
        emb_ = s_kernel[:embed_dim]
        h_ = s_kernel[embed_dim + model_dim :]
        s_kernel_ = numpy.concatenate([emb_, h_], axis=0)
        assert s_kernel_.shape[0] == s_kernel.shape[0] - model_dim
        assert s_kernel_.shape[1] == s_kernel.shape[1]

        if name == "output/rec/s_wo_att/rec/lstm_cell/kernel":
            return s_kernel_  # modified
        elif name == "output/rec/s_wo_att/rec/lstm_cell/bias":
            return s_bias

    return None


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
    ignored_keys_for_reduce_dim=None,
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
            if ignored_keys_for_reduce_dim and key in ignored_keys_for_reduce_dim:
                continue
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
                if ignored_keys_for_reduce_dim and key in ignored_keys_for_reduce_dim:
                    continue
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


def create_config(
    training_datasets,
    encoder_args: EncoderArgs,
    decoder_args: DecoderArgs,
    with_staged_network=True,
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
    max_seqs=200,
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
    global_stats=None,
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
    chunked_decoder=True,
    eoc_idx=0,
    search_type=None,
    dump_alignments_dataset=None,  # train, dev, etc
    dump_ctc=False,
    dump_ctc_dataset=None,  # train, dev, etc
    enable_check_align=True,
    recog_ext_pipeline=False,
    window_left_padding=None,
    end_slice_start=None,
    end_slice_size=None,
    conf_mem_opts=None,
    gpu_mem=11,
    remove_att_ctx_from_dec_state=False,
    use_curr_enc_for_dec_state=False,
    lm_mask_layer_name=None,
    ilm_mask_layer_name=None,
    eos_cond_layer_name=None,
    handle_eos_for_ilm=False,
    renorm_wo_eos=False,
    asr_eos_no_scale=False,
    eos_asr_lm_scales=None,
    mask_always_eos_for_ilm=True,
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
        "max_seqs": max_seqs,
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
    assert isinstance(encoder_args, ConformerEncoderArgs) or isinstance(encoder_args, ConformerEncoderV2Args)
    encoder_type = ConformerEncoder

    assert isinstance(decoder_args, ChunkwiseRNNDecoderArgs)
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

        if conf_mem_opts is not None:
            encoder_args["memory_variant_opts"] = ConformerMemoryVariantOpts(
                split_batch_time_base="_input_chunked",
                chunked_time_dim=chunked_time_dim,
                self_att_version=conf_mem_opts["self_att_version"],
                chunk_size=chunk_size,
                chunk_size_dim=chunk_size_dim,
                mem_size=conf_mem_opts.get("mem_size", 1),
                conv_cache_size=conf_mem_opts.get("conv_cache_size", None),
                use_cached_prev_kv=conf_mem_opts.get("use_cached_prev_kv", False),
                mem_slice_size=conf_mem_opts.get("mem_slice_size", None),
                mem_slice_start=conf_mem_opts.get("mem_slice_start", None),
                use_emformer_mem=conf_mem_opts.get("use_emformer_mem", False),
                apply_tanh_on_emformer_mem=conf_mem_opts.get("apply_tanh_on_emformer_mem", False),
            )

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
            assert end_slice_size is None  # not implemented
            assert window_left_padding is None, "not implemented"
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

            if window_left_padding is None:
                if in_chunk_size > 1:
                    window_left_padding = (
                        (in_chunk_size // 2 - 1) * (in_chunk_size - in_chunk_step) // (in_chunk_size - 1)
                    )
                else:
                    window_left_padding = 0

            conformer_encoder.network["_input_chunked"] = {
                "class": "window",
                "from": input_,
                "window_dim": input_chunk_size_dim,
                "stride": in_chunk_step,
                "out_spatial_dim": chunked_time_dim,
                "window_left": window_left_padding,
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
            src = "__encoder"
            if end_slice_size is not None:
                new_chunk_size_dim = SpatialDim("sliced-chunk-size", end_slice_size)
                # TODO: this will break hashes for left-context only exps so needs to handle this later
                conformer_encoder.network["___encoder"] = {
                    "class": "slice",
                    "from": "__encoder",
                    "axis": chunk_size_dim,
                    "slice_start": end_slice_start,
                    "slice_end": end_slice_start + end_slice_size,
                    "out_dim": new_chunk_size_dim,
                }
                chunk_size_dim = new_chunk_size_dim
                src = "___encoder"
            conformer_encoder.network["encoder"] = {
                "class": "reinterpret_data",
                "from": src,
                "set_axes": {"T": chunked_time_dim},
            }

        else:
            raise ValueError(f"invalid chunk_level: {chunk_level!r}")

        # if not dump_ctc_dataset and not dump_ctc and not dump_alignments_dataset:
        #     if encoder_args["with_ctc"]:
        #         conformer_encoder.network["ctc_encoder"] = {
        #             "class": "fold",
        #             "from": "encoder",  # [B,C,W,D]
        #             "in_spatial_dim": chunked_time_dim,  # C
        #             "window_dim": chunk_size_dim,  # W
        #         }  # [B,C*W,D]
        #         conformer_encoder.network["ctc"]["from"] = "ctc_encoder"
    else:
        conformer_encoder = encoder_type(**encoder_args)
        conformer_encoder.create_network()

        chunk_size_dim = None
        chunked_time_dim = None
        conformer_encoder.network.add_copy_layer("encoder", "encoder_full_seq")

    decoder_args = asdict(decoder_args)
    decoder_args.update({"target": target, "beam_size": beam_size})

    chunked_decoder_trained_with_fs = False

    if chunked_decoder:
        decoder_args["enc_chunks_dim"] = chunked_time_dim
        decoder_args["enc_time_dim"] = chunk_size_dim
        decoder_args["eos_id"] = eoc_idx
        decoder_args["search_type"] = search_type
        decoder_args["enable_check_align"] = enable_check_align  # just here to keep some old changes

        if decoder_args["full_sum_simple_approx"] and is_recog:
            chunked_decoder_trained_with_fs = True
            decoder_args["full_sum_simple_approx"] = False
            decoder_args["masked_computation_blank_idx"] = eoc_idx
    elif chunk_size and not dump_ctc_dataset and not dump_ctc and not dump_alignments_dataset:
        # chunked encoder and non-chunked decoder so we need to merge encoder chunks
        assert "encoder_fold_inp" not in conformer_encoder.network
        conformer_encoder.network["encoder_fold_inp"] = copy.deepcopy(conformer_encoder.network["encoder"])
        conformer_encoder.network["encoder"] = {
            "class": "fold",
            "from": "encoder_fold_inp",  # [B,C,W,D]
            "in_spatial_dim": chunked_time_dim,  # C
            "window_dim": chunk_size_dim,  # W
        }  # [B,C*W,D]
        if encoder_args["with_ctc"]:
            conformer_encoder.network["ctc"]["from"] = "encoder"

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
            mask_layer_name=lm_mask_layer_name,
            ilm_mask_layer_name=ilm_mask_layer_name,
            eos_cond_layer_name=eos_cond_layer_name,
            handle_eos_for_ilm=handle_eos_for_ilm,
            renorm_wo_eos=renorm_wo_eos,
            asr_eos_no_scale=asr_eos_no_scale,
            eos_asr_lm_scales=eos_asr_lm_scales,
            mask_always_eos_for_ilm=mask_always_eos_for_ilm,
        )
        transformer_decoder.create_network()

    # add full network
    exp_config["network"] = conformer_encoder.network.get_net()  # type: dict

    # do not add decoder when dumping with CTC
    if not dump_ctc_dataset:
        exp_config["network"].update(transformer_decoder.network.get_net())

    if feature_extraction_net:
        exp_config["network"].update(feature_extraction_net)
        if global_stats:
            add_global_stats_norm(global_stats, exp_config["network"])

    if ext_lm_opts:
        if lm_mask_layer_name:
            exp_config["network"]["output"]["unit"][lm_mask_layer_name] = {
                "class": "compare",
                "from": "output",
                "kind": "not_equal",
                "value": 0,
                "initial_output": True,
            }

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

    if conf_mem_opts and conf_mem_opts["self_att_version"] == 1:
        assert retrain_checkpoint_opts is None
        retrain_checkpoint_opts = {}
        retrain_checkpoint_opts["custom_missing_load_func"] = load_qkv_mats

    if remove_att_ctx_from_dec_state:
        assert retrain_checkpoint_opts is None
        retrain_checkpoint_opts = {}
        retrain_checkpoint_opts["custom_missing_load_func"] = load_params_v2
        # TODO: hacky way for now
        exp_config["network"]["output"]["unit"]["s_wo_att"] = copy.deepcopy(
            exp_config["network"]["output"]["unit"]["s"]
        )
        exp_config["network"]["output"]["unit"].pop("s", None)

        # change inputs
        if decoder_args["full_sum_simple_approx"] or chunked_decoder_trained_with_fs:
            exp_config["network"]["output"]["unit"]["s_wo_att"]["from"] = "prev_target_embed"
        else:
            exp_config["network"]["output"]["unit"]["s_wo_att"]["from"] = "prev:target_embed"
        exp_config["network"]["output"]["unit"]["s_transformed"]["from"] = "s_wo_att"
        assert exp_config["network"]["output"]["unit"]["readout_in"]["from"][0] == "s"
        exp_config["network"]["output"]["unit"]["readout_in"]["from"][0] = "s_wo_att"
    elif use_curr_enc_for_dec_state:
        assert chunk_size == 1
        assert retrain_checkpoint_opts is None
        retrain_checkpoint_opts = {}
        exp_config["network"]["output"]["unit"]["enc_h_t_"] = {
            "class": "gather",
            "from": "base:encoder_full_seq",  # [B,C,1,D]
            "position": "chunk_idx",
            "axis": chunked_time_dim,
        }  # [B,1,D]
        exp_config["network"]["output"]["unit"]["enc_h_t"] = {
            "class": "squeeze",
            "from": "enc_h_t_",
            "axis": chunk_size_dim,
        }  # [B,D]
        # change inputs
        exp_config["network"]["output"]["unit"]["s"]["from"] = ["prev:target_embed", "enc_h_t"]

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

    if conf_mem_opts and conf_mem_opts["self_att_version"] == 1:
        assert retrain_checkpoint_opts is not None, "preload_from_files should be used."

    # seems to only work only when TF_FORCE_GPU_ALLOW_GROWTH is set to True in settings.py
    # otherwise I get CUDNN not loaded error. Also some error related to conv ops.
    if gpu_mem == 24:
        post_config["tf_session_opts"] = {"gpu_options": {"per_process_gpu_memory_fraction": 0.94}}

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


def add_global_stats_norm(global_stats: dict, net):
    from sisyphus.delayed_ops import DelayedFormat

    global_mean_delayed = DelayedFormat("{}", global_stats["mean"])
    global_stddev_delayed = DelayedFormat("{}", global_stats["stddev"])

    net["log10_"] = copy.deepcopy(net["log10"])
    net["global_mean"] = {
        "class": "constant",
        "value": CodeWrapper(
            f"eval(\"exec('import numpy') or numpy.loadtxt('{global_mean_delayed}', dtype='float32')\")"
        ),
        "dtype": "float32",
    }
    net["global_stddev"] = {
        "class": "constant",
        "value": CodeWrapper(
            f"eval(\"exec('import numpy') or numpy.loadtxt('{global_stddev_delayed}', dtype='float32')\")"
        ),
        "dtype": "float32",
    }
    net["log10"] = {
        "class": "eval",
        "from": ["log10_", "global_mean", "global_stddev"],
        "eval": "(source(0) - source(1)) / source(2)",
    }
