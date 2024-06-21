"""
https://arxiv.org/abs/1905.04226
Reference: https://github.com/rwth-i6/returnn-experiments/blob/master/2019-lm-transformers/librispeech/bpe_10k/transfo_24_d00.4096_1024.sgd.lr1.8_heads.config
"""

from __future__ import annotations

from typing import Dict

import numpy

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.schmitt.returnn_frontend.convert.checkpoint import ConvertTfCheckpointToRfPtJob

import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo import model as trafo_lm

_returnn_tf_ckpt_filename = "/u/zhou/asr-exps/librispeech/dependencies/kazuki_lstmlm_27062019/network.040"
TrafoLmOpts = {
    "vocab_dim": 10_025,
    "model_dim": 1024,
    "embed_dim": 128,
    "num_layers": 24,
    "decoder_layer_opts": {"self_att_opts": {"with_bias": False, "att_dropout_broadcast": False}},
    "input_embedding_scale": 1.0,
    "share_embedding": False,
    "logits_with_bias": True,
    "input_dropout": 0.1,
}

_ParamMapping = {}  # type: Dict[str,str]


def get_tf_checkpoint_path() -> tk.Path:
    """
    :return: Sisyphus tk.Path to the original TF checkpoint file

    https://arxiv.org/abs/1905.04226
    Reference: https://github.com/rwth-i6/returnn-experiments/blob/master/2019-lm-transformers/librispeech/bpe_10k/transfo_24_d00.4096_1024.sgd.lr1.8_heads.config
    """
    return tk.Path(
        _returnn_tf_ckpt_filename, hash_overwrite="librispeech-2018-kazuki-transfo_24_d00.4096_1024.sgd.lr1.8_heads"
    )


def get_pt_checkpoint_path() -> tk.Path:
    """
    :return: Sisyphus tk.Path to the PyTorch checkpoint file

    https://arxiv.org/abs/1905.04226
    Reference: https://github.com/rwth-i6/returnn-experiments/blob/master/2019-lm-transformers/librispeech/bpe_10k/transfo_24_d00.4096_1024.sgd.lr1.8_heads.config
    """
    old_tf_ckpt_path = get_tf_checkpoint_path()
    old_tf_ckpt = Checkpoint(index_path=old_tf_ckpt_path)
    make_model_func = trafo_lm.MakeModel(**TrafoLmOpts)  # eos_label=0
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=old_tf_ckpt,
        make_model_func=make_model_func,
        map_func=map_param_func_v2,
        epoch=1,
        step=0,
    )
    return converter.out_checkpoint


def _add_params():
    _ParamMapping.update(
        {
            "input_embedding.weight": "output/rec/target_embed_raw/W",
            "input_embedding_proj.weight": "output/rec/target_embed_lin/W",
            "final_layer_norm.scale": "output/rec/decoder/scale",
            "final_layer_norm.bias": "output/rec/decoder/bias",
            "logits.weight": "output/rec/output/W",
            "logits.bias": "output/rec/output/b",
        }
    )

    for layer_idx in range(TrafoLmOpts["num_layers"]):
        # FF
        _ParamMapping[f"layers.{layer_idx}.ff.linear_ff.weight"] = f"output/rec/dec_{layer_idx}_ff_conv1/W"
        _ParamMapping[f"layers.{layer_idx}.ff.linear_ff.bias"] = f"output/rec/dec_{layer_idx}_ff_conv1/b"
        _ParamMapping[f"layers.{layer_idx}.ff.linear_out.weight"] = f"output/rec/dec_{layer_idx}_ff_conv2/W"
        _ParamMapping[f"layers.{layer_idx}.ff.linear_out.bias"] = f"output/rec/dec_{layer_idx}_ff_conv2/b"
        _ParamMapping[f"layers.{layer_idx}.ff_layer_norm.scale"] = f"output/rec/dec_{layer_idx}_ff_laynorm/scale"
        _ParamMapping[f"layers.{layer_idx}.ff_layer_norm.bias"] = f"output/rec/dec_{layer_idx}_ff_laynorm/bias"

        # self-att
        _ParamMapping[f"layers.{layer_idx}.self_att.qkv.weight"] = f"output/rec/dec_{layer_idx}_self_att_att/QKV"
        _ParamMapping[f"layers.{layer_idx}.self_att.proj.weight"] = f"output/rec/dec_{layer_idx}_self_att_lin/W"
        _ParamMapping[f"layers.{layer_idx}.self_att_layer_norm.scale"] = (
            f"output/rec/dec_{layer_idx}_self_att_laynorm/scale"
        )
        _ParamMapping[f"layers.{layer_idx}.self_att_layer_norm.bias"] = (
            f"output/rec/dec_{layer_idx}_self_att_laynorm/bias"
        )


_add_params()


def map_param_func_v2(reader, name: str, var: rf.Parameter) -> numpy.ndarray:
    """map params, TF to RF"""
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader

    assert isinstance(reader, CheckpointReader)
    assert isinstance(var, rf.Parameter)

    tf_var_name = name.replace(".", "/")
    if reader.has_tensor(tf_var_name):
        return reader.get_tensor(tf_var_name)

    if name in _ParamMapping:
        var_name = _ParamMapping[name]
        assert reader.has_tensor(var_name)
        value = reader.get_tensor(var_name)
        assert isinstance(value, numpy.ndarray)
        assert (
            value.shape == var.batch_shape
        ), f"new param {name} {var.batch_shape} vs ckpt param {var_name} {value.shape}"
        assert value.dtype.name == var.dtype, f"new param {name} {var.dtype} vs ckpt param {var_name} {value.dtype}"
        return value

    raise NotImplementedError(f"cannot map {name!r} {var}")
