from __future__ import annotations

from typing import Dict

import os
import sys
import numpy

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from .trafo_lm import MakeModel

_returnn_tf_ckpt_filename = "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023.index"
_load_existing_ckpt_in_test = True
_trafo_lm_opts = {
    "vocab_dim": 10_025,
    "model_dim": 1024,
    "embed_dim": 128,
    "num_layers": 24,
    "decoder_layer_opts": {"self_att_opts": {"with_bias": False}},
    "input_embedding_scale": 1.0,
    "share_embedding": False,
    "input_dropout": 0,
    "logits_with_bias": True,
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
    make_model_func = MakeModel(**_trafo_lm_opts)  # eos_label=0
    # TODO: problems with hash:
    #  make_model_func, map_func: uses full module name (including "zeyer"), should use sth like unhashed_package_root
    #  https://github.com/rwth-i6/sisyphus/issues/144
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

    for layer_idx in range(_trafo_lm_opts["num_layers"]):
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


# See comment below, use `py = test_import_forward` to easily run this.
def test_import_forward():
    from returnn.util import better_exchook

    better_exchook.install()

    from pprint import pprint

    # Pick some layers to check outputs for equality.
    # See the func tracing logic below, entries in captured_tensors.
    # RETURNN layer name -> trace point in RF/PT model forwarding.
    _layer_mapping = {
        "output/target_embed_raw": (TransformerDecoder.__call__, 0, "decoded", 0),
        "output/target_embed_with_pos": (TransformerDecoder.__call__, 0, "decoded", 1),
        "output/target_embed_lin": (TransformerDecoder.__call__, 0, "decoded", 2),
        "output/dec_0_self_att_laynorm": (TransformerDecoderLayer.__call__, 0, "x_sa_ln", 0),
        "output/dec_0_self_att_drop": (TransformerDecoderLayer.__call__, 0, "x_sa", 0),
        "output/dec_0_self_att_out": (TransformerDecoderLayer.__call__, 0, "x", 1),
        "output/dec_0_ff_out": (TransformerDecoderLayer.__call__, 0, "x", 2),
        "output/dec_0": (TransformerDecoderLayer.__call__, 1, "x", 0),
        "output/dec_23_ff_out": (TransformerDecoderLayer.__call__, -1, "x", 2),
        "output/dec_23": (TransformerDecoder.__call__, 0, "decoded", -2),
        "output/output": (TransformerDecoder.__call__, 0, "logits", 0),
    }

    from i6_experiments.common.setups.returnn_common import serialization

    exec(serialization.PythonEnlargeStackWorkaroundNonhashedCode.code)  # setrecursionlimit etc

    from returnn.datasets.util.vocabulary import Vocabulary

    target_dim = Dim(name="target", dimension=10_025, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor("target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim)
    model_dim = Dim(1024, name="model")

    from ._trafo_lm_kazuki_net_dict import network as net_dict
    from returnn.config import Config
    from returnn.util import BehaviorVersion

    config = Config(
        dict(
            log_verbositiy=5,
            network=net_dict,
            extern_data={
                "data": {"dim_tags": target.dims, "sparse_dim": target.sparse_dim},
            },
        )
    )
    BehaviorVersion.set_min_behavior_version(20)

    from returnn.tensor.utils import tensor_dict_fill_random_numpy_
    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_
    from returnn.tf.network import TFNetwork
    import tensorflow as tf
    import numpy.testing
    import tempfile
    import atexit
    import shutil

    ckpt_dir = tempfile.mkdtemp("returnn-import-test")
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    print("*** Construct TF graph for old model")
    extern_data = TensorDict()
    extern_data.update(config.typed_dict["extern_data"], auto_convert=True)
    tensor_dict_fill_random_numpy_(
        extern_data, dyn_dim_max_sizes={target_spatial_dim: 20}, dyn_dim_min_sizes={target_spatial_dim: 10}
    )  # raw sample level
    extern_data["data"].raw_tensor[:, 0] = 0  # set BOS
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    tf1 = tf.compat.v1
    with tf1.Graph().as_default() as graph, tf1.Session(graph=graph).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        if _load_existing_ckpt_in_test:
            ckpt_path = get_tf_checkpoint_path()
            print(f"*** Load model params from {ckpt_path.get_path()}")
            net.load_params_from_file(ckpt_path.get_path(), session=session)
            old_tf_ckpt_path = ckpt_path
        else:
            print("*** Random init old model")
            net.initialize_params(session)
            print("*** Save old model to disk")
            net.save_params_to_file(ckpt_dir + "/old_model/model", session=session)
            old_tf_ckpt_path = tk.Path(ckpt_dir + "/old_model/model.index")

        print("*** Forwarding ...")

        extern_data_tf_raw_dict = net.extern_data.as_raw_tensor_dict()
        assert set(extern_data_tf_raw_dict.keys()) == set(extern_data_numpy_raw_dict.keys())
        feed_dict = {extern_data_tf_raw_dict[k]: extern_data_numpy_raw_dict[k] for k in extern_data_numpy_raw_dict}
        fetches = net.get_fetches_dict()
        old_model_outputs_data = {}
        for old_layer_name, _ in _layer_mapping.items():
            layer = net.get_layer(old_layer_name)
            out = layer.output.copy_as_batch_major()
            old_model_outputs_data[old_layer_name] = out
            fetches["layer:" + old_layer_name] = out.placeholder
            for i, tag in enumerate(out.dim_tags):
                if tag.is_batch_dim():
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.get_dim_value()
                elif tag.dyn_size_ext:
                    old_model_outputs_data[f"{old_layer_name}:size{i}"] = tag.dyn_size_ext
                    fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.placeholder
        old_model_outputs_fetch = session.run(fetches, feed_dict=feed_dict)

    def _make_new_model():
        opts = _trafo_lm_opts.copy()
        opts.pop("vocab_dim")
        opts.pop("model_dim")
        return MakeModel(target_dim, model_dim, **opts)()

    _make_new_model.__module__ = "<dummy-main>"  # avoid error with hashing
    rf.select_backend_torch()

    print("*** Convert old model to new model")
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=Checkpoint(index_path=old_tf_ckpt_path),
        make_model_func=_make_new_model,
        map_func=map_param_func_v2,
        epoch=1,
        step=0,
    )
    converter._out_model_dir = tk.Path(ckpt_dir + "/new_model")
    converter.out_checkpoint = tk.Path(ckpt_dir + "/new_model/checkpoint.pt")
    converter.run()

    print("*** Create new model")
    new_model = _make_new_model()

    rf.init_train_step_run_ctx(train_flag=False)
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_numpy_raw_dict)
    tensor_dict_numpy_to_torch_(extern_data)

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(ckpt_dir + "/new_model/checkpoint.pt")
    pt_module.load_state_dict(checkpoint_state["model"])
    pt_module.eval()

    print("*** Forwarding with tracing ...")

    funcs_to_trace_list = [
        TransformerDecoder.__call__,
        TransformerDecoderLayer.__call__,
    ]
    code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
    captured_tensors = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

    def _trace_func(frame, event, arg):
        """
        Trace func to get intermediate outputs.
        """
        func = code_obj_to_func.get(frame.f_code)
        if func:
            if event == "call":
                captured_tensors.setdefault(func, []).append({})
            else:
                for k, v in frame.f_locals.items():
                    if not isinstance(v, Tensor):
                        continue
                    prev = captured_tensors[func][-1].get(k, None)
                    if prev is None or prev[-1] is not v:
                        print(f"{func.__qualname__} tensor var changed: {k} = {v}")
                        captured_tensors[func][-1].setdefault(k, []).append(v)
            return _trace_func

    sys.settrace(_trace_func)
    new_model(
        extern_data["data"],
        spatial_dim=target_spatial_dim,
        state=new_model.default_initial_state(batch_dims=[batch_dim]),
    )
    sys.settrace(None)
    pprint(captured_tensors)

    print("*** Getting values from trace ...")
    fetches = {}
    for old_layer_name, new_var_path in _layer_mapping.items():
        new_out = captured_tensors
        try:
            for k in new_var_path:
                new_out = new_out[k]
        except KeyError as exc:
            raise Exception(f"{exc.__class__.__name__} {exc}, new_var_path: {new_var_path}")
        assert isinstance(new_out, Tensor), f"new_out: {new_out}, new_var_path: {new_var_path}"
        old_out = old_model_outputs_data[old_layer_name]
        assert old_out.batch_ndim == new_out.batch_ndim
        mapped_axes = new_out.find_matching_dim_map(
            old_out,
            list(range(old_out.batch_ndim)),
            is_equal_opts=dict(
                allow_same_feature_dim=True,
                allow_same_spatial_dim=True,
                treat_feature_as_spatial=True,
                allow_old_behavior=True,
            ),
        )
        out = new_out.copy_transpose([mapped_axes[i] for i in range(old_out.batch_ndim)])
        fetches["layer:" + old_layer_name] = out.raw_tensor
        for i, tag in enumerate(out.dim_tags):
            if tag.dyn_size_ext:
                fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.raw_tensor
    fetches = {k: v.detach().cpu().numpy() for (k, v) in fetches.items()}
    new_model_outputs_fetch = fetches

    print("*** Comparing ...")
    print("**** target spatial len:", extern_data_numpy_raw_dict["data"].shape[1])
    for out_step in range(extern_data_numpy_raw_dict["data"].shape[1]):
        for old_layer_name, new_var_path in _layer_mapping.items():
            out = old_model_outputs_data[old_layer_name]
            if out_step > 0 and target_spatial_dim not in out.dim_tags:
                continue
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext:
                    old_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    new_v = new_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    numpy.testing.assert_equal(old_v, new_v, err_msg=f"{tag} mismatch")
            old_v = old_model_outputs_fetch["layer:" + old_layer_name]
            new_v = new_model_outputs_fetch["layer:" + old_layer_name]
            for i, tag in enumerate(out.dim_tags):
                if tag.dyn_size_ext and tag.dyn_size_ext.dim_tags:  # dynamic, and not scalar dyn sizes
                    assert tag.dyn_size_ext.dim_tags == (batch_dim,)  # not implemented otherwise
                    assert out.batch_dim_axis == 0  # not implemented otherwise but should be ensured above
                    size_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    for b in range(old_v.shape[0]):
                        idx = tuple([slice(b, b + 1)] + [slice(None, None)] * (i - 1) + [slice(size_v[b], None)])
                        old_v[idx] = 0
                        new_v[idx] = 0
            print(f"* Comparing {out}: {old_layer_name!r} vs {new_var_path!r}")
            assert old_v.shape == new_v.shape
            if target_spatial_dim in out.dim_tags:
                assert out.get_axis_from_description(target_spatial_dim) == 1  # not implemented otherwise
                out = out.copy_template_excluding_axis(1)
                print("** comparing out_step", out_step, out)
                old_v = old_v[:, out_step]
                new_v = new_v[:, out_step]
            # Using equal_nan=False because we do not want any nan in any of the values.
            rtol, atol = 0.2, 5e-5
            if numpy.allclose(old_v, new_v, rtol=rtol, atol=atol):
                continue
            print("** not all close. close:")
            # Iterate over all indices, and check if the values are close.
            # If not, add the index to the mismatches list.
            remarks = []
            count_mismatches = 0
            for idx in sorted(numpy.ndindex(old_v.shape), key=sum):
                if numpy.isnan(old_v[idx]) and numpy.isnan(new_v[idx]):
                    remarks.append("[%s]:? (both are nan)" % ",".join([str(i) for i in idx]))
                    count_mismatches += 1
                    continue
                close = numpy.allclose(old_v[idx], new_v[idx], rtol=rtol, atol=atol)
                if not close:
                    count_mismatches += 1
                remarks.append(
                    "[%s]:" % ",".join([str(i) for i in idx])
                    + ("✓" if close else "✗ (%.5f diff)" % abs(old_v[idx] - new_v[idx]))
                )
                if len(remarks) >= 50 and count_mismatches > 0:
                    remarks.append("...")
                    break
            print("\n".join(remarks))
            numpy.testing.assert_allclose(
                old_v,
                new_v,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                err_msg=f"{old_layer_name!r} vs {new_var_path!r} mismatch",
            )
            raise Exception(f"should not get here, mismatches: {remarks}")

    print("*** Done, all correct (!), exit now ***")
    raise SystemExit("done")


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
# py = test_import_search
py = test_import_forward


if __name__ == "__main__":
    mod_name = __package__
    if mod_name.startswith("recipe."):
        mod_name = mod_name[len("recipe.") :]
    mod_name += "." + os.path.basename(__file__)[: -len(".py")]
    map_param_func_v2.__module__ = mod_name
    py()
