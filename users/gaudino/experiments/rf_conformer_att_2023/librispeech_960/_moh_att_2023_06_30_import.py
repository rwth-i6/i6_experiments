from __future__ import annotations

from typing import Dict

import os
import sys
import numpy

from sisyphus import tk

from i6_core.returnn.training import Checkpoint
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import (
    ConvertTfCheckpointToRfPtJob,
)

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    Model,
    MakeModel,
    from_scratch_training,
    model_recog,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_ctc_greedy import (
    model_recog_ctc,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_recog_time_sync import (
    model_recog_time_sync,
)
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config"
# # E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
_load_existing_ckpt_in_test = True
_lm_path = "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/re_i128_m2048_m2048_m2048_m2048.sgd_b32_lr0_cl2.newbobabs.d0.0.1350/bk-net-model/network.035"


_ParamMapping = {}  # type: Dict[str,str]


def _get_tf_checkpoint_path() -> tk.Path:
    """
    :return: Sisyphus tk.Path to the checkpoint file
    """
    return generic_job_output(_returnn_tf_ckpt_filename)


def _get_pt_checkpoint_path() -> tk.Path:
    old_tf_ckpt_path = _get_tf_checkpoint_path()
    old_tf_ckpt = Checkpoint(index_path=old_tf_ckpt_path)
    make_model_func = MakeModel(80, 10_025, eos_label=0, num_enc_layers=12)
    # TODO: problems with hash:
    #  make_model_func, map_func: uses full module name (including "zeyer"), should use sth like unhashed_package_root
    #  https://github.com/rwth-i6/sisyphus/issues/144
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=old_tf_ckpt,
        make_model_func=make_model_func,
        map_func=map_param_func_v3,
        epoch=1,
        step=0,
    )
    # converter.run()
    return converter.out_checkpoint


def _add_params():
    # frontend
    for layer_idx in [0, 1, 2]:
        orig_name = "conv0" if layer_idx == 0 else f"subsample_conv{layer_idx - 1}"
        _ParamMapping.update(
            {
                f"encoder.input_layer.conv_layers.{layer_idx}.filter": f"{orig_name}/W",
                f"encoder.input_layer.conv_layers.{layer_idx}.bias": f"{orig_name}/bias",
            }
        )
    _ParamMapping.update(
        {
            "encoder.input_projection.weight": "source_linear/W",
            "enc_ctx.weight": "enc_ctx/W",
            "enc_ctx.bias": "enc_ctx/b",
            "ctc.weight": "ctc/W",
            "ctc.bias": "ctc/b",
            "inv_fertility.weight": "inv_fertility/W",
            "target_embed.weight": "output/rec/target_embed0/W",
            "weight_feedback.weight": "output/rec/weight_feedback/W",
            "s_transformed.weight": "output/rec/s_transformed/W",
            "energy.weight": "output/rec/energy/W",
            "readout_in.weight": "output/rec/readout_in/W",
            "readout_in.bias": "output/rec/readout_in/b",
            "output_prob.weight": "output/rec/output_prob/W",
            "output_prob.bias": "output/rec/output_prob/b",
        }
    )
    # conformer
    for layer_idx in range(12):
        # FF
        for sub in [1, 2]:
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/W"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/b"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/W"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/b"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/scale"
            _ParamMapping[
                f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/bias"
        # conv
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/b"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/bias"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/b"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.conv_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/bias"
        # self-att
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.qkv.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att/QKV"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.proj.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_linear/W"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/bias"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln_rel_pos_enc/encoding_matrix"
        # final layer norm
        _ParamMapping[
            f"encoder.layers.{layer_idx}.final_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/scale"
        _ParamMapping[
            f"encoder.layers.{layer_idx}.final_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/bias"


_add_params()


def map_param_func_v3(reader, name: str, var: rf.Parameter) -> numpy.ndarray:
    """map params, TF to RF"""
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    from i6_experiments.users.zeyer.returnn.convert.params import (
        numpy as convert_params_np,
    )
    from i6_experiments.users.zeyer.returnn.convert.params import (
        tf_to_rf_np as convert_params_tf_to_rf_np,
    )

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
        if name.endswith(".filter"):
            value = convert_params_np.convert_tf_conv_to_pt_conv_filter(value)
        assert (
            value.shape == var.batch_shape
        ), f"new param {name} {var.batch_shape} vs ckpt param {var_name} {value.shape}"
        assert (
            value.dtype.name == var.dtype
        ), f"new param {name} {var.dtype} vs ckpt param {var_name} {value.dtype}"
        return value

    if name == "s.ff_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_ff(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.rec_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_rec(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.bias":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")
        value = convert_params_np.convert_tf_lstm_to_native_lstm_bias(
            value, forget_gate_bias=1.0
        )
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if ".conv_block.norm." in name:
        assert name.startswith("encoder.layers.")
        layer_idx = int(name.split(".")[2])
        value = convert_params_tf_to_rf_np.convert_tf_batch_norm_to_rf(
            reader=reader,
            rf_name=name,
            rf_prefix_name=f"encoder.layers.{layer_idx}.conv_block.norm.",
            tf_prefix_name=f"conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/",
            var=var,
        )
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    raise NotImplementedError(f"cannot map {name!r} {var}")


# See comment below, use `py = test_import` to easily run this.
def test_import_forward():
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
    )
    from pprint import pprint

    # Pick some layers to check outputs for equality.
    # See the func tracing logic below, entries in captured_tensors.
    # RETURNN layer name -> trace point in RF/PT model forwarding.
    _layer_mapping = {
        "source": (Model.encode, 0, "source", -1),
        "conv_merged": (ConformerEncoder.__call__, 0, "x_subsample", 0),
        "source_linear": (ConformerEncoder.__call__, 0, "x_linear", 0),
        "conformer_block_01_ffmod_1_drop2": (
            ConformerEncoderLayer.__call__,
            0,
            "x_ffn1",
            0,
        ),
        "conformer_block_01_ffmod_1_res": (
            ConformerEncoderLayer.__call__,
            0,
            "x_ffn1_out",
            0,
        ),
        "conformer_block_01_self_att_ln": (
            ConformerEncoderLayer.__call__,
            0,
            "x_mhsa_ln",
            0,
        ),
        "conformer_block_01_self_att_linear": (
            ConformerEncoderLayer.__call__,
            0,
            "x_mhsa",
            0,
        ),
        "conformer_block_01_self_att_res": (
            ConformerEncoderLayer.__call__,
            0,
            "x_mhsa_out",
            0,
        ),
        "conformer_block_01_conv_mod_res": (
            ConformerEncoderLayer.__call__,
            0,
            "x_conv_out",
            0,
        ),
        "conformer_block_01_ffmod_2_res": (
            ConformerEncoderLayer.__call__,
            0,
            "x_ffn2_out",
            0,
        ),
        "conformer_block_01": (ConformerEncoderLayer.__call__, 1, "inp", 0),
        "encoder": (Model.encode, 0, "enc", 0),
        "inv_fertility": (Model.encode, 0, "inv_fertility", 0),
        "enc_ctx": (Model.encode, 0, "enc_ctx", 0),
        "output/prev:target_embed": (from_scratch_training, 0, "input_embeddings", -1),
        # Note: Some of these commented-out checks are not available anymore because we cleaned up the code.
        # If we want to test this again, we need to re-add the corresponding locals and outputs from rf.scan.
        # "output/weight_feedback": (from_scratch_training, 0, "weight_feedback", 0),
        "output/s": (Model.decode_logits, 0, "s", 0),
        # "output/s_transformed": (from_scratch_training, 0, "s_transformed", 0),
        # "output/energy": (from_scratch_training, 0, "energy", 0),
        # "output/att_weights": (from_scratch_training, 0, "att_weights", 0),
        "output/att": (Model.decode_logits, 0, "att", 0),
        "output/output_prob": (from_scratch_training, 0, "logits", 0),
    }

    from i6_experiments.common.setups.returnn_common import serialization

    exec(serialization.PythonEnlargeStackWorkaroundNonhashedCode.code)

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
    time_dim = Dim(
        name="time",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("time_size", dims=[batch_dim], dtype="int32"),
    )
    target_dim = Dim(name="target", dimension=10_025, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
        [str(i) for i in range(target_dim.dimension)], eos_label=0
    )
    data = Tensor("data", dim_tags=[batch_dim, time_dim])
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor(
        "target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim
    )

    from ._moh_att_2023_04_24_BxqgICRSGkgb_net_dict import net_dict

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            network=net_dict,
            extern_data={
                "audio_features": {"dim_tags": data.dims},
                "bpe_labels": {
                    "dim_tags": target.dims,
                    "sparse_dim": target.sparse_dim,
                },
            },
        )
    )

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
        extern_data,
        dyn_dim_max_sizes={time_dim: 2000},
        dyn_dim_min_sizes={time_dim: 1000},
    )  # raw sample level
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    tf1 = tf.compat.v1
    with tf1.Graph().as_default() as graph, tf1.Session(
        graph=graph
    ).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        if _load_existing_ckpt_in_test:
            ckpt_path = _get_tf_checkpoint_path()
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
        assert set(extern_data_tf_raw_dict.keys()) == set(
            extern_data_numpy_raw_dict.keys()
        )
        feed_dict = {
            extern_data_tf_raw_dict[k]: extern_data_numpy_raw_dict[k]
            for k in extern_data_numpy_raw_dict
        }
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
                    old_model_outputs_data[
                        f"{old_layer_name}:size{i}"
                    ] = tag.dyn_size_ext
                    fetches[
                        f"layer:{old_layer_name}:size{i}"
                    ] = tag.dyn_size_ext.placeholder
        old_model_outputs_fetch = session.run(fetches, feed_dict=feed_dict)

    def _make_new_model():
        return MakeModel.make_model(in_dim, target_dim, num_enc_layers=num_layers)

    rf.select_backend_torch()

    print("*** Convert old model to new model")
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=Checkpoint(index_path=old_tf_ckpt_path),
        make_model_func=_make_new_model,
        map_func=map_param_func_v3,
        epoch=1,
        step=0,
    )
    converter._out_model_dir = tk.Path(ckpt_dir + "/new_model")
    converter.out_checkpoint = tk.Path(ckpt_dir + "/new_model/checkpoint.pt")
    converter.run()

    print("*** Create new model")
    new_model = _make_new_model()

    rf.init_train_step_run_ctx(train_flag=False, step=0)
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_numpy_raw_dict)
    tensor_dict_numpy_to_torch_(extern_data)

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(ckpt_dir + "/new_model/checkpoint.pt")
    pt_module.load_state_dict(checkpoint_state["model"])

    print("*** Forwarding with tracing ...")

    funcs_to_trace_list = [
        Model.encode,
        Model.decode_logits,
        ConformerEncoder.__call__,
        ConformerEncoderLayer.__call__,
        ConformerConvSubsample.__call__,
        from_scratch_training,
    ]
    code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
    captured_tensors = (
        {}
    )  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

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
    from_scratch_training(
        model=new_model,
        data=extern_data["audio_features"],
        data_spatial_dim=time_dim,
        targets=extern_data["bpe_labels"],
        targets_spatial_dim=target_spatial_dim,
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
            raise Exception(
                f"{exc.__class__.__name__} {exc}, new_var_path: {new_var_path}"
            )
        assert isinstance(
            new_out, Tensor
        ), f"new_out: {new_out}, new_var_path: {new_var_path}"
        old_out = old_model_outputs_data[old_layer_name]
        assert old_out.batch_ndim == new_out.batch_ndim
        mapped_axes = new_out.find_matching_dim_map(
            old_out, list(range(old_out.batch_ndim))
        )
        out = new_out.copy_transpose(
            [mapped_axes[i] for i in range(old_out.batch_ndim)]
        )
        fetches["layer:" + old_layer_name] = out.raw_tensor
        for i, tag in enumerate(out.dim_tags):
            if tag.dyn_size_ext:
                fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.raw_tensor
    fetches = {k: v.detach().cpu().numpy() for (k, v) in fetches.items()}
    new_model_outputs_fetch = fetches

    print("*** Comparing ...")
    print("**** target spatial len:", extern_data_numpy_raw_dict["bpe_labels"].shape[1])
    for out_step in range(extern_data_numpy_raw_dict["bpe_labels"].shape[1]):
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
                if (
                    tag.dyn_size_ext and tag.dyn_size_ext.dim_tags
                ):  # dynamic, and not scalar dyn sizes
                    assert tag.dyn_size_ext.dim_tags == (
                        batch_dim,
                    )  # not implemented otherwise
                    assert (
                        out.batch_dim_axis == 0
                    )  # not implemented otherwise but should be ensured above
                    size_v = old_model_outputs_fetch[f"layer:{old_layer_name}:size{i}"]
                    for b in range(old_v.shape[0]):
                        idx = tuple(
                            [slice(b, b + 1)]
                            + [slice(None, None)] * (i - 1)
                            + [slice(size_v[b], None)]
                        )
                        old_v[idx] = 0
                        new_v[idx] = 0
            print(f"* Comparing {out}: {old_layer_name!r} vs {new_var_path!r}")
            assert old_v.shape == new_v.shape
            if target_spatial_dim in out.dim_tags:
                assert (
                    out.get_axis_from_description(target_spatial_dim) == 1
                )  # not implemented otherwise
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
                    remarks.append(
                        "[%s]:? (both are nan)" % ",".join([str(i) for i in idx])
                    )
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


def test_import_search():
    from returnn.util import debug

    debug.install_lib_sig_segfault()
    debug.install_native_signal_handler()
    debug.init_faulthandler()

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
    time_dim = Dim(
        name="time",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("time_size", dims=[batch_dim], dtype="int32"),
    )
    target_dim = Dim(name="target", dimension=10_025, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(
        [str(i) for i in range(target_dim.dimension)], eos_label=0
    )
    data = Tensor(
        "data",
        dim_tags=[batch_dim, time_dim, Dim(1, name="dummy-feature")],
        feature_dim_axis=-1,
    )
    target_spatial_dim = Dim(
        name="target_spatial",
        dimension=None,
        kind=Dim.Types.Spatial,
        dyn_size_ext=Tensor("target_spatial_size", dims=[batch_dim], dtype="int32"),
    )
    target = Tensor(
        "target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim
    )

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            extern_data={
                "audio_features": {"dim_tags": data.dims, "feature_dim_axis": -1},
                "bpe_labels": {
                    "dim_tags": target.dims,
                    "sparse_dim": target.sparse_dim,
                },
            },
        )
    )

    # data e.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
    search_data_opts = {
        "class": "MetaDataset",
        "data_map": {
            "audio_features": ("zip_dataset", "data"),
            "bpe_labels": ("zip_dataset", "classes"),
        },
        "datasets": {
            "zip_dataset": {
                "class": "OggZipDataset",
                "path": generic_job_output(
                    "i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M/output/out.ogg.zip"
                ).get_path(),
                "use_cache_manager": True,
                "audio": {
                    "features": "raw",
                    "peak_normalization": True,
                    "preemphasis": None,
                },
                "targets": {
                    "class": "BytePairEncoding",
                    "bpe_file": generic_job_output(
                        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes"
                    ).get_path(),
                    "vocab_file": generic_job_output(
                        "i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
                    ).get_path(),
                    "unknown_label": "<unk>",
                    "seq_postfix": [0],
                },
                "segment_file": None,
                "partition_epoch": 1,
                # "seq_ordering": "sorted_reverse",
            }
        },
        "seq_order_control_dataset": "zip_dataset",
    }

    print("*** Construct input minibatch")
    extern_data = TensorDict()
    extern_data.update(config.typed_dict["extern_data"], auto_convert=True)

    from returnn.datasets.basic import init_dataset, Batch
    from returnn.tf.data_pipeline import FeedDictDataProvider, BatchSetGenerator

    dataset = init_dataset(search_data_opts)
    dataset.init_seq_order(
        epoch=1,
        # seq_list=["dev-other/5543-27761-0074/5543-27761-0074"],
        seq_list=[
            f"dev-other/116-288045-{i:04d}/116-288045-{i:04d}" for i in range(33)
        ],
    )
    batch_num_seqs = 1
    # batch_num_seqs = 10
    dataset.load_seqs(0, batch_num_seqs)
    batch = Batch()
    for seq_idx in range(batch_num_seqs):
        batch.add_sequence_as_slice(
            seq_idx=seq_idx, seq_start_frame=0, length=dataset.get_seq_length(seq_idx)
        )
    batches = BatchSetGenerator(dataset, generator=iter([batch]))
    data_provider = FeedDictDataProvider(
        extern_data=extern_data,
        data_keys=list(extern_data.data.keys()),
        dataset=dataset,
        batches=batches,
    )
    batch_data = data_provider.get_next_batch()

    for key, data in extern_data.data.items():
        data.placeholder = batch_data[key]
        key_seq_lens = f"{key}_seq_lens"
        if key_seq_lens in batch_data:
            seq_lens = data.dims[1]
            if not seq_lens.dyn_size_ext:
                seq_lens.dyn_size_ext = Tensor(
                    key_seq_lens, dims=[batch_dim], dtype="int32"
                )
            seq_lens.dyn_size_ext.placeholder = batch_data[key_seq_lens]
    if not batch_dim.dyn_size_ext:
        batch_dim.dyn_size_ext = Tensor("batch_dim", dims=[], dtype="int32")
    batch_dim.dyn_size_ext.placeholder = numpy.array(
        batch_data["batch_dim"], dtype="int32"
    )
    extern_data_numpy_raw_dict = extern_data.as_raw_tensor_dict()
    extern_data.reset_content()

    rf.select_backend_torch()

    print("*** Convert old model to new model")
    # pt_checkpoint_path = _get_pt_checkpoint_path()
    pt_checkpoint_path = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_10_18/average.pt"
    print(pt_checkpoint_path)

    print("*** Create new model")
    ctc_only = False
    time_sync = False
    time_sync_search_args = {
        "beam_size": 2,
        "att_scale": 0.65,
        "ctc_scale": 0.35,
        "mask_eos": False,
    }
    search_args = {
        "beam_size": 12,
        "use_ctc": True,
        "att_scale": 0.0,
        "ctc_scale": 1.0,
        "ctc_state_fix": True,
    }
    if time_sync:
        search_args = time_sync_search_args

    model_args = {
        "add_lstm_lm": True,
    }
    # returnn/torch/fonrtend/_backend 1635: sizes_raw = torch.reshape(sizes.raw_tensor, [batch_dim]).to('cpu')
    new_model = MakeModel.make_model(
        in_dim,
        target_dim,
        num_enc_layers=num_layers,
        search_args=search_args,
        model_args=model_args,
    )

    from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_

    rf.init_train_step_run_ctx(train_flag=False, step=0)
    extern_data.reset_content()
    extern_data.assign_from_raw_tensor_dict_(extern_data_numpy_raw_dict)
    tensor_dict_numpy_to_torch_(extern_data)

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(pt_checkpoint_path)
    # checkpoint_state = torch.load(pt_checkpoint_path.get_path())
    pt_module.load_state_dict(checkpoint_state["model"])

    cuda = torch.device("cuda")
    pt_module.to(cuda)
    extern_data["audio_features"].raw_tensor = extern_data[
        "audio_features"
    ].raw_tensor.to(cuda)

    print("*** Search ...")

    with torch.no_grad():
        with rf.set_default_device_ctx("cuda"):
            # manually switch between different decoders
            if ctc_only:
                seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog_ctc(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
            elif time_sync:
                (
                    seq_targets,
                    seq_log_prob,
                    out_spatial_dim,
                    beam_dim,
                ) = model_recog_time_sync(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
            else:
                seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog(
                    model=new_model,
                    data=extern_data["audio_features"],
                    data_spatial_dim=time_dim,
                )
    print(seq_targets, seq_targets.raw_tensor)  # seq_targets [T,Batch,Beam]
    print("Out spatial dim:", out_spatial_dim)

    # serialize output
    vocab_1 = Vocabulary(
        "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
        eos_label=0,
    )
    for batch_idx in range(batch_dim.get_dim_value()):
        # process seq
        hyps = seq_targets.raw_tensor[:, batch_idx, :]
        scores = seq_log_prob.raw_tensor[batch_idx, :]
        hyps_len = seq_targets.dims[0].dyn_size_ext.raw_tensor[:, batch_idx]
        num_beam = hyps.shape[1]
        only_max = True
        if only_max:
            max_score_idx = torch.argmax(scores)
            score = float(scores[max_score_idx])
            hyp_ids = hyps[: hyps_len[max_score_idx], max_score_idx]
            hyp_serialized = vocab_1.get_seq_labels(hyp_ids)
            print(f"  ({score!r}, {hyp_serialized!r}),\n")
            continue
        for i in range(num_beam):
            score = float(scores[i])
            hyp_ids = hyps[: hyps_len[i], i]
            hyp_serialized = vocab_1.get_seq_labels(hyp_ids)
            print(f"  ({score!r}, {hyp_serialized!r}),\n")


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_import_search


if __name__ == "__main__":
    mod_name = __package__
    if mod_name.startswith("recipe."):
        mod_name = mod_name[len("recipe.") :]
    mod_name += "." + os.path.basename(__file__)[: -len(".py")]
    map_param_func_v3.__module__ = mod_name
    test_import_search()
