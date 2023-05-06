"""Param Import
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence
import numpy
import sys

from sisyphus import tk

from returnn.tensor import Tensor, Dim, batch_dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_core.returnn.training import Checkpoint

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef, ModelWithCheckpoint
from i6_experiments.users.zeyer.recog import recog_model
from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob


def sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    task = get_switchboard_task_bpe1k()

    epoch = 300
    new_chkpt = ConvertTfCheckpointToRfPtJob(
        checkpoint=Checkpoint(
            index_path=tk.Path(
                f"/u/zeyer/setups/combined/2021-05-31"
                f"/alias/exp_fs_base/old_nick_att_conformer_lrs2/train/output/models/epoch.{epoch:03}.index"
            )
        ),
        make_model_func=MakeModel(
            extern_data_dict=task.train_dataset.get_extern_data(),
            default_input_key=task.train_dataset.get_default_input(),
            default_target_key=task.train_dataset.get_default_target(),
        ),
        map_func=map_param_func_v2,
    ).out_checkpoint
    model_with_checkpoint = ModelWithCheckpoint(definition=from_scratch_model_def, checkpoint=new_chkpt)

    res = recog_model(task, model_with_checkpoint, model_recog)
    tk.register_output(prefix_name + f"/recog_results_per_epoch/{epoch:03}", res.output)


class MakeModel:
    """for import"""

    def __init__(self, *, extern_data_dict, default_input_key, default_target_key):
        self.extern_data_dict = extern_data_dict
        self.default_input_key = default_input_key
        self.default_target_key = default_target_key

    def __call__(self) -> Model:
        data = Tensor(name=self.default_input_key, **self.extern_data_dict[self.default_input_key])
        targets = Tensor(name=self.default_target_key, **self.extern_data_dict[self.default_target_key])
        in_dim = data.feature_dim_or_sparse_dim
        target_dim = targets.feature_dim_or_sparse_dim
        return self.make_model(in_dim, target_dim)

    @classmethod
    def make_model(cls, in_dim: Dim, target_dim: Dim, *, num_enc_layers: int = 12) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            nb_target_dim=target_dim,
            wb_target_dim=target_dim + 1,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
        )


_ParamMapping = {}  # type: Dict[str,str]


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
        if name.endswith(".filter"):
            value = _convert_tf_conv_to_pt_conv_filter(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.ff_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = _convert_tf_lstm_to_native_lstm_ff(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.rec_weight":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
        value = _convert_tf_lstm_to_native_lstm_rec(value)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if name == "s.bias":
        value = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")
        value = _convert_tf_lstm_to_native_lstm_bias(value, forget_gate_bias=1.0)
        assert value.shape == var.batch_shape, name
        assert value.dtype.name == var.dtype, name
        return value

    if ".conv_block.norm." in name:
        assert name.startswith("encoder.layers.")
        layer_idx = int(name.split(".")[2])
        value = _convert_tf_batch_norm_to_rf(
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


def _convert_tf_conv_to_pt_conv_filter(tf_filter: numpy.ndarray) -> numpy.ndarray:
    # in: (*filter_size, in_dim, out_dim)
    # out: (out_dim, in_dim, *filter_size)
    assert tf_filter.ndim >= 3
    n = tf_filter.ndim
    return tf_filter.transpose([n - 1, n - 2] + list(range(n - 2)))


def _convert_tf_batch_norm_to_rf(
    *,
    reader,
    rf_name: str,
    rf_prefix_name: str,
    tf_prefix_name: str,
    var: rf.Parameter,
) -> numpy.ndarray:
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader

    assert isinstance(reader, CheckpointReader)
    assert rf_name.startswith(rf_prefix_name)
    rf_suffix = rf_name[len(rf_prefix_name) :]
    tf_suffix = {"running_mean": "mean", "running_variance": "variance", "gamma": "gamma", "beta": "beta"}[rf_suffix]

    # TF model with earlier BN versions has strange naming
    tf_var_names = [
        name
        for name in reader.get_variable_to_shape_map()
        if name.startswith(tf_prefix_name) and name.endswith("_" + tf_suffix)
    ]
    assert len(tf_var_names) == 1, f"found {tf_var_names} for {rf_name}"
    value = reader.get_tensor(tf_var_names[0])
    assert var.batch_ndim == 1
    value = numpy.squeeze(value)
    assert value.ndim == 1 and value.shape == var.batch_shape
    return value


def _convert_tf_lstm_to_native_lstm_ff(old_w_ff_re: numpy.ndarray) -> numpy.ndarray:
    # in: (in_dim+dim, 4*dim)
    # out: (4*dim, in_dim)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_w_ff_re.ndim == 2
    assert old_w_ff_re.shape[1] % 4 == 0
    n_out = old_w_ff_re.shape[1] // 4
    assert old_w_ff_re.shape[0] > n_out
    n_in = old_w_ff_re.shape[0] - n_out
    old_w_ff, _ = numpy.split(old_w_ff_re, [n_in], axis=0)  # (in_dim,4*dim)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: i, j, f, o; Input: [inputs, h]
    # NativeLstm2: j, i, f, o
    old_w_ff_i, old_w_ff_j, old_w_ff_f, old_w_ff_o = numpy.split(old_w_ff, 4, axis=1)
    new_w_ff = numpy.concatenate([old_w_ff_j, old_w_ff_i, old_w_ff_f, old_w_ff_o], axis=1)  # (in_dim,4*dim)
    new_w_ff = new_w_ff.transpose()  # (4*dim,in_dim)
    return new_w_ff


def _convert_tf_lstm_to_native_lstm_rec(old_w_ff_re: numpy.ndarray) -> numpy.ndarray:
    # in: (in_dim+dim, 4*dim)
    # out: (4*dim, in_dim)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_w_ff_re.ndim == 2
    assert old_w_ff_re.shape[1] % 4 == 0
    n_out = old_w_ff_re.shape[1] // 4
    assert old_w_ff_re.shape[0] > n_out
    n_in = old_w_ff_re.shape[0] - n_out
    _, old_w_rec = numpy.split(old_w_ff_re, [n_in], axis=0)  # (dim,4*dim)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: i, j, f, o; Input: [inputs, h]
    # NativeLstm2: j, i, f, o
    old_w_rec_i, old_w_rec_j, old_w_rec_f, old_w_rec_o = numpy.split(old_w_rec, 4, axis=1)
    new_w_rec = numpy.concatenate([old_w_rec_j, old_w_rec_i, old_w_rec_f, old_w_rec_o], axis=1)  # (dim,4*dim)
    new_w_rec = new_w_rec.transpose()  # (4*dim,dim)
    return new_w_rec


def _convert_tf_lstm_to_native_lstm_bias(old_bias: numpy.ndarray, *, forget_gate_bias: float = 1.0) -> numpy.ndarray:
    # in: (4*dim,)
    # out: (4*dim,)
    # See CustomCheckpointLoader MakeLoadBasicToNativeLstm.
    assert old_bias.ndim == 1
    assert old_bias.shape[0] % 4 == 0
    n_out = old_bias.shape[0] // 4
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    # BasicLSTM: i, j, f, o; Input: [inputs, h]
    # NativeLstm2: j, i, f, o
    old_bias_i, old_bias_j, old_bias_f, old_bias_o = numpy.split(old_bias, 4, axis=0)
    old_bias_f += forget_gate_bias
    new_bias = numpy.concatenate([old_bias_j, old_bias_i, old_bias_f, old_bias_o], axis=0)  # (4*dim,)
    return new_bias


# See comment below, use `py = test_import` to easily run this.
def test_import():
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
    from pprint import pprint

    # Pick some layers to check outputs for equality.
    # See the func tracing logic below, entries in captured_tensors.
    _layer_mapping = {
        "source": (Model.encode, 0, "source", -1),
        "conv_merged": (ConformerEncoder.__call__, 0, "x_subsample", 0),
        "source_linear": (ConformerEncoder.__call__, 0, "x_linear", 0),
        "conformer_block_01_ffmod_1_drop2": (ConformerEncoderLayer.__call__, 0, "x_ffn1", 0),
        "conformer_block_01_ffmod_1_res": (ConformerEncoderLayer.__call__, 0, "x_ffn1_out", 0),
        "conformer_block_01_self_att_res": (ConformerEncoderLayer.__call__, 0, "x_mhsa_out", 0),
        "conformer_block_01_conv_mod_res": (ConformerEncoderLayer.__call__, 0, "x_conv_out", 0),
        "conformer_block_01_ffmod_2_res": (ConformerEncoderLayer.__call__, 0, "x_ffn2_out", 0),
        "conformer_block_01": (ConformerEncoderLayer.__call__, 1, "inp", 0),
        "encoder": (Model.encode, 0, "enc", 0),
    }

    from i6_experiments.common.setups.returnn_common import serialization

    exec(serialization.PythonEnlargeStackWorkaroundNonhashedCode.code)

    from returnn.datasets.util.vocabulary import Vocabulary

    in_dim = Dim(name="in", dimension=80, kind=Dim.Types.Feature)
    time_dim = Dim(name="time", dimension=None, kind=Dim.Types.Spatial)
    target_dim = Dim(name="target", dimension=1030, kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels([str(i) for i in range(target_dim.dimension)], eos_label=0)
    data = Tensor("data", dim_tags=[batch_dim, time_dim])
    target_spatial_dim = Dim(name="target_spatial", dimension=None, kind=Dim.Types.Spatial)
    target = Tensor("target", dim_tags=[batch_dim, target_spatial_dim], sparse_dim=target_dim)

    from ._moh_att_2023_04_24_BxqgICRSGkgb import net_dict

    num_layers = 12

    from returnn.config import Config

    config = Config(
        dict(
            log_verbositiy=5,
            network=net_dict,
            extern_data={
                "audio_features": {"dim_tags": data.dims},
                "bpe_labels": {"dim_tags": target.dims, "sparse_dim": target.sparse_dim},
            },
        )
    )

    from returnn.tf.network import TFNetwork
    import tensorflow as tf
    import numpy.testing
    import tempfile
    import atexit
    import shutil

    ckpt_dir = tempfile.mkdtemp("returnn-import-test")
    atexit.register(lambda: shutil.rmtree(ckpt_dir))

    print("*** Construct TF graph for old model")
    n_batch = 3
    n_time = 1231  # raw sample level
    x_sizes = [n_time, n_time - 100, n_time - 200]
    rnd = numpy.random.RandomState(42)
    x_np = rnd.rand(n_batch, n_time).astype("float32")

    tf1 = tf.compat.v1
    with tf1.Graph().as_default() as graph, tf1.Session(graph=graph).as_default() as session:
        net = TFNetwork(config=config)
        net.construct_from_dict(config.typed_dict["network"])
        print("*** Random init old model")
        net.initialize_params(session)
        print("*** Save old model to disk")
        net.save_params_to_file(ckpt_dir + "/old_model/model", session=session)

        print("*** Forwarding ...")

        x = net.extern_data.data["audio_features"]
        feed_dict = {x.placeholder: x_np, x.size_placeholder[0]: x_sizes}
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
        return MakeModel.make_model(in_dim, target_dim, num_enc_layers=num_layers)

    rf.select_backend_torch()

    print("*** Convert old model to new model")
    converter = ConvertTfCheckpointToRfPtJob(
        checkpoint=Checkpoint(index_path=tk.Path(ckpt_dir + "/old_model/model.index")),
        make_model_func=_make_new_model,
        map_func=map_param_func_v2,
        epoch=1,
        step=0,
    )
    converter._out_model_dir = tk.Path(ckpt_dir + "/new_model")
    converter.out_checkpoint = tk.Path(ckpt_dir + "/new_model/model.pt")
    converter.run()

    print("*** Create new model")
    new_model = _make_new_model()

    rf.init_forward_step_run_ctx()
    batch_dim.dyn_size_ext = rf.convert_to_tensor(n_batch, dims=())
    batch_dim.batch = None
    time_dim.dyn_size_ext = rf.convert_to_tensor(numpy.array(x_sizes), dims=[batch_dim])
    time_dim.batch = None
    x = rf.convert_to_tensor(x_np, dims=[batch_dim, time_dim])

    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("*** Load new model params from disk")
    pt_module = rf_module_to_pt_module(new_model)
    checkpoint_state = torch.load(ckpt_dir + "/new_model/model.pt")
    pt_module.load_state_dict(checkpoint_state["model"])

    print("*** Forwarding with tracing ...")

    funcs_to_trace_list = [
        Model.encode,
        ConformerEncoder.__call__,
        ConformerEncoderLayer.__call__,
        ConformerConvSubsample.__call__,
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
    y = new_model.encode(x, in_spatial_dim=time_dim)[0]["enc"]
    sys.settrace(None)
    pprint(captured_tensors)
    y.mark_as_default_output()

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
        mapped_axes = new_out.find_matching_dim_map(old_out, list(range(old_out.batch_ndim)))
        out = new_out.copy_transpose([mapped_axes[i] for i in range(old_out.batch_ndim)])
        fetches["layer:" + old_layer_name] = out.raw_tensor
        for i, tag in enumerate(out.dim_tags):
            if tag.dyn_size_ext:
                fetches[f"layer:{old_layer_name}:size{i}"] = tag.dyn_size_ext.raw_tensor
    fetches = {k: v.detach().cpu().numpy() for (k, v) in fetches.items()}
    new_model_outputs_fetch = fetches

    print("*** Comparing ...")
    for old_layer_name, new_var_path in _layer_mapping.items():
        out = old_model_outputs_data[old_layer_name]
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
        print("* Comparing", old_layer_name, "vs", new_var_path)
        numpy.testing.assert_almost_equal(old_v, new_v, decimal=5, err_msg=f"{old_layer_name}/{new_var_path} mismatch")

    print("*** Done, all correct (!), exit now ***")
    raise SystemExit("done")


# `py` is the default sis config function name. so when running this directly, run the import test.
# So you can just run:
# `sis m recipe/i6_experiments/users/zeyer/experiments/....py`
py = test_import


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        nb_target_dim: Dim,
        wb_target_dim: Dim,
        blank_idx: int,
        bos_idx: int,
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = Dim(name="enc_win_dim", dimension=5)

        self.inv_fertility = rf.Linear(self.encoder.out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(nb_target_dim, Dim(name="target_embed", dimension=640))

        self.s = rf.LSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="lstm", dimension=1024),
            # TODO
            # zoneout_factor_cell=0.15,
            # zoneout_factor_output=0.05,
        )

        self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder.out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, nb_target_dim)

        for p in self.parameters():
            p.weight_decay = l2

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim, in_dim_ = rf.stft(
            source, in_spatial_dim=in_spatial_dim, frame_step=160, frame_length=400, fft_length=512
        )
        source = rf.abs(source) ** 2.0
        source = rf.mel_filterbank(source, in_dim=in_dim_, out_dim=self.in_dim, sampling_rate=16000)
        source = rf.safe_log(source, eps=1e-10) / 2.3026
        # TODO specaug
        # source = specaugment_wei(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)  # TODO
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = self.inv_fertility(enc)
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, rf.Tensor]) -> Dict[str, rf.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = rf.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
        """Default initial state"""
        return rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
            accum_att_weights=rf.zeros(list(batch_dims) + [enc_spatial_dim, self.att_num_heads]),
        )

    def decode(
        self,
        *,
        enc: rf.Tensor,
        enc_ctx: rf.Tensor,
        inv_fertility: rf.Tensor,
        enc_spatial_dim: Dim,
        prev_nb_target: Optional[rf.Tensor] = None,  # non-blank
        prev_nb_target_spatial_dim: Optional[Dim] = None,
        state: Optional[rf.State] = None,
    ) -> Tuple[rf.Tensor, rf.State]:
        """decoder step, or operating on full seq. output logits + state"""
        if state is None:
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
        state_ = rf.State()

        prev_target_embed = self.target_embed(prev_nb_target)
        prev_att = state.att

        s, state_.s = self.s(
            rf.concat_features(prev_target_embed, prev_att), spatial_dim=prev_nb_target_spatial_dim, state=state.s
        )

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim)
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
        state_.att = att

        readout_in = self.readout_in(rf.concat_features(s, prev_target_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2)
        readout = rf.dropout(readout, drop_prob=0.3, axis=readout.feature_dim)
        logits = self.output_prob(readout)

        return logits, state_


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    epoch  # noqa
    return MakeModel.make_model(in_dim, target_dim)


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 14


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    prev_targets, prev_targets_spatial_dim = rf.prev_target_seq(
        targets, spatial_dim=targets_spatial_dim, bos_idx=model.bos_idx, out_one_longer=True
    )
    probs, _ = model.decode(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        all_combinations_out=True,
        prev_nb_target=prev_targets,
        prev_nb_target_spatial_dim=prev_targets_spatial_dim,
    )
    out_log_prob = probs.get_wb_label_log_probs()
    loss = rf.transducer_time_sync_full_sum_neg_log_prob(
        log_probs=out_log_prob,
        labels=targets,
        input_spatial_dim=enc_spatial_dim,
        labels_spatial_dim=targets_spatial_dim,
        prev_labels_spatial_dim=prev_targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss("full_sum")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    targets_dim: Dim,  # noqa
) -> rf.Tensor:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return: recog results including beam
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    loop = rf.Loop(axis=enc_spatial_dim)  # time-sync transducer
    loop.max_seq_len = rf.dim_value(enc_spatial_dim) * 2
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims)
    loop.state.target = rf.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
    with loop:
        enc = model.encoder_unstack(enc_args)
        probs, loop.state.decoder = model.decode(
            **enc,
            enc_spatial_dim=single_step_dim,
            wb_target_spatial_dim=single_step_dim,
            prev_wb_target=loop.state.target,
            state=loop.state.decoder,
        )
        log_prob = probs.get_wb_label_log_probs()
        loop.state.target = rf.choice(
            log_prob, input_type="log_prob", target=None, search=True, beam_size=beam_size, length_normalization=False
        )
        res = loop.stack(loop.state.target)

    assert model.blank_idx == targets_dim.dimension  # added at the end
    res.feature_dim.vocab = rf.Vocabulary.create_vocab_from_labels(
        targets_dim.vocab.labels + ["<blank>"], user_defined_symbols={"<blank>": model.blank_idx}
    )
    return res


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
