from __future__ import annotations

from typing import Dict

import os
import sys
import numpy

from sisyphus import tk

# from i6_core.returnn.training import Checkpoint
# from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
# from i6_experiments.users.gaudino.returnn.convert_ckpt_rf import (
#     ConvertTfCheckpointToRfPtJob,
# )

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    MakeModel,
)

import returnn.frontend as rf


_returnn_tf_ckpt_filename = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average"
_ted2_lm_ckpt_filename = "/work/asr4/michel/setups-data/language_modelling/tedlium/neurallm/trafo_kazuki19/net-model/network.020"

# def _get_pt_checkpoint_path() -> tk.Path:
#     old_tf_ckpt_path = generic_job_output(lm_path)
#     old_tf_ckpt = Checkpoint(index_path=old_tf_ckpt_path)
#     make_model_func = MakeModel(10_025, 10_025)
#     # TODO: problems with hash:
#     #  make_model_func, map_func: uses full module name (including "zeyer"), should use sth like unhashed_package_root
#     #  https://github.com/rwth-i6/sisyphus/issues/144
#     converter = ConvertTfCheckpointToRfPtJob(
#         checkpoint=old_tf_ckpt,
#         make_model_func=make_model_func,
#         map_func=map_param_func,
#         epoch=1,
#         step=0,
#     )
#     # converter.run()
#     return converter.out_checkpoint


def test_convert_checkpoint():
    """run"""
    import returnn.frontend as rf
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.util.basic import model_epoch_from_filename
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    import torch
    import numpy

    out_dir = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/baseline_w_trafo_lm_23_11_09"

    reader = CheckpointReader(_returnn_tf_ckpt_filename)
    reader_lm = CheckpointReader(lm_path)

    print("Input checkpoint:")
    print(reader.debug_string().decode("utf-8"))
    print(reader_lm.debug_string().decode("utf-8"))
    print()

    print("Creating model...")
    rf.select_backend_torch()
    search_args = {
        "add_lstm_lm": False,
    }
    model_args = {
        "target_embed_dim": 256
    }
    model = MakeModel(80, 1_057, model_args=model_args, search_args=search_args)()
    print("Created model:", model)
    print("Model parameters:")
    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)
        print(f"{name}: {param}")
    print()

    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)

        value = map_param_func_v3(reader, name, param)
        assert isinstance(value, numpy.ndarray)
        # noinspection PyProtectedMember
        param._raw_backend.set_parameter_initial_value(param, value)

    epoch = 1
    if epoch is None:
        epoch = model_epoch_from_filename(tk.Path(_returnn_tf_ckpt_filename))
        pass

    step = 0
    if step is None:
        assert reader.has_tensor("global_step")
        step = int(reader.get_tensor("global_step"))

    ckpt_name = os.path.basename(_returnn_tf_ckpt_filename)

    pt_model = rf_module_to_pt_module(model)

    save_model = False
    if save_model:
        os.makedirs(out_dir, exist_ok=True)
        filename = out_dir + "/" + ckpt_name + ".pt"
        print(f"*** saving PyTorch model checkpoint: {filename}")
        torch.save({"model": pt_model.state_dict(), "epoch": epoch, "step": step}, filename)

        if ckpt_name != "checkpoint":
            symlink_filename = out_dir + "/checkpoint.pt"
            print(
                f"*** creating symlink {symlink_filename} -> {os.path.basename(filename)}"
            )
            os.symlink(os.path.basename(filename), symlink_filename)
            # create meta information
            meta_filename = out_dir + "/" + ckpt_name + "." + str(epoch) + ".meta"
            open(meta_filename, "w").close()
            symlink_filename_1 = out_dir + "/checkpoint." + str(epoch) + ".meta"
            symlink_filename_2 = out_dir + "/" + ckpt_name + ".meta"
            os.symlink(os.path.basename(meta_filename), symlink_filename_1)
            os.symlink(os.path.basename(meta_filename), symlink_filename_2)
        # assert os.path.exists(self.out_checkpoint.get_path())


_ParamMapping = {}  # type: Dict[str,str]


def _add_params_conformer():
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


_add_params_conformer()


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
        assert reader.has_tensor(var_name), f"missing {var_name}"
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

if __name__ == "__main__":
    test_convert_checkpoint()
