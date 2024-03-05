from __future__ import annotations

from typing import Dict, Optional

import os
import sys
import torch
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

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.lm_import_2023_11_09 import (
    MakeModel as MakeModelLM,
)

import returnn.frontend as rf

from i6_core.returnn.training import Checkpoint

from itertools import product


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

models = {
    "model_baseline": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.yB4JK4GDCxWG/output/model/average.index"
            )
        ),
    },
    # ctcScale models
    "model_ctc0.43_att1.0": {  # ctcScale 0.3
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.nCrQhRfqIRiZ/output/model/average.index"
            )
        ),
    },
    "model_ctc0.25_att1.0": {  # ctcScale 0.2
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.CknpN55pjOHo/output/model/average.index"
            )
        ),
    },
    "model_ctc0.2_att1.0": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.ro9g9W6DBJpW/output/model/average.index"
            )
        ),
    },
    # 1-y models
    "model_ctc0.3_att0.7": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.jGxeW6yzeoG7/output/model/average.index"
            )
        ),
    },
    "model_ctc0.2_att0.8": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.6qWPnvXHalfJ/output/model/average.index"
            )
        ),
    },
    "model_ctc0.1_att0.9": {  # pre 4
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.MEtpESN5M4oD/output/model/average.index"
            )
        ),
    },
    "model_ctc0.001_att0.999": {  # pre 4
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.eEEAEAZQiFvO/output/model/average.index"
            )
        ),
    },
    # att only
    "model_att_only_currL": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.io6cKw6ETnHp/output/model/average.index"
            )
        ),
    },
    "model_att_only_adjSpec": {
        "ckpt": Checkpoint(
            tk.Path(
                "work/i6_core/returnn/training/AverageTFCheckpointsJob.9f6nlw1UOxVO/output/model/average.index"
            )
        ),
    },
    # ctc only
    "model_ctc_only": {
        "ckpt": Checkpoint(
            tk.Path(
                "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400.index"
            )
        ),  # last
    },
}


def convert_checkpoint(
    *,
    model_args,
    ckpt_path: str,
    ckpt_path_lm: str,
    ckpt_path_sep: Optional[str] = None,
    out_dir: str,
):
    """run"""
    import returnn.frontend as rf
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.util.basic import model_epoch_from_filename
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader

    search_args = {
        "add_lstm_lm": False,
    }

    reader = CheckpointReader(ckpt_path)
    reader_lm = CheckpointReader(ckpt_path_lm)
    if model_args.get("encoder_ctc", False):
        reader_2 = CheckpointReader(ckpt_path_sep)

    print("Input checkpoint:")
    print(reader.debug_string().decode("utf-8"))
    print(reader_lm.debug_string().decode("utf-8"))
    if model_args.get("encoder_ctc", False):
        print(reader_2.debug_string().decode("utf-8"))
    print()

    print("Creating model...")
    rf.select_backend_torch()
    model = MakeModel(80, 1_057, model_args=model_args, search_args=search_args)()
    print("Created model:", model)
    print("Model parameters:")
    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)
        print(f"{name}: {param}")
    print()

    print("Create ParamMapping...")
    param_mapping = {}
    _add_params_conformer(param_mapping, prefix="")
    _add_params_att_decoder(param_mapping)
    _add_params_trafo_lm(param_mapping)
    # if model_args.get("encoder_ctc", False):
    #     _add_params_conformer(param_mapping, prefix="sep_enc_ctc_")

    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)

        if name.startswith("trafo_lm."):
            sub_name = name.removeprefix("trafo_lm.")
            value = map_param_func_trafo_lm(reader_lm, sub_name, param, param_mapping)
        elif name.startswith("sep_enc_ctc_"):
            sub_name = name.removeprefix("sep_enc_ctc_")
            value = map_param_func_v3(reader_2, sub_name, param, param_mapping)
        else:
            value = map_param_func_v3(reader, name, param, param_mapping)
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

    save_model = True

    if save_model:
        os.makedirs(out_dir, exist_ok=True)
        filename = out_dir + "/" + ckpt_name + ".pt"
        print(f"*** saving PyTorch model checkpoint: {filename}")
        torch.save(
            {"model": pt_model.state_dict(), "epoch": epoch, "step": step}, filename
        )

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


def convert_lm(ckpt_path_lm, out_dir, model_target_dim, model_args):
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader
    from returnn.torch.frontend.bridge import rf_module_to_pt_module

    print("Loading checkpoint...")
    reader_lm = CheckpointReader(ckpt_path_lm)

    print("Creating model...")
    rf.select_backend_torch()
    model = MakeModelLM(model_target_dim, model_target_dim, model_args=model_args)()

    print("Create ParamMapping...")
    param_mapping = {}
    _add_params_trafo_lm(param_mapping)

    print("Mapping parameters...")
    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)
        value = map_param_func_trafo_lm(reader_lm, name, param, param_mapping)

        assert isinstance(value, numpy.ndarray)
        # noinspection PyProtectedMember
        param._raw_backend.set_parameter_initial_value(param, value)

    epoch = 1
    step = 0

    print("Converting rf module to pt module...")
    ckpt_name = os.path.basename(ckpt_path_lm)
    pt_model = rf_module_to_pt_module(model)

    save_model = True
    if save_model:
        os.makedirs(out_dir, exist_ok=True)
        filename = out_dir + "/" + ckpt_name + ".pt"
        print(f"Saving PyTorch model checkpoint: {filename}")
        torch.save(
            {"model": pt_model.state_dict(), "epoch": epoch, "step": step}, filename
        )


def _add_params_trafo_lm(param_mapping: Dict[str, str]):
    # add params of trafo lm
    for layer_idx in range(30):
        param_mapping.update(
            {
                f"layers.{layer_idx}.ff_conv1.weight": f"dec_{layer_idx}_ff_conv1/W",
                f"layers.{layer_idx}.ff_conv1.bias": f"dec_{layer_idx}_ff_conv1/b",
                f"layers.{layer_idx}.ff_conv2.weight": f"dec_{layer_idx}_ff_conv2/W",
                f"layers.{layer_idx}.ff_conv2.bias": f"dec_{layer_idx}_ff_conv2/b",
                f"layers.{layer_idx}.ff_layer_norm.scale": f"dec_{layer_idx}_ff_laynorm/scale",
                f"layers.{layer_idx}.ff_layer_norm.bias": f"dec_{layer_idx}_ff_laynorm/bias",
                f"layers.{layer_idx}.self_att.qkv.weight": f"dec_{layer_idx}_self_att_att/QKV",
                f"layers.{layer_idx}.self_att_lin.weight": f"dec_{layer_idx}_self_att_lin/W",
                f"layers.{layer_idx}.self_att_layer_norm.scale": f"dec_{layer_idx}_self_att_laynorm/scale",
                f"layers.{layer_idx}.self_att_layer_norm.bias": f"dec_{layer_idx}_self_att_laynorm/bias",
            }
        )

    param_mapping.update(
        {
            "decoder.scale": "decoder/scale",
            "decoder.bias": "decoder/bias",
            "output.weight": "output/W",
            "output.bias": "output/b",
            "target_embed_lin.weight": "target_embed_lin/W",
            "target_embed_raw.weight": "target_embed_raw/W",
        }
    )


def _add_params_conformer(param_mapping: Dict[str, str], prefix: str):
    # frontend
    for layer_idx in [0, 1, 2]:
        orig_name = "conv0" if layer_idx == 0 else f"subsample_conv{layer_idx - 1}"
        param_mapping.update(
            {
                prefix
                + f"encoder.input_layer.conv_layers.{layer_idx}.filter": f"{orig_name}/W",
                prefix
                + f"encoder.input_layer.conv_layers.{layer_idx}.bias": f"{orig_name}/bias",
            }
        )
    param_mapping.update(
        {
            prefix + "encoder.input_projection.weight": "source_linear/W",
            prefix + "ctc.weight": "ctc/W",
            prefix + "ctc.bias": "ctc/b",
        }
    )
    # conformer
    for layer_idx in range(12):
        # FF
        for sub in [1, 2]:
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/W"
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}.linear_ff.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff1/b"
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.weight"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/W"
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}.linear_out.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ff2/b"
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.scale"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/scale"
            param_mapping[
                prefix + f"encoder.layers.{layer_idx}.ffn{sub}_layer_norm.bias"
            ] = f"conformer_block_{layer_idx + 1:02d}_ffmod_{sub}_ln/bias"
        # conv
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/W"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.positionwise_conv1.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv1/b"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.filter"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/W"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.depthwise_conv.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_depthwise_conv2/bias"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/W"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_block.positionwise_conv2.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_pointwise_conv2/b"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/scale"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.conv_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_conv_mod_ln/bias"
        # self-att
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.self_att.qkv.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att/QKV"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.self_att.proj.weight"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_linear/W"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.self_att_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/scale"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.self_att_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln/bias"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.self_att.learned_pos_emb.pos_emb"
        ] = f"conformer_block_{layer_idx + 1:02d}_self_att_ln_rel_pos_enc/encoding_matrix"
        # final layer norm
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.final_layer_norm.scale"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/scale"
        param_mapping[
            prefix + f"encoder.layers.{layer_idx}.final_layer_norm.bias"
        ] = f"conformer_block_{layer_idx + 1:02d}_ln/bias"


def _add_params_att_decoder(param_mapping: Dict[str, str]):
    param_mapping.update(
        {
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


def map_param_func_v3(
    reader, name: str, var: rf.Parameter, param_mapping: Dict[str, str]
) -> numpy.ndarray:
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

    if name in param_mapping:
        var_name = param_mapping[name]
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


def map_param_func_trafo_lm(
    reader, name: str, var: rf.Parameter, param_mapping: Dict[str, str]
) -> numpy.ndarray:
    """map params, TF to RF"""
    from tensorflow.python.training.py_checkpoint_reader import CheckpointReader

    assert isinstance(reader, CheckpointReader)
    assert isinstance(var, rf.Parameter)

    tf_var_name = name.replace(".", "/")
    if reader.has_tensor(tf_var_name):
        return reader.get_tensor(tf_var_name)

    if name in param_mapping:
        var_name = "output/rec/" + param_mapping[name]
        assert reader.has_tensor(var_name), f"missing {var_name}"
        value = reader.get_tensor(var_name)
        assert isinstance(value, numpy.ndarray)

        assert (
            value.shape == var.batch_shape
        ), f"new param {name} {var.batch_shape} vs ckpt param {var_name} {value.shape}"
        assert (
            value.dtype.name == var.dtype
        ), f"new param {name} {var.dtype} vs ckpt param {var_name} {value.dtype}"
        return value

    raise NotImplementedError(f"cannot map {name!r} {var}")


def import_models():
    for model_name, sep_enc in product(models.keys(), [True, False]):
        model_args = {
            "target_embed_dim": 256,
            "add_ted2_trafo_lm": False,
            "encoder_ctc": sep_enc,
        }

        print(
            f"Converting model {model_name}"
            + (" with separate ctc only encoder" if sep_enc else "")
            + " ..."
        )
        out_dir = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/"
        out_dir_postfix = model_name + ("__ctc_only" if sep_enc else "") + "_24_02_02"

        convert_checkpoint(
            model_args=model_args,
            ckpt_path=models[model_name]["ckpt"].ckpt_path,
            ckpt_path_lm=None,
            ckpt_path_sep=models["model_ctc_only"]["ckpt"].ckpt_path
            if sep_enc
            else None,
            out_dir=out_dir + out_dir_postfix,
        )
        print(
            f"Model {model_name}"
            + (" with separate ctc only encoder" if sep_enc else "")
            + " converted."
        )


if __name__ == "__main__":
    # import_models()
    convert_lm(
        _ted2_lm_ckpt_filename,
        "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/tedlium2/trafo_lm_only_24_02_05",
        1057,
    )
