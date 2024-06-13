from __future__ import annotations

from typing import Dict, Optional

import os
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

from i6_experiments.users.gaudino.models.asr.rf.conformer_rnnt.model_conformer_rnnt import MakeModel as MakeModelRNNT

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_11_09 import (
    MakeModel as MakeModelLM,
)

from i6_experiments.users.gaudino.models.asr.rf.nn_lm.lm_import_2023_09_03 import (
    MakeModel as MakeModelLSTMLM,
)

from i6_experiments.users.gaudino.models.asr.rf.ilm_import_2024_04_17 import (
    MakeModel as MakeModelILM,
)

import returnn.frontend as rf

from itertools import product

_nick_pure_torch_rnnt_ckpt_path = "/work/asr4/rossenbach/sisyphus_work_folders/tts_decoder_asr_work/i6_core/returnn/training/ReturnnTrainingJob.6lwn4XuFkhkI/output/models/epoch.250.pt"

from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.model_ckpt_info import models


def convert_checkpoint(
    *,
    ckpt_path: str,
    ckpt_path_lm: Optional[str] = None,
    ckpt_path_sep: Optional[str] = None,
    out_dir: str,
    print_params: bool = False,
    save_model: bool = True,
):
    """run"""
    import returnn.frontend as rf
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.util.basic import model_epoch_from_filename

    print("Input checkpoint:" + ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if print_params:
        for k, v in ckpt["model"].items():
            print(f"{k}: {v.shape if hasattr(v, 'shape') else v}")
        # print(reader.debug_string().decode("utf-8"))


    print()


    print("Creating model...")
    rf.select_backend_torch()
    model = MakeModelRNNT(80, 1_057)()

    print("Created model:", model)
    print("Model parameters:")
    for name, param in model.named_parameters():
        assert isinstance(name, str)
        assert isinstance(param, rf.Parameter)
        if print_params:
            print(f"{name}: {param}")
    print()

    print("Create ParamMapping...")
    param_mapping = {}
    _add_params_predictor_joiner(param_mapping)
    # _add_params_conformer(param_mapping, prefix="")
    # if not ctc_only:
    #     _add_params_att_decoder(param_mapping)
    # _add_params_trafo_lm(param_mapping)
    # if model_args.get("encoder_ctc", False):
    #     _add_params_conformer(param_mapping, prefix="sep_enc_ctc_")

    for name, param in model.named_parameters():
        if name in param_mapping:
            assert isinstance(name, str)
            assert isinstance(param, rf.Parameter)

            value = map_param_func(ckpt, name, param, param_mapping)
            assert isinstance(value, numpy.ndarray)
            # noinspection PyProtectedMember
            param._raw_backend.set_parameter_initial_value(param, value)

    epoch = 1
    if epoch is None:
        epoch = model_epoch_from_filename(tk.Path(ckpt_path))
        pass

    step = 0
    # if step is None:
    #     assert reader.has_tensor("global_step")
    #     step = int(reader.get_tensor("global_step"))

    ckpt_name = os.path.basename(ckpt_path)

    pt_model = rf_module_to_pt_module(model)

    breakpoint()

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

def _add_params_conformer(param_mapping: Dict[str, str], prefix: str):
    # rf -> pt
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
            # prefix + "ctc.weight": "ctc/W",
            # prefix + "ctc.bias": "ctc/b",
            prefix + "enc_aux_logits_12.weight": "ctc/W",
            prefix + "enc_aux_logits_12.bias": "ctc/b",
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

def _add_params_predictor_joiner(param_mapping: Dict[str, str]):
    # add params of trafo lm
    for layer_idx in range(1):
        param_mapping.update(
            {
                f"predictor.layers.{layer_idx}.ff_weight": f"predictor.lstm_layers.{layer_idx}.weight_ih_l0",

                f"predictor.layers.{layer_idx}.rec_weight": f"predictor.lstm_layers.{layer_idx}.weight_hh_l0",
                f"predictor.layers.{layer_idx}.bias": f"predictor.lstm_layers.{layer_idx}.bias_ih_l0",
            }
        )

    param_mapping.update(
        {
            "predictor.embedding.weight": "predictor.embedding.weight",
            "predictor.input_layer_norm.scale": "predictor.input_layer_norm.weight",
            "predictor.input_layer_norm.bias": "predictor.input_layer_norm.bias",
            "predictor.linear.weight": "predictor.linear.weight",
            "predictor.linear.bias": "predictor.linear.bias",
            "predictor.output_layer_norm.scale": "predictor.output_layer_norm.weight",
            "predictor.output_layer_norm.bias": "predictor.output_layer_norm.bias",
            # joiner
            "joiner.linear.weight": "joiner.linear.weight",
            "joiner.linear.bias": "joiner.linear.bias",
        }
    )


def map_param_func(
    ckpt, name: str, var: rf.Parameter, param_mapping: Dict[str, str]
) -> numpy.ndarray:
    """map params, TF to RF"""
    from i6_experiments.users.zeyer.returnn.convert.params import (
        numpy as convert_params_np,
    )
    from i6_experiments.users.zeyer.returnn.convert.params import (
        tf_to_rf_np as convert_params_tf_to_rf_np,
    )

    assert isinstance(var, rf.Parameter)

    if name in param_mapping:
        breakpoint()
        var_name = param_mapping[name]
        assert name in ckpt["model"].keys(), f"missing {var_name}"
        value = ckpt["model"][name].numpy()
        assert isinstance(value, numpy.ndarray)

        assert (
            value.shape == var.batch_shape
        ), f"new param {name} {var.batch_shape} vs ckpt param {var_name} {value.shape}"
        assert (
            value.dtype.name == var.dtype
        ), f"new param {name} {var.dtype} vs ckpt param {var_name} {value.dtype}"
        return value

    # if name == "s.ff_weight":
    #     value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
    #     value = convert_params_np.convert_tf_lstm_to_native_lstm_ff(value)
    #     assert value.shape == var.batch_shape, name
    #     assert value.dtype.name == var.dtype, name
    #     return value
    #
    # if name == "s.rec_weight":
    #     value = reader.get_tensor("output/rec/s/rec/lstm_cell/kernel")
    #     value = convert_params_np.convert_tf_lstm_to_native_lstm_rec(value)
    #     assert value.shape == var.batch_shape, name
    #     assert value.dtype.name == var.dtype, name
    #     return value
    #
    # if name == "s.bias":
    #     value = reader.get_tensor("output/rec/s/rec/lstm_cell/bias")
    #     value = convert_params_np.convert_tf_lstm_to_native_lstm_bias(
    #         value, forget_gate_bias=1.0
    #     )
    #     assert value.shape == var.batch_shape, name
    #     assert value.dtype.name == var.dtype, name
    #     return value
    #
    # if ".conv_block.norm." in name:
    #     assert name.startswith("encoder.layers.")
    #     layer_idx = int(name.split(".")[2])
    #     value = convert_params_tf_to_rf_np.convert_tf_batch_norm_to_rf(
    #         reader=reader,
    #         rf_name=name,
    #         rf_prefix_name=f"encoder.layers.{layer_idx}.conv_block.norm.",
    #         tf_prefix_name=f"conformer_block_{layer_idx + 1:02d}_conv_mod_bn/batch_norm/",
    #         var=var,
    #     )
    #     assert value.shape == var.batch_shape, name
    #     assert value.dtype.name == var.dtype, name
    #     return value

    raise NotImplementedError(f"cannot map {name!r} {var}")



if __name__ == "__main__":
    convert_checkpoint(ckpt_path=_nick_pure_torch_rnnt_ckpt_path, print_params=True, out_dir="", save_model=False)