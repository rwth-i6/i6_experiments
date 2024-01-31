from typing import Dict, List, Optional, Tuple, Union, Any

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFunction
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config
from i6_core.returnn import CodeWrapper
from i6_experiments.users.berger.network.helpers.conformer import add_conformer_stack as add_conformer_stack_simon
from ...ctc.feat.network_helpers.specaug import add_specaug_layer, add_specaug_layer_v2
from ...ctc.feat.network_helpers.specaug_sort_layer2 import add_specaug_layer as add_specaug_layer_sort_layer2
from ...ctc.feat.network_helpers.conformer_wei import add_conformer_stack as add_conformer_stack_wei
from ...ctc.feat.network_helpers.conformer_wei import add_vgg_stack as add_vgg_stack_wei


def segmental_loss(source):
    """
    L_boost of speech segment
    """
    import tensorflow as tf

    loss = source(0, enforce_batch_major=True)
    mask = source(1, enforce_batch_major=True)
    return tf.where(mask, loss, tf.zeros_like(loss))


def add_transducer_viterbi_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    dropout: Optional[float] = None,
    recognition: bool = False,
    **kwargs,
):
    network[name] = {
        "class": "rec",
        "from": from_list,
        "target": "classes",
        "cheating": False,
        "unit": {
            "embedding": {
                "L2": l2,
                "activation": None,
                "class": "linear",
                "from": "base:mask_label",
                "n_out": 128,
                "with_bias": False,
            },
            "mask_embedding": {
                "axes": "T",
                "class": "pad",
                "from": "embedding",
                "mode": "constant",
                "padding": (1, 0),
                "value": 0,
            },
            "label_lm_1": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": ["embedding" if recognition else "mask_embedding"],
                "n_out": 640,
            },
            "label_lm_2": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": "label_lm_1",
                "n_out": 640,
            },
            "mask_flag": {
                "amount": 1,
                "axis": "T",
                "class": "shift_axis",
                "from": "base:mask_flag",
                "pad": True,
            },
            "unmask_context": {
                "class": "unmask",
                "from": "label_lm_2",
                "mask": "mask_flag",
                "skip_initial": True,
            },
            "unmask_context_reinterpret": {
                "class": "reinterpret_data",
                "from": "unmask_context",
                "size_base": "data:classes",
            },
            "joint_encoding": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": ["data:source", "label_lm_2" if recognition else "unmask_context_reinterpret"],
                "n_out": 1024,
            },
            "ce_loss": {
                "class": "loss",
                "from": "output",
                "loss_": "ce",
            },
            "segmental_loss": {
                "class": "eval",
                "eval": "self.network.get_config().typed_value('segmental_loss')(source)",
                "from": ["ce_loss", "base:mask_flag"],
                "loss": "as_is",
                "loss_opts": {"scale": 5.0},
            },
            "output": {
                "class": "softmax",
                "from": "joint_encoding",
                "loss": "ce",
                "loss_opts": {"label_smoothing": 0.2},
            },
        },
    }
    if recognition:
        network[name]["unit"].update({
            "label_context": {
                "class": "choice",
                "from": "output",
                "input_type": "log_prob",
                "target": "classes",
                "beam_size": 1,
                "initial_output": num_outputs + 1,
            },
            "embedding": {
                "L2": l2,
                "activation": None,
                "class": "linear",
                "from": "prev:label_context",
                "n_out": 128,
                "with_bias": False,
                "initial_output": None,
                "safe_embedding": True,
            },
            "output": {
                "class": "linear",
                "from": "joint_encoding",
                "activation": "log_softmax",
                "n_out": num_outputs,
            },
        })
        for layer in [
            "ce_loss", "mask_embedding", "mask_flag", "segmental_loss", "unmask_context", "unmask_context_reinterpret"
        ]:
            network[name]["unit"].pop(layer)
    return name


def make_conformer_viterbi_transducer_model(
    num_outputs: int,
    conformer_args: Optional[Dict] = None,
    output_args: Optional[Dict] = None,
    conformer_type: str = "wei",
    specaug_old: Optional[Dict[str, Any]] = None,
    recognition: bool = False,
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    from_list = ["data"]

    if recognition:
        python_code = []
    else:
        if specaug_old is not None:
            sort_layer2 = specaug_old.pop("sort_layer2", False)
            specaug_func = add_specaug_layer_sort_layer2 if sort_layer2 else add_specaug_layer
            specaug_old_args = {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
                **specaug_old,
            }
            from_list, python_code = specaug_func(network, from_list=from_list, **specaug_old_args)
        else:
            from_list, python_code = add_specaug_layer_v2(network, from_list=from_list)

        python_code += [segmental_loss]

    if conformer_type == "wei":
        network, from_list = add_vgg_stack_wei(network, from_list)
        conformer_args_full = {
            "pos_enc_clip": 32,
            "batch_norm_fix": True,
            "switch_conv_mhsa_module": True,
            "l2": 5e-06,
            "dropout": 0.1,
            **(conformer_args or {}),
        }
        network, from_list = add_conformer_stack_wei(network, from_list, **conformer_args_full)
    else:
        raise NotImplementedError

    network["encoder"] = {
        "class": "reinterpret_data",
        "from": from_list,
        "size_base": "data:classes",
    }
    add_transducer_viterbi_output_layer(
        network, from_list="encoder", num_outputs=num_outputs, recognition=recognition,
        **{**conformer_args_full, **(output_args or {})}
    )
    if not recognition:
        network.update({
            "enc_output": {
                "class": "softmax",
                "from": "encoder",
                "loss": "ce",
                "loss_opts": {"focal_loss_factor": 1.0},
            },
            "enc_output_loss": {
                "class": "softmax",
                "from": "conformer_6_output",
                "loss": "ce",
                "loss_opts": {"focal_loss_factor": 1.0},
                "loss_scale": 0.3,
            },
            "mask_flag": {
                "class": "compare",
                "from": "data:classes",
                "kind": "not_equal",
                "value": 0,
            },
            "mask_label": {
                "class": "masked_computation",
                "from": "data:classes",
                "mask": "mask_flag",
                "unit": {"class": "copy"},
            },
        })

    return network, python_code
