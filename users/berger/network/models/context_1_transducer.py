from typing import Tuple, List, Dict, Optional
from i6_core.rasr.config import RasrConfig
from i6_experiments.users.berger.network.helpers.rnnt_loss import (
    add_rnnt_loss,
    add_rnnt_loss_compressed,
)

from i6_experiments.users.berger.network.helpers.specaug import (
    add_specaug_layer,
    get_specaug_funcs,
)
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.loss_boost import (
    add_loss_boost,
    loss_boost_func,
)
import i6_experiments.users.berger.network.helpers.label_context as label_context


def make_context_1_blstm_transducer_blank(
    num_outputs: int,
    blank_index: int = 0,
    loss_boost_scale: float = 5.0,
    encoder_loss: bool = False,
    specaug_args: Dict = {},
    blstm_args: Dict = {},
    decoder_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, List]:

    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, **specaug_args)
    python_code += get_specaug_funcs()

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {
        "class": "reinterpret_data",
        "from": from_list,
        "enforce_time_major": True,
        "size_base": "data:classes",
    }

    if encoder_loss:
        add_softmax_output(
            network,
            from_list="encoder",
            name="encoder_output",
            num_outputs=num_outputs,
            target="classes",
            focal_loss_factor=1.0,
        )

    base_labels = "data:classes"
    context_labels, mask_non_blank = label_context.add_context_label_sequence_blank(
        network,
        base_labels=base_labels,
        blank_index=blank_index,
    )

    joint_output, decoder_unit = label_context.add_context_1_decoder(
        network,
        context_labels=context_labels,
        mask_non_blank=mask_non_blank,
        encoder="encoder",
        **decoder_args,
    )

    add_softmax_output(
        decoder_unit,
        from_list=joint_output,
        name="output",
        num_outputs=num_outputs,
        target="classes",
        **output_args,
    )

    if loss_boost_scale:
        add_loss_boost(
            decoder_unit,
            boost_positions_mask=f"base:{mask_non_blank}",
            scale=loss_boost_scale,
        )
        python_code.append(loss_boost_func)

    return network, python_code


def make_context_1_blstm_transducer_noblank(
    num_outputs: int,
    nonword_labels: Optional[List[int]] = None,
    loss_boost_scale: float = 5.0,
    blstm_args: Dict = {},
    decoder_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, List]:

    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network)
    python_code += get_specaug_funcs()

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {"class": "copy", "from": from_list}

    base_labels = "data:classes"

    context_labels, mask_first_label = label_context.add_context_label_sequence_noblank(
        network,
        num_outputs=num_outputs,
        nonword_labels=nonword_labels,
        base_labels=base_labels,
    )

    joint_output, decoder_unit = label_context.add_context_1_decoder(
        network,
        context_labels=context_labels,
        mask_first_label=mask_first_label,
        encoder="encoder",
        **decoder_args,
    )

    add_softmax_output(
        decoder_unit,
        from_list=joint_output,
        name="output",
        num_outputs=num_outputs,
        target="classes",
        **output_args,
    )

    if loss_boost_scale > 0:
        decoder_unit["boost_positions"] = {
            "class": "copy",
            "from": f"base:{mask_first_label}",
        }
        add_loss_boost(
            decoder_unit,
            boost_positions_mask="boost_positions",
            scale=loss_boost_scale,
        )

    return network, python_code


def make_context_1_blstm_transducer_fullsum(
    num_outputs: int,
    blank_index: int = 0,
    compress_joint_input: bool = True,
    specaug_args: Dict = {},
    blstm_args: Dict = {},
    decoder_args: Dict = {},
) -> Tuple[Dict, List]:

    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, **specaug_args)
    python_code += get_specaug_funcs()

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {
        "class": "copy",
        "from": from_list,
    }

    base_labels = "data:classes"

    context_labels, _ = label_context.add_context_label_sequence_blank(
        network,
        base_labels=base_labels,
        blank_index=blank_index,
    )

    (
        joint_output,
        decoder_unit,
        decoder_python,
    ) = label_context.add_context_1_decoder_fullsum(
        network,
        context_labels=context_labels,
        encoder="encoder",
        compress_joint_input=compress_joint_input,
        **decoder_args,
    )
    python_code += decoder_python

    if compress_joint_input:
        python_code += add_rnnt_loss_compressed(
            decoder_unit,
            encoder="base:base:encoder",
            joint_output=joint_output,
            targets=f"base:base:{context_labels}",
            num_classes=num_outputs,
            blank_index=blank_index,
        )
    else:
        python_code += add_rnnt_loss(
            decoder_unit,
            encoder="base:base:encoder",
            joint_output=joint_output,
            targets=f"base:base:{context_labels}",
            num_classes=num_outputs,
            blank_index=blank_index,
        )

    return network, python_code


def make_context_1_blstm_transducer_recog(
    num_outputs: int,
    blstm_args: Dict = {},
    decoder_args: Dict = {},
) -> Tuple[Dict, List]:

    network = {}
    python_code = []

    from_list = ["data"]

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {"class": "copy", "from": from_list}

    joint_output, decoder_unit = label_context.add_context_1_decoder_recog(
        network,
        num_outputs=num_outputs,
        encoder="encoder",
        **decoder_args,
    )

    decoder_unit["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code


def get_viterbi_transducer_alignment_config(reduction_factor: int) -> RasrConfig:
    alignment_config = RasrConfig()
    alignment_config.neural_network_trainer["*"].force_single_state = True
    alignment_config.neural_network_trainer[
        "*"
    ].reduce_alignment_factor = reduction_factor
    alignment_config.neural_network_trainer["*"].peaky_alignment = True
    alignment_config.neural_network_trainer["*"].peak_position = 1.0

    return alignment_config


def pretrain_construction_algo(idx, net_dict):
    num_layers = idx + 1
    remaining_reduction = 1

    enc_layer_idx = 0
    while "fwd_lstm_%i" % (enc_layer_idx + 1) in net_dict:
        enc_layer_idx += 1
        if enc_layer_idx > num_layers:
            del net_dict["fwd_lstm_%i" % enc_layer_idx]
            del net_dict["bwd_lstm_%i" % enc_layer_idx]
        if enc_layer_idx >= num_layers and "max_pool_%i" % enc_layer_idx in net_dict:
            remaining_reduction *= net_dict["max_pool_%i" % enc_layer_idx]["pool_size"][
                0
            ]
            del net_dict["max_pool_%i" % enc_layer_idx]

    if num_layers <= enc_layer_idx:
        # only encoder
        fromList = ["fwd_lstm_%i" % num_layers, "bwd_lstm_%i" % num_layers]
        if remaining_reduction > 1:
            net_dict["max_pool_lstm_%i" % num_layers] = {
                "class": "pool",
                "mode": "max",
                "padding": "same",
                "pool_size": (remaining_reduction,),
                "from": fromList,
                "trainable": False,
            }
            fromList = ["max_pool_lstm_%i" % num_layers]

        net_dict["encoder"]["from"] = fromList
        # add encoder ce loss
        net_dict["encoder_output"] = {
            "class": "softmax",
            "from": "encoder",
            "loss": "ce",
        }

        # remove decoder
        del net_dict["output"]

    else:
        num_dec_layers = num_layers - enc_layer_idx

        dec_layer_idx = 0
        while "dec_ff_%i" % (dec_layer_idx + 1) in net_dict["output"]["unit"]:
            dec_layer_idx += 1
            if dec_layer_idx > num_dec_layers:
                del net_dict["output"]["unit"]["dec_ff_%i" % dec_layer_idx]

        if num_dec_layers <= dec_layer_idx:
            # partial decoder
            net_dict["output"]["unit"]["decoder"]["from"] = [
                "dec_ff_%i" % num_dec_layers
            ]
        else:  # full encoder and full decoder -> finished pre-training
            return None

    return net_dict
