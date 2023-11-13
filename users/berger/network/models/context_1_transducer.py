from typing import Tuple, List, Dict
from i6_core.rasr.config import RasrConfig
from i6_experiments.users.berger.network.helpers.rnnt_loss import (
    add_rnnt_loss,
    add_rnnt_loss_compressed,
)

from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack

from i6_experiments.users.berger.network.helpers.conformer_wei import (
    add_conformer_stack,
    add_initial_conv,
)
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.loss_boost import (
    add_loss_boost,
    loss_boost_func,
)
import i6_experiments.users.berger.network.helpers.label_context as label_context
from i6_experiments.users.berger.network.helpers.specaug import get_specaug_funcs, add_specaug_layer


def make_context_1_conformer_transducer(
    num_outputs: int,
    blank_index: int = 0,
    loss_boost_scale: float = 5.0,
    intermediate_loss: float = 0.3,
    specaug_args: Dict = {},
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    decoder_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, from_list=from_list, **specaug_args)
    python_code += get_specaug_funcs()

    from_list = add_initial_conv(network, from_list, **vgg_args)
    from_list, blocks = add_conformer_stack(network, from_list, **conformer_args)

    network["encoder"] = {
        "class": "reinterpret_data",
        "from": from_list,
        "enforce_time_major": True,
        "size_base": "data:classes",
    }

    if intermediate_loss != 0:
        mid_block = len(blocks) // 2
        add_softmax_output(
            network,
            from_list=blocks[mid_block - 1],
            name=f"encoder_output_{mid_block}",
            num_outputs=num_outputs,
            target="classes",
            scale=intermediate_loss,
            focal_loss_factor=1.0,
        )

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


def make_context_1_conformer_transducer_fullsum(
    num_outputs: int,
    blank_index: int = 0,
    compress_joint_input: bool = True,
    specaug_args: Dict = {},
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    decoder_args: Dict = {},
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, from_list=from_list, **specaug_args)
    python_code += get_specaug_funcs()

    from_list = add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = add_conformer_stack(network, from_list, **conformer_args)

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

    network["pred_labels_int32"] = {
        "class": "cast",
        "from": context_labels,
        "dtype": "int32",
    }
    context_labels = "pred_labels_int32"

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


def make_context_1_conformer_transducer_recog(
    num_outputs: int,
    vgg_args: Dict = {},
    conformer_args: Dict = {},
    decoder_args: Dict = {},
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_initial_conv(network, from_list, **vgg_args)
    from_list, _ = add_conformer_stack(network, from_list, **conformer_args)

    network["encoder"] = {"class": "copy", "from": from_list}

    # bn_layers = []
    # for layer_name, layer_desc in network.items():
    #     if layer_desc.get("class") == "batch_norm":
    #         bn_layers.append(layer_name)
    # for layer_name in bn_layers:
    #     skip_layer(network, layer_name)
    # if layer_desc.get("class") == "batch_norm":
    #    layer_desc["param_version"] = 0
    #    layer_desc.pop("delay_sample_update")

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


def make_context_1_blstm_transducer(
    num_outputs: int,
    blank_index: int = 0,
    loss_boost_scale: float = 5.0,
    specaug_args: Dict = {},
    blstm_args: Dict = {},
    decoder_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    from_list = ["data"]

    from_list = add_specaug_layer(network, from_list=from_list, **specaug_args)
    python_code += get_specaug_funcs()

    from_list, _ = add_blstm_stack(network, from_list, **blstm_args)

    network["encoder"] = {
        "class": "reinterpret_data",
        "from": from_list,
        "enforce_time_major": True,
        "size_base": "data:classes",
    }

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

    from_list = add_specaug_layer(network, from_list=from_list, **specaug_args)
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

    network["pred_labels_int32"] = {
        "class": "cast",
        "from": context_labels,
        "dtype": "int32",
    }
    context_labels = "pred_labels_int32"

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

    # bn_layers = []
    # for layer_name, layer_desc in network.items():
    #     if layer_desc.get("class") == "batch_norm":
    #         bn_layers.append(layer_name)
    # for layer_name in bn_layers:
    #     skip_layer(network, layer_name)
    # if layer_desc.get("class") == "batch_norm":
    #    layer_desc["param_version"] = 0
    #    layer_desc.pop("delay_sample_update")

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
    alignment_config.neural_network_trainer["*"].reduce_alignment_factor = reduction_factor
    alignment_config.neural_network_trainer["*"].peaky_alignment = True
    alignment_config.neural_network_trainer["*"].peak_position = 1.0

    return alignment_config
