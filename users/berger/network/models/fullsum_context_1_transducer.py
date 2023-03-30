from typing import Tuple, List, Dict, Optional
from i6_core.rasr.config import RasrConfig

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


def make_context_1_blstm_transducer(
    num_outputs: int,
    context_transformation_func: Optional[str] = None,
    context_label_dim: Optional[int] = None,
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
            focal_loss=1.0,
        )

    base_labels = "data:classes"
    if context_transformation_func:
        assert context_label_dim
        network["transformed_classes"] = {
            "class": "eval",
            "from": base_labels,
            "eval": context_transformation_func,
            "out_type": {"dim": context_label_dim},
        }
        base_labels = "transformed_classes"

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


def make_context_1_blstm_transducer_recog(
    num_outputs: int,
    context_transformation_func: Optional[str] = None,
    context_label_dim: Optional[int] = None,
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
        context_transformation_func=context_transformation_func,
        context_label_dim=context_label_dim,
        **decoder_args,
    )

    decoder_unit["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    return network, python_code
