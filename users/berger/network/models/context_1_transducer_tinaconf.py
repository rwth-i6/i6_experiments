from typing import Tuple, List, Dict
from i6_core.rasr.config import RasrConfig
from i6_experiments.users.berger.network.helpers.rnnt_loss import (
    add_rnnt_loss,
    add_rnnt_loss_compressed,
)

from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack

from i6_experiments.users.raissi.setups.common import encoder
from i6_experiments.users.berger.network.helpers.conformer_wei import (
    add_conformer_stack,
    add_initial_conv,
)
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.loss_boost import (
    add_loss_boost,
    loss_boost_func_v1,
    loss_boost_func_v2,
)
import i6_experiments.users.berger.network.helpers.label_context as label_context
from i6_experiments.users.berger.network.helpers.specaug import get_specaug_funcs, add_specaug_layer


def make_context_1_conformer_transducer(
    num_inputs: int,
    num_outputs: int,
    blank_index: int = 0,
    loss_boost_scale: float = 5.0,
    intermediate_loss: float = 0.3,
    specaug_args: Dict = {},
    decoder_args: Dict = {},
    output_args: Dict = {},
    loss_boost_v2: bool = True,
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    network.update(
        encoder.get_best_conformer_network(
            size=512,
            num_classes=num_outputs,
            num_input_feature=num_inputs,
            time_tag_name=None,
            upsample_by_transposed_conv=False,
            chunking="400:200",
            label_smoothing=0.0,
            additional_args={
                "feature_stacking": False,
                "reduction_factor": (1, 4),
                "use_spec_augment": False,
            },
        ).network
    )

    from_list = add_specaug_layer(network, from_list="data", **specaug_args)
    python_code += get_specaug_funcs()

    network["source0"]["from"] = from_list

    network["encoder-output"] = {
        "class": "reinterpret_data",
        "from": "encoder",
        "enforce_time_major": True,
        "size_base": "data:classes",
    }

    # if intermediate_loss != 0:
    #     add_softmax_output(
    #         network,
    #         from_list="enc_006",
    #         name="encoder_output_006",
    #         num_outputs=num_outputs,
    #         target="classes",
    #         scale=intermediate_loss,
    #         focal_loss_factor=1.0,
    #     )

    # add_softmax_output(
    #     network,
    #     from_list="encoder",
    #     name="encoder_output",
    #     num_outputs=num_outputs,
    #     target="classes",
    #     focal_loss_factor=1.0,
    # )

    base_labels = "data:classes"
    context_labels, mask_non_blank = label_context.add_context_label_sequence_blank(
        network,
        base_labels=base_labels,
        blank_index=blank_index,
    )

    joint_output, decoder_unit = label_context.add_context_1_decoder(
        network,
        num_outputs=num_outputs,
        context_labels=context_labels,
        mask_non_blank=mask_non_blank,
        encoder="encoder-output",
        output_args=output_args,
        **decoder_args,
    )

    if loss_boost_scale:
        add_loss_boost(
            decoder_unit,
            boost_positions_mask=f"base:{mask_non_blank}",
            scale=loss_boost_scale,
            v2=loss_boost_v2,
        )
        if loss_boost_v2:
            python_code.append(loss_boost_func_v2)
        else:
            python_code.append(loss_boost_func_v1)

    return network, python_code


def make_context_1_conformer_transducer_recog(
    num_inputs: int,
    num_outputs: int,
    decoder_args: Dict = {},
) -> Tuple[Dict, List]:
    network = {}
    python_code = []

    network.update(
        encoder.get_best_conformer_network(
            size=512,
            num_classes=num_outputs,
            num_input_feature=num_inputs,
            time_tag_name=None,
            upsample_by_transposed_conv=False,
            chunking="400:200",
            label_smoothing=0.0,
            additional_args={
                "feature_stacking": False,
                "reduction_factor": (1, 4),
                "use_spec_augment": False,
            },
        ).network
    )

    network["encoder-output"] = {
        "class": "copy",
        "from": "encoder",
    }

    label_context.add_context_1_decoder_recog(
        network,
        num_outputs=num_outputs,
        encoder="encoder-output",
        **decoder_args,
    )

    return network, python_code
