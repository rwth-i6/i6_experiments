from functools import reduce
from i6_experiments.users.berger.network.models.lstm_lm import make_lstm_lm_recog_model
from i6_experiments.users.berger.util import skip_layer
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from typing import Dict, List, Optional, Tuple, Union
from i6_experiments.users.berger.network.helpers.blstm import add_blstm_stack
from i6_experiments.users.berger.network.helpers.ctc_loss import (
    CtcLossType,
    add_ctc_output_layer,
)
from i6_experiments.users.berger.network.helpers.specaug import (
    add_specaug_layer,
    get_specaug_funcs,
)


def make_blstm_fullsum_ctc_dual_output_model(
    num_outputs: int,
    from_0: Union[str, List[str]],
    from_1: Union[str, List[str]],
    target_key_0: str,
    target_key_1: str,
    from_mix: Optional[Union[str, List[str]]] = None,
    specaug_01_args: Dict = {},
    specaug_mix_args: Optional[Dict] = None,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    output_args: Dict = {},
) -> Tuple[Dict, List[str]]:

    network = {}
    python_code = get_specaug_funcs()

    use_mix = bool(from_mix)

    # Specaug
    from_0 = add_specaug_layer(
        network, name="specaug_0", from_list=from_0, **specaug_01_args
    )
    from_1 = add_specaug_layer(
        network, name="specaug_1", from_list=from_1, **specaug_01_args
    )
    if use_mix:
        specaug_mix_args = specaug_mix_args or specaug_01_args
        from_mix = add_specaug_layer(
            network, name="specaug_mix", from_list=from_mix, **specaug_mix_args
        )

    # Encoders of separated audio

    # Assert that pooling is compatible, i.e. time axes have the same length
    if use_mix:
        if blstm_01_args.get("num_layers", 1) > 0:
            total_01_pool = reduce(
                lambda x, y: x * y, blstm_01_args.get("max_pool", []), 1
            )
        else:
            total_01_pool = 1

        if blstm_mix_args.get("num_layers", 1) > 0:
            total_mix_pool = reduce(
                lambda x, y: x * y, blstm_mix_args.get("max_pool", []), 1
            )
        else:
            total_mix_pool = 1

        assert total_01_pool == total_mix_pool

    from_0, _ = add_blstm_stack(
        network,
        from_list=from_0,
        name="lstm_0",
        **blstm_01_args,
    )
    from_1, _ = add_blstm_stack(
        network,
        from_list=from_1,
        name="lstm_1",
        reuse_from_name="lstm_0",
        **blstm_01_args,
    )
    from_mix, _ = add_blstm_stack(
        network,
        from_list=from_mix,
        name="lstm_mix",
        **blstm_mix_args,
    )

    # Joining

    if use_mix:
        from_0 += from_mix
        from_1 += from_mix

        from_0, _ = add_blstm_stack(
            network,
            from_list=from_0,
            name="lstm_0+mix",
            **blstm_01_mix_args,
        )

        from_1, _ = add_blstm_stack(
            network,
            from_list=from_1,
            name="lstm_1+mix",
            reuse_from_name="lstm_0+mix",
            **blstm_01_mix_args,
        )

    add_ctc_output_layer(
        CtcLossType.ReturnnFastBW,
        network=network,
        from_list=from_0,
        num_outputs=num_outputs,
        target_key=target_key_0,
        name="output_0",
        **output_args,
    )
    add_ctc_output_layer(
        CtcLossType.ReturnnFastBW,
        network=network,
        from_list=from_1,
        num_outputs=num_outputs,
        target_key=target_key_1,
        name="output_1",
        reuse_from_name="output_0",
        **output_args,
    )

    return network, python_code


def make_blstm_fullsum_ctc_dual_output_recog_model(
    num_outputs: int,
    from_0: Union[str, List[str]],
    from_1: Union[str, List[str]],
    target_key_0: str,
    target_key_1: str,
    from_mix: Optional[Union[str, List[str]]] = None,
    blstm_01_args: Dict = {},
    blstm_mix_args: Dict = {},
    blstm_01_mix_args: Dict = {},
    lm_path: Optional[tk.Path] = None,
    lm_scale: float = 0.0,
    lm_args: Dict = {},
    blank_penalty: float = 0.0,
    prior_path: Optional[tk.Path] = None,
    prior_scale: float = 0.0,
    output_args: Dict = {},
) -> Tuple[Dict, List[str]]:

    network = {}

    blank_index = output_args.get("blank_index", num_outputs - 1)

    if prior_scale:
        assert prior_path
        python_code = [
            DelayedFormat(
                'def get_prior_vector():\n    return np.loadtxt("{}", dtype=np.float32)\n',
                prior_path,
            )
        ]
    else:
        python_code = []

    # Basic network

    network, _ = make_blstm_fullsum_ctc_dual_output_model(
        num_outputs=num_outputs,
        from_0=from_0,
        from_1=from_1,
        target_key_0=target_key_0,
        target_key_1=target_key_1,
        from_mix=from_mix,
        blstm_01_args=blstm_01_args,
        blstm_mix_args=blstm_mix_args,
        blstm_01_mix_args=blstm_01_mix_args,
        output_args=output_args,
    )

    skip_layer(network, "specaug_0")
    skip_layer(network, "specaug_1")
    skip_layer(network, "specaug_mix")

    for index in [0, 1]:
        out_name = f"output_{index}"
        network[out_name].update(
            {
                "class": "linear",
                "activation": "log_softmax",
            }
        )
        network.pop(f"{out_name}_ctc_loss", None)
        network.pop(f"{out_name}_apply_loss", None)

        # Blank penalty
        if blank_penalty:
            network[f"{out_name}_blank_pen"] = {
                "class": "eval",
                "from": out_name,
                "eval": DelayedFormat(
                    "source(0) - tf.expand_dims(tf.one_hot([{}], {}, on_value={}, dtype=tf.float32), axis=0)",
                    blank_index,
                    num_outputs,
                    blank_penalty,
                ),
            }
            out_name = f"{out_name}_blank_pen"

        # Prior
        if prior_scale:
            network[f"{out_name}_prior"] = {
                "class": "eval",
                "from": out_name,
                "eval": f'source(0) - {prior_scale} * self.network.get_config().typed_value("get_prior_vector")()',
            }

        # Beam search
        network[f"beam_search_{index}"] = {
            "class": "rec",
            "from": out_name,
            "unit": {
                "output": {
                    "class": "choice",
                    "from": "data:source",
                    "input_type": "log_prob",
                    "target": "bpe_b",
                    "beam_size": 16,
                    "explicit_search_source": "prev:output",
                    "initial_output": blank_index,
                }
            },
        }

        # LM
        if lm_scale:
            assert lm_path

            network[f"beam_search_{index}"]["unit"].update(
                {
                    "mask_non_blank": {
                        "class": "compare",
                        "from": "output",
                        "value": blank_index,
                        "kind": "not_equal",
                        "initial_output": True,
                    },
                    "prev_output_reinterpret": {
                        "class": "reinterpret_data",
                        "from": "prev:output",
                        "increase_sparse_dim": -1,
                    },
                    "lm_masked": {
                        "class": "masked_computation",
                        "from": "prev_output_reinterpret",
                        "mask": "prev:mask_non_blank",
                        "unit": {
                            "class": "subnetwork",
                            "load_on_init": DelayedFormat("{}.index", lm_path),
                            "subnetwork": make_lstm_lm_recog_model(
                                num_outputs=num_outputs - 1, **lm_args
                            ),
                        },
                    },
                    "lm_padded": {
                        "class": "pad",
                        "from": "lm_masked",
                        "axes": "f",
                        "padding": (0, 1),
                        "value": 0,
                        "mode": "constant",
                    },
                    "combined_scores": {
                        "class": "eval",
                        "from": ["data:source", "lm_padded"],
                        "eval": f"source(0) + {lm_scale} * source(1)",
                    },
                }
            )
            network[f"beam_search_{index}"]["unit"]["output"][
                "from"
            ] = "combined_scores"

        # CTC decoding
        network[f"ctc_decode_{index}"] = {
            "class": "subnetwork",
            "is_output_layer": True,
            "target": target_key_0 if index == 0 else target_key_1,
            "subnetwork": {
                "decision": {
                    "class": "decide",
                    "from": f"base:beam_search_{index}",
                },
                "decision_shifted": {
                    "class": "shift_axis",
                    "from": "decision",
                    "axis": "T",
                    "amount": 1,
                    "pad_value": -1,
                    "adjust_size_info": False,
                },
                "mask_unique": {
                    "class": "compare",
                    "from": ["decision", "decision_shifted"],
                    "kind": "not_equal",
                },
                "mask_non_blank": {
                    "class": "compare",
                    "from": "decision",
                    "kind": "not_equal",
                    "value": blank_index,
                },
                "mask_label": {
                    "class": "combine",
                    "from": ["mask_unique", "mask_non_blank"],
                    "kind": "logical_and",
                },
                "decision_unique_labels": {
                    "class": "masked_computation",
                    "from": "decision",
                    "mask": "mask_label",
                    "unit": {"class": "copy"},
                },
                "output": {
                    "class": "reinterpret_data",
                    "from": "decision_unique_labels",
                    "increase_sparse_dim": -1,
                    "target": target_key_0 if index == 0 else target_key_1,
                    "loss": "edit_distance",
                },
            },
        }

    return network, python_code
