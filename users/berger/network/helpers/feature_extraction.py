from typing import Any, Dict, List, Optional, Tuple, Union
from i6_experiments.users.berger.network.helpers.specaug import add_specaug_layer_v2
from returnn_common.asr import gt


def add_gt_feature_extraction(
    network: Dict[str, Any],
    from_list: Union[str, List[str]],
    sample_rate: int,
    name="gammatone",
    specaug_before_dct: bool = False,
    specaug_after_dct: bool = True,
    channels: Optional[int] = None,
    filterbank_size: Optional[int] = None,
    tempint_length: float = 0.025,
    tempint_shift: float = 0.01,
    max_freq: Optional[int] = None,
    padding: Optional[Tuple[int, int]] = None,
) -> Tuple[str, Union[str, List[str]]]:
    python_code = []

    channels = (
        channels
        or {
            8000: 40,
            16000: 50,
        }[sample_rate]
    )

    max_freq = (
        max_freq
        or {
            8000: 3800,
            16000: 7500,
        }[sample_rate]
    )

    filterbank_size = filterbank_size or sample_rate // 25

    gt_net = gt.get_net_dict_v1(
        num_channels=channels,
        sample_rate=sample_rate,
        gt_filterbank_size=filterbank_size,
        temporal_integration_size=int(sample_rate * tempint_length),
        temporal_integration_strides=int(sample_rate * tempint_shift),
        normalization="time",
        freq_max=max_freq,
    )

    if padding:
        gt_net["gammatone_filterbank"]["padding"] = "same"
        gt_net["preemphasis_padded"] = {
            "class": "pad",
            "from": "preemphasis",
            "axes": "T",
            "padding": padding,
        }
        gt_net["gammatone_filterbank"]["from"] = "preemphasis_padded"

    if specaug_before_dct:
        specaug_name, python_code = add_specaug_layer_v2(
            gt_net, from_list=gt_net["dct"]["from"]
        )
        gt_net["dct"]["from"] = specaug_name

    if specaug_after_dct:
        specaug_name, python_code = add_specaug_layer_v2(
            gt_net, from_list=gt_net["output"]["from"]
        )
        gt_net["output"]["from"] = specaug_name

    for layer_attr in gt_net.values():
        if layer_attr.get("from", "") == "data":
            layer_attr["from"] = "cast_input"

    gt_net["cast_input"] = {
        "class": "cast",
        "from": "data",
        "dtype": "float32",
    }

    network[name] = {
        "class": "subnetwork",
        "subnetwork": gt_net,
        "from": from_list,
        "trainable": False,
    }

    return name, python_code
