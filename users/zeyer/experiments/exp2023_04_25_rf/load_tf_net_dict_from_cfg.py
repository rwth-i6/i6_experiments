from __future__ import annotations
from typing import Dict, Any


def load_net_dict_from_cfg(cfg_filename: str, *, output_probs_output_layer: bool = False) -> Dict[str, Any]:
    from returnn.config import Config

    cfg = Config()
    cfg.load_file(cfg_filename)
    net_dict = cfg.typed_dict["network"]
    assert isinstance(net_dict, dict)

    # TODO potentially patch the net dict a bit, removing stuff like specaug, simplifying...

    if "source" in net_dict and net_dict["source"]["class"] == "eval":
        net_dict["source"]["class"] = "copy"
        del net_dict["source"]["eval"]

    if output_probs_output_layer:
        rec_layer = net_dict["output"]
        assert rec_layer["class"] == "rec"
        subnet = rec_layer["unit"]
        assert isinstance(subnet, dict)
        subnet["output_prob"]["is_output_layer"] = True

    return net_dict
