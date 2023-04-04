from typing import Dict, Union, List, Optional


def add_softmax_output(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    target: str = "classes",
    scale: float = 1.0,
    l2: Optional[float] = None,
    dropout: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    focal_loss_factor: Optional[float] = None,
    initializer: Optional[str] = None,
    reuse_from_name: Optional[str] = None,
):

    loss_opts = {}
    if label_smoothing:
        loss_opts["label_smoothing"] = label_smoothing
    if focal_loss_factor:
        loss_opts["focal_loss_factor"] = focal_loss_factor

    network[name] = {
        "class": "softmax",
        "from": from_list,
        "n_out": num_outputs,
        "loss": "ce",
        "target": target,
    }

    if scale != 1.0:
        network[name]["loss_scale"] = scale
    if l2 is not None:
        network[name]["L2"] = l2
    if dropout is not None:
        network[name]["dropout"] = dropout
    if loss_opts:
        network[name]["loss_opts"] = loss_opts
    if initializer:
        network[name]["forward_weights_init"] = initializer
        network[name]["bias_init"] = initializer
    if reuse_from_name:
        network[name]["reuse_params"] = reuse_from_name

    return name
