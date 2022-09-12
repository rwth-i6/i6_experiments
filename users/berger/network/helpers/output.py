from typing import Dict, Union, List, Optional


def add_softmax_output(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    target: str = "classes",
    l2: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    focal_loss: Optional[float] = None,
):

    loss_opts = {}
    if label_smoothing:
        loss_opts["label_smoothing"] = label_smoothing
    if focal_loss:
        loss_opts["focal_loss_factor"] = focal_loss

    network[name] = {
        "class": "softmax",
        "from": from_list,
        "n_out": num_outputs,
        "loss": "ce",
        "target": target,
    }

    if l2 is not None:
        network[name]["L2"] = l2
    if loss_opts:
        network[name]["loss_opts"] = loss_opts

    return name
