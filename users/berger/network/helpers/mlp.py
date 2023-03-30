from typing import Dict, Union, List, Optional


def add_feed_forward_stack(
    network: Dict,
    from_list: Union[str, List[str]],
    name: str = "ff",
    num_layers: int = 2,
    size: int = 1024,
    activation: str = "tanh",
    dropout: Optional[float] = 0.1,
    l2: Optional[float] = 0.01,
    reuse_from_name: Optional[str] = None,
    trainable: bool = True,
    initializer: Optional[str] = None,
) -> List[str]:

    for layer_idx in range(num_layers):
        layer_name = f"{name}_{layer_idx + 1}"
        network[layer_name] = {
            "class": "linear",
            "from": from_list,
            "n_out": size,
            "activation": activation,
        }
        if dropout is not None:
            network[layer_name]["dropout"] = dropout
        if l2 is not None:
            network[layer_name]["L2"] = l2
        if reuse_from_name is not None:
            network[layer_name]["reuse_params"] = f"{reuse_from_name}_{layer_idx + 1}"
        if not trainable:
            network[layer_name]["trainable"] = False
        if initializer:
            network[layer_name]["forward_weights_init"] = initializer

        from_list = [layer_name]

    return from_list
