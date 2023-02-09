from typing import Optional, Dict, Tuple, Union, List
import copy


def add_blstm_layer(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    size: int = 500,
    max_pool: Optional[int] = None,
    dropout: Optional[float] = 0.1,
    l2: Optional[float] = 0.01,
    trainable: bool = True,
    reuse_from_name: Optional[str] = None,
) -> List[str]:
    layer_dict = {
        "class": "rec",
        "unit": "nativelstm2",
        "from": from_list,
        "n_out": size,
    }
    if dropout is not None:
        layer_dict["dropout"] = dropout
    if l2 is not None:
        layer_dict["L2"] = l2
    if not trainable:
        layer_dict["trainable"] = False

    network[f"fwd_{name}"] = copy.deepcopy(layer_dict)
    network[f"fwd_{name}"]["direction"] = 1
    network[f"bwd_{name}"] = copy.deepcopy(layer_dict)
    network[f"bwd_{name}"]["direction"] = -1
    if reuse_from_name is not None:
        network[f"fwd_{name}"]["reuse_params"] = f"fwd_{reuse_from_name}"
        network[f"bwd_{name}"]["reuse_params"] = f"bwd_{reuse_from_name}"

    output = [f"fwd_{name}", f"bwd_{name}"]

    if max_pool is not None and max_pool > 1:
        network[f"max_pool_{name}"] = {
            "class": "pool",
            "from": output,
            "mode": "max",
            "padding": "same",
            "pool_size": (max_pool,),
        }
        output = [f"max_pool_{name}"]

    return output


def add_blstm_stack(
    network: Dict,
    from_list: Union[str, List[str]],
    name: str = "lstm",
    num_layers: int = 6,
    max_pool: Optional[List[int]] = None,
    reuse_from_name: Optional[str] = None,
    **kwargs,
) -> Tuple[List[str], List[List[str]]]:

    layers = []

    for layer_idx in range(num_layers):
        pool = None
        if max_pool is not None and len(max_pool) > layer_idx:
            pool = max_pool[layer_idx]
        if reuse_from_name is not None:
            layer_reuse_name = f"{reuse_from_name}_{layer_idx + 1}"
        else:
            layer_reuse_name = None
        from_list = add_blstm_layer(
            network,
            f"{name}_{layer_idx + 1}",
            from_list,
            max_pool=pool,
            reuse_from_name=layer_reuse_name,
            **kwargs,
        )
        layers.append(from_list)

    return from_list, layers
