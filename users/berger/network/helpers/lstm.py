from typing import Optional, Dict, Union, List
import copy


def add_lstm_layer(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    size: int = 500,
    max_pool: Optional[int] = None,
    dropout: Optional[float] = 0.1,
    l2: Optional[float] = 0.01,
    trainable: bool = True,
    initializer: Optional[str] = None,
) -> List[str]:

    network[name] = {
        "class": "rec",
        "unit": "nativelstm2",
        "from": from_list,
        "n_out": size,
        "direction": 1,
    }
    if dropout is not None:
        network[name]["dropout"] = dropout
    if l2 is not None:
        network[name]["L2"] = l2
    if not trainable:
        network[name]["trainable"] = False
    if initializer:
        network[name]["forward_weights_init"] = initializer
        network[name]["recurrent_weights_init"] = initializer
        network[name]["bias_init"] = initializer

    output = [name]

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


def add_lstm_stack(
    network: Dict,
    from_list: Union[str, List[str]],
    name: str = "lstm",
    num_layers: int = 6,
    max_pool: Optional[List[int]] = None,
    **kwargs,
) -> List[str]:

    for layer_idx in range(num_layers):
        pool = None
        if max_pool is not None and len(max_pool) > layer_idx:
            pool = max_pool[layer_idx]
        from_list = add_lstm_layer(
            network, f"{name}_{layer_idx + 1}", from_list, max_pool=pool, **kwargs
        )

    return from_list
