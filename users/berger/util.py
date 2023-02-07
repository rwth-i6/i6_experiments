from typing import List, Union


def skip_layer(network: dict, layer_name: str) -> None:
    """Removes a layer from the network and connects its outputs to its inputs directly instead."""
    if not isinstance(network, dict):
        return

    layer_dict = network.pop(layer_name, None)
    if not layer_dict:
        return

    layer_from = layer_dict.get("from", "data")

    change_source_name(network, orig_name=layer_name, new_name=layer_from)


def change_source_name(network: dict, orig_name: str, new_name: Union[str, List[str]]):
    """Goes through the network and changes all appearances of orig_name in fromLists to new_name."""
    if not isinstance(network, dict):
        return

    if isinstance(new_name, str):
        new_name = [new_name]

    for x, attributes in network.items():
        if not isinstance(attributes, dict):
            continue
        if "from" in attributes:
            from_list = attributes["from"]
            if isinstance(from_list, str) and from_list == orig_name:
                attributes["from"] = new_name
            elif isinstance(from_list, list) and orig_name in from_list:
                index = from_list.index(orig_name)
                attributes["from"] = (
                    from_list[:index] + new_name + from_list[index + 1 :]
                )

        if "subnetwork" in attributes:
            change_source_name(
                attributes["subnetwork"],
                "base:" + orig_name,
                [f"base:{name}" for name in new_name],
            )

        if "unit" in attributes:
            change_source_name(
                attributes["unit"],
                "base:" + orig_name,
                [f"base:{name}" for name in new_name],
            )


def recursive_update(orig_dict: dict, update: dict):
    """Recursively updates dict and sub-dicts."""

    for k, v in update.items():
        if isinstance(v, dict):
            orig_dict[k] = recursive_update(orig_dict.get(k, {}), v)
        else:
            orig_dict[k] = v

    return orig_dict
