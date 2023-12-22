import os
import copy
from sisyphus import *


def add_loss_to_layer(network, name, loss, loss_opts=None, target=None, **kwargs):
    assert loss is not None
    network[name]["loss"] = loss
    if loss_opts:
        network[name]["loss_opts"] = loss_opts
    if target is not None:
        network[name]["target"] = target
    return network


def add_gather_layer(network, name, from_layers, axis="F", position=0):
    network[name] = {
        "class": "gather",
        "from": from_layers,
        "axis": axis,
        "position": position,
    }
    return network, name


def add_compare_layer(network, name, from_layers, value=None, kind="not_equal"):
    network[name] = {"class": "compare", "from": from_layers, "kind": kind}
    if value is not None:
        network[name]["value"] = value
    return network, name


def add_length_layer(network, name, from_layers, axis="T"):
    network[name] = {"class": "length", "from": from_layers, "axis": axis}
    return network, name


def add_range_from_length_layer(network, name, from_layers):
    network[name] = {"class": "range_from_length", "from": from_layers}
    return network, name


def add_combine_layer(network, name, from_layers, kind="logical_and", sources=None):
    network[name] = {"class": "combine", "from": from_layers, "kind": kind}
    if sources is not None:
        network[name]["sources"] = sources
    return network, name


# note: no 'from' entry
def add_switch_layer(network, name, condition, true_from=1, false_from=0):
    network[name] = {
        "class": "switch",
        "condition": condition,
        "true_from": true_from,
        "false_from": false_from,
    }
    return network, name


def add_reduce_layer(network, name, from_layers, axis="T", mode="sum"):
    network[name] = {"class": "reduce", "from": from_layers, "axis": axis, "mode": mode}
    return network, name


def add_slice_layer(network, name, from_layers, axis="F", start=None, end=None, step=None):
    network[name] = {
        "class": "slice",
        "from": from_layers,
        "axis": axis,
        "slice_start": start,
        "slice_end": end,
        "slice_step": step,
    }
    return network, name


def add_shift_layer(
    network, name, from_layers, axis="T", amount=1, pad=True, pad_value=None, **kwargs
):
    network[name] = {
        "class": "shift_axis",
        "from": from_layers,
        "axis": axis,
        "amount": amount,
        "pad": pad,
        "pad_value": pad_value,
    }
    if pad_value is not None:  # default 0 (modified RETURNN)
        network[name]["pad_value"] = pad_value
    if kwargs.get("adjust_size", None) is not None:
        network[name]["adjust_size_info"] = kwargs.get("adjust_size", None)

    return network, name


# Note: RETURNN source(i, auto_convert=True, enforce_batch_major=False, as_data=False)
def add_eval_layer(network, name, from_layers, eval_str, **kwargs):
    network[name] = {"class": "eval", "from": from_layers, "eval": eval_str}
    if kwargs.get("loss", None) is not None:
        network = add_loss_to_layer(network, name, **kwargs)
    if kwargs.get("initial", None) is not None:
        network[name]["initial_output"] = kwargs.get("initial", None)
    if kwargs.get("n_out", None) is not None:
        network[name]["n_out"] = kwargs.get("n_out", None)
    if kwargs.get("out_type", None) is not None:
        network[name]["out_type"] = kwargs.get("out_type", None)
    if kwargs.get("is_output", False):
        network[name]["is_output_layer"] = True
    return network, name


# masked computation
def add_mask_layer(network, name, from_layers, mask, unit={"class": "copy"}, **kwargs):
    network[name] = {
        "class": "masked_computation",
        "from": from_layers,
        "mask": mask,
        "unit": unit,
    }
    # more likely to be used in training where input is already masked elsewhere: directly use
    if kwargs.get("masked_from", None) is not None:
        network[name]["masked_from"] = kwargs.get("masked_from", None)
    # heuristics likely not needed anymore, use pad layer to achieve the same
    if kwargs.get("initial", None) is not None:
        network[name]["unit"]["initial_output"] = kwargs.get("initial", None)
    if kwargs.get("keep_last_for_prev", False):
        network[name]["keep_last_for_prev"] = True
    if kwargs.get("is_output", False):
        network[name]["is_output_layer"] = True
    return network, name


def add_reinterpret_data_layer(network, name, fromList, size_base=None, **kwargs):
    network[name] = { 
        "class": "reinterpret_data",
        "from": fromList
    }
    if kwargs.get('loss', None) is not None:
        network = add_loss_to_layer(network, name, **kwargs)
    if size_base is not None:
        network[name]['size_base'] = size_base
    if kwargs.get('enforce_batch_major', False):
        network[name]['enforce_batch_major'] = True
    if kwargs.get('enforce_time_major', False):
        network[name]['enforce_time_major'] = True
    if kwargs.get('set_sparse', None) is not None:
        network[name]['set_sparse'] = kwargs.get('set_sparse', None)
    if kwargs.get('set_sparse_dim', None) is not None:
        network[name]['set_sparse_dim'] = kwargs.get('set_sparse_dim', None)
    if kwargs.get('is_output', False):
        network[name]['is_output_layer'] = True
    return network, name
