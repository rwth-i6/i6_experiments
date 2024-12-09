"""
Report on GPUs (free/used) in the Slurm cluster
"""

import sys
import os
from functools import reduce
from typing import TypeVar


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.slurm_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


from .report_gpus import parse_scontrol_show_node, parse_tres


def main():
    nodes_info = parse_scontrol_show_node()

    count_args = ["gres/gpu"]
    for node_name, node_info in nodes_info.items():
        state, *state_flags = node_info["State"].split("+")
        if state not in {"ALLOCATED", "IDLE", "MIXED"}:
            continue
        if "RESERVED" in state_flags:
            continue
        cfg_tres = parse_tres(node_info["CfgTRES"])
        alloc_tres = parse_tres(node_info.get("AllocTRES", ""))
        has_free = []
        for count_arg in count_args:
            if count_arg not in cfg_tres:
                continue
            node_total_count = cfg_tres[count_arg]
            node_alloc_count = alloc_tres.get(count_arg, 0)
            if node_total_count > node_alloc_count:
                has_free.append(count_arg)
        if has_free:
            print(f"Node {node_name} ({node_info['Partitions']}) has free {','.join(count_args)}:")
            for k, v in cfg_tres.items():
                f = _identity
                if k == "mem":
                    f = human_bytes_size
                print(f"  {k}: {f(v)} total, {f(alloc_tres.get(k, 0))} used, {f(v - alloc_tres.get(k, 0))} free")
            print(f"  State: {node_info['State']}")


def _identity(x):
    return x


def human_size(n, factor=1000, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: for each of the units K, M, G, T
    :param float frac: when to go over to the next bigger unit
    :param int prec: how much decimals after the dot
    :return: human readable size, using K, M, G, T
    :rtype: str
    """
    postfixes = ["", "K", "M", "G", "T"]
    i = 0
    while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
        i += 1
    if i == 0:
        return str(n)
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


def human_bytes_size(n, factor=1024, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: see :func:`human_size`. 1024 by default for bytes
    :param float frac: see :func:`human_size`
    :param int prec: how much decimals after the dot
    :return: human readable byte size, using K, M, G, T, with "B" at the end
    :rtype: str
    """
    return human_size(n, factor=factor, frac=frac, prec=prec) + "B"


if __name__ == "__main__":
    main()
