"""
Report on GPUs (free/used) in the Slurm cluster
"""

import sys
import os
import re
from functools import reduce
from subprocess import check_output
from collections import defaultdict
from typing import TypeVar, Dict


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


def main():
    nodes_info = parse_scontrol_show_node()

    count_args = ["gres/gpu"]
    # by (arg, partition)
    total = defaultdict(int)
    alloc = defaultdict(int)
    down = defaultdict(int)
    for _, node_info in nodes_info.items():
        for partition in node_info["Partitions"].split(","):
            state, *state_flags = node_info["State"].split("+")
            cfg_tres = parse_tres(node_info["CfgTRES"])
            alloc_tres = parse_tres(node_info.get("AllocTRES", ""))
            for count_arg in count_args:
                if count_arg not in cfg_tres:
                    continue
                node_total_count = cfg_tres[count_arg]
                node_alloc_count = alloc_tres.get(count_arg, 0)
                key = (count_arg, partition)
                if state in {"ALLOCATED", "IDLE", "MIXED"}:
                    total[key] += node_total_count
                    alloc[key] += node_alloc_count
                else:
                    down[key] += node_total_count

    for key in total:
        total_ = total[key]
        alloc_ = alloc[key]
        down_ = down[key]
        print(f"Count {key}: {alloc_}/{total_} used, {total_ - alloc_}/{total_} free, {down_} down")


def parse_tres(tres_str: str) -> Dict[str, int]:
    res = {}
    if not tres_str:
        return res
    for part in tres_str.split(","):
        key, value = part.split("=")
        factor = 1
        if value.endswith("M"):
            value = value[:-1]
            factor = 1024 * 1024
        elif value.endswith("G"):
            value = value[:-1]
            factor = 1024 * 1024 * 1024
        res[key] = (float(value) if "." in value else int(value)) * factor
    return res


def parse_scontrol_show_node() -> Dict[str, Dict[str, str]]:
    nodes_res = {}
    out_lines = check_output(["scontrol", "-o", "show", "node"]).decode("utf-8").splitlines()
    # example out, with "-o" one line per node::
    # NodeName=n23g0007 CoresPerSocket=48  CPUAlloc=0 CPUEfctv=96 CPUTot=96 ...
    # ... CfgTRES=cpu=96,mem=499800M,billing=96,gres/gpu=4,gres/gpu:hopper=4 ...
    # ... Reason=Not responding [slurmadm@2024-12-05T16:14:59] Comment=BadPerformance. Testing.
    # Without "-o", full node example:
    # NodeName=n23g0002 Arch=x86_64 CoresPerSocket=12
    #    CPUAlloc=47 CPUEfctv=96 CPUTot=96 CPULoad=127.44
    #    AvailableFeatures=sapphire,spr8468,hostok,hpcwork,Rocky8
    #    ActiveFeatures=sapphire,spr8468,hostok,hpcwork,Rocky8
    #    Gres=gpu:hopper:4(S:0-7)
    #    NodeAddr=n23g0002 NodeHostName=n23g0002 Version=22.05.4
    #    OS=Linux 4.18.0-553.27.1.el8_10.x86_64 #1 SMP Wed Nov 6 14:29:02 UTC 2024
    #    RealMemory=499800 AllocMem=211744 FreeMem=468883 Sockets=8 Boards=1
    #    State=MIXED ThreadsPerCore=1 TmpDisk=0 Weight=150933 Owner=N/A MCS_label=N/A
    #    Partitions=c23g,c23g_low
    #    BootTime=2024-12-08T16:28:22 SlurmdStartTime=2024-12-08T16:28:43
    #    LastBusyTime=2024-12-08T21:07:28
    #    CfgTRES=cpu=96,mem=499800M,billing=96,gres/gpu=4,gres/gpu:hopper=4
    #    AllocTRES=cpu=47,mem=211744M,gres/gpu=4,gres/gpu:hopper=4
    #    CapWatts=n/a
    #    CurrentWatts=528 AveWatts=319
    #    ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
    # Another example of a node in bad state:
    # NodeName=n23g0003 CoresPerSocket=48
    #    CPUAlloc=0 CPUEfctv=96 CPUTot=96 CPULoad=N/A
    #    AvailableFeatures=sapphire,spr8468,hostok,hpcwork,Rocky8
    #    ActiveFeatures=sapphire,spr8468,hostok,hpcwork,Rocky8
    #    Gres=gpu:hopper:4
    #    NodeAddr=n23g0003 NodeHostName=n23g0003
    #    RealMemory=499800 AllocMem=0 FreeMem=N/A Sockets=2 Boards=1
    #    State=DOWN+DRAIN+NOT_RESPONDING ThreadsPerCore=1 TmpDisk=0 Weight=150933 Owner=N/A MCS_label=N/A
    #    Partitions=c23g,c23g_low
    #    BootTime=None SlurmdStartTime=None
    #    LastBusyTime=2024-12-08T03:38:34
    #    CfgTRES=cpu=96,mem=499800M,billing=96,gres/gpu=4,gres/gpu:hopper=4
    #    AllocTRES=
    #    CapWatts=n/a
    #    CurrentWatts=0 AveWatts=0
    #    ExtSensorsJoules=n/s ExtSensorsWatts=0 ExtSensorsTemp=n/s
    #    Reason=gets too hot [mw445520@2024-12-03T17:06:40]

    for line in out_lines:
        node_res = {}
        # Parse the whole line
        for part in re.split(" +(?=\\S+=)", line):
            m = re.match("^([A-Za-z_]+)=(.*)$", part)
            assert m, f"Failed to match part: {part}"
            key, value = m.groups()
            node_res[key] = value
        node_name = node_res["NodeName"]
        assert node_name not in nodes_res, f"Node {node_name} appears twice? {nodes_res}"
        nodes_res[node_name] = node_res
    return nodes_res


if __name__ == "__main__":
    main()
