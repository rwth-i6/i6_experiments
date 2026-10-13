"""
Config code that pins a torch distributed rank to the CPUs local to its GPU.

Bench-only probe (no RETURNN / sisyphus / torchrun code pins anything),
appended to a RETURNN config via ``extra_config_code``.
"""

from __future__ import annotations

__all__ = ["numa_pin_config_code", "socket_slice_pin_config_code"]


def numa_pin_config_code() -> str:
    """
    :return: python for a RETURNN config,
        pins the process (and the workers it forks) to the CPU affinity of GPU ``LOCAL_RANK``
        as ``nvidia-smi topo -m`` reports it (PCI order, CUDA_DEVICE_ORDER makes the CUDA index follow it).
        Everything lives in a function and is deleted again:
        the config namespace gets pickled for the dataset workers.
    """
    return (
        "def _numa_pin():\n"
        "    import os, re, subprocess\n"
        "    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'\n"
        "    lr = int(os.environ['LOCAL_RANK'])\n"
        "    rows = [l.split() for l in subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True).splitlines()]\n"
        "    row = [r for r in rows if r and r[0] == f'GPU{lr}']\n"
        "    assert len(row) == 1, f'GPU{lr} not in nvidia-smi topo: {rows[:8]}'\n"
        "    rng = re.compile(r'^\\d+(-\\d+)?(,\\d+(-\\d+)?)*$')\n"
        "    cl = next(t for t in row[0][1:] if rng.match(t))\n"
        "    cpus = set()\n"
        "    for part in cl.split(','):\n"
        "        a, _, b = part.partition('-')\n"
        "        cpus.update(range(int(a), int(b or a) + 1))\n"
        "    allowed = os.sched_getaffinity(0)\n"
        "    cpus &= allowed\n"
        "    assert cpus, f'no allowed cpus for GPU{lr}: {cl} vs {sorted(allowed)}'\n"
        "    os.sched_setaffinity(0, cpus)\n"
        "    print(f'NUMA pin: local rank {lr} cpus {cl} -> {len(cpus)} of {len(allowed)}')\n"
        "\n"
        "\n"
        "_numa_pin()\n"
        "del _numa_pin\n"
    )


def socket_slice_pin_config_code() -> str:
    """
    :return: python for a RETURNN config, the CPU selection of returnn#1951 (``select_gpu_local_cpus``):
        the GPU's NUMA node when it covers the rank's share of the allowed CPUs (allowed / local ranks),
        else its socket; ranks whose GPUs share that set get contiguous disjoint slices when each slice
        still covers the share. GPU topology from ``nvidia-smi topo -m`` as in :func:`numa_pin_config_code`.
    """
    return r'''
def _socket_slice_pin():
    import os, re, subprocess
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    lr = int(os.environ['LOCAL_RANK'])
    n = int(os.environ['LOCAL_WORLD_SIZE'])
    rows = [l.split() for l in subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True).splitlines()]
    rng = re.compile(r'^\d+(-\d+)?(,\d+(-\d+)?)*$')

    def cpus_of(i):
        row = [r for r in rows if r and r[0] == f'GPU{i}']
        assert len(row) == 1, f'GPU{i} not in nvidia-smi topo: {rows[:8]}'
        cl = next(t for t in row[0][1:] if rng.match(t))
        cpus = set()
        for part in cl.split(','):
            a, _, b = part.partition('-')
            cpus.update(range(int(a), int(b or a) + 1))
        return cpus

    def socket(c):
        with open(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id') as f:
            return int(f.read())

    allowed = os.sched_getaffinity(0)
    share = max(1, len(allowed) // n)
    local = [cpus_of(i) for i in range(n)]
    for widen in (lambda s: s & allowed, lambda s: {c for c in allowed if socket(c) in {socket(x) for x in s}}):
        cand = [widen(s) for s in local]
        cpus = cand[lr]
        if len(cpus) < share:
            continue
        peers = [i for i, o in enumerate(cand) if o == cpus]
        if len(cpus) >= share * len(peers):
            cs = sorted(cpus)
            k = len(cs) // len(peers)
            idx = peers.index(lr)
            cpus = set(cs[idx * k:(idx + 1) * k if idx + 1 < len(peers) else None])
        break
    else:
        print(f'socket-slice pin: local rank {lr} nothing pinned, {len(allowed)} allowed')
        return
    os.sched_setaffinity(0, cpus)
    print(f'socket-slice pin: local rank {lr} -> {len(cpus)} of {len(allowed)}: {sorted(cpus)}')


_socket_slice_pin()
del _socket_slice_pin
'''
