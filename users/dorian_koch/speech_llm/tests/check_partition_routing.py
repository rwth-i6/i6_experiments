"""Verify settings.py routes jobs to partitions from cluster-agnostic requirements.

Extracts the real source (rather than reimplementing it) and exercises it on the rqmt shapes our
jobs actually produce. Guards three things that fail silently in production:
  * every submission carries the broken-node exclusion,
  * a capability requirement reaches a partition that provides it,
  * jobs never need to name a partition themselves.
"""

import textwrap

src = open("settings.py").read()
start = src.index("    #: What each GPU partition on THIS cluster actually offers")
end = src.index("    def engine():")
ns = {}
exec(textwrap.dedent(src[start:end]), ns)
check_engine_limits = ns["check_engine_limits"]
GPU_PARTITIONS = ns["GPU_PARTITIONS"]

# BROKEN_NODES is defined just above the block we sliced; pull it in too.
bn_start = src.index("    BROKEN_NODES = [")
exec(textwrap.dedent(src[bn_start : src.index("    def check_engine_limits")]), ns)
BROKEN_NODES = ns["BROKEN_NODES"]


def route(rqmt):
    """Return (partition, sbatch_args) for an rqmt, asserting the invariants that hold for ALL jobs."""
    out = check_engine_limits(dict(rqmt), None)
    args = out["sbatch_args"]
    assert "-x" in args, f"node exclusion missing -> {args}"
    excluded = args[args.index("-x") + 1].split(",")
    assert set(BROKEN_NODES) <= set(excluded), f"not all broken nodes excluded -> {args}"
    assert args.count("-p") == 1, f"expected exactly one partition -> {args}"
    # our own requirement keys must be consumed, never leaked to sbatch
    assert "gpu_mem_gb" not in out and "requires" not in out, out
    return args[args.index("-p") + 1], args


def expect(name, rqmt, partition):
    got, args = route(rqmt)
    assert got == partition, f"{name}: routed to {got}, expected {partition} -> {args}"
    print(f"[ok] {name:38s} -> {' '.join(args)}")


# --- automatic routing by walltime ---------------------------------------------------------------
expect("gpu 8h", {"gpu": 1, "time": 8}, "c25g")
expect("gpu 24h (over c25g 12h cap)", {"gpu": 1, "time": 24}, "c23g")
expect("cpu job", {"cpu": 4, "time": 2}, "c23ms")

# --- capability-driven routing -------------------------------------------------------------------
# The audio jobs (MoshiAnnotate, ChatterboxInference) declare this instead of naming c23g.
expect("gpu 4h + system_ffmpeg", {"gpu": 1, "time": 4, "requires": ["system_ffmpeg"]}, "c23g")
# A judge needing more than c25g's 80 GB per card must land on the bigger cards.
expect("gpu 8h + 90 GB/GPU", {"gpu": 1, "time": 8, "gpu_mem_gb": 90}, "c23g")
expect("gpu 8h + 70 GB/GPU (fits c25g)", {"gpu": 1, "time": 8, "gpu_mem_gb": 70}, "c25g")

# --- escape hatches still work -------------------------------------------------------------------
expect("raw -p pin (legacy)", {"gpu": 1, "time": 8, "sbatch_args": ["-p", "c23g"]}, "c23g")
out = check_engine_limits({"gpu": 1, "time": 8, "sbatch_args": ["-x", "n23g0001"]}, None)
assert out["sbatch_args"].count("-x") == 1, out["sbatch_args"]
print(f"[ok] {'job-supplied -x preserved':38s} -> {' '.join(out['sbatch_args'])}")

# --- unsatisfiable requirements must fail loudly, not silently mis-route -------------------------
# NB: an over-long `time` is NOT in this list -- check_engine_limits clamps walltime to 72h before
# routing, so it becomes satisfiable on c23g rather than unsatisfiable. Asserted just below.
for bad in ({"gpu": 1, "time": 8, "requires": ["nonexistent_feature"]},
            {"gpu": 1, "time": 8, "gpu_mem_gb": 500}):
    try:
        check_engine_limits(dict(bad), None)
    except ValueError:
        pass
    else:
        raise SystemExit(f"FAIL: unsatisfiable rqmt {bad} was silently routed")
print(f"[ok] {'unsatisfiable requirements rejected':38s} (2 cases)")
expect("gpu 100h (clamped to 72h)", {"gpu": 1, "time": 100}, "c23g")

# --- the capability table must be self-consistent ------------------------------------------------
for name, spec in GPU_PARTITIONS.items():
    assert spec["gpu_mem_gb"] > 0 and spec["max_time_h"] > 0, (name, spec)
    assert isinstance(spec["features"], set), (name, spec)
print(f"[ok] {'partition table well-formed':38s} ({', '.join(GPU_PARTITIONS)})")

print("\nALL CHECKS PASSED")
