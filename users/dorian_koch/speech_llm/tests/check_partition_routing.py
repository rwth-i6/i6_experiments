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

# The sections below up to "routing by estimated start" assert the requirement / spread / preference rules
# on their own: no live SLURM probe and no manual pins, so they are deterministic on any machine.
ns["GPU_PARTITION_BY_START"] = False
ns["GPU_PARTITION_PINS"] = {}


class _FakeTask:
    """A task whose work path and DECLARED rqmt (what the job itself asks for) we control."""

    def __init__(self, p, rqmt=None):
        self._p = p
        self._rqmt = dict(rqmt or {})

    def path(self):
        return self._p

    def rqmt(self):
        return dict(self._rqmt)


def route(rqmt, task=None):
    """Return (partition, sbatch_args) for an rqmt, asserting the invariants that hold for ALL jobs."""
    out = check_engine_limits(dict(rqmt), task)
    args = out["sbatch_args"]
    assert "-x" in args, f"node exclusion missing -> {args}"
    excluded = args[args.index("-x") + 1].split(",")
    assert set(BROKEN_NODES) <= set(excluded), f"not all broken nodes excluded -> {args}"
    assert args.count("-p") == 1, f"expected exactly one partition -> {args}"
    # our own requirement keys must be consumed, never leaked to sbatch
    assert "gpu_mem_gb" not in out and "requires" not in out, out
    return args[args.index("-p") + 1], args


def expect(name, rqmt, partition, task=None):
    got, args = route(rqmt, task)
    assert got == partition, f"{name}: routed to {got}, expected {partition} -> {args}"
    print(f"[ok] {name:38s} -> {' '.join(args)}")


def expect_any(name, rqmt, allowed):
    """For jobs that fit SEVERAL partitions: assert only that the choice is a legal one.

    Pinning a specific partition here would be asserting the spread function's hash, which is an
    implementation detail -- and it is what made this check fail the moment the preference order was
    re-measured. What must hold is the requirement, not the destination.
    """
    got, args = route(rqmt)
    assert got in allowed, f"{name}: routed to {got}, expected one of {sorted(allowed)} -> {args}"
    print(f"[ok] {name:38s} -> {' '.join(args)}")


# --- automatic routing by walltime ---------------------------------------------------------------
# A job that fits both may land on either; spreading is asserted separately below.
expect_any("gpu 8h (fits both)", {"gpu": 1, "time": 8}, {"c23g", "c25g"})
expect_any("gpu 24h (exactly c25g's cap)", {"gpu": 1, "time": 24}, {"c23g", "c25g"})
expect("gpu 25h (over c25g's 24h cap)", {"gpu": 1, "time": 25}, "c23g")
expect("cpu job", {"cpu": 4, "time": 2}, "c23ms")

# --- capability-driven routing -------------------------------------------------------------------
# The audio jobs (MoshiAnnotate, ChatterboxInference) declare this instead of naming c23g.
expect("gpu 4h + system_ffmpeg", {"gpu": 1, "time": 4, "requires": ["system_ffmpeg"]}, "c23g")
# A judge needing more than c25g's 80 GB per card must land on the bigger cards.
expect("gpu 8h + 90 GB/GPU", {"gpu": 1, "time": 8, "gpu_mem_gb": 90}, "c23g")
expect_any("gpu 8h + 70 GB/GPU (fits both)", {"gpu": 1, "time": 8, "gpu_mem_gb": 70}, {"c23g", "c25g"})

# --- spreading: deterministic, and it really spreads ----------------------------------------------
# We cannot let SLURM pick: a comma partition list is rejected for both our accounts, so the choice
# is made per job in settings.py. Two properties matter and both fail silently.
select_gpu_partition = ns["select_gpu_partition"]


# (1) Deterministic -- the same job must not move between manager restarts. Python's salted hash()
#     would break this and nothing downstream would notice.
_k = "work/i6_experiments/.../SpeechFinetune.abc123"
_first = select_gpu_partition(time_h=8, spread_key=_k)
assert all(select_gpu_partition(time_h=8, spread_key=_k) == _first for _ in range(20)), (
    "partition choice is not deterministic for a fixed job -- a resubmission would silently move"
)
print(f"[ok] {'spread is deterministic':38s} -> {_k.split('/')[-1]} always {_first}")

# (2) It actually spreads. A 'spread' that always returns the preferred partition is the bug this
#     exists to prevent, and it looks identical from one job.
_seen = {select_gpu_partition(time_h=8, spread_key=f"job{i}") for i in range(200)}
assert _seen == {"c23g", "c25g"}, f"spreading only ever produced {_seen} -- it is not spreading"
print(f"[ok] {'spread covers both partitions':38s} -> {sorted(_seen)}")

# (2b) LONG jobs must not be spread. Spreading is right for many interchangeable short jobs; for a
#      training run it can pin the headline experiment behind the worse queue permanently, because
#      the choice is deterministic and does not change on resubmission. That happened to a41 on
#      2026-09-16 (36 h estimated wait on the partition the hash picked).
_long = {select_gpu_partition(time_h=24, spread_key=f"job{i}") for i in range(200)}
assert len(_long) == 1, f"a 24 h job was spread across {_long} -- long runs must take the preference"
_pref = ns["GPU_PARTITION_PREFERENCE"][0]
assert _long == {_pref}, f"long job went to {_long}, expected the preferred partition {_pref}"
print(f"[ok] {'long jobs are NOT spread':38s} -> 24h always {sorted(_long)[0]}")

# (3) Spreading must never override a real requirement.
assert select_gpu_partition(time_h=8, features=["system_ffmpeg"], spread_key="x") == "c23g"
assert select_gpu_partition(time_h=25, spread_key="x") == "c23g"
assert select_gpu_partition(time_h=8, gpu_mem_gb=90, spread_key="x") == "c23g"
print(f"[ok] {'requirements beat spreading':38s} -> ffmpeg / >24h / 90GB all pinned to c23g")

# (4) The task-derived key must survive a task object that cannot supply one.
assert ns["_spread_key"](None) is None, "_spread_key must tolerate a missing task, not raise"
assert ns["_spread_key"](_FakeTask("some/job/path")) == "some/job/path"
print(f"[ok] {'spread key is fail-safe':38s} -> None task -> falls back to preference order")

# --- escape hatches still work -------------------------------------------------------------------
# A raw pin is a "-p" the job DECLARES in its own Task rqmt (discouraged, not portable).
_raw = {"gpu": 1, "time": 8, "sbatch_args": ["-p", "c23g"]}
expect("raw -p pin (legacy, declared)", _raw, "c23g", task=_FakeTask("job/raw", _raw))
out = check_engine_limits({"gpu": 1, "time": 8, "sbatch_args": ["-x", "n23g0001"]}, None)
assert out["sbatch_args"].count("-x") == 1, out["sbatch_args"]
print(f"[ok] {'job-supplied -x preserved':38s} -> {' '.join(out['sbatch_args'])}")

# --- unsatisfiable requirements must fail loudly, not silently mis-route -------------------------
# NB: an over-long `time` is NOT in this list -- check_engine_limits clamps walltime to 72h before
# routing, so it becomes satisfiable on c23g rather than unsatisfiable. Asserted just below.
for bad in ({"gpu": 1, "time": 8, "requires": ["nonexistent_feature"]}, {"gpu": 1, "time": 8, "gpu_mem_gb": 500}):
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

# --- routing by estimated start (GPU_PARTITION_BY_START) ----------------------------------------------
# Hit 2026-10-18: the hash spread is blind to queues and put about half of all short jobs on a stalled c25g
# (27 jobs estimated to start 2-4 days out) while c23g started jobs in minutes. The probe is replaced by a
# fake here; one live probe is smoke-tested at the end.
ns["GPU_PARTITION_BY_START"] = True
_real_probe = ns.get("_estimated_start")
_now = 1_800_000_000.0


def _fake(starts):
    ns["_estimated_start"] = lambda partition, rqmt: starts[partition]


_r8 = {"gpu": 1, "cpu": 4, "mem": 48, "time": 8}
for _early, _late in (("c23g", "c25g"), ("c25g", "c23g")):
    _fake({_early: _now + 3600, _late: _now + 3 * 86400})
    _got = {select_gpu_partition(time_h=8, spread_key=f"job{i}", rqmt=_r8) for i in range(200)}
    assert _got == {_early}, f"{_early} starts 3 days sooner but jobs went to {_got}"
    _long = {select_gpu_partition(time_h=20, spread_key=f"job{i}", rqmt=dict(_r8, time=20)) for i in range(50)}
    assert _long == {_early}, f"a 20 h job ignored the earlier start on {_early}: {_long}"
print(f"[ok] {'earliest estimated start wins':38s} -> both directions, short and long jobs")

_fake({"c23g": _now, "c25g": _now + 600})  # 10 min apart: within GPU_START_TIE_H -> tie
_seen = {select_gpu_partition(time_h=8, spread_key=f"job{i}", rqmt=_r8) for i in range(200)}
assert _seen == {"c23g", "c25g"}, f"tied estimates must still spread, got {_seen}"
print(f"[ok] {'tied estimates still spread':38s} -> {sorted(_seen)}")

_fake({"c23g": None, "c25g": _now})  # one probe failed -> the old rules, unchanged
_seen = {select_gpu_partition(time_h=8, spread_key=f"job{i}", rqmt=_r8) for i in range(200)}
assert _seen == {"c23g", "c25g"}, f"a failed probe must fall back to spreading, got {_seen}"
print(f"[ok] {'failed probe falls back to spread':38s} -> {sorted(_seen)}")

_fake({"c23g": _now + 3 * 86400, "c25g": _now})
assert select_gpu_partition(time_h=25, spread_key="x", rqmt=dict(_r8, time=25)) == "c23g"
assert select_gpu_partition(time_h=8, gpu_mem_gb=90, spread_key="x", rqmt=_r8) == "c23g"
print(f"[ok] {'requirements beat the estimate':38s} -> >24h / 90GB stay on c23g though c25g is sooner")

# A resubmission carries our OWN earlier "-p" back in (update_engine_rqmt copies the last run's requested
# resources); it must be routed afresh, else a job on a stalled partition never leaves it.
_fake({"c23g": _now, "c25g": _now + 3 * 86400})
_hist = {"gpu": 1, "cpu": 4, "mem": 48, "time": 8, "sbatch_args": ["-p", "c25g", "-A", "p0023565", "-x", "n0"]}
_job = _FakeTask("work/.../SpeechInference.abc", {"gpu": 1, "cpu": 4, "mem": 48, "time": 8})
_out = check_engine_limits(dict(_hist), _job)["sbatch_args"]
assert _out.count("-p") == 1 and _out[_out.index("-p") + 1] == "c23g", f"history -p kept: {_out}"
print(f"[ok] {'resubmission is routed afresh':38s} -> {' '.join(_out)}")
_decl = dict(_hist, sbatch_args=["-p", "c25g"])
_out = check_engine_limits(dict(_hist), _FakeTask("work/.../X.abc", _decl))["sbatch_args"]
assert _out.count("-p") == 1 and _out[_out.index("-p") + 1] == "c25g", f"declared -p lost: {_out}"
print(f"[ok] {'a declared -p still pins':38s} -> {' '.join(_out)}")

# Live smoke: the real probe returns a time (login node) or None (no sbatch here) -- never raises.
ns["_estimated_start"] = _real_probe
_t = _real_probe("c23g", _r8)
assert _t is None or isinstance(_t, float), _t
print(
    f"[ok] {'live sbatch --test-only probe':38s} -> {'no sbatch here' if _t is None else 'start %+.1f h' % ((_t - __import__('time').time()) / 3600)}"
)

print("\nALL CHECKS PASSED")
