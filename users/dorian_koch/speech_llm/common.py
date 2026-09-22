from sisyphus import tk
import json
import os
import signal
import sys
import subprocess
import threading
import time
import socket
import random
from contextlib import contextmanager

#: The HF **hub** cache -- the directory that holds ``models--*`` / ``datasets--*``. This is the
#: value for ``HF_HUB_CACHE``, NOT for ``HF_HOME``.
HF_CACHE_DIR = tk.Path("/hpcwork/p0023999/common_hf_home/hub", hash_overwrite="HF_CACHE_DIR")

#: The HF **home** root, one level up. ``huggingface_hub`` derives its hub cache as
#: ``$HF_HOME/hub``, so assigning the *hub* path to ``HF_HOME`` makes it append ``/hub`` a second
#: time and quietly opens a SECOND cache at ``common_hf_home/hub/hub`` -- which is exactly what
#: happened: 14 repos / 50 GB were re-downloaded there because ``run_worker_script`` set only
#: ``HF_HOME``. ``finetune.py`` escaped it by also setting ``HF_HUB_CACHE``, which wins. Both are set
#: together everywhere now, so neither variable alone can move the cache. Run()-side, not hashed.
HF_HOME_DIR = tk.Path("/hpcwork/p0023999/common_hf_home", hash_overwrite="HF_HOME_DIR")


# ---------------------------------------------------------------------------
# Subprocess-server scaffolding (shared by vllm_server + moshi_client.moshi_server)
# ---------------------------------------------------------------------------
#
# Both servers boot a local model server in a subprocess and need the exact same
# lifecycle: pick a free port, Popen in its own process group, tail stdout in a
# thread until a "ready" line appears, poll the port with a timeout, and tear the
# whole process group down on exit. None of this is hashed (pure runtime).


def pick_free_port(base: int) -> int:
    """Pick a free port near ``base``.

    Seeded by ``SLURM_JOB_ID`` (so concurrent array tasks spread out) and then
    bumped past any port already in use.
    """
    if "SLURM_JOB_ID" in os.environ:
        port = base + (int(os.environ["SLURM_JOB_ID"]) % 1000)
    else:
        port = base + random.randint(0, 999)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(50):
            try:
                s.bind(("localhost", port))
                break
            except OSError:
                port += 1
    return port


def _terminate_process_group(proc, log_prefix: str) -> None:
    """SIGTERM (then SIGKILL) the server's whole process group."""
    if not (proc and proc.poll() is None):
        return
    print(f"Stopping {log_prefix} server...")
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
        print(f"{log_prefix} server stopped gracefully")
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        print(f"Force killing {log_prefix} server...")
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
        else:
            proc.kill()
        proc.wait()


@contextmanager
def managed_subprocess_server(
    cmd,
    *,
    port: int,
    ready_substrings,
    log_prefix: str,
    cwd: str | None = None,
    env: dict | None = None,
    drop_line: str | None = None,
    max_wait: int = 15 * 60,
):
    """Run ``cmd`` as a model server, yield once it is ready, tear it down after.

    Streams the server's stdout (dropping lines containing ``drop_line`` if set),
    marks the server ready when any of ``ready_substrings`` is seen, and waits for
    the port to accept connections. Raises if the process dies during startup or
    is not ready within ``max_wait`` seconds.
    """
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if env:
        full_env.update(env)
    print(f"Starting {log_prefix} server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        cwd=cwd,
        env=full_env,
    )

    ready = {"flag": False}

    def read_server_output():
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                if line and (drop_line is None or drop_line not in line):
                    print(f"[{log_prefix} Server] {line.rstrip()}", flush=True)
                if any(sub in line for sub in ready_substrings):
                    ready["flag"] = True
        print(f"{log_prefix} server output thread exiting")

    threading.Thread(target=read_server_output, daemon=True).start()

    start_time = time.time()
    while time.time() - start_time < max_wait:
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            raise RuntimeError(f"{log_prefix} server died during startup:\n{stdout}")
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            if ready["flag"]:
                print(f"{log_prefix} server is ready and accepting connections")
                break
            print(f"{log_prefix} server port is open but server not ready yet")
        except (ConnectionRefusedError, socket.timeout):
            pass
        time.sleep(0.5)
    else:
        raise TimeoutError(f"{log_prefix} server not ready after {max_wait} seconds")
    print(f"{log_prefix} server started successfully")

    try:
        yield port
    finally:
        _terminate_process_group(proc, log_prefix)


# Per-model vLLM serving overrides (context length etc.); models not listed use vLLM defaults.
# gemma keeps its exact prior args so existing dialogue-gen jobs are unaffected. GPT-OSS-120B is an
# MoE shipping MXFP4 weights (~63 GB) -> fits one H100; if it OOMs on an 80 GB card, raise the
# dialogue shard's gpu rqmt and add "--tensor-parallel-size 2" here.
_VLLM_MODEL_ARGS: dict[str, list[str]] = {
    "google/gemma-4-31B-it": ["--max-model-len", "65536"],
    "Qwen/Qwen3-32B": ["--max-model-len", "32768"],
    "openai/gpt-oss-120b": ["--max-model-len", "32768"],
}


@contextmanager
def vllm_server(
    hf_model: str,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    enforce_eager: bool = False,
):
    # `max_model_len` override: a short-context caller (e.g. LLMGrading, whose prompts measure ~2.5k
    # tokens worst case over the real data, median 249) can pass a small value so the judge's KV cache
    # fits c25g's 80 GB H100 at TP=1 -- otherwise the
    # dict's large context (gemma 65536 -> ~12 GiB KV) only fits c23g's 94 GB cards, forcing the job onto
    # the scarce c23g queue. None keeps the per-model dict default (dialogue-gen needs the long context).
    #
    # `gpu_memory_utilization` override: 0.9 demands 71.26 of the card's 79.18 GiB and vLLM REFUSES TO
    # START if that much is not free, so a few GiB left on the card by anyone else kills the job --
    # and one errored job stalls the entire Sisyphus graph. Seen 2026-09-06: LLMGrading landed on
    # n25g0004 with 10.8 GiB already resident (68.38 free vs 71.26 wanted) and died. Lower this ONLY
    # for callers with KV headroom to spare; it is a straight trade of KV cache for tolerance of a
    # dirty card, and LLMPreprocess (12288) has none to give.
    port = pick_free_port(18998)
    print(f"Selected port {port} for vLLM server")
    # Tensor-parallel over every GPU the job was handed (backlog G1). On a 1-GPU allocation this is
    # exactly the old command; on a 4-GPU one the weights are split 4 ways, which is both faster
    # and what makes the big judges fit a 80 GB card WITHOUT the max_model_len workaround (kept as
    # a floor for the 1-GPU case). The RL judge/trainer split pins this process to one card via
    # CUDA_VISIBLE_DEVICES, so it sees 1 and stays at TP=1.
    n_gpus = max(1, len(visible_gpus()))
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        hf_model,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--enable-prefix-caching",
        "true",
    ]
    # `enforce_eager` skips torch.compile + CUDA-graph capture. Available, but OFF everywhere by
    # default, and the measurement is why:
    #
    #   run              weights  compile  capture  engine-init
    #   yuHZaBu9zg08      98.3 s   33.2 s    10 s      69.0 s     <- COLD node
    #   jQxiG0smmHPT      41.4 s   12.2 s     7 s      40.9 s
    #   oocY3FqKpufd      48.5 s   10.5 s     8 s      40.7 s
    #
    # A first reading of the cold run alone said "43 s of compile+capture, 70% of the job is boot"
    # and enabled eager on the judge. Two warm runs withdrew that: compile is ~11 s and capture
    # ~8 s once the torch.compile cache is warm, so eager buys ~19 s of boot and pays it back in
    # slower decoding across every request. On ~1,000 gradings that is very likely a net loss.
    #
    # Two things that reading got right and are worth keeping: the compile cache ALREADY works here
    # (33 s cold -> 10-12 s warm) because ~/.cache is symlinked to shared hpcwork, so every node
    # reuses it; and weight loading is the dominant phase (41-98 s), which is what to attack.
    #
    # So: measure with the "[vllm-timing]" line below before enabling this for any caller. Do not
    # infer it from one cold boot, which is exactly the mistake this comment records.
    if enforce_eager:
        cmd += ["--enforce-eager"]
    if n_gpus > 1:
        print(f"vLLM: tensor-parallel over {n_gpus} visible GPUs", flush=True)
        cmd += ["--tensor-parallel-size", str(n_gpus)]
    model_args = list(_VLLM_MODEL_ARGS.get(hf_model, []))
    if max_model_len is not None:
        if "--max-model-len" in model_args:
            model_args[model_args.index("--max-model-len") + 1] = str(max_model_len)
        else:
            model_args += ["--max-model-len", str(max_model_len)]
    cmd += model_args

    # Boot cost is invisible unless it is recorded: this job's wall time reads as "grading was
    # slow" when in fact the model spent most of it loading. One line, so any future caller can see
    # its own split without re-deriving it from vLLM's own log.
    _t0 = time.time()
    print(f"[vllm-timing] starting server for {hf_model} (enforce_eager={enforce_eager})", flush=True)
    with managed_subprocess_server(
        cmd,
        port=port,
        ready_substrings=("Uvicorn running on", "Application startup complete"),
        log_prefix="vLLM",
        # A cold Lustre load of a big checkpoint (gemma-4-31B: 58 GiB) can take ~11-12 min for
        # weights alone before CUDA-graph capture -- the 900s default tips over on a c25g node whose
        # page cache is cold, though warm c23g nodes booted fine. Give cold loads generous headroom;
        # a genuinely dead server is still caught immediately by the proc.poll() check.
        max_wait=30 * 60,
    ):
        _boot = time.time() - _t0
        print(f"[vllm-timing] server ready after {_boot:.1f}s", flush=True)
        try:
            yield f"http://localhost:{port}/v1"
        finally:
            _total = time.time() - _t0
            _work = _total - _boot
            _pct = 100.0 * _boot / _total if _total > 0 else 0.0
            print(
                f"[vllm-timing] boot {_boot:.1f}s + work {_work:.1f}s = {_total:.1f}s "
                f"({_pct:.0f}% of this job was vLLM startup)",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Worker-script subprocess launcher (shared by the Job.run() wrappers that shell
# out to a sibling worker script in their own venv).
# ---------------------------------------------------------------------------


def add_cuda_npp_to_env(venv_python_path, env: dict) -> None:
    """Put a venv's CUDA NPP libraries on ``LD_LIBRARY_PATH``.

    torchcodec's ``libtorchcodec_core*.so`` link against ``libnppicc.so.12`` (NVIDIA Performance
    Primitives) for colour conversion. torch does NOT bundle NPP -- a torch venv ships
    cublas/cudnn/cufft/... under ``site-packages/nvidia/`` but no ``npp`` -- so the import dies with
    ``libnppicc.so.12: cannot open shared object file`` on any node that does not provide it
    system-wide. Installing ``nvidia-npp-cu12`` puts it in the venv; this puts it on the path.

    Pair with :meth:`InstallFFmpeg.add_to_env`. Together they are what make torchcodec
    node-independent, i.e. what lets a job drop ``requires: ["system_ffmpeg"]``. Runtime only --
    nothing here is hashed.
    """
    import glob as _glob

    from i6_experiments.users.dorian_koch.jobs.sqsh_venv import venv_prefix

    base = venv_prefix(venv_python_path)  # a packed venv's bin/ is a launcher skeleton with no lib/
    hits = _glob.glob(os.path.join(base, "lib", "python*", "site-packages", "nvidia", "npp", "lib"))
    if not hits:
        print(f"[npp] WARNING: no nvidia/npp/lib under {base} -- is nvidia-npp-cu12 installed?", flush=True)
        return
    env["LD_LIBRARY_PATH"] = hits[0] + ":" + env.get("LD_LIBRARY_PATH", "")


def add_venv_python_lib_to_env(venv_python_path, env: dict) -> None:
    """Put the venv's BASE INTERPRETER ``lib/`` on ``LD_LIBRARY_PATH``.

    ``libtorchcodec_core*.so`` resolves ``libpython3.12.so.1.0`` transitively, and our interpreter is
    a uv-managed CPython under ``~/.local/share/uv/python/``, not a system one -- so nothing puts its
    ``lib/`` on the loader path. c23g happens to provide a compatible libpython system-wide and
    **c25g does not**, which is the same class of node-dependency as the missing NPP: measured
    2026-09-16, a c25g job got past ``libnppicc`` and then died on ``libpython3.12.so.1.0``.

    Pair with :func:`add_cuda_npp_to_env` and :meth:`InstallFFmpeg.add_to_env`. Those three together
    are what let a job drop ``requires: ["system_ffmpeg"]`` -- verified on a real c25g node with no
    system FFmpeg and no system NPP: torchcodec imports and a ``datasets`` ``Audio()`` column round
    trips. Runtime only, nothing hashed.

    The base interpreter is named by ``pyvenv.cfg``'s ``home`` (its ``bin/``), so its ``lib/`` is the
    sibling. Read from the venv rather than from ``sys`` -- the caller is the MANAGER's interpreter,
    which is a different Python from the job's.
    """
    import os as _os

    from i6_experiments.users.dorian_koch.jobs.sqsh_venv import venv_prefix

    base = venv_prefix(venv_python_path)
    cfg = _os.path.join(base, "pyvenv.cfg")
    home = None
    try:
        with open(cfg) as fh:
            for line in fh:
                if line.startswith("home"):
                    home = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass
    if not home:
        print(f"[pylib] WARNING: no `home` in {cfg} -- libpython may not resolve", flush=True)
        return
    lib = _os.path.join(_os.path.dirname(home), "lib")
    if not _os.path.isdir(lib):
        print(f"[pylib] WARNING: {lib} does not exist -- libpython may not resolve", flush=True)
        return
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")


def run_worker_script(
    python_exe,
    script_path,
    args,
    *,
    log_label: str,
    with_hf_home: bool = True,
    extra_env: dict | None = None,
    env_hook=None,
) -> None:
    """Run ``python_exe script_path *args`` as a checked subprocess.

    Sets ``PYTHONUNBUFFERED`` (and ``HF_HOME`` when ``with_hf_home``), then applies
    ``extra_env`` and finally ``env_hook(env)`` (in-place mutation, e.g. to splice in
    an FFmpeg install) before launching. Pure runtime helper; nothing here is hashed.
    """
    cmd = [str(python_exe), str(script_path), *[str(a) for a in args]]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if with_hf_home:
        env["HF_HOME"] = HF_HOME_DIR.get()
        env["HF_HUB_CACHE"] = HF_CACHE_DIR.get()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    if env_hook is not None:
        env_hook(env)
    print(f"Running {log_label}: {' '.join(cmd)}", flush=True)
    if with_hf_home:
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
    subprocess.run(cmd, env=env, check=True)


# ---------------------------------------------------------------------------
# Multi-GPU fan-out (backlog G1): a job scales to however many GPUs the cluster hands it.
# ---------------------------------------------------------------------------
#
# A job declares the SMALLEST allocation it can use (usually ``gpu: 1``); ``settings.py`` may raise
# that to the cluster's minimum (``MIN_GPUS_PER_JOB`` -- 1 here, 4 on a cluster whose smallest GPU
# node is 4 cards). Nothing in the recipe hardcodes the count: at run time the job asks
# ``visible_gpus()`` what it actually got and fans its embarrassingly-parallel worker out over them,
# one subprocess per card. ``rqmt`` is not hashed, so none of this moves a job hash.
#
# Determinism contract: the fan-out must not change WHAT a job produces, only how fast. Every worker
# that participates therefore (a) derives per-item randomness from the item INDEX (TTS seeds are
# ``SEED + i``), (b) writes to index-disjoint locations or per-shard parts the job merges in index
# order, and (c) never depends on the order in which shards finish.


def visible_gpus() -> list[str]:
    """The GPU ids this process may use, as CUDA sees them.

    Reads ``CUDA_VISIBLE_DEVICES`` (SLURM sets it inside the job cgroup; the RL judge/trainer split
    sets it by hand). An unset variable means "everything on the node" -- ask ``nvidia-smi`` -- and an
    explicitly EMPTY one means no GPU. Never imports torch: the caller may be a login-node process
    or a job venv without it.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [x.strip() for x in env.split(",") if x.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith("GPU ")]


def run_worker_script_per_gpu(
    python_exe,
    script_path,
    args_for_shard,
    *,
    log_label: str,
    max_gpus: int | None = None,
    with_hf_home: bool = True,
    extra_env: dict | None = None,
    env_hook=None,
) -> int:
    """Run ``script_path`` once per visible GPU, each pinned to its card, and wait for all.

    ``args_for_shard(k, n)`` returns the argv for worker ``k`` of ``n`` -- the worker's own in-job
    shard flags plus any per-shard output path. With one visible GPU this is exactly
    :func:`run_worker_script`, so the single-GPU path is unchanged byte for byte.

    Each worker runs in its own subdirectory ``gpu<k>/`` of the job work dir, so a worker's
    ``progress.json`` (and any other cwd-relative scratch) cannot collide; ``job_progress_fraction``
    sums those. Worker stdout/stderr go to ``gpu<k>/worker.log`` AND are echoed, prefixed, to the
    job log, so a failure still reads from ``log.run.1``. The first worker to fail kills the rest
    and the job fails -- a partial output must never be merged.

    Returns the number of workers that ran, so the caller knows how many parts to merge.
    """
    gpus = visible_gpus()
    n = len(gpus)
    if max_gpus is not None:
        n = min(n, max_gpus)
    if n <= 1:
        run_worker_script(
            python_exe,
            script_path,
            args_for_shard(0, 1),
            log_label=log_label,
            with_hf_home=with_hf_home,
            extra_env=extra_env,
            env_hook=env_hook,
        )
        return 1

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    if with_hf_home:
        base_env["HF_HOME"] = HF_HOME_DIR.get()
        base_env["HF_HUB_CACHE"] = HF_CACHE_DIR.get()
    if extra_env:
        base_env.update({k: str(v) for k, v in extra_env.items()})
    if env_hook is not None:
        env_hook(base_env)

    print(f"[fan-out] {log_label}: {n} workers over GPUs {gpus[:n]}", flush=True)
    procs = []
    for k in range(n):
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpus[k]
        cwd = os.path.join(os.getcwd(), f"gpu{k}")
        os.makedirs(cwd, exist_ok=True)
        cmd = [str(python_exe), str(script_path), *[str(a) for a in args_for_shard(k, n)]]
        print(f"[fan-out] worker {k}/{n} on GPU {gpus[k]}: {' '.join(cmd)}", flush=True)
        log = open(os.path.join(cwd, "worker.log"), "w")
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((k, proc, log))

    failed = None
    try:
        # Poll rather than wait() in order: a worker that dies early must stop the others.
        pending = {k for k, _p, _l in procs}
        while pending:
            for k, proc, _log in procs:
                if k not in pending:
                    continue
                rc = proc.poll()
                if rc is None:
                    continue
                pending.discard(k)
                if rc != 0 and failed is None:
                    failed = (k, rc)
                    for j, other, _l in procs:
                        if j in pending and other.poll() is None:
                            other.terminate()
            if pending:
                time.sleep(5)
    finally:
        for _k, _proc, log in procs:
            log.close()
    for k, _proc, _log in procs:
        path = os.path.join(os.getcwd(), f"gpu{k}", "worker.log")
        try:
            with open(path) as f:
                tail = f.readlines()[-40:]
        except OSError:
            tail = []
        print(f"[fan-out] ---- worker {k} log tail ({path}) ----", flush=True)
        for line in tail:
            print(f"[gpu{k}] {line.rstrip()}", flush=True)
    if failed is not None:
        k, rc = failed
        raise RuntimeError(
            f"{log_label}: worker {k}/{n} exited with {rc}; the other workers were terminated and "
            f"nothing was merged. Full log: {os.path.join(os.getcwd(), f'gpu{k}', 'worker.log')}"
        )
    return n


def contiguous_slice(total: int, k: int, n: int) -> tuple[int, int]:
    """``[start, end)`` of part ``k`` of ``n`` over ``total`` items, the way
    ``datasets.Dataset.shard(num_shards=n, index=k, contiguous=True)`` cuts them: the first
    ``total % n`` parts get one extra item. Used by workers that sub-shard a dataset per GPU and
    must keep GLOBAL row numbering (ids, seeds) identical to the single-GPU run."""
    assert 0 <= k < n, (k, n)
    div, mod = divmod(int(total), n)
    start = k * div + min(k, mod)
    return start, start + div + (1 if k < mod else 0)


def nested_shard(shard: int | None, num_shards: int | None, k: int, n: int) -> tuple[int, int]:
    """Sub-shard ``k`` of ``n`` INSIDE Sisyphus-level shard ``shard`` of ``num_shards``, for a
    worker whose sharding rule is strided (``items[shard::num_shards]``): ``(shard + k*num_shards,
    num_shards*n)`` selects exactly the items ``i`` with ``i % num_shards == shard`` and
    ``(i // num_shards) % n == k`` -- a partition of the outer shard over the ``n`` GPUs with no
    change to the worker at all."""
    s, S = (0, 1) if shard is None or num_shards is None else (int(shard), int(num_shards))
    return s + k * S, S * n


def merge_hf_parts(out_path, parts) -> int:
    """Concatenate per-GPU HF dataset parts -- written from CONTIGUOUS sub-shards, in order -- into
    ``out_path``, so the result has the row order a single worker would have produced. Rows keep
    their ids untouched (unlike ``HfMergeShards``, which prefixes Sisyphus-shard ids). A
    ``stats.json`` of integer counters, if the parts carry one, is summed. The parts are deleted
    after a successful write. Returns the row count."""
    import shutil

    from datasets import concatenate_datasets, load_from_disk

    parts = [str(p) for p in parts]
    merged = concatenate_datasets([load_from_disk(p) for p in parts])
    assert len(merged) > 0, f"per-GPU parts are all empty: {parts}"
    merged.save_to_disk(str(out_path))
    stats = None
    for p in parts:
        sp = os.path.join(p, "stats.json")
        if os.path.isfile(sp):
            with open(sp) as f:
                d = json.load(f)
            stats = d if stats is None else {k: stats.get(k, 0) + v for k, v in d.items()}
    if stats is not None:
        with open(os.path.join(str(out_path), "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    for p in parts:
        shutil.rmtree(p, ignore_errors=True)
    print(f"[fan-out] merged {len(parts)} parts -> {out_path} ({len(merged)} rows)", flush=True)
    return len(merged)


def merge_jsonl_parts(out_path, parts) -> int:
    """Merge per-GPU ``<part>`` jsonl files, each with a ``<part>.idx`` sidecar (one clip index per
    line, same order), into ``out_path`` in ascending clip-index order -- the order a single worker
    would have written. Returns the row count. The sidecar keeps the record schema untouched."""
    rows = []
    for part in parts:
        with open(part) as f, open(str(part) + ".idx") as g:
            recs, idx = f.read().splitlines(), g.read().splitlines()
        assert len(recs) == len(idx), f"{part}: {len(recs)} records but {len(idx)} indices"
        rows.extend((int(i), r) for i, r in zip(idx, recs))
    rows.sort(key=lambda t: t[0])
    seen = [i for i, _ in rows]
    assert len(set(seen)) == len(seen), "per-GPU parts overlap: the same clip index appears twice"
    with open(out_path, "w") as f:
        for _i, r in rows:
            f.write(r + "\n")
    return len(rows)


# ---------------------------------------------------------------------------
# Progress reporting helpers (for Job.completed_fraction)
# ---------------------------------------------------------------------------
#
# Sisyphus calls Job.completed_fraction() manager-side for every running job and
# prints "[XX.X%]". It must be cheap and read progress off the shared filesystem.
# Two flavours:
#   - job_progress_fraction(job): reads a tiny progress.json our worker scripts
#     write into their cwd (= the job work dir) as {"done", "total"}.
#   - last_jsonl_value(path, field): tail-reads the last JSON line of a .jsonl
#     metrics file (e.g. moshi-finetune's metrics.train.jsonl "percent_done").


def map_concurrent(fn, items, *, concurrency: int, progress_path: str | None = None, progress_every: int = 10) -> list:
    """Apply ``fn`` to every item with ``concurrency`` threads; results come back in INPUT order.

    For I/O-bound calls against one of our LLM servers (:func:`vllm_server`): one thread per request
    in flight, so the server sees ``concurrency`` requests at once and batches them. The concurrency
    is what decides the server's utilisation, so set it from a measurement -- vLLM logs
    ``Running: N reqs, Waiting: M reqs, GPU KV cache usage: X%`` every 10 s. Measured 2026-09-22 on an
    H100 with gpt-oss-120b: 32 in flight ran at 5% KV use and 1.6k generated tok/s; 192 ran at ~28%
    and 5.3k tok/s. (32 was the ``datasets.map(num_proc=32)`` of the job it replaced; a serial loop
    is 1 in flight.) Jobs whose log already shows the KV cache full with requests waiting gain
    nothing from more.

    An exception in ``fn`` propagates, as it would from a serial loop; handle per-item failures
    (retries, None results) inside ``fn``. ``progress_path``: a :func:`write_progress` marker, updated
    every ``progress_every`` completions, for ``Job.completed_fraction``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    items = list(items)
    out = [None] * len(items)
    with ThreadPoolExecutor(max(1, int(concurrency))) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for n_done, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if progress_path and (n_done % progress_every == 0 or n_done == len(items)):
                try:
                    write_progress(n_done, len(items), progress_path)
                except Exception:
                    pass
    return out


def write_progress(done: int, total: int, path: str = "progress.json") -> None:
    """Worker-side: atomically write a {done, total} progress marker.

    Call this from an inference loop; the file lands in the process cwd, which is
    the Sisyphus job work dir, so Job.completed_fraction() can read it back.
    """
    import json
    import os
    import tempfile

    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".progress-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"done": int(done), "total": int(total)}, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def job_progress_fraction(job) -> "float | None":
    """Manager-side: read a worker-written progress.json from the job work dir.

    :return: fraction in [0, 1], or None if no usable progress yet.
    """
    import json
    import os
    from sisyphus import global_settings as gs

    import glob

    work = job._sis_path(gs.JOB_WORK_DIR)
    # One progress.json for a single worker; under the per-GPU fan-out each worker writes its own
    # in gpu<k>/, and the job's progress is their sum.
    paths = [os.path.join(work, "progress.json")] + sorted(glob.glob(os.path.join(work, "gpu*", "progress.json")))
    done = total = 0
    for path in paths:
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        done += int(d.get("done", 0) or 0)
        total += int(d.get("total", 0) or 0)
    if total <= 0:
        return None
    return max(0.0, min(1.0, done / total))


def last_jsonl_value(path: str, field: str):
    """Tail-read the last complete JSON line of a .jsonl file and return field.

    Cheap: only the file tail is read. Returns None if unavailable/unparseable.
    """
    import json
    import os

    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size == 0:
        return None
    with open(path, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read()
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if field in obj:
            return obj[field]
    return None
