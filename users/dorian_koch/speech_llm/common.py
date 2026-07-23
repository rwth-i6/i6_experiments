from sisyphus import tk
import os
import signal
import sys
import subprocess
import threading
import time
import socket
import random
from contextlib import contextmanager

HF_CACHE_DIR = tk.Path("/hpcwork/p0023999/common_hf_home/hub", hash_overwrite="HF_CACHE_DIR")


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
def vllm_server(hf_model: str, max_model_len: int | None = None):
    # `max_model_len` override: a short-context caller (e.g. LLMGrading, whose prompts are <1k tokens)
    # can pass a small value so the judge's KV cache fits c25g's 80 GB H100 at TP=1 -- otherwise the
    # dict's large context (gemma 65536 -> ~12 GiB KV) only fits c23g's 94 GB cards, forcing the job onto
    # the scarce c23g queue. None keeps the per-model dict default (dialogue-gen needs the long context).
    port = pick_free_port(18998)
    print(f"Selected port {port} for vLLM server")
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
        "0.9",
        "--enable-prefix-caching",
        "true",
    ]
    model_args = list(_VLLM_MODEL_ARGS.get(hf_model, []))
    if max_model_len is not None:
        if "--max-model-len" in model_args:
            model_args[model_args.index("--max-model-len") + 1] = str(max_model_len)
        else:
            model_args += ["--max-model-len", str(max_model_len)]
    cmd += model_args

    with managed_subprocess_server(
        cmd,
        port=port,
        ready_substrings=("Uvicorn running on", "Application startup complete"),
        log_prefix="vLLM",
    ):
        yield f"http://localhost:{port}/v1"


# ---------------------------------------------------------------------------
# Worker-script subprocess launcher (shared by the Job.run() wrappers that shell
# out to a sibling worker script in their own venv).
# ---------------------------------------------------------------------------


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
        env["HF_HOME"] = HF_CACHE_DIR.get()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    if env_hook is not None:
        env_hook(env)
    print(f"Running {log_label}: {' '.join(cmd)}", flush=True)
    if with_hf_home:
        print(f"Using HF cache directory: {HF_CACHE_DIR}")
    subprocess.run(cmd, env=env, check=True)


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

    try:
        with open(os.path.join(job._sis_path(gs.JOB_WORK_DIR), "progress.json")) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    total = d.get("total") or 0
    if total <= 0:
        return None
    return max(0.0, min(1.0, d.get("done", 0) / total))


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
