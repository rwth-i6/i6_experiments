from sisyphus import tk
import os
import signal
import sys
import subprocess
import time
import socket
import random
from contextlib import contextmanager
from pathlib import Path

HF_CACHE_DIR = tk.Path(
    "/hpcwork/p0023999/common_hf_home/hub", hash_overwrite="HF_CACHE_DIR"
)


@contextmanager
def vllm_server(hf_model: str):
    # first: find a free port
    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        port = 18998 + (job_id % 1000)
    else:
        port = 18998 + random.randint(0, 999)
    # check if in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(50):
            try:
                s.bind(("localhost", port))
                break
            except OSError:
                port += 1

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

    print(f"Starting vLLM server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    max_wait = 15 * 60  # seconds
    start_time = time.time()
    server_ready = False

    def read_server_output():
        nonlocal server_ready
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ""):
                if line:
                    print(f"[vLLM Server] {line.rstrip()}", flush=True)
                if (
                    "Uvicorn running on" in line
                    or "Application startup complete" in line
                ):
                    server_ready = True
        print("vLLM server output thread exiting")

    import threading

    output_thread = threading.Thread(target=read_server_output, daemon=True)
    output_thread.start()

    while time.time() - start_time < max_wait:
        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            raise RuntimeError(f"vLLM server died during startup:\n{stdout}")

        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            if server_ready:
                print("vLLM server is ready and accepting connections")
                break
            else:
                print("vLLM server port is open but server not ready yet")
        except (ConnectionRefusedError, socket.timeout):
            pass

        time.sleep(0.5)
    else:
        raise TimeoutError(f"vLLM server not ready after {max_wait} seconds")
    print("vLLM server started successfully")

    try:
        yield f"http://localhost:{port}/v1"
    finally:
        if server_process and server_process.poll() is None:
            print("Stopping vLLM server...")
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                else:
                    server_process.terminate()

                server_process.wait(timeout=10)
                print("vLLM server stopped gracefully")
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                print("Force killing vLLM server...")
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    server_process.kill()
                server_process.wait()

