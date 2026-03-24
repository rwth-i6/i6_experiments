from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
import os
import signal
import sys
import subprocess
from .common import HF_CACHE_DIR
from contextlib import contextmanager
import shutil
import time
import random
import socket

# ln -s ../projects/2025-10-speech-llm/src/moshified_full_duplex_bench/v1_v1.5 moshified_fdb_v1_v15
import moshified_fdb_v1_v15.evaluation.evaluate as moshified_fdb_v1_v15_evaluate
from moshified_fdb_v1_v15.model_inference.moshi.inference import (
    _ws_url,
    MoshiFileClient,
)
from moshified_fdb_v1_v15.evaluation.evaluate import main as evaluate_main
from moshified_fdb_v1_v15.get_transcript.asr import get_time_aligned_transcription


def fdb_files_for_tasks(ds_path: Path, tasks: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for t in tasks:
        pattern = ds_path / f"{t}/*/input.wav"
        files += [(t, Path(p)) for p in sorted(glob(str(pattern)))]
    return files


def get_fdb_asr_download():
    repo = DownloadHuggingFaceRepoJob(model_id="kyutai/moshiko-pytorch-bf16")
    repo.out_hub_cache_dir = HF_CACHE_DIR
    return repo.out_hub_cache_dir


@contextmanager
def moshi_server():
    # first: find a free port
    # take my slurm jub id and modulo
    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        port = 8998 + (job_id % 1000)
    else:
        port = 8998 + random.randint(0, 999)
    # check if in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        for _ in range(50):
            try:
                s.bind(("localhost", port))
                break
            except OSError:
                port += 1

    print(f"Selected port {port} for Moshi server")

    cmd = [
        sys.executable,
        "-m",
        "moshi.moshi.server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    print(f"Starting Moshi server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,  # Create process group
        cwd="/home/tt201262/setups/2026-01-speech-llm/projects/moshi",  # TODO...
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # Wait for server to be ready (check port 8998)
    max_wait = 15 * 60  # seconds
    start_time = time.time()
    server_ready = False

    # Also read output to look for ready signal
    def read_server_output():
        nonlocal server_ready
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ""):
                if line and "frame handled" not in line:
                    print(f"[Moshi Server] {line.rstrip()}", flush=True)
                if "Access the Web UI directly at" in line:
                    server_ready = True
        print("Moshi server output thread exiting")

    # Start output reading thread
    import threading

    output_thread = threading.Thread(target=read_server_output, daemon=True)
    output_thread.start()

    # Wait for server to be ready
    while time.time() - start_time < max_wait:
        # Check if server process died
        if server_process.poll() is not None:
            stdout, _ = server_process.communicate()
            raise RuntimeError(f"Moshi server died during startup:\n{stdout}")

        # Try to connect to port
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            if server_ready:
                print("Moshi server is ready and accepting connections")
                break
            else:
                print("Moshi server port is open but server not ready yet")
        except (ConnectionRefusedError, socket.timeout):
            pass

        time.sleep(0.5)
    else:
        raise TimeoutError(f"Moshi server not ready after {max_wait} seconds")
    print("Moshi server started successfully")

    try:
        yield f"localhost:{port}"
    finally:
        # Stop the Moshi server if it's still running
        if server_process and server_process.poll() is None:
            print("Stopping Moshi server...")
            try:
                # Send SIGTERM to process group (Unix) or terminate (Windows)
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                else:
                    server_process.terminate()

                # Wait for graceful termination
                server_process.wait(timeout=10)
                print("Moshi server stopped gracefully")
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                print("Force killing Moshi server...")
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                else:
                    server_process.kill()
                server_process.wait()


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


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


FDB_TASK_MAP = {
    "candor_pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "synthetic_pause_handling": "pause_handling",
    "synthetic_user_interruption": "user_interruption",
}


class FullDuplexBenchEval_Inference(Job):
    def __init__(self, *, fdb_task: str, model):
        self.fdb_data = tk.Path(
            "/home/tt201262/setups/2026-01-speech-llm/projects/Full-Duplex-Bench/v1_v1.5/dataset/v1.0",
            hash_overwrite="FullDuplexBench-datasets",
        )  # TODO... the dataset is available as a google drive link, so no good way to write a download job?

        self.fdb_task = fdb_task
        self.model = model

        self.out_audios = self.output_path("audios", directory=True)

        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 4,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        # check if dataset is valid
        assert os.path.exists(
            os.path.join(self.fdb_data, "candor_pause_handling/1/pause.json")
        ), f"Dataset not found at {self.fdb_data}"

        files = fdb_files_for_tasks(Path(self.fdb_data.get_path()), [self.fdb_task])
        assert len(files) > 0, (
            f"No files found for task {self.fdb_task} in dataset {self.fdb_data.get_path()}"
        )

        with self.model() as url:
            url = _ws_url(url)
            print(f"Running inference with Moshi server at {url}...")

            # Run inference on all files
            for task, inp in files:
                ind = (
                    inp.parent.name
                )  # e.g. "1" in "v1.0/candor_pause_handling/1/input.wav"

                out = Path(self.out_audios.get_path()) / str(ind) / "output.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                print("[RUN]", task, inp)
                for _ in range(3):  # retry a few times if it fails
                    try:
                        MoshiFileClient(url, inp, out).run()
                        break
                    except Exception as e:
                        print(f"Error processing {inp}: {e}")
                        time.sleep(1)

                # Find all other .json files in that folder, and copy them
                for json_file in inp.parent.glob("*.json"):
                    out_json = out.parent / json_file.name
                    shutil.copy(json_file, out_json)

        # pys v1_v1.5/get_transcript/asr.py --root_dir v1_v1.5/dataset/v1.0/icc_backchannel --task default

        """
        choices=[
            "backchannel",
            "pause_handling",
            "smooth_turn_taking",
            "user_interruption",
            "behavior",
            "general_before_after",
        ],
        """

        # this takes output.wav and puts output.json in the same folder
        get_time_aligned_transcription(
            self.out_audios.get_path(), FDB_TASK_MAP.get(self.fdb_task, self.fdb_task)
        )


class FullDuplexBenchEval_Evaluation(Job):
    def __init__(
        self,
        *,
        fdb_task: str,
        in_audios: tk.Path,
        hf_model: str = "openai/gpt-oss-120b",
    ):
        self.fdb_task = fdb_task

        self.in_audios = in_audios
        self.out_log = self.output_path("evaluation_output.txt")
        self.out_eval = self.output_path("eval.json")
        self.needs_llm_inference = FDB_TASK_MAP.get(self.fdb_task, self.fdb_task) in [
            "user_interruption",
            "behavior",
            "general_before_after",
        ]
        if self.needs_llm_inference:
            self.rqmt = {
                "gpu": 1,
                "cpu": 2,
                "mem": 16,
                "time": 2,
            }
            self.hf_model = hf_model
        else:
            self.rqmt = None

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 5
        return super().hash(d)

    def tasks(self):
        if self.rqmt is None:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        # pys evaluate.py --task backchannel --root_dir ../dataset/v1.0/icc_backchannel
        # hacky...
        sys.argv = [
            "evaluate.py",
            "--task",
            FDB_TASK_MAP.get(self.fdb_task, self.fdb_task),
            "--root_dir",
            self.in_audios.get_path(),
        ]
        # also now record stdout into a file (but still also regular stdout) in self.output_path("evaluation_output.txt")
        with open(self.out_log, "w", encoding="utf-8") as f:
            # redirect stdout to both console and file
            tee = Tee(sys.stdout, f)
            sys.stdout = tee

            # add the path of evaluate.py to sys.path so it can import modules
            sys.path.append(str(Path(moshified_fdb_v1_v15_evaluate.__file__).parent))

            if self.needs_llm_inference:
                # TODO fdb code would be faster if it utilized batching...
                with vllm_server(self.hf_model) as llm_url:
                    sys.argv += ["--llm-api-url", llm_url, "--llm-model", self.hf_model]
                    # set OPENAI_API_KEY
                    os.environ["OPENAI_API_KEY"] = "fake_key_for_vllm"
                    res = evaluate_main()
            else:
                res = evaluate_main()
            with open(self.out_eval, "w", encoding="utf-8") as f_eval:
                json.dump(res, f_eval, indent=4)
            sys.stdout = sys.__stdout__  # restore original stdout


def moshified_fdb_eval(fdb_task: str, model):
    infer = FullDuplexBenchEval_Inference(fdb_task=fdb_task, model=model)
    _eval = FullDuplexBenchEval_Evaluation(
        fdb_task=fdb_task, in_audios=infer.out_audios
    )
    return _eval.out_eval
