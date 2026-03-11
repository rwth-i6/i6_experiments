from glob import glob
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
import os
import signal
import sys
import subprocess
from .common import HF_CACHE_DIR
from contextlib import contextmanager

def fdp_files_for_tasks(ds_path: Path, tasks: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for t in tasks:
        pattern = ds_path / f"{t}/*/input.wav"
        files += [(t, Path(p)) for p in sorted(glob(str(pattern)))]
    return files

def get_fdp_asr_download():
    repo = DownloadHuggingFaceRepoJob(model_id="kyutai/moshiko-pytorch-bf16")
    repo.out_hub_cache_dir = HF_CACHE_DIR
    return repo.out_hub_cache_dir

@contextmanager
def moshi_server():
     # Build command for Moshi server
    cmd = [sys.executable, "-m", "moshi.server", "--host", "localhost", "--port", "8998"]
    
    print(f"Starting Moshi server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create process group
    )
    
    # Wait for server to be ready (check port 8998)
    import socket
    import time
    max_wait = 120  # seconds
    start_time = time.time()
    server_ready = False
    
    # Also read output to look for ready signal
    def read_server_output():
        nonlocal server_ready
        if server_process.stdout:
            for line in iter(server_process.stdout.readline, ''):
                if line:
                    print(f"[Moshi Server] {line.rstrip()}")
                if "Access the Web UI directly at" in line:
                    server_ready = True
    
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
            sock = socket.create_connection(('localhost', 8998), timeout=1)
            sock.close()
            if server_ready:
                print("Moshi server is ready and accepting connections")
                break
        except (ConnectionRefusedError, socket.timeout):
            pass
        
        time.sleep(0.5)
    else:
        raise TimeoutError(f"Moshi server not ready after {max_wait} seconds")
    
    try:
        yield
    finally:
        # Stop the Moshi server if it's still running
        if server_process and server_process.poll() is None:
            print("Stopping Moshi server...")
            try:
                # Send SIGTERM to process group (Unix) or terminate (Windows)
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                else:
                    server_process.terminate()
                
                # Wait for graceful termination
                server_process.wait(timeout=10)
                print("Moshi server stopped gracefully")
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                print("Force killing Moshi server...")
                if hasattr(os, 'killpg'):
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

class FullDuplexBenchEval(Job):
    def __init__(self, *, fdp_task: str, model):
        self.fdp_data = tk.Path(
            "/home/tt201262/setups/2026-01-speech-llm/projects/Full-Duplex-Bench/v1_v1.5/dataset",
            hash_overwrite="FullDuplexBench-datasets",
        )  # TODO... the dataset is available as a google drive link, so no good way to write a download job?

        self.fdp_task = fdp_task  # TODO test if task is valid...
        self.model = model
        
        self.out_audios = self.output_path("audios", directory=True)
        self.out_eval = self.output_path("evaluation_output.txt")

        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 4,
            "time": 4,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        # check if dataset is valid
        assert os.path.exists(
            os.path.join(self.fdp_data, "v1.0/candor_pause_handling/1/pause.json")
        ), f"Dataset not found at {self.fdp_data}"

        # ln -s ../projects/Full-Duplex-Bench/v1_v1.5 fdp_v1_v15
        from fdp_v1_v15.model_inference.moshi.inference import _ws_url, MoshiFileClient

        with self.model():
            url = _ws_url("localhost")
            
            # Run inference on all files
            for task, inp in fdp_files_for_tasks(Path(self.fdp_data.get_path()), [self.fdp_task]):
                ind = inp.parent.name  # e.g. "1" in "v1.0/candor_pause_handling/1/input.wav"

                out = Path(self.out_audios.get_path()) / str(ind) / "output.wav" 
                out.parent.mkdir(parents=True, exist_ok=True)
                print("[RUN]", task, inp)
                MoshiFileClient(url, inp, out).run()
            

        # pys v1_v1.5/get_transcript/asr.py --root_dir v1_v1.5/dataset/v1.0/icc_backchannel --task default
        from fdp_v1_v15.get_transcript.asr import get_time_aligned_transcription

        task_map = {
            "icc_backchannel": "backchannel"
        }

        # this takes output.wav and puts output.json in the same folder
        # TODO: for user_interruption, also needs interrupt.json!!!
        assert self.fdp_task != "user_interruption"
        get_time_aligned_transcription(self.out_audios.get_path(), task_map.get(self.fdp_task, self.fdp_task))

        # pys evaluate.py --task backchannel --root_dir ../dataset/v1.0/icc_backchannel
        from fdp_v1_v15.evaluation.evaluate import main as evaluate_main
        # hacky...
        sys.argv = ["evaluate.py", "--task", task_map.get(self.fdp_task, self.fdp_task), "--root_dir", self.out_audios.get_path()]
        # also now record stdout into a file (but still also regular stdout) in self.output_path("evaluation_output.txt")
        with open(self.out_eval, "w", encoding="utf-8") as f:
            # redirect stdout to both console and file
            tee = Tee(sys.stdout, f)
            sys.stdout = tee

            evaluate_main()
            sys.stdout = sys.__stdout__  # restore original stdout
        