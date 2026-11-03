"""Guard: the Parakeet reward worker survives the allocator handing its cache back to the driver (2026-11-03).

MT v3 (RLFinetuneV2.caKAG7FoO87k) died at its first 25 s long-window step with ``ASR worker parakeet: illegal memory
access``. Reproduced on the test node: NeMo's CUDA-graph TDT decoder (``use_cuda_graph_decoder``, NeMo's default) reads
freed memory once cached blocks go back to the driver -- ``torch.cuda.empty_cache()``, or the allocator's own release
when a cudaMalloc fails, as on the judge card the worker shares with vLLM. ``parakeet_worker.ParakeetAsr`` now decodes
with the loop decoder.

Drives the real ``ParakeetAsr`` (real checkpoint, ``transcribe_words_many``) in a child process per mode, since an
illegal access poisons the CUDA context: a 32 x 10 s batch, ``empty_cache``, a 32 x 25 s batch, a 32 x 10 s batch.

  [1] the worker as shipped: no crash, and its decoding config has the CUDA-graph decoder off
  [2] CAN FAIL: the same with the graph decoder switched back on crashes (illegal memory access)

Run from the setup root on a GPU, in the NeMo venv (~2 min):
    CUDA_HOME=/usr <fdb_asr_venv python> recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_parakeet_worker_cache.py
"""

import importlib.util
import os
import subprocess
import sys

SETUP = os.getcwd()
WORKER = os.path.join(SETUP, "recipe/speech_llm/full_duplex/moshi_family/rl/parakeet_worker.py")
MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def child(mode: str) -> None:
    import numpy as np
    import torch
    from omegaconf import open_dict

    spec = importlib.util.spec_from_file_location("pw", WORKER)
    pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    asr = pw.ParakeetAsr(MODEL, "cuda:0")
    graph = bool(asr.model.cfg.decoding.greedy.get("use_cuda_graph_decoder"))
    if mode == "mutant":
        for d in (asr.dec_conf, asr.dec_plain):
            with open_dict(d):
                d.greedy.use_cuda_graph_decoder = True
        asr.model.change_decoding_strategy(asr.dec_conf, verbose=False)
    rng = np.random.default_rng(0)

    def clips(n, sec):  # speech-band noise with pauses: the decoder runs its full loop either way
        out = []
        for _ in range(n):
            x = rng.standard_normal(int(24000 * sec)).astype(np.float32) * 0.1
            x[: int(len(x) * rng.uniform(0, 0.3))] *= 0.01
            out.append(x)
        return out

    asr.transcribe_words_many(clips(32, 10), 24000)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    asr.transcribe_words_many(clips(32, 25), 24000)
    asr.transcribe_words_many(clips(32, 10), 24000)
    torch.cuda.synchronize()
    print(f"CHILD_OK graph_decoder_at_init={graph}", flush=True)


def run(mode: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, __file__, "--child", mode], capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


if __name__ == "__main__" and sys.argv[1:2] == ["--child"]:
    child(sys.argv[2])
    sys.exit(0)

fails = []
rc, out = run("shipped")
ok = rc == 0 and "CHILD_OK graph_decoder_at_init=False" in out
print(
    f"[{'ok' if ok else 'FAIL'}] [1] shipped worker: 10 s / empty_cache / 25 s / 10 s batches decode, graph decoder off"
)
if not ok:
    fails.append("shipped")
    print(out[-3000:])
rc, out = run("mutant")
ok = rc != 0 and "illegal memory access" in out
print(f"[{'ok' if ok else 'FAIL'}] [2] mutant (graph decoder on) crashes with an illegal memory access (rc {rc})")
if not ok:
    fails.append("mutant not detected")
    print(out[-2000:])
if fails:
    sys.exit(f"FAILED: {fails}")
print("\nall Parakeet worker cache checks passed")
