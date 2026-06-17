"""Shared model-access harness for both benchmarks (knowledge + Full-Duplex-Bench).

Both benchmarks reduce to the *same* operation: given a list of
``(input_wav -> output_wav)`` pairs and a backend, produce one reply wav per
input -- either by

* **streaming** each clip through a realtime server (a ``server`` context manager +
  a ``FileClient`` honouring ``(url, inp, out, *, lead_in_s, capture_s).run()``), or
* running an **offline** driver script once over the whole set.

This module owns that one access pattern so ``MoshiInference`` (knowledge) and
``FullDuplexBenchEval_Inference`` (FDB) share identical model-access code; the jobs
differ only in how they *enumerate* the pairs and what they do downstream (ASR,
grading, ...). These are **pure runtime helpers** -- nothing here is hashed by
Sisyphus, so routing both jobs through this module does not change any job hash.

Backends plug in via the same callables a ``BackendSpec`` carries (``server``,
``file_client``, ``ws_url``, ``offline_script``); see ``speech_backends.py``. A cloud
realtime API (Gemini / OpenAI) is just a streaming backend whose ``server`` launches
nothing local and whose ``file_client`` talks to the remote websocket.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class StreamOptions:
    """Knobs for the streaming loop. Defaults match the knowledge benchmark; FDB
    flips ``resume``/``copy_sidecars``/``length_check`` on and uses the client's
    legacy capture mode (``lead_in_s=0``, ``capture_s=None``)."""

    lead_in_s: float = 0.0
    capture_s: float | None = None
    retries: int = 3
    retry_sleep_s: float = 0.0
    resume: bool = False  # skip a clip whose output.wav already exists (non-empty)
    copy_sidecars: bool = False  # copy inp.parent/*.json next to each output
    length_check: bool = False  # warn if reply << input (alignment regression guard)
    progress_every: int = 50


def stream_inference(
    *,
    server: Callable,
    file_client: type,
    ws_url: Callable[[str], str],
    items: Sequence[tuple[Path, Path]],
    server_kwargs: dict,
    opts: StreamOptions,
) -> None:
    """Bring up ``server(**server_kwargs)`` once and stream every ``(inp, out)`` pair
    through ``file_client``. Backend-agnostic: the server ctx-mgr takes the uniform
    ``(*, lora_weights, lora_config, python_exe, unmute_llm)`` signature and ignores
    what it does not use; cloud-API backends launch nothing and just yield a handle."""
    n = len(items)
    with server(**server_kwargs) as handle:
        url = ws_url(handle)
        server_name = getattr(server, "__name__", "server")
        print(f"[{server_name}] streaming {n} clips at {url}", flush=True)
        for i, (inp, out) in enumerate(items):
            inp, out = Path(inp), Path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            if opts.resume and out.exists() and out.stat().st_size > 0:
                print("[SKIP]", out.parent.name, inp.name, "(already done)", flush=True)
                continue
            for attempt in range(opts.retries):
                try:
                    file_client(
                        url,
                        inp,
                        out,
                        lead_in_s=opts.lead_in_s,
                        capture_s=opts.capture_s,
                    ).run()
                    break
                except Exception as e:
                    print(f"[RETRY {attempt + 1}/{opts.retries}] {inp.name}: {e}", flush=True)
                    if opts.retry_sleep_s:
                        time.sleep(opts.retry_sleep_s)
            if opts.length_check:
                _warn_if_truncated(inp, out)
            if opts.copy_sidecars:
                for json_file in inp.parent.glob("*.json"):
                    shutil.copy(json_file, out.parent / json_file.name)
            if opts.progress_every and (i + 1) % opts.progress_every == 0:
                print(f"[inference] {i + 1}/{n} clips done", flush=True)


def _warn_if_truncated(inp: Path, out: Path) -> None:
    """A served full-duplex reply must span (roughly) the whole conversation timeline,
    not just the spoken segment -- otherwise FDB latency is measured from the wrong
    zero (the Unmute output-alignment bug). Warn loudly if the reply is far short."""
    try:
        import soundfile as _sf

        olen = _sf.info(str(out)).frames / _sf.info(str(out)).samplerate
        ilen = _sf.info(str(inp)).frames / _sf.info(str(inp)).samplerate
        if olen < 0.5 * ilen:
            print(
                f"[WARN] {out.parent.name}: reply {olen:.2f}s << input {ilen:.2f}s -- output may be "
                "truncated to the spoken segment (breaks latency alignment)",
                flush=True,
            )
    except Exception as e:
        print(f"[WARN] could not length-check {out}: {e}", flush=True)


def write_pair_manifest(items: Sequence[tuple[Path, Path]], *, copy_sidecars: bool = False) -> str:
    """Write a JSON manifest of ``[[in_wav, out_wav], ...]`` for an offline driver and
    (optionally) copy each input's sidecar ``*.json`` next to its output. Returns the
    manifest path (in the job's cwd)."""
    pairs = []
    for inp, out in items:
        inp, out = Path(inp), Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pairs.append([str(inp), str(out)])
        if copy_sidecars:
            for json_file in inp.parent.glob("*.json"):
                shutil.copy(json_file, out.parent / json_file.name)
    manifest = os.path.join(os.getcwd(), "offline_manifest.json")
    with open(manifest, "w") as f:
        json.dump(pairs, f)
    return manifest


def run_offline_driver(
    *,
    python_exe: str,
    script_path: str,
    in_dir: str | None = None,
    out_dir: str | None = None,
    manifest: str | None = None,
    lead_in_s: float | None = None,
    capture_s: float | None = None,
    batch_size: int | None = None,
    shard: int | None = None,
    num_shards: int | None = None,
    lora_weights: str | None = None,
    lora_config: str | None = None,
    extra_args: Sequence[str] = (),
) -> None:
    """Build + run an offline inference driver as a subprocess. Supports both the
    knowledge dir-mode (``--in_dir/--out_dir/--batch_size/--shard``) and the FDB
    manifest-mode (``--manifest``); the caller passes whichever set its driver wants.
    argparse is order-independent, so the two callers share one builder."""
    cmd = [python_exe, str(script_path)]
    if manifest is not None:
        cmd += ["--manifest", manifest]
    if in_dir is not None:
        cmd += ["--in_dir", str(in_dir)]
    if out_dir is not None:
        cmd += ["--out_dir", str(out_dir)]
    if lead_in_s is not None:
        cmd += ["--lead_in_s", str(lead_in_s)]
    if capture_s is not None:
        cmd += ["--capture_s", str(capture_s)]
    if batch_size is not None:
        cmd += ["--batch_size", str(batch_size)]
    if lora_weights is not None:
        cmd += ["--lora_weights", str(lora_weights)]
        if lora_config is not None:
            cmd += ["--lora_config", str(lora_config)]
    if shard is not None and num_shards is not None:
        cmd += ["--shard", str(shard), "--num_shards", str(num_shards)]
    cmd += list(extra_args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("[offline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
