"""Offline MoshiRAG inference driver for the benchmarks.

MoshiRAG (kyutai-labs/moshi-rag, arXiv 2604.12928) is a full-duplex Moshi variant that
emits a ``<ret>`` token to *asynchronously* trigger a text-in/text-out retrieval backend
and folds the returned reference into its reply via an ARC-Encoder conditioner. The fork
ships a batch wav->wav entry point, ``python -m moshi.run_inference`` (writes one
``<stem>.wav`` reply + a ``<stem>.json`` trace per input wav), which needs:

* a **reference-encoder server** (``moshi.server_conditioner``, conditioner
  ``reference_with_time``) reachable at ``REFERENCE_ENCODER_URL`` -- co-launched here, and
* a **retrieval LLM** at ``LLM_BASE_URL`` / ``LLM_MODEL_NAME`` (OpenAI-compatible) -- the
  benchmark job brings up our existing vLLM Gemma and passes its url in via
  ``--llm_base_url`` (see ``MoshiInference._run_offline``); or ``--use_gt_reference`` skips
  retrieval and reads a gold ``gt_reference_text`` sidecar instead.

This file is a thin *adapter* over the fork's CLI -- it never ``import moshi`` itself (it
shells out with ``sys.executable``, the moshirag venv python), so it lives directly under
``dorian_koch/`` like ``moshi_offline_inference.py`` without the package-shadowing dance.

Two invocation modes, matching the benchmark jobs (same driver serves both):
* **knowledge** -- ``--in_dir``/``--out_dir`` (wavs named ``<i>.wav``), ``--shard``/
  ``--num_shards``; reply wavs land in ``--out_dir`` with the same names.
* **FDB** -- ``--manifest`` (JSON ``[[in_wav, out_wav], ...]``); each reply is written to its
  ``out_wav`` (``.../<ind>/output.wav``) with the trace alongside as ``output.json``.

GATED: the model (``kyutai/moshika-rag-pytorch-bf16``, CC-BY) must be staged into the HF
cache first; see ``projects/2026-01-speech-llm/moshirag.md``. ``lead_in_s``/``capture_s``
are accepted for CLI parity but ignored -- the fork does its own VAD turn-taking, so the
reply wav already spans the full conversation timeline (satisfies the harness length-check).
``batch_size`` is forwarded to the fork.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_REPO = "kyutai/moshika-rag-pytorch-bf16"
# The fork reads the reference-encoder endpoint from this env var (get_reference_encoder_url()).
_REF_ENCODER_ENV = "REFERENCE_ENCODER_URL"
# Ungated mirror of the gated meta-llama/Llama-3.2-3B-Instruct (the ARC-Encoder's text
# tokenizer). Byte-identical token ids; see _materialize_arc_tokenizer.
ARC_TOKENIZER_MIRROR = "unsloth/Llama-3.2-3B-Instruct"
# The fork's run_inference shares its GPU with the co-resident reference encoder (~23 GB),
# and lm_gen pre-allocates a KV cache sized to batch_size. batch 32 OOMs a 94 GB GPU, so cap
# the fork batch here (the benchmark passes a generic 32). Tune up if headroom allows.
MOSHIRAG_FORK_MAX_BATCH = 8
# We run the fork's batch inference through moshirag_run_inference_patched.py, which gives the
# retrieval OpenAI client a finite request timeout so an abandoned worker thread returns instead
# of blocking asyncio.run's shutdown-executor join forever (see that file for the full mechanism
# of the fork's exit-hang). The watchdog below is now only a backstop: poll the output dir and,
# if the process self-exits, return; if it ever finishes all clips but lingers, force-kill after
# _FORK_EXIT_GRACE_S; if it makes no progress for _FORK_STALL_S, fail loudly (a genuine hang).
_FORK_RUN_WRAPPER = "moshirag_run_inference_patched.py"
_FORK_POLL_S = 10.0
_FORK_EXIT_GRACE_S = 120.0
_FORK_STALL_S = 1800.0
# Max clips we'll salvage with silent outputs when the fork hits its tail deadlock (a few clips on the
# final partial batch wedge in a circular feeder<->step-loop wait); a larger shortfall = real failure.
_MAX_TAIL_SALVAGE = 16


def _write_silence_wav(path: Path, *, seconds: float = 0.5, sr: int = 24000) -> None:
    """Write a short mono int16 silent WAV (stdlib only). Salvages clips the fork never answered
    (tail deadlock) so downstream transcription/grading still has a file -- it scores as an empty
    answer, the honest outcome for a clip the model never produced."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, proc: subprocess.Popen, *, what: str, timeout: float = 600.0) -> None:
    """Block until ``port`` accepts connections, failing loudly if the process dies first."""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} died during startup (exit {proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                print(f"[moshirag] {what} ready on port {port}", flush=True)
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{what} not ready after {timeout}s")


def _materialize_arc_tokenizer(dest: Path) -> None:
    """Build a local Llama-3.2 tokenizer dir the fork's ArcEncoderTokenizer can load.

    The ARC encoder tokenizes reference text with
    ``AutoTokenizer.from_pretrained(name, use_fast=False)``. The original
    ``meta-llama/Llama-3.2-3B-Instruct`` is gated; the ungated mirror's
    tokenizer_config declares ``tokenizer_class="PreTrainedTokenizer"`` (slow), and
    Llama-3.2 ships no slow SentencePiece model, so ``use_fast=False`` cannot build a
    slow tokenizer (transformers 4.55.3 silently returns ``False``). We re-save the
    *fast* tokenizer and mark it ``PreTrainedTokenizerFast`` so ``use_fast=False``
    resolves to the fast tokenizer -- byte-identical ids to meta-llama, which itself
    declares ``PreTrainedTokenizerFast`` (so this reproduces kyutai's setup). Reads from
    the HF cache (pre-staged), so it works offline on compute nodes.
    """
    from transformers import PreTrainedTokenizerFast

    ft = PreTrainedTokenizerFast.from_pretrained(ARC_TOKENIZER_MIRROR)
    ft.save_pretrained(str(dest))
    cfg_path = dest / "tokenizer_config.json"
    tc = json.loads(cfg_path.read_text())
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg_path.write_text(json.dumps(tc))


def _patched_conditioner_config(repo: str, tokenizer_dir: Path, dest: Path) -> Path:
    """Copy the model's config.json and pin the ARC tokenizer to our local dir.

    The released config omits ``tokenizer_name`` (so the fork falls back to the gated
    meta-llama default); we inject our local ungated tokenizer dir instead.
    server_conditioner loads ``--config`` via ``loaders.hf_get`` which accepts a local
    path, so we hand it this patched file while ``--moshi-weight`` stays the hf:// URI
    (read from the staged cache).
    """
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(repo, "config.json")
    cfg = json.loads(Path(src).read_text())
    arc = cfg["conditioners"]["reference_with_time"]["multi_arc_encoder"]
    arc["tokenizer_name"] = str(tokenizer_dir)
    dest.write_text(json.dumps(cfg))
    return dest


def _start_reference_encoder(repo: str, device: str) -> tuple[subprocess.Popen, str]:
    """Co-launch the ARC-Encoder reference conditioner server and return (proc, url)."""
    port = _free_port()
    # Materialize the ungated tokenizer + a config.json that points at it (kept for the
    # life of the server; tiny, cleaned with the job's tmp at exit).
    work = Path(tempfile.mkdtemp(prefix="moshirag_refenc_"))
    tok_dir = work / "arc_tokenizer"
    _materialize_arc_tokenizer(tok_dir)
    cfg_file = _patched_conditioner_config(repo, tok_dir, work / "config.json")
    cmd = [
        sys.executable,
        "-m",
        "moshi.server_conditioner",
        "--config",
        str(cfg_file),
        "--moshi-weight",
        f"hf://{repo}/model.safetensors",
        "--conditioner",
        "reference_with_time",
        "--cuda-device",
        "0",  # the driver's (single) visible GPU; see job-side pinning
        "--port",
        str(port),
    ]
    print("[moshirag] starting reference encoder:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    _wait_for_port(port, proc, what="reference encoder")
    return proc, f"http://127.0.0.1:{port}"


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort terminate then kill a subprocess that won't exit on its own."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run_fork_inference(
    in_dir: Path,
    out_dir: Path,
    *,
    repo: str,
    device: str,
    batch_size: int,
    use_gt_reference: bool,
    env: dict,
    n_expected: int,
) -> None:
    """Run the fork's batched inference via the timeout-hardened wrapper, with a stall backstop.

    moshirag_run_inference_patched.py fixes the fork's exit-hang at the cause (bounded OpenAI
    request timeout), so the process now self-exits and proc.poll() returns normally. The poll
    loop here is a backstop: a non-zero exit with fewer than ``n_expected`` traces is a real
    failure; if the fork ever finishes all clips but lingers, force-kill after _FORK_EXIT_GRACE_S;
    no new trace for _FORK_STALL_S is a genuine hang and fails loudly. The post-run wav mapping in
    main() still asserts every clip produced audio, so partial results can never ship silently.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = str(Path(__file__).with_name(_FORK_RUN_WRAPPER))
    cmd = [
        sys.executable,
        wrapper,
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_dir),
        "--hf-repo",
        repo,
        "--device",
        f"{device}:0" if ":" not in device else device,
        "--batch-size",
        str(batch_size),
        # Paper/README setting so the model stops a reasonable time after the user's turn.
        "--max-consecutive-silence-frames",
        "40",
    ]
    if use_gt_reference:
        cmd.append("--use-gt-reference")
    print("[moshirag] run_inference:", " ".join(cmd), flush=True)

    def _n_traces() -> int:
        return len(list(out_dir.glob("*.json")))

    proc = subprocess.Popen(cmd, env={**env, "PYTHONUNBUFFERED": "1"})
    done_at = None
    last_n = 0
    last_progress = time.time()
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                n_done = _n_traces()
                # A non-zero exit with most clips done is the fork's tail deadlock (a few clips on the
                # final partial batch wedge). Tolerate it -- the caller salvages the missing clips with
                # silent outputs so the run still completes. Only a large shortfall aborts.
                if rc != 0 and n_done < n_expected - _MAX_TAIL_SALVAGE:
                    raise RuntimeError(f"moshi.run_inference exited {rc} after {n_done}/{n_expected} clips")
                if rc != 0:
                    print(
                        f"[moshirag] fork exited {rc} at {n_done}/{n_expected}; "
                        f"salvaging {n_expected - n_done} wedged tail clip(s) downstream",
                        flush=True,
                    )
                return
            n = _n_traces()
            if n > last_n:
                last_n = n
                last_progress = time.time()
            if n >= n_expected:
                # All clips written; let it exit cleanly, else break the known exit-hang.
                if done_at is None:
                    done_at = time.time()
                elif time.time() - done_at > _FORK_EXIT_GRACE_S:
                    print(
                        f"[moshirag] fork finished all {n_expected} clips but did not exit after "
                        f"{_FORK_EXIT_GRACE_S:.0f}s (known exit-hang); terminating",
                        flush=True,
                    )
                    _terminate(proc)
                    return
            elif time.time() - last_progress > _FORK_STALL_S:
                _terminate(proc)
                raise RuntimeError(
                    f"moshi.run_inference stalled: no new trace for {_FORK_STALL_S:.0f}s at "
                    f"{n}/{n_expected} clips (mid-work hang)"
                )
            time.sleep(_FORK_POLL_S)
    finally:
        if proc.poll() is None:
            _terminate(proc)


def main():
    p = argparse.ArgumentParser(description="Offline MoshiRAG inference")
    p.add_argument("--in_dir", help="Dir of question wavs named <i>.wav (knowledge mode)")
    p.add_argument("--out_dir", help="Output dir for reply wavs (knowledge mode)")
    p.add_argument("--manifest", help="JSON list of [in_wav, out_wav] pairs (FDB mode)")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--lead_in_s", type=float, default=2.0, help="Accepted for CLI parity; ignored (VAD).")
    p.add_argument("--capture_s", type=float, default=24.0, help="Accepted for CLI parity; ignored (VAD).")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--hf_repo", type=str, default=DEFAULT_REPO)
    p.add_argument("--device", type=str, default="cuda")
    # Retrieval backend: an OpenAI-compatible LLM url (our vLLM Gemma, passed by the job),
    # or --use_gt_reference to read a gold gt_reference_text sidecar and skip the retriever.
    p.add_argument("--llm_base_url", type=str, default=None)
    p.add_argument("--llm_model_name", type=str, default=None)
    p.add_argument("--use_gt_reference", action="store_true")
    # MoshiRAG LoRA finetune is scaffolded only (no released training); fail loudly if asked.
    p.add_argument("--lora_weights", type=str, default=None)
    p.add_argument("--lora_config", type=str, default=None)
    args = p.parse_args()

    if args.lora_weights is not None:
        raise NotImplementedError("MoshiRAG LoRA inference is not wired (training is scaffold-only); see moshirag.md.")
    if not args.use_gt_reference and not args.llm_base_url:
        raise ValueError(
            "MoshiRAG retrieval needs --llm_base_url (the vLLM retrieval backend) or "
            "--use_gt_reference. The benchmark job supplies the url via vllm_server."
        )

    # Build the (input_wav -> output_wav) pairs and a stem->output map. The fork globs an
    # input dir and writes <stem>.wav, so we stage a temp dir of uniquely-named symlinks and
    # map the results back to the requested output paths.
    # FDB output path is .../<ind>/output.wav -> stem is the <ind> dir name; knowledge mode
    # names outputs <i>.wav -> stem is the file stem.
    def _stem_manifest(out):
        return Path(out).parent.name

    def _stem_dir(out):
        return Path(out).stem

    if args.manifest:
        pairs = [tuple(x) for x in json.loads(Path(args.manifest).read_text())]
        stem_of = _stem_manifest
    else:
        assert args.in_dir and args.out_dir, "knowledge mode needs --in_dir and --out_dir"
        in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(in_dir.glob("*.wav"), key=lambda q: int(q.stem))
        if args.shard is not None and args.num_shards is not None:
            wavs = wavs[args.shard :: args.num_shards]
        pairs = [(str(w), str(out_dir / w.name)) for w in wavs]
        stem_of = _stem_dir

    if not pairs:
        print("[moshirag] no inputs; nothing to do", flush=True)
        return

    env = os.environ.copy()
    if args.llm_base_url:
        env["LLM_BASE_URL"] = args.llm_base_url
        if args.llm_model_name:
            env["LLM_MODEL_NAME"] = args.llm_model_name
        # The fork's llm/client.py builds an openai.OpenAI client which refuses to init
        # without an api_key; vLLM ignores the value, so pass a dummy unless one is set.
        env.setdefault("LLM_API_KEY", os.environ.get("LLM_API_KEY", "EMPTY"))

    staging = Path(tempfile.mkdtemp(prefix="moshirag_in_"))
    fork_out = Path(tempfile.mkdtemp(prefix="moshirag_out_"))
    ref_proc = None
    try:
        # Stage uniquely-named symlinks; remember stem -> requested output wav.
        out_for_stem: dict[str, str] = {}
        for inp, out in pairs:
            stem = stem_of(out)
            out_for_stem[stem] = out
            link = staging / f"{stem}.wav"
            if link.exists():
                link.unlink()
            link.symlink_to(Path(inp).resolve())
            # For --use_gt_reference the fork reads <stem>.json's gt_reference_text; if a
            # sidecar exists next to the input, carry it over so retrieval can be skipped.
            src_json = Path(inp).with_suffix(".json")
            if args.use_gt_reference and src_json.exists():
                shutil.copy(src_json, staging / f"{stem}.json")

        ref_proc, ref_url = _start_reference_encoder(args.hf_repo, args.device)
        env[_REF_ENCODER_ENV] = ref_url

        effective_batch = min(args.batch_size, MOSHIRAG_FORK_MAX_BATCH)
        if effective_batch != args.batch_size:
            print(
                f"[moshirag] capping fork batch_size {args.batch_size} -> {effective_batch} "
                "(reference encoder shares the GPU)",
                flush=True,
            )
        print(f"[moshirag] loading {args.hf_repo} ({len(pairs)} clips)", flush=True)
        _run_fork_inference(
            staging,
            fork_out,
            repo=args.hf_repo,
            device=args.device,
            batch_size=effective_batch,
            use_gt_reference=args.use_gt_reference,
            env=env,
            n_expected=len(pairs),
        )

        # Map each produced <stem>.wav (+ trace .json) back to its requested output path. Clips the
        # fork never answered (tail deadlock) get a salvaged silent wav + empty trace so the batch
        # still completes; they score as empty answers (honest -- the model produced nothing).
        n_ok = 0
        n_salvaged = 0
        for stem, out in out_for_stem.items():
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            produced = fork_out / f"{stem}.wav"
            if produced.exists() and produced.stat().st_size > 0:
                shutil.move(str(produced), str(out_path))
            else:
                _write_silence_wav(out_path)
                n_salvaged += 1
            trace = fork_out / f"{stem}.json"
            dst_trace = out_path.parent / ("output.json" if out_path.name == "output.wav" else f"{stem}.json")
            if trace.exists():
                shutil.copy(trace, dst_trace)
            else:
                dst_trace.write_text(
                    json.dumps({"model_text": [], "user_text": [], "_salvaged": True}, indent=2),
                    encoding="utf-8",
                )
            n_ok += 1
        print(f"[moshirag] finished {n_ok}/{len(pairs)} clips ({n_salvaged} salvaged silent)", flush=True)
    finally:
        if ref_proc is not None and ref_proc.poll() is None:
            ref_proc.terminate()
            try:
                ref_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                ref_proc.kill()
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(fork_out, ignore_errors=True)


if __name__ == "__main__":
    main()
