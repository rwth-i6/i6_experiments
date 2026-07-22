"""Offline MoshiRAG inference driver for the benchmarks.

MoshiRAG (kyutai-labs/moshi-rag, arXiv 2604.12928) is a full-duplex Moshi variant that emits a
``<ret>`` token to trigger a text-in/text-out retrieval LLM and folds the returned reference into its
reply via an ARC-Encoder conditioner. This driver runs inference through **our own engine**
(``moshirag_engine.py``) -- a synchronous, single-clip pump over the fork's model components -- instead
of the fork's batched ``moshi.run_inference`` server, whose feed<->step<->output gating deadlocks on a
batch tail (the 193/200 strand; see ``projects/2026-01-speech-llm/moshirag.md``). The engine cannot
wedge: there is no shared queue / ``step_index`` coupling, so every clip completes and we **assert** it,
rather than salvaging missing clips with silence as the old fork-driver path had to.

It still needs two services the engine talks to (kept here, unchanged):
* a **reference-encoder server** (``moshi.server_conditioner``, ``reference_with_time``) at
  ``REFERENCE_ENCODER_URL`` -- co-launched here on the driver's GPU; the engine's ``load_models`` sees
  that env var and therefore skips the in-model reference conditioner (the ARC encode is remote), and
* a **retrieval LLM** at ``LLM_BASE_URL`` / ``LLM_MODEL_NAME`` (OpenAI-compatible, our vLLM Gemma,
  passed by the job via ``--llm_base_url``); or ``--use_gt_reference`` reads a gold ``gt_reference_text``
  sidecar and skips the retriever.

This file is run with the moshirag venv python, so ``import moshi`` (inside the engine) resolves to the
fork in site-packages; it lives under ``dorian_koch/`` (no ``speech_llm`` on sys.path) so that holds.

Two modes (same driver):
* **knowledge** -- ``--in_dir``/``--out_dir`` (wavs ``<i>.wav``), ``--shard``/``--num_shards``; the engine
  pads each input with ``--lead_in_s`` + ``--capture_s`` silence so the spoken answer lands, and writes
  the reply to ``--out_dir/<i>.wav``.
* **FDB** -- ``--manifest`` (JSON ``[[in_wav, out_wav], ...]``); each clip is processed as-is (+ a short
  tail) and the reply written to its ``out_wav`` (``.../<ind>/output.wav``), trace alongside.

GATED: stage ``kyutai/moshika-rag-pytorch-bf16`` (CC-BY) into the HF cache first; see moshirag.md.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_REPO = "kyutai/moshika-rag-pytorch-bf16"
_REF_ENCODER_ENV = "REFERENCE_ENCODER_URL"
# Ungated mirror of the gated meta-llama/Llama-3.2-3B-Instruct (the ARC-Encoder's text tokenizer).
# Byte-identical token ids; see _materialize_arc_tokenizer.
ARC_TOKENIZER_MIRROR = "unsloth/Llama-3.2-3B-Instruct"


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

    The ARC encoder tokenizes reference text with ``AutoTokenizer.from_pretrained(name, use_fast=False)``.
    ``meta-llama/Llama-3.2-3B-Instruct`` is gated; the ungated mirror's tokenizer_config declares the
    slow ``PreTrainedTokenizer`` and Llama-3.2 ships no slow SentencePiece model, so ``use_fast=False``
    cannot build it. Re-save the *fast* tokenizer and mark it ``PreTrainedTokenizerFast`` so
    ``use_fast=False`` resolves to it -- byte-identical ids to meta-llama (which itself declares the fast
    class). Reads the pre-staged HF cache, so it works offline on compute nodes.
    """
    from transformers import PreTrainedTokenizerFast

    ft = PreTrainedTokenizerFast.from_pretrained(ARC_TOKENIZER_MIRROR)
    ft.save_pretrained(str(dest))
    cfg_path = dest / "tokenizer_config.json"
    tc = json.loads(cfg_path.read_text())
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg_path.write_text(json.dumps(tc))


def _patched_conditioner_config(repo: str, tokenizer_dir: Path, dest: Path) -> Path:
    """Copy the model's config.json and pin the ARC tokenizer to our local ungated dir.

    The released config omits ``tokenizer_name`` (so the fork falls back to the gated meta-llama
    default); inject our local dir instead. server_conditioner loads ``--config`` via ``loaders.hf_get``
    which accepts a local path.
    """
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(repo, "config.json")
    cfg = json.loads(Path(src).read_text())
    arc = cfg["conditioners"]["reference_with_time"]["multi_arc_encoder"]
    arc["tokenizer_name"] = str(tokenizer_dir)
    dest.write_text(json.dumps(cfg))
    return dest


def _start_reference_encoder(repo: str) -> tuple[subprocess.Popen, str]:
    """Co-launch the ARC-Encoder reference conditioner server (on the driver's visible GPU)."""
    port = _free_port()
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
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    p = argparse.ArgumentParser(description="Offline MoshiRAG inference")
    p.add_argument("--in_dir", help="Dir of question wavs named <i>.wav (knowledge mode)")
    p.add_argument("--out_dir", help="Output dir for reply wavs (knowledge mode)")
    p.add_argument("--manifest", help="JSON list of [in_wav, out_wav] pairs (FDB mode)")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--lead_in_s", type=float, default=2.0, help="Silence prepended (knowledge mode).")
    p.add_argument("--capture_s", type=float, default=24.0, help="Trailing silence to capture the answer (knowledge).")
    p.add_argument("--tail_s", type=float, default=2.0, help="Extra trailing silence to flush the model's last word.")
    p.add_argument("--batch_size", type=int, default=1, help="Accepted for CLI parity; the engine is single-clip.")
    p.add_argument("--hf_repo", type=str, default=DEFAULT_REPO)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--rag_timeout", type=float, default=10.0)
    p.add_argument("--max_reference_tokens", type=int, default=64)
    # Retrieval backend: an OpenAI-compatible LLM url (our vLLM Gemma, passed by the job), or
    # --use_gt_reference to read a gold gt_reference_text sidecar and skip the retriever.
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
        raise ValueError("MoshiRAG retrieval needs --llm_base_url (the vLLM retrieval backend) or --use_gt_reference.")

    # Build (input_wav -> output_wav) pairs. The engine reads each input and writes its reply directly,
    # so no input staging is needed (unlike the old fork driver, which globbed a dir).
    if args.manifest:
        pairs = [tuple(x) for x in json.loads(Path(args.manifest).read_text())]
        lead_in_s, capture_s = 0.0, 0.0  # FDB: clips are pre-timed; process as-is (+ tail).
    else:
        assert args.in_dir and args.out_dir, "knowledge mode needs --in_dir and --out_dir"
        in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(in_dir.glob("*.wav"), key=lambda q: int(q.stem))
        if args.shard is not None and args.num_shards is not None:
            wavs = wavs[args.shard :: args.num_shards]
        pairs = [(str(w), str(out_dir / w.name)) for w in wavs]
        lead_in_s, capture_s = args.lead_in_s, args.capture_s

    if not pairs:
        print("[moshirag] no inputs; nothing to do", flush=True)
        return

    # Retrieval LLM env (read by the engine's reference generator) -- set on os.environ since the
    # engine runs in-process.
    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url
        if args.llm_model_name:
            os.environ["LLM_MODEL_NAME"] = args.llm_model_name
        # The fork's openai client refuses to init without a key; vLLM ignores the value.
        os.environ.setdefault("LLM_API_KEY", os.environ.get("LLM_API_KEY", "EMPTY"))

    # Optional gt-reference sidecars (skip the retriever): map in_wav -> gt_reference_text.
    gt_reference_for: dict[str, str] = {}
    if args.use_gt_reference:
        for inp, _ in pairs:
            sc = Path(inp).with_suffix(".json")
            if sc.exists():
                try:
                    gt_reference_for[inp] = json.loads(sc.read_text()).get("gt_reference_text", "")
                except (OSError, json.JSONDecodeError):
                    pass

    ref_proc = None
    try:
        ref_proc, ref_url = _start_reference_encoder(args.hf_repo)
        os.environ[_REF_ENCODER_ENV] = ref_url  # must be set before the engine's load_models()

        # Import the engine only now -- it imports moshi (the fork) + reads REFERENCE_ENCODER_URL.
        import moshirag_engine as eng

        print(f"[moshirag] loading {args.hf_repo} ({len(pairs)} clips)", flush=True)
        model = eng.MoshiRagModel(
            args.hf_repo,
            device=f"{args.device}:0" if ":" not in args.device else args.device,
            rag_timeout=args.rag_timeout,
            max_reference_tokens=args.max_reference_tokens,
        )
        n_done = asyncio.run(
            eng.run_pairs(
                model,
                pairs,
                lead_in_s=lead_in_s,
                capture_s=capture_s,
                tail_s=args.tail_s,
                gt_reference_for=gt_reference_for,
            )
        )

        # Every clip must have produced a reply wav -- no silent salvage. The synchronous engine cannot
        # deadlock, so a missing output is a real error, not a tolerable tail loss.
        missing = [out for _, out in pairs if not (Path(out).exists() and Path(out).stat().st_size > 0)]
        assert not missing, f"{len(missing)} clip(s) produced no output (first: {missing[:3]})"
        print(f"[moshirag] finished {n_done}/{len(pairs)} clips (all produced output)", flush=True)
    finally:
        if ref_proc is not None and ref_proc.poll() is None:
            _terminate(ref_proc)


if __name__ == "__main__":
    main()
