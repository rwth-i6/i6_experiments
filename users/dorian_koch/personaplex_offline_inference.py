"""Offline PersonaPlex inference driver for the benchmarks.

Thin CLI over `speech_llm/personaplex_engine.py` (the science layer). Lives directly under
`dorian_koch/` -- like `moshi_offline_inference.py` -- so a bare `import moshi` resolves to
PersonaPlex's fork in the active venv, not the sibling `speech_llm/` package.

Two modes, matching how the benchmark jobs invoke it (same driver serves both):
* **knowledge** -- `--in_dir`/`--out_dir` (named `<i>.wav`), `--shard`/`--num_shards`; each
  input is padded with `--lead_in_s` + `--capture_s` silence so PersonaPlex's
  output (trimmed to input length) captures the full spoken reply. This is the CLI
  `MoshiInference._run_offline` emits.
* **FDB** -- `--manifest` (JSON list of `[input_wav, output_wav]`); processed as-is. This is
  the CLI `FullDuplexBenchEval_Inference._run_offline` emits.

PersonaPlex inference is inherently single-stream (LMGen `streaming_forever(1)`), so the
shared `--batch_size` is accepted-and-ignored. The model download (`nvidia/personaplex-7b-v1`
+ voices.tgz) is gated on HF_TOKEN, which Sisyphus injects into the job env; see
`projects/2026-01-speech-llm/personaplex.md`.
"""

import argparse
import importlib.util
import json
from pathlib import Path

# Load the engine straight from its file, WITHOUT putting speech_llm/ on sys.path: that dir
# holds our recipe `moshi.py`, which would shadow the PersonaPlex fork's `moshi` package and
# break `import moshi` inside the engine (it imports i6_experiments, absent in this venv).
# The engine has no speech_llm-relative imports, so a standalone file load is safe.
_engine_path = Path(__file__).resolve().parent / "speech_llm" / "personaplex_engine.py"
_spec = importlib.util.spec_from_file_location("personaplex_engine", _engine_path)
ppx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppx)

DEFAULT_REPO = "nvidia/personaplex-7b-v1"


def main():
    p = argparse.ArgumentParser(description="Offline PersonaPlex inference")
    p.add_argument("--in_dir", help="Dir of question wavs named <i>.wav (knowledge mode)")
    p.add_argument("--out_dir", help="Output dir for reply wavs (knowledge mode)")
    p.add_argument("--manifest", help="JSON list of [in_wav, out_wav] pairs (FDB mode)")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--lead_in_s", type=float, default=2.0)
    p.add_argument("--capture_s", type=float, default=24.0)
    p.add_argument("--batch_size", type=int, default=1, help="Accepted for CLI parity; ignored.")
    p.add_argument("--hf_repo", type=str, default=DEFAULT_REPO)
    p.add_argument("--voice", type=str, default=ppx.DEFAULT_VOICE)
    p.add_argument("--system_prompt", type=str, default=ppx.DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    # PersonaPlex offline has no LoRA path yet (no released finetune); fail loudly if asked.
    p.add_argument("--lora_weights", type=str, default=None)
    p.add_argument("--lora_config", type=str, default=None)
    args = p.parse_args()

    if args.lora_weights is not None:
        raise NotImplementedError("PersonaPlex LoRA inference is not wired (no released finetune); see personaplex.md.")

    if args.manifest:
        pairs = [tuple(x) for x in json.loads(Path(args.manifest).read_text())]
        lead_in_s, capture_s = 0.0, 0.0  # FDB: process clips as-is (pre-timed)
    else:
        assert args.in_dir and args.out_dir, "knowledge mode needs --in_dir and --out_dir"
        in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wavs = sorted(in_dir.glob("*.wav"), key=lambda q: int(q.stem))
        if args.shard is not None and args.num_shards is not None:
            wavs = wavs[args.shard :: args.num_shards]
        pairs = [(str(w), str(out_dir / w.name)) for w in wavs]
        lead_in_s, capture_s = args.lead_in_s, args.capture_s

    print(f"[personaplex] loading {args.hf_repo} ({len(pairs)} clips)", flush=True)
    model = ppx.PersonaPlexModel(args.hf_repo, device=args.device, cpu_offload=args.cpu_offload)
    ppx.run_pairs(
        model,
        pairs,
        hf_repo=args.hf_repo,
        system_prompt=args.system_prompt,
        voice=args.voice,
        lead_in_s=lead_in_s,
        capture_s=capture_s,
    )
    print(f"[personaplex] finished {len(pairs)} clips", flush=True)


if __name__ == "__main__":
    main()
