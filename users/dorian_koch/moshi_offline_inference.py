"""Offline batched Moshi inference for the knowledge benchmark (fast path).

Runs Moshi as-fast-as-the-GPU in batches via ``moshi.run_inference.InferenceState`` instead of
the realtime websocket server (which is hardcoded batch_size=1 and paces to ~1x realtime). For
each clip we feed ``[lead_in silence + question + capture_s trailing silence]``; Moshi emits one
output frame per input frame and stops when the input ends, so the captured output is its full
reply (greeting + answer) -- nothing skipped or trimmed. Used by MoshiInference(backend="offline").

Optionally applies a fine-tuned LoRA adapter (``--lora_weights`` + ``--lora_config``) on top of
the base model, so the same harness can benchmark both the base and the fine-tuned Moshi.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sphn
import torch

from moshi.models import loaders
from moshi.run_inference import InferenceState


def main():
    p = argparse.ArgumentParser(description="Offline batched Moshi inference")
    p.add_argument("--in_dir", required=True, help="Directory of question (TTS) wavs named <i>.wav")
    p.add_argument("--out_dir", required=True, help="Output dir for Moshi reply wavs")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--lead_in_s", type=float, default=2.0)
    p.add_argument("--capture_s", type=float, default=60.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hf_repo", type=str, default=loaders.DEFAULT_REPO)
    p.add_argument("--cfg_coef", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to a fine-tuned LoRA adapter (safetensors) to apply on top of the base model.",
    )
    p.add_argument(
        "--lora_config",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint's config.json (supplies lora_rank/lora_scaling).",
    )
    p.add_argument(
        "--tail_check_s",
        type=float,
        default=4.0,
        help="Flag clips still non-silent in the last N seconds (likely truncated by capture_s).",
    )
    args = p.parse_args()

    device = args.device
    print(f"[moshi-offline] loading model {args.hf_repo}", flush=True)
    ckpt = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)
    if args.lora_weights is not None:
        # Apply a fine-tuned LoRA adapter on top of the base model. We mutate the base lm_config
        # (rather than passing the checkpoint's config.json as config_path) so the base model's
        # lm_gen_config / tokenizer / mimi stay identical to the base run -- only the LoRA layers
        # are added, keeping base-vs-finetuned a fair comparison.
        lora_cfg = json.loads(Path(args.lora_config).read_text()) if args.lora_config else {}
        if ckpt.lm_config is None:
            ckpt.lm_config = dict(loaders._lm_kwargs)
        ckpt.lm_config["lora"] = True
        ckpt.lm_config["lora_rank"] = lora_cfg.get("lora_rank", 128)
        ckpt.lm_config["lora_scaling"] = lora_cfg.get("lora_scaling", 2.0)
        ckpt.lora_weights = Path(args.lora_weights)
        print(
            f"[moshi-offline] applying LoRA {args.lora_weights} "
            f"(rank={ckpt.lm_config['lora_rank']}, scaling={ckpt.lm_config['lora_scaling']})",
            flush=True,
        )
    mimi = ckpt.get_mimi(device=device)
    text_tokenizer = ckpt.get_text_tokenizer()
    lm = ckpt.get_moshi(device=device, dtype=torch.bfloat16)
    lm.eval()
    sr = mimi.sample_rate
    print(f"[moshi-offline] model loaded (sample_rate={sr})", flush=True)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"), key=lambda q: int(q.stem))
    if args.shard is not None and args.num_shards is not None:
        wavs = wavs[args.shard :: args.num_shards]
    print(f"[moshi-offline] {len(wavs)} clips, batch_size={args.batch_size}", flush=True)

    lead = np.zeros(int(round(args.lead_in_s * sr)), dtype=np.float32)
    tail = np.zeros(int(round(args.capture_s * sr)), dtype=np.float32)

    def load_input(path):
        a, _ = sphn.read(str(path), sample_rate=sr)  # -> (channels, T) float32 at sr
        a = a[0] if a.ndim > 1 else a
        return np.concatenate([lead, a.astype(np.float32), tail])

    # Create the streaming state once for a fixed batch size; reset_streaming() between batches
    # keeps each batch independent. Re-creating InferenceState would re-enter streaming_forever on
    # the shared mimi and raise "already streaming". Partial last batch is padded up to bs.
    bs = args.batch_size
    state = InferenceState(ckpt, mimi, text_tokenizer, lm, bs, args.cfg_coef, device, **ckpt.lm_gen_config)

    done = 0
    truncated = 0
    tail_n = int(round(args.tail_check_s * sr))
    speech_threshold = 0.02  # float PCM peak above this is speech (Moshi silence ~0.0006)
    for start in range(0, len(wavs), bs):
        batch = wavs[start : start + bs]
        inputs = [load_input(w) for w in batch]
        t_max = max(len(x) for x in inputs)
        arr = np.zeros((bs, 1, t_max), dtype=np.float32)  # pad partial last batch up to bs
        for b, x in enumerate(inputs):
            arr[b, 0, : len(x)] = x
        in_pcms = torch.from_numpy(arr).to(device)

        state.mimi.reset_streaming()
        state.lm_gen.reset_streaming()
        with torch.no_grad():
            out_items = state.run(in_pcms)
        for b, w in enumerate(batch):  # save only the real clips (ignore padding rows)
            out_pcm = out_items[b][1][0].detach().cpu().numpy()  # mono (T_out,)
            sphn.write_wav(str(out_dir / w.name), out_pcm, sr)
            if out_pcm.size >= tail_n and float(np.max(np.abs(out_pcm[-tail_n:]))) >= speech_threshold:
                truncated += 1
                print(
                    f"[moshi-offline] WARN possibly truncated: {w.name} still speaking in last {args.tail_check_s}s",
                    flush=True,
                )
        done += len(batch)
        print(f"[moshi-offline] {done}/{len(wavs)} clips done", flush=True)

    print(
        f"[moshi-offline] finished {done} clips; {truncated} possibly truncated (non-silent in last {args.tail_check_s}s of the {args.capture_s}s capture)",
        flush=True,
    )


if __name__ == "__main__":
    main()
