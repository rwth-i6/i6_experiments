# import torch
# import math
# import time
# import returnn.util.basic as util  # noqa
# from i6_experiments.users.zeyer.external_models.huggingface import (
#     DownloadHuggingFaceRepoJob,
#     get_content_dir_from_hub_cache_dir,
# )
#!/usr/bin/env python3
import os, time, math, argparse
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_dir: str):
    model_path = model_dir  # you can wrap with your get_content_dir_from_hub_cache_dir
    print("Loading model/tokenizer...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer_max_length: {tokenizer.model_max_length}")

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - start_time:.1f}s")
    return model, tokenizer


def build_concat_ids(
    tokenizer,
    prompts: List[str],
    hyps: List[str],
    add_bos_once: bool = True,
) -> Tuple[List[List[int]], List[int]]:
    """
    For each sample: ids = [BOS?] + prompt_ids + hyp_ids
    Returns:
      - list of per-sample token id lists
      - list of hyp_start positions (index of first hyp token for each sample)
    """
    assert len(prompts) == len(hyps)
    bos_id = tokenizer.bos_token_id
    out_ids: List[List[int]] = []
    hyp_starts: List[int] = []

    for p, h in zip(prompts, hyps):
        # tokenize w/o special tokens so we control BOS once
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        h_ids = tokenizer.encode(h, add_special_tokens=False)

        seq: List[int] = []
        if add_bos_once and bos_id is not None:
            seq.append(bos_id)

        seq.extend(p_ids)
        hyp_start = len(seq)  # first hyp token index
        seq.extend(h_ids)

        out_ids.append(seq)
        hyp_starts.append(hyp_start)
    return out_ids, hyp_starts


def pad_right(batch_ids: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch to max length; return input_ids and attention_mask."""
    maxlen = max(len(x) for x in batch_ids) if batch_ids else 0
    input_ids = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch_ids), maxlen), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
    return input_ids, attn


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Explicit position_ids that increment only on real tokens:
    pos = cumsum(attention_mask)-1, with pads set to 0.
    This mirrors how many HF models derive positions internally.
    """
    pos = attention_mask.cumsum(dim=-1) - 1
    pos = pos.clamp_min(0)
    pos = pos.masked_fill(attention_mask == 0, 0)
    return pos.to(torch.long)


@torch.inference_mode()
def score_batch(
    model,
    tokenizer,
    batch_prompts: List[str],
    batch_hyps: List[str],
    pass_position_ids: bool = False,
    include_eos_in_hyp: bool = False,
) -> List[Tuple[float, int]]:
    """
    Returns list of (nll, T), where nll = -sum log p(hyp_tokens | prompt),
    T = number of scored tokens (hyp length (+1 if include_eos)).
    """
    # 1) Build per-sample concats
    cat_ids, hyp_starts = build_concat_ids(tokenizer, batch_prompts, batch_hyps)

    # Optionally append EOS to hyp for scoring
    if include_eos_in_hyp and tokenizer.eos_token_id is not None:
        for i, (h, s) in enumerate(zip(batch_hyps, hyp_starts)):
            # append EOS only if sequence is non-empty so target exists
            cat_ids[i].append(tokenizer.eos_token_id)

    # 2) Single pad at batch level (no interior pads)
    input_ids, attention_mask = pad_right(cat_ids, tokenizer.pad_token_id)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # 3) Optional explicit position_ids
    model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if pass_position_ids:
        position_ids = make_position_ids(attention_mask)
        model_kwargs["position_ids"] = position_ids

    # 4) Forward
    logits = model(**model_kwargs).logits  # [B, L, V]
    # Shift to get p(x_t | x_<t)
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, L-1, V]
    tokens = input_ids[:, 1:]  # next tokens as labels

    # 5) For each sample, gather only hyp tokens
    results: List[Tuple[float, int]] = []
    for b in range(input_ids.size(0)):
        L = attention_mask[b].sum().item()
        hyp_start = hyp_starts[b]
        # hyp tokens live in positions [hyp_start .. L-1] in the *input*;
        # their conditional probs are taken from logprobs indices [hyp_start-1 .. L-2]
        left = max(hyp_start - 1, 0)
        right = L - 1  # inclusive index in input for last conditional; slice end is right
        if right <= left:
            # empty hyp -> length 0 (nll = 0.0)
            results.append((0.0, 0))
            continue

        lp_slice = logprobs[b, left:right, :]                # [T, V]
        tok_slice = tokens[b, left:right]                    # [T]
        tok_lp = lp_slice.gather(1, tok_slice.unsqueeze(1)).squeeze(1)  # [T]
        nll = -tok_lp.sum().double().item()
        T = tok_slice.numel()
        results.append((nll, T))
    return results


def run_compare(
    model, tokenizer, data: List[Tuple[str, str]],
    batch_size: int, pass_pos_ids: bool, include_eos: bool
):
    prompts = [p for p, _ in data]
    hyps = [h for _, h in data]

    # Non-batched (true per-sample)
    nb_nll_T = []
    for p, h in zip(prompts, hyps):
        r = score_batch(model, tokenizer, [p], [h],
                        pass_position_ids=pass_pos_ids,
                        include_eos_in_hyp=include_eos)
        nb_nll_T.append(r[0])

    # Batched in chunks
    b_nll_T = []
    for i in range(0, len(data), batch_size):
        chunk_p = prompts[i:i+batch_size]
        chunk_h = hyps[i:i+batch_size]
        r = score_batch(model, tokenizer, chunk_p, chunk_h,
                        pass_position_ids=pass_pos_ids,
                        include_eos_in_hyp=include_eos)
        b_nll_T.extend(r)

    # Compare
    assert len(nb_nll_T) == len(b_nll_T)
    print("\n=== Per-sample comparison (non-batch vs batched) ===")
    max_abs_diff = 0.0
    for idx, ((n1, T1), (n2, T2)) in enumerate(zip(nb_nll_T, b_nll_T)):
        diff = abs(n1 - n2)
        max_abs_diff = max(max_abs_diff, diff)
        ppl1 = math.exp(n1 / T1) if T1 > 0 else float("nan")
        ppl2 = math.exp(n2 / T2) if T2 > 0 else float("nan")
        print(f"[{idx:03d}] T={T1}  NLL(nb)={n1:.6f}  NLL(b)={n2:.6f}  Δ={diff:.6g}  "
              f"PPL(nb)={ppl1:.4f}  PPL(b)={ppl2:.4f}")
    print(f"\nMax |Δ NLL| = {max_abs_diff:.6g}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Local HF model directory")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--pass_position_ids", action="store_true",
                    help="If set, pass explicit position_ids derived from attention_mask")
    ap.add_argument("--include_eos", action="store_true",
                    help="If set, score EOS right after the hypothesis")
    ap.add_argument("--stdin_pairs", action="store_true",
                    help="Read prompt<TAB>hyp pairs from stdin instead of built-ins")
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_dir)

    # Provide a tiny default set; or read from stdin as TSV lines
    data: List[Tuple[str, str]] = [
        ("", "hello world"),
        ("The capital of France is", " Paris."),
        ("User: Hi\nAssistant:", " Hello! How can I help you today?"),
        ("Context: speech recognition system logs\nQuery:", " show the top five errors."),
    ]
    if args.stdin_pairs:
        import sys
        data = []
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                raise ValueError("stdin_pairs expects lines as: <prompt>\\t<hyp>")
            p, h = line.split("\t", 1)
            data.append((p, h))

    run_compare(
        model, tokenizer, data,
        batch_size=args.batch_size,
        pass_pos_ids=args.pass_position_ids,
        include_eos=args.include_eos
    )


if __name__ == "__main__":
    main()
