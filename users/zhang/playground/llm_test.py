import torch
import math
import time
import returnn.util.basic as util  # noqa
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
    get_content_dir_from_hub_cache_dir,
)
if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"


    def _report_dev_memory_stats():
        dev = torch.device(device_str)
        if dev.type == "cuda":
            stats = [
                f"alloc cur {util.human_bytes_size(torch.cuda.memory_allocated(dev))}",
                f"alloc peak {util.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
                f"reserved cur {util.human_bytes_size(torch.cuda.memory_reserved(dev))}",
                f"reserved peak {util.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
            ]
            print(f"Memory usage ({device_str}):", " ".join(stats))


    from transformers import AutoModelForCausalLM, AutoTokenizer

    start_time = time.time()
    print("Loading model...")

    model_path = get_content_dir_from_hub_cache_dir(self.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"\nTokenizer_max_length:{tokenizer.model_max_length}\n")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[
        0] >= 8 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,  # load directly in low precision
        device_map={"": 0},  # put the whole model on cuda:0 (single GPU)
        low_cpu_mem_usage=True,  # stream weights, smaller CPU peak
        # attn_implementation="flash_attention_2",  # faster runtime; f
    )
    # model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
    device = torch.device(device_str)
    # model.to(device)

    print(f"({time.time() - start_time} secs)")
    _report_dev_memory_stats()

    for s in ["A BUSINESS. ", "a business \n"]:
        enc = tokenizer(s, return_tensors="pt")
        print(s, "→ token IDs:", enc.input_ids.tolist())
    print(f"bos_token id:{tokenizer.bos_token}")
    print(f"eos_token id:{tokenizer.eos_token}")

    total_logprob = 0.0
    total_tokens = 0
    total_word_tokens = 0

    lines_seen = 0
    total_lines = 0
    batch_count = 0
    log_every = 1000  # print a message every 1k lines
    d_rec = dict()
    import i6_core.util as cutil

    for text_file in self.text_file:
        d_rec.update(eval(cutil.uopen(text_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")}))
        # Iterate records
        lines_seen = 0
    total_lines = sum(len(n_best) for _, n_best in d_rec.items())
    from i6_experiments.users.zhang.datasets.utils import sort_dict_by_record, extract_record_id

    if self.use_prev_context:
        d_rec = sort_dict_by_record(d_rec)
    ctx_rec = None
    if self.context is not None:
        ctx_rec = eval(cutil.uopen(self.context, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
    batch_lines, batch_prompt = [], []
    eos_symbol = (
            " " + tokenizer.eos_token) if self.eos_symbol == "eos" else self.eos_symbol  # (" "+tokenizer.eos_token) makes ppl worse
    eos_symbol = eos_symbol if self.add_eos_to_completion else ""
    last_record = None
    for seq_tag, raw_line in d_rec.items():
        boundary = False
        line = raw_line.strip().lower() if self.lower_case else raw_line.strip()
        if not line:
            continue
        total_lines += 1
        total_word_tokens += len(line.split()) + (1 if self.eos_symbol else 0)
        current_record = extract_record_id(seq_tag)
        batch_lines.append(line + eos_symbol)
        if self.use_prev_context and current_record != last_record:
            boundary = True  # This will skip one time following prompt append
            self.clear_prompt()  # makes self.prompt = '', double guard
            # Ensure there is no empty prompt inside a prompt batch
            # Process current batch, separately handle the boundary sequence
            if len(batch_lines) == 1:
                batch_count += 1
                nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
                total_logprob += nll
                total_tokens += tok_count
            else:
                batch_count += 2
                nll, tok_count = _score_batch(batch_lines[:-1], batch_prompt, tokenizer, model, device)
                nll_boundary, tok_count_boundary = _score_batch([batch_lines[-1]], [], tokenizer, model, device)
                total_logprob += nll + nll_boundary
                total_tokens += tok_count + tok_count_boundary

            if batch_count % 1000 == 0:
                print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

            batch_lines, batch_prompt = [], []  # clear for next batch
            print(f"Clear context for record {last_record}")

        if self.prompt and not boundary:  # Never append " "
            batch_prompt.append(self.prompt.lower() if self.lower_case else self.prompt)
        lines_seen += 1

        # Log after every `log_every` lines
        if lines_seen % log_every == 0:
            print(f"[Line {lines_seen:,}/{total_lines}] {100 * lines_seen / total_lines:.2f}% processed…")
            print(f"current lines:{batch_lines}")
            _report_dev_memory_stats()
        # Once we have `batch_size` lines, tokenize & process them
        if len(batch_lines) == self.batch_size:
            batch_count += 1
            nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
            total_logprob += nll
            total_tokens += tok_count

            if batch_count % 1000 == 0:
                print(f"  → Completed {batch_count:,} batches; Tokens so far: {total_tokens:,}")

            batch_lines, batch_prompt = [], []  # clear for next batch

        if self.use_prev_context:
            if ctx_rec is not None:
                ctx = ctx_rec[seq_tag]
                print(f"Ctx for {seq_tag}: {ctx}")
            else:
                ctx = line
                print(f"Transcription for {seq_tag}: {ctx}")
            self.update_prompt(ctx)
            print(f"Current context len: {len(self.prompt.split())}")
            last_record = current_record

    # Process any leftover lines (if total lines % batch_size != 0)
    if batch_lines:
        nll, tok_count = _score_batch(batch_lines, batch_prompt, tokenizer, model, device)
        total_logprob += nll
        total_tokens += tok_count

    # print(f"(Assumed batch size 1)Average bpe seq length:{bpe_length/total_lines:.2f}")
    print(f"Average bpe/word length ratio:{total_tokens}/{total_word_tokens}->{total_tokens / total_word_tokens:.2f}")
    # Explicit cleanup to avoid stuck CG state
    del model
    del tokenizer
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    print("Finished and cleaned up.")

    # Finally compute PPL
    bpe_ppl = math.exp(-total_logprob / total_tokens)
    ppl = math.exp(-total_logprob / total_word_tokens) if self.word_ppl else bpe_ppl
    with open(self.out_ppl.get_path(), "w") as out_f:
        out_f.write(f"Average bpe/word length ratio:: {total_tokens / total_word_tokens:.2f}\n")
        out_f.write(f"Total word tokens:: {total_word_tokens}\n")
        out_f.write(f"Total bpe tokens:: {total_tokens}\n")
        out_f.write(f"bpe level: {bpe_ppl}\n")
        out_f.write(f"Perplexity: {ppl}\n")