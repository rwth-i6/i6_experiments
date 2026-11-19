"""Candidate generation and fixed-transcript acoustic scores for two-pass fusion."""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from sisyphus import Job, Task


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def ordered_candidates(candidates, n):
    unique = {}
    for candidate in sorted(candidates, key=lambda c: c["first_pass_cost"]):
        if not math.isfinite(candidate["first_pass_cost"]):
            raise ValueError("Nonfinite first-pass cost")
        unique.setdefault(candidate["text"], candidate)
    if not unique:
        raise ValueError("First pass returned no candidates")
    return list(unique.values())[:n]


def generated_ids(row, eos, pad):
    """Keep an emitted EOS, remove only padding, and retain capped sequences."""
    for i, token in enumerate(row):
        if token in eos:
            return row[: i + 1], row[:i]
        if token == pad:
            return row[:i], row[:i]
    return row, row


def cohere_costs(adapter, sequences, batch_size=4):
    """Teacher force each candidate using one shared encoder output and prompt."""
    import torch

    result = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        prompt_length = len(adapter.prompt)
        # The last scored token is a target only, so never feed it to the decoder.
        length = max(prompt_length + max(0, len(s) - 1) for s in batch)
        ids = torch.full((len(batch), length), adapter.prompt[-1], device=adapter.device, dtype=torch.long)
        for i, tokens in enumerate(batch):
            prefix = adapter.prompt + tokens[:-1]
            ids[i, : len(prefix)] = torch.tensor(prefix, device=adapter.device)
        decoded = adapter.model.model.decoder(
            input_ids=ids,
            encoder_hidden_states=adapter.encoder.last_hidden_state.expand(len(batch), -1, -1),
            encoder_attention_mask=adapter.encoder.attention_mask.expand(len(batch), -1),
            use_cache=False,
        )
        logits = adapter.model.proj_out(decoded.last_hidden_state[:, prompt_length - 1 :]).float()
        for i, tokens in enumerate(batch):
            target = torch.tensor(tokens, device=adapter.device, dtype=torch.long)
            result.append(
                float(torch.nn.functional.cross_entropy(logits[i, : len(tokens)], target, reduction="sum"))
                if tokens
                else 0.0
            )
    return result


def seq2seq_nbest(backend, waveform, family, n):
    if family == "qwen":
        inputs = backend.prepare(waveform)
        prompt_length = inputs["input_ids"].shape[-1]
        eos = backend.eos_ids
        tokenizer = backend.tokenizer

        def decode(ids):
            return backend.processor.decode(ids, return_format="transcription_only")
    else:
        inputs = backend.inputs(waveform)
        prompt_length = inputs["decoder_input_ids"].shape[-1]
        eos = [backend.eos]
        tokenizer = backend.tokenizer
        decode = backend.detokenize
    raw = backend.model.generate(
        **inputs,
        num_beams=n,
        num_return_sequences=n,
        length_penalty=0.0,
        do_sample=False,
        max_new_tokens=500,
    )
    raw = raw.sequences if hasattr(raw, "sequences") else raw
    sequences, surfaces = zip(
        *(generated_ids(row, eos, tokenizer.pad_token_id) for row in raw[:, prompt_length:].tolist())
    )
    if family == "qwen":
        backend.set_audio(inputs)
        costs = [sum(c) for c in backend.score_continuations([backend.initial] * len(sequences), sequences)]
    else:
        backend.encode_inputs(inputs)
        costs = cohere_costs(backend, sequences)
    return ordered_candidates(
        [
            dict(
                text=decode(ids).strip(),
                first_pass_cost=cost,
                native_rank=rank,
                truncated=not tokens or tokens[-1] not in eos,
            )
            for rank, (ids, tokens, cost) in enumerate(zip(surfaces, sequences, costs))
        ],
        n,
    )


def cohere_adapter(repo):
    from .hf_benchmark import HfBenchmarkAdapter

    adapter = HfBenchmarkAdapter(repo, "aed")
    # Both registered dev sets are below 40 s. Use one encoder context even
    # for the few utterances that the default processor would split.
    fe = adapter.processor.feature_extractor
    fe.max_audio_clip_s = 45.0 + fe.overlap_chunk_second
    return adapter


class GenerateFusionNbestJob(Job):
    def __init__(self, dataset, repo, *, family, n, code_version, rasr=None):
        self.dataset, self.repo, self.family, self.n = dataset, repo, family, n
        self.code_version, self.rasr = code_version, rasr
        self.out_candidates = self.output_path("candidates.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_log = self.output_path("rasr.log") if rasr else None

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=32, gpu=1, gpu_mem=48, time=48, sbatch_args=["-p", "gpu_48gb"]))

    def run(self):
        import torch
        from datasets import load_from_disk
        from .hf_benchmark import audio_array, check_transformers_version
        from .parakeet_tdt import decode_tokens, encoder_frames, load_model, waveform_inputs
        from .qwen_speech import QwenSpeechScorer
        from .speech_qwen_round import pack_nbest

        check_transformers_version()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        repo = self.repo.get_path()
        if self.family == "qwen":
            backend = QwenSpeechScorer(repo, batch_size=4)
        elif self.family == "aed":
            backend = cohere_adapter(repo)
        else:
            model, processor = load_model(repo, "cuda")
            rasr = self.rasr
            gate = json.loads(Path(rasr["gate"].get_path()).read_text())
            if not gate["passed"]:
                raise ValueError("TDT ONNX parity gate failed")
            if model.generation_config.eos_token_id != rasr["eos"]:
                raise ValueError("TDT checkpoint EOS differs from the search configuration")
            vocab = {piece: i for i, piece in enumerate(Path(rasr["vocab"].get_path()).read_text().splitlines())}
            sys.path.insert(0, rasr["binary"].get_path())
            import librasr

            cfg = librasr.Configuration()
            if not cfg.set_from_file(rasr["config"].get_path()):
                raise ValueError("Invalid TDT n-best search config")
            for channel in ("log", "warning", "error", "statistics"):
                cfg.set(f"*.{channel}.channel", self.out_log.get_path())
            librasr.init_logging(cfg)
            search = librasr.SearchAlgorithm(config=cfg)

            def render(trace):
                return decode_tokens(model, processor, [vocab[x.lemma] for x in trace if x.lemma])

        dataset = load_from_disk(self.dataset.get_path())
        seconds = 0.0
        sizes = []
        empty_first_pass = 0
        started = time.perf_counter()
        with torch.inference_mode(), open(self.out_candidates.get_path(), "w") as dst:
            for index, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                if self.rasr:
                    features = encoder_frames(model, waveform_inputs(processor, waveform, "cuda")).cpu().numpy()
                    traces = search.recognize_segment_n_best(features, self.n)
                    first = search.get_current_best_traceback()
                    candidates = pack_nbest(traces, first, render)
                else:
                    candidates = seq2seq_nbest(backend, waveform, self.family, self.n)
                write_row(dst, dict(id=str(sample["id"]), candidates=candidates))
                sizes.append(len(candidates))
                empty_first_pass += not candidates[0]["text"].strip()
                seconds += len(waveform) / 16000
                if index % 100 == 0:
                    print(f"{index + 1}/{len(dataset)}: {len(candidates)} candidates", flush=True)
        if not seconds or max(sizes) < 2:
            raise ValueError("N-best generation did not produce alternatives")
        elapsed = time.perf_counter() - started
        Path(self.out_runtime.get_path()).write_text(
            json.dumps(
                dict(
                    rtf=elapsed / seconds,
                    elapsed_seconds=elapsed,
                    audio_seconds=seconds,
                    mean_candidates=float(np.mean(sizes)),
                    min_candidates=min(sizes),
                    max_candidates=max(sizes),
                    empty_first_pass=empty_first_pass,
                )
            )
        )


def scoring_surface(text, family):
    """Reference-independent rendering used only for the auxiliary score."""
    text = text.replace("’", "'").replace("‘", "'")
    if family in ("ctc", "rnnt", "aed"):
        text = text.lower()
        if family == "ctc":
            # Preserve decimal points and internal word punctuation in Granite.
            text = re.sub(r"(?<!\w)[^\w\s']|[^\w\s'](?!\w)", " ", text)
        else:
            text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def fixed_tokens(tokenizer, text, family):
    # Reject unknown/special symbols and tokenizer transformations that lose
    # content, such as a digit on a letter-only tokenizer.
    surface = scoring_surface(text, family)
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if set(ids).intersection(tokenizer.all_special_ids):
        return None
    kw = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if family in ("ctc", "rnnt", "tdt"):
        kw["group_tokens"] = False
    recovered = tokenizer.decode(ids, **kw)
    if scoring_surface(recovered, family) != surface:
        return None
    return ids


def transducer_viterbi(blank, label, durations=None):
    """Best complete alignment. A blank advances time, a label advances text."""
    frames, prefixes = blank.shape
    dp = np.full((frames + 1, prefixes), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for t in range(frames):
        current = dp[t]
        continuation = label[t] if durations is None else label[t] + durations[t, :-1, 0]
        cumulative = np.concatenate(([0.0], np.cumsum(continuation, dtype=np.float64)))
        current[:] = cumulative + np.minimum.accumulate(current - cumulative)
        if durations is None:
            np.minimum(dp[t + 1], current + blank[t], out=dp[t + 1])
        else:
            for d in range(1, durations.shape[-1]):
                dest = dp[min(t + d, frames)]
                np.minimum(dest, current + blank[t] + durations[t, :, d], out=dest)
                np.minimum(dest[1:], current[:-1] + label[t] + durations[t, :-1, d], out=dest[1:])
    return float(dp[-1, -1])


def score_transducer(model, frames, targets, blank, durations):
    """Compute token and duration costs in bounded tiles, then align each string."""
    import torch

    results = []
    for tokens in targets:
        prefix = torch.tensor([[blank, *tokens]], device=frames.device)
        predictor = model.decoder(prefix, cache=None)[0]
        blank_cost, token_cost, duration_cost = [], [], []
        for tile in frames.split(max(1, min(32, 4096 // len(prefix[0])))):
            logits = model.joint(
                decoder_hidden_states=predictor[None, :, :],
                encoder_hidden_states=tile[:, None, :],
            ).float()
            vocab = model.config.vocab_size
            costs = -logits[..., :vocab].log_softmax(-1)
            blank_cost.append(costs[..., blank].cpu().numpy())
            idx = prefix[0, 1:].view(1, -1, 1).expand(len(tile), -1, 1)
            token_cost.append(costs[:, :-1].gather(2, idx).squeeze(-1).cpu().numpy())
            if durations:
                duration_cost.append((-logits[..., vocab:].log_softmax(-1)).cpu().numpy())
        results.append(
            transducer_viterbi(
                np.concatenate(blank_cost),
                np.concatenate(token_cost),
                np.concatenate(duration_cost) if durations else None,
            )
        )
    return results


def score_fixed(backend, family, waveform, texts):
    import torch
    from .parakeet_tdt import encoder_frames, waveform_inputs

    if family == "tdt":
        model, processor = backend
        tokenizer = processor.tokenizer
    else:
        tokenizer = backend.tokenizer
    encoded = [fixed_tokens(tokenizer, text, family) for text in texts]
    valid = {tuple(ids): [] for ids in encoded if ids is not None}
    for index, ids in enumerate(encoded):
        if ids is not None:
            valid[tuple(ids)].append(index)
    result = [None] * len(texts)
    if not valid:
        return result
    targets = [list(ids) for ids in valid]
    if family == "ctc":
        features = backend.inputs(waveform)
        log_probs = backend.ctc_logits(features).float().log_softmax(-1)
        costs = []
        for start in range(0, len(targets), 4):
            batch = targets[start : start + 4]
            lengths = torch.tensor([len(ids) for ids in batch], dtype=torch.long)
            concatenated = torch.tensor([i for ids in batch for i in ids], dtype=torch.long, device=log_probs.device)
            costs.extend(
                torch.nn.functional.ctc_loss(
                    log_probs[:, None, :].expand(-1, len(batch), -1).contiguous(),
                    concatenated,
                    torch.full((len(batch),), len(log_probs), dtype=torch.long),
                    lengths,
                    blank=backend.blank,
                    reduction="none",
                ).tolist()
            )
    elif family == "aed":
        backend.encode_inputs(backend.inputs(waveform))
        costs = cohere_costs(backend, [ids + [backend.eos] for ids in targets])
    elif family == "rnnt":
        backend.encode(waveform)
        costs = score_transducer(backend.model, backend.encoder[0, : backend.num_frames], targets, backend.blank, ())
    else:
        frames = encoder_frames(model, waveform_inputs(processor, waveform, "cuda"))
        eos = model.generation_config.eos_token_id
        costs = score_transducer(
            model,
            frames,
            [ids + ([eos] if eos is not None else []) for ids in targets],
            model.config.blank_token_id,
            model.config.durations,
        )
    for indices, cost in zip(valid.values(), costs, strict=True):
        if math.isnan(cost) or cost == -math.inf:
            raise ValueError("Invalid auxiliary cost")
        for index in indices:
            result[index] = cost if math.isfinite(cost) else None
    return result


class ScoreFixedFusionJob(Job):
    def __init__(self, dataset, candidates, repo, *, family, code_version):
        self.dataset, self.candidates, self.repo, self.family = dataset, candidates, repo, family
        self.code_version = code_version
        self.out_scores = self.output_path("scored.jsonl")
        self.out_runtime = self.output_path("runtime.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=32, gpu=1, gpu_mem=48, time=48, sbatch_args=["-p", "gpu_48gb"]))

    def run(self):
        import torch
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, check_transformers_version
        from .parakeet_tdt import load_model

        check_transformers_version()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        family, repo = self.family, self.repo.get_path()
        if family == "tdt":
            backend = load_model(repo, "cuda")
        elif family == "aed":
            backend = cohere_adapter(repo)
        else:
            backend = HfBenchmarkAdapter(repo, family)
        dataset = load_from_disk(self.dataset.get_path())
        count = available = skipped = 0
        seconds = 0.0
        started = time.perf_counter()
        with torch.inference_mode(), open(self.candidates.get_path()) as src, open(
            self.out_scores.get_path(), "w"
        ) as dst:
            for index, sample in enumerate(dataset):
                line = next(src, None)
                if line is None:
                    raise ValueError(f"Missing n-best candidates for {sample['id']}")
                row = json.loads(line)
                if row["id"] != str(sample["id"]) or not row["candidates"]:
                    raise ValueError("Audio and n-best candidates do not match")
                waveform = audio_array(sample["audio"])
                candidates = row["candidates"]
                scores = score_fixed(backend, family, waveform, [c["text"] for c in candidates])
                write_row(
                    dst, dict(id=row["id"], candidates=[dict(c, auxiliary_cost=s) for c, s in zip(candidates, scores)])
                )
                count += len(scores)
                available += sum(s is not None for s in scores)
                skipped += all(s is None for s in scores)
                seconds += len(waveform) / 16000
                if index % 100 == 0:
                    print(f"Scored {available}/{count} candidates", flush=True)
            if next(src, None) is not None:
                raise ValueError("Extra candidate segments")
        if not count or not available:
            raise ValueError("No scoreable candidates")
        elapsed = time.perf_counter() - started
        Path(self.out_runtime.get_path()).write_text(
            json.dumps(
                dict(
                    rtf=elapsed / seconds,
                    elapsed_seconds=elapsed,
                    audio_seconds=seconds,
                    candidates=count,
                    scoreable=available,
                    segments_with_no_scoreable_candidates=skipped,
                )
            )
        )


def pick_candidate(candidates, *, scale, speech_policy):
    if scale == 0:
        return candidates[0]

    def cost(candidate):
        if speech_policy is None:
            return candidate["auxiliary_cost"]
        return candidate["speech"][speech_policy]["nll"]

    return min(
        (c for c in candidates if cost(c) is not None),
        key=lambda c: c["first_pass_cost"] + scale * cost(c),
        default=candidates[0],
    )


class SelectBidirectionalJob(Job):
    def __init__(self, candidates, *, n, scales, speech_policy=None, code_version):
        self.candidates, self.n, self.scales = candidates, n, scales
        self.speech_policy, self.code_version = speech_policy, code_version
        self.out_hypotheses = {s: self.output_path(f"scale-{s:g}.jsonl") for s in scales}

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        from contextlib import ExitStack

        with ExitStack() as stack, open(self.candidates.get_path()) as src:
            outputs = {s: stack.enter_context(open(out.get_path(), "w")) for s, out in self.out_hypotheses.items()}
            for line in src:
                row = json.loads(line)
                for scale, out in outputs.items():
                    best = pick_candidate(row["candidates"][: self.n], scale=scale, speech_policy=self.speech_policy)
                    write_row(out, dict(id=row["id"], text=best["text"]))
