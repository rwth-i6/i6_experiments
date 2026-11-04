"""Isolated jobs for the 9804f5aa dynamic LLM search prototype."""

import json
import time

from sisyphus import Job, Task


class GraniteRawPiecesJob(Job):
    """Lexicon-free orthographies, not sanitized acoustic phoneme names."""

    def __init__(self, repo, *, code_version):
        self.repo, self.code_version = repo, code_version
        self.out_vocab = self.output_path("pieces.txt")
        self.out_report = self.output_path("inventory.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from transformers import AutoConfig, AutoTokenizer
        from .dynamic_llm import decode_byte_pieces

        config = AutoConfig.from_pretrained(self.repo.get_path())
        tokenizer = AutoTokenizer.from_pretrained(self.repo.get_path())
        labels = tokenizer.convert_ids_to_tokens(list(range(config.vocab_size)))
        blank = getattr(config, "blank_token_id", 0)
        labels[blank] = "[BLANK]"
        if any(not s or "\n" in s or "\r" in s for s in labels) or len(set(labels)) != len(labels):
            raise ValueError("Raw acoustic inventory must be dense and injective")
        for piece in labels:
            if piece.count("▁") > 1 or ("▁" in piece and not piece.startswith("▁")) or any(c.isspace() for c in piece):
                raise ValueError(f"Unsupported multi-boundary acoustic piece: {piece!r}")
        for pieces in (["▁", "<0xC2>", "<0xA2>", "3", "7"], ["▁", "<0xC2>"]):
            decoded = tokenizer.convert_tokens_to_string(pieces)
            reconstructed = decode_byte_pieces("".join(pieces).removeprefix("▁"))
            if decoded != reconstructed:
                raise ValueError(f"Byte reconstruction differs from Granite detokenizer: {pieces!r}")
        with open(self.out_vocab.get_path(), "w") as f:
            f.write("\n".join(labels) + "\n")
        with open(self.out_report.get_path(), "w") as f:
            json.dump(
                dict(
                    num_labels=len(labels),
                    blank=blank,
                    marker="▁",
                    raw_output_order=True,
                    word_start_pieces=sum(p.startswith("▁") for p in labels),
                ),
                f,
                indent=2,
            )


class DynamicQwenGateJob(Job):
    """Native forced piece paths vs independent full-string teacher forcing."""

    def __init__(self, binary, repo, *, code_version):
        self.binary, self.repo, self.code_version = binary, repo, code_version
        self.out_report = self.output_path("validation.json")
        self.out_log = self.output_path("rasr.validation.log")
        self.rqmt = dict(cpu=2, mem=16, gpu=1, gpu_mem=24, time=2)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import subprocess
        import sys

        # FLF owns a process-global lexicon: every SearchAlgorithm must run in
        # its own interpreter, not merely be deleted between validation cases.
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        runner = (
            "import json,sys; "
            "from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.dynamic_llm_jobs "
            "import run_dynamic_qwen_gate_case; "
            "result=run_dynamic_qwen_gate_case(*json.loads(sys.argv[1])); "
            "json.dump(result,open(sys.argv[2],'w'),indent=2)"
        )
        checks = []
        with open(self.out_log.get_path(), "w") as combined:
            for case, pieces, expected in (
                ("multi-piece", ["▁he", "llo", "▁world"], "hello world"),
                ("byte-fallback", ["▁", "<0xC2>", "<0xA2>", "3", "7", "▁and"], "¢37 and"),
            ):
                vocab_path = self.output_path(f"{case}.txt").get_path()
                with open(vocab_path, "w") as f:
                    f.write("\n".join(["[BLANK]"] + pieces) + "\n")
                for index, (scale, penalty, casings) in enumerate(
                    (
                        (0.3, 0.0, "lower"),
                        (0.3, -1.0, "lower"),
                        (1.0, 0.0, "lower"),
                        (0.3, 0.0, "original,lower,capitalized,upper"),
                    )
                ):
                    report_path = self.output_path(f"{case}-{index}.json").get_path()
                    log_path = self.output_path(f"{case}-{index}.log").get_path()
                    arguments = [
                        self.binary.get_path(),
                        self.repo.get_path(),
                        vocab_path,
                        log_path,
                        case,
                        pieces,
                        expected,
                        scale,
                        penalty,
                        casings,
                    ]
                    print(f"Validate {case}: scale={scale}, penalty={penalty}, casings={casings}", flush=True)
                    try:
                        subprocess.run(
                            [sys.executable, "-c", runner, json.dumps(arguments), report_path],
                            check=True,
                            env=environment,
                        )
                    finally:
                        if os.path.exists(log_path):
                            with open(log_path, encoding="utf-8", errors="replace") as f:
                                combined.write(f"\n=== {case}-{index} ===\n")
                                combined.write(f.read())
                            combined.flush()
                    with open(report_path) as f:
                        checks.append(json.load(f))
        with open(self.out_report.get_path(), "w") as f:
            json.dump(dict(passed=True, checks=checks, dtype="float32", process_isolation=True), f, indent=2)


def run_dynamic_qwen_gate_case(binary, repo, vocab_path, log_path, case, pieces, expected, scale, penalty, casings):
    """One native validation case per fresh process; never recreate FLF here."""
    import os
    import sys
    import numpy as np
    import torch
    from .dynamic_llm import register_qwen_scorer, casing_surfaces

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, binary)
    import librasr

    if not hasattr(librasr, "LlmScorer"):
        raise RuntimeError("Build does not contain the LLM Python interface")
    instances = []
    register_qwen_scorer(librasr, instances=instances)
    cfg = librasr.Configuration()
    cfg.set("lib-rasr.lexicon.file", f"vocab-text:{vocab_path}")
    cfg.set("lib-rasr.label-scorer.type", "no-op")
    selector = "lib-rasr.search-algorithm."
    for key, value in dict(
        type="llm-timesync-beam-search",
        **{
            "max-beam-size": 1,
            "pre-llm-max-beam-size": 1,
            "score-threshold": 100,
            "blank-label-index": 0,
            "collapse-repeated-labels": "true",
            "llm-scale": scale,
            "word-penalty": penalty,
            "log-stepwise-statistics": "false",
            "llm.type": "qwen-dynamic",
            "llm.model": repo,
            "llm.case-variants": casings,
            "llm.max-batch-size": 8,
        },
    ).items():
        cfg.set(selector + key, str(value))
    for channel in ("log", "warning", "error", "statistics"):
        cfg.set(f"*.{channel}.channel", log_path)
    librasr.init_logging(cfg)
    search = librasr.SearchAlgorithm(config=cfg)
    path = [0]
    for index in range(1, len(pieces) + 1):
        path.extend([index, index, 0])
    costs = np.full((len(path), len(pieces) + 1), 1e6, dtype=np.float32)
    costs[np.arange(len(path)), path] = 0
    trace = search.recognize_segment(costs)
    backend = instances[-1].backend
    # Native independent full-string reference, not the adapter's
    # continuation slicing implementation.
    history = backend.initial_tokens()[:]
    chosen_surfaces = []
    words = expected.split()
    for w, word in enumerate(words):
        options = []
        for surface in casing_surfaces((" " if w else "") + word, casings.split(",")):
            word_ids = backend.tokenizer.encode(surface, add_special_tokens=False)
            suffix = word_ids + (backend.sentence_end_tokens() if w == len(words) - 1 else [])
            trial = torch.tensor([history + suffix], device="cuda")
            with torch.inference_mode():
                logits = backend.model(input_ids=trial, use_cache=False).logits[0].float()
                cost = torch.nn.functional.cross_entropy(
                    logits[len(history) - 1 : len(history) + len(suffix) - 1],
                    trial[0, len(history) :],
                    reduction="sum",
                ).item()
            options.append((cost, surface, word_ids))
        _, surface, selected = min(options, key=lambda item: item[0])
        history += selected
        chosen_surfaces.append(surface)
    tokens = history + backend.sentence_end_tokens()
    chosen_surface = "".join(chosen_surfaces)
    if backend.tokenizer.encode(chosen_surface, add_special_tokens=False) != history[len(backend.initial) :]:
        raise ValueError("Wordwise tokenization does not equal full-string tokenization in native gate")
    ids = torch.tensor([tokens], device="cuda")
    with torch.inference_mode():
        logits = backend.model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        reference = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="sum").item()
    expected_cost = scale * reference + penalty * len(expected.split())
    actual = float(trace[-1].am_score + trace[-1].lm_score)
    if abs(actual - expected_cost) > 0.01:
        raise ValueError(f"Native Qwen cost mismatch: {case}, {actual} != {expected_cost}")
    emitted = [item.lemma for item in trace if item.lemma != "[BLANK]"]
    if emitted != pieces:
        raise ValueError(f"Native piece path changed: {emitted}")
    # Unequal-length requests exercise padding/slicing/reordering.
    prefixes = [tokens[:1], tokens[:3]]
    continuations = [tokens[1:3], tokens[3:]]
    scored = backend.score_continuations(prefixes, continuations)
    for prefix, continuation, scores in zip(prefixes, continuations, scored):
        sequence = torch.tensor([prefix + continuation], device="cuda")
        with torch.inference_mode():
            logits = backend.model(input_ids=sequence, use_cache=False).logits[0].float()
            target = sequence[0, len(prefix) :]
            expected_scores = (
                torch.nn.functional.cross_entropy(
                    logits[len(prefix) - 1 : len(prefix) + len(continuation) - 1], target, reduction="none"
                )
                .cpu()
                .tolist()
            )
        if max((abs(a - b) for a, b in zip(scores, expected_scores)), default=0.0) > 0.002:
            raise ValueError("Padded Qwen continuation scores differ from independent forwards")

    return dict(
        case=case,
        scale=scale,
        penalty=penalty,
        casings=casings,
        chosen_surface=chosen_surface,
        expected=expected_cost,
        actual=actual,
        pid=os.getpid(),
    )


class DynamicGraniteQwenJob(Job):
    def __init__(self, dataset, repo, qwen_repo, vocabulary, binary, rasr_config, gate, *, zero=False, code_version):
        self.dataset, self.repo, self.qwen_repo, self.vocabulary = dataset, repo, qwen_repo, vocabulary
        self.binary, self.rasr_config, self.gate = binary, rasr_config, gate
        self.zero, self.code_version = zero, code_version
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_log = self.output_path("rasr.recog.log")
        self.rqmt = dict(cpu=2, mem=24, gpu=1, gpu_mem=24, time=24)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import numpy as np
        import torch
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result, collapse_ctc
        from .dynamic_llm import register_qwen_scorer

        with open(self.gate.get_path()) as f:
            if not json.load(f)["passed"]:
                raise ValueError("Dynamic Qwen native preflight did not pass")
        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        sys.path.insert(0, self.binary.get_path())
        import librasr

        instances = []
        register_qwen_scorer(librasr, zero=self.zero, instances=instances)
        cfg = librasr.Configuration()
        if not cfg.set_from_file(self.rasr_config.get_path()):
            raise ValueError("Failed to read prototype config")
        cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        adapter = HfBenchmarkAdapter(self.repo.get_path(), "ctc", "float32")
        with open(self.vocabulary.get_path()) as f:
            vocabulary = {piece.rstrip("\n"): i for i, piece in enumerate(f)}
        dataset = load_from_disk(self.dataset.get_path())
        audio_seconds = elapsed = 0.0
        with open(self.out_hypotheses.get_path(), "w") as dst:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    features = adapter.encode_inputs(adapter.inputs(waveform))
                trace = search.recognize_segment(np.ascontiguousarray(features, dtype=np.float32))
                result = traceback_result(trace, adapter, vocabulary, True)
                pieces = adapter.tokenizer.convert_ids_to_tokens(result["token_ids"])
                pending_pieces = max_pieces = 0
                for piece in pieces:
                    if piece.startswith("▁"):
                        pending_pieces = 0
                    if piece != "▁":
                        pending_pieces += 1
                        max_pieces = max(max_pieces, pending_pieces)
                result["num_output_words"] = len(result["text"].split())
                result["max_word_pieces"] = max_pieces
                if self.zero:
                    argmax_ids = collapse_ctc(
                        np.asarray(features).reshape(-1, len(vocabulary)).argmin(-1).tolist(), adapter.blank
                    )
                    result["greedy_text"] = adapter.detokenize(argmax_ids)
                    result["greedy_matches"] = result["text"] == result["greedy_text"]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                audio_seconds += len(waveform) / 16000
                elapsed += seconds
                result.update(id=str(sample["id"]), elapsed_seconds=seconds)
                backend = instances[-1].backend
                if backend:
                    result["llm_requests"] = backend.requests
                    result["llm_forward_batches"] = backend.forward_batches
                    result["llm_max_request_length"] = backend.max_request_length
                dst.write(json.dumps(result, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)} {sample['id']}: RTF={elapsed / audio_seconds:.3f}", flush=True)
        with open(self.out_runtime.get_path(), "w") as f:
            json.dump(
                dict(
                    num_segments=len(dataset),
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    dtype="float32",
                    zero=self.zero,
                ),
                f,
                indent=2,
            )
