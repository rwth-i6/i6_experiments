"""Isolated Speech-Qwen rescoring runtimes; do not rehash completed recognitions.

References are only used by the explicitly labelled oracle, never by scoring
or selection. N-best AM/LM costs are cumulative, already-scaled trace totals.
"""

import json
import math
import time
from contextlib import contextmanager

import numpy as np
from sisyphus import Job, Task

from .rnnt_llm import RnntQwenRecogJob, RnntLlmGateJob


def pack_nbest(traces, first, render):
    def cost(trace):
        return (float(trace[-1].am_score), float(trace[-1].lm_score)) if trace else (0.0, 0.0)
    candidates = {}
    for rank, trace in enumerate(traces):
        am, lm = cost(trace)
        if not math.isfinite(am + lm):
            raise ValueError("Nonfinite n-best score")
        text = render(trace)
        row = dict(text=text, am_cost=am, lm_cost=lm, first_pass_cost=am + lm, native_rank=rank)
        if text not in candidates or am + lm < candidates[text]["first_pass_cost"]:
            candidates[text] = row
    ordered = sorted(candidates.values(), key=lambda r: r["first_pass_cost"])
    if not ordered or ordered[0]["text"] != render(first) or abs(ordered[0]["first_pass_cost"] - sum(cost(first))) > .01:
        raise ValueError("RNN-T n-best/one-best parity failure; inspect native lattice scores")
    return ordered


@contextmanager
def nbest_search(librasr, callback, n):
    """Reuse the reviewed initialization/forward loop in an isolated worker."""
    original = librasr.SearchAlgorithm
    class Search:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def recognize_segment(self, features):
            traces = self.inner.recognize_segment_n_best(features, n)
            first = self.inner.recognize_segment(features)
            callback(traces, first)
            return first
    librasr.SearchAlgorithm = Search
    try:
        yield
    finally:
        librasr.SearchAlgorithm = original


class RnntSpeechNbestGateJob(RnntLlmGateJob):
    """Run the scaled word/EOS CPU topology gate through native n-best too."""
    def run(self):
        import sys
        sys.path.insert(0, self.binary.get_path())
        import librasr
        checked = []
        def check(traces, first):
            checked.append(pack_nbest(traces, first, lambda t: " ".join(x.lemma or "" for x in t)))
        with nbest_search(librasr, check, 5):
            super().run()
        if not checked:
            raise ValueError("Native n-best gate was not exercised")
        with open(self.out_report.get_path()) as src:
            report = json.load(src)
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(dict(report, nbest_one_best_scaled_score_parity=True), dst)


class RnntSpeechNbestJob(RnntQwenRecogJob):
    def __init__(self, *, n=50, **kwargs):
        super().__init__(**kwargs)
        if self.kind != "zero" or n < 1:
            raise ValueError("Speech rescoring requires AM-only candidate generation")
        self.n = n
        self.out_candidates = self.output_path("candidates.jsonl")

    def run(self):
        import sys
        from datasets import load_from_disk
        from . import hf_benchmark
        sys.path.insert(0, self.binary.get_path())
        import librasr
        ids = list(load_from_disk(self.dataset.get_path())["id"])
        with open(self.vocabulary.get_path()) as src:
            vocab = {p.rstrip("\n"): i for i, p in enumerate(src)}
        original, adapters = hf_benchmark.HfBenchmarkAdapter, []
        def adapter(*args, **kwargs):
            obj = original(*args, **kwargs)
            adapters.append(obj)
            return obj
        sizes = []
        with open(self.out_candidates.get_path(), "w") as dst:
            def record(traces, first):
                if len(adapters) != 1:
                    raise ValueError("Expected one acoustic adapter")
                rows = pack_nbest(traces, first, lambda t: hf_benchmark.traceback_result(t, adapters[0], vocab, True)["text"])
                dst.write(json.dumps(dict(id=str(ids[len(sizes)]), candidates=rows), ensure_ascii=False) + "\n")
                dst.flush()
                sizes.append(len(rows))
            hf_benchmark.HfBenchmarkAdapter = adapter
            try:
                with nbest_search(librasr, record, self.n):
                    super().run()
            finally:
                hf_benchmark.HfBenchmarkAdapter = original
        if len(sizes) != len(ids) or not sizes:
            raise ValueError("Incomplete n-best generation")
        if self.n > 1 and max(sizes) == 1:
            raise ValueError("Only singleton candidate lists; inspect/widen n-best generation before rescoring")
        with open(self.out_validation.get_path()) as src:
            report = json.load(src)
        with open(self.out_validation.get_path(), "w") as dst:
            json.dump(dict(report, nbest_parity=True, requested_n=self.n, mean_unique=float(np.mean(sizes)),
                           min_unique=min(sizes), max_unique=max(sizes), recombination="viterbi",
                           singleton_fraction=sizes.count(1) / len(sizes),
                           includes_independent_one_best_parity_decode=True), dst)


def speech_surface(text, policy):
    if policy == "original":
        return text
    from .nbest_rescoring import lm_surface
    return lm_surface(text, policy)


def score_speech_candidates(backend, candidates, policies):
    """Whole-string teacher forcing, exact prompt once and one turn-end token."""
    surfaces = list(dict.fromkeys(speech_surface(c["text"], p) for c in candidates for p in policies))
    tokens = [backend.tokenizer.encode(s, add_special_tokens=False) for s in surfaces]
    for ids in tokens:
        if set(ids) & set(backend.tokenizer.all_special_ids):
            raise ValueError("Special token in candidate transcript")
    rows = backend.score_continuations([backend.initial] * len(tokens), [ids + backend.end for ids in tokens])
    scored = {}
    for surface, ids, costs in zip(surfaces, tokens, rows):
        if len(costs) != len(ids) + len(backend.end) or not np.isfinite(costs).all():
            raise ValueError("Invalid teacher-forced candidate scores")
        scored[surface] = dict(nll=sum(costs), token_costs=costs, num_tokens=len(ids))
    return [dict(c, speech={p: scored[speech_surface(c["text"], p)] for p in policies}) for c in candidates]


class ScoreSpeechNbestJob(Job):
    def __init__(self, dataset, candidates, speech_repo, *, code_version, batch_size=8,
                 policies=("original", "initial-capital", "initial-capital-period")):
        self.dataset, self.candidates, self.speech_repo = dataset, candidates, speech_repo
        self.code_version, self.batch_size, self.policies = code_version, batch_size, policies
        self.out_scores = self.output_path("speech_candidates.jsonl")
        self.out_runtime = self.output_path("runtime.json")
        self.out_validation = self.output_path("validation.json")
        self.rqmt = dict(cpu=2, mem=96, gpu=1, gpu_mem=48, time=48)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import torch
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import audio_array
        from .qwen_speech import QwenSpeechScorer
        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = QwenSpeechScorer(self.speech_repo.get_path(), batch_size=self.batch_size)
        dataset = load_from_disk(self.dataset.get_path())
        elapsed = audio_seconds = count = 0
        validation = None
        with open(self.candidates.get_path()) as src, open(self.out_scores.get_path(), "w") as dst:
            for sample in dataset:
                line = src.readline()
                if not line:
                    raise ValueError("Missing candidate segment")
                row = json.loads(line)
                if row["id"] != str(sample["id"]) or not row["candidates"]:
                    raise ValueError("Audio/candidate id or coverage mismatch")
                waveform = audio_array(sample["audio"])
                torch.cuda.synchronize()
                started = time.perf_counter()
                backend.set_audio(backend.prepare(waveform))
                if validation is None:
                    validation = backend.validate()
                scored = score_speech_candidates(backend, row["candidates"], self.policies)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                audio_seconds += len(waveform) / 16000
                dst.write(json.dumps(dict(id=row["id"], candidates=scored), ensure_ascii=False) + "\n")
                dst.flush()
                count += 1
                print(f"{count}/{len(dataset)}: rescoring RTF={elapsed / audio_seconds:.3f}", flush=True)
            if src.readline():
                raise ValueError("Extra candidate segments")
        if not count or not audio_seconds:
            raise ValueError("Empty rescoring dataset")
        with open(self.out_runtime.get_path(), "w") as dst:
            json.dump(dict(rtf=elapsed / audio_seconds, elapsed_seconds=elapsed, audio_seconds=audio_seconds,
                           num_segments=count, batch_size=self.batch_size, dtype="float32", audio_conditioned=True), dst)
        with open(self.out_validation.get_path(), "w") as dst:
            json.dump(validation, dst)


def choose_speech_candidate(candidates, *, n, scale, word_penalty, policy):
    if n < 1 or not candidates or scale < 0 or not np.isfinite([scale, word_penalty]).all():
        raise ValueError("Invalid selection settings")
    return min(candidates[:n], key=lambda c: c["first_pass_cost"] + scale * c["speech"][policy]["nll"]
               + word_penalty * len(c["text"].split()))


class SelectSpeechNbestJob(Job):
    def __init__(self, candidates, generation_runtime, scoring_runtime, *, n, scale, word_penalty, policy, code_version):
        self.candidates, self.generation_runtime, self.scoring_runtime = candidates, generation_runtime, scoring_runtime
        self.n, self.scale, self.word_penalty, self.policy, self.code_version = n, scale, word_penalty, policy, code_version
        self.out_hypotheses = self.output_path("hypotheses.jsonl")
        self.out_runtime = self.output_path("runtime.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        with open(self.candidates.get_path()) as src, open(self.out_hypotheses.get_path(), "w") as dst:
            for line in src:
                row = json.loads(line)
                c = choose_speech_candidate(row["candidates"], n=self.n, scale=self.scale,
                                            word_penalty=self.word_penalty, policy=self.policy)
                dst.write(json.dumps(dict(id=row["id"], text=c["text"]), ensure_ascii=False) + "\n")
        with open(self.generation_runtime.get_path()) as src:
            generation = json.load(src)
        with open(self.scoring_runtime.get_path()) as src:
            scoring = json.load(src)
        if not math.isclose(generation["audio_seconds"], scoring["audio_seconds"], rel_tol=1e-6):
            raise ValueError("Generation/rescoring audio duration differs")
        elapsed = generation["elapsed_seconds"] + scoring["elapsed_seconds"]
        with open(self.out_runtime.get_path(), "w") as dst:
            json.dump(dict(rtf=elapsed / scoring["audio_seconds"], elapsed_seconds=elapsed,
                           audio_seconds=scoring["audio_seconds"], includes_independent_one_best_parity_decode=True,
                           note="Includes all formatting variants and up to 50 candidates; excludes cheap selection/scoring"), dst)


class OracleSpeechNbestJob(Job):
    """Reference-dependent diagnostic only; never a deployable recognition."""
    def __init__(self, references, candidates, *, n, code_version):
        self.references, self.candidates, self.n, self.code_version = references, candidates, n, code_version
        self.out_hypotheses = self.output_path("oracle.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from kaldialign import edit_distance
        from .benchmark_scoring import read_jsonl
        from .hf_scoring import normalize_hf_english, normalize_compound_pair
        refs = read_jsonl(self.references.get_path())
        if any(r["group_id"] != r["id"] or r["chunk_index"] != 0 for r in refs):
            raise ValueError("Dev n-best oracle requires independent utterances")
        refs = {r["id"]: r["reference"] for r in refs}
        seen = set()
        with open(self.candidates.get_path()) as src, open(self.out_hypotheses.get_path(), "w") as dst:
            for line in src:
                row = json.loads(line)
                def errors(c):
                    ref, hyp = normalize_compound_pair(normalize_hf_english(refs[row["id"]]), normalize_hf_english(c["text"]))
                    return edit_distance(ref.split(), hyp.split())["total"]
                best = min(row["candidates"][:self.n], key=errors)
                dst.write(json.dumps(dict(id=row["id"], text=best["text"])) + "\n")
                if row["id"] in seen:
                    raise ValueError("Duplicate oracle segment")
                seen.add(row["id"])
        if seen != set(refs):
            raise ValueError("Incomplete oracle coverage")
