"""Fixed dev-selected two-pass Qwen recognition, including grouped corpora."""

import json
import time

from sisyphus import Job, Task

from .nbest_rescoring import GraniteNbestJob, ScoreQwenNbestJob
from .qwen_full_dev import selection_indices


class CompactGraniteNbestJob(GraniteNbestJob):
    """Keep full parity checks but do not serialize every candidate's trace."""

    def run(self):
        import sys
        import numpy as np
        import torch
        import transformers
        from datasets import load_from_disk
        from .hf_benchmark import HfBenchmarkAdapter, audio_array, traceback_result

        if transformers.__version__ != "5.16.1":
            raise RuntimeError("Use benchmark transformers==5.16.1")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        adapter = HfBenchmarkAdapter(self.repo.get_path(), "ctc", "float32")
        sys.path.insert(0, self.binary.get_path())
        import librasr

        cfg = librasr.Configuration()
        if not cfg.set_from_file(self.rasr_config.get_path()):
            raise ValueError("Invalid config")
        cfg.set("lib-rasr.search-algorithm.log-stepwise-statistics", "false")
        for channel in ("log", "warning", "error", "statistics"):
            cfg.set(f"*.{channel}.channel", self.out_log.get_path())
        librasr.init_logging(cfg)
        search = librasr.SearchAlgorithm(config=cfg)
        dataset = load_from_disk(self.dataset.get_path())
        sizes, audio_seconds = [], 0.0
        started = time.perf_counter()
        with open(self.out_candidates.get_path(), "w") as dst, open(self.out_first_pass.get_path(), "w") as first_out:
            for i, sample in enumerate(dataset):
                waveform = audio_array(sample["audio"])
                with torch.inference_mode():
                    features = adapter.encode_inputs(adapter.inputs(waveform))
                features = np.ascontiguousarray(features, dtype=np.float32)
                traces = search.recognize_segment_n_best(features, self.n)
                first = traceback_result(search.recognize_segment(features), adapter, {}, False)
                candidates = {}
                for rank, trace in enumerate(traces):
                    result = traceback_result(trace, adapter, {}, False)
                    am = float(trace[-1].am_score) if trace else 0.0
                    lm = float(trace[-1].lm_score) if trace else 0.0
                    if not np.isfinite(am + lm):
                        raise ValueError("Non-finite score")
                    c = dict(text=result["text"], am_cost=am, lm_cost=lm, first_pass_cost=am + lm, native_rank=rank)
                    if c["text"] not in candidates or c["first_pass_cost"] < candidates[c["text"]]["first_pass_cost"]:
                        candidates[c["text"]] = c
                ordered = sorted(candidates.values(), key=lambda c: c["first_pass_cost"])
                trace = first["traceback"]
                cost = trace[-1]["am_score"] + trace[-1]["lm_score"] if trace else 0
                if (
                    not ordered
                    or ordered[0]["text"] != first["text"]
                    or abs(ordered[0]["first_pass_cost"] - cost) > 0.01
                ):
                    raise ValueError(f"N-best/one-best parity failure: {sample['id']}")
                sizes.append(len(ordered))
                audio_seconds += len(waveform) / 16000
                dst.write(json.dumps(dict(id=str(sample["id"]), candidates=ordered)) + "\n")
                first_out.write(json.dumps(dict(id=str(sample["id"]), text=first["text"])) + "\n")
                dst.flush()
                print(f"{i + 1}/{len(dataset)}: {len(ordered)} candidates", flush=True)
        elapsed = time.perf_counter() - started
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(
                dict(
                    requested_n=self.n,
                    num_segments=len(sizes),
                    mean_unique=sum(sizes) / len(sizes),
                    min_unique=min(sizes),
                    max_unique=max(sizes),
                    audio_seconds=audio_seconds,
                    elapsed_seconds=elapsed,
                    rtf=elapsed / audio_seconds,
                    includes_independent_one_best_parity_decode=True,
                ),
                dst,
                indent=2,
            )


class FixedScoreQwenNbestJob(ScoreQwenNbestJob):
    def __init__(self, candidates, repo, *, code_version):
        super().__init__(candidates, repo, code_version=code_version, batch_size=8)
        self.policies = ("initial-capital-period",)
        self.rqmt.update(mem=64, time=24)


class SelectFixedQwenNbestJob(Job):
    def __init__(self, candidates, *, code_version):
        self.candidates, self.code_version = candidates, code_version
        self.out_hypotheses = self.output_path("hypotheses.jsonl")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        setting = dict(
            policy="initial-capital-period",
            scale=0.3,
            eos_weight=1,
            word_penalty=0,
            token_penalty=-0.5,
            retain_first_pass_lm=1,
        )
        with open(self.candidates.get_path()) as src, open(self.out_hypotheses.get_path(), "w") as dst:
            for line in src:
                row = json.loads(line)
                i = selection_indices(row["candidates"][:50], [setting])[0]
                dst.write(json.dumps(dict(id=row["id"], text=row["candidates"][i]["text"])) + "\n")


class QwenBenchmarkOverviewJob(Job):
    def __init__(self, scores, runtimes, resources):
        self.scores, self.runtimes, self.resources = scores, runtimes, resources
        self.out_report = self.output_path("overview.md")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_report.get_path(), "w") as dst:
            dst.write("# Granite + Qwen: fixed LibriSpeech settings\n\n")
            dst.write(
                "Loquacious subsets share their parent's decoding; no subset-specific RTF is claimed. "
                "Two-pass RTF includes the independent one-best parity decode.\n\n"
            )
            dst.write("| Dataset | Method | First-pass lexicon | LM | WER (%) | RTF |\n|---|---|---|---|---:|---:|\n")
            for method, datasets in sorted(self.scores.items()):
                resource = self.resources[method]
                for name, path in sorted(datasets.items()):
                    with open(path.get_path()) as src:
                        score = json.load(src)
                    runtime = self.runtimes[method].get(name)
                    rtf = "—"
                    if runtime is not None:
                        with open(runtime.get_path()) as src:
                            rtf = f"{json.load(src)['rtf']:.4f}"
                    dst.write(
                        f"| {name} | {method} | {resource['lexicon']} | {resource['lm']} | "
                        f"{score['wer_percent']:.3f} | {rtf} |\n"
                    )


class QwenTwoPassRuntimeJob(Job):
    def __init__(self, generation, lm_scoring):
        self.generation, self.lm_scoring = generation, lm_scoring
        self.out_report = self.output_path("runtime.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.generation.get_path()) as src:
            generation = json.load(src)
        with open(self.lm_scoring.get_path()) as src:
            lm = json.load(src)
        elapsed = generation["elapsed_seconds"] + lm["elapsed_seconds"]
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(
                dict(
                    rtf=elapsed / generation["audio_seconds"],
                    elapsed_seconds=elapsed,
                    generation_rtf=generation["rtf"],
                    qwen_rtf=lm["elapsed_seconds"] / generation["audio_seconds"],
                    includes_independent_one_best_parity_decode=True,
                    note="Conservative job-wall RTF; generation includes parity re-decoding; excludes selection/scoring",
                ),
                dst,
                indent=2,
            )
