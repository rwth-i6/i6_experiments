"""Oracle-guided dev diagnostics, not a reportable tuning/evaluation set.

References select examples and evaluate rankings only. They never construct
scoring text, prompts or punctuation. All candidates of a selected segment are
retained, so beating a single hand-picked competitor is not counted as success.
"""

import itertools
import json

from sisyphus import Job, Task

from .benchmark_scoring import read_jsonl
from .hf_scoring import normalize_hf_english, normalize_compound_pair
from .nbest_rescoring import lm_surface


def surfaces(text):
    """Reference-independent, deterministic formatting probes."""
    lower = text.lower()
    capital = lm_surface(text, "initial-capital")
    normalized = normalize_hf_english(text)
    result = {
        "lower": lower,
        "capital": capital,
        "period": lm_surface(text, "initial-capital-period"),
        "question": capital.rstrip(".!?") + "?",
        "upper": text.upper(),
        "title": lower.title(),
        "normalized-period": lm_surface(normalized, "initial-capital-period"),
        "pronoun-period": " ".join("I" if w == "i" else w for w in lm_surface(lower, "initial-capital-period").split()),
    }
    # Commas before conjunctions are a deliberately coarse punctuation probe,
    # not an oracle-dependent rewrite or a proposed punctuation restorer.
    words = capital.split()
    for i in range(1, len(words)):
        if words[i].lower() in {"but", "although", "because"}:
            words[i - 1] = words[i - 1].rstrip(",") + ","
    result["clause-period"] = lm_surface(" ".join(words), "initial-capital-period")
    for name, value in result.items():
        if normalize_hf_english(value) != normalized:
            raise ValueError(f"Formatting changes scoring transcription: {name}: {text!r} -> {value!r}")
    return result


def fit_examples(rows, targets):
    """Sweep a shared configuration; never pick a different knob per utterance."""
    import numpy as np

    policies = sorted(rows[0]["candidates"][0]["qwen"])
    grid = itertools.product(
        policies,
        (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
        (0.5, 1.0, 2.0),
        (0.0, 1.0),
        (0.0, 0.5, 1.0),
        (-1.0, 0.0, 1.0),
        (-0.5, 0.0, 0.5),
    )
    arrays = {}
    for row in rows:
        cs = row["candidates"]
        arrays[row["id"]] = dict(
            am=np.array([c["am_cost"] for c in cs]),
            lm=np.array([c["lm_cost"] for c in cs]),
            words=np.array([len(c["text"].split()) for c in cs]),
        )
        for policy in policies:
            scores = [c["qwen"][policy] for c in cs]
            arrays[row["id"]][policy] = (
                np.array([s["nll"] - s["token_costs"][-1] for s in scores]),
                np.array([s["token_costs"][-1] for s in scores]),
                np.array([len(s["token_costs"]) - 1 for s in scores]),
            )
    rankings = []
    for policy, scale, acoustic, keep_lm, eos, word_penalty, token_penalty in grid:
        excess = matches = exact = 0
        chosen = {}
        for row in rows:
            cs, a, target = row["candidates"], arrays[row["id"]], targets[row["id"]]
            text_nll, eos_nll, tokens = a[policy]
            costs = (
                acoustic * a["am"]
                + keep_lm * a["lm"]
                + scale * (text_nll + eos * eos_nll)
                + word_penalty * a["words"]
                + token_penalty * tokens
            )
            i = int(np.argmin(costs))
            errors = target["candidate_errors"][i]
            excess += errors - target["oracle_errors"]
            matches += errors == target["oracle_errors"]
            exact += cs[i]["text"] == target["oracle_text"]
            chosen[row["id"]] = cs[i]["text"]
        rankings.append(
            dict(
                excess_errors=excess,
                oracle_error_matches=matches,
                exact_oracle_matches=exact,
                policy=policy,
                scale=scale,
                acoustic_scale=acoustic,
                retain_first_pass_lm=keep_lm,
                eos_weight=eos,
                word_penalty=word_penalty,
                token_penalty=token_penalty,
                chosen=chosen,
            )
        )
    rankings.sort(key=lambda r: (r["excess_errors"], -r["exact_oracle_matches"]))
    return dict(
        oracle_guided=True,
        num_examples=len(rows),
        num_configurations=len(rankings),
        warning="Selected dev examples only; no unbiased WER claim. Validate shared settings on all dev data.",
        best=rankings[:10],
    )


class SelectQwenOracleExamplesJob(Job):
    def __init__(self, references, candidates, selected, oracle, *, code_version, num_examples=6):
        self.references, self.candidates = references, candidates
        self.selected, self.oracle = selected, oracle
        self.code_version, self.num_examples = code_version, num_examples
        self.out_candidates = self.output_path("candidates.jsonl")
        self.out_examples = self.output_path("examples.json")
        self.out_cached_sweep = self.output_path("cached_sweep.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        from kaldialign import edit_distance

        refs = {r["id"]: r["reference"] for r in read_jsonl(self.references.get_path())}
        selected = {r["id"]: r for r in read_jsonl(self.selected.get_path())}
        oracle = {r["id"]: r for r in read_jsonl(self.oracle.get_path())}
        gaps = sorted(
            (
                (-(selected[k]["counts"]["total"] - oracle[k]["counts"]["total"]), k)
                for k in selected
                if selected[k]["counts"]["total"] > oracle[k]["counts"]["total"]
            )
        )
        # Largest gaps are useful probes, not a representative sample.
        ids = {k for _, k in gaps[: self.num_examples]}
        rows, targets = [], {}
        with open(self.candidates.get_path()) as src:
            for line in src:
                row = json.loads(line)
                if row["id"] not in ids:
                    continue
                k = row["id"]
                errors = []
                for c in row["candidates"]:
                    ref, hyp = normalize_compound_pair(normalize_hf_english(refs[k]), normalize_hf_english(c["text"]))
                    errors.append(edit_distance(ref.split(), hyp.split())["total"])
                targets[k] = dict(
                    reference=refs[k],
                    selected_text=selected[k]["text"],
                    selected_errors=selected[k]["counts"]["total"],
                    oracle_text=oracle[k]["text"],
                    oracle_errors=min(errors),
                    candidate_errors=errors,
                )
                # No references/oracle texts in the input to the GPU scorer.
                rows.append(
                    dict(
                        id=k,
                        candidates=[
                            {key: c[key] for key in ("text", "am_cost", "lm_cost", "first_pass_cost", "qwen")}
                            for c in row["candidates"]
                        ],
                    )
                )
        if len(rows) != len(ids):
            raise ValueError("Missing selected candidate lists")
        with open(self.out_candidates.get_path(), "w") as dst:
            for row in rows:
                dst.write(json.dumps(row) + "\n")
        with open(self.out_examples.get_path(), "w") as dst:
            json.dump(targets, dst, indent=2)
        with open(self.out_cached_sweep.get_path(), "w") as dst:
            json.dump(fit_examples(rows, targets), dst, indent=2)


class ProbeQwenOracleExamplesJob(Job):
    def __init__(self, candidates, repo, *, code_version):
        self.candidates, self.repo, self.code_version = candidates, repo, code_version
        self.out_scores = self.output_path("scores.jsonl")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=2, mem=16, gpu=1, gpu_mem=24, time=2))

    def run(self):
        import torch
        from .dynamic_llm import QwenContinuationScorer

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        scorer = QwenContinuationScorer(self.repo.get_path(), device="cuda", batch_size=8)
        rows = read_jsonl(self.candidates.get_path())
        prompts = {"plain": "", "transcript": "Transcript: ", "speech": "The following is a transcription of speech:\n"}
        requests = {}
        for row in rows:
            for c in row["candidates"]:
                for style, surface in surfaces(c["text"]).items():
                    for prompt, prefix in prompts.items():
                        requests[(prefix, surface)] = None
        keys = list(requests)
        prefixes = [scorer.initial_tokens() + scorer.tokenizer.encode(p, add_special_tokens=False) for p, s in keys]
        continuations = [
            scorer.tokenizer.encode(s, add_special_tokens=False) + scorer.sentence_end_tokens() for p, s in keys
        ]
        costs = scorer.score_continuations(prefixes, continuations)
        for key, ids, token_costs in zip(keys, continuations, costs):
            requests[key] = dict(surface=key[1], token_ids=ids, token_costs=token_costs, nll=sum(token_costs))
        with open(self.out_scores.get_path(), "w") as dst:
            for row in rows:
                for c in row["candidates"]:
                    c["qwen"] = {
                        f"{prompt}/{style}": requests[(prefix, surface)]
                        for style, surface in surfaces(c["text"]).items()
                        for prompt, prefix in prompts.items()
                    }
                dst.write(json.dumps(row) + "\n")


class AnalyzeQwenOracleProbesJob(Job):
    def __init__(self, examples, scores, *, code_version):
        self.examples, self.scores, self.code_version = examples, scores, code_version
        self.out_report = self.output_path("probe_sweep.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=1))

    def run(self):
        with open(self.examples.get_path()) as src:
            targets = json.load(src)
        with open(self.out_report.get_path(), "w") as dst:
            json.dump(fit_examples(read_jsonl(self.scores.get_path()), targets), dst, indent=2)
