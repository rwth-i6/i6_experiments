"""Full-dev validation of shared rescoring settings using cached Qwen scores.

Selection uses hypotheses and cached scores only. References are consulted for
evaluation afterwards. Removing the first-pass LM column removes fallback costs
as well; that condition is explicitly an ablation, not pure 4gram replacement.
"""

import json
from functools import lru_cache

from sisyphus import Job, Task

from .benchmark_scoring import read_jsonl
from .hf_scoring import normalize_hf_english, normalize_compound_pair


def settings():
    result = []
    for policy in ("lowercase", "initial-capital", "initial-capital-period"):
        for scale in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            for eos, word, token in ((1, 0, 0), (0, 0, 0), (1, -1, 0), (1, 0, -0.5), (1, -1, -0.5)):
                if not scale and (eos, word, token) != (1, 0, 0):
                    continue
                result.append(
                    dict(
                        policy=policy,
                        scale=scale,
                        eos_weight=eos,
                        word_penalty=word,
                        token_penalty=token,
                        retain_first_pass_lm=1,
                    )
                )
        for scale in (0.2, 0.5, 1.0):
            for eos in (0, 1):
                result.append(
                    dict(
                        policy=policy,
                        scale=scale,
                        eos_weight=eos,
                        word_penalty=-2,
                        token_penalty=-1,
                        retain_first_pass_lm=0,
                    )
                )
    # Last grid matches the sample's AM scale .5 and rewards -1/-.5,
    # after dividing the objective by .5. Include scale 1 for that exact point.
    return result


def selection_indices(candidates, grid):
    import numpy as np

    am = np.array([c["am_cost"] for c in candidates])
    lm = np.array([c["lm_cost"] for c in candidates])
    words = np.array([len(c["text"].split()) for c in candidates])
    arrays = {}
    for policy in {g["policy"] for g in grid}:
        scores = [c["qwen"][policy] for c in candidates]
        arrays[policy] = (
            np.array([s["nll"] - s["token_costs"][-1] for s in scores]),
            np.array([s["token_costs"][-1] for s in scores]),
            np.array([len(s["token_costs"]) - 1 for s in scores]),
        )
    return [
        int(
            np.argmin(
                am
                + g["retain_first_pass_lm"] * lm
                + g["scale"] * (arrays[g["policy"]][0] + g["eos_weight"] * arrays[g["policy"]][1])
                + g["word_penalty"] * words
                + g["token_penalty"] * arrays[g["policy"]][2]
            )
        )
        for g in grid
    ]


class EvaluateQwenFullDevJob(Job):
    def __init__(self, references, candidates, *, code_version):
        self.references, self.candidates, self.code_version = references, candidates, code_version
        self.out_summary = self.output_path("summary.json")
        self.out_best = self.output_path("best_hypotheses.jsonl")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=3))

    def run(self):
        from kaldialign import edit_distance

        references = read_jsonl(self.references.get_path())
        refs = {r["id"]: r["reference"] for r in references}
        if len(refs) != len(references) or any(r["group_id"] != r["id"] or r["chunk_index"] for r in references):
            raise ValueError("Requires independent utterances with unique IDs")
        grid = settings()
        totals = [
            dict(ins=0, **{"del": 0}, sub=0, total=0, ref_len=0, improved_utterances=0, worsened_utterances=0)
            for g in grid
        ]
        seen = set()
        choices = []

        @lru_cache(maxsize=20000)
        def normalized(text):
            return normalize_hf_english(text)

        with open(self.candidates.get_path()) as src:
            for line in src:
                row = json.loads(line)
                key, cs = row["id"], row["candidates"]
                if key in seen or key not in refs:
                    raise ValueError("Duplicate/unexpected candidate ID")
                seen.add(key)
                indices = selection_indices(cs, grid)
                counts = {}
                for i in set(indices) | {0}:
                    ref, hyp = normalize_compound_pair(normalized(refs[key]), normalized(cs[i]["text"]))
                    counts[i] = edit_distance(ref.split(), hyp.split())
                baseline = counts[0]["total"]
                for total, i in zip(totals, indices):
                    for field, value in counts[i].items():
                        if field in ("ins", "del", "sub", "total", "ref_len"):
                            total[field] += value
                    total["improved_utterances"] += counts[i]["total"] < baseline
                    total["worsened_utterances"] += counts[i]["total"] > baseline
                choices.append((key, [cs[i]["text"] for i in indices]))
                if len(seen) % 250 == 0:
                    print(f"Evaluated {len(seen)}/{len(refs)} utterances", flush=True)
        if seen != refs.keys():
            raise ValueError("Candidate/reference coverage mismatch")
        results = [
            dict(configuration=g, **t, wer_percent=100 * t["total"] / t["ref_len"]) for g, t in zip(grid, totals)
        ]
        best = min(range(len(results)), key=lambda i: results[i]["wer_percent"])
        with open(self.out_summary.get_path(), "w") as dst:
            json.dump(
                dict(
                    num_utterances=len(seen),
                    num_settings=len(grid),
                    split="dev-other",
                    selection="full-dev tuning; requires held-out evaluation",
                    first_pass_lm_ablation="Removing LM also removes stored fallback penalties",
                    best=results[best],
                    results=sorted(results, key=lambda r: r["wer_percent"]),
                ),
                dst,
                indent=2,
            )
        with open(self.out_best.get_path(), "w") as dst:
            for key, texts in choices:
                dst.write(json.dumps(dict(id=key, text=texts[best])) + "\n")
