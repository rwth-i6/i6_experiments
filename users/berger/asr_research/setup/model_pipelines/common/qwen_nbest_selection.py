"""Cheap development scale tuning over already computed Qwen n-best scores."""

import json
from sisyphus import Job, Task
from .qwen_full_dev import selection_indices


class SelectQwenScaleJob(Job):
    def __init__(self, candidates, scale, *, code_version):
        self.candidates, self.scale, self.code_version = candidates, scale, code_version
        self.out_hypotheses = self.output_path("hypotheses.jsonl")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        setting = dict(policy="initial-capital-period", scale=self.scale, eos_weight=1,
                       word_penalty=0, token_penalty=-0.5, retain_first_pass_lm=1)
        with open(self.candidates.get_path()) as src, open(self.out_hypotheses.get_path(), "w") as dst:
            for line in src:
                row = json.loads(line)
                i = selection_indices(row["candidates"][:50], [setting])[0]
                dst.write(json.dumps(dict(id=row["id"], text=row["candidates"][i]["text"])) + "\n")
