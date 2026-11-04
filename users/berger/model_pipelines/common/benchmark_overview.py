"""Explicit corpus-resource rows in the benchmark result overview."""

import json

from sisyphus import Job, Task


class BenchmarkResourceOverviewJob(Job):
    def __init__(self, scores, runtimes, resources):
        self.scores, self.runtimes, self.resources = scores, runtimes, resources
        self.out_report = self.output_path("resource-overview.md")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out_report.get_path(), "w") as output:
            output.write("# Open ASR results by lexicon and LM source\n\n")
            output.write(
                "Every registered corpus-resource comparison is shown explicitly. "
                "Corpus lexicon + LibriSpeech Transformer is a transferred LM, "
                "not a corpus-trained Transformer.\n\n"
            )
            output.write(
                "| Model | Dataset | Condition | Lexicon | LM | WER (%) | RTF |\n|---|---|---|---|---|---:|---:|\n"
            )
            for key, datasets in sorted(self.scores.items()):
                model, condition = key.split("/", 1)
                resource = self.resources[key]
                for dataset, path in sorted(datasets.items()):
                    with open(path.get_path()) as source:
                        score = json.load(source)
                    with open(self.runtimes[key][dataset].get_path()) as source:
                        runtime = json.load(source)
                    output.write(
                        f"| {model} | {dataset} | {condition} | {resource['lexicon']} | {resource['lm']} "
                        f"| {score['wer_percent']:.3f} | {runtime['rtf']:.4f} |\n"
                    )
