from sisyphus import Job, Task
from i6_core.util import uopen


class BpeVocabToVocabFileJob(Job):
    def __init__(self, bpe_vocab):
        self.bpe_vocab = bpe_vocab

        self.out_vocab = self.output_path("bpe.vocab", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with uopen(self.bpe_vocab, "rt") as in_f:
            with uopen(self.out_vocab, "wt") as out_f:
                for line in in_f:
                    if line.startswith("{") or line.startswith("}") or "</s>" in line:
                        continue
                    token, idx = line.split()
                    token = token[1:-2]  # Change "'<token>':" to "<token>"
                    idx = idx[:-1]  # Remove ending comma
                    out_f.write(f"{token} {idx}\n")
