import pprint

from sisyphus import tk, Job, Task
from i6_core.lib import corpus


class UtteranceLMScoresFromWordScoresFileJob(Job):
    """
    This job collects utterance-wise LM scores for a bliss corpus.
    It expects as inputs a bliss corpus and a file with word scores. The word score file is expected to be created by a
    `ComputePerplexityJob` with the corresponding LM and as text file with each utterance of the corresponding corpus on
    a separate line. It also assumes that the order of utterances in the bliss corpus and the word score file are
    identical.
    """
    def __init__(
        self,
        bliss_corpus: tk.Path,
        word_score_file: tk.Path,
    ):
        self.bliss_corpus = bliss_corpus
        self.word_score_file = word_score_file

        self.out_score_file = self.output_path("utterance_scores")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # load corpus
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get())
        segment_iterator = c.segments()

        utterance_scores = {}
        orth = []
        score = 0.0
        for line in open(self.word_score_file.get(), "r").readlines():
            word, _, word_score = line.split()
            orth.append(word)
            score += float(word_score)
            if word == "\\n":
                seg = next(segment_iterator)
                ref_orth_words = " ".join([w for w in seg.orth.split() if "[" not in w])
                assert ref_orth_words == " ".join(orth[:-1]), (
                    f"segments in corpus and word score file do not match:\n{seg.orth}\n{' '.join(orth[:-1])}"
                )
                utterance_scores[seg.fullname()] = score
                orth = []
                score = 0.0

        with open(self.out_score_file.get(), "w") as f:
            f.write("{\n " + pprint.pformat(utterance_scores, width=140, indent=4, sort_dicts=False)[1:-1] + "\n}")
