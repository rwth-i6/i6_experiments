from sisyphus import Job, Task, tk
import gzip

from i6_core.lib.corpus import Corpus, Recording, Segment

class CreateBlissFromTextLinesJob(Job):

    def __init__(self, text_file: tk.Path, corpus_name: str, sequence_prefix: str):
        """

        :param text_file:
        :param corpus_name:
        :param sequence_prefix:
        """

        self.text_file = text_file
        self.corpus_name = corpus_name
        self.sequence_prefix = sequence_prefix

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        c = Corpus()
        c.name = self.corpus_name

        fopen = gzip.open if self.text_file.get_path().endswith(".gz") else open
        text = fopen(self.text_file.get_path(), "rt")

        for i, line in enumerate(text):
            r = Recording()
            r.name = "recording_%i" % i
            s = Segment()
            s.start = 0
            s.end = len(line)
            s.orth = line
            s.name = self.sequence_prefix + "_%i" % i
            r.add_segment(s)
            c.add_recording(r)

        c.dump(self.out_corpus.get_path())
