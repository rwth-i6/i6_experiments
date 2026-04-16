from sisyphus import Job, Task
from i6_core.lib import corpus
from i6_core.util import uopen


class GetSeqTagsFromCorpusJob(Job):
    """
    Extract seq_tags from a Bliss corpus and store as raw txt or gzipped txt
    """

    def __init__(self, bliss_corpus, gzip=False):
        """

        :param Path bliss_corpus: Bliss corpus
        :param bool gzip: gzip the output text file
        """
        self.set_vis_name("Extract TXT from Corpus")

        self.bliss_corpus = bliss_corpus
        self.gzip = gzip

        self.out_txt = self.output_path("segments.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        with uopen(self.out_txt.get_path(), "wt") as f:
            for segment in c.segments():
                f.write(segment.fullname() + "\n")