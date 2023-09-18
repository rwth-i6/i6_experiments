from sisyphus import *

Path = setup_path(__package__)

from i6_core.lib import corpus

class CorpusLengthJob(Job):
    def __init__(self, bliss_corpus, segment_path):
        self.bliss_corpus = bliss_corpus
        self.segment_path = segment_path
    
        self.out_duration = self.output_var("duration")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        with open(tk.uncached_path(self.segment_path)) as f:
            segments = set(f.read().splitlines())

        duration = 0
        for segment in c.segments():
            if segment.fullname() in segments:
                duration += segment.end - segment.start 
        
        self.out_duration.set(duration)
    
    @classmethod
    def from_crp(cls, crp):
        return cls(
            crp.corpus_config.file,
            crp.segment_path
        )
