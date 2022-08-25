from sisyphus import tk, Job, Task

from i6_core import util
from i6_core.lib.lexicon import Lexicon
from i6_core.lib import corpus

class RemovePronunciationVariantsJob(Job):
    def __init__(self, lexicon: tk.Path):
        self.lexicon = lexicon

        self.out_lexicon = self.output_path("lexicon.gz")
    
    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.lexicon.get_path())
        for lemma in lexicon.lemmata:
            if len(lemma.phon) == 0:
                print("Warning: lemma with orth {} has no phon".format(lemma.orth))
                continue
            lemma.phon = [lemma.phon[0]]
        util.write_xml(self.out_lexicon.get_path(), lexicon.to_xml())

class VariantStatisticsJob(Job):
    def __init__(self, bliss_corpus, bliss_lexicon):
        self.bliss_corpus = bliss_corpus
        self.bliss_lexicon = bliss_lexicon

        self.out_lexicon_stats = self.output_var("lexicon.stats")
        self.out_corpus_stats = self.output_var("corpus.stats")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.bliss_lexicon.get_path())

        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        from collections import Counter, defaultdict
        lookup_dict = {}
        variant_counts = Counter()
        for lemma in lexicon.lemmata:
            orth = lemma.orth[0] if isinstance(lemma.orth, list) else lemma.orth
            lookup_dict[orth] = lemma.phon
            variant_counts[len(lemma.phon)] += 1
        lexicon_stats = dict(variant_counts)
        lexicon_stats["total"] = sum(variant_counts.values())
        self.out_lexicon_stats.set(lexicon_stats)

        corpus_counts = Counter()
        for seg in c.segments():
            words = seg.orth.split()
            for word in words:
                phons = lookup_dict[word]
                corpus_counts[len(phons)] += 1
        corpus_stats = dict(corpus_counts)
        corpus_stats["total"] = sum(corpus_counts.values())
        self.out_corpus_stats.set(corpus_stats)
        