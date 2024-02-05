from enum import Enum, auto
import xml.etree.ElementTree as ET

from sisyphus import tk, Job, Task

from i6_core.util import uopen
from i6_core.lib.corpus import Corpus


class ReplaceUnknownWordsJob(Job):
    def __init__(self, corpus_file, lexicon_file, unknown_token="[UNKNOWN]"):
        self.corpus_file = corpus_file
        self.lexicon_file = lexicon_file
        self.unknown_token = unknown_token
        self.out_corpus_file = self.output_path("corpus.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        with uopen(self.lexicon_file, "rt") as f:
            lex_root = ET.parse(f)

        vocabulary = set([o.text.strip() if o.text else "" for o in lex_root.findall(".//orth")])

        corpus = Corpus()
        corpus.load(self.corpus_file.get_path())

        def replace_func(sc):
            for rec in sc.recordings:
                for seg in rec.segments:
                    seg.orth = " ".join([w if w in vocabulary else self.unknown_token for w in seg.orth.split()])
            for ssc in sc.subcorpora:
                replace_func(ssc)

        replace_func(corpus)

        corpus.dump(self.out_corpus_file.get_path())


class TranscriptionTransform(Enum):
    ALL_LOWERCASE = auto()
    ALL_UPPERCASE = auto()


class TransformTranscriptionsJob(Job):
    def __init__(self, corpus_file: tk.Path, transform: TranscriptionTransform):
        self.corpus_file = corpus_file
        self.transform = transform

        self.out_corpus_file = self.output_path("corpus.xml.gz", cached=True)

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        corpus = Corpus()
        corpus.load(self.corpus_file.get_path())

        if self.transform == TranscriptionTransform.ALL_LOWERCASE:
            transform_func = lambda s: s.lower()
        elif self.transform == TranscriptionTransform.ALL_UPPERCASE:
            transform_func = lambda s: s.upper()
        else:
            raise NotImplementedError

        def replace_func(sc):
            for rec in sc.recordings:
                for seg in rec.segments:
                    seg.orth = transform_func(seg.orth)
            for ssc in sc.subcorpora:
                replace_func(ssc)

        replace_func(corpus)

        corpus.dump(self.out_corpus_file.get_path())
