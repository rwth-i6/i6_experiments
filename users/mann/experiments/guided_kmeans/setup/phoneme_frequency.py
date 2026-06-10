from sisyphus import tk, gs, Job

import inspect
import re
import random
from collections import Counter

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.corpus.stats import CountCorpusWordFrequenciesJob
from i6_core.lib.lexicon import Lexicon
from i6_core.lib.corpus import Corpus

from .corpus_setup import py as setup_corpus

class SampleSegmentsWithMinPhonemeCountJobV(Job):
    def __init__(
        self,
        phonetic_corpus: tk.Path,
        min_phoneme_count: int,
        lexicon: tk.Path,
        exclude_phonemes: set[str] | str = r"\[\S+\]"
    ):
        super().__init__()
        self.phonetic_corpus = phonetic_corpus
        self.min_phoneme_count = min_phoneme_count
        self.lexicon = lexicon
        self.exclude_phonemes = exclude_phonemes

        self.out_segments = self.output_path("sampled_segments.txt")
        self.out_counts = self.output_var("phoneme.counts")
    
    def run(self):
        lexicon = Lexicon()
        lexicon.load(self.lexicon.get_path())
        phoneme_set = set(lexicon.phonemes)
        
        # Exclude specified phonemes
        if isinstance(self.exclude_phonemes, str):
            exclude_pattern = re.compile(self.exclude_phonemes)
            phoneme_set = {p for p in phoneme_set if not exclude_pattern.match(p)}
        elif isinstance(self.exclude_phonemes, set):
            phoneme_set = phoneme_set - self.exclude_phonemes
        

        corpus = Corpus()
        corpus.load(self.phonetic_corpus.get_path())
        

        phoneme_counts = Counter()
        # sample segments at random until each phoneme has at least min_phoneme_count occurrences
        segments_list = list(corpus.segments())
        random.shuffle(list(segments_list))
        random_segment_iter = iter(segments_list)
        sampled_segments = set()
        while any(phoneme_counts[p] < self.min_phoneme_count for p in phoneme_set):
            segment = next(random_segment_iter)
            sampled_segments.add(segment)
            for phoneme in segment.orth.split():
                phoneme_counts[phoneme] += 1
        
        with open(self.out_segments.get_path(), "w") as f:
            for segment in sampled_segments:
                f.write(f"{segment.fullname()}\n")

        self.out_counts.set(dict(phoneme_counts))


def py():
    # print(gs.worker_wrapper)
    # print(inspect.getsource(gs.worker_wrapper))
    # gs.worker_wrapper = None

    setup_result = setup_corpus()

    phoneme_corpus = ApplyLexiconToCorpusJob(setup_result.corpus, setup_result.lexicon).out_corpus
    tk.register_output("corpus/phonemes.xml.gz", phoneme_corpus)

    phoneme_frequencies = CountCorpusWordFrequenciesJob(phoneme_corpus).out_word_counts
    tk.register_output("stats/phoneme.counts", phoneme_frequencies)

    sample_segments = SampleSegmentsWithMinPhonemeCountJobV(
        phonetic_corpus=phoneme_corpus,
        min_phoneme_count=5,
        lexicon=setup_result.lexicon,
    )

    tk.register_output("corpus/min_phon_sample_segments.txt", sample_segments.out_segments)
    tk.register_output("stats/min_phon_sample_phoneme_counts", sample_segments.out_counts)

    return sample_segments.out_segments
