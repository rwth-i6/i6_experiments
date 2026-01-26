import os.path
from typing import Iterator

from sisyphus import Job, Task, tk
import enum
import random
import numpy as np

from i6_core.lib import corpus
from i6_core.lib.corpus import Corpus, Recording, Segment
from i6_core.lib.lexicon import Lexicon
from i6_core.corpus.transform import ApplyLexiconToCorpusJob, MergeStrategy, MergeCorporaJob

class LexiconStrategy(enum.Enum):
    PICK_FIRST = 0

class ApplyLexiconToTranscriptions(ApplyLexiconToCorpusJob):
    """
        Placeholder only
    """
    def __init__(
        self,
        bliss_corpus,
        bliss_lexicon,
        word_separation_orth,
        strategy=LexiconStrategy.PICK_FIRST,
    ):
        super().__init__(
            bliss_corpus=bliss_corpus,
            bliss_lexicon=bliss_lexicon,
            word_separation_orth=word_separation_orth,
            strategy=strategy
        )


class RandomizeWordOrderJob(Job):
    """

    """
    def __init__(self, bliss_corpus: tk.Path):
        """

        :param bliss_corpus: corpus xml
        """
        self.bliss_corpus = bliss_corpus

        self.out_corpus = self.output_path("random.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        c = Corpus()
        c.load(self.bliss_corpus.get())
        random.seed(42)

        for r in c.all_recordings():
            for s in r.segments:
                words = s.orth.split()
                random.shuffle(words)
                s.orth = " ".join(words)

        c.dump(self.out_corpus.get())


class FakeCorpusFromLexicon(Job):
    """

    """
    def __init__(self, lexicon: tk.Path, num_sequences, len_min, len_max):
        """

        :param lexicon:
        """
        self.lexicon = lexicon
        self.num_sequences = num_sequences
        self.len_min = len_min
        self.len_max = len_max

        self.out_corpus = self.output_path("lex_random_corpus.xml.gz")

    def run(self):
        lex = Lexicon()
        lex.load(self.lexicon.get())

        rand = np.random.RandomState(seed=42)

        word_list = []
        for lemma in lex.lemmata:
            if lemma.special is not None:
                continue
            for orth in lemma.orth:
                word_list.append(orth)
        word_array = np.asarray(word_list)

        c = Corpus()
        for i in range(self.num_sequences):
            rec = Recording()
            rec.name = "non_rec_%i" % i
            seg = Segment()
            seg.name = "lex_random_seg_%i" % i
            length = rand.randint(low=self.len_min, high=self.len_max + 1)
            seg.orth = " ".join(rand.choice(a=word_array, size=length))
            rec.add_segment(seg)
            c.add_recording(rec)

        c.dump(self.out_corpus.get())



class RandomAssignSpeakersFromCorpus(Job):
    """
    Takes another bliss corpus as speaker reference and randomly distributes speaker tags

    Used e.g. for synthetic TTS data
    """


    __sis_hash_exclude__ = {"randomize_speaker": True}

    def __init__(self, bliss_corpus: tk.Path, speaker_reference_bliss_corpus: tk.Path, seed: int = 42,
                 randomize_speaker: bool = True):
        """

        :param bliss_corpus: bliss corpus to assign speakers to
        :param speaker_reference_bliss_corpus: bliss corpus to take speakers from
        :param seed: random seed for deterministic behavior
        :param randomize_speaker:
        True: samples speakers at random
        False: every speaker appears at least once if bliss_size > num_speakers
        """
        self.bliss_corpus = bliss_corpus
        self.speaker_reference_bliss_corpus = speaker_reference_bliss_corpus
        self.seed = seed
        self.randomize_speaker = randomize_speaker
        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        # set seed for deterministic behavior
        np.random.seed(self.seed)

        out_corpus = Corpus()
        out_corpus.load(self.bliss_corpus.get_path())

        speaker_corpus = Corpus()
        speaker_corpus.load(self.speaker_reference_bliss_corpus.get_path())

        out_corpus.speakers = speaker_corpus.speakers

        # shuffle speaker list for randomization of sequence to speaker mapping
        speaker_name_list = list(out_corpus.speakers.keys())
        random.shuffle(speaker_name_list)
        num_speakers = len(speaker_name_list)

        for recording in out_corpus.all_recordings():
            recording.speaker_name = None
            recording.default_speaker = None
            for segment in recording.segments:
                if self.randomize_speaker:
                    segment.speaker_name = speaker_name_list[np.random.randint(num_speakers)]
                else:
                    if len(speaker_name_list)>0:
                        segment.speaker_name = speaker_name_list.pop()
                    else:
                        # re-initialize random speaker list
                        speaker_name_list = list(out_corpus.speakers.keys())
                        random.shuffle(speaker_name_list)
                        segment.speaker_name = speaker_name_list.pop()

        out_corpus.dump(self.out_corpus.get_path())


class MergeCorporaWithPathResolveJob(MergeCorporaJob):
    """
    Merges Bliss Corpora files into a single file as subcorpora or flat

    resolves relative paths to absolute
    """

    def run(self):
        merged_corpus = corpus.Corpus()
        merged_corpus.name = self.name
        for corpus_path in self.bliss_corpora:
            c = corpus.Corpus()
            c.load(corpus_path.get_path())

            # Make all audio paths absolute
            corpus_dir = os.path.dirname(corpus_path.get_path())
            for recording in c.all_recordings():
                absolute_audio = os.path.join(corpus_dir, recording.audio)
                assert os.path.exists(absolute_audio)
                recording.audio = absolute_audio

            if self.merge_strategy == MergeStrategy.SUBCORPORA:
                merged_corpus.add_subcorpus(c)
            elif self.merge_strategy == MergeStrategy.FLAT:
                for rec in c.all_recordings():
                    merged_corpus.add_recording(rec)
                merged_corpus.speakers.update(c.speakers)
            elif self.merge_strategy == MergeStrategy.CONCATENATE:
                for subcorpus in c.top_level_subcorpora():
                    merged_corpus.add_subcorpus(subcorpus)
                for rec in c.top_level_recordings():
                    merged_corpus.add_recording(rec)
                for speaker in c.top_level_speakers():
                    merged_corpus.add_speaker(speaker)
            else:
                assert False, "invalid merge strategy"

        merged_corpus.dump(self.out_merged_corpus.get_path())
