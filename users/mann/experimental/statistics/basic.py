__all__ = [
    "AllophoneCounts",
    "PhonemeCounts",
    "SilenceAtSegmentBoundaries",
    "AlignmentStatisticsJob",
    "ApplyStateTyingToPhonemeStats",
    "ApplyStateTyingToAllophoneStats"
]
from sisyphus import *

Path = setup_path(__package__)

import sys, os
import subprocess
import matplotlib.pyplot as plt
from itertools import filterfalse
from tabulate import tabulate
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Union

# import recipe.lib.sprint_cache as sc
from i6_core.lib import rasr_cache as sc
from i6_core.lib import corpus
from i6_core.rasr.command import RasrCommand
from i6_core.util import uopen, get_val

from i6_experiments.users.mann.setups.state_tying import Allophone

def allophone_and_state_generator(lines):
    for l in lines:
        if "<" in l:
            continue
        l.replace("\n", "")
        parts = l.split("\t")
        if len(parts) < 10:
            continue
        state = int(parts[-1])
        allophone = parts[5]
        yield allophone, state

class StateParser:
    def __init__(self, lines):
        self.lines = lines
        self._line_iterator = iter(lines)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # l = "" # dummy
        l = next(self._line_iterator, None)
        while l is not None: 
            if "<" in l:
                l = next(self._line_iterator, None)
                continue
            l.replace("\n", "")
            parts = l.split("\t")
            if len(parts) < 10:
                l = next(self._line_iterator, None)
                continue
            state = int(parts[-1])
            allophone = parts[5]
            return allophone, state
        raise StopIteration
    
    def __reversed__(self):
        return type(self)(reversed(self.lines))


class SilenceCounter:
    def __init__(self):
        self.last_silence = False
        self.counts = []
    
    def _incr_current(self):
        self.counts[-1] += 1
    
    def _new_counter(self):
        self.counts += [0]
    
    def feed(self, allo, *_ignored):
        if "#" in allo:
            if not self.last_silence:
                self._new_counter()
            self._incr_current()
            self.last_silence = True
            return
        # else
        self.last_silence = False

class ContextualSilenceCounter:
    """Counts only segment inner silence.
    Counts silence insertions.
    """
    def __init__(self):
        self.last_silence = False
        self.sil_count = 0
        self.word_end_count = 0
        self.is_inner = False
        self.last_allo = None
    
    @classmethod
    def is_silence(cls, allo):
        return "#" in allo
    
    @classmethod
    def is_speech_end(cls, allo):
        return "@f" in allo and not cls.is_silence(allo)

    @classmethod
    def is_lemma_start(cls, allo):
        return "@i" in allo
    
    def feed(self, allo, *_ignored):
        if self.last_allo is None:
            self.last_allo = allo
            return
        if self.is_speech_end(self.last_allo):
            if self.is_lemma_start(allo):
                self.word_end_count += 1
            if self.is_silence(allo):
                self.sil_count += 1
        self.last_allo = allo
    
    def maybe_remove_last(self):
        if self.is_silence(self.last_allo):
            if self.sil_count > 0:
                self.sil_count -= 1
            if self.word_end_count > 0:
                self.word_end_count -= 1

def resolve_bundle(alignment):
    if not isinstance(alignment, (str, tk.Path)):
        return alignment
    with uopen(alignment) as f:
        alignment = {i: line.rstrip("\n") for i, line in enumerate(f, 1)}
    return alignment

class AlignmentStatisticsJob(Job):

    def __init__(self, alignment, allophones, segments, concurrent, archiver_exe=None):
        # self.csp = csp
        self.alignment = alignment
        self.allophones = allophones
        self.segments = segments
        self.concurrent = concurrent

        self.exe = RasrCommand.select_exe(archiver_exe, "archiver")
        self.single_counts = {i: self.output_var("single_counts.{}".format(i)) for i in range(1, self.concurrent + 1)}
        self.counts = self.output_var("counts")

        self.rqmt = {'time': 1,
                    'cpu' : 1,
                    'gpu' : 0,
                    'mem' : 1}

    def tasks(self):
        yield Task('run', resume='run', rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task('gather', resume='gather', mini_task=True)
    
    def archive(self, alignment, segment):
        args = [self.exe,
                '--allophone-file',
                tk.uncached_path(self.allophones),
                '--mode', 'show',
                '--type', 'align',
                alignment, segment]
        res = subprocess.run(args,
                            stdout=subprocess.PIPE)
        lines = res.stdout.decode('utf-8').split('\n')
        return lines
    

    def run(self, task_id):
        alignment_path = tk.uncached_path(resolve_bundle(self.alignment)[task_id])
        segment_path = tk.uncached_path(self.segments[task_id])

        silence_counts = []
        state_count = 0

        with open(segment_path, "r") as segment_file:
            for seg in segment_file:
                lines = self.archive(alignment_path, seg.rstrip("\n"))

                sil_counter = SilenceCounter()
                for allo, state in allophone_and_state_generator(lines):
                    sil_counter.feed(allo)
                    state_count += 1

                silence_counts.append(sil_counter.counts)
        
        res = dict(
            prepended_silence = sum(c[0] for c in silence_counts if len(c) > 0),
            appended_silence = sum(c[-1] for c in silence_counts if len(c) > 0),
            total_silence = sum(sum(c) for c in silence_counts),
            total_states = state_count
        )
        self.single_counts[task_id].set(res)
    
    def gather(self):
        res = defaultdict(int)
        for i in range(1, self.concurrent + 1):
            cs = self.single_counts[i].get()
            for key, value in cs.items():
                res[key] += value
        self.counts.set(dict(res))


class SilenceAtSegmentBoundaries(AlignmentStatisticsJob):

    def run(self, task_id):
        alignment_path = tk.uncached_path(resolve_bundle(self.alignment)[task_id])
        segment_path = tk.uncached_path(self.segments[task_id])

        silence_counts = defaultdict(int)

        with open(segment_path, "r") as segment_file:
            for seg in segment_file:
                lines = self.archive(alignment_path, seg.rstrip("\n"))

                allo_gen = StateParser(lines)
                
                first_allo, _ = next(allo_gen)
                if "#" in first_allo:
                    silence_counts["start"] += 1
                
                last_allo, _ = next(reversed(allo_gen))
                if "#" in last_allo:
                    silence_counts["end"] += 1

                silence_counts["total"] += 1
        
        self.single_counts[task_id].set(dict(silence_counts))
    
    def gather(self):
        counter = defaultdict(int)
        for i in range(1, self.concurrent + 1):
            cs = self.single_counts[i].get()
            for key, value in cs.items():
                counter[key] += value
        # normalize
        res = {
            "start": counter["start"] / counter["total"],
            "end"  : counter["end"]   / counter["total"],
            "total": counter["total"]
        } 
        self.counts.set(res)


class PositionalSilenceCounter(AlignmentStatisticsJob):

    def run(self, task_id):
        alignment_path = tk.uncached_path(resolve_bundle(self.alignment)[task_id])
        segment_path = tk.uncached_path(self.segments[task_id])

        silence_counts = []

        with open(segment_path, "r") as segment_file:
            for seg in segment_file:
                lines = self.archive(alignment_path, seg.rstrip("\n"))

                sil_counter = SilenceCounter()
                for allo, state in allophone_and_state_generator(lines):
                    sil_counter.feed(allo)

                silence_counts.append(sil_counter.counts)
        
        res = dict(
            start_loops = sum(c[0] - 1 for c in silence_counts),
            end_loops   = sum(c[-1] - 1 for c in silence_counts),
            inner_loops = sum(sum(c[1:-1]) - len(c) + 2 for c in silence_counts),
            inner_count = sum(len(c) - 2 for c in silence_counts),
            seg_count   = len(silence_counts)
        )
        self.single_counts[task_id].set(res)
    
    def gather(self):
        counter = defaultdict(int)
        for i in range(1, self.concurrent + 1):
            cs = self.single_counts[i].get()
            for key, value in cs.items():
                counter[key] += value
        # normalize
        res = {
            "start_loops": counter["start_loops"] / counter["seg_count"],
            "end_loops"  : counter["end_loops"]   / counter["seg_count"],
            "inner_loops": counter["inner_loops"] / counter["inner_count"]
        } 
        self.counts.set(res)


class SilenceBetweenWords(AlignmentStatisticsJob):

    def run(self, task_id):
        alignment_path = tk.uncached_path(resolve_bundle(self.alignment)[task_id])
        segment_path = tk.uncached_path(self.segments[task_id])

        silence_counts = defaultdict(int)

        with open(segment_path, "r") as segment_file:
            for seg in segment_file:
                lines = self.archive(alignment_path, seg.rstrip("\n"))

                sil_counter = ContextualSilenceCounter()
                for allo, state in allophone_and_state_generator(lines):
                    sil_counter.feed(allo)
                sil_counter.maybe_remove_last()

                silence_counts["silence_insertions"] += sil_counter.sil_count
                silence_counts["word_transitions"]   += sil_counter.word_end_count
        
        self.single_counts[task_id].set(
            dict(silence_counts)
        )
    
    def gather(self):
        counter = defaultdict(int)
        for i in range(1, self.concurrent + 1):
            cs = self.single_counts[i].get()
            for key, value in cs.items():
                counter[key] += value
        # normalize
        self.counts.set(counter["silence_insertions"] / counter["word_transitions"])

class AllophoneSequencer:
    def __init__(self, corpus, lexicon, state_tying, hmm_partition):
        self.corpus = corpus
        self.lexicon = lexicon
        self.state_tying = state_tying
        self.states_per_phone = hmm_partition

        self.lexicon_dict = {}
        self.corpus_dict = {}
        self.state_tying_dict = {}
    
    def init_lexicon_dict(self):
        lex = lexicon.Lexicon()
        lex.load(self.lexicon.get_path())
        # build lookup dict
        for lemma in lex.lemmata:
            for orth in lemma.orth:
                if orth:
                    self.lexicon_dict[orth] = [phon.split(" ") for phon in lemma.phon]
        return

    def init_state_tying_dict(self):
        state_tying = self.state_tying.get_path()
        with open(state_tying, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                allo = Allophone.parse_line(line)
                self.state_tying_dict[allo.write(omit_idx=True)] = allo.idx
        return

    def init_corpus_dict(self):
        c = corpus.Corpus()
        c.load(self.corpus.get_path())
        for segment in c.segments():
            words = segment.orth.split(" ")
            self.corpus_dict[segment.fullname()] = words
        return

    def get_allophone_sequence(self, seq_tag: str) -> List[str]:
        word_seq = self.corpus_dict[seq_tag]
        allo_seq = []
        for word in word_seq:
            allo_seq.append(
                Allophone("[SILENCE]", "#", "#", initial=True, final=True, state=0)
            )
            allo_seq.append(
                [self.get_lemma(phon_seq) for phon_seq in self.lexicon_dict[word][:1]]
            )
        allo_seq.append(
            Allophone("[SILENCE]", "#", "#", initial=True, final=True, state=0)
        )

class SkipCounter:
    def __init__(self):
        self.counter = 0
    
    def set_allophone_sequence(self, allophone_sequence):
        self.allophone_sequence = allophone_sequence
    
    def feed(self, allophone):
        if allophone == self.allophone_sequence[self.counter]:
            self.counter += 1
        if self.counter == len(self.allophone_sequence):
            self.counter = 0


class SkipCountsJob(AlignmentStatisticsJob):

    def __init__(self,
        alignment,
        allophones,
        segments,
        corpus,
        lexicon,
        concurrent,
        archiver_exe=None
    ):
        super().__init__(alignment, allophones, segments, concurrent, archiver_exe)
        self.lexicon = lexicon
        self.corpus = corpus
    
    def run(self, task_id):
        alignment_path = tk.uncached_path(resolve_bundle(self.alignment)[task_id])
        segment_path = tk.uncached_path(self.segments[task_id])

        allophone_seqs = AllophoneSequencer(self.corpus, self.lexicon, None, hmm_partition=None)
        allophone_seqs.init_lexicon_dict()
        allophone_seqs.init_corpus_dict()




class PhonemeCounts(Job):
    def __init__(self, bliss_corpus):
        self.bliss_corpus = bliss_corpus
        self.counts = self.output_var("counts")
        self.total = self.output_var("total")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))
        
        from collections import Counter
        counts = Counter()
        total = 0
        for segment in c.segments():
            phons = segment.orth.split(" ")
            counts.update(phons)
            total += len(phons)
        self.counts.set(dict(counts))
        self.total.set(total)

class VirtualAllophone:
    def __init__(self, allophone, virtual):
        self.allophone = allophone
        self.virtual = virtual

class AllophoneCounter:
    def __init__(self):
        self.counter = Counter()
        self.virtual_counter = Counter

    @staticmethod
    def get_allophones(subsegment):
        is_initial = (subsegment[1] == "#")
        is_final = (subsegment[3] == "#")
        is_virtual = (subsegment[0] != "#" or subsegment)
        yield Allophone(subsegment[2], subsegment[1], subsegment[3], is_initial, is_final)
        if is_initial and subsegment[0] != "#":
            yield Allophone(subsegment[2], subsegment[0], subsegment[3], is_initial, is_final)
        if is_final and subsegment[4] != "#":
            yield Allophone(subsegment[2], subsegment[1], subsegment[4], is_initial, is_final)
        if is_initial and subsegment[0] != "#" and is_final and subsegment[4] != "#":
            yield Allophone(subsegment[2], subsegment[0], subsegment[4], is_initial, is_final)
    
    def update(self, segment):
        phons = ["#"] * 2 + segment.orth.split(" ") + ["#"] * 2
        for subsegment in zip(phon[i:] for i in range(5)):
            is_initial = (subsegment[1] == "#")
            is_final = (subsegment[3] == "#")
            if is_initial:
                pass



class AllophoneCounts(Job):
    def __init__(self, bliss_corpus, lemma_end_probability=0.0):
        assert 0.0 <= lemma_end_probability <= 1.0
        self.bliss_corpus = bliss_corpus
        self.lemma_end_probability = lemma_end_probability
        self.counts = self.output_var("counts")
        self.total = self.output_var("total")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        c = corpus.Corpus()
        c.load(tk.uncached_path(self.bliss_corpus))
        
        from collections import Counter
        from itertools import chain
        from functools import reduce
        import numpy as np
        counts = Counter()
        left_ambiguous_counts = Counter()
        right_ambiguous_counts = Counter()
        both_ambiguous_counts = Counter()
        lemma_start_counts = Counter()
        lemma_end_counts = Counter()
        print("new routine")
            
        for segment in c.segments():
            phons = ["#"] * 2 + segment.orth.split(" ") + ["#"] * 2
            for subsegment in zip(phons[0:], phons[1:], phons[2:], phons[3:], phons[4:]): # center, second left, left, right, second right
                if subsegment[2] == "#": # not a valid center phoneme
                    continue
                center_idx = 2
                l2, l1, c, r1, r2 = subsegment
                center = subsegment[center_idx]
                is_speech = lambda p: ("[{}]".format(p[1:-1]) != p) # is not of the form [ABC]
                is_initial = (l1 == "#")
                is_final = (r1 == "#")

                allo = lambda c, l, r: Allophone(c, l, r, is_initial, is_final).write_phon()
                is_left_ambiguous = is_initial and is_speech(c) and is_speech(l2)
                is_right_ambiguous = is_final and is_speech(c) and is_speech(r2)
                if is_left_ambiguous and is_right_ambiguous:
                    both_ambiguous_counts.update([allo(c, l2, r2)])
                elif is_left_ambiguous:
                    left_ambiguous_counts[allo(c, l2, r1)] += 1
                elif is_right_ambiguous:
                    right_ambiguous_counts[allo(c, l1, r2)] += 1
                else:
                    counts[allo(c, l1, r1)] += 1
        
        for a, c in left_ambiguous_counts.items():
            counts[a] += c * (1 - self.lemma_end_probability)
            counts[a.replace("prev", "#")] += c * self.lemma_end_probability
        for a, c in right_ambiguous_counts.items():
            counts[a] += c * (1 - self.lemma_end_probability)
            counts[a.replace("right", "#")] += c * self.lemma_end_probability
        for a, c in both_ambiguous_counts.items():
            counts[a] += c * (1 - self.lemma_end_probability)**2
            counts[a.replace("prev", "#")] += c * self.lemma_end_probability * (1 - self.lemma_end_probability)
            counts[a.replace("next", "#")] += c * self.lemma_end_probability * (1 - self.lemma_end_probability)
            counts[a.replace("next", "#").replace("prev", "#")] += c * self.lemma_end_probability**2

        self.counts.set(dict(counts))
        self.total.set(sum(counts.values()))


class ApplyStateTyingToPhonemeStats(Job):
    def __init__(self, phoneme_stats, state_tying):
        self.phoneme_stats = phoneme_stats
        self.state_tying = state_tying

        self.stats = self.output_var("stats")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        phon_stats = get_val(self.phoneme_stats)
        state_stats = {}
        # make lookup table
        with open(tk.uncached_path(self.state_tying), "r") as f:
            for line in f:
                phon = line.split("{")[0]
                state = int(line.split(" ")[-1])
                value = phon_stats.get(phon, 0)
                if state in state_stats:
                    assert value == state_stats[state], "State-tying file is possibly inconsistent " \
                        + "for phoneme {} and state {}".format(phon, state)
                    continue
                state_stats[state] = value
        
        # save
        self.stats.set(state_stats)
                
class ApplyStateTyingToAllophoneStats(Job):
    def __init__(self, allophone_stats, state_tying):
        self.allophone_stats = allophone_stats
        self.state_tying = state_tying

        self.stats = self.output_var("stats")
    
    def tasks(self):
        yield Task("run", mini_task=True)
    
    def run(self):
        phon_stats = get_val(self.allophone_stats)
        from collections import defaultdict
        state_stats = defaultdict(int)
        # make lookup table
        with open(tk.uncached_path(self.state_tying), "r") as f:
            for line in f:
                allophone = Allophone.parse_line(line)
                value = phon_stats.get(allophone.write_phon(), 0)
                state_stats[allophone.idx] += value
        # save
        self.stats.set(dict(state_stats))
