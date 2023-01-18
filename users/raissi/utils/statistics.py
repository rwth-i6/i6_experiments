__all__ = ['CalculateSilenceStateLabel', 'get_prior_from_transcription']
#credits reserved to Daniel Mann
import numpy as np
from sisyphus import tk

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.lib import corpus
from i6_core.rasr.command import RasrCommand
from i6_core.util import uopen, get_val




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


class AlignmentStatisticsJob(Job):

    def __init__(self, alignment, allophones, segments, concurrent, archiver_exe=None):
        self.alignment = alignment
        self.allophones = allophones
        self.segments = segments
        self.concurrent = concurrent

        self.exe = RasrCommand.select_exe(archiver_exe, "archiver")
        self.single_counts = {i: self.output_var("single_counts.{}".format(i)) for i in range(1, self.concurrent + 1)}
        self.counts = self.output_var("counts")

        self.rqmt = {'time': 1,
                     'cpu': 1,
                     'gpu': 0,
                     'mem': 1}

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
            prepended_silence=sum(c[0] for c in silence_counts if len(c) > 0),
            appended_silence=sum(c[-1] for c in silence_counts if len(c) > 0),
            total_silence=sum(sum(c) for c in silence_counts),
            total_states=state_count
        )
        self.single_counts[task_id].set(res)

    def gather(self):
        res = defaultdict(int)
        for i in range(1, self.concurrent + 1):
            cs = self.single_counts[i].get()
            for key, value in cs.items():
                res[key] += value
        self.counts.set(dict(res))


class CalculateSilenceStateLabel(Job):
    def __init__(self, train_crp):
        self.train_crp = train_crp
        self.silence_state_id = self.output_var("silence_state_id")

    def tasks(self):
        yield Task('run', mini_task=True)

    def run(self):
        dst = DumpStateTyingJob(self.train_crp)
        path = dst.state_tying.get_path()
        with open(path, "rt") as f:
            st_file = f.read().splitlines()
        for ele in st_file:
            split_ele = ele.split(" ")
            if split_ele[0] == '[SILENCE]{#+#}@i@f.0':
                self.silence_state_id = int(split_ele[1])
            break



def get_prior_from_transcription(train_crp, avg_phoneme_frame_len=8, num_state_classes=126):
    transcribe_job = ApplyLexiconToCorpusJob(
        train_crp.corpus_config.file,
        train_crp.lexicon_config.file
    )

    count_phonemes = PhonemeCounts(
        transcribe_job.out_corpus,
    )

    dst = DumpStateTyingJob(train_crp)
    apply_states = ApplyStateTyingToPhonemeStats(
        count_phonemes.counts, dst.state_tying
    )

    align_stats = AlignmentStatisticsJob(
        system.alignments["train"]["init_align"].value,
        system.csp["train"].acoustic_model_config.allophones.add_from_file,
        system.csp["train"].segment_path.hidden_paths,
        system.csp["train"].concurrent
    )

    total_frames = align_stats.counts["total_states"]
    non_speech_frames = align_stats.counts["total_states"] \
        - avg_phoneme_frame_len * count_phonemes.total
    stats = {
        "Total phonemes": count_phonemes.total,
        "Total frames": total_frames,
        "Average phoneme frames": avg_phoneme_frame_len,
        "Total non speech frames": non_speech_frames,
    }
    #ToDo: what is this part?
    counts = apply_states.stats
    counts = [counts[i] / 3 / count_phonemes.total for i in range(211)]

    counts[207] = non_speech_frames / total_frames
    for i in range(207):
        counts[i] *= (1 - counts[207])

    return counts
