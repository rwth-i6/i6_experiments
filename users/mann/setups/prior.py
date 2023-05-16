from sisyphus import *
from sisyphus.tools import try_get

import os

from i6_experiments.users.mann.experimental.statistics import (
    AllophoneCounts,
    PhonemeCounts,
    AlignmentStatisticsJob,
    ApplyStateTyingToPhonemeStats,
    ApplyStateTyingToAllophoneStats
)
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_experiments.users.mann.setups.reports import DescValueReport, SimpleValueReport
from i6_experiments.users.mann.setups.state_tying import Allophone
from i6_core.util import uopen, get_val, instanciate_delayed

def output(name, value):
    opath = os.path.join(fname, name)
    if isinstance(value, dict):
        tk.register_report(opath, DescValueReport(value))
        return
    tk.register_report(opath, SimpleValueReport(value))

from sisyphus.delayed_ops import DelayedBase

class DelayedGetDefault(DelayedBase):
    def __init__(self, a, b, default=None):
        super().__init__(a, b)
        self.default = default

    def get(self):
        try:
            return try_get(self.a)[try_get(self.b)]
        except KeyError:
            return self.default

class PlotPrior:
    def __init__(self, priors, name):
        self.priors = priors
        self.name = os.path.join(fname, name)
    
    def __call__(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.priors)
        plt.save_fig(self.name)

def plot_prior(priors, fname, name):
    priors = [try_get(v) for v in priors]
    name = os.path.join("output", fname, name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(priors)
    plt.yscale("log")
    plt.savefig(name)

def get_prior_from_transcription(system):
    system.init_oov_lexicon()
    transcribe_job = ApplyLexiconToCorpusJob(
        system.csp["train"].corpus_config.file,
        system.csp["train"].lexicon_config.file
    )

    count_phonemes = PhonemeCounts(
        transcribe_job.out_corpus,
    )

    from recipe.allophones import DumpStateTying
    dst = DumpStateTying(system.csp["train"])
    apply_states = ApplyStateTyingToPhonemeStats(
        count_phonemes.counts, dst.state_tying
    )

    align_stats = AlignmentStatisticsJob(
        system.alignments["train"]["init_align"].value,
        system.csp["train"].acoustic_model_config.allophones.add_from_file,
        system.csp["train"].segment_path.hidden_paths,
        system.csp["train"].concurrent
    )

    # tk.register_output("train.corpus.transcription.xml.gz", transcribe_job.out_corpus)

    # output("phoneme_counts", count_phonemes.counts)
    # output("state_stats", apply_states.stats)
    counts = count_phonemes.counts

    AVERAGE_PHON_FRAMES = 8
    total_frames = align_stats.counts["total_states"]
    non_speech_frames = align_stats.counts["total_states"] \
        - AVERAGE_PHON_FRAMES * count_phonemes.total
    stats = {
        "Total phonemes": count_phonemes.total,
        "Total frames": total_frames,
        "Average phoneme frames": AVERAGE_PHON_FRAMES,
        "Total non speech frames": non_speech_frames,
    }

    counts = apply_states.stats
    counts = [counts[i] / 3 / count_phonemes.total for i in range(211)]

    counts[207] = non_speech_frames / total_frames
    for i in range(207):
        counts[i] *= (1 - counts[207])

    # output("prior_from_transcription", counts)

    # tk.register_callback(plot_prior, counts, "prior_from_transcription.png")

    # output("corpus_totals", stats)
    return counts

def get_prior_from_transcription_new(
        system,
        output_stats=False, 
        num_states=211,
        silence_idx=207,
        hmm_partition=3,
        state_tying=None,
        total_frames=None,
    ):
    # system.init_oov_lexicon()
    transcribe_job = ApplyLexiconToCorpusJob(
        system.crp["train"].corpus_config.file,
        system.crp["train"].lexicon_config.file,
        word_separation_orth="#",
    )

    count_phonemes = AllophoneCounts(
        transcribe_job.out_corpus,
    )

    if state_tying is None:
        from recipe.allophones import DumpStateTying
        dst = DumpStateTying(system.crp["train"])
        state_tying = dst.state_tying
    apply_states = ApplyStateTyingToAllophoneStats(
        count_phonemes.counts, state_tying
    )

    counts = count_phonemes.counts

    AVERAGE_PHON_FRAMES = 8
    non_speech_frames = total_frames \
        - AVERAGE_PHON_FRAMES * count_phonemes.total
    stats = {
        "Total phonemes": count_phonemes.total,
        "Total frames": total_frames,
        "Average phoneme frames": AVERAGE_PHON_FRAMES,
        "Total non speech frames": non_speech_frames,
    }

    counts = apply_states.stats
    def default_get(variable, idx, default=None):
        try:
            return variable[idx]
        except KeyError:
            return default
        
    counts = [DelayedGetDefault(counts, i, 0) / hmm_partition / count_phonemes.total for i in range(num_states)]

    counts[silence_idx] = non_speech_frames / total_frames
    for i in range(num_states):
        if i != silence_idx:
            counts[i] *= 1 - counts[silence_idx]

    if output_stats:
        tk.register_output("train.corpus.transcription.xml.gz", transcribe_job.out_corpus)
        tk.register_output("allophone_counts", count_phonemes.counts)
        if isinstance(state_tying, tk.AbstractPath):
            tk.register_output("state_tying", state_tying)
        # output("phoneme_counts", count_phonemes.counts)
        tk.register_output("state_stats", apply_states.stats)
        # output("prior_from_transcription", counts)
        # tk.register_callback(plot_prior, counts, "prior_from_transcription.png")
        # output("corpus_totals", stats)
    return counts



class PriorFromTranscriptionCounts(Job):
    __sis_hash_exclude__ = {
        "eps": 0.0,
        "num_states": None,
    }

    def __init__(
        self,
        allophone_counts,
        total_count,
        state_tying,
        num_frames,
        average_phoneme_frames=8,
        hmm_partition=None,
        eps=0.0,
        num_states=None,
        silence_phoneme="[SILENCE]"
    ):
        self.allophone_counts = allophone_counts
        self.total_count = total_count
        self.state_tying = state_tying
        self.num_frames = num_frames
        self.average_phoneme_frames = average_phoneme_frames
        self.hmm_partition = hmm_partition
        self.eps = eps
        self.num_states = num_states
        self.silence_phoneme = silence_phoneme

        self.out_prior_txt_file = self.output_path("prior.txt")
        self.out_prior_xml_file = self.output_path("prior.xml")
        self.out_prior_png_file = self.output_path("prior.png")
    
    def tasks(self):
        yield Task("run", mini_task=True)
        yield Task("plot", mini_task=True, resume="plot")
    
    def run(self):
        phon_stats = get_val(self.allophone_counts)
        from collections import defaultdict
        state_stats = defaultdict(int)
        # make lookup table
        allophones = []
        with open(tk.uncached_path(self.state_tying), "r") as f:
            for line in f:
                allophone = Allophone.parse_line(line)
                allophones.append(allophone)
                value = phon_stats.get(allophone.write_phon(), 0)
                state_stats[allophone.idx] += value
        # retrieve statistics
        if self.hmm_partition is None:
            # try to guess from state tying
            self.hmm_partition = max(map(lambda a: a.state, allophones)) + 1
        num_states = instanciate_delayed(self.num_states) \
            or max(map(lambda a: a.idx, allophones)) + 1
        silence_idx = next(filter(lambda a: a.phon == self.silence_phoneme, allophones)).idx

        total_phoneme_count = get_val(self.total_count)
        non_speech_frames = self.num_frames - self.average_phoneme_frames * total_phoneme_count
        counts = [state_stats[i] / self.hmm_partition / total_phoneme_count for i in range(num_states)]

        counts[silence_idx] = non_speech_frames / self.num_frames
        for i in range(num_states):
            if i != silence_idx:
                counts[i] *= (1 - counts[silence_idx])
            counts[i] += self.eps
        
        # save
        import numpy as np
        np.savetxt(self.out_prior_txt_file.get_path(), counts, delimiter=" ")
        with open(self.out_prior_xml_file.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % len(counts)
            )
            f.write(" ".join("%.20e" % np.log(s) for s in counts) + "\n")
            f.write("</vector-f32>")

    def plot(self):
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(self.out_prior_txt_file.get_path(), "rt") as f:
            merged_scores = np.loadtxt(f, delimiter=" ")

        xdata = range(len(merged_scores))
        plt.semilogy(xdata, np.exp(merged_scores))
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig(self.out_prior_png_file.get_path())

def get_prior_from_transcription_job(
    system,
    total_frames=None,
    hmm_partition=None,
    lemma_end_prob=0.0,
    eps=0.0,
    extra_num_states=False,
):
    transcribe_job = ApplyLexiconToCorpusJob(
        system.csp["train"].corpus_config.file,
        system.csp["train"].lexicon_config.file,
        word_separation_orth="#",
    )

    count_phonemes = AllophoneCounts(
        transcribe_job.out_corpus,
        lemma_end_probability=lemma_end_prob,
    )

    from i6_core.lexicon.allophones import DumpStateTyingJob
    dst = DumpStateTyingJob(system.csp["train"])

    prior_job = PriorFromTranscriptionCounts(
        count_phonemes.counts,
        count_phonemes.total,
        dst.out_state_tying,
        num_frames=total_frames,
        hmm_partition=hmm_partition,
        eps=eps,
        num_states=system.num_classes() if extra_num_states else None,
    )

    tk.register_output("prior_from_transcription", prior_job.out_prior_txt_file)
    tk.register_output("prior_from_transcription.png", prior_job.out_prior_png_file)

    return {
        "txt": prior_job.out_prior_txt_file,
        "xml": prior_job.out_prior_xml_file,
        "png": prior_job.out_prior_png_file
    }

class PriorSystem:
    def __init__(self,
        system,
        total_frames=None,
        eps=0.0,
        extra_hmm_partition=None,
        lemma_end_probability=0.0,
        extra_num_states=False,
        legacy=False,
    ):
        self.system = system
        self.total_frames = total_frames
        self.eps = eps
        self.hmm_partition = extra_hmm_partition
        self.lemma_end_probability = lemma_end_probability
        self._eps = None
        self.extra_num_states = extra_num_states
        self.extract_prior()
    
    def get_prior_from_transcription(self):
        total_frames=self.total_frames
        hmm_partition=self.hmm_partition
        lemma_end_prob=self.lemma_end_probability
        eps=self.eps
        extra_num_states=self.extra_num_states

        transcribe_job = ApplyLexiconToCorpusJob(
            self.system.csp["train"].corpus_config.file,
            self.system.csp["train"].lexicon_config.file,
            word_separation_orth="#",
        )

        count_phonemes = AllophoneCounts(
            transcribe_job.out_corpus,
            lemma_end_probability=lemma_end_prob,
        )

        prior_job = PriorFromTranscriptionCounts(
            count_phonemes.counts,
            count_phonemes.total,
            self.system.get_state_tying_file(),
            num_frames=total_frames,
            hmm_partition=hmm_partition,
            eps=eps,
            num_states=self.system.num_classes() if extra_num_states else None,
        )

        return {
            "txt": prior_job.out_prior_txt_file,
            "xml": prior_job.out_prior_xml_file,
            "png": prior_job.out_prior_png_file
        }
    
    def extract_prior(self):
        assert self.total_frames is not None
        self.prior_files = self.get_prior_from_transcription()
        self.prior_txt_file = self.prior_files["txt"]
        self.prior_xml_file = self.prior_files["xml"]
        self.prior_png_file = self.prior_files["png"] 
    
    def add_to_config(self, config):
        from i6_experiments.users.mann.nn import prior
        prior.prepare_static_prior(config, prob=True)
        prior.add_static_prior(config, self.prior_txt_file, eps=self._eps)


class PriorSystemV2(PriorSystem):
    def get_prior_from_transcription(self):
        from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
        lexicon_w_we = AddEowPhonemesToLexiconJob(
            self.system.crp["train"].lexicon_config.file,
            boundary_marker=" #", # important to have space before #
        )

        transcribe_job = ApplyLexiconToCorpusJob(
            self.system.csp["train"].corpus_config.file,
            lexicon_w_we.out_lexicon,
        )

        count_phonemes = AllophoneCounts(
            transcribe_job.out_corpus,
            lemma_end_probability=self.lemma_end_probability,
        )

        prior_job = PriorFromTranscriptionCounts(
            count_phonemes.counts,
            count_phonemes.total,
            self.system.get_state_tying_file(),
            num_frames=self.total_frames,
            hmm_partition=self.hmm_partition,
            eps=self.eps,
            num_states=self.system.num_classes() if self.extra_num_states else None,
        )

        return {
            "txt": prior_job.out_prior_txt_file,
            "xml": prior_job.out_prior_xml_file,
            "png": prior_job.out_prior_png_file
        }