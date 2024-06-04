from sisyphus import *
from sisyphus.tools import try_get

import os

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob


from i6_experiments.users.mann.experimental.statistics import AllophoneCounts
from i6_experiments.users.mann.setups.prior import PriorFromTranscriptionCounts


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


def get_prior_from_transcription(
    crp,
    total_frames,
    average_phoneme_frames,
    epsilon=1e-12,
    lemma_end_probability=0.0,

):

    lexicon_w_we = AddEowPhonemesToLexiconJob(
        crp.lexicon_config.file,
        boundary_marker=" #", # the prepended space is important
    )

    corpus = crp.corpus_config.file
    if not isinstance(crp.corpus_config.file, tk.Path):
        corpus = tk.Path(crp.corpus_config.file)


    transcribe_job = ApplyLexiconToCorpusJob(
        corpus,
        lexicon_w_we.out_lexicon,
    )

    count_phonemes = AllophoneCounts(
        transcribe_job.out_corpus,
        lemma_end_probability=lemma_end_probability,
    )

    state_tying_file = DumpStateTyingJob(crp).out_state_tying



    prior_job = PriorFromTranscriptionCounts(
        allophone_counts=count_phonemes.counts,
        total_count=count_phonemes.total,
        state_tying=state_tying_file,
        average_phoneme_frames=average_phoneme_frames,
        num_frames=total_frames,
        eps=epsilon,
    )

    return {
        "txt": prior_job.out_prior_txt_file,
        "xml": prior_job.out_prior_xml_file,
        "png": prior_job.out_prior_png_file
    }