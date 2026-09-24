"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_jobs.py (the window-replay step of
``LexiconTrieBuildJob.run``, split out as its own job in the port).

:class:`WordWindowReplayJob` writes the word lines of the prior's window -- the text the in-house word
trigram is trained on -- so that i6_core's ``KenLMplzJob`` and ``CreateBinaryLMJob`` can train and
binarise that trigram as jobs of their own (:func:`.word_lm.get_lexicon_trie`).  The replay itself is
``prior_gap``'s, called with the arguments ``LexiconTrieBuildJob.run`` passed it: the phonemization
lexicon, the counted/held split of the window and the verified replay of the window's word lines.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from sisyphus import Job, Task, tk

__all__ = ["WordWindowReplayJob"]


class WordWindowReplayJob(Job):
    """The counted window lines as words (``out_words``) and as phones (``out_phones``).

    :param bliss_lexicon: ``PhonemizeWithSilJob``'s own bliss lexicon.
    :param g2p_lexicon: its Sequitur g2p lexicon (``None`` = bliss only).
    :param word_corpus: the text the phonemization ran on.
    :param window_phn: the banked ``SampleLinesJob`` window the live prior is fitted on.
    :param n_window_lines: the window's line count (``None`` = the prior's count + held lines).
    :param held_stride: every ``held_stride``-th window line is held out (``None`` = the prior's).
    :param sample_seed: the ``SampleLinesJob`` seed of the window.

    ``out_stats`` holds the replay's counts (kept corpus lines, window lines, counted lines).
    """

    def __init__(
        self,
        *,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path],
        word_corpus: tk.Path,
        window_phn: tk.Path,
        n_window_lines: Optional[int] = None,
        held_stride: Optional[int] = None,
        sample_seed: int = 0,
    ):
        super().__init__()
        from .phone_prior import DEFAULT_COUNT_LINES, DEFAULT_HELD_LINES, HELD_STRIDE

        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.word_corpus = word_corpus
        self.window_phn = window_phn
        self.n_window_lines = int(
            DEFAULT_COUNT_LINES + DEFAULT_HELD_LINES if n_window_lines is None else n_window_lines)
        self.held_stride = int(HELD_STRIDE if held_stride is None else held_stride)
        self.sample_seed = int(sample_seed)
        self.out_words = self.output_path("window.words.txt")
        self.out_phones = self.output_path("window.phn.txt")
        self.out_stats = self.output_path("window.json")
        #: the Step 0 job ran this replay and lmplz together in 302.85 s on 4 CPUs / 16 GB
        self.rqmt = {"cpu": 4, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .prior_gap import load_phonemization_lexicon, replay_window_texts, window_line_flags

        out_dir = os.path.abspath("window")
        os.makedirs(out_dir, exist_ok=True)
        w2p_all = load_phonemization_lexicon(
            self.bliss_lexicon.get_path(),
            None if self.g2p_lexicon is None else self.g2p_lexicon.get_path(),
        )
        flags, _held = window_line_flags(self.window_phn.get_path(), n_lines=self.n_window_lines,
                                         held_stride=self.held_stride)
        window = replay_window_texts(
            word_corpus=self.word_corpus.get_path(), window_phn=self.window_phn.get_path(),
            word_to_phones=w2p_all, counted=flags, out_dir=out_dir,
            n_window_lines=self.n_window_lines, sample_seed=self.sample_seed)
        print(f"window: {window['counted_lines']} counted lines", flush=True)
        shutil.move(window["words"], self.out_words.get_path())
        shutil.move(window["phones"], self.out_phones.get_path())
        with open(self.out_stats.get_path(), "w") as fh:
            json.dump({k: window[k] for k in sorted(window) if isinstance(window[k], (int, float))},
                      fh, indent=2, sort_keys=True)
