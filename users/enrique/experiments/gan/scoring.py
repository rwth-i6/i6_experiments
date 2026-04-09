import gzip
import json
import math
import os
from random import random
import shutil
import subprocess as sp
from typing import Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
import i6_core.util as util
import numpy as np
import glob
from sisyphus import tools
import gc


from sisyphus import tk
from sisyphus.job_path import Variable

from recipe.i6_core.tools.git import CloneGitRepositoryJob
from recipe.i6_core.lm.kenlm import CompileKenLMJob


def load_vocab(path):
    vocab = set()
    for line in open(path):
        tok = line.split()[0].strip()
        if tok:
            vocab.add(tok)
    return vocab

def score_with_lm(seqs, lm, vocab, sil=None):
    """
    Implements the LM scoring + vocabulary usage exactly as in the wav2vec-U paper.
    Returns:
       avg_NLL   : average per-utterance negative log-likelihood per token
       total_logprob : sum of all log probabilities across tokens (natural log)
       vocab_usage: fraction of vocab types used
    """
    total_logprob = 0.0
    total_nll_sum = 0.0
    total_utts = 0
    vocab_used = set()

    ln10 = math.log(10.0)

    for seq in seqs:
        # remove SIL
        toks = [t for t in seq if t != sil] if sil else seq
        if not toks:
            continue

        sent = " ".join(toks)

        # kenlm gives log10 probability
        log10_p = lm.score(sent, bos=False, eos=False)
        ln_p = log10_p * ln10

        M = len(toks)

        avg_nll_utt = -(ln_p) / M

        total_nll_sum += avg_nll_utt
        total_logprob += ln_p
        total_utts += 1

        # vocabulary usage
        for t in toks:
            if t in vocab:
                vocab_used.add(t)

    if total_utts == 0:
        avg_nll = float("inf")
    else:
        avg_nll = total_nll_sum / total_utts

    vocab_usage = len(vocab_used) / max(1, len(vocab))

    return avg_nll, total_logprob, vocab_usage, len(vocab_used)

class KenLMScoreJob(Job):
    def __init__(
        self,
        kenlm_binary_path: tk.Path,
        lm_path: tk.Path,
        vocab_path: tk.Path,
        input_text: tk.Path,
        sil: str = "<SIL>",
    ):
        self.kenlm_binary_path = kenlm_binary_path
        self.lm_path = lm_path
        self.vocab_path = vocab_path
        self.input_text = input_text
        self.sil = sil

        self.output_score = self.output_var("score")

        self.rqmt = {"time": 100, "cpu": 1, "mem": 4}