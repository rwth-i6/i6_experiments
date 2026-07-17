"""SAE §1c text side — the SIL-augmented phoneme corpus the GAN's discriminator matches against.

Why a new job rather than reusing §0b's 𝒯_φ: `TextToPhonemeJob` writes `" ".join(phons)`, which
destroys word boundaries at write time, so the wav2vec-U silence protocol (`<SIL>` at *word*
boundaries with probability p_sil) cannot be recovered from `tphi.txt.gz`. The lexicon/G2P lookups
are reused verbatim; only the join changes.

Silence protocol mirrors fairseq `scripts/phonemize_with_sil.py` exactly: surround every sentence
with `<SIL>`, insert `<SIL>` after word i (internal boundaries only) with probability p_sil, drop
lines containing any OOV word. Sampled once offline -- fairseq freezes the text before binarizing,
so silence patterns are not resampled per epoch.

p_sil is a *distribution-matching* knob, not an alignment: rVAD removes bulk silence from the audio
but leaves the short inter-word gaps (measured recall 0.13 for 1-2 frame gaps, 0.81 for >=520 ms),
so the residue must be modelled on the text side or the discriminator wins on length alone.
The published values (1.0: 0.25, 2.0: 0.5) are tuned to *their* VAD; ours is swept and checked
against the measured residual-silence rate (see `residual_sil_rate` in features.py).
"""

from __future__ import annotations

import os
import subprocess as sp
from typing import Optional

import numpy as np
from sisyphus import Job, Task, gs, tk

from i6_core.util import uopen
from i6_experiments.users.wu.experiments.posterior_hmm.data.phon_lm import (
    _load_g2p_lexicon,
    _load_lexicon_word_to_phon,
)

SIL = "<SIL>"


def phonemize_line(words, word_to_phon, sil_prob, surround, rng):
    """One line -> phone tokens, or None if any word is OOV.

    Draw order and count mirror fairseq `phonemize_with_sil.py` exactly (one `random_sample(W-1)`
    per kept line, consumed left to right), so a shared seed reproduces its output token for token.
    """
    prons = [word_to_phon.get(w) for w in words]
    if not words or any(p is None for p in prons):
        return None
    phones = [SIL] if surround else []
    draws = rng.random_sample(len(words) - 1) if sil_prob > 0 and len(words) > 1 else None
    for i, pron in enumerate(prons):
        phones.extend(pron.split())
        if draws is not None and i < len(draws) and draws[i] < sil_prob:
            phones.append(SIL)
    if surround:
        phones.append(SIL)
    return phones

# Interpreter of the `w2vu` env (fairseq 0.12.2 + CUDA torch). settings.py's worker_wrapper rewrites
# every worker's argv[0] to the *speech_llm* conda python, so fairseq must be invoked explicitly.
W2VU_PYTHON = tk.Path(
    getattr(gs, "W2VU_PYTHON_EXE", "/e/project1/spell/wu24/env/conda/envs/w2vu/bin/python"),
    hash_overwrite="W2VU_ENV_PYTHON",
)


class PhonemizeWithSilJob(Job):
    """Text -> ARPAbet lines with the wav2vec-U silence protocol. One deterministic pron per word."""

    def __init__(
        self,
        *,
        text_file: tk.Path,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path] = None,
        sil_prob: float = 0.5,
        surround: bool = True,
        seed: int = 0,
        max_lines: Optional[int] = None,
    ):
        super().__init__()
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.sil_prob = sil_prob
        self.surround = surround
        self.seed = seed
        self.max_lines = max_lines

        self.out_text = self.output_path("text.phn.gz")
        self.out_stats = self.output_path("stats.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        w2p = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        if self.g2p_lexicon is not None:
            for w, p in _load_g2p_lexicon(self.g2p_lexicon.get_path()).items():
                w2p.setdefault(w, p)

        rng = np.random.RandomState(self.seed)
        n_in = n_out = n_sil = n_tok = 0
        with uopen(self.text_file.get_path(), "rt") as inf, uopen(self.out_text.get_path(), "wt") as outf:
            for line in inf:
                if self.max_lines is not None and n_in >= self.max_lines:
                    break
                n_in += 1
                phones = phonemize_line(line.split(), w2p, self.sil_prob, self.surround, rng)
                if phones is None:
                    continue

                outf.write(" ".join(phones) + "\n")
                n_out += 1
                n_tok += len(phones)
                n_sil += sum(1 for p in phones if p == SIL)

        with uopen(self.out_stats.get_path(), "wt") as f:
            f.write(
                f"lines_in={n_in}\nlines_out={n_out}\ndropped_oov={n_in - n_out}\n"
                f"sil_prob={self.sil_prob}\nsurround={self.surround}\nseed={self.seed}\n"
                f"tokens={n_tok}\nsil_tokens={n_sil}\nsil_token_rate={n_sil / max(n_tok, 1):.6f}\n"
            )


class FairseqPreprocessTextJob(Job):
    """Binarize the phoneme corpus into fairseq's `text_data` layout (dict.phn.txt + {split}.bin/idx).

    `--thresholdsrc` prunes rare phones (README uses 1000). `--padding-factor 1` suppresses fairseq's
    `madeup` dictionary padding, which would otherwise inflate the generator's output vocabulary with
    dead symbols that the phone-diversity loss then tries to use.
    """

    def __init__(self, *, text_file: tk.Path, threshold: int = 1000, workers: int = 4,
                 python_exe: tk.Path = W2VU_PYTHON):
        super().__init__()
        self.text_file = text_file
        self.threshold = threshold
        self.workers = workers
        self.python_exe = python_exe

        self.out_dir = self.output_path("text_data", directory=True)
        # `unpaired_audio_text` loads exactly `<text_data>/dict.txt` (falling back to
        # `<data>/dict.{labels}.txt`), which is the name fairseq-preprocess writes -- do not rename it.
        self.out_dict = self.output_path("text_data/dict.txt")
        self.rqmt = {"cpu": workers, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        plain = os.path.abspath("train.phn")  # fairseq-preprocess needs an uncompressed --trainpref
        with uopen(self.text_file.get_path(), "rt") as inf, open(plain, "wt") as outf:
            for line in inf:
                outf.write(line)

        cmd = [
            os.fspath(self.python_exe), "-m", "fairseq_cli.preprocess",
            "--dataset-impl", "mmap", "--trainpref", plain,
            "--only-source", "--destdir", self.out_dir.get_path(),
            "--thresholdsrc", str(self.threshold),
            "--padding-factor", "1", "--workers", str(self.workers),
        ]
        print("RUN:", " ".join(cmd), flush=True)
        sp.check_call(cmd)

        d = self.out_dir.get_path()
        for f in ("dict.txt", "train.bin", "train.idx"):
            if not os.path.exists(os.path.join(d, f)):
                raise FileNotFoundError(f"fairseq-preprocess did not write {f} into {d}")
        # `has_unpaired_text` is per split: only train.idx exists, so valid runs label-free and is
        # scored by weighted_lm_ppl -- which is what keeps checkpoint selection unsupervised.
        print("text_data:", sorted(os.listdir(d)), flush=True)
