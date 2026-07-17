"""Ground-truth test: our SIL protocol must reproduce fairseq's `phonemize_with_sil.py` exactly.

The reference script has no --seed, so it is copied with `np.random.seed(0)` injected. Our
`phonemize_line` consumes a RandomState(0) with the same draw order/count, so a match is
token-for-token, not merely distributional.

Run:  PYTHONPATH=recipe:tools/sisyphus:recipe/i6_models \
      /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python -m pytest <this file> -q
"""

import os
import subprocess as sp
import sys
import tempfile

import numpy as np

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import SIL, phonemize_line

_REF = "/e/project1/spell/wu24/2026-07-13_unsupervised/tools/w2vu_reference/fs_phonemize_with_sil.py"

_LEX = {"THE": "DH AH", "CAT": "K AE T", "SAT": "S AE T", "ON": "AA N", "MAT": "M AE T", "A": "AH"}
_TEXT = [
    "THE CAT SAT ON THE MAT",
    "A CAT",
    "THE MAT",
    "CAT SAT ON A MAT THE CAT",
    "A",
    "THE UNKNOWNWORD CAT",  # dropped: OOV
]


def _ours(sil_prob, surround, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for line in _TEXT:
        p = phonemize_line(line.split(), _LEX, sil_prob, surround, rng)
        if p is not None:
            out.append(" ".join(p))
    return out


def _fairseq(sil_prob, surround, seed=0):
    with tempfile.TemporaryDirectory() as d:
        lex = os.path.join(d, "lex.txt")
        with open(lex, "w") as f:
            for w, p in _LEX.items():
                f.write(f"{w} {p}\n")
        ref = os.path.join(d, "ref_seeded.py")
        with open(_REF) as f:
            src = f.read().replace("import numpy as np", f"import numpy as np\nnp.random.seed({seed})", 1)
        with open(ref, "w") as f:
            f.write(src)
        cmd = [sys.executable, ref, "--lexicon", lex, "--sil-prob", str(sil_prob)]
        if surround:
            cmd.append("--surround")
        r = sp.run(cmd, input="\n".join(_TEXT) + "\n", capture_output=True, text=True, check=True)
        return r.stdout.splitlines()


def test_matches_fairseq_reference():
    for sil_prob in (0.0, 0.25, 0.5, 1.0):
        for surround in (True, False):
            assert _ours(sil_prob, surround) == _fairseq(sil_prob, surround), (sil_prob, surround)


def test_oov_line_dropped():
    assert len(_ours(0.5, True)) == len(_TEXT) - 1


def test_surround_and_sil_bounds():
    # p_sil=0 -> silence only at the two sentence edges; p_sil=1 -> every internal boundary too.
    none_ = _ours(0.0, True)[0].split()
    assert none_[0] == SIL and none_[-1] == SIL and none_.count(SIL) == 2
    all_ = _ours(1.0, True)[0].split()
    assert all_.count(SIL) == 2 + (len(_TEXT[0].split()) - 1)
    assert _ours(0.0, False)[0].split().count(SIL) == 0
