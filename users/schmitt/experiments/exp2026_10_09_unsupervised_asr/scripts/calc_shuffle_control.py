#!/usr/bin/env python3
"""
Shuffled-reference control: does a recognition output carry ANY utterance-specific
information, or is it explained entirely by phoneme-frequency coincidence?

Motivation
----------
PER alone cannot answer this. A hypothesis that is pure LM babble still scores a
non-trivial "% Correct" in sclite, because an edit alignment between two phoneme
strings drawn from the same unigram distribution matches ~15-25% of tokens by
chance -- and that chance rate *grows* with hypothesis length, so an
over-generating system looks deceptively good on "Corr".

The control removes the acoustics while holding everything else fixed: score each
hypothesis against a *different* utterance's reference, chosen to have nearly the
same length (rotation within a length-sorted order). Any correctness the system
gets from actually listening to the audio must show up as

    matched Corr  >>  shuffled Corr

If the two are equal, the output is independent of the input audio.

Calibration (dev-other, 400 utts)
---------------------------------
    supervised linear probe (34.7% PER)   matched 67.2   shuffled 16.9   gap +50.3
    unsup CTC greedy        (108.3% PER)  matched 24.1   shuffled 24.2   gap  -0.1
    unsup CTC + LM fusion   ( 83.5% PER)  matched 16.6   shuffled 16.3   gap  +0.3
    unsup AED GAN variants  (~190% PER)   matched ~37    shuffled ~37    gap  ~0.0

So the gap, not the PER, is the quantity that says whether a model has learned
anything. Note the GAN rows: the highest "Corr" of any model here (37.6%) is
*entirely* chance -- it is high only because those systems insert 142% extra
tokens, giving the aligner more chances to hit.

Usage
-----
    ./calc_shuffle_control.py <search_out.py.gz> [more...]
    ./calc_shuffle_control.py --stm OTHER.stm <search_out.py.gz>
    ./calc_shuffle_control.py --num-utts 1000 <search_out.py.gz>

The default --stm is the wo-silence dev-other phoneme reference used by the
sclite scoring jobs. Plain script, no sisyphus graph load.
"""

import argparse
import ast
import gzip
import re
import sys

DEFAULT_STM = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_core/text/convert/"
    "TextDictToStmJob.GlEbcEUNuQ2r/output/corpus.stm"
)

_STM_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(<[^>]*>)?\s*(.*)$")


def load_stm(path):
    ref = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue
            m = _STM_RE.match(line)
            if m:
                ref[m.group(1)] = m.group(7).split()
    assert ref, f"no segments parsed from {path}"
    return ref


def lev_counts(ref, hyp):
    """(corr, sub, del, ins) from an optimal edit alignment, as sclite counts them."""
    m = len(hyp)
    # each cell: (cost, corr, sub, del, ins)
    prev = [(j, 0, 0, 0, j) for j in range(m + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, 0, i, 0)] + [None] * m
        for j in range(1, m + 1):
            match = ref[i - 1] == hyp[j - 1]
            d, u, l = prev[j - 1], prev[j], cur[j - 1]
            cur[j] = min(
                (d[0] + (not match), d[1] + match, d[2] + (not match), d[3], d[4]),
                (u[0] + 1, u[1], u[2], u[3] + 1, u[4]),
                (l[0] + 1, l[1], l[2], l[3], l[4] + 1),
                key=lambda t: t[0],
            )
        prev = cur
    return prev[m][1:]


def score(pairs):
    C = S = D = I = N = 0
    for r, h in pairs:
        c, s, d, i = lev_counts(r, h)
        C, S, D, I, N = C + c, S + s, D + d, I + i, N + len(r)
    return C, S, D, I, N


def report(name, pairs):
    C, S, D, I, N = score(pairs)
    print(
        f"  {name:<26} Corr={100*C/N:5.1f}  Sub={100*S/N:5.1f}  Del={100*D/N:5.1f}  "
        f"Ins={100*I/N:5.1f}  PER={100*(S+D+I)/N:6.1f}"
    )
    return 100 * C / N


def run(path, ref, num_utts, num_trials):
    d = ast.literal_eval(gzip.open(path, "rt").read())
    tags = sorted(set(d) & set(ref))
    assert tags, f"no overlap between {path} and the reference"
    # deterministic subsample: every k-th tag, so no RNG and no seed to remember
    if len(tags) > num_utts:
        step = len(tags) / num_utts
        tags = [tags[int(i * step)] for i in range(num_utts)]
    hyps = {t: d[t][0][1].split() for t in tags}

    print(f"### {path}")
    print(f"  ({len(tags)} utts, {sum(len(ref[t]) for t in tags)} ref tokens)")
    matched = report("matched", [(ref[t], hyps[t]) for t in tags])

    # length-sorted rotation -> wrong utterance, near-identical length
    order = sorted(tags, key=lambda t: len(ref[t]))
    shuffled = []
    for shift in range(1, num_trials + 1):
        perm = order[shift:] + order[:shift]
        shuffled.append(
            report(f"shuffled (rot {shift})", [(ref[a], hyps[b]) for a, b in zip(order, perm)])
        )
    mean = sum(shuffled) / len(shuffled)
    gap = matched - mean
    verdict = "INFORMATIVE" if gap > 3.0 else "NO SIGNAL (chance-level)"
    print(f"  --> matched {matched:.1f} vs shuffled {mean:.1f}   gap {gap:+.1f}   {verdict}\n")
    return gap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("search_out", nargs="+", help="search_out.py.gz file(s)")
    p.add_argument("--stm", default=DEFAULT_STM, help="reference stm (default: wo-sil dev-other)")
    p.add_argument("--num-utts", type=int, default=400, help="utterances to score (default 400)")
    p.add_argument("--num-trials", type=int, default=3, help="shuffle rotations (default 3)")
    args = p.parse_args()

    ref = load_stm(args.stm)
    for path in args.search_out:
        run(path, ref, args.num_utts, args.num_trials)


if __name__ == "__main__":
    sys.exit(main())
