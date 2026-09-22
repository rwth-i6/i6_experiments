"""Translating this setup's RASR search parameters into a `backoff_fb` graph.

Runtime logic only -- no sisyphus -- so it sits in ``lib`` beside
:mod:`..parallel_recognizer` rather than in ``setup``.

The CUDA forward-backward op (``tools/backoff_fb``, see its SPEC.md) takes a
compiled graph rather than a RASR config. The three things that have to line up
are the label inventory, the emission sign, and the HMM topology, and each of
them is easy to get subtly wrong:

* **Labels.** The FB search reads scores by lemma id, so score column *k* is
  lemma *k* of the forward-backward lexicon: id 0 is the silence lemma (LM
  token ``<SIL>``) and 1..N are the phonemes in the lexicon's own order. The LM
  symbol list handed to the compiler must be exactly that sequence, which is
  why :func:`lexicon_symbols` reads the lexicon rather than sorting the ARPA's
  unigrams. Sorting happens to give the same answer here -- ``<`` sorts before
  ``A`` -- and relying on that coincidence would break silently the first time
  a phoneme set changed.

* **Sign.** ``ScoreModel.scores`` returns *costs* (squared Euclidean or
  Mahalanobis distances, lower is better); the op wants emission
  log-likelihoods. The conversion is ``o = -distance_scale * scores``, applied
  in the recognizer.

* **Topology.** ``loop_probability`` and ``silence_loop_probability`` are
  per-phoneme, and this setup runs them at 0: the features are segment-pooled
  (one vector per alignment segment, silence dropped), so a segment is one
  phoneme and there is no self-loop to take. ``HmmTopology.uniform`` cannot
  express that -- it calls ``math.log(p_loop)`` -- so the arrays are built here.
"""

from __future__ import annotations

__all__ = ["ensure_importable", "lexicon_symbols", "build_topology",
           "build_graph", "GraphSpec"]

import dataclasses
import gzip
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree

import numpy as np

def ensure_importable(root) -> None:
    """Put the op on ``sys.path``. Idempotent, and safe to call from anywhere.

    Every ``import backoff_fb`` in this setup has to be preceded by this, so
    call it explicitly rather than relying on some other import having run
    first: a job that imports ``backoff_fb.batching`` before touching this
    module otherwise dies with ModuleNotFoundError inside the task, which is
    the expensive place to find out.

    `root` comes from ``tools.BACKOFF_FB``. It must be a path the job container
    binds -- /work/asr4 is, /u/mann is not (see settings.py:worker_wrapper).
    """
    root = str(root)
    if not os.path.isdir(root):
        raise ValueError(
            f"backoff_fb root {root!r} is not a directory. Inside a job this "
            f"usually means the path is not bound into the container; "
            f"settings.py:worker_wrapper binds /work/asr4 but not /u/mann.")
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_backoff_fb(root):
    ensure_importable(root)
    import backoff_fb  # noqa: E402
    return backoff_fb


#: Lemmas that exist for the search's bookkeeping rather than as labels. The
#: LM handles sentence boundaries itself (BOS and EOS are not in Sigma) and
#: `<unk>` n-grams are dropped at load, so none of them is a score column.
NON_LABEL_TOKENS = frozenset({"<s>", "</s>", "<unk>", "<UNK>"})


def _lemma_token(lemma) -> str:
    """A lemma's LM token: its ``<synt>`` when it has one (the silence lemma
    carries ``<SIL>``), otherwise its ``<orth>``."""
    synt = lemma.find("synt")
    if synt is not None:
        toks = [t.text for t in synt.findall("tok") if t.text]
        return toks[0] if toks else (synt.text or "").strip()
    orth = lemma.find("orth")
    return (orth.text or "").strip() if orth is not None else ""


def lexicon_symbols(lexicon_path: str) -> List[str]:
    """LM symbols in score-column order, read off the forward-backward lexicon.

    The FB search reads scores by lemma id, so this is the mapping from score
    column to LM token. The lexicon also carries `<unk>`, `<s>` and `</s>`
    lemmas, which are not labels; `create_fb_lexicon` builds it with
    ``phonemes_before_special=True`` precisely so the label lemmas occupy ids
    0..N-1, and that is asserted here rather than assumed -- with the specials
    first every score column would be off by three.
    """
    opener = gzip.open if str(lexicon_path).endswith(".gz") else open
    with opener(lexicon_path, "rt") as f:
        root = ElementTree.parse(f).getroot()

    tokens = []
    for i, lemma in enumerate(root.findall("lemma")):
        token = _lemma_token(lemma)
        if not token:
            raise ValueError(f"lemma {i} in {lexicon_path} has no usable LM token")
        tokens.append(token)
    if not tokens:
        raise ValueError(f"no lemmas found in {lexicon_path}")

    labels = [t for t in tokens if t not in NON_LABEL_TOKENS]
    if tokens[: len(labels)] != labels:
        raise ValueError(
            f"{lexicon_path} interleaves non-label lemmas with the labels "
            f"({tokens}); the forward-backward lexicon must be built with "
            f"phonemes_before_special=True or the score columns do not line up")
    return labels


def build_topology(
    symbols: List[str],
    *,
    loop_probability: float,
    silence_loop_probability: Optional[float] = None,
    transition_scale: float = 1.0,
    silence_symbol: str = "<SIL>",
    backoff_fb_root=None,
):
    """An S=1 left-to-right topology matching the RASR search parameters.

    With S = 1 there is one state per phoneme: it loops with probability
    ``p`` and exits with ``1 - p``. ``p = 0`` (this setup's setting) gives
    ``tau_loop = -inf``, which the op handles -- ``LSE(-inf, x) = x``, so every
    frame must then arrive through an LM transition.
    """
    if backoff_fb_root is not None:
        ensure_importable(backoff_fb_root)
    from backoff_fb_ref import HmmTopology

    V = len(symbols)
    if silence_loop_probability is None:
        silence_loop_probability = loop_probability

    def log(p: float) -> float:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"loop probability must be in [0, 1), got {p}")
        return -math.inf if p == 0.0 else math.log(p)

    tau_loop = np.empty((V, 1), dtype=np.float64)
    tau_exit = np.empty(V, dtype=np.float64)
    for a, sym in enumerate(symbols):
        p = silence_loop_probability if sym == silence_symbol else loop_probability
        tau_loop[a, 0] = transition_scale * log(p)
        # log1p(-p) rather than log(1 - p): exact at the small-p end, and equal
        # to 0.0 at p = 0 where the exit is certain.
        tau_exit[a] = transition_scale * math.log1p(-p)

    # S = 1, so there is no forward arc; compile.py expects -inf in the last
    # column regardless.
    tau_fwd = np.full((V, 1), -math.inf, dtype=np.float64)
    # Label k is emitted by phoneme k, which is what makes the score columns
    # and the emission labels the same index.
    emis = np.arange(V, dtype=np.int32).reshape(V, 1)
    return HmmTopology(V=V, S=1, tau_loop=tau_loop, tau_fwd=tau_fwd,
                       tau_exit=tau_exit, emis=emis)


@dataclass(frozen=True)
class GraphSpec:
    """Everything that determines the compiled graph, and nothing else.

    Used as the cache key, so it must not carry scheduling knobs.
    """
    arpa_path: str
    lexicon_path: str
    lm_scale: float
    loop_probability: float
    silence_loop_probability: Optional[float]
    transition_scale: float
    #: Where the op is checked out; see ``tools.BACKOFF_FB``.
    backoff_fb_root: str = ""
    #: Whether a path pays ``ln P(</s> | g)`` to terminate (spec 2.3).
    #: True is the correct model and what KenLM scores against. RASR's
    #: forward-backward does *not* apply it -- measured: its log Z converges
    #: from below onto the apply_sentence_end=False value as its beam grows
    #: (1000 -> 10k -> 50k -> 200k gives -136.56, -134.65, -134.31, -134.26
    #: against -134.07, while the correct value is -138.16). Set False only to
    #: reproduce the RASR arm exactly.
    apply_sentence_end: bool = True


#: One compiled graph per process. Compiling the 5-gram takes ~86 s and each
#: chunk task is a fresh process, so this is per-task rather than global; it
#: still saves recompiling for every epoch within a task.
_CACHE: Dict[Tuple, object] = {}


def build_graph(spec: GraphSpec, device: str = "cuda"):
    """Compile (or fetch) the `BackoffHmmGraph` for `spec`."""
    key = (spec, device)
    if key in _CACHE:
        return _CACHE[key]

    bf = _import_backoff_fb(spec.backoff_fb_root)
    from backoff_fb_ref import compile_graph

    symbols = lexicon_symbols(spec.lexicon_path)
    # The label inventory and the LM's own inventory must be the same set, or
    # some column is being scored against the wrong symbol's n-grams.
    arpa_symbols = bf.phonemes_from_arpa(spec.arpa_path)
    if set(symbols) != set(arpa_symbols):
        only_lex = sorted(set(symbols) - set(arpa_symbols))
        only_lm = sorted(set(arpa_symbols) - set(symbols))
        raise ValueError(
            f"lexicon and LM disagree on the label inventory: "
            f"{len(symbols)} lexicon labels vs {len(arpa_symbols)} LM symbols; "
            f"only in lexicon: {only_lex}; only in LM: {only_lm}")
    lm = bf.load_arpa(spec.arpa_path, symbols)
    topo = build_topology(
        symbols,
        loop_probability=spec.loop_probability,
        silence_loop_probability=spec.silence_loop_probability,
        transition_scale=spec.transition_scale,
    )
    compiled = compile_graph(lm, topo, lm_scale=spec.lm_scale, check=False)
    if not spec.apply_sentence_end:
        # compile_graph sets phi(g) = tau_exit[last g] + ln P(</s> | g); drop
        # the LM term so a path may end anywhere, as RASR's search does.
        # dataclasses.replace rather than mutation: the compiled object is
        # shared with whatever else holds it.
        compiled = dataclasses.replace(
            compiled,
            chain_final=topo.tau_exit[compiled.chain_last].astype(np.float64))
    graph = bf.BackoffHmmGraph.from_compiled(compiled, device=device)
    _CACHE[key] = graph
    return graph
