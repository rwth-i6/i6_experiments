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
           "build_graph", "GraphSpec", "LabelLayout"]

import contextlib
import dataclasses
import gzip
import math
import os
import shutil
import sys
import tempfile
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


@dataclass(frozen=True)
class LabelLayout:
    """How score columns map onto phonemes and HMM states.

    Tied (the default, and every run before untied states existed): all ``S``
    states of a phoneme emit that phoneme's label, so column ``k`` *is* phoneme
    ``k`` and there are ``V`` columns. Untied: every state has its own label,
    phoneme-major, so column ``k`` is state ``k % S`` of phoneme ``k // S`` and
    there are ``V * S`` columns - a phoneme's states are adjacent rows of the
    model's table.

    Everything downstream that has to know what a column means - the
    statistics counter, hypotheses read off posteriors, the emission balance -
    goes through this rather than re-deriving the convention.
    """
    num_phonemes: int
    num_states: int = 1
    tie_states: bool = True

    def __post_init__(self):
        if self.num_phonemes < 1 or self.num_states < 1:
            raise ValueError(f"invalid layout {self}")

    @property
    def untied(self) -> bool:
        """True when columns are states rather than phonemes. Untying one state
        changes nothing, so S = 1 counts as tied whatever the flag says."""
        return not self.tie_states and self.num_states > 1

    @property
    def num_labels(self) -> int:
        return self.num_phonemes * self.num_states if self.untied else self.num_phonemes

    def emission_table(self) -> np.ndarray:
        """``emis[a, j]``, the label state ``j`` of phoneme ``a`` emits."""
        V, S = self.num_phonemes, self.num_states
        if self.untied:
            return np.arange(V * S, dtype=np.int32).reshape(V, S)
        return np.repeat(np.arange(V, dtype=np.int32).reshape(V, 1), S, axis=1)

    def phoneme_of(self) -> np.ndarray:
        """``[num_labels]``: the phoneme each column belongs to."""
        k = np.arange(self.num_labels)
        return k // self.num_states if self.untied else k

    def state_of(self) -> np.ndarray:
        """``[num_labels]``: the state each column is, 0 for every tied column."""
        k = np.arange(self.num_labels)
        return k % self.num_states if self.untied else np.zeros_like(k)

    def phoneme_posteriors(self, posteriors: np.ndarray) -> np.ndarray:
        """``[T, num_labels]`` -> ``[T, num_phonemes]``, summing each phoneme's
        states. The identity for a tied layout."""
        posteriors = np.asarray(posteriors)
        if not self.untied:
            return posteriors
        return posteriors.reshape(len(posteriors), self.num_phonemes, self.num_states).sum(axis=2)

    def argmax_segments(self, posteriors: np.ndarray) -> np.ndarray:
        """Phoneme of every segment in the frame-wise argmax, in order.

        A new segment starts where the argmax phoneme changes - and, untied,
        also where the state index goes *down* within one phoneme: a
        left-to-right phoneme cannot re-enter an earlier state, so ``a1 -> a0``
        is a second ``a``. That is something a tied layout cannot see, which
        is why there two consecutive realizations of one phoneme merge.

        Frames without posterior mass (a sequence with no path through the
        graph) yield no segments rather than the argmax of a zero row.
        """
        posteriors = np.asarray(posteriors)
        if posteriors.size == 0:
            return np.zeros((0,), dtype=np.int64)
        live = posteriors.sum(axis=1) > 0
        labels = posteriors.argmax(axis=1)[live]
        if labels.size == 0:
            return np.zeros((0,), dtype=np.int64)
        phonemes = self.phoneme_of()[labels]
        states = self.state_of()[labels]
        starts = np.r_[True, (phonemes[1:] != phonemes[:-1]) | (states[1:] < states[:-1])]
        return phonemes[starts]


def build_topology(
    symbols: List[str],
    *,
    loop_probability: float,
    silence_loop_probability: Optional[float] = None,
    transition_scale: float = 1.0,
    silence_symbol: str = "<SIL>",
    num_states: int = 1,
    tie_states: bool = True,
    zero_topology: bool = False,
    backoff_fb_root=None,
):
    """A left-to-right topology matching the RASR search parameters.

    Every phoneme gets ``num_states`` states, each looping with probability
    ``p`` and leaving with ``1 - p`` -- forward to the next state, or, out of
    the last one, exiting the phoneme. Uniform S across phonemes is what the op
    assumes (SPEC.md section 13); per-phoneme lengths would need a
    ``chain_state_ptr``.

    ``p = 0`` gives ``tau_loop = -inf``, which the op handles -- ``LSE(-inf, x)
    = x`` -- so each state then consumes exactly one frame.

    **Duration.** A state's occupancy is geometric with mean ``1 / (1 - p)``, so
    a phoneme's is the sum of ``S`` of them: mean ``S / (1 - p)``, i.e. negative
    binomial with ``r = S``. Inverting that is how a config picks ``p`` from a
    target duration ``d`` in frames::

        p = 1 - num_states / d

    At S = 1 this is the geometric MLE and at S = 2 the negative-binomial one.
    Raising S at fixed mean concentrates the distribution (the geometric's mode
    is always 1 frame, which is the thing S > 1 exists to fix) without changing
    the mean, which is what makes the two comparable.

    **Emissions.** Tied (``tie_states=True``, the default): every state of a
    phoneme emits that phoneme's label (``emis[a, j] = a``), so the op's
    per-label reduction (SPEC.md section 7) already sums the states and nothing
    downstream of the op sees S at all. Untied: each state emits its own label,
    ``emis[a, j] = a * S + j`` (:class:`LabelLayout`), so a phoneme's onset and
    offset get separate emission models and the score matrix has ``V * S``
    columns. The op allows any map (SPEC.md section 2) and its conformance grid
    covers untied labels; the transitions are the same either way.

    **Zero topology** (``zero_topology=True``): every loop, forward step and
    exit weighs 0, i.e. the graph has no transition model of its own beyond
    the LM. For per-frame transition scores that are meant to be the whole
    transition model (the op's SPEC.md 14.1). The loop parameters then have
    nothing to set and must stay at their defaults. The last state's forward
    arc stays -inf as always: leaving the last state is the exit.
    """
    if backoff_fb_root is not None:
        ensure_importable(backoff_fb_root)
    from backoff_fb_ref import HmmTopology

    if num_states < 1:
        raise ValueError(f"num_states must be >= 1, got {num_states}")

    V = len(symbols)
    S = num_states
    if zero_topology:
        if loop_probability != 0.0 or silence_loop_probability is not None or transition_scale != 1.0:
            raise ValueError(
                f"zero_topology=True sets every transition weight to 0, so "
                f"loop_probability ({loop_probability}), silence_loop_probability "
                f"({silence_loop_probability}) and transition_scale "
                f"({transition_scale}) would be ignored; leave them at their "
                f"defaults (0.0, None, 1.0)")
        tau_fwd = np.zeros((V, S), dtype=np.float64)
        tau_fwd[:, S - 1] = -math.inf
        return HmmTopology(V=V, S=S, tau_loop=np.zeros((V, S), dtype=np.float64),
                           tau_fwd=tau_fwd, tau_exit=np.zeros(V, dtype=np.float64),
                           emis=LabelLayout(V, S, tie_states).emission_table())
    if silence_loop_probability is None:
        silence_loop_probability = loop_probability

    def log(p: float) -> float:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"loop probability must be in [0, 1), got {p}")
        return -math.inf if p == 0.0 else math.log(p)

    def scaled(weight: float) -> float:
        """``transition_scale`` applied to a finite weight, left alone otherwise.

        ``transition_scale`` down-weights a *prior*; it is not a licence to
        reopen a transition the topology forbids. Multiplying through would make
        ``0.0 * -inf`` a NaN and put it in the graph, where it propagates
        silently through the forward recursion -- the spec requires that no NaN
        appear even for a dead utterance (section 4.4). The combination is
        reachable: ``loop_probability=0`` is what every segment-level arm in
        this family uses, and a transition-scale sweep that includes 0 would hit
        it. A forbidden transition stays forbidden at any scale.
        """
        return weight if math.isinf(weight) else transition_scale * weight

    tau_loop = np.empty((V, S), dtype=np.float64)
    tau_fwd = np.empty((V, S), dtype=np.float64)
    tau_exit = np.empty(V, dtype=np.float64)
    for a, sym in enumerate(symbols):
        p = silence_loop_probability if sym == silence_symbol else loop_probability
        # log1p(-p) rather than log(1 - p): exact at the small-p end, and equal
        # to 0.0 at p = 0 where leaving is certain.
        leave = scaled(math.log1p(-p))
        tau_loop[a, :] = scaled(log(p))
        tau_fwd[a, :] = leave
        # The last state leaves by exiting the phoneme, not by a forward arc;
        # compile.py expects -inf in that column.
        tau_fwd[a, S - 1] = -math.inf
        tau_exit[a] = leave

    # Tied, label k is emitted by phoneme k in every state, which is what makes
    # the score columns and the phonemes the same index; untied, by state
    # k % S of phoneme k // S. LabelLayout owns the convention.
    emis = LabelLayout(V, S, tie_states).emission_table()
    return HmmTopology(V=V, S=S, tau_loop=tau_loop, tau_fwd=tau_fwd,
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
    #: HMM states per phoneme. 1 is one state per phoneme (geometric duration);
    #: 2 gives a negative-binomial duration at the same mean. Defaulted so the
    #: existing S=1 callers keep the cache key they had.
    num_states: int = 1
    #: Whether a phoneme's states share one emission label; see
    #: :func:`build_topology`. Defaulted for the same reason as num_states.
    tie_states: bool = True
    #: Every transition weight 0, for per-frame transition scores that carry
    #: the whole transition model; see :func:`build_topology`.
    zero_topology: bool = False
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


@contextlib.contextmanager
def _plain_arpa(path):
    """The ARPA at ``path`` as an uncompressed file, removed afterwards.

    ``load_arpa`` decompresses a ``.gz`` itself, into the default temp
    directory - node-local ``/tmp`` inside a job, which is shared with every
    other job on the node and is nobody's to clean. On 2026-10-12 cn-225's was
    full and two chunks died with ENOSPC before compiling anything. So the
    decompression happens here instead, with a fallback: the temp directory
    first, then the task's working directory (the job's ``work/``, on shared
    storage and removed with the job). The file is 241 MB for the 5-gram and
    exists only while the LM loads.
    """
    path = str(path)
    if not path.endswith(".gz"):
        yield path
        return
    error = None
    for directory in (tempfile.gettempdir(), os.getcwd()):
        plain = None
        try:
            fd, plain = tempfile.mkstemp(suffix=".arpa", dir=directory)
            with os.fdopen(fd, "wb") as out, gzip.open(path, "rb") as src:
                shutil.copyfileobj(src, out)
        except OSError as e:
            if plain is not None:
                os.unlink(plain)
            error = e
            print(f"[WARNING] could not decompress {path} into {directory}: {e}",
                  flush=True)
            continue
        try:
            yield plain
        finally:
            os.unlink(plain)
        return
    raise error


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
    with _plain_arpa(spec.arpa_path) as arpa:
        lm = bf.load_arpa(arpa, symbols)
    topo = build_topology(
        symbols,
        loop_probability=spec.loop_probability,
        silence_loop_probability=spec.silence_loop_probability,
        transition_scale=spec.transition_scale,
        num_states=spec.num_states,
        tie_states=spec.tie_states,
        zero_topology=spec.zero_topology,
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
