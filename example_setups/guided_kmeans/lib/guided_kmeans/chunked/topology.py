"""
Stationary transition models: the loop probabilities of the HMM topology, and
their re-estimation from the expected transition counts.

Every run before this one held the topology fixed: one loop probability ``p``
(or one for speech and one for silence), turned into static transition weights
by :func:`..backoff_fb_graph.build_topology` as ``tau_loop = s * ln p`` and
``tau_fwd = tau_exit = s * ln(1 - p)``, with ``s`` the transition scale. On
request the ``backoff_fb`` op also returns how often a path uses each of those
weights (SPEC.md section 15 in its checkout). For every sequence that is a
``[V, S, 2]`` array of expected counts, ``(stay, move)`` per (phoneme, state);
a move out of the last state is the phoneme's exit, the final transition
included. Summed over the corpus they are the sufficient statistics of the EM
update of the loop probabilities, with the emissions and the LM held fixed::

    p_G = sum_G stay / sum_G (stay + move)

over a *tying group* ``G``: a set of (phoneme, state) pairs that share one
probability (:data:`TYINGS`). The update is EM at any transition scale and any
emission scale, so the summed log Z of the scaled model it is computed under
cannot decrease (SPEC.md 15.6).

There are two objects here, mirroring a score model and its accumulator:

* :class:`StationaryTopology` is the ``[V, S]`` table a search runs with. An
  epoch job writes it as a directory (``loop_probabilities.npy`` next to a
  ``topology.json`` manifest), and the next epoch recognizes with it.
* :class:`TopologyAccumulator` holds the summed counts. Its ``merge`` is
  associative, which keeps ``num_chunks`` a scheduling knob, and its
  ``finalize`` is the M-step.

Both stay separate from the per-frame transition models of :mod:`.transitions`
(the theta side). Those are functions of the features and would be
re-estimated from per-frame transition posteriors; this one is a constant
table. A fixed per-phoneme exit offset is the search's business, not the
table's (``BackoffFBRecognizer(exit_offset=...)``): the update never touches it,
and the counts it produces are the same derivatives with or without it.
"""

from __future__ import annotations

__all__ = [
    "TYINGS",
    "tying_groups",
    "StationaryTopology",
    "TopologyAccumulator",
    "TOPOLOGY_MANIFEST",
]

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .models import _save_array_atomically

#: The tyings :class:`TopologyAccumulator` knows, coarsest first:
#:
#: * ``"global"``: one probability for every (phoneme, state), silence included
#: * ``"speech_silence"``: one for the states of the silence symbol, one for
#:   all others
#: * ``"phoneme"``: one per phoneme, shared by its ``S`` states
#: * ``"state"``: one per (phoneme, state)
TYINGS = ("global", "speech_silence", "phoneme", "state")

TOPOLOGY_MANIFEST = "topology.json"


def tying_groups(
    tying: str,
    symbols: Sequence[str],
    num_states: int,
    silence_symbol: str = "<SIL>",
) -> Tuple[np.ndarray, List[str]]:
    """``([V, S] group index, group names)``: which group each (phoneme, state)
    pair belongs to under ``tying`` (one of :data:`TYINGS`)."""
    symbols = list(symbols)
    V, S = len(symbols), num_states
    if tying == "global":
        return np.zeros((V, S), dtype=np.int64), ["all"]
    if tying == "speech_silence":
        if silence_symbol not in symbols:
            raise ValueError(
                f"speech_silence tying needs the silence symbol {silence_symbol!r}, "
                f"which is not among the {V} symbols")
        groups = np.zeros((V, S), dtype=np.int64)
        groups[symbols.index(silence_symbol)] = 1
        return groups, ["speech", "silence"]
    if tying == "phoneme":
        return np.repeat(np.arange(V, dtype=np.int64)[:, None], S, axis=1), symbols
    if tying == "state":
        return (np.arange(V * S, dtype=np.int64).reshape(V, S),
                [f"{sym}.{j}" for sym in symbols for j in range(S)])
    raise ValueError(f"unknown tying {tying!r}; expected one of {TYINGS}")


def _lambda(p: Union[float, np.ndarray], transition_scale: float):
    """``s * logit(p)``, the per-state boundary price that is all the posterior
    sees of a global ``p`` (SPEC.md 15.6). ``-inf`` at ``p = 0``."""
    with np.errstate(divide="ignore"):
        return transition_scale * (np.log(p) - np.log1p(-np.asarray(p, dtype=np.float64)))


class StationaryTopology:
    """
    The loop probabilities a stationary HMM topology runs with.

    ``loop_probabilities[a, j]`` is the probability that state ``j`` of phoneme
    ``symbols[a]`` loops; it leaves with ``1 - p``, forward to the next state or,
    out of the last one, through the LM to the next phoneme. ``p = 0`` forbids
    the loop, and the update keeps it forbidden.

    :param loop_probabilities: ``[V, S]``, as an array or as the path of a
        ``.npy``, each entry in ``[0, 1)``
    :param symbols: the ``V`` LM symbols in row order, which is the
        forward-backward lexicon's order
        (:func:`..backoff_fb_graph.lexicon_symbols`)
    """

    ARTIFACT_NAMES = ("loop_probabilities",)

    def __init__(self, loop_probabilities: Union[np.ndarray, str], symbols: Sequence[str]):
        if isinstance(loop_probabilities, str):
            loop_probabilities = np.load(loop_probabilities)
        p = np.array(loop_probabilities, dtype=np.float64)
        symbols = tuple(symbols)
        if p.ndim != 2 or p.shape[0] != len(symbols) or p.shape[1] < 1:
            raise ValueError(
                f"loop probabilities have shape {p.shape}, expected "
                f"({len(symbols)}, num_states): one row per symbol")
        if not ((p >= 0.0) & (p < 1.0)).all():
            raise ValueError(
                f"loop probabilities must lie in [0, 1), got values in "
                f"[{np.nanmin(p)}, {np.nanmax(p)}]"
                + (" and NaN" if np.isnan(p).any() else ""))
        self.loop_probabilities = p
        self.symbols = symbols

    @property
    def num_phonemes(self) -> int:
        return self.loop_probabilities.shape[0]

    @property
    def num_states(self) -> int:
        return self.loop_probabilities.shape[1]

    @classmethod
    def from_scalars(
        cls,
        symbols: Sequence[str],
        num_states: int,
        loop_probability: float,
        silence_loop_probability: Optional[float] = None,
        silence_symbol: str = "<SIL>",
    ) -> "StationaryTopology":
        """The table that :func:`..backoff_fb_graph.build_topology` fills from
        its two scalars: ``silence_loop_probability`` (default: the same as
        everything else) for the silence symbol's states, ``loop_probability``
        for all others."""
        silence = loop_probability if silence_loop_probability is None else silence_loop_probability
        table = [[silence if sym == silence_symbol else loop_probability] * num_states
                 for sym in symbols]
        return cls(np.array(table, dtype=np.float64), symbols)

    def lambdas(self, transition_scale: float) -> np.ndarray:
        """``[V, S]``: ``s * logit(p)`` per state."""
        return _lambda(self.loop_probabilities, transition_scale)

    def artifacts(self) -> Dict[str, np.ndarray]:
        return {"loop_probabilities": self.loop_probabilities}

    def save(self, directory: str) -> None:
        """``loop_probabilities.npy`` and the ``topology.json`` manifest, the
        same layout a model directory has (:meth:`.models.ArtifactModel.save`)."""
        os.makedirs(directory, exist_ok=True)
        for name, array in self.artifacts().items():
            _save_array_atomically(os.path.join(directory, f"{name}.npy"), array)
        manifest = {
            "class": type(self).__name__,
            "artifacts": sorted(self.artifacts()),
            "symbols": list(self.symbols),
            "num_states": self.num_states,
        }
        tmp = os.path.join(directory, f"{TOPOLOGY_MANIFEST}.{os.getpid()}.tmp")
        with open(tmp, "w") as fp:
            json.dump(manifest, fp, indent=2)
        os.replace(tmp, os.path.join(directory, TOPOLOGY_MANIFEST))

    @classmethod
    def load(cls, directory: str) -> "StationaryTopology":
        with open(os.path.join(directory, TOPOLOGY_MANIFEST)) as fp:
            manifest = json.load(fp)
        if manifest.get("class") != cls.__name__:
            raise ValueError(f"{directory} holds a {manifest.get('class')!r}, not a {cls.__name__}")
        topology = cls(os.path.join(directory, "loop_probabilities.npy"), manifest["symbols"])
        if topology.num_states != manifest["num_states"]:
            raise ValueError(
                f"{directory}: the manifest says {manifest['num_states']} states, the "
                f"table has {topology.num_states}")
        return topology

    def __repr__(self) -> str:
        p = self.loop_probabilities
        return (f"StationaryTopology(V={self.num_phonemes}, S={self.num_states}, "
                f"p in [{p.min():.4g}, {p.max():.4g}])")


class TopologyAccumulator:
    """
    Expected transition counts summed over an epoch, and the M-step on them.

    The state is a plain sum - ``[V, S, 2]`` counts ``(stay, move)`` and the
    number of sequences - so ``merge`` is associative and ``scale`` is exact,
    the two properties an :class:`.interfaces.Accumulator` promises. It is
    built through a zero-argument factory in the reduce step, so the shape is
    taken from the first counts it sees.

    Three regularizations of the update are described in SPEC.md 15.6; two
    are here. The clamp is not implemented.

    :param tying: one of :data:`TYINGS`
    :param min_frames: the floor. A group whose expected frame count
        ``sum (stay + move)`` is below this keeps its previous probability. A
        group with no frames at all always keeps it, since the update is then
        undefined. It acts only on groups that are empty or nearly so and
        never constrains a value.
    :param prior_weight: ``m`` of a Beta prior with mean ``prior_mean``
        (MAP-EM): ``p = (sum stay + m * p0) / (sum (stay + move) + m)``, i.e. ``m``
        pseudo-frames at ``p0``. Its pull fades as the group's data grow. 0, the
        default, is plain EM. Since an empty group then gets ``p0``, a prior
        replaces the floor rather than combining with it.
    :param prior_mean: ``p0``: a probability, e.g. ``1 - S / d`` for a target
        duration of ``d`` frames, or ``"global"`` for the same epoch's global
        estimate, which shrinks each group toward the global value
    :param silence_symbol: what ``"speech_silence"`` calls silence
    """

    def __init__(
        self,
        tying: str = "global",
        min_frames: float = 0.0,
        prior_weight: float = 0.0,
        prior_mean: Union[float, str, None] = None,
        silence_symbol: str = "<SIL>",
    ):
        if tying not in TYINGS:
            raise ValueError(f"unknown tying {tying!r}; expected one of {TYINGS}")
        if not min_frames >= 0.0:
            raise ValueError(f"min_frames must be >= 0, got {min_frames}")
        if not prior_weight >= 0.0:
            raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
        if prior_weight > 0.0:
            if prior_mean != "global" and not (
                    isinstance(prior_mean, (int, float)) and 0.0 <= prior_mean < 1.0):
                raise ValueError(
                    f"prior_mean must be a probability in [0, 1) or 'global', got {prior_mean!r}")
            if min_frames > 0.0:
                raise ValueError(
                    "a prior gives an empty group its mean, so it replaces the floor; "
                    "set prior_weight or min_frames, not both")
        elif prior_mean is not None:
            raise ValueError(f"prior_mean={prior_mean!r} without a prior_weight does nothing")
        self.tying = tying
        self.min_frames = float(min_frames)
        self.prior_weight = float(prior_weight)
        self.prior_mean = prior_mean
        self.silence_symbol = silence_symbol
        self.counts: Optional[np.ndarray] = None
        self.num_seqs = 0

    def observe(self, counts: np.ndarray) -> None:
        """Add one sequence's ``[V, S, 2]`` counts."""
        counts = np.asarray(counts, dtype=np.float64)
        if counts.ndim != 3 or counts.shape[2] != 2:
            raise ValueError(f"transition counts have shape {counts.shape}, expected (V, S, 2)")
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("transition counts must be finite and non-negative")
        if self.counts is None:
            self.counts = counts.copy()
        else:
            self._check_shape(counts.shape)
            self.counts += counts
        self.num_seqs += 1

    def _check_shape(self, shape) -> None:
        if self.counts is not None and tuple(shape) != self.counts.shape:
            raise ValueError(f"transition counts of shape {tuple(shape)} cannot be added to "
                             f"{self.counts.shape}")

    def merge(self, other: "TopologyAccumulator") -> "TopologyAccumulator":
        if other.counts is not None:
            if self.counts is None:
                self.counts = other.counts.copy()
            else:
                self._check_shape(other.counts.shape)
                self.counts += other.counts
        self.num_seqs += other.num_seqs
        return self

    def scale(self, factor: float) -> "TopologyAccumulator":
        if self.counts is not None:
            self.counts *= factor
        return self

    def state_dict(self) -> dict:
        return {"counts": None if self.counts is None else self.counts.copy(),
                "num_seqs": self.num_seqs}

    def load_state_dict(self, state: dict) -> "TopologyAccumulator":
        counts = state["counts"]
        self.counts = None if counts is None else np.array(counts, dtype=np.float64)
        self.num_seqs = int(state["num_seqs"])
        return self

    def _group_sums(self, previous: StationaryTopology):
        if self.counts is None:
            raise RuntimeError("no transition counts were observed; nothing to update from")
        if self.counts.shape[:2] != previous.loop_probabilities.shape:
            raise ValueError(
                f"counts for {self.counts.shape[:2]} (phoneme, state) pairs, but the "
                f"topology has {previous.loop_probabilities.shape}")
        groups, names = tying_groups(self.tying, previous.symbols, previous.num_states,
                                     self.silence_symbol)
        # A forbidden loop stays forbidden: its stay count is exactly 0, and
        # its frames must not dilute the group's estimate.
        free = previous.loop_probabilities > 0.0
        stay = self.counts[..., 0]
        total = stay + self.counts[..., 1]
        stay_g = np.bincount(groups[free], weights=stay[free], minlength=len(names))
        total_g = np.bincount(groups[free], weights=total[free], minlength=len(names))
        return groups, names, free, stay_g, total_g

    def global_estimate(self, previous: StationaryTopology) -> float:
        """What the global tying would set: the corpus' stays over its frames.
        Equals ``1 - S / d`` for ``d`` the expected frames per phoneme."""
        _, _, free, stay_g, total_g = self._group_sums(previous)
        total = total_g.sum()
        return float(stay_g.sum() / total) if total > 0 else math.nan

    def finalize(self, previous: StationaryTopology) -> StationaryTopology:
        """The next epoch's topology. ``previous`` is the one this epoch ran
        with: it gives the symbols, the forbidden loops, and the value a
        floored group keeps."""
        groups, names, free, stay_g, total_g = self._group_sums(previous)
        if self.prior_weight > 0.0:
            p0 = (self.global_estimate(previous) if self.prior_mean == "global"
                  else float(self.prior_mean))
            if not math.isfinite(p0):
                raise RuntimeError("prior_mean='global' needs counts, and every group is empty")
            p_g = (stay_g + self.prior_weight * p0) / (total_g + self.prior_weight)
            update = np.ones(len(names), dtype=bool)
        else:
            update = (total_g > 0.0) & (total_g >= self.min_frames)
            p_g = np.divide(stay_g, total_g, out=np.zeros_like(stay_g), where=total_g > 0.0)
        if (p_g[update] >= 1.0).any():
            # Every visit to a state ends with one move, so stays can only
            # make up all of a group's frames if it was never left.
            bad = [names[g] for g in np.flatnonzero(update & (p_g >= 1.0))]
            raise FloatingPointError(f"the update would give groups {bad[:5]} p = 1 (no moves)")
        table = previous.loop_probabilities.copy()
        mask = free & update[groups]
        table[mask] = p_g[groups[mask]]
        return StationaryTopology(table, previous.symbols)

    def summary(
        self,
        previous: StationaryTopology,
        updated: StationaryTopology,
        transition_scale: float,
    ) -> Dict[str, Any]:
        """What an epoch's ``statistics.json`` records about the update.

        The scalars feed report columns: the global estimate and its
        ``lambda = s * logit(p)`` (whatever the tying, that is the estimate the
        global tying would make), and the expected frames per phoneme, against
        which ``average_segment_duration_argmax`` and the alignment's 3.99 can
        be read. ``topology_groups`` has every group's frames, previous and new
        ``p`` and ``lambda``.
        """
        groups, names, free, stay_g, total_g = self._group_sums(previous)
        frames = float(self.counts.sum())
        # The last state's move is the exit, one per phoneme occurrence.
        occurrences = float(self.counts[:, -1, 1].sum())
        p_global = self.global_estimate(previous)

        def value(table, sel):
            values = table[sel]
            if values.size and (values == values[0]).all():
                return float(values[0])
            return [float(v) for v in values]

        per_group = {}
        for g, name in enumerate(names):
            sel = groups == g
            new = value(updated.loop_probabilities, sel)
            per_group[name] = {
                "frames": float(total_g[g]),
                "stay": float(stay_g[g]),
                "p_previous": value(previous.loop_probabilities, sel),
                "p": new,
                "lambda": (float(_lambda(new, transition_scale)) if isinstance(new, float)
                           else [float(x) for x in _lambda(np.array(new), transition_scale)]),
                "updated": bool(not np.array_equal(updated.loop_probabilities[sel],
                                                   previous.loop_probabilities[sel])),
            }
        return {
            "topology_tying": self.tying,
            "topology_transition_scale": float(transition_scale),
            "topology_min_frames": self.min_frames,
            "topology_prior_weight": self.prior_weight,
            "topology_prior_mean": self.prior_mean,
            "topology_num_seqs": self.num_seqs,
            "topology_frames": frames,
            "topology_occurrences": occurrences,
            "topology_mean_duration": frames / occurrences if occurrences > 0 else math.nan,
            "topology_p_global": p_global,
            "topology_lambda_global": float(_lambda(p_global, transition_scale)),
            "topology_groups": per_group,
        }
