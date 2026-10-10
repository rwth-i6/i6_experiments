"""Sequential race over first-pass CTC scales, paired per utterance, in place of the n-best re-ranking tuner."""

import ast
import gzip
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy import stats

from returnn.forward_iface import ForwardCallbackIface


RACE_ATTR = "ctc_scale_race"
PICKS = ("winner", "smallest", "largest")


def word_edit_distance(hyp: Sequence[str], ref: Sequence[str]) -> int:
    """
    Levenshtein distance between two word sequences.

    :param hyp: hypothesis words
    :param ref: reference words
    :return: the number of substitutions, insertions and deletions
    """
    return Levenshtein.distance(list(hyp), list(ref))


def student_t_quantile(p: float, dof: float) -> float:
    """
    Upper quantile of Student's t.

    :param p: probability in (0.5, 1)
    :param dof: degrees of freedom
    :return: the x with P(T <= x) = p
    """
    assert 0.5 < p < 1.0, p
    assert dof >= 1, dof

    return float(stats.t.ppf(p, dof))


def load_text_dict(path: str) -> Dict[str, str]:
    """
    Reads a RETURNN text dict file, plain or gzipped.

    :param path: the file
    :return: sequence tag to text
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    data = ast.literal_eval(raw.decode("utf-8"))
    assert isinstance(data, dict), type(data)
    return {str(tag): str(text) for tag, text in data.items()}


class ScaleRace:
    """
    Race of candidate scales on the first-pass decodes of one dataset, one look after every doubling of the utterances.

    Every alive candidate decodes every utterance, so two candidates compare on the same utterances through the mean
    of their per-utterance error differences, the matched-pairs test of Gillick and Cox (1989). A candidate drops out
    once its mean difference to some other alive candidate has a one-sided Student t lower bound above zero, at the
    race delta split over the ordered candidate pairs and the maximal number of looks. The utterance order must be
    random for the early looks to be samples of the dataset.
    """

    def __init__(
        self,
        *,
        candidates: Sequence[float],
        refs: Dict[str, str],
        decode: Callable[[List[int]], str],
        delta: float = 0.05,
        first_look: int = 128,
        max_looks: int = 32,
        pick: str = "winner",
    ) -> None:
        """
        :param candidates: the scales, at least two, zero decodes without the CTC term
        :param refs: reference text per sequence tag
        :param decode: token ids to text
        :param delta: the probability that some candidate drops out although it is not worse
        :param first_look: utterances before the first look, then doubling
        :param max_looks: bound on the looks, part of the level split
        :param pick: the scale the race reports, the survivor with the fewest errors, the smallest or the largest
        """
        scales = sorted(set(float(c) for c in candidates))
        assert len(scales) >= 2, scales
        assert all(scale >= 0.0 for scale in scales), scales
        assert pick in PICKS, pick
        assert first_look >= 2 and max_looks >= 1 and 0.0 < delta < 1.0, (first_look, max_looks, delta)
        self.candidates = scales
        self.refs = refs
        self.decode = decode
        self.delta = float(delta)
        self.first_look = int(first_look)
        self.max_looks = int(max_looks)
        self.pick = pick
        self.level = self.delta / (self.max_looks * len(scales) * (len(scales) - 1))
        self.alive = {scale: True for scale in scales}
        self.tags: List[str] = []
        self.ref_words: List[int] = []
        self.errors: Dict[float, List[int]] = {scale: [] for scale in scales}
        self.hyps: Dict[float, List[str]] = {scale: [] for scale in scales}
        self.eliminated: Dict[float, Dict[str, Any]] = {}
        self.looks: List[Dict[str, Any]] = []
        self.next_look = self.first_look

    @property
    def num_seqs(self) -> int:
        return len(self.tags)

    def alive_candidates(self) -> List[float]:
        """
        :return: the scales still in the race, ascending
        """
        return [scale for scale in self.candidates if self.alive[scale]]

    def run_batch(self, *, seq_tags: Sequence[str], search: Callable[[float], Sequence[Sequence[int]]]) -> None:
        """
        Decodes one batch with every alive candidate and looks when the utterance count crossed the next threshold.

        :param seq_tags: the batch's sequence tags
        :param search: decodes the batch at a scale, token ids per sequence
        """
        tags = [str(tag) for tag in seq_tags]
        missing = [tag for tag in tags if tag not in self.refs]
        assert not missing, f"{len(missing)} sequences without reference, e.g. {missing[:3]}"
        ref_words = [self.refs[tag].split() for tag in tags]
        for scale in self.alive_candidates():
            tokens = search(scale)
            assert len(tokens) == len(tags), (len(tokens), len(tags))
            for words, seq_tokens in zip(ref_words, tokens):
                text = self.decode([int(t) for t in seq_tokens])
                self.hyps[scale].append(text)
                self.errors[scale].append(word_edit_distance(text.split(), words))
        self.tags.extend(tags)
        self.ref_words.extend(len(words) for words in ref_words)
        if self.num_seqs >= self.next_look:
            self._look()
            self.next_look = 2 * self.num_seqs

    def _wer(self, scale: float) -> float:
        seen = len(self.errors[scale])
        return 100.0 * sum(self.errors[scale]) / max(sum(self.ref_words[:seen]), 1)

    def _look(self) -> None:
        n = self.num_seqs
        alive = self.alive_candidates()
        quantile = student_t_quantile(1.0 - self.level, n - 1)
        errors = {scale: np.asarray(self.errors[scale], dtype=np.float64) for scale in alive}
        dropped = []
        for scale in alive:
            for other in alive:
                if other == scale:
                    continue
                diff = errors[scale] - errors[other]
                mean = float(diff.mean())
                std = float(diff.std(ddof=1))
                lower = mean - quantile * std / math.sqrt(n)
                if lower > 0.0:
                    dropped.append(
                        {"scale": scale, "against": other, "mean_diff": mean, "std": std, "lower_bound": lower}
                    )
                    break
        for entry in dropped:
            self.alive[entry["scale"]] = False
            self.eliminated[entry["scale"]] = {**entry, "seqs": n, "look": len(self.looks)}
        self.looks.append(
            {
                "seqs": n,
                "quantile": quantile,
                "alive_before": alive,
                "wer": {str(scale): self._wer(scale) for scale in alive},
                "dropped": dropped,
            }
        )
        assert len(self.looks) <= self.max_looks, (len(self.looks), self.max_looks)

    def finish(self) -> Dict[str, Any]:
        """
        Closes the race with a look at the whole dataset and reports every candidate.

        :return: the survivors, the chosen scale and the per-candidate and per-look records
        """
        n = self.num_seqs
        assert n > 0, "no utterances raced"
        if not self.looks or self.looks[-1]["seqs"] < n:
            self._look()
        survivors = self.alive_candidates()
        winner = min(survivors, key=lambda scale: (sum(self.errors[scale]), scale))
        chosen = {"winner": winner, "smallest": survivors[0], "largest": survivors[-1]}[self.pick]
        per_candidate = {}
        for scale in self.candidates:
            seen = len(self.errors[scale])
            per_candidate[str(scale)] = {
                "seqs": seen,
                "errors": int(sum(self.errors[scale])),
                "ref_words": int(sum(self.ref_words[:seen])),
                "wer": self._wer(scale),
                "alive": self.alive[scale],
                "eliminated": self.eliminated.get(scale),
            }
        return {
            "candidates": self.candidates,
            "delta": self.delta,
            "level_per_test": self.level,
            "first_look": self.first_look,
            "max_looks": self.max_looks,
            "pick": self.pick,
            "seqs": n,
            "ref_words": int(sum(self.ref_words)),
            "survivors": survivors,
            "band": [survivors[0], survivors[-1]],
            "winner": winner,
            "chosen": chosen,
            "per_candidate": per_candidate,
            "looks": self.looks,
        }

    def decoded(self, scale: float) -> List[tuple]:
        """
        :param scale: a candidate
        :return: (sequence tag, hypothesis text) of every utterance the candidate decoded, in order
        """
        return list(zip(self.tags, self.hyps[scale]))


def active_race(model: Any) -> Optional[ScaleRace]:
    """
    :param model: the model of the forward
    :return: the race the forward callback attached to it, if any
    """
    return getattr(model, RACE_ATTR, None)


class ScaleRaceForwardCallback(ForwardCallbackIface):
    """Runs the race of the ``ctc_scale_race`` option and writes its result and the chosen scale's decode."""

    result_filename = "race_result.json"
    out_filename = "search_out.py"

    def __init__(self, make_decoder: Callable[[Any], Callable[[List[int]], str]], race: Dict[str, Any]) -> None:
        """
        :param make_decoder: gives the function turning label ids into text, for the model
        :param race: the arguments of the race, the reference text dict under "ref"
        """
        self._make_decoder = make_decoder
        self._race_opts = dict(race)
        self.race: Optional[ScaleRace] = None
        self._model: Any = None
        self._processed = 0

    def init(self, *, model: Any) -> None:
        opts = dict(self._race_opts)
        refs = load_text_dict(str(opts.pop("ref")))
        self.race = ScaleRace(refs=refs, decode=self._make_decoder(model), **opts)
        self._model = model
        setattr(model, RACE_ATTR, self.race)

    def process_seq(self, *, seq_tag: str, outputs: Any) -> None:
        del seq_tag, outputs
        self._processed += 1

    def finish(self) -> None:
        assert self.race is not None
        assert self._processed == self.race.num_seqs, (self._processed, self.race.num_seqs)
        result = self.race.finish()
        with open(self.result_filename, "wt") as f:
            json.dump(result, f, indent=1)
        with open(self.out_filename, "wt") as f:
            f.write("{\n")
            for tag, text in self.race.decoded(result["chosen"]):
                f.write(f"{tag!r}: {text!r},\n")
            f.write("}\n")
        delattr(self._model, RACE_ATTR)
