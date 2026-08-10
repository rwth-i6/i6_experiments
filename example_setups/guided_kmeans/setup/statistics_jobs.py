"""
Jobs that turn a guided-k-means ``epoch_statistics.json`` into Variables.

The statistics file is written by ``EpochwiseStatisticsLogger`` during the RETURNN
forward pass, so it is a plain output file of the clustering job. Reports want
individual numbers out of it, and want them without opening the file themselves -
hence these two small jobs, both of which publish a ``{epoch: ...}`` Variable.
"""

__all__ = [
    "ExtractEpochStatisticsJob",
    "PhonemePriorDistanceJob",
    "load_epoch_statistics",
    "read_count_file",
]

import json
import logging
from typing import Iterable

from sisyphus import Job, Task, tk


def load_epoch_statistics(path: str) -> dict:
    """
    Read an ``epoch_statistics.json`` and key it by ``int`` epoch.

    :param path: path to the json file, whose top level maps epoch (as a string,
        json has no integer keys) to that epoch's statistics dict
    """
    with open(path, "rt") as f:
        raw = json.load(f)
    return {int(epoch): stats for epoch, stats in raw.items()}


def read_count_file(path: str) -> dict:
    """
    Read a ``<count>\\t<word>`` count file, as written by ``CountCorpusWordFrequenciesJob``.

    :param path: path to the counts file
    """
    counts = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            count, word = line.split("\t", 1)
            counts[word] = int(count)
    return counts


class ExtractEpochStatisticsJob(Job):
    """
    Reduce a guided-k-means ``epoch_statistics.json`` to its scalar entries.

    The file mixes scalars (``average_am_score``, ``relative_loop_frequency``, ...)
    with whole distributions (``relative_phoneme_frequencies``) and debug output
    (``sampled_tracebacks``). Everything that is not a number is dropped here, so
    the result stays small enough to live in a Variable.

    Note on epoch numbering: statistics epoch ``e`` is the recognition pass that ran
    with ``centroids.e.npy`` and produced ``centroids.{e+1}.npy``, so a decoding run
    on epoch ``e``'s centroids lines up with statistics epoch ``e``. A run with
    ``num_epochs`` epochs writes centroids ``0..num_epochs`` but statistics only for
    ``0..num_epochs-1``; the final centroids have no recognition pass of their own.
    """

    def __init__(self, statistics: tk.Path):
        """
        :param statistics: ``epoch_statistics.json`` of a clustering job
        """
        self.statistics = statistics

        self.out_statistics = self.output_var("statistics")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        stats = load_epoch_statistics(self.statistics.get_path())
        scalars = {
            epoch: {
                key: value
                for key, value in epoch_stats.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            for epoch, epoch_stats in stats.items()
        }
        self.out_statistics.set(scalars)


class PhonemePriorDistanceJob(Job):
    """
    Distance per epoch between the unigram phoneme distribution the guided search
    produced and a reference (transcription) unigram distribution.

    Both sides are restricted to the phonemes the reference knows about and
    renormalized over that shared support, so the comparison is between two proper
    distributions. This matters because the hypothesis side contains ``[SILENCE]``,
    which transcription counts do not - without excluding it, the distance would
    mostly measure how much silence the search emitted.
    """

    def __init__(
        self,
        statistics: tk.Path,
        reference_counts: tk.Path,
        exclude_phonemes: Iterable[str] = ("[SILENCE]",),
        renormalize: bool = True,
        order: int = 1,
    ):
        """
        :param statistics: ``epoch_statistics.json``; each epoch needs either
            ``relative_phoneme_frequencies`` or ``absolute_phoneme_counts``
        :param reference_counts: ``<count>\\t<phoneme>`` file, e.g.
            ``CountCorpusWordFrequenciesJob.out_word_counts`` run on a phoneme corpus
            (``constants.PHONEME_UNIGRAM_PRIORS``)
        :param exclude_phonemes: dropped from both sides before normalizing
        :param renormalize: renormalize the hypothesis over the shared support. With
            ``False`` the mass on excluded/unknown phonemes is simply dropped and the
            hypothesis no longer sums to one.
        :param order: p of the L_p distance; 1 (the default) gives the L1 distance
        """
        self.statistics = statistics
        self.reference_counts = reference_counts
        # sorted list rather than a set, so the job hash does not depend on set order
        self.exclude_phonemes = sorted(set(exclude_phonemes))
        self.renormalize = renormalize
        self.order = order

        self.out_distances = self.output_var("distances")
        self.out_reference_priors = self.output_var("reference_priors")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _normalize(counts: dict) -> dict:
        total = sum(counts.values())
        assert total > 0, "cannot normalize an all-zero distribution"
        return {key: value / total for key, value in counts.items()}

    def run(self):
        excluded = set(self.exclude_phonemes)

        raw_reference = read_count_file(self.reference_counts.get_path())
        reference = self._normalize({p: c for p, c in raw_reference.items() if p not in excluded})
        support = sorted(reference)
        self.out_reference_priors.set(reference)

        distances = {}
        for epoch, epoch_stats in load_epoch_statistics(self.statistics.get_path()).items():
            counts = epoch_stats.get("relative_phoneme_frequencies")
            if counts is None:
                counts = epoch_stats.get("absolute_phoneme_counts")
            assert counts is not None, (
                f"epoch {epoch} has neither relative_phoneme_frequencies nor absolute_phoneme_counts"
            )

            unknown = sorted(set(counts) - set(reference) - excluded)
            if unknown:
                logging.warning("epoch %s: phonemes not in the reference, dropped: %s", epoch, unknown)

            hypothesis = {p: counts.get(p, 0.0) for p in support}
            total = sum(hypothesis.values())
            if total <= 0:
                logging.warning("epoch %s: no mass on the reference support, distance is nan", epoch)
                distances[epoch] = float("nan")
                continue
            if self.renormalize:
                hypothesis = {p: value / total for p, value in hypothesis.items()}

            distance = sum(abs(hypothesis[p] - reference[p]) ** self.order for p in support)
            distances[epoch] = distance ** (1.0 / self.order)

        self.out_distances.set(distances)
