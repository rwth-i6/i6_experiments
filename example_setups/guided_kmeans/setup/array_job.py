"""
A base class for the small jobs that reshape arrays into other arrays.

Model initialization is a growing pile of one-liners - sample K frames, stack
identities, split each centroid in n, tile one covariance - and written out
longhand each is about seven lines of mechanism around one line of arithmetic.
:class:`ArrayJob` takes the mechanism: the mini-task declaration, the output
declaration, loading inputs, saving results, and the shape checks that turn a
silent broadcast into an error naming the file that caused it.

What it deliberately does *not* take is the constructor. It is tempting to
generate the whole job from a function - ``@array_job`` over a plain
``def split(centroids, n, seed=42)`` - and it does not work, because sisyphus
hashes a job from ``inspect.getcallargs(cls.__init__, ...)``. A generated
``__init__(self, *args, **kwargs)`` never fills in defaults, so three spellings
of one call produce three different hashes and three copies of the same
computation:

    Explicit signature      f(1), f(1, 2), f(1, seed=42) -> one hash
    *args/**kwargs          f(1), f(1, 2), f(1, seed=42) -> three hashes

and ``__sis_hash_exclude__``, which matches on parameter names, stops working
entirely. A decorator could exec a real signature to dodge this, but then every
job's identity depends on a code generator - a bad trade for the two lines it
would save. Writing ``self.x = x`` out is also what makes a job's hash
greppable, which is the i6_core convention here (see its CONTRIBUTING.md: set
inputs, set outputs, then rqmt).

Subclass like this::

    class UniformMixturesJob(ArrayJob):
        OUTPUTS = ("mixtures",)

        def __init__(self, num_labels: int, num_densities: int):
            self.num_labels = num_labels
            self.num_densities = num_densities
            super().__init__()

        def compute(self):
            return np.full((self.num_labels, self.num_densities), 1 / self.num_densities)

which gets ``out_mixtures`` pointing at ``mixtures.npy``. Return a dict keyed
by :attr:`OUTPUTS` for more than one output, and set ``self.rqmt`` for a job
too heavy to run as a mini-task. Anything needing several tasks or a
non-``.npy`` output should just be a plain ``Job``.
"""

from __future__ import annotations

__all__ = ["ArrayJob"]

from typing import ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from sisyphus import Job, Task, tk


class ArrayJob(Job):
    """One mini-task producing one ``.npy`` file per name in :attr:`OUTPUTS`."""

    #: Output names. Each becomes ``self.out_<name>``, a path to ``<name>.npy``.
    OUTPUTS: ClassVar[Tuple[str, ...]] = ("out",)

    def __init__(self):
        for name in self.OUTPUTS:
            setattr(self, f"out_{name}", self.output_path(f"{name}.npy"))

    def tasks(self):
        # A mini-task unless the job declared requirements. Reshaping a few
        # arrays belongs on the local engine; streaming 29 GB of features to
        # accumulate a covariance does not, and i6_core's convention is that a
        # job with non-trivial requirements says so via self.rqmt.
        rqmt = getattr(self, "rqmt", None)
        if rqmt:
            yield Task("run", rqmt=rqmt)
        else:
            yield Task("run", mini_task=True)

    def compute(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """The actual work: an array, or ``{output_name: array}``."""
        raise NotImplementedError

    def run(self):
        result = self.compute()
        if not isinstance(result, dict):
            if len(self.OUTPUTS) != 1:
                raise TypeError(
                    f"{type(self).__name__} declares {len(self.OUTPUTS)} outputs "
                    f"{self.OUTPUTS}, so compute() has to return a dict keyed by them"
                )
            result = {self.OUTPUTS[0]: result}
        if set(result) != set(self.OUTPUTS):
            raise ValueError(
                f"{type(self).__name__}.compute() returned {sorted(result)} but "
                f"OUTPUTS declares {sorted(self.OUTPUTS)}"
            )
        for name, array in result.items():
            array = np.asarray(array)
            # Same guard the models apply on save: np.save turns a None or a
            # ragged list into a 0-d object array that needs allow_pickle to
            # read back, and fails far from whatever produced it.
            if array.dtype == object:
                raise TypeError(
                    f"{type(self).__name__} output {name!r} is not a numeric array "
                    f"(got {type(result[name]).__name__})"
                )
            np.save(getattr(self, f"out_{name}").get_path(), array)

    @staticmethod
    def load(
        path: tk.Path,
        *,
        ndim: Optional[Union[int, Sequence[int]]] = None,
        name: str = "array",
    ) -> np.ndarray:
        """
        Read an input ``.npy``, optionally asserting its rank.

        The rank check is the point: these jobs mostly index and broadcast, and
        a ``[D, D]`` covariance passed where a ``[K, D, D]`` stack was meant
        broadcasts happily and produces a wrong array rather than an error. One
        check here names the file instead.
        """
        array = np.load(path.get_path())
        if ndim is not None:
            allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
            if array.ndim not in allowed:
                raise ValueError(
                    f"{name} from {path.get_path()} has shape {array.shape} "
                    f"({array.ndim}-D); expected {' or '.join(f'{n}-D' for n in allowed)}"
                )
        return array
