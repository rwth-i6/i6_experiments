"""
Lightweight dependency injection for the chunked clustering jobs.

A :class:`Spec` is a picklable, sisyphus-hashable *description* of an object
(a class plus its constructor arguments) that gets built inside the job task
rather than in the graph-building process. This is deliberately not the
serializer machinery used for RETURNN configs (``Import``/``CallImport``/
``CodeWrapper``): those exist because a RETURNN config is Python *source text*
executed by a separate interpreter, so a class reference has to be turned into
an import statement. A sisyphus job's ``run()`` already executes with the
recipe on ``sys.path``, so it can simply hold the class object and call it.

Two sisyphus properties make this work without any extra bookkeeping:

* ``sis_hash_helper`` hashes a class by ``(module, qualname)``, so swapping
  the class changes the job hash while editing the class body does not.
* ``extract_paths`` recurses into arbitrary objects via ``get_object_state``,
  so ``tk.Path`` objects nested anywhere inside ``kwargs`` are registered as
  job inputs automatically.
"""

from __future__ import annotations

__all__ = ["Spec", "resolve"]

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Type, TypeVar

from sisyphus.delayed_ops import DelayedBase

T = TypeVar("T")


def resolve(value: Any) -> Any:
    """
    Turn graph-time placeholders into runtime values.

    ``tk.Path``/``tk.Variable`` are both ``DelayedBase``, and their ``get()``
    returns the path string / variable value respectively. Nested
    :class:`Spec` objects are built, which is what makes specs composable
    (e.g. an accumulator spec carrying a model spec).
    """
    if isinstance(value, Spec):
        return value.build()
    if isinstance(value, DelayedBase):
        return value.get()
    if isinstance(value, dict):
        return {k: resolve(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(resolve(v) for v in value)
    return value


@dataclass(frozen=True)
class Spec(Generic[T]):
    """
    :param cls: the class to instantiate inside the job
    :param kwargs: constructor arguments; may contain ``tk.Path``,
        ``tk.Variable`` and nested :class:`Spec` objects
    :param unhashed_kwargs: arguments that affect *how* the object runs but
        not what it computes (worker counts, timeouts, verbosity). Jobs drop
        these before hashing via :meth:`hashed`, mirroring the
        ``hashed_arguments``/``unhashed_arguments`` split this setup already
        uses for RETURNN ``CallImport`` objects.
    """

    cls: Type[T]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    unhashed_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, **runtime: Any) -> T:
        """
        Instantiate. ``runtime`` carries values only known inside the task
        (chunk index, number of clusters, ...) and must not collide with the
        statically configured kwargs.
        """
        configured = {**self.kwargs, **self.unhashed_kwargs}
        overlap = set(configured) & set(runtime)
        if overlap:
            raise TypeError(
                f"{self.cls.__name__}: runtime argument(s) {sorted(overlap)} already "
                f"set in the spec kwargs; remove them from one side"
            )
        resolved = {k: resolve(v) for k, v in configured.items()}
        return self.cls(**resolved, **runtime)

    def hashed(self) -> "Spec[T]":
        """This spec with the unhashed arguments removed, for job hashing."""
        if not self.unhashed_kwargs:
            return self
        return Spec(self.cls, self.kwargs)

    def __repr__(self) -> str:
        return f"Spec({self.cls.__name__}, {sorted(self.kwargs.items(), key=lambda x: x[0])})"
