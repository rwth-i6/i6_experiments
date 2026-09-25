"""The k2 lexicon leg with a PER-CHUNK backward, for the P1 lift-ladder arms (SAE_i6_P1.md, Task B, G1.M).

WHY.  ``model.lexlat_k2_train.LexlatK2Runtime.step`` (the HELD path) intersects the batch with the HLG
in chunks of ``chunk_seqs`` sequences but keeps EVERY chunk's lattice and autograd graph alive until
RETURNN's outer ``backward``, so ``lexlat_k2_chunk_seqs`` does not bound the memory the leg holds
(JUPITER A13: 80-84 GB reserved at steps 1-2 of the rt arms).  This module computes the SAME term,
the SAME monitors and the SAME gradient, but runs each chunk's backward as soon as that chunk's
lattice is read, and drops the lattice before the next chunk is intersected.

HOW (exactness argument, pinned by ``tests/test_rt_chunked_backward.py``):

* ``dense_in`` is built by the held ``dense`` (the live tensor, with its graph to ``log_q``); the HLG
  chunks read a DETACHED LEAF copy of it, so each chunk's backward stops at the leaf and lands in
  ``leaf.grad`` (the chunks read disjoint row slices, so every element receives exactly one non-zero
  contribution -- the same as the held path's slice backward).
* The upstream gradient each chunk's ``log Z_HLG`` receives in the held path is
  ``lam * d term / d z_hlg_i``.  It depends on the row only through ``keep_i``, ``retained_i``,
  ``n_keep``, ``lam`` and whether ``z_hlg_i`` is finite (the empty-lattice rule), never on the
  value of ``z_hlg``.  It is computed ONCE per step by autograd on the held expression itself
  (:func:`z_hlg_upstream`, a float64 leaf in place of ``z_hlg``, every kept row taken as scored), and
  each chunk takes its rows of it with the rows whose lattice came out empty set to 0 -- exactly what
  the held ``torch.where(scored, ...)`` backward produces.
* The leaf's gradient enters RETURNN's backward through :class:`_InjectGrad`: a function of the
  LIVE ``dense_in`` whose value is exactly 0 and whose backward hands ``dense_in`` the accumulated
  gradient times ``incoming / lam``.  RETURNN multiplies the marked loss by its ``scale`` = ``lam``
  in the term's float64, so ``incoming / lam`` is exactly 1.0 and ``dense_in`` receives the held
  path's gradient bit for bit; from there the cat / temperature / dtype backward to ``log_q`` and on
  into theta is the held path's own.  This is the surrogate ``(log_q * dense_in.grad).sum()`` of the
  design review (item 1) written so that its VALUE is structurally zero (a product with ``-inf``
  emissions cannot turn it into NaN) and its gradient does not depend on how the scale is rounded.
* ``log Z_H`` (the unpruned ``H`` leg) is unchanged: one ``k2.intersect_dense`` call on the live
  ``dense_in``, its graph held until RETURNN's backward as in the held path.
* The term's VALUE is the held expression on the same numbers; the monitors are the held
  ``_monitors`` dict built from per-chunk reads (lattice sizes, expected word / escape counts) taken
  BEFORE each chunk's lattice is dropped.  The stability read, the empty-lattice rule
  (``n_empty > empty_raise_frac * b`` per BATCH, then :meth:`_abort_on_empty`) and ``check_bed`` are
  the held methods, called at the held points.
* Under ``torch.no_grad`` (RETURNN's dev pass) or with a ``log_q`` that needs no gradient, no chunk
  backward runs and the term is the held value.

TWO MONITORS CHANGE MEANING (not value-compared by the test): ``lexlat_k2_sec`` and
``lexlat_k2_peak_reserved_gib`` time and measure the leg INCLUDING the per-chunk backwards (in the
held path those ran later, inside RETURNN's outer backward).

THREE MEMORY MONITORS ARE ADDED (G1.M; not in the held path's dict, emitted after
``lexlat_k2_peak_reserved_gib`` as the same ``as_error`` monitors, so they never enter the loss; 0.0
and no CUDA call on CPU):

* ``lexlat_k2_pre_peak_allocated_gib`` / ``lexlat_k2_pre_peak_reserved_gib``: the allocator's peak
  read JUST BEFORE this step's ``reset_peak_memory_stats``.  At step n it covers everything since the
  previous reset: step n-1's k2 leg, backward and optimizer step, then step n's forward and the
  once-per-sub-epoch stability read (at step 1 of a sub-epoch: since RETURNN's epoch-start reset).
  The per-step peak G1.M reads is the max of this and the post-reset peak (allocated: RETURNN's
  ``mem_usage:cuda:0``; reserved: ``lexlat_k2_peak_reserved_gib``).
* ``lexlat_k2_device_used_gib``: ``torch.cuda.mem_get_info`` total minus free at the end of the k2
  leg, a POINT sample (not a peak) of the whole device, including the CUDA context and allocations
  outside the torch allocator.

A NON-FINITE STABILITY READ IS OMITTED, NOT EMITTED (the one monitor whose key set differs from the
held path's).  The held read catches its own failure and returns ``nan`` (``LexlatK2Runtime._stability``:
"MAY NOT KILL THE ARM"), but the train step marks every monitor as an ``as_error`` loss and the rt
config sets ``stop_on_nonfinite_train_score = True``; RETURNN's check (``returnn/torch/engine.py``,
``if self._stop_on_nonfinite_train_score``) scans every accumulated loss, so a ``nan`` read would raise
``Inf/nan score`` and end the arm.  Here ``lexlat_k2_stability`` is emitted only when it is finite; a
failed or empty read leaves it out for that sub-epoch (the value is cached per sub-epoch, so it is
absent from every step of it).  The read stays recoverable: the held read's print line (``stability at
sub-epoch N: median nan ... over 0 of n`` or ``... FAILED (...)``) is in the job log, and that
sub-epoch's ``learning_rates`` entry carries the other ``lexlat_k2_*`` columns but no
``lexlat_k2_stability`` -- a missing read, never a number.  The non-finite stop still covers the total
loss and every other loss and monitor.  No value, gradient or job hash moves.

SELECTION.  Nothing in ``model/`` or ``training/`` imports this module.  The rt arms' RETURNN config
rebinds ``train_step`` to :func:`train_step` here in its (unhashed) ``python_epilog``
(:data:`EPILOG`, used by ``config/sae_i6_p1_ladder.py``), which swaps the model's runtime class on its
first call and then runs the held ``model.train_step.train_step`` unchanged.

Every ``k2`` import is local, as in ``model/lexlat_k2*.py``.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import torch

from ..model import lexlat_k2 as K
from ..model import lexlat_k2_train as KT

__all__ = ["ChunkedBackwardLexlatK2Runtime", "z_hlg_upstream", "install_chunked_backward", "train_step",
           "EPILOG"]

_PKG = "i6_experiments.users.wu.experiments.unsupervised_asr"

#: the RETURNN config epilog that selects this path.  It runs after the prolog (which put the recipe
#: on ``sys.path`` and bound ``train_step`` to the held step) and rebinds ``train_step``.
EPILOG = (
    "# P1 rt arms: the k2 lexicon leg with a per-chunk backward (exact; reverse_model/rt_chunked_backward.py)\n"
    f"from {_PKG}.reverse_model.rt_chunked_backward import train_step as train_step\n"
)


def z_hlg_upstream(*, keep: torch.Tensor, retained: torch.Tensor, lam: float, term_dtype: torch.dtype,
                   z_dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The held path's gradient of ``lam * term`` w.r.t. ``log Z_HLG``, every kept row taken as scored.

    The held expression (``LexlatK2Runtime.step``) evaluated by autograd on a leaf in place of
    ``z_hlg``: ``term = ((-where(scored, z - z_h, 0) / retained) * w).sum() / n_keep``, returned
    ``.to(term_dtype)``, backpropagated from ``lam`` in ``term_dtype`` (RETURNN's ``loss * scale``).
    The value of ``z`` and ``z_h`` does not enter the gradient, so zeros stand in for both.  A row
    whose lattice is empty takes 0 instead (the caller applies that per chunk).
    """
    device = keep.device
    b = int(keep.shape[0])
    z = torch.zeros(b, dtype=z_dtype, device=device, requires_grad=True)
    with torch.enable_grad():
        z_h = torch.zeros(b, dtype=z_dtype, device=device)
        scored = keep > 0
        w = scored.to(z.dtype)
        l_lex = torch.where(scored, z - z_h, torch.zeros_like(z))
        n_keep = keep.sum().clamp(min=1)
        term = ((-l_lex / retained.to(l_lex.dtype)) * w).sum() / n_keep.to(l_lex.dtype)
        term = term.to(term_dtype)
        (grad,) = torch.autograd.grad(term, z, grad_outputs=torch.tensor(float(lam), dtype=term_dtype,
                                                                          device=device))
    return grad.detach()


class _InjectGrad(torch.autograd.Function):
    """Value 0; backward hands ``dense_in`` the pre-computed ``grad`` times ``incoming / lam``."""

    @staticmethod
    def forward(ctx, dense_in, grad, lam_t):  # noqa: D102
        ctx.save_for_backward(grad, lam_t)
        return torch.zeros((), dtype=lam_t.dtype, device=lam_t.device)

    @staticmethod
    def backward(ctx, incoming):  # noqa: D102
        grad, lam_t = ctx.saved_tensors
        ratio = incoming / lam_t  # exactly 1.0 when RETURNN's scale is lam (same dtype, same float)
        return grad * ratio.to(grad.dtype), None, None


class ChunkedBackwardLexlatK2Runtime(KT.LexlatK2Runtime):
    """``LexlatK2Runtime`` whose :meth:`step` runs each HLG chunk's backward before the next chunk.

    Only :meth:`step` differs; every other method (graph cache, ``check_bed``, the stability read,
    the abort marker, the expected-word read of one chunk) is the held class's own.
    """

    def step(self, log_q: torch.Tensor, *, feat_lens: torch.Tensor, retained: torch.Tensor,
             keep: torch.Tensor, epoch: int, temperature: float, cfg, global_step: Optional[int] = None):
        """The held ``step``'s ``(term, monitors)`` (module doc); the gradient reaches ``log_q``
        through RETURNN's backward of ``lam * term`` exactly as in the held path."""
        assert self.active(epoch), (
            f"step() at sub-epoch {epoch}, where lam_lex = 0; the train step must not call the "
            "lexicon leg before the on-set (Design 3's curriculum)"
        )
        self.check_bed(cfg)
        facts = self.facts()
        assert int(log_q.shape[2]) == int(facts["n_phones"]), (log_q.shape, facts["n_phones"])
        b = int(log_q.shape[0])
        cuda = log_q.is_cuda
        dense_in = self.dense(log_q, temperature)
        stability = self._stability(dense_in, feat_lens=feat_lens, retained=retained, epoch=epoch,
                                    temperature=temperature)
        pre_alloc_gib = pre_reserved_gib = 0.0
        if cuda:
            torch.cuda.synchronize()
            # G1.M: the peak since the previous reset (including the stability read above), read
            # before the reset below discards it
            pre_alloc_gib = torch.cuda.max_memory_allocated() / 2 ** 30
            pre_reserved_gib = torch.cuda.max_memory_reserved() / 2 ** 30
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        need_grad = bool(torch.is_grad_enabled() and dense_in.requires_grad)
        lam = float(self.lam(epoch))
        upstream = (z_hlg_upstream(keep=keep, retained=retained, lam=lam, term_dtype=log_q.dtype)
                    if need_grad else None)
        leaf = dense_in.detach().requires_grad_(need_grad)
        z_hlg, reads = self.log_z_hlg_chunked_backward(leaf, feat_lens, temperature, upstream=upstream)
        z_h = self.log_z_h(dense_in, feat_lens)

        empty = ~torch.isfinite(z_hlg)
        n_empty = int(empty.sum())
        assert bool(torch.isfinite(z_h).all()), (
            "log Z_H is not finite on every utterance: H accepts EVERY frame string, so this is a "
            "dense-tensor or supervision-segment defect, not a pruning loss"
        )
        if n_empty > self.spec.empty_raise_frac * b:
            self._abort_on_empty(n_empty=n_empty, b=b, epoch=epoch, global_step=global_step)
        scored = (keep > 0) & (~empty)
        w = scored.to(z_hlg.dtype)
        l_lex = torch.where(scored, z_hlg - z_h, torch.zeros_like(z_hlg))
        n_keep = keep.sum().clamp(min=1)
        term = ((-l_lex / retained.to(l_lex.dtype)) * w).sum() / n_keep.to(l_lex.dtype)
        monitors = self._monitors_from_reads(reads, l_lex=l_lex.detach(), scored=w, retained=retained,
                                             feat_lens=feat_lens, epoch=epoch, n_empty=n_empty, b=b)
        if cuda:
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        peak_gib = (torch.cuda.max_memory_reserved() / 2 ** 30) if cuda else 0.0
        device_used_gib = 0.0
        if cuda:
            free_b, total_b = torch.cuda.mem_get_info(log_q.device)
            device_used_gib = (total_b - free_b) / 2 ** 30
        dtype, device = l_lex.dtype, l_lex.device
        monitors["lexlat_k2_sec"] = torch.as_tensor(float(sec), dtype=dtype, device=device)
        monitors["lexlat_k2_peak_reserved_gib"] = torch.as_tensor(
            float(peak_gib), dtype=dtype, device=device)
        for key, value in (("lexlat_k2_pre_peak_allocated_gib", pre_alloc_gib),
                           ("lexlat_k2_pre_peak_reserved_gib", pre_reserved_gib),
                           ("lexlat_k2_device_used_gib", device_used_gib)):
            monitors[key] = torch.as_tensor(float(value), dtype=dtype, device=device)
        # the stability read is emitted only when it is a number: a failed or empty read (``nan``) is
        # OMITTED, so RETURNN's non-finite stop (which scans every loss, ``as_error`` ones included)
        # cannot end the arm on a diagnostic, and the sub-epoch's ``learning_rates`` entry has no
        # ``lexlat_k2_stability`` column -- the Gate's missing read, never a number (module doc)
        if math.isfinite(float(stability)):
            monitors["lexlat_k2_stability"] = torch.as_tensor(
                float(stability), dtype=dtype, device=device)
        # the term: the held expression, with z_hlg a constant here and z_h still on its graph (the
        # H leg's gradient reaches dense_in through it, as held); the HLG leg's gradient is the
        # accumulated leaf gradient, injected at the live dense tensor by a zero-valued addend
        out = term.to(log_q.dtype)
        if not need_grad:
            return out, monitors
        grad = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
        lam_t = torch.tensor(lam, dtype=log_q.dtype, device=log_q.device)
        return out + _InjectGrad.apply(dense_in, grad, lam_t), monitors

    def log_z_hlg_chunked_backward(self, leaf: torch.Tensor, feat_lens: torch.Tensor, temperature: float,
                                   *, upstream: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """``lexlat_k2.chunked_tot_scores``' loop (same bounds, segments, calls), with per chunk:
        the monitor reads, then ``tot.backward(upstream rows, 0 where the lattice is empty)`` when
        ``upstream`` is given, then the lattice is dropped.

        :return: ``(log Z_HLG detached, reads)``; ``reads`` = ``{"sizes": lattice_sizes' dict,
            "words", "escapes"}`` in the batch's order.
        """
        import k2

        graph = self.graph(leaf.device, temperature)
        b = int(leaf.shape[0])
        lens_i32 = feat_lens.detach().cpu().to(torch.int32)
        assert int(lens_i32.numel()) == b, (lens_i32.shape, leaf.shape)
        facts = self.facts()
        n_words, unk = int(facts["n_words"]), int(facts["unk_word"])
        parts, words, escapes = [], [], []
        sizes = {"n_fsas": 0, "states": 0, "arcs": 0}
        for start, stop in K.chunk_bounds(b, int(self.spec.chunk_seqs)):
            n = stop - start
            seg = torch.stack([torch.arange(n, dtype=torch.int32),
                               torch.zeros(n, dtype=torch.int32),
                               lens_i32[start:stop]], dim=1)
            dense = k2.DenseFsaVec(leaf[start:stop], seg)
            lattice = k2.intersect_dense_pruned(
                graph, dense, search_beam=float(self.spec.search_beam),
                output_beam=float(self.spec.output_beam),
                min_active_states=int(self.spec.min_active_states),
                max_active_states=int(self.spec.max_active))
            tot = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
            # the monitors' reads of this chunk, before its lattice goes (lexlat_k2.lattice_sizes and
            # LexlatK2Runtime._expected_words, chunk by chunk)
            for key, value in K._sizes(lattice).items():
                sizes[key] += int(value)
            with torch.no_grad():
                w_c, e_c = self._expected_words_one(lattice, n_words=n_words, unk=unk)
            words.append(w_c)
            escapes.append(e_c)
            if upstream is not None:
                rows = upstream[start:stop].to(device=tot.device, dtype=tot.dtype)
                g = torch.where(torch.isfinite(tot.detach()), rows, torch.zeros_like(rows))
                tot.backward(g)
            parts.append(tot.detach())
            del lattice, dense, tot
        words_t, escapes_t = torch.cat(words), torch.cat(escapes)
        assert int(words_t.numel()) == b, (
            f"the chunks cover {int(words_t.numel())} sequences and the batch has {b}; "
            "intersect_dense_pruned returned one FSA per supervision segment until now"
        )
        return torch.cat(parts), {"sizes": sizes, "words": words_t, "escapes": escapes_t}

    def _monitors_from_reads(self, reads, *, l_lex, scored, retained, feat_lens, epoch, n_empty, b):
        """The held ``_monitors`` dict (same keys, order and expressions) from per-chunk reads."""
        device = l_lex.device
        dtype = l_lex.dtype
        n_scored = scored.sum().clamp(min=1)
        sizes = reads["sizes"]
        frames = float(feat_lens.sum())
        words, escapes = reads["words"], reads["escapes"]
        as_t = lambda v: torch.as_tensor(float(v), dtype=dtype, device=device)  # noqa: E731
        return {
            "lexlat_k2_lam": as_t(self.lam(epoch)),
            "lexlat_k2_term_mean": ((l_lex / retained.to(dtype)) * scored).sum() / n_scored,
            "lexlat_k2_n_empty": as_t(n_empty),
            "lexlat_k2_empty_frac": as_t(n_empty / max(b, 1)),
            "lexlat_k2_lattice_arcs_per_frame": as_t(sizes["arcs"] / max(frames, 1.0)),
            "lexlat_k2_lattice_states_per_frame": as_t(sizes["states"] / max(frames, 1.0)),
            "lexlat_k2_expected_words": (words.to(dtype) * scored).sum() / n_scored,
            "lexlat_k2_expected_escape_words": (escapes.to(dtype) * scored).sum() / n_scored,
        }


def install_chunked_backward(runtime) -> ChunkedBackwardLexlatK2Runtime:
    """Switch a held ``LexlatK2Runtime`` to this class in place (its caches and spec are kept)."""
    if isinstance(runtime, ChunkedBackwardLexlatK2Runtime):
        return runtime
    assert type(runtime) is KT.LexlatK2Runtime, (
        f"expected the held LexlatK2Runtime, got {type(runtime)!r}")
    runtime.__class__ = ChunkedBackwardLexlatK2Runtime
    print(f"rt_chunked_backward: per-chunk k2 backward installed (chunk_seqs = {runtime.spec.chunk_seqs})",
          flush=True)
    return runtime


def train_step(*, model, extern_data, **kwargs):
    """The held ``model.train_step.train_step`` with the model's k2 runtime on the per-chunk backward.

    Refuses a model without the k2 leg: this step is selected only for the rt arms, which carry it.
    """
    from ..model import train_step as held

    runtime = getattr(model, "lexlat_k2", None)
    assert runtime is not None, (
        "rt_chunked_backward.train_step is selected for a model without the k2 lexicon leg; only the "
        "P1 rt arms (lexlat_k2_hlg stated) use it")
    install_chunked_backward(runtime)
    return held.train_step(model=model, extern_data=extern_data, **kwargs)
