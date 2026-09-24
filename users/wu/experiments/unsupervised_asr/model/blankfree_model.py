"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/model/definitions/sae_blankfree.py.

The blank-free EMC model of the phase-4a bed (ctrl_20, k2lat_20_*, D15 ``_rp``, durinit / durfrz, the
lexlat_v2 L2-0 / L2-1 phi-first arms and the p0 supervised recognizer all build this class).

Port note: the flag-gated terms that are off in every reference config are CUT -- the InfoMax terms
(``ent_schedule``, ``lam_cons``, ``cons_views``), the trie-DP lexicon leg (``lexlat_*``), the
straight-through soft scorer (``soft_*``) and the score-function scorer (``sf_*``).  Their keyword
arguments are still accepted AT THEIR SOURCE DEFAULTS (:data:`REMOVED_BLANKFREE_FLAGS`), any other
value raises ``ValueError``, and the model no longer carries the attributes ``ent_schedule``,
``lexlat``, ``soft``, ``sf``.  The k2 lexicon leg (``lexlat_k2_*``), D15's ``prior_weight_schedule``,
L2-1's ``null_recognizer`` and the duration options are the source's, line for line.
Also cut after the port review (review_model.md section 4), each off in every reference config and
set by no other builder of this package: ``zero_reverse_emission`` (attrib step 2),
``permute_frames_seed`` (control E4; ``blankfree_permute`` itself stays, genmarg calls it
directly) and ``agg_order = 3`` with its ``agg_trigram_weight`` (the order-3 L_agg of attrib step
5).  They are refused like the terms above (:data:`REMOVED_BLANKFREE_FLAGS`); ``agg_grad_mode``
stays.
:func:`get_model` is ``SaeBlankfreeModelV1(**kwargs)``, the same call the source configs made
through ``functools.partial``.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Dict, List, Optional

import torch

from .agg import AggLoss
from .blankfree import expected_run_counts, project_text_bigram
from .emc_model import SaeEmcModelV1, refuse_removed_flags, schedule_value

__all__ = [
    "BlankfreeAggLoss",
    "REMOVED_BLANKFREE_FLAGS",
    "SaeBlankfreeModelV1",
    "get_model",
    "max_entropy_duration_logits",
]

#: The source keyword arguments of :class:`SaeBlankfreeModelV1` whose terms were removed in the
#: port, with the SOURCE defaults (the literal defaults of the source signature).  At these values
#: the source built and trained exactly what the port does.
REMOVED_BLANKFREE_FLAGS: Dict[str, object] = {
    "ent_schedule": None,
    "lam_cons": 0.0,
    "cons_views": (),
    "lexlat_resources": None,
    "lexlat_onset": 8,
    "lexlat_ramp": 3,
    "lexlat_full_lam": 1.0,
    "lexlat_contexts": 1024,
    "lexlat_escape_budget": 64,
    "lexlat_escape": True,
    "lexlat_shuffled": False,
    "lexlat_word_lm_order": 3,
    "lexlat_max_candidates": None,
    "soft_scorer": None,
    "soft_lam": 0.0,
    "soft_unigram_npz": None,
    "soft_derangement_seed": None,
    "soft_checkpoint": None,
    "sf_scorer": None,
    "sf_lam": 0.0,
    "sf_unigram_npz": None,
    "sf_num_samples": 8,
    "sf_sample_seed": 1234,
    "sf_checkpoint": None,
    # unreached branches (port review, review_model.md section 4)
    "zero_reverse_emission": False,
    "agg_order": 2,
    "agg_trigram_weight": 1.0,
    "permute_frames_seed": None,
}


class BlankfreeAggLoss(AggLoss):
    """L_agg on the RUN-COLLAPSED string, at order 2 (the bed) or order 3 (SAE_4A_attrib step 5).

    (port) Order 3 is cut: ``order`` must be 2, and the ``trigram_weight`` / ``text_tri`` arguments,
    the ``text_tri`` / ``ema_tri`` buffers and the ``*_tri`` diagnostics are gone.  The text below is
    the source's.

    Two options, both at the value the bed has always run, so ``agg_order = 2`` with
    ``agg_grad_mode = "kl"`` is the previous module bit for bit (same tensors, same order of
    operations, same ``state_dict``: the order-3 buffers are registered ONLY at order 3, so a
    banked order-2 checkpoint still loads strictly).

    ``agg_grad_mode``
        ``"kl"`` -- the bed: ``sum_o w_o KL(p_text_o || f_ema_o)``, the EMA blend inside the log, so
        only the ``(1 - decay)`` fraction of the blend carries this step's gradient.
        ``"ratio"`` -- the Empirical-ODM surrogate of the step-5 design: per order

            ``loss_o = - sum_w p_text_o(w) * f_batch_o(w) / max(f_ema_o(w).detach(), fl_o(w))``
            ``fl_o(w) = max(1e-6, 0.01 * p_text_o(w))``

        with ``f_batch`` the FREQUENCIES of this step (expected counts / total expected count, never
        raw counts) and ``f_ema`` the same EMA the KL uses, entering DETACHED and only in the
        denominator.  The floor is PER CELL and RELATIVE, so the per-cell weight
        ``p_text(w) / max(f_ema(w), fl(w))`` is at most ``1 / 0.01 = 100``: an absolute 1e-6 floor
        alone would pay a frequent trigram the model has never emitted up to 1e6, i.e. one missing
        cell would own the whole gradient.  A cell whose target is itself below 1e-6 keeps the
        absolute floor and so stays below 1 in weight.  It is the first-order expansion of the
        forward cross entropy
        ``-sum_w p_text log f`` around ``f = f_ema``: the coverage-seeking direction of Liu et al.
        (NeurIPS 2017) with the corpus-level frequency estimated across steps, but with the current
        step's gradient at FULL weight instead of ``(1 - decay)``.  At the fixed point
        ``f_batch = f_ema = p_text`` its value is exactly -1 per order and its gradient equals that
        of the forward CE at the same point.

    Diagnostics (``as_error`` only -- they never enter the trained total) are reported under their
    OWN RETURNN keys, and only for an arm that actually turns one of the two options on, so the
    bed's reported columns do not move: per order the forward CE ``-sum p_text log clamp(f_ema)``,
    the constant floor ``H(p_text_o)`` (CE minus floor is the KL a reader wants), the p_text mass
    sitting on n-grams whose ``f_ema`` is below their floor ``fl(w)`` (the cells the surrogate pays
    at the capped weight), the largest ``f_batch / f_ema`` ratio -- the UNBOUNDED one, guarded only
    against a division by zero, so a reader sees the blow-up the floor caps -- the per-order
    surrogate value, and the trigram JSD against ``p_text``.
    """

    #: the ABSOLUTE frequency floor of the ratio denominator (SAE_4A_attrib.md step 5)
    RATIO_FLOOR = 1e-6
    #: the RELATIVE floor: ``max(f_ema, max(RATIO_FLOOR, RATIO_REL_FLOOR * p_text))``, so no cell is
    #: paid more than ``1 / RATIO_REL_FLOOR`` = 100 (code review of 2026-09-19)
    RATIO_REL_FLOOR = 0.01

    def __init__(
        self,
        cfg,
        text_uni: torch.Tensor,
        text_bi: torch.Tensor,
        *,
        order: int = 2,
        grad_mode: str = "kl",
    ):
        super().__init__(cfg, text_uni, text_bi)
        if int(order) != 2:
            raise ValueError(f"agg_order is 2 (the bed); order 3 was removed in the port, not {order!r}")
        if str(grad_mode) not in ("kl", "ratio"):
            raise ValueError(f"agg_grad_mode is 'kl' (the bed) or 'ratio', not {grad_mode!r}")
        self.order = int(order)
        self.grad_mode = str(grad_mode)
        # Only an arm that changes the objective reports the extra columns; the bed's reported set
        # must not move under a running job.
        self.report_diagnostics = self.grad_mode != "kl"

    def _ratio_floor(self, p_text: torch.Tensor) -> torch.Tensor:
        """The per-cell floor ``max(1e-6, 0.01 * p_text(w))`` of the ratio denominator."""
        return (self.RATIO_REL_FLOOR * p_text).clamp(min=self.RATIO_FLOOR)

    def _ratio(self, p_text: torch.Tensor, f_batch: torch.Tensor, f_ema: torch.Tensor):
        """``-sum_w p_text(w) f_batch(w) / max(f_ema(w).detach(), max(1e-6, 0.01 p_text(w)))``."""
        denom = torch.maximum(f_ema.detach(), self._ratio_floor(p_text))
        return -(p_text * f_batch / denom).sum()

    def _diagnostics(self, suffix: str, p_text, f_batch, f_hat, *, jsd: bool = False):
        """The ``as_error`` readouts of one order, as 0-dim tensors (no host sync here)."""
        f = f_hat.detach()
        floored = f < self._ratio_floor(p_text)
        out = {
            f"agg_ce_{suffix}": -(p_text * f.clamp(min=self.cfg.floor).log()).sum(),
            f"agg_htext_{suffix}": -(p_text * p_text.clamp(min=self.cfg.floor).log()).sum(),
            f"agg_floored_mass_{suffix}": (p_text * floored.to(p_text.dtype)).sum(),
            f"agg_max_ratio_{suffix}": (f_batch.detach() / f.clamp(min=self.RATIO_FLOOR)).max(),
            f"agg_ratio_{suffix}": self._ratio(p_text, f_batch.detach(), f),
        }
        if jsd:
            m = 0.5 * (p_text + f)
            log_m = m.clamp(min=self.cfg.floor).log()
            out[f"agg_jsd_{suffix}"] = 0.5 * (
                (p_text * (p_text.clamp(min=self.cfg.floor).log() - log_m)).sum()
                + (f * (f.clamp(min=self.cfg.floor).log() - log_m)).sum()
            )
        return out

    def forward(self, log_probs: torch.Tensor, lens: torch.Tensor, *, update_ema: bool = True):
        cfg = self.cfg
        uni, bi = expected_run_counts(log_probs, lens)
        uni_b, bi_b = uni.sum(0), bi.sum(0)
        uni_n = uni_b / uni_b.sum().clamp(min=cfg.floor)
        bi_n = bi_b / bi_b.sum().clamp(min=cfg.floor)
        first = self.ema_steps == 0
        uni_hat = self._blend(uni_n, self.ema_uni, first)
        bi_hat = self._blend(bi_n, self.ema_bi, first)
        text_uni = self.text_uni.to(uni_hat.dtype)
        text_bi = self.text_bi.to(bi_hat.dtype)
        kl_uni = (text_uni * (text_uni.clamp(min=cfg.floor).log() - uni_hat.clamp(min=cfg.floor).log())).sum()
        kl_bi = (text_bi * (text_bi.clamp(min=cfg.floor).log() - bi_hat.clamp(min=cfg.floor).log())).sum()
        if self.grad_mode == "kl":
            loss = cfg.unigram_weight * kl_uni + cfg.bigram_weight * kl_bi
        else:
            loss = (cfg.unigram_weight * self._ratio(text_uni, uni_n, uni_hat)
                    + cfg.bigram_weight * self._ratio(text_bi, bi_n, bi_hat))
        extra = {}
        if self.report_diagnostics:
            extra.update(self._diagnostics("uni", text_uni, uni_n, uni_hat))
            extra.update(self._diagnostics("bi", text_bi, bi_n, bi_hat))
        if update_ema and self.training:
            with torch.no_grad():
                self.ema_uni.copy_(uni_hat.detach().float())
                self.ema_bi.copy_(bi_hat.detach().float())
                self.ema_steps += 1
        if extra and self.training:
            self._report(extra)
        stats = {
            "agg/kl_unigram": float(kl_uni.detach()),
            "agg/kl_bigram": float(kl_bi.detach()),
            "agg/expected_tokens": float(uni_b.sum().detach()),
        }
        stats.update(extra)
        return loss, stats

    @staticmethod
    def _report(values):
        """``mark_as_loss(..., as_error=True)`` for each diagnostic, when a train step is running.

        The blank-free train step marks the terms it knows; these belong to this module's own
        options, so they are marked here rather than by editing a step every other arm shares.  The
        0-dim tensors go in as tensors (a ``float()`` would sync the host on every step), and the
        lookup is skipped outside a RETURNN train step (the gradient-norm profile job calls this
        loss directly, with no run context of its own).  The guard reads ``stage``, a string:
        ``train_flag`` may be a TENSOR, and testing it would sync.
        """
        import returnn.frontend as rf

        ctx = rf.get_run_ctx()
        if ctx is None or ctx.stage != "train_step":
            return
        for name, value in values.items():
            ctx.mark_as_loss(value.reshape(1), name, as_error=True, use_normalized_loss=False)


def _max_entropy_duration_logits(cfg, mean: float) -> torch.Tensor:
    """``[n_types, d_cap]`` float64 duration logits: per type, ``lambda_k * d`` on the legal support
    ``[d_min, D_k]`` (zero elsewhere; masked by the reverse model), with lambda_k solved by
    bisection so that the law's mean is ``mean`` frames -- the maximum-entropy law with that mean."""
    d = torch.arange(1, cfg.d_cap + 1, dtype=torch.float64)
    out = torch.zeros(cfg.n_types, cfg.d_cap, dtype=torch.float64)
    for k in range(cfg.n_types):
        legal = (d >= cfg.d_min) & (d <= cfg.type_d_max(k))
        support = d[legal]
        assert float(support[0]) < mean < float(support[-1]), (k, mean, float(support[0]), float(support[-1]))

        def law_mean(lam: float) -> float:
            w = torch.softmax(lam * support, 0)
            return float((w * support).sum())

        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if law_mean(mid) < mean:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        assert abs(law_mean(lam) - mean) < 1e-9, (k, law_mean(lam), mean)
        out[k, legal] = lam * support
    return out


class SaeBlankfreeModelV1(SaeEmcModelV1):
    # ``zero_reverse_emission`` is the delta of SAE_4A_attrib.md step 2 (as amended 2026-09-19):
    # the reverse EMISSION score is identically zero for every segment, while the duration model
    # stays a proper distribution over the legal d (zeroing it too would leave an unnormalized
    # segmentation count that rewards the maximal phone rate).  It is read by the blank-free
    # train step, which then never calls the emission head at all.  Because the duration head must
    # stay at its COLD initialization, the arm also has to freeze phi, and the two are asserted
    # together here rather than left to the config.  False = the bed as it has always run.
    #
    # ``agg_order`` / ``agg_grad_mode`` / ``agg_trigram_weight`` are the step-5 options of
    # :class:`BlankfreeAggLoss`; 2 / "kl" / 1.0 is the bed, bit for bit.
    #
    # ``permute_frames_seed`` is the DESTROYED-STRUCTURE control E4 of SAE_4A_attrib.md step 6
    # (``emc.blankfree_permute``, which states the whole semantics): an int turns on ONE
    # per-utterance frame permutation of every per-frame stream of the TRAINING step, drawn from
    # this seed and the utterance tag alone, inside the valid length only -- lengths, padding, eta,
    # the schedule and every weight unchanged, the frame-level (feature, unit) pairs intact, only
    # the temporal order destroyed.  The features and the units share one 50 Hz clock here, so the
    # permutation is PER FRAME (the recognizer's stride 3 is internal to it: both streams take the
    # SAME permutation, so the pairing at its output clock is preserved).  ``None`` = no
    # permutation, which is every other arm and is the model bit for bit: the flag is only read by
    # the train step, and the forward/decode path never sees it.
    #
    # (port) The source's class comment on ``ent_schedule`` / ``lam_cons`` / ``cons_views``,
    # ``lexlat_*``, ``soft_*`` and ``sf_*`` is dropped with those terms (REMOVED_BLANKFREE_FLAGS).
    #
    # ``lexlat_k2_*`` is the K2 ARM of ``SAE_4A_lexlat.md`` amendment 9: the SAME lexicalised
    # lattice term as ``lexlat_*``, priced by a compiled ``H . L . G`` through k2's pruned dense
    # intersection instead of the trie DP, and just as default-off -- ``lexlat_k2_hlg = None``
    # (every other blank-free arm and every banked checkpoint) leaves ``model.lexlat_k2`` at
    # ``None``, does not import ``emc.lexlat_k2_train`` and the step marks no extra loss at all.
    # The runtime is a PLAIN object (the graph lives in its per-device cache, never as a submodule),
    # so it adds nothing to ``state_dict`` and a banked checkpoint still loads strictly.  It runs
    # under the k2 environment's python, which only a k2 arm's config states.  ``lexlat_k2_onset``
    # / ``_ramp`` / ``_full_lam`` are the SAME curriculum constants as the trie arm's (amendment
    # 9.2: the schedule is unchanged, only the way ``L_lex`` is computed differs), and
    # ``lexlat_k2_max_active`` is the pruning width the settling probe's summary states -- there is
    # no default for it, because a width chosen here rather than read off that probe would set the
    # arm's approximation quality silently (amendment 9.4).  ``lexlat_k2_expected_build`` is the
    # build the arm's config STATES it is training against; the runtime asserts it against both
    # graphs' own ``build.json`` (``backoff_loops`` / ``escape`` / ``sil_prob`` / ``theta``, with
    # ``shuffled`` the one field a null graph may differ in), so a repointed or stale graph stops
    # the run instead of silently changing what the lexicon term prices.  It must be named and
    # forwarded here: an unknown key would otherwise be absorbed by the base model's
    # ``**_returnn_fwd_compat`` sink and the check would never run.  ``lexlat_k2_chunk_seqs`` is
    # the sequences per pruned-intersection call (default None: the runtime's own resolution); it
    # is stated in a config so the launch granularity is readable off the arm's returnn.config.  An arm carries the trie leg or the k2
    # leg, never both: they are two computations of one term under one reported column, and the
    # pack runs them as separate arms.
    #
    # ``prior_weight_schedule`` is the PER-SUB-EPOCH phone-trigram weight beta of SAE_4A_lexlat.md
    # D15 (the REPLACEMENT ablation: beta(e) = 1 - lam_lex(e) / full_lam on a k2 arm, so the phone
    # trigram gives way where the word graph carries the text).  A list read like
    # ``temperature_schedule`` (``sae_emc.schedule_value``: entry ``epoch - 1``, the last entry held
    # beyond the list).  It is read by the TRAINING step alone, through :meth:`prior_weight_at`,
    # which is the ONE beta every DP of the step sees: the l_tau lattice (``lattice._prior_term``,
    # including its skip at exactly 0), the rate term's finite-difference passes (same ``dp``) and
    # the dev evaluation, which RETURNN runs through the same step at the same sub-epoch -- so a dev
    # score is that sub-epoch's objective.  ``None`` -- every other arm and every banked config --
    # leaves ``model.prior_weight`` the config's scalar and :meth:`prior_weight_at` returns that
    # very float, so the step is bit for bit the one it has always been.  With a schedule the
    # scalar ``prior_weight`` is NOT read and ``model.prior_weight`` is set to ``None``, so any
    # reader of beta that was not plumbed through :meth:`prior_weight_at` fails loudly instead of
    # silently running a constant 1.  No parameter or buffer is added: a banked checkpoint loads
    # strictly either way.
    #
    # ``null_recognizer`` is L2-1 of SAE_4A_lexlat_v2.md (phi-first EM): theta is replaced by a
    # NULL recognizer: ``log_q = -log V`` exactly (V = ``lattice_cfg.n_symbols`` = 40), the
    # log-softmax of zero logits, at every recognizer frame, and the recognizer is never run.
    # It is read by the TRAINING step alone, through :meth:`null_log_q`.  It has to be stated
    # together with ``freeze_recognizer = True`` (asserted here, the ``zero_reverse_emission``
    # precedent), which keeps theta out of ``emc_param_groups`` and so out of the optimizer; with
    # the forward skipped theta has no gradient path at all.  Under it:
    #   * ``l_tau`` is the lattice term on the uniform q.  Every path through the band has the SAME
    #     acoustic score ``T' log(1/V)`` (T' = (T + 2) // 3 recognizer frames), so
    #         l_tau = -log Z_tau = (T' log V) / tau
    #                 - log sum_{a in band} exp((1/tau)[log P_psi(B(a)) + log p_phi(z, seg(a) | B(a), eta)])
    #     i.e. it EQUALS the path-space generative objective -log sum_a [P_psi p_phi]^(1/tau) up to
    #     the phi-independent constant (T'/tau) log V, with T' the RECOGNIZER frames (not the unit
    #     frames T the step divides by).  It is NOT the string-level marginal
    #     -log sum_y P(y) p_phi(z | y) plus a constant: the two differ by the number of in-band
    #     alignments A(y, seg) of each string/segmentation, which is not constant (an implicit
    #     length prior the band imposes).  At tau = 1 the lattice posterior is EXACTLY the
    #     posterior of that path-space model, so tau = 1 is the exact E-step for it, and
    #     the gradient on phi is the exact generalised-EM M-step direction.  Both statements are
    #     checked against a brute-force enumeration in ``emc.test_lexlat_v2_em_config``.
    #   * ``agg`` and ``rate`` get their gradient through ``log_q`` only (theta), so under the null
    #     neither has a gradient path; the step still computes and logs them (``as_error``), with
    #     ``agg`` a constant (a function of the uniform q alone) and ``rate`` the expected phone
    #     rate of the phi/prior posterior (a phi-dependent VALUE, never trained on).  The rate
    #     term's finite-difference passes, which exist only to form theta's surrogate, are skipped.
    #   * a k2 word-graph leg (``lexlat_k2_*``) prices ``log_q`` alone and is therefore a
    #     phi-INDEPENDENT constant under the null (it cannot move phi); it is not refused but
    #     announced at construction.  Terms that exist to train or read theta (the InfoMax terms, BT,
    #     the anchor / self-distillation, the content head, the soft / sf scorers) and
    #     ``zero_reverse_emission`` (which would leave nothing trainable) are refused.
    # ``False`` -- every other arm and every banked config -- adds no attribute the step reads
    # differently, no parameter and no buffer: the step's default branch is the one it always ran.
    #
    # ``reverse_duration_freeze_mean`` (L2-1 only, needs ``null_recognizer``; default ``None`` =
    # nothing changes) is the design review's B2 fallback against rate drift under the null (no
    # term prices the token count there): phi's duration logits are set to a rate-matched,
    # label-free init and frozen (``requires_grad = False``, so ``emc_param_groups`` drops them).
    # For every type k the duration law on its legal support [d_min, D_k] is the maximum-entropy
    # law with the given mean, ``p(d | k) ∝ exp(lambda_k d)`` with lambda_k solved by bisection in
    # float64; the value registered for it is 50 / 9.66 frames per token (the bed's rho).  It
    # overwrites phi's random / loaded duration table, so it is refused with
    # ``reverse_checkpoint_path``.
    #
    # ``reverse_duration_prior`` / ``reverse_duration_prior_mode`` are the GIVEN duration prior of
    # SAE_4A_lexlat.md D17 and SAE_4A_lexlat_v2.md A9 (general knowledge only, no alignment
    # statistic).  ``reverse_duration_prior`` is the ``prior.json`` of
    # ``emc.blankfree_duration_prior_jobs.BlankfreeDurationPriorMeanJob``; its ``mean_frames`` m is
    # (50 / rho) x (retained / original frames) of the bed's train stream (that module states why).
    # At construction every NON-SIL row of phi's duration logits is set to the maximum-entropy law
    # on [d_min, D_k] with mean m (``_max_entropy_duration_logits``, the freeze-mean law above); the
    # SIL row is not touched, so it keeps the bed's uniform init and trains in both modes.  Mode
    # ``"init"``: the rows are a trainable start.  Mode ``"freeze"``: a gradient hook on
    # ``reverse.dur_logits`` zeroes the phone rows' gradient (the parameter stays ONE tensor, so the
    # ``state_dict`` keys and every reader of a checkpoint are unchanged); with the bed's Adam at
    # ``weight_decay = 0`` a gradient that is exactly zero from step 1 gives an update that is
    # exactly zero (both moments stay 0), which the tests check through RETURNN's own updater.  A
    # decoupled weight decay would move the frozen rows: the hook freezes the gradient, not the
    # value.  Everything happens in the constructor, before RETURNN loads a checkpoint, so a restart
    # resumes the trained table.  Refused with ``reverse_checkpoint_path`` (it overwrites the loaded
    # table), with ``reverse_duration_freeze_mean`` (two laws for one table) and with
    # ``freeze_reverse`` (phi frozen whole leaves nothing for the mode to decide).  ``None`` -- every
    # other arm and every banked config -- adds no parameter, buffer or hook.
    def __init__(self, *, zero_reverse_emission: bool = False, agg_order: int = 2,  # (port) refused
                 agg_grad_mode: str = "kl", agg_trigram_weight: float = 1.0,
                 permute_frames_seed: Optional[int] = None,
                 ent_schedule: Optional[list] = None, lam_cons: float = 0.0,
                 cons_views: tuple = (),
                 lexlat_resources: Optional[str] = None, lexlat_onset: int = 8,
                 lexlat_ramp: int = 3, lexlat_full_lam: float = 1.0,
                 lexlat_contexts: int = 1024, lexlat_escape_budget: int = 64,
                 lexlat_escape: bool = True, lexlat_shuffled: bool = False,
                 lexlat_word_lm_order: int = 3,
                 lexlat_max_candidates: Optional[int] = None,
                 soft_scorer: Optional[str] = None, soft_lam: float = 0.0,
                 soft_unigram_npz: Optional[str] = None,
                 soft_derangement_seed: Optional[int] = None,
                 soft_checkpoint: Optional[int] = None,
                 sf_scorer: Optional[str] = None, sf_lam: float = 0.0,
                 sf_unigram_npz: Optional[str] = None,
                 sf_num_samples: int = 8, sf_sample_seed: int = 1234,
                 sf_checkpoint: Optional[int] = None,
                 lexlat_k2_hlg: Optional[str] = None,
                 lexlat_k2_stats: Optional[str] = None,
                 lexlat_k2_resources: Optional[str] = None,
                 lexlat_k2_max_active: Optional[int] = None,
                 lexlat_k2_onset: int = 8, lexlat_k2_ramp: int = 3,
                 lexlat_k2_full_lam: float = 1.0,
                 lexlat_k2_search_beam: Optional[float] = None,
                 lexlat_k2_output_beam: Optional[float] = None,
                 lexlat_k2_min_active_states: Optional[int] = None,
                 lexlat_k2_expected_build: Optional[dict] = None,
                 lexlat_k2_chunk_seqs: Optional[int] = None,
                 prior_weight_schedule: Optional[List[float]] = None,
                 null_recognizer: bool = False,
                 reverse_duration_freeze_mean: Optional[float] = None,
                 reverse_duration_prior: Optional[str] = None,
                 reverse_duration_prior_mode: Optional[str] = None, **kwargs):
        refuse_removed_flags(
            "SaeBlankfreeModelV1",
            dict(
                ent_schedule=ent_schedule, lam_cons=lam_cons, cons_views=cons_views,
                lexlat_resources=lexlat_resources, lexlat_onset=lexlat_onset,
                lexlat_ramp=lexlat_ramp, lexlat_full_lam=lexlat_full_lam,
                lexlat_contexts=lexlat_contexts, lexlat_escape_budget=lexlat_escape_budget,
                lexlat_escape=lexlat_escape, lexlat_shuffled=lexlat_shuffled,
                lexlat_word_lm_order=lexlat_word_lm_order,
                lexlat_max_candidates=lexlat_max_candidates, soft_scorer=soft_scorer,
                soft_lam=soft_lam, soft_unigram_npz=soft_unigram_npz,
                soft_derangement_seed=soft_derangement_seed, soft_checkpoint=soft_checkpoint,
                sf_scorer=sf_scorer, sf_lam=sf_lam, sf_unigram_npz=sf_unigram_npz,
                sf_num_samples=sf_num_samples, sf_sample_seed=sf_sample_seed,
                sf_checkpoint=sf_checkpoint, zero_reverse_emission=zero_reverse_emission,
                agg_order=agg_order, agg_trigram_weight=agg_trigram_weight,
                permute_frames_seed=permute_frames_seed,
            ),
            REMOVED_BLANKFREE_FLAGS,
        )
        kwargs["recognizer_kwargs"] = dict(
            in_dim=1024, n_out=40, kernel=9, stride=3, n_layers=1,
            dropout=0.1, batch_norm=30.0, residual=True, bias=False,
        )
        kwargs["prior_history"] = "trigram"
        kwargs["prior_order"] = 3
        assert reverse_duration_freeze_mean is None or not kwargs.get("reverse_checkpoint_path"), (
            "reverse_duration_freeze_mean overwrites phi's duration table: refused with a loaded phi")
        assert reverse_duration_prior is None or not kwargs.get("reverse_checkpoint_path"), (
            "reverse_duration_prior overwrites phi's duration table: refused with a loaded phi")
        super().__init__(**kwargs)
        # (port) ``zero_reverse_emission`` / ``permute_frames_seed`` are refused above; the model
        # no longer carries them and the train step has no branch for them.
        # The lattice SKIPS the prior addend at prior_weight = 0 (``lattice._prior_term``) instead
        # of multiplying by zero; every other weight needs the table itself to be finite, which the
        # interpolated Witten-Bell fit guarantees.  Checked ONCE, at construction (a per-step check
        # would sync the host), so a table with a -inf entry is a job-start error and not a NaN
        # loss hours in.
        assert bool(torch.isfinite(self.prior_log_bi).all()), (
            "the fitted prior table carries a non-finite entry; the lattice scores it at every arc"
        )
        # -- D15's per-sub-epoch beta; at the default nothing below changes a thing --------------
        self.prior_weight_schedule = None
        if prior_weight_schedule is not None:
            schedule = [float(v) for v in prior_weight_schedule]
            assert schedule, (
                "an empty prior_weight_schedule states no beta for any sub-epoch; state None to run "
                "the scalar prior_weight"
            )
            assert all(math.isfinite(v) and v >= 0.0 for v in schedule), (
                f"prior_weight_schedule {schedule} carries a negative or non-finite beta"
            )
            self.prior_weight_schedule = schedule
            self.prior_weight = None  # see the class comment: an unplumbed reader must fail
        self.lattice_cfg = replace(self.lattice_cfg, topology="blankfree", recognizer_stride=3)
        # (port) the order-3 target of ``agg_order = 3`` is cut (refused above)
        self.agg = BlankfreeAggLoss(
            self.agg_cfg, self.agg.text_uni.detach(),
            project_text_bigram(self.agg.text_bi.detach()),
            order=int(agg_order), grad_mode=str(agg_grad_mode),
        )
        # -- the k2 lexicalised lattice of SAE_4A_lexlat.md amendment 9; default: nothing changes -
        self.lexlat_k2 = None
        if lexlat_k2_hlg is not None:
            # Imported HERE, inside the guard (the BT / InfoMax / lexlat / soft / sf idiom), so an
            # arm without a compiled graph never loads the module -- which matters more here than
            # anywhere above: importing it imports k2, and only a k2 arm runs under a python that
            # has it.
            from . import lexlat_k2_train

            assert lexlat_k2_stats is not None, (
                "the k2 leg reads the graph's build.json: it carries the bed the graph was "
                "compiled for (d_min, stride, symbol set) and the arm asserts it against the live "
                "lattice_cfg.  State lexlat_k2_stats next to lexlat_k2_hlg"
            )
            if lexlat_k2_max_active is None:
                # Amendment 9.4: the width comes from the settling probe's OWN summary (the rung
                # where log Z stops moving), never from a literal typed here.
                raise ValueError(
                    "lexlat_k2_max_active is not stated.  It is the pruning width read off "
                    "LexlatK2SettlingProbeJob's summary for THIS graph; there is no default, "
                    "because a width chosen anywhere else sets the arm's approximation quality "
                    "silently (SAE_4A_lexlat.md amendment 9.4)"
                )
            beams = {}
            if lexlat_k2_search_beam is not None:
                beams["search_beam"] = float(lexlat_k2_search_beam)
            if lexlat_k2_output_beam is not None:
                beams["output_beam"] = float(lexlat_k2_output_beam)
            if lexlat_k2_min_active_states is not None:
                beams["min_active_states"] = int(lexlat_k2_min_active_states)
            if lexlat_k2_chunk_seqs is not None:
                # the launch granularity of the pruned intersection (sequences per k2 call), not
                # an arm constant: k2 prunes per sequence, so no number moves; None keeps the
                # runtime's own resolution ($LEXLAT_K2_CHUNK_SEQS, then lexlat_k2.CHUNK_SEQS)
                beams["chunk_seqs"] = int(lexlat_k2_chunk_seqs)
            self.lexlat_k2 = lexlat_k2_train.LexlatK2Runtime(
                hlg=str(lexlat_k2_hlg), stats=str(lexlat_k2_stats),
                resources=None if lexlat_k2_resources is None else str(lexlat_k2_resources),
                max_active=int(lexlat_k2_max_active), onset=int(lexlat_k2_onset),
                ramp=int(lexlat_k2_ramp), full_lam=float(lexlat_k2_full_lam),
                expected_build=lexlat_k2_expected_build, **beams,
            )
            if isinstance(self.temperature_schedule, (list, tuple)):
                # The ramp has to FINISH inside the run, exactly as the trie arm's does (the same
                # schedule, checked at construction).
                lexlat_k2_train.assert_ramp(
                    len(self.temperature_schedule), onset=int(lexlat_k2_onset),
                    ramp=int(lexlat_k2_ramp), full=float(lexlat_k2_full_lam))
            print(f"k2 lexicon term: {self.lexlat_k2.describe()}", flush=True)
        # -- L2-1's null recognizer (SAE_4A_lexlat_v2.md); at the default nothing changes ---------
        self.null_recognizer = bool(null_recognizer)
        if self.null_recognizer:
            assert self.freeze_recognizer, (
                "null_recognizer replaces theta by zero logits, so theta must stay out of the "
                "optimizer: state freeze_recognizer = True as well"
            )
            # (port) every entry of the source's refusal list (InfoMax / BT / content / soft / sf,
            # zero_reverse_emission, lam_selfdistill, the anchor) is a removed flag, refused at any
            # non-default value by refuse_removed_flags / SaeEmcModelV1 above
            if self.lexlat_k2 is not None:
                print("null_recognizer: the k2 lexicon term prices log_q alone, which is uniform "
                      "here, so it is a phi-independent constant and gives phi no gradient",
                      flush=True)
            print(f"null_recognizer: log_q = -log {self.lattice_cfg.n_symbols} at every frame, "
                  "recognizer never run", flush=True)
        self.reverse_duration_freeze_mean = None
        if reverse_duration_freeze_mean is not None:
            assert self.null_recognizer, "reverse_duration_freeze_mean is L2-1's, under null_recognizer"
            self.reverse_duration_freeze_mean = float(reverse_duration_freeze_mean)
            with torch.no_grad():
                self.reverse.dur_logits.copy_(_max_entropy_duration_logits(
                    self.reverse.cfg, self.reverse_duration_freeze_mean
                ).to(self.reverse.dur_logits))
            self.reverse.dur_logits.requires_grad_(False)
            mean = (self.reverse.duration_log_probs().double().exp()
                    * torch.arange(1, self.reverse.cfg.d_cap + 1, dtype=torch.float64)).sum(-1)
            print(f"reverse_duration_freeze_mean: phi's duration logits frozen at the max-entropy "
                  f"law of mean {self.reverse_duration_freeze_mean:.6f} frames per type "
                  f"(realised {float(mean.min()):.6f}..{float(mean.max()):.6f})", flush=True)
        # -- D17 / A9's given duration prior; at the default nothing below changes a thing -------
        self.reverse_duration_prior_mode = None
        if reverse_duration_prior is not None or reverse_duration_prior_mode is not None:
            assert reverse_duration_prior is not None and reverse_duration_prior_mode in ("init", "freeze"), (
                f"reverse_duration_prior needs its json and a mode in ('init', 'freeze'); got "
                f"{reverse_duration_prior!r} / {reverse_duration_prior_mode!r}")
            assert self.reverse_duration_freeze_mean is None, (
                "reverse_duration_prior and reverse_duration_freeze_mean both set phi's duration table")
            assert not self.freeze_reverse, (
                "reverse_duration_prior under freeze_reverse: phi is frozen whole, so the mode "
                "would decide nothing")
            with open(str(reverse_duration_prior)) as fh:
                spec = json.load(fh)
            prior_mean = float(spec["mean_frames"])
            cfg = self.reverse.cfg
            sil = int(cfg.sil_id)
            phone_rows = torch.arange(cfg.n_types) != sil
            table = _max_entropy_duration_logits(cfg, prior_mean).to(self.reverse.dur_logits)
            with torch.no_grad():
                self.reverse.dur_logits[phone_rows] = table[phone_rows]
            if reverse_duration_prior_mode == "freeze":
                def _sil_row_only(grad: torch.Tensor, sil: int = sil) -> torch.Tensor:
                    out = torch.zeros_like(grad)
                    out[sil] = grad[sil]
                    return out

                self.reverse.dur_logits.register_hook(_sil_row_only)
            self.reverse_duration_prior_mode = str(reverse_duration_prior_mode)
            self.reverse_duration_prior_mean = prior_mean
            with torch.no_grad():
                probs = self.reverse.duration_log_probs().double().exp()
            e_d = (probs * torch.arange(1, cfg.d_cap + 1, dtype=torch.float64)).sum(-1)
            assert torch.allclose(probs.sum(-1), torch.ones_like(e_d), atol=1e-6), probs.sum(-1)
            print(f"reverse_duration_prior ({self.reverse_duration_prior_mode}): phone rows at the "
                  f"max-entropy law of mean {prior_mean:.6f} frames from {reverse_duration_prior}; "
                  f"SIL row untouched and trainable", flush=True)
            print("reverse_duration_prior E[d] per type at start: " + ", ".join(
                f"{k}:{float(v):.4f}" for k, v in enumerate(e_d)), flush=True)

    def null_log_q(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``[B, (T + 2) // 3, V]`` float32 ``-log V`` (the log-softmax of zero logits): L2-1's null theta.

        The shape and dtype of ``self.recognizer(feats, lengths)`` (its forward ends in
        ``log_softmax(x.float())`` at stride 3), with no parameter involved, so nothing upstream
        of the lattice carries a gradient.  ``lengths`` is unused: padded frames are masked by
        ``feat_lens`` downstream exactly as the recognizer's padded outputs are.
        """
        del lengths
        b, t = int(feats.shape[0]), int(feats.shape[1])
        v = int(self.lattice_cfg.n_symbols)
        return torch.full((b, (t + 2) // 3, v), -math.log(v), dtype=torch.float32, device=feats.device)

    def prior_weight_at(self, epoch: int) -> float:
        """beta of sub-epoch ``epoch`` (1-based): the scalar ``prior_weight``, or D15's schedule.

        Without a schedule this returns ``self.prior_weight`` itself -- the float the step has
        always handed the DP -- so nothing moves.  ``getattr``: a model object built before the
        knob existed has no ``prior_weight_schedule`` attribute and is the scalar case.
        """
        schedule = getattr(self, "prior_weight_schedule", None)
        if schedule is None:
            return self.prior_weight
        return schedule_value(schedule, int(epoch))


def get_model(**kwargs) -> SaeBlankfreeModelV1:
    """RETURNN ``get_model``: the source configs' ``functools.partial(SaeBlankfreeModelV1, **args)``."""
    return SaeBlankfreeModelV1(**kwargs)


# public aliases of helpers used by reverse_model/ (same objects)
max_entropy_duration_logits = _max_entropy_duration_logits
