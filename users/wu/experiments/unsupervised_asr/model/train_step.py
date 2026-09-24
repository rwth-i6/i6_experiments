"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/model/train_steps/sae_blankfree.py.

The blank-free EMC train step: ``l_tau`` (the exact-marginal lattice), ``agg``, the label-free rate
term with its finite-difference surrogate, and -- on a k2 arm from its on-set -- the k2 lexicon term.

Port note: the flag-gated branches that are off in every reference config are CUT -- the BT
dispatch and frame pool (``lam_bt``), the trie-DP lexicon leg (``model.lexlat``: the step always
takes the ``lattice.lattice_loss`` / ``rate_term._fd_passes`` path the source took without it), the
soft and sf scorer terms and the InfoMax terms, with their monitor loops; after the port review
also the ``permute_frames_seed`` and ``zero_reverse_emission`` branches and the one-pass
``rate_fd_mode = "forward"`` difference.  The model refuses those flags at construction
(``blankfree_model.REMOVED_BLANKFREE_FLAGS``, ``emc_model.REMOVED_EMC_FLAGS``).
The helpers ``_lens`` / ``_scalar_int`` / ``_seq_tags`` are the source's
``train_steps/sae_emc.py`` helpers, verbatim.  Every kept statement is the source's, in its order,
so the marked losses and their insertion order are unchanged.
"""

from __future__ import annotations

import time
from typing import List

import torch
import returnn.frontend as rf
from returnn.tensor import TensorDict

from .lattice import build_segment_table, lattice_loss
from .rate_term import _fd_passes, expected_nonsil_tokens

__all__ = ["train_step", "seq_lens", "seq_tags"]


def _lens(tensor) -> torch.Tensor:
    return tensor.dims[1].dyn_size_ext.raw_tensor.to(device=tensor.raw_tensor.device).long()


def _scalar_int(value) -> int:
    """``run_ctx.step`` is an int under the torch engine but may be a Tensor (run_ctx.py:94)."""
    return int(getattr(value, "raw_tensor", value))


def _seq_tags(extern_data: TensorDict, key: str, batch: int) -> List[str]:
    if key not in extern_data.data:
        raise ValueError(f"the eta lookup needs extern_data[{key!r}] (INTERFACES.md §3)")
    raw = extern_data[key].raw_tensor
    tags = [str(t) for t in (raw.tolist() if hasattr(raw, "tolist") else list(raw))]
    assert len(tags) == batch, (len(tags), batch)
    return tags


def train_step(*, model, extern_data, **kwargs):
    """The blank-free EMC step (``SaeBlankfreeModelV1``)."""
    ctx = rf.get_run_ctx()
    now = time.perf_counter()
    feats_ = extern_data[model.features_key]
    units_ = extern_data[model.units_key]
    feats, units = feats_.raw_tensor.float(), units_.raw_tensor.long()
    lengths, unit_lens = _lens(feats_), _lens(units_)
    original = extern_data["original_length"].raw_tensor.long().reshape(-1)
    assert torch.equal(lengths, unit_lens)
    assert torch.all(original >= lengths)
    b = feats.shape[0]
    tags = _seq_tags(extern_data, model.seq_tag_key, b)
    # (port) the source's ``permute_frames_seed`` branch (control E4) is cut; the model refuses it
    eta = model.lookup_eta(tags, feats.device)
    # L2-1 of SAE_4A_lexlat_v2.md: the NULL recognizer (``SaeBlankfreeModelV1.null_log_q``, whose
    # class comment states the objective this makes of l_tau).  ``getattr``: a model built before
    # the knob existed has no attribute and is the default, where this is the recognizer forward
    # it has always been.
    null_theta = bool(getattr(model, "null_recognizer", False))
    if null_theta:
        log_q = model.null_log_q(feats, lengths)
    else:
        log_q = model.recognizer(feats, lengths)
    assert log_q.shape[1] == (feats.shape[1] + 2) // 3
    # (port) the source's ``zero_reverse_emission`` branch is cut; the model refuses it
    seg = build_segment_table(model.reverse, units, eta)
    tau = model.temperature(int(ctx.epoch))
    # beta of THIS sub-epoch (SAE_4A_lexlat.md D15): the config's scalar ``prior_weight`` -- the
    # very float this line has always read -- unless the arm states ``prior_weight_schedule``.  It
    # enters ``dp`` once, and every DP below reads it from there: the l_tau lattice, the rate
    # term's tilted passes (``_fd_passes`` copies ``dp``) and the soft / sf / trie legs.  RETURNN
    # evaluates the dev set through this same step at the same ``ctx.epoch``, so the dev scores
    # are this sub-epoch's objective.
    prior_weight = model.prior_weight_at(int(ctx.epoch))
    dp = dict(
        prior_log_bi=model.prior_log_bi.to(log_q.dtype), feat_lens=model.recognizer.output_lengths(lengths),
        unit_lens=unit_lens, cfg=model.lattice_cfg, temperature=tau,
        anchor_weight=model.anchor_weight(int(ctx.epoch)), prior_weight=prior_weight,
        log_q_init=None, collect_stats=model.collect_stats, history=model.prior_history,
        reduction=model.lattice_reduction, checkpoint=model.lattice_checkpoint,
    )
    dp_log_q, dp_seg = log_q, seg
    if model.lattice_float64:
        dp_log_q, dp_seg = log_q.double(), seg.double()
        dp["prior_log_bi"] = dp["prior_log_bi"].double()
    loss_utt, out = lattice_loss(dp_log_q, dp_seg, **dp)
    keep = (~out.z_zero).to(loss_utt.dtype)
    n_keep = keep.sum().clamp(min=1)
    retained = lengths.to(loss_utt.dtype)
    cycle = ((loss_utt / retained) * keep).sum() / n_keep
    agg, agg_stats = model.agg(log_q, dp["feat_lens"])
    ctx.mark_as_loss(cycle, "l_tau", scale=model.lam_tau, use_normalized_loss=False)
    if null_theta:
        # agg reads log_q alone, which is the uniform null here: a constant, logged, never trained
        ctx.mark_as_loss(agg, "agg", as_error=True, use_normalized_loss=False)
    else:
        ctx.mark_as_loss(agg, "agg", scale=model.lam_agg, use_normalized_loss=False)

    rho = model.rate_rho_hz / model.lattice_cfg.frame_rate_hz
    expected = expected_nonsil_tokens(out, model.lattice_cfg)
    rate = expected / original.to(expected.dtype)
    if null_theta:
        # The rate term's gradient reaches theta alone (the surrogate multiplies log_q), so under the
        # null it has none; its tilted passes exist only to form that surrogate and are skipped.
        # The VALUE -- the expected phone rate of the phi / prior posterior -- is still logged.
        value = ((rate - rho) / rho).square()
        value = torch.where(out.z_zero, torch.zeros_like(value), value)
        ctx.mark_as_loss((value * keep.to(value.dtype)).sum() / n_keep.to(value.dtype), "rate",
                         as_error=True, use_normalized_loss=False)
        fd = None
    eps = model.rate_fd_eps
    if not null_theta:
        fd = _fd_passes(dp_log_q, dp_seg, dp, model.lattice_cfg, eps=eps,
                        mode=model.rate_fd_mode, batched=model.rate_fd_batched)
        # (port) central only: the source's one-pass "forward" branch is cut, the model refuses it
        assert model.rate_fd_mode == "central", model.rate_fd_mode
        g = (fd.post_plus - fd.post_minus) / (2 * eps)
        fd_expected = tau * (fd.log_z[0] - fd.log_z[1]) / (2 * eps)
        value = ((rate - rho) / rho).square()
        coef = 2 * (rate - rho) / (rho * rho * original.to(rate.dtype))
        surrogate = coef.to(dp_log_q.dtype) * (g.to(dp_log_q.dtype) * dp_log_q).sum((1, 2))
        value = torch.where(out.z_zero, torch.zeros_like(value), value).to(surrogate.dtype)
        surrogate = surrogate * keep
        rate_loss = surrogate + (value - surrogate).detach()
        ctx.mark_as_loss((rate_loss * keep).sum() / n_keep, "rate", scale=model.lam_rate,
                         use_normalized_loss=False)

    # -- the k2 lexicalised lattice of SAE_4A_lexlat.md amendment 9 (emc.lexlat_k2_train), OFF ----
    # ``getattr``: a model built before this knob existed -- every banked blank-free arm, and any of
    # them re-importing this tree on a resubmit -- has no ``lexlat_k2`` attribute, so this block
    # does not run, does not import the module (and so never imports k2, which only the k2
    # environment's python has) and marks nothing, and the step is the one it has always been.
    # With a graph stated, the term is the SAME ``L_lex`` the trie arm adds, priced by the compiled
    # ``H . L . G``: ``log Z_HLG(e / temperature) - log Z_H(e / temperature)`` on THIS step's
    # emissions, added as ``mean_kept[(-L_lex) / n_unit_frames]`` at ``lam_lex`` of this sub-epoch.
    # It is an ADDED TERM, not a replacement DP: the bed's lattice above ran the banked
    # ``lattice.py`` path (the k2 arm states no ``lexlat_resources``), so ``l_tau``, the rate term
    # and every ``blankfree_*`` column are the bed's own, and the lexicon enters through this
    # gradient alone -- which reaches the EMISSIONS only.  Before the on-set sub-epoch the term is
    # not computed at all (Design 3's curriculum; the runtime refuses to be called there).
    k2_monitors = {}
    if getattr(model, "lexlat_k2", None) is not None and model.lexlat_k2.active(int(ctx.epoch)):
        k2_term, k2_monitors = model.lexlat_k2.step(
            dp_log_q, feat_lens=dp["feat_lens"], retained=retained, keep=keep,
            epoch=int(ctx.epoch), temperature=tau, cfg=model.lattice_cfg,
            # the step amendment 9.6's abort marker names, so the sub-epoch's abort can be placed
            # in the run's own learning_rates curve
            global_step=_scalar_int(ctx.step))
        ctx.mark_as_loss(k2_term, "lexlat_k2", scale=model.lexlat_k2.lam(int(ctx.epoch)),
                         use_normalized_loss=False)

    with torch.no_grad():
        tokens = out.expected_tokens
        elapsed = now - model._last_step_time if model._last_step_time else 0.0
        model._last_step_time = now
        values = {
            "l_tau_per_frame": cycle, "agg_kl_unigram": agg_stats["agg/kl_unigram"],
            "agg_kl_bigram": agg_stats["agg/kl_bigram"],
            "reverse_per_frame": ((out.expected_reverse / retained) * keep).sum() / n_keep,
            "prior_per_token": ((out.expected_prior / tokens.clamp(min=1e-6)) * keep).sum() / n_keep,
            "phone_rate_original_hz": ((tokens * 50 / original) * keep).sum() / n_keep,
            "phone_rate_retained_hz": ((tokens * 50 / retained) * keep).sum() / n_keep,
            "expected_phone_rate_hz": ((rate * 50) * keep).sum() / n_keep,
            "expected_tokens": (tokens * keep).sum() / n_keep,
            "z_zero_frac": out.z_zero.to(loss_utt.dtype).mean(),
        }
        if not null_theta:
            # (under the null recognizer these two describe finite-difference passes that did not
            # run, so they are not marked; insertion order keeps every other column where it was)
            values["rate_fd_check"] = ((fd_expected - expected).abs() * keep).sum() / (expected * keep).sum().clamp(min=1e-3)
            values["rate_dp_calls"] = fd.n_dp_calls
        else:
            # SAE_4A_lexlat_v2.md A2's rate: expected NON-SIL tokens per retained (unit) frame x 50,
            # pooled over the batch (``phone_rate_retained_hz`` above counts SIL tokens as well)
            values["nonsil_rate_retained_hz"] = (
                50 * (expected * keep).sum() / (retained * keep).sum().clamp(min=1))
        values["temperature"] = tau
        values["frames_per_sec"] = retained.sum() / elapsed if elapsed > 0 else 0.0
        for key, value in values.items():
            ctx.mark_as_loss(torch.tensor([float(value)], device=feats.device),
                             f"blankfree_{key}", as_error=True, use_normalized_loss=False)
        if getattr(model, "prior_weight_schedule", None) is not None:
            # Under its OWN key, and only on an arm that states a schedule, so no reported column
            # of a banked arm moves: the beta the DPs of this step actually ran at (``dp``), which
            # RETURNN's sub-epoch mean turns into that sub-epoch's beta in ``learning_rates``.
            # float64, so the logged 2/3 is the beta the DP ran at and not its float32 rounding.
            ctx.mark_as_loss(
                torch.tensor([float(dp["prior_weight"])], dtype=torch.float64, device=feats.device),
                "prior_weight_eff", as_error=True, use_normalized_loss=False)
        # Under their OWN keys (``lexlat_k2_...``), and only from the on-set of a k2 arm, so no
        # reported column of a banked arm moves.  ``LexlatK2Runtime.step`` says what each one is;
        # the ones the amendment reads are ``lexlat_k2_term_mean`` (``L_lex`` per unit frame -- the
        # trie arm's ``lexlat_gap_per_frame`` on the other DP), ``lexlat_k2_empty_frac`` (the share
        # of utterances whose PRUNED lattice came out empty, monitored per sub-epoch as amendment
        # 9.6 requires), ``lexlat_k2_expected_escape_words`` against ``lexlat_k2_expected_words``
        # (the escape rate the Gate's engagement clause is written on) and ``lexlat_k2_sec`` (the
        # CUDA-synchronised wall clock of the two intersections, the efficiency read).
        for key, value in k2_monitors.items():
            ctx.mark_as_loss(torch.tensor([float(value)], device=feats.device), key,
                             as_error=True, use_normalized_loss=False)


# public aliases of helpers used by reverse_model/ (same objects)
seq_lens = _lens
seq_tags = _seq_tags
