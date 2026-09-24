"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/model/definitions/sae_emc.py.

Port note: the base model of the blank-free arms.  The flag-gated terms that are off in every
reference config are CUT -- the entropy term (``lam_ent`` / ``ent_ramp_epochs``), the S3b-C
consistency term (``lam_cons``, ``cons_views``, ``cons_specaug``, ``primary_specaug_until_epoch``,
``primary_specaug``, ``cons_pert_features_key``), the S3b-F content term (``lam_content``,
``content_codes_key``), the BT auxiliary (``lam_bt``, ``bt_depth``, ``bt_ramp_epochs``,
``bt_batch_sents``, ``bt_text_path``) and the candidate-LM training (``candidate_prior_order``,
``candidate_path_draws``, ``candidate_counts_path``, ``candidate_source_sha256``).  Their keyword
arguments are still accepted AT THEIR SOURCE DEFAULTS, so a config written for the source builds the
same model; any other value raises ``ValueError`` (:data:`REMOVED_EMC_FLAGS`).  The model no longer
carries the matching attributes (``lam_ent``, ``lam_cons``, ``cons_*``, ``primary_specaug*``,
``lam_content``, ``content_codes_key``, ``lam_bt``, ``bt_*``, ``candidate_*``).  Everything else --
the constructor order, the buffers, the checkpoint loads, the validation -- is the source's.

Also refused (port review, review_model.md section 4): knobs the blank-free train step never
reads, and branches no reference config reaches.  ``rate_hinge`` (no step reads it),
``lam_selfdistill != 0`` and a nonzero ``anchor_weight_schedule`` (both only built the frozen
``q_init`` copy; the step hands the lattice ``log_q_init = None``, so neither ever reached the
objective), ``prior_classes`` / ``prior_history = "class_trigram"`` (the blank-free model forces the
full trigram) and ``rate_fd_mode = "forward"``.  Any value other than the source default raises
``ValueError``; ``q_init``, its ``train()`` override, the class-trigram table and the attributes
``lam_selfdistill`` / ``rate_hinge`` / ``prior_classes`` are gone.  The source text below still
names them.

``get_model`` container for the SAE §4a EMC loop (SAE_4A.md:132-140, stage S0b).

One ``ReturnnTrainingJob`` trains the whole cycle, so every piece of the objective lives here as a
submodule or a buffer:

* ``recognizer`` -- ``recognizer.ConvRecognizer`` (theta): the small conv net over the frozen
  wav2vec2-lv60 L15 features, blank + 40 phones, stride 1 so T = S (INTERFACES.md §2).
* ``reverse`` -- ``reverse.SegmentalReverseModel`` (phi): the segmental reverse model
  p(z | y, eta) with explicit durations, d_min = 2, D = 25 / D_sil = 50.
* ``q_init`` -- a frozen deep copy of ``recognizer`` at its init checkpoint: the anchor of S2 arm B
  (SAE_4A.md:77-83) and the teacher of S2c arm E's self-distillation. Built ONLY when the anchor
  weight can be nonzero or ``lam_selfdistill != 0``, so arm A's module tree, its state_dict and its
  resume path are untouched by either option.
* ``agg`` -- ``agg.AggLoss``, holding the EMA of the expected m-gram counts in its buffers so a
  resume continues the same average.
* buffers ``prior_log_bi`` (frozen P_psi, the ``[|h|, K]`` table of the configured
  ``prior_history``) and ``eta_table`` (frozen PCA-16 speaker vectors).

Trained parameters: theta and phi. Everything else is frozen by construction -- the prior tables and
eta are buffers with no gradient at all, and ``q_init`` is ``requires_grad_(False)`` and pinned to
``eval()``.

Constants with no default here (they change the objective and the plan fixes no value):
``temperature_schedule``, ``lam_agg``, ``count_ema_decay``. A config must state them.
``lam_tau`` and ``lam_selfdistill`` DO default (1.0 / 0.0 = the loop as it has always run); only
S2c arm E sets them, to 0 and 1.  ``lam_rate`` defaults to 0 for the same reason (the S3b-R rate
term off), and its ``rate_rho_hz`` has no default because the target rate is a number the stage
must state -- see the constructor.  ``rate_hinge`` (the one-sided form of that term) defaults to
off for the same reason.
Constants that DO trace: ``band = 25`` (SAE_4A.md:76), ``prior_weight = 1`` (SAE_4A.md:134,
"beta = 1"), ``anchor_weight = 0`` = anchor off (orchestrator brief 2026-09-15), the duration bounds
and d_min from ``reverse.ReverseConfig``, the 50 Hz clock (SAE_4A.md:43-49).

``prior_history`` / ``prior_classes`` choose the history the prior conditions on inside L_tau (the
lattice's ``PriorHistory``); ``"bigram"`` is what every stage up to S2c ran and is the default, so a
config that does not mention them builds exactly the model it always did. A class trigram states its
partition (a list, or the named ``"manner8"``), and ``prior_order`` must agree with the history.

``lattice_reduction`` / ``lattice_checkpoint`` are the two knobs of the DP itself (D3 and D4 of
reports/estimate_prior_order_2026-09-15.full.md): how a frame reduces over the history axis, and
the stride of the checkpointed forward table. Their defaults (``"auto"`` / ``0`` = the whole table
resident) are what every bigram stage ran. Above the bigram the un-checkpointed table does not fit,
so ``lattice_checkpoint`` must be set -- the constructor refuses that combination rather than let a
job pay the queue and die (review_lattice_d3d4_2026-09-15, F1). No keyword is swallowed any more:
an unknown ``model_args`` key raises here, at job start, instead of being a silent no-op.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from .agg import AggConfig, AggLoss, text_target_counts
from .lattice import LatticeConfig, _use_matmul, build_prior_history
from .reverse import ReverseConfig, SegmentalReverseModel

__all__ = [
    "PRIOR_HISTORY_ORDER",
    "REMOVED_EMC_FLAGS",
    "SaeEmcModelV1",
    "refuse_removed_flags",
    "schedule_value",
]

Schedule = Union[float, int, Sequence[float]]

# The order of the LM term INSIDE the objective is the lattice's history axis, so ``prior_order``
# and ``prior_history`` are one statement made twice and must agree (user ruling 2026-09-15: no new
# stage carries a bigram term; reports/estimate_prior_order_2026-09-15.full.md section 6).
PRIOR_HISTORY_ORDER = {"bigram": 2, "trigram": 3}  # (port) "class_trigram" cut, refused

#: The source keyword arguments whose terms were removed in the port, with the SOURCE defaults
#: (``entropy_term.DEFAULT_RAMP_EPOCHS = 4``, ``consistency.DEFAULT_VIEWS = ("specaug",)``,
#: ``consistency.PERT_FEATURES_KEY = "features_pert"``, ``content_term.CONTENT_CODES_KEY =
#: "content_codes"``, the BT literals ``"full"`` / 4 / 128).  At these values the source built and
#: trained exactly what the port does; any other value turned a removed term on.
REMOVED_EMC_FLAGS: Dict[str, object] = {
    "lam_ent": 0.0,
    "ent_ramp_epochs": 4,
    "lam_cons": 0.0,
    "cons_views": ("specaug",),
    "cons_specaug": None,
    "primary_specaug_until_epoch": 0,
    "primary_specaug": None,
    "cons_pert_features_key": "features_pert",
    "lam_content": 0.0,
    "content_codes_key": "content_codes",
    "lam_bt": 0.0,
    "bt_depth": "full",
    "bt_ramp_epochs": 4,
    "bt_batch_sents": 128,
    "bt_text_path": None,
    "candidate_prior_order": 0,
    "candidate_path_draws": 0,
    "candidate_counts_path": None,
    "candidate_source_sha256": None,
    # dead knobs and unreached branches (port review, review_model.md section 4)
    "lam_selfdistill": 0.0,
    "rate_hinge": False,
    "prior_classes": None,
}


def _is_default(value, default) -> bool:
    if isinstance(default, tuple):
        return value is not None and not isinstance(value, str) and tuple(value) == default
    if default is None:
        return value is None
    return value == default


def refuse_removed_flags(
    owner: str, given: Mapping[str, object], defaults: Mapping[str, object]
) -> None:
    """Raise ``ValueError`` naming every removed flag that is set to a non-default value."""
    bad = {k: given[k] for k in defaults if not _is_default(given[k], defaults[k])}
    if bad:
        raise ValueError(
            f"{owner}: {sorted(bad)} = {[bad[k] for k in sorted(bad)]!r} turn on terms that were "
            f"removed in the port (off in every reference run); the accepted values are the "
            f"source defaults {({k: defaults[k] for k in sorted(bad)})!r}"
        )


def schedule_value(schedule: Schedule, epoch: int) -> float:
    """A per-sub-epoch constant: a scalar, or a list indexed by sub-epoch (1-based, last held).

    The plan's two schedules are exactly this shape -- ``alpha`` decaying 1 -> 0 over sub-epochs 1-4
    (SAE_4A.md:77-83) and ``tau`` annealed 8 -> 2 over sub-epochs 1-4 (SAE_4A.md:158-160) -- so the
    config states the per-sub-epoch values and this function does no interpolation of its own.
    """
    if isinstance(schedule, (int, float)):
        return float(schedule)
    seq = list(schedule)
    assert seq, "an empty schedule states nothing"
    return float(seq[min(max(int(epoch), 1), len(seq)) - 1])


def _load_state(
    module: nn.Module, path: str, *, name: str, allow_missing_prefix: Optional[str] = None
) -> None:
    """Load a standalone checkpoint into ``module`` and refuse a silent miss.

    ``allow_missing_prefix`` is the ONE exception (the source used it for the content head, which
    is removed in the port; kept as the source's signature).  Every other missing key -- and every
    unexpected key, always -- is a hard error.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    res = module.load_state_dict(sd, strict=False)
    missing = list(res.missing_keys)
    if allow_missing_prefix:
        fresh = [k for k in missing if k.startswith(allow_missing_prefix)]
        missing = [k for k in missing if not k.startswith(allow_missing_prefix)]
        if fresh:
            print(f"{name}: {len(fresh)} key(s) left at their fresh init: {sorted(fresh)}", flush=True)
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} key(s) not loaded from {path}, "
                           f"e.g. {missing[:4]}")
    if res.unexpected_keys:
        raise RuntimeError(f"{name}: {len(res.unexpected_keys)} unexpected key(s) in {path}, "
                           f"e.g. {res.unexpected_keys[:4]}")


class SaeEmcModelV1(nn.Module):
    def __init__(
        self,
        *,
        # --- the objective's undetermined constants: no defaults (see the module docstring)
        temperature_schedule: Schedule,
        lam_agg: float,
        count_ema_decay: float,
        # The scale of the CYCLE objective itself. 1.0 is the loop as every stage up to S2b runs
        # it, so a config that does not mention it is unchanged; ``lam_tau = 0`` is S2c arm E, the
        # control that keeps L_tau computed and REPORTED but out of the optimized total (RETURNN
        # drops a scale-0 loss from the total, returnn/frontend/run_ctx.py:423). It is applied in
        # the train step, as ``lam_agg`` is, never inside the lattice.
        lam_tau: float = 1.0,
        # Self-distillation from the frozen init recognizer: the per-frame KL(q_init || q_theta),
        # applied in the train step. 0 = off, which is every stage up to S2b, so a config that does
        # not mention it is unchanged; S2c arm E sets 1. It needs the same frozen ``q_init`` copy
        # the anchor uses, and therefore the same ``recognizer_checkpoint_path``.
        lam_selfdistill: float = 0.0,
        # The LABEL-FREE rate term of SAE_4A.md "S3b-R" (rate_term), applied in the train step.
        # 0 = off, which is every stage up to S3 and is what the loop has always run, so a config
        # that does not mention these keys builds exactly the model it always did -- and the train
        # step marks neither the loss nor its two monitors at 0. ``rate_rho_hz`` has NO default: it
        # is the target phone rate in Hz, computed label-free (phones per word of the phonemized
        # text T_phi x 2.7 words/s), and the stage that trains on it must state it. The dev
        # gold phone rate (SAE_1f.md:533-536) is REPORTING-ONLY and is never this number.
        lam_rate: float = 0.0,
        rate_rho_hz: Optional[float] = None,
        # The finite difference the term's gradient is taken by (rate_term.DEFAULT_FD_EPS /
        # DEFAULT_FD_MODE): the step in nats per non-SIL token, and "central" (two tilted passes,
        # O(eps^2)) or "forward" (one, O(eps)). ``rate_fd_batched`` runs the two central tilts as
        # ONE stacked DP call where the lattice batch budget allows; False is the memory fallback.
        rate_fd_eps: float = 0.25,
        rate_fd_mode: str = "central",
        rate_fd_batched: bool = True,
        # The ONE-SIDED form of that term, max(0, rho - r)^2 (rate_term, "ONE-SIDED HINGE"): a
        # DISCLOSED DEVIATION from the user's two-sided form, so the term cannot push emissions DOWN
        # while the expectation is still above rho.  False = the stated form, which is what the
        # three S3b-R arms run, so a config that does not mention it is unchanged.
        rate_hinge: bool = False,
        # --- removed in the port (REMOVED_EMC_FLAGS): accepted at the source default only
        lam_ent: float = 0.0,
        ent_ramp_epochs: int = 4,
        lam_cons: float = 0.0,
        cons_views: Sequence[str] = ("specaug",),
        cons_specaug: Optional[Dict] = None,
        primary_specaug_until_epoch: int = 0,
        primary_specaug: Optional[Dict] = None,
        cons_pert_features_key: str = "features_pert",
        lam_content: float = 0.0,
        content_codes_key: str = "content_codes",
        lam_bt: float = 0.0,
        bt_depth: str = "full",
        bt_ramp_epochs: int = 4,
        bt_batch_sents: int = 128,
        bt_text_path: Optional[str] = None,
        # --- frozen inputs
        prior_npz_path: str,
        eta_table_path: Optional[str] = None,
        recognizer_checkpoint_path: Optional[str] = None,
        reverse_checkpoint_path: Optional[str] = None,
        # --- module shapes
        recognizer_kwargs: Optional[Dict] = None,
        reverse_kwargs: Optional[Dict] = None,
        # --- lattice topology (all traced; see the module docstring)
        band: int = 25,
        prior_weight: float = 1.0,
        prior_order: int = 2,
        # The history the prior conditions on, i.e. the order of the LM term inside L_tau.
        # "bigram" is what every stage up to S2c ran, so a config that does not mention these two
        # keys is unchanged -- and ``prior_order`` must agree with it.
        prior_history: str = "bigram",
        prior_classes: Optional[Union[str, Sequence[int]]] = None,
        # How the DP reduces a frame over h and over the band offset, and the stride of its
        # checkpointed forward table (lattice._use_matmul / lattice_forward_backward). The defaults
        # are the DP every stage up to S2c ran: "auto" is the elementwise reduction at the bigram
        # and the D3 matrix path above it, 0 keeps the whole [B, T, O, |h|, 2] forward table.
        lattice_reduction: str = "auto",
        lattice_checkpoint: int = 0,
        lattice_float64: bool = False,
        # --- removed in the port (REMOVED_EMC_FLAGS): accepted at the source default only
        candidate_prior_order: int = 0,
        candidate_path_draws: int = 0,
        candidate_counts_path: Optional[str] = None,
        candidate_source_sha256: Optional[Dict[str, str]] = None,
        anchor_weight_schedule: Schedule = 0.0,
        # --- L_agg
        agg_unigram_weight: float = 1.0,
        agg_bigram_weight: float = 1.0,
        # --- training modes
        freeze_recognizer: bool = False,  # phi warm-up: one sub-epoch of phi alone (SAE_4A.md:149)
        # the mirror of it (SAE_4A.md "S3b-OR"): phi is LOADED from a checkpoint and held, so the
        # arm trains the recognizer alone against a fixed reverse model
        freeze_reverse: bool = False,
        collect_stats: bool = True,
        # --- extern_data key names (INTERFACES.md §3)
        features_key: str = "features",
        units_key: str = "units",
        seq_tag_key: str = "seq_tag",
        eta_key: str = "eta",
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        device=None,  # RETURNN hands get_model the run's device; the model is moved by the engine
        **_returnn_fwd_compat,
    ):
        super().__init__()
        # STRICT config keys (review_lattice_d3d4_2026-09-15, F1). RETURNN's get_model call passes
        # epoch, step, device and one randomly named forward-compatibility sentinel
        # (returnn/torch/engine.py:1244-1245, returnn/util/basic.py:4552-4557) -- nothing else.
        # Anything else left over is a ``model_args`` key this model does not know, which used to be
        # swallowed in silence: a mistyped or unplumbed knob was a no-op that only showed up hours
        # later, as an OOM or as an arm that quietly ran the default.
        stray = sorted(k for k in _returnn_fwd_compat if not k.startswith("__fwd_compat"))
        if stray:
            raise TypeError(
                f"SaeEmcModelV1 got unknown model_args key(s) {stray}. A knob only reaches the "
                "model if this constructor names it; nothing is swallowed."
            )
        refuse_removed_flags(
            "SaeEmcModelV1",
            dict(
                lam_ent=lam_ent, ent_ramp_epochs=ent_ramp_epochs, lam_cons=lam_cons,
                cons_views=cons_views, cons_specaug=cons_specaug,
                primary_specaug_until_epoch=primary_specaug_until_epoch,
                primary_specaug=primary_specaug, cons_pert_features_key=cons_pert_features_key,
                lam_content=lam_content, content_codes_key=content_codes_key, lam_bt=lam_bt,
                bt_depth=bt_depth, bt_ramp_epochs=bt_ramp_epochs, bt_batch_sents=bt_batch_sents,
                bt_text_path=bt_text_path, candidate_prior_order=candidate_prior_order,
                candidate_path_draws=candidate_path_draws,
                candidate_counts_path=candidate_counts_path,
                candidate_source_sha256=candidate_source_sha256,
                lam_selfdistill=lam_selfdistill, rate_hinge=rate_hinge,
                prior_classes=prior_classes,
            ),
            REMOVED_EMC_FLAGS,
        )
        # (port) the three refusals whose accepted value is not a single literal default
        if prior_history == "class_trigram":
            raise ValueError(
                "SaeEmcModelV1: prior_history = 'class_trigram' was removed in the port (the "
                "blank-free model forces the full trigram); accepted: 'bigram' or 'trigram'"
            )
        if rate_fd_mode != "central":
            raise ValueError(
                f"SaeEmcModelV1: rate_fd_mode = {rate_fd_mode!r}; the one-pass 'forward' difference "
                "was removed in the port (every reference run used 'central')"
            )
        if self._anchor_can_be_nonzero(anchor_weight_schedule):
            raise ValueError(
                f"SaeEmcModelV1: anchor_weight_schedule = {anchor_weight_schedule!r}; a nonzero "
                "anchor was removed in the port (the train step never passed q_init to the "
                "lattice, so it had no effect); accepted: 0.0 at every sub-epoch"
            )
        assert prior_history in PRIOR_HISTORY_ORDER, (
            f"unknown prior_history {prior_history!r} (known: {sorted(PRIOR_HISTORY_ORDER)})"
        )
        assert prior_order == PRIOR_HISTORY_ORDER[prior_history], (
            f"prior_order = {prior_order} but the lattice history is {prior_history!r}, which is an "
            f"order-{PRIOR_HISTORY_ORDER[prior_history]} term; the two state the same thing and the "
            "table loaded below is the one the DP indexes"
        )
        from .recognizer import ConvRecognizer

        self.recognizer = ConvRecognizer(**dict(recognizer_kwargs or {}))
        rcfg = ReverseConfig(**dict(reverse_kwargs or {}))
        self.reverse = SegmentalReverseModel(rcfg)
        self.lattice_cfg = LatticeConfig(
            n_phones=rcfg.n_types, sil_id=rcfg.sil_id, band=int(band),
            d_min=rcfg.d_min, d_max=rcfg.d_max, d_max_sil=rcfg.d_max_sil,
        )
        # The history axis h of the DP and the prior table the DP indexes with it. The table is
        # NOT a second statement of the history: its rows ARE the h of ``prior_history``, and the
        # assertion below is the check that the two agree (it replaces the order-2 pin).
        # (port) no class partition: ``prior_classes`` is refused above.
        self.prior_history_name = str(prior_history)
        self.prior_history = build_prior_history(self.lattice_cfg, self.prior_history_name)
        # The DP's own two knobs, validated HERE so a wrong value is a job-start error and not a
        # crash inside the first train step. The legal reductions are stated once, by the lattice's
        # own resolver (it raises on anything outside auto / matmul / elementwise).
        self.lattice_reduction = str(lattice_reduction)
        self.lattice_checkpoint = int(lattice_checkpoint)
        self.lattice_float64 = bool(lattice_float64)
        _use_matmul(self.lattice_reduction, self.prior_history)
        assert self.lattice_checkpoint >= 0, (
            f"lattice_checkpoint = {self.lattice_checkpoint} is a frame stride S >= 0 (0 = keep the "
            "whole forward table)"
        )
        # D4 is not optional above the bigram: the un-checkpointed forward table is
        # [B, T, O, |h|, 2], i.e. 1.37 GiB at the bigram and ~56 GiB at the full trigram at the S2
        # shape B = 125, T = 704 (lattice.py, the D4 comment in lattice_forward_backward). Without
        # this the leg queues, allocates and dies (review_lattice_d3d4_2026-09-15, F1).
        if self.prior_history_name != "bigram" and self.lattice_checkpoint == 0:
            raise ValueError(
                f"prior_history = {self.prior_history_name!r} (|h| = {self.prior_history.n_hist}) "
                f"with lattice_checkpoint = 0 keeps the whole [B, T, O, |h|, 2] forward table "
                f"resident: ~56 GiB at the full trigram against 1.37 GiB at the bigram (B = 125, "
                f"T = 704). Set lattice_checkpoint = S > 0, the D4 frame stride."
            )
        if recognizer_checkpoint_path:
            _load_state(self.recognizer, recognizer_checkpoint_path, name="recognizer")
        if reverse_checkpoint_path:
            _load_state(self.reverse, reverse_checkpoint_path, name="reverse")

        prior = self._load_prior(prior_npz_path, rcfg.n_types)
        self.register_buffer(
            "prior_log_bi",
            torch.as_tensor(
                self._prior_table(prior, self.prior_history_name),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        assert tuple(self.prior_log_bi.shape) == (self.prior_history.n_hist, rcfg.n_types), (
            f"the {self.prior_history_name} prior table is {tuple(self.prior_log_bi.shape)}, the "
            f"lattice history wants ({self.prior_history.n_hist}, {rcfg.n_types})"
        )
        text_uni, text_bi = text_target_counts(prior, n_phones=rcfg.n_types)
        self.agg_cfg = AggConfig(
            count_ema_decay=float(count_ema_decay), n_phones=rcfg.n_types,
            unigram_weight=float(agg_unigram_weight), bigram_weight=float(agg_bigram_weight),
        )
        self.agg = AggLoss(self.agg_cfg, text_uni, text_bi)

        # eta: a frozen lookup by seq tag (INTERFACES.md §3). persistent=False -- it is an INPUT,
        # rebuilt from its path on every construction, and has no business growing the checkpoint.
        self.eta_tags: Dict[str, int] = {}
        eta = torch.zeros(1, rcfg.eta_dim)
        if eta_table_path:
            eta, self.eta_tags = self._load_eta(eta_table_path, rcfg.eta_dim)
        self.register_buffer("eta_table", eta, persistent=False)

        self.temperature_schedule = temperature_schedule
        self.anchor_weight_schedule = anchor_weight_schedule
        self.lam_agg = float(lam_agg)
        self.lam_tau = float(lam_tau)
        self.lam_rate = float(lam_rate)
        self.rate_rho_hz = None if rate_rho_hz is None else float(rate_rho_hz)
        self.rate_fd_eps = float(rate_fd_eps)
        self.rate_fd_mode = str(rate_fd_mode)  # (port) always "central"; anything else refused above
        self.rate_fd_batched = bool(rate_fd_batched)
        if self.lam_rate:
            # Refused HERE, at job start, rather than at the first step of a queued run.
            if not self.rate_rho_hz or self.rate_rho_hz <= 0.0:
                raise ValueError(
                    "lam_rate != 0 needs rate_rho_hz > 0, the LABEL-FREE target phone rate in Hz"
                )
            if self.rate_fd_eps <= 0.0:
                raise ValueError("rate_fd_eps is the finite-difference step in b, in nats, > 0")
        self.prior_weight = float(prior_weight)
        self.collect_stats = bool(collect_stats)
        self.features_key = features_key
        self.units_key = units_key
        self.seq_tag_key = seq_tag_key
        self.eta_key = eta_key

        # phi warm-up (SAE_4A.md:149: "one sub-epoch of phi on the init recognizer's tempered
        # posterior, theta frozen"). Frozen here AND detached in the train step: the first keeps
        # theta out of the optimizer and out of DDP's reduction, the second makes the severed path
        # explicit at the loss.
        self.freeze_recognizer = bool(freeze_recognizer)
        if self.freeze_recognizer:
            self.recognizer.requires_grad_(False)

        # The mirror image (SAE_4A.md "S3b-OR"): phi is loaded from ``reverse_checkpoint_path``
        # above and HELD, so the arm trains theta alone against a fixed reverse model. Set AFTER
        # the checkpoint load, so the frozen weights are the loaded ones; ``emc_param_groups``
        # drops every parameter with requires_grad = False, which leaves the phi group absent.
        # No detach is needed in the train step: with phi's parameters frozen the segment table it
        # builds carries no gradient path at all (the units and eta inputs are data).
        self.freeze_reverse = bool(freeze_reverse)
        if self.freeze_reverse:
            self.reverse.requires_grad_(False)

        # (port) the source built the anchor copy ``q_init`` here when the anchor weight could be
        # nonzero or ``lam_selfdistill != 0``; both are refused above, so it is never built.
        self._last_step_time: Optional[float] = None

    # -- construction helpers -------------------------------------------------------------------

    @staticmethod
    def _anchor_can_be_nonzero(schedule: Schedule) -> bool:
        if isinstance(schedule, (int, float)):
            return float(schedule) != 0.0
        return any(float(v) != 0.0 for v in schedule)

    @staticmethod
    def _prior_table(prior, history: str):
        """The ``[|h|, K]`` table the DP indexes with ``h`` -- all three from the SAME banked fit.

        ``bigram`` is ``log_bi`` (the table every stage up to S2c ran), ``trigram`` the banked
        ``log_tri`` (rows ``h2 * N_CTX + h1``, which is the lattice's own ``outer * n_ctx + last``),
        and ``class_trigram`` the marginalisation of that fit's trigram COUNTS over the outer symbol
        inside each class, re-smoothed against the same ``log_bi``
        (``prior.class_trigram_log_table``). No new LM fit and no new upstream job for any of them
        (estimate 2026-09-15, D1).  (port) The class-trigram branch is cut.
        """
        if history == "bigram":
            return prior.log_bi
        if history == "trigram":
            return prior.log_tri
        raise ValueError(f"prior history {history!r}: only 'bigram' and 'trigram' are ported")

    @staticmethod
    def _load_prior(path: str, n_types: int):
        from .prior import PhoneNgramPrior

        prior = PhoneNgramPrior.load(path)
        assert prior.log_bi.shape == (n_types + 1, n_types), (
            f"the stored prior is {prior.log_bi.shape}, the lattice wants ({n_types + 1}, {n_types})"
        )
        return prior

    @staticmethod
    def _load_eta(path: str, eta_dim: int):
        import numpy as np

        d = np.load(path, allow_pickle=False)
        eta = torch.as_tensor(np.asarray(d["eta"], dtype=np.float32))
        tags = [str(t) for t in d["tags"]]
        assert eta.shape == (len(tags), eta_dim), (eta.shape, len(tags), eta_dim)
        return eta, {t: i for i, t in enumerate(tags)}

    # -- runtime ----------------------------------------------------------------------------------

    def lookup_eta(self, tags: List[str], device) -> torch.Tensor:
        """``[B, eta_dim]`` frozen speaker vectors. A missing tag is an error, never a zero vector."""
        if not self.eta_tags:
            raise ValueError("no eta table was loaded; pass eta_table_path or the 'eta' extern_data key")
        try:
            idx = torch.tensor([self.eta_tags[t] for t in tags], dtype=torch.long)
        except KeyError as e:
            raise KeyError(f"seq tag {e} is not in the frozen eta table") from None
        return self.eta_table.index_select(0, idx.to(self.eta_table.device)).to(device)

    def temperature(self, epoch: int) -> float:
        return schedule_value(self.temperature_schedule, epoch)

    def anchor_weight(self, epoch: int) -> float:
        return schedule_value(self.anchor_weight_schedule, epoch)
