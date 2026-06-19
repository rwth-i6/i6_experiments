"""Per-parameter-group LR multiplier for RETURNN-torch (BEST-RQ-style two-rate finetune).

Used as ``param_groups_custom`` in the optimizer options, together with a
``learning_rate_multipliers_by_patterns`` dict (``{fnmatch-pattern: multiplier}``). Each trainable
parameter whose FULL name (``module.param``) matches a pattern gets that multiplier on the scheduled
LR -- RETURNN multiplies the (OCLR) learning rate by the group's ``learning_rate_multiplier`` every
step (see returnn/torch/updater.py: ``_update_effective_learning_rate``); unmatched params stay at 1x.

This realizes the two-rate finetune (BEST-RQ: fresh decoder ~3x the pretrained encoder LR): set the
GLOBAL peak_lr to the low backbone rate and lift only the fresh head group via the multiplier.

Frozen params (``requires_grad is False``) are dropped (they are not optimized anyway). Weight decay is
left to the optimizer-level default, so it is byte-identical to the single-group setup -- this changes
ONLY the per-group LR, nothing else. Self-contained (stdlib ``fnmatch`` only) so make_local_package_copy
snapshots it cleanly; RETURNN calls it as ``f(model=, rf_model=, optimizer_class=, optimizer_opts=, ...)``.
"""

from fnmatch import fnmatchcase


def optimizer_param_groups_custom_lr_multiplier(*, model, optimizer_opts, **_kwargs):
    patterns = optimizer_opts.pop("learning_rate_multipliers_by_patterns")
    by_mult = {}  # multiplier -> [params]; insertion order preserved
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = [m for pat, m in patterns.items() if fnmatchcase(name, pat)]
        assert len(matched) <= 1, f"param {name!r} matches >1 LR-multiplier pattern: {matched}"
        mult = matched[0] if matched else 1.0
        by_mult.setdefault(mult, []).append(param)
    groups = []
    for mult, params in by_mult.items():
        group = {"params": params}
        if mult != 1.0:  # absent => RETURNN treats as 1.0 (one extra-opts entry per group still created)
            group["learning_rate_multiplier"] = mult
        groups.append(group)
    return groups
