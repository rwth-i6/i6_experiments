"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/emc_train_jobs.py (``emc_param_groups``,
lines 342-386, verbatim).

The optimizer ``param_groups_custom`` of every EMC / blank-free leg: one group for theta, one for
phi, each with its own learning-rate multiplier.
"""

__all__ = ["emc_param_groups"]


def emc_param_groups(
    *,
    model,
    recognizer_lr_multiplier: float = 1.0,
    reverse_lr_multiplier: float = 1.0,
    **_kwargs,
):
    """``optimizer.param_groups_custom``: one group for theta, one for phi.

    RETURNN calls this as ``param_groups_custom(model=..., rf_model=..., optimizer_class=...,
    optimizer_opts=..., **fwd_compat)`` (``returnn/torch/updater.py:531-544``) and multiplies each
    group's base learning rate by its ``learning_rate_multiplier``
    (``returnn/torch/updater.py:98, :198``).  Frozen parameters (``q_init``, the recognizer during
    the phi warm-up, and the reverse model under ``freeze_reverse``) carry ``requires_grad = False``
    and are dropped here, so the optimizer never holds state for a parameter that cannot move -- a
    frozen side leaves its group ABSENT, which is what the remaining assert allows.
    """
    theta, phi, other = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        short = name[len("module.") :] if name.startswith("module.") else name  # DDP wrapping
        if short.startswith("recognizer."):
            theta.append(param)
        elif short.startswith("reverse."):
            phi.append(param)
        else:
            other.append(short)
    assert not other, (
        f"trainable parameters outside recognizer/reverse: {sorted(other)[:8]}. The EMC model's "
        "prior, eta and expected-count tables are BUFFERS by construction; a parameter here means "
        "the container changed and the two learning rates no longer describe the run."
    )
    groups = []
    if theta:
        groups.append({"params": theta, "learning_rate_multiplier": float(recognizer_lr_multiplier)})
    if phi:
        groups.append({"params": phi, "learning_rate_multiplier": float(reverse_lr_multiplier)})
    assert groups, "no trainable parameter at all; the leg would be a no-op"
    print(
        f"emc_param_groups: theta {len(theta)} tensors x{recognizer_lr_multiplier:g}, "
        f"phi {len(phi)} tensors x{reverse_lr_multiplier:g}",
        flush=True,
    )
    return groups
