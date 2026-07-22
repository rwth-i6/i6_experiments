"""
Optimizer helpers serialized into RETURNN configs, reproducing the fairseq wav2vec-U *composite*
optimizer with a single RETURNN optimizer.

fairseq computes the loss for the active step (generator or discriminator), backpropagates so that
**both** param groups receive gradients (the loss flows through both), clips the **global** grad norm
over all params, and then steps **only the active group** (``CompositeOptimizer.step(groups=...)``).

We reproduce this exactly:

* ``wav2vec_u_param_groups`` splits the parameters into a ``generator`` and a ``discriminator`` group
  (by the ``.param_group`` tag the model sets), each with its own ``weight_decay`` and per-group
  learning-rate multiplier, and tags each group with ``param_group_name``.
* The wav2vec-U train step does **not** freeze the inactive group, so both groups get gradients and
  RETURNN's global grad-norm clip sees both (matching fairseq's clip scope). The train step records
  which group is active for the current update via :func:`set_active_param_group`.
* :class:`GanAlternatingAdamW` runs as RETURNN's optimizer. RETURNN clips *before* calling
  ``optimizer.step()``; inside ``step()`` we drop the inactive group's gradients (set to ``None``) so
  AdamW updates only the active group (no momentum / weight-decay drift on the frozen group).

Referenced from the config via ``optimizer = {"class": CodeWrapper("GanAlternatingAdamW"),
"param_groups_custom": CodeWrapper("wav2vec_u_param_groups"), ...}`` plus a ``PartialImport`` that
binds the per-group weight decays / lr multipliers.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------------------------
# shared state: the train step tells the optimizer which group to step this update
# ---------------------------------------------------------------------------------------------

# process-global; the train step and optimizer.step() run sequentially in the same process (single
# GPU, or one such pair per rank under DDP), so this stays in sync and is resume-safe (the train step
# derives the active group from the true global step).
_GAN_ACTIVE_GROUP = {"name": "generator"}


def set_active_param_group(name: str):
    """Set the param group to be stepped on the next optimizer update ('generator'|'discriminator')."""
    assert name in ("generator", "discriminator"), name
    _GAN_ACTIVE_GROUP["name"] = name


def get_active_param_group() -> str:
    return _GAN_ACTIVE_GROUP["name"]


class GanAlternatingAdamW(torch.optim.AdamW):
    """AdamW that steps only the currently active param group (fairseq composite-optimizer behavior).

    RETURNN clips the global grad norm over all params *before* this ``step()`` (so the clip sees both
    groups, like fairseq). Here we drop the inactive group's gradients so AdamW updates only the active
    group. Groups must be tagged with ``param_group_name`` (see ``wav2vec_u_param_groups``).
    """

    def step(self, closure=None):
        active = _GAN_ACTIVE_GROUP["name"]
        for group in self.param_groups:
            name = group.get("param_group_name", None)
            if name is not None and name != active:
                for p in group["params"]:
                    p.grad = None
        return super().step(closure)


def wav2vec_u_param_groups(
    *,
    model,
    generator_weight_decay: float = 0.0,
    generator_lr_multiplier: float = 1.0,
    discriminator_weight_decay: float = 1e-4,
    discriminator_lr_multiplier: float = 1.0,
    **_kwargs,
):
    """
    :param model: the (PyTorch) network, passed by RETURNN's updater.
    :return: list of param-group dicts (``params`` + ``weight_decay`` + ``learning_rate_multiplier``
        + ``param_group_name``).
    """
    gen_params = []
    disc_params = []
    for name, p in model.named_parameters():
        group = getattr(p, "param_group", None)
        if group is None:
            # fall back to the module path (robust if the tag did not survive wrapping)
            group = "discriminator" if "discriminator" in name else "generator"
        if group == "discriminator":
            disc_params.append(p)
        else:
            gen_params.append(p)

    groups = []
    if gen_params:
        groups.append(
            {
                "params": gen_params,
                "weight_decay": generator_weight_decay,
                "learning_rate_multiplier": generator_lr_multiplier,
                "param_group_name": "generator",
            }
        )
    if disc_params:
        groups.append(
            {
                "params": disc_params,
                "weight_decay": discriminator_weight_decay,
                "learning_rate_multiplier": discriminator_lr_multiplier,
                "param_group_name": "discriminator",
            }
        )
    return groups
