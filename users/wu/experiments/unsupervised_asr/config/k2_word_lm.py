"""Entry point: the k2 word-LM arms, with the same reads as ``config.base`` plus paired contrasts
against the control.

* ``k2_word_lm`` -- the port's DEFAULT k2 arm (PORT_WORK design decision 2): the official LibriSpeech
  4-gram HLG (theta 5.0; escape resources from ``LexlatOfficialResourcesJob``), max_active 1000,
  ``phone_trigram="rampout"`` (D15's schedule: full weight, then ramped to 0 over the word graph's
  on-set).  This combination was NEVER RUN as such; it has no banked number.
  ``py(phone_trigram="full")`` gives ``off4_k2lat_20``'s treatment (the same config, so sisyphus
  merges the two into one job); ``"off"`` trains with no LM term before the on-set.
* Presets reproducing banked arms (``phone_trigram="full"``):
    - ``k2lat_20_ma3000``: in-house word trigram HLG, max_active 3000; banked dev-other PER 0.818615
      at sub-epoch 20;
    - ``k2lat_20_ma3000_x60``: the same as one 60-sub-epoch job; banked 0.823991 at 60;
    - ``off4_k2lat_20``: the official 4-gram HLG, max_active 1000.

Reads: every arm as ``config.common.train_and_read``; at the final checkpoint the JS rows and the
paired PER delta of each k2 arm against the control of the same length (``ctrl_20`` /
``ctrl_20_x60``, built by ``config.base``; the k2 pack's registered ``(treatment, ctrl)`` pair).
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["DEFAULT_PHONE_TRIGRAM", "k2_arms", "py"]

#: the default arm's phone-trigram mode (``training.schedules.PHONE_TRIGRAM_MODES``)
DEFAULT_PHONE_TRIGRAM = "rampout"


def k2_arms(inputs=None, *, phone_trigram: str = DEFAULT_PHONE_TRIGRAM) -> Dict[str, Dict[str, Any]]:
    """``{arm name: train_and_read(...)}`` for the default arm and the three presets.

    :param phone_trigram: the DEFAULT arm's mode only; the presets always run ``"full"`` (banked)."""
    from ..inputs import get_graph, get_inputs
    from ..training.arms import k2_word_lm, k2lat_20_ma3000, off4_k2lat_20
    from .common import train_and_read

    inputs = get_inputs() if inputs is None else inputs
    official, inhouse = get_graph("official_4gram"), get_graph("inhouse_3gram")
    arms = [
        k2_word_lm(data=inputs.data, graph=official, phone_trigram=phone_trigram),
        k2lat_20_ma3000(data=inputs.data, graph=inhouse),
        k2lat_20_ma3000(data=inputs.data, graph=inhouse, num_sub_epochs=60),
        off4_k2lat_20(data=inputs.data, graph=official),
    ]
    return {arm.name: train_and_read(arm, inputs) for arm in arms}


def py(phone_trigram: str = DEFAULT_PHONE_TRIGRAM) -> Dict[str, Any]:
    from ..inputs import get_inputs
    from .base import ctrl_arms, register_inputs
    from .common import js_rows, paired_delta

    inputs = get_inputs()
    register_inputs(inputs)
    ctrl = ctrl_arms(inputs)
    k2 = k2_arms(inputs, phone_trigram=phone_trigram)
    counterpart = {name: ("ctrl_20_x60" if r["arm"].num_epochs == 60 else "ctrl_20") for name, r in k2.items()}
    paired = {name: paired_delta(k2[name], ctrl[base], inputs) for name, base in counterpart.items()}
    rows = [{"tag": f"{name}_vs_{base}", "cand": name, "base": base} for name, base in counterpart.items()]
    js = js_rows(f"k2_word_lm_{phone_trigram}", {**ctrl, **k2}, inputs, rows)
    return {"inputs": inputs, "ctrl": ctrl, "arms": k2, "paired": paired, "js_rows": js}
