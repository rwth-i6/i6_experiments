"""
Generate ``exp2025_10_21_chunked_ctc.md``, the results summary next to the recipe,
and the scaling-extrapolation figure it embeds.

Every number is read from the setup's ``output/`` tree at call time,
so no WER in the doc is ever hand-typed.
The prose lives here in :data:`_TEMPLATE`:
why an experiment was run is not on disk and cannot be generated,
while every number inside that prose can be.

Placeholders are ``{{<recog>:<variant>[:<tag>]:<field>}}`` for measured cells,
e.g. ``{{ctc:base:dev_test}}`` -> ``7.32 / 8.10``,
and ``{{fit:<split>:<key>}}`` for values derived from the scaling fit.
An unresolvable cell renders as ``--`` instead of raising,
so a doc can be regenerated while some recogs are still pending.

Reads ``output/`` rather than rebuilding the Sisyphus graph:
the alias tree is what the manager already resolved,
and this keeps the generator a plain script with no sisyphus import or settings bootstrap.

The setup dir cannot be derived from this file's location:
``recipe`` is a symlink into a checkout shared by many setups,
so the setup is taken from the cwd, or from ``--setup-dir``.

Needs numpy and matplotlib for the fit and the figure,
so run it with the py-env interpreter, from the setup dir::

    /home/az668407/work/py-envs/py3.12-torch2.7/bin/python \\
        recipe/i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc_summary.py
    ... exp2025_10_21_chunked_ctc_summary.py --check   # diff only, write nothing
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["render", "main"]

_my_dir = os.path.dirname(os.path.realpath(__file__))
_doc_path = f"{_my_dir}/exp2025_10_21_chunked_ctc.md"
_out_rel = "output/exp2025_10_21_chunked_ctc/aed"

MISSING = "--"
PLOT_NAME = "exp2025_10_21_chunked_ctc.scaling.png"

_DYN = "chunked-L80-C5-R4-v2.3-dyn-rope-ctembed"

# Where each recog kind writes its result file, relative to the variant's output dir.
# "ctc+lm" has a single LM subdir whose name encodes the LM, so it is globbed.
_RECOG_PATHS = {
    "ctc": "aed+ctc/ctc-only-res.txt",
    "aedctc": "aed+ctc/recog-1stpass-res.txt",
    "ctclm": "ctc+lm-v2/*/recog-1stpass-res.txt",
    "sweep": "ctc-recog-sweep/{tag}",
    # One epoch of the per-epoch WER curve, for runs whose last epoch is not the one to quote.
    "ep": "recog_results_per_epoch/{tag}",
    # Any other registered output under the variant dir, named by the tag.
    "out": "{tag}",
    # Plain-scalar outputs, exposed as the field "value".
    "hours": "train_time_hours",
}

# The three scaling curves, at 1x / 2x / 4x.
# "dyn offline" is the streaming checkpoint decoded at chunk_size=None,
# which separates the cost of training under a chunk pool
# from the cost of decoding in chunks.
SCALES = [1, 2, 4]
_CURVES = {
    "base offline": [("ctc", "base", None), ("ctc", "base-2xtrain", None), ("ctc", "base-4xtrain", None)],
    "dyn offline": [
        ("sweep", _DYN, "offline"),
        ("sweep", f"{_DYN}-2xtrain", "offline"),
        ("sweep", f"{_DYN}-4xtrain", "offline"),
    ],
    "dyn online": [("ctc", _DYN, None), ("ctc", f"{_DYN}-2xtrain", None), ("ctc", f"{_DYN}-4xtrain", None)],
}
_CURVE_COLORS = {"base offline": "#1f77b4", "dyn offline": "#2ca02c", "dyn online": "#ff7f0e"}
_CURVE_KEYS = {"base offline": "base", "dyn offline": "dynoff", "dyn online": "dynon"}


def _find_out_dir(setup_dir: Optional[str] = None) -> str:
    """Resolve the setup's output dir, and fail loudly if it is not there.

    Silently rendering every cell as missing is the failure mode to avoid,
    since it looks like "results pending" rather than "wrong directory".
    """
    setup_dir = setup_dir or os.getcwd()
    out_dir = f"{setup_dir}/{_out_rel}"
    if not os.path.isdir(out_dir):
        raise SystemExit(f"no {_out_rel} under {setup_dir!r}; run from the setup dir or pass --setup-dir")
    return out_dir


def _read_res(out_dir: str, variant: str, recog: str, tag: Optional[str] = None) -> Dict[str, Any]:
    """Result dict for one (variant, recog), or empty if not on disk yet."""
    rel = _RECOG_PATHS[recog]
    base = f"{out_dir}/{variant}"
    if "{tag}" in rel:
        assert tag, f"recog {recog!r} needs a tag"
        path = f"{base}/{rel.format(tag=tag)}"
    elif "*" in rel:
        head, _, tail = rel.partition("*")
        d = f"{base}/{head}".rstrip("/")
        if not os.path.isdir(d):
            return {}
        subs = sorted(os.listdir(d))
        if len(subs) != 1:
            # More than one LM variant would make "the" ctc+lm number ambiguous.
            return {}
        path = f"{d}/{subs[0]}/{tail.lstrip('/')}"
    else:
        path = f"{base}/{rel}"
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return {}
    # Some outputs are a bare scalar (train_time_hours), which is also valid JSON,
    # so normalize anything non-dict to {"value": ...}.
    try:
        v = json.loads(text)
    except ValueError:
        try:
            v = float(text)
        except ValueError:
            return {}
    return v if isinstance(v, dict) else {"value": v}


def _get(res: Dict[str, Any], field: str) -> Any:
    """Look up a field, flat key first, then as a dotted path.

    Flat first because some result keys contain a dot themselves
    (``tedlium-seg.test``), while others nest (``summary.max_of_max_abs_diff``).
    """
    if field in res:
        return res[field]
    cur: Any = res
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt(res: Dict[str, Any], field: str) -> str:
    """Render one field. ``dev_test`` is the paired form used in most tables.

    Two decimals by default, so a WER that happens to be a round number
    (JSON 8.1) still lines up with the rest of the column as 8.10.
    A trailing ``|<spec>`` overrides that, for values that are not WERs.
    """
    field, _, spec = field.partition("|")
    spec = spec or ".2f"
    if field == "dev_test":
        dev, test = res.get("dev"), res.get("test")
        if dev is None or test is None:
            return MISSING
        return f"{dev:{spec}} / {test:{spec}}"
    v = _get(res, field)
    return MISSING if v is None else f"{v:{spec}}"


# ---------------------------------------------------------------- scaling fit


def _curve_values(out_dir: str, split: str) -> Optional[Dict[str, List[float]]]:
    """The three scaling curves for one split, or None if any of the 9 cells is missing."""
    out = {}
    for name, cells in _CURVES.items():
        vals = []
        for recog, variant, tag in cells:
            v = _read_res(out_dir, variant, recog, tag).get(split)
            if v is None:
                return None
            vals.append(float(v))
        out[name] = vals
    return out


def _fit_joint(curves: Dict[str, List[float]], c_grid) -> Tuple[float, Dict[str, Tuple[float, float]], float]:
    """Fit W(s) = E + b * s**-c with one exponent c shared by all curves.

    Each curve alone is 3 parameters on 3 points, so its floor is unidentifiable
    and the fitted floors come out unordered (the streaming floor sinks below its
    own offline floor, which inference cannot do). Sharing c is the better-conditioned
    fit rather than a richer one: 7 parameters on 9 points, and the ordering holds.
    Given c the model is linear in (E, b), so c is profiled on a grid and the rest solved in closed form.
    """
    import numpy as np

    s = np.array(SCALES, dtype=float)
    basis = s[None, :] ** -c_grid[:, None]
    basis_m = basis - basis.mean(axis=1, keepdims=True)
    basis_var = (basis_m**2).sum(axis=1)

    per, total = {}, 0.0
    for name, vals in curves.items():
        w = np.array(vals)
        b = (basis_m @ (w - w.mean())) / basis_var
        e = w.mean() - b * basis.mean(axis=1)
        rss = ((w[None, :] - (e[:, None] + b[:, None] * basis)) ** 2).sum(axis=1)
        per[name] = (e, b, rss)
        total = total + rss
    i = int(total.argmin())
    return float(c_grid[i]), {k: (float(v[0][i]), float(v[1][i])) for k, v in per.items()}, float(total[i])


def _fit_context(out_dir: str) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    """Placeholder values derived from the fit, plus what the plot needs."""
    try:
        import numpy as np
    except ImportError:
        raise SystemExit("numpy is required; run with the py-env interpreter (see the module docstring)")

    c_grid = np.linspace(0.05, 3.0, 4000)
    ctx: Dict[str, str] = {}
    state: Dict[str, Any] = {}
    for split in ("dev", "test"):
        curves = _curve_values(out_dir, split)
        if curves is None:
            continue
        c, pars, rss = _fit_joint(curves, c_grid)
        state[split] = {"curves": curves, "c": c, "pars": pars}
        ctx[f"fit:{split}:c"] = f"{c:.2f}"
        ctx[f"fit:{split}:rms"] = f"{math.sqrt(rss / 9):.3f}"
        for name, key in _CURVE_KEYS.items():
            ctx[f"fit:{split}:E_{key}"] = f"{pars[name][0]:.2f}"
    # Paired dev / test forms, for the "inf" row of the scale table.
    if "dev" in state and "test" in state:
        for name, key in _CURVE_KEYS.items():
            ctx[f"fit:devtest:E_{key}"] = f"{state['dev']['pars'][name][0]:.2f} / {state['test']['pars'][name][0]:.2f}"
    return ctx, state or None


def _write_plot(path: str, state: Dict[str, Any]) -> None:
    """Two panels, dev and test: measured points, the joint fit, and its floors."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sx = np.logspace(0, math.log10(64), 300)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.0))
    for ax, split in zip(axes, ("dev", "test")):
        if split not in state:
            continue
        st = state[split]
        for name, vals in st["curves"].items():
            e, b = st["pars"][name]
            col = _CURVE_COLORS[name]
            ax.plot(SCALES, vals, "o", color=col, ms=7, zorder=5, label=name)
            ax.plot(sx, e + b * sx ** -st["c"], "-", color=col, lw=1.8)
            ax.axhline(e, color=col, ls=":", lw=1.0)
            ax.text(64, e, f"{e:.2f}", color=col, fontsize=9, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["1x", "2x", "4x", "8", "16", "32", "64"])
        ax.set_xlabel("training scale (x epochs)")
        ax.set_ylabel(f"CTC-only WER [%] ({split})")
        ax.set_title(f"{split}: joint fit, shared exponent c={st['c']:.2f}", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8.5)
    fig.suptitle("WER vs training scale, power-law fit with one shared exponent; dotted = extrapolated floor")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- rendering

_PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")


def _resolve(out_dir: str, ctx: Dict[str, str], key: str) -> str:
    if key.startswith("fit:"):
        return ctx.get(key, MISSING)
    parts = key.split(":")
    if len(parts) == 3:
        recog, variant, field = parts
        tag = None
    elif len(parts) == 4:
        recog, variant, tag, field = parts
    else:
        raise ValueError(f"bad placeholder {key!r}")
    return _fmt(_read_res(out_dir, variant, recog, tag), field)


def render(setup_dir: Optional[str] = None, plot_path: Optional[str] = None) -> str:
    """The full doc, with every placeholder resolved against current disk state.

    Writes the figure too when ``plot_path`` is given.
    """
    out_dir = _find_out_dir(setup_dir)
    ctx, state = _fit_context(out_dir)
    if plot_path and state:
        _write_plot(plot_path, state)
    return _PLACEHOLDER.sub(lambda m: _resolve(out_dir, ctx, m.group(1).strip()), _TEMPLATE)


# language=Markdown
_TEMPLATE = (
    """\
# Chunked CTC / Conformer streaming ASR: results

Code: `i6_experiments/users/zeyer/experiments/exp2025_10_21_chunked_ctc.py`.
This file is generated by `exp2025_10_21_chunked_ctc_summary.py`; edit the template there, not this file.

## Overview / setup

Study: how close chunked streaming attention gets to the offline model.

Task: Loquacious, `large` subset (~25 kh).
1x training = `total_k_hours=100` = 4 full epochs (100 subepochs, split 25).

Model: Conformer encoder + AED (Transformer) decoder, with CTC aux heads.
- encoder: 16 layers, dim 1024, 8 heads, relu-square FF (no bias),
  `ConformerConvSubsample` /6 downsampling (`out_dims=[32,64,64]`).
- decoder: `TransformerDecoder`, 6 layers, dim 1024, RMSNorm, `FeedForwardGated`, rotary causal self-att.
- CTC aux: encoder layers `[4,10,16]`, decoder layer `[3]`; `feature_batch_norm`.

Train: bf16, batch 100k, weight_decay 1e-2, lrlin OCLR base_lr 0.5, max input 19.5 s.
Vocab: spm10k (sampling BPE, breadth_prob 0.01).

Recog:
- CTC-only (`ctc_model_recog`) = headline.
- AED+CTC time-sync (`aed_ctc_timesync_recog_recomb_auto_scale`).
- CTC+LM label-sync + prior (`ctc_recog_recomb_labelwise_prior_auto_scale`),
  LM = Trafo n32-d1024 spm10k.
Encoders: `nn_rf/encoder/chunked_conformer_v1`, `nn_rf/encoder/chunked_conformer_v2.py`.

Eval: Loquacious dev / test, aggregate + per-domain (voxpopuli, commonvoice, librispeech, yodas).
Metric below: CTC-only WER [%], dev / test aggregate, last epoch.

## Baselines

Offline (full-context) reference:

| model | dev | test |
| --- | --- | --- |
| base (offline conformer) | {{ctc:base:dev}} | {{ctc:base:test}} |

Vocab size (chunked L80-C5-R4, v1; CTC-only dev):

| vocab | dev |
| --- | --- |
| spm1k | {{ctc:chunked-L80-C5-R4-spm1k:dev}} |
| spm5k | {{ctc:chunked-L80-C5-R4-spm5k:dev}} |
| spm10k | {{ctc:chunked-L80-C5-R4-spm10k:dev}} |

Default: spm10k.

## Comparisons

Each section states the question it answers and lists the runs that answer it.
All numbers are CTC-only WER, dev / test, last epoch.
`h` is `train_time_hours`.

### Chunk geometry: history, center, lookahead

How much left history, center chunk and right lookahead does a chunked model need?
The early sweep used the v1 encoder; most of these runs are retired, the numbers stay here.

History, at C20-R15:

| history | dev / test |
| --- | --- |
| L0 | {{ctc:chunked-L0-C20-R15:dev_test}} |
| L20 | {{ctc:chunked-L20-C20-R15:dev_test}} |
| L40 | {{ctc:chunked-L40-C20-R15:dev_test}} |
| L60 | {{ctc:chunked-L60-C20-R15:dev_test}} |
| L80 | {{ctc:chunked-L80-C20-R15:dev_test}} |
| L160 | {{ctc:chunked-L160-C20-R15:dev_test}} |

History matters most from 0 to 40 and then plateaus.
Repeated at the tight C5-R4 geometry with the v2.3 encoder, where the L0 case is far more extreme:
L0 {{ctc:chunked-L0-C5-R4-v2.3:dev_test}},
L40 {{ctc:chunked-L40-C5-R4-v2.3:dev_test}},
L80 {{ctc:chunked-L80-C5-R4-v2.3:dev_test}}.

Center and lookahead, at L40:

| geometry | dev / test |
| --- | --- |
| C5-R30 | {{ctc:chunked-L40-C5-R30:dev_test}} |
| C10-R25 | {{ctc:chunked-L40-C10-R25:dev_test}} |
| C10-R30 | {{ctc:chunked-L40-C10-R30:dev_test}} |
| C20-R20 | {{ctc:chunked-L40-C20-R20:dev_test}} |
| C40-R15 | {{ctc:chunked-L40-C40-R15:dev_test}} |
| C40-R0 | {{ctc:chunked-L40-C40-R0:dev_test}} |

Dropping the lookahead (C40-R0) is the one clear loss, and history does not buy it back:
at C40-R0, L40 {{ctc:chunked-L40-C40-R0:dev}},
L80 {{ctc:chunked-L80-C40-R0:dev}},
L120 {{ctc:chunked-L120-C40-R0:dev}}.
That is the first sign that lookahead, not history, is what the model leans on.

Extremes: a tiny chunk is bad ({{ctc:chunked-L80-C2-R3:dev_test}} at C2-R3),
and a very loose chunk approaches offline
({{ctc:chunked-L100-C100-R15:dev_test}} at C100-R15) but at useless latency.
Middle point for reference: C10-R8 {{ctc:chunked-L80-C10-R8:dev_test}}.

### Implementation version and cost knobs

Not a WER question but a correctness and throughput one:
does the rewritten chunked encoder reproduce v1, and what do the options cost?
v1 and v2.2 used a wrong chunking implementation, fixed from `version=3`.

| variant | dev / test | h |
| --- | --- | --- |
| v1 | {{ctc:chunked-L80-C5-R4:dev_test}} | {{hours:chunked-L80-C5-R4:value|.1f}} |
| v2.1 (bugged) | {{ctc:chunked-L80-C5-R4-v2:dev_test}} | {{hours:chunked-L80-C5-R4-v2:value|.1f}} |
| v2.2 (bugged) | {{ctc:chunked-L80-C5-R4-v2.2:dev_test}} | {{hours:chunked-L80-C5-R4-v2.2:value|.1f}} |
| v2.3 | {{ctc:chunked-L80-C5-R4-v2.3:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3:value|.1f}} |
| v2.3, no short-seq adapt | {{ctc:chunked-L80-C5-R4-v2.3-compat:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-compat:value|.1f}} |
| v2.3 + grad checkpointing | {{ctc:chunked-L80-C5-R4-v2.3-gdckpt:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-gdckpt:value|.1f}} |

v2.3 matches v1 while training faster.
The short-sequence adaptation is WER-neutral.
Grad checkpointing is WER-neutral too and trades time for memory.

### Positional encoding

Relative position vs RoPE vs learnable relative position,
run in both the fixed and the dynamic setting, with an offline control
to separate "helps under chunking" from "helps in general".

| setting | relpos | rope | learnable relpos |
| --- | --- | --- | --- |
| offline | {{ctc:base:dev_test}} | {{ctc:base-rope:dev_test}} | |
| fixed chunk | {{ctc:chunked-L80-C5-R4-v2.3:dev_test}} | {{ctc:chunked-L80-C5-R4-v2.3-rope:dev_test}} | |
| dynamic chunk, +ctembed | {{ctc:chunked-L80-C5-R4-v2.3-dyn-ctembed:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-relposL-ctembed:dev_test}} |

RoPE is neutral offline and helps under chunking; learnable relpos is the worst of the three.
Why RoPE helps only under chunking was never resolved, and the investigation was stopped deliberately.

### Chunk-type embedding

Does tagging each frame as center or lookahead by its in-chunk position help?
The comparison only makes sense against whether the geometry varies during training,
so it is run in both settings with everything else held constant.

| setting | without ctembed | with ctembed |
| --- | --- | --- |
| fixed chunk, rope | {{ctc:chunked-L80-C5-R4-v2.3-rope:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-rope-ctembed:dev_test}} |
| dynamic chunk, rope | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |

It pays off only when the chunk geometry varies during training, and slightly hurts at fixed geometry.

### Fixed vs dynamic chunking

Sampling the chunk geometry per batch costs some WER but buys one model for all recog chunk sizes,
and trains much faster, since a fixed small chunk means many chunks per sequence.

| features | fixed | h | dynamic | h |
| --- | --- | --- | --- | --- |
| plain | {{ctc:chunked-L80-C5-R4-v2.3:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3:value|.1f}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3-dyn:value|.1f}} |
| + rope | {{ctc:chunked-L80-C5-R4-v2.3-rope:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3-rope:value|.1f}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3-dyn-rope:value|.1f}} |

### Dynamic train-pool composition

Given dynamic chunking, what should the pools contain?
Pools are chunk_size / history / lookahead; rope and ctembed are held fixed.

| pool variant | pools | dev / test |
| --- | --- | --- |
| dyn | [5,10,20,40,None] / [80,40] / [4,2] | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| dynCx3 | oversample C=5 | {{ctc:chunked-L80-C5-R4-v2.3-dynCx3-rope-ctembed:dev_test}} |
| dynV4 | + 0 in lookahead | {{ctc:chunked-L80-C5-R4-v2.3-dynV4-rope-ctembed:dev_test}} |
| dynV3 | + 0 in history too | {{ctc:chunked-L80-C5-R4-v2.3-dynV3-rope-ctembed:dev_test}} |
| dynV2 | + offline oversampled | {{ctc:chunked-L80-C5-R4-v2.3-dynV2-rope-ctembed:dev_test}} |

Every zero added to a pool costs WER, and oversampling the deployment chunk buys nothing.
The plain pool is the best of the five.

### Generalization across recog-time chunk size

Sweeping the chunk size and lookahead of a trained checkpoint at recog time,
with controls that never saw varying geometry during training.

Dynamic model:

| recog geometry | dev / test |
| --- | --- |
| C5-R2 | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C5-R2:dev_test}} |
| C5-R4 (deployment) | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C5-R4:dev_test}} |
| C10-R4 | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C10-R4:dev_test}} |
| C20-R4 | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C20-R4:dev_test}} |
| C40-R4 | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C40-R4:dev_test}} |
| offline | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:offline:dev_test}} |

A clean monotone curve, and the offline end is what the scaling section uses.
The fixed-chunk controls degrade off-canonical and collapse at offline recog:
rope {{sweep:chunked-L80-C5-R4-v2.3-rope:offline:dev_test}},
rope-ctembed {{sweep:chunked-L80-C5-R4-v2.3-rope-ctembed:offline:dev_test}}.

The complementary control loads the offline-trained base into the chunked encoder,
verified bit-exact at `chunk_size=None`, then chunks it at recog time only:

| recog geometry | dev / test |
| --- | --- |
| offline | {{sweep:base-via-v2.3:offline:dev_test}} |
| C20-R15 | {{sweep:base-via-v2.3:L80-C20-R15:dev_test}} |
| C10-R8 | {{sweep:base-via-v2.3:L80-C10-R8:dev_test}} |
| C5-R4 | {{sweep:base-via-v2.3:L80-C5-R4:dev_test}} |

So chunk-aware training, not just chunk-aware inference, is what matters.

### Overlapping chunks

Overlapping the chunks and averaging the views helps on its own,
but it doubles the compute, so the control is a plain run at twice the budget.

| variant | dev / test | h |
| --- | --- | --- |
| plain v2.3 | {{ctc:chunked-L80-C5-R4-v2.3:dev_test}} | {{hours:chunked-L80-C5-R4-v2.3:value|.1f}} |
| + overlap | {{ctc:chunked-L80-C5-R4-v2.3-overlap:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-overlap:value|.1f}} |
| + overlap + MSE | {{ctc:chunked-L80-C5-R4-v2.3-overlap-mse:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-overlap-mse:value|.1f}} |
| 2x budget, no overlap | {{ctc:chunked-L80-C5-R4-v2.3-2xtrain:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-2xtrain:value|.1f}} |

The 2x control is the one run in this doc whose last epoch must not be read:
its top CTC head diverged at epoch 200, so the comparable value is its best epoch 190,
{{ep:chunked-L80-C5-R4-v2.3-2xtrain:190:dev_test}}.
Read that way it matches overlap for the same compute, so overlap's gain is bought by compute,
not by overlap.
On top of the stronger dyn-rope-ctembed base it regresses outright:

| variant | dev / test |
| --- | --- |
| no overlap | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| overlap | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-overlap:dev_test}} |
| overlap + MSE | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-overlap-mse:dev_test}} |
| overlap dynamic | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-overlapD:dev_test}} |
| overlap dynamic + ctembedfix | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-overlapD-ctembedfix:dev_test}} |
| overlap dynamic, no ctembed | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-overlapD:dev_test}} |

Two further questions closed this line.
Turning overlap on at recog only, for a model never trained with it, hurts badly:
C5-R4 {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C5-R4-ov2:dev_test}},
C5-R2 {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C5-R2-ov2:dev_test}}.
And overlap does not remove the need for lookahead:
overlap at R0 gives {{ctc:chunked-L80-C5-R0-v2.3-overlap:dev_test}}.

### Offline init vs from scratch, at matched budget

Is it better to warm-start the streaming model from the offline one, or train it from scratch?
`impBase` initializes from the 1x base and finetunes for 1x, so 100 + 100 matches the 2x controls.

| variant | dev / test |
| --- | --- |
| from scratch, 1x | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| impBase, finetune LR 0.1 | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-impBase-baseLr0.1:dev_test}} |
| impBase, finetune LR 0.25 | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-impBase-baseLr0.25:dev_test}} |
| impBase, finetune LR 0.5 | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-impBase-baseLr0.5:dev_test}} |
| from scratch, 2x | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-2xtrain:dev_test}} |

Warm-starting beats 1x from scratch but loses to 2x from scratch,
so at equal compute the streaming model is better trained from scratch.

### Encoder architecture: attention vs linear-attention recurrence

Can a recurrent layer carry long context more cheaply than chunked attention?
Eight standard chunked conformer layers interleaved with eight recurrent ones,
outer chunk structure held fixed so the comparison is direct.

| encoder | dev / test | h |
| --- | --- | --- |
| conformer (dyn-rope-ctembed) | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:value|.1f}} |
| + Mamba-2 | {{ctc:chunked-L80-C5-R4-v2.3-mamba2:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-mamba2:value|.1f}} |
| + Mamba-2, bidirectional | {{ctc:chunked-L80-C5-R4-v2.3-mamba2-bidir-ssdchunk256:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-mamba2-bidir-ssdchunk256:value|.1f}} |
| + DeltaNet | {{ctc:chunked-L80-C5-R4-v2.3-deltanet:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-deltanet:value|.1f}} |
| + DeltaNet, bidirectional | {{ctc:chunked-L80-C5-R4-v2.3-deltanet-bidir:dev_test}} \
| {{hours:chunked-L80-C5-R4-v2.3-deltanet-bidir:value|.1f}} |

All worse than the conformer, Mamba-2 the best of the set,
and going bidirectional hurts both, which was the opposite of the expectation.

### Context extremes

What the chunk is worth, bracketed from both sides.

| model | dev / test |
| --- | --- |
| offline, full context | {{ctc:base:dev_test}} |
| chunked C5-R4 | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| fully causal, unlimited history, no lookahead | {{ctc:base-causal:dev_test}} |
| chunked C5-R4, no history | {{ctc:chunked-L0-C5-R4-v2.3:dev_test}} |
| feed-forward encoder, 12 layers | {{ctc:ff12:dev_test}} |
| feed-forward encoder, 6 layers | {{ctc:ff6:dev_test}} |

The fully-causal model has unlimited left context and still loses badly to the chunked one,
so the small lookahead does work that history cannot replace.
The feed-forward rows are the zero-context floor, a sanity bound rather than a competitor.

### Streaming inference correctness

An equivalence check, not a WER comparison: does the KV-cache streaming encoder,
which carries state across 10 s segments, match batched chunked inference?

Encoder log-probs on 3 sequences:
max abs diff {{out:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-consistency-kvcache.json\
:summary.max_of_max_abs_diff|.1e}},
mean abs diff {{out:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-consistency-kvcache.json\
:summary.mean_of_mean_abs_diff|.1e}}.

End-to-end WER, same checkpoint:
streaming {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-kvcache-v2-seg10:dev_test}}
vs batched chunked {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:L80-C5-R4:dev_test}}.

### Long-form vs segmented

Does a streaming model degrade on full-length recordings?
TEDLium, streaming-KV recog, on the dyn-rope-ctembed model only.

| eval set | WER |
| --- | --- |
| segmented (leaderboard) | {{out:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-kvcache-v2-seg10-tedlium:tedlium-seg.test}} |
| segmented, restricted to the long-form talks \
| {{out:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-kvcache-v2-seg10-tedlium:tedlium-seg-filtered.test}} |
| long-form (11 full talks) \
| {{out:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:streaming-kvcache-v2-seg10-tedlium:tedlium-long.test}} |

The filtered row is the honest comparison, since its references are word-for-word identical to long-form.
Segmented is level with or slightly better than long-form on the same content,
and the raw segmented gap is just the extra talk the long-form set omits.
Only one model has been run here, so this is not yet a cross-model comparison.

## Training scale

CTC-only WER, dev / test, last epoch.
base: offline recog.
dyn-rope-ctembed: streaming recog at the deployment chunk (C5, R4).
The `inf` row is not measured: it is the extrapolated floor of the fit below,
and is far less certain than the measured rows.

| scale | base (offline) | dyn-rope-ctembed (streaming) |
| --- | --- | --- |
| 1x | {{ctc:base:dev_test}} | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| 2x | {{ctc:base-2xtrain:dev_test}} | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-2xtrain:dev_test}} |
| 4x | {{ctc:base-4xtrain:dev_test}} | {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-4xtrain:dev_test}} |
| inf (fit) | {{fit:devtest:E_base}} | {{fit:devtest:E_dynon}} |

## Extrapolation to infinite training scale

Decoding the same streaming checkpoint at `chunk_size=None`, which its train pool includes,
splits the streaming cost into two parts:
the price of training under a chunk pool, and the price of decoding in chunks.

| scale | base offline | dyn offline | dyn online |
| --- | --- | --- | --- |
| 1x | {{ctc:base:dev_test}} | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:offline:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed:dev_test}} |
| 2x | {{ctc:base-2xtrain:dev_test}} | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-2xtrain:offline:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-2xtrain:dev_test}} |
| 4x | {{ctc:base-4xtrain:dev_test}} | {{sweep:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-4xtrain:offline:dev_test}} \
| {{ctc:chunked-L80-C5-R4-v2.3-dyn-rope-ctembed-4xtrain:dev_test}} |
| inf (fit) | {{fit:devtest:E_base}} | {{fit:devtest:E_dynoff}} | {{fit:devtest:E_dynon}} |

![WER vs training scale, with the shared-exponent power-law fit]("""
    + PLOT_NAME
    + """)

Fit `W(s) = E + b * s^-c` per curve.
Fitted alone that is 3 parameters on 3 points, so the floor E is unidentifiable,
and the floors come out unordered: the streaming floor sinks below its own offline floor,
which inference cannot do.
Sharing one exponent c across all three curves keeps E and b per curve,
giving 7 parameters on 9 points, and the ordering holds.
Result: dev c={{fit:dev:c}} (rms {{fit:dev:rms}} abs), test c={{fit:test:c}} (rms {{fit:test:rms}} abs).
"""
)


def main() -> int:
    """Write the doc and figure, or with --check only report how the doc differs from disk."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="Print a diff and write nothing.")
    p.add_argument("--out", default=_doc_path, help="Target path (default: next to the recipe).")
    p.add_argument("--setup-dir", help="Setup dir holding output/ (default: cwd).")
    args = p.parse_args()

    plot_path = None if args.check else os.path.join(os.path.dirname(args.out), PLOT_NAME)
    new = render(args.setup_dir, plot_path)
    old = ""
    if os.path.exists(args.out):
        with open(args.out) as f:
            old = f.read()

    if old != new:
        print(
            "".join(
                difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="on disk", tofile="generated")
            ),
            end="",
        )
    if args.check:
        return 1 if old != new else 0
    with open(args.out, "w") as f:
        f.write(new)
    print(f"wrote: {args.out}")
    if plot_path:
        print(f"wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
