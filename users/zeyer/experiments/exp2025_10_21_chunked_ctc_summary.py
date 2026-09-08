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

# Run-to-run WER noise, from the run2 duplicate of the dyn baseline (9.41 vs 9.52).
# Used only for the sensitivity band on the extrapolated floors.
WER_SIGMA = 0.07

_DYN = "chunked-L80-C5-R4-v2.3-dyn-rope-ctembed"

# Where each recog kind writes its result file, relative to the variant's output dir.
# "ctc+lm" has a single LM subdir whose name encodes the LM, so it is globbed.
_RECOG_PATHS = {
    "ctc": "aed+ctc/ctc-only-res.txt",
    "aedctc": "aed+ctc/recog-1stpass-res.txt",
    "ctclm": "ctc+lm-v2/*/recog-1stpass-res.txt",
    "sweep": "ctc-recog-sweep/{tag}",
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
            return json.loads(f.read().strip())
    except (OSError, ValueError):
        return {}


def _fmt(res: Dict[str, Any], field: str) -> str:
    """Render one field. ``dev_test`` is the paired form used in most tables.

    Two decimals always, so a WER that happens to be a round number
    (JSON 8.1) still lines up with the rest of the column as 8.10.
    """
    if field == "dev_test":
        dev, test = res.get("dev"), res.get("test")
        if dev is None or test is None:
            return MISSING
        return f"{dev:.2f} / {test:.2f}"
    v = res.get(field)
    return MISSING if v is None else f"{v:.2f}"


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


def _gap_ci(curves: Dict[str, List[float]], c_grid, n: int = 2000) -> Dict[str, Tuple[float, float]]:
    """10-90% range of the two asymptotic gaps under WER run noise.

    The floors themselves move far more than the gaps between them,
    so the gaps are what the doc quotes.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    train, stream = [], []
    for _ in range(n):
        pert = {k: list(np.array(v) + rng.normal(0, WER_SIGMA, len(v))) for k, v in curves.items()}
        _, pars, _ = _fit_joint(pert, c_grid)
        train.append(pars["dyn offline"][0] - pars["base offline"][0])
        stream.append(pars["dyn online"][0] - pars["dyn offline"][0])
    return {
        "gap_train": (float(np.percentile(train, 10)), float(np.percentile(train, 90))),
        "gap_stream": (float(np.percentile(stream, 10)), float(np.percentile(stream, 90))),
    }


def _fit_context(out_dir: str) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    """Placeholder values derived from the fit, plus what the plot needs."""
    try:
        import numpy as np
    except ImportError:
        raise SystemExit("numpy is required; run with the py-env interpreter (see the module docstring)")

    c_grid = np.linspace(0.05, 3.0, 4000)
    ctx: Dict[str, str] = {"fit:sigma:noise": f"{WER_SIGMA}"}
    state: Dict[str, Any] = {}
    for split in ("dev", "test"):
        curves = _curve_values(out_dir, split)
        if curves is None:
            continue
        c, pars, rss = _fit_joint(curves, c_grid)
        ci = _gap_ci(curves, c_grid)
        state[split] = {"curves": curves, "c": c, "pars": pars}
        ctx[f"fit:{split}:c"] = f"{c:.2f}"
        ctx[f"fit:{split}:rms"] = f"{math.sqrt(rss / 9):.3f}"
        for name, key in _CURVE_KEYS.items():
            ctx[f"fit:{split}:E_{key}"] = f"{pars[name][0]:.2f}"
        for gap, (lo, hi) in ci.items():
            ctx[f"fit:{split}:{gap}"] = f"{lo:.2f} to {hi:.2f}"
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

Read the gaps, not the floors.
The objective is nearly flat in c, and E slides along with c,
so the floor values above are not meaningful to two decimals.
Under {{fit:sigma:noise}} abs of run noise the differences are far better determined than the floors:

| asymptotic cost | dev (10-90%) | test (10-90%) |
| --- | --- | --- |
| chunked training (dyn offline - base offline) | {{fit:dev:gap_train}} | {{fit:test:gap_train}} |
| streaming inference (dyn online - dyn offline) | {{fit:dev:gap_stream}} | {{fit:test:gap_stream}} |

So training under a chunk pool costs little or nothing at infinite scale,
while decoding in chunks costs about 1.2 abs and scale does not remove it.
Three points cannot pin an asymptote; one more scale point (3x or 8x) would constrain it properly.
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
