"""Auto-generated grad-align figure DATA dumps (manifest + blobs), wired into the Sis graph.

``build_figures(results)`` (called at ``py()`` end) emits one :class:`WriteFigureDataJob` per
(model/tokenization, seq): the Whisper-large gradient (char) and cross-attention (subword) variants,
on a curated subset of internal-silence Buckeye-segA examples. Each dumps ``figures-data/<name>/``
(``fig.json`` + ``blobs/*.npy``).

Split preview pipeline, exactly mirroring the tables (``build_preview_tables`` + ``table_data.py
--refresh-preview``):

- ``build_figures_preview`` writes one tiny manifest per current figure to
  ``output/figures-data-preview/<name>.fig-manifest.json`` (the job's output dir), pruning manifests of
  renamed/removed figures. Written once per config load.
- ``figure_builders.py --refresh-preview <dir>`` re-resolves each manifest from current disk state at
  any time, with no manager restart: a finished figure becomes a symlink ``<name>`` to its output, a
  pending one is skipped. ``scripts/sync_figures.sh`` runs this before rsyncing the resolved set, so a
  figure whose job just finished shows up without a manager restart, and a still-pending one is simply
  left out (the figure analog of a table's pending cell).
"""

import json
import os
import sys
from functools import reduce

# Standalone-runnable for the preview refresh (python figure_builders.py --refresh-preview <dir>):
# add the recipe + sisyphus dirs to sys.path so the module's own sisyphus import resolves. A normal
# import already has __package__ set and skips this. Mirrors table_data.py.
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(6), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        os.environ.setdefault("SIS_GLOBAL_SETTINGS_FILE", f"{_setup_base_dir}/settings.py")


_setup()

from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.write_figure_data import WriteFigureDataJob

_ALIGN_OPTS = {"apply_softmax_over_time": True, "blank_score": -5}
# Curated internal-silence example seqs for the paper figures.
_PICK = [4, 5, 13]
# When a list (preview mode), ``_emit`` captures (name, out_dir) instead of registering an output.
_CAPTURE = None
_MANIFEST_SUFFIX = ".fig-manifest.json"


def _emit(results, ds, name, hdf_reg, model, *, seqs, tokenizer_reg=None):
    hdf = results.get(f"{hdf_reg}.hdf")
    if hdf is None:
        return
    tok = results.get(tokenizer_reg) if tokenizer_reg else None
    for si in seqs:
        job = WriteFigureDataJob(
            grad_score_hdf=hdf,
            grad_score_key="data",
            dataset_dir=ds,
            dataset_key="test",
            dataset_offset_factors=1,  # segA gold start/stop in samples
            align_opts=_ALIGN_OPTS,
            seq_idx=si,
            audio_energy_pow=0.5,
            blank_silence_energy_scale=1.0,
            boundary_source="word_detail",
            model=model,
            family="AED",
            dataset="buckeye-segA",
            tokenizer_dir=tok,
        )
        out_name = f"{name}-seq{si}"
        if _CAPTURE is not None:
            _CAPTURE.append((out_name, os.path.abspath(str(job.out_dir))))
        else:
            tk.register_output(f"figures-data/{out_name}", job.out_dir)


def build_figures(results):
    ds = results.get("buckeye-segA-5h-dataset")
    if ds is None:
        return
    # Whisper-large gradient (char) + cross-attention (subword), on curated internal-silence examples.
    _emit(
        results,
        ds,
        "whisper-large-buckeye-segA",
        "whisper-large-v3-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc",
        "Whisper-large",
        seqs=_PICK,
    )
    _emit(
        results,
        ds,
        "whisper-large-crossattn-buckeye-segA",
        "baseline-whisper-large-v3-crossattn-auto-buckeye-segA-5h",
        "Whisper-large cross-att",
        seqs=_PICK,
        tokenizer_reg="whisper-large-v3-model",
    )


def build_figures_preview(results):
    """Write one manifest per current figure to output/figures-data-preview/, pruning renamed/removed
    ones, then resolve once. See module docstring -- the figure analog of build_preview_tables."""
    global _CAPTURE
    _CAPTURE = []
    try:
        build_figures(results)  # capture (name, out_dir) instead of registering
        defs = list(_CAPTURE)
    finally:
        _CAPTURE = None
    out_dir = "output/figures-data-preview"
    os.makedirs(out_dir, exist_ok=True)
    keep = {name for name, _ in defs}
    for fn in os.listdir(out_dir):  # drop manifests + resolved links of renamed/removed figures
        base = fn[: -len(_MANIFEST_SUFFIX)] if fn.endswith(_MANIFEST_SUFFIX) else fn
        if base not in keep:
            os.remove(os.path.join(out_dir, fn))
    for name, job_out in defs:
        with open(os.path.join(out_dir, name + _MANIFEST_SUFFIX), "w") as f:
            json.dump({"out_dir": job_out}, f)
    refresh_preview(out_dir)


def refresh_preview(manifest_dir):
    """Re-resolve each figure manifest from current disk state: a finished job -> a symlink <name> to
    its output dir, a pending one -> no symlink (skipped). No manager restart needed."""
    n_ready = n_pending = 0
    for fn in sorted(os.listdir(manifest_dir)):
        if not fn.endswith(_MANIFEST_SUFFIX):
            continue
        name = fn[: -len(_MANIFEST_SUFFIX)]
        with open(os.path.join(manifest_dir, fn)) as f:
            out = json.load(f)["out_dir"]
        link = os.path.join(manifest_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        if os.path.exists(os.path.join(out, "fig.json")):
            os.symlink(out, link)
            n_ready += 1
        else:
            n_pending += 1
    print(f"refreshed figure preview in {manifest_dir}: {n_ready} ready, {n_pending} pending")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="resolve the figure-data preview symlinks from current disk state")
    parser.add_argument("--refresh-preview", required=True, metavar="DIR", help="the output/figures-data-preview dir")
    refresh_preview(parser.parse_args().refresh_preview)
