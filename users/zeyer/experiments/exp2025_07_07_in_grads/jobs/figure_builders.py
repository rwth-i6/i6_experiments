"""Auto-generated grad-align figure DATA dumps (manifest + blobs), wired into the Sis graph.

A figure is one or more *parts* (panels), the analog of a table's cells.
Each part is a :class:`WriteFigureDataJob` that dumps ``figures-data/<part>/`` (``fig.json`` + blobs) --
the actual alignment data, loaded by the local renderer.

``_figures`` lists the figures.
By default each (alignment kind, seq) is its own *single-part* figure named by its part,
so the structure file and the rendered PDF keep that part's name.
To combine kinds into one figure (e.g. gradient + cross-attention side by side),
yield them together under one figure name instead;
the renderer stacks the parts and draws a placeholder panel for any pending one.

Split preview pipeline, mirroring the tables (``build_preview_tables`` + ``table_data.py --refresh-preview``).
A figure's parts are the analog of a table's cells, and a pending part the analog of a pending cell:

- ``build_figures_preview`` writes one manifest per figure
  to ``output/figures-data-preview/<fig>.fig-manifest.json`` (its parts and their job out_dirs),
  pruning manifests of renamed/removed figures.
- ``figure_builders.py --refresh-preview <dir>`` re-resolves every manifest from current disk state,
  with no manager restart:
  it writes the complete structure ``<fig>.fig.json`` (each part flagged ``ready`` or not),
  and symlinks the ready parts' data dirs.
  ``scripts/sync_figures.sh`` runs this before rsyncing, then the renderer draws a placeholder panel
  for any part still flagged not-ready -- exactly as a table renders ``·`` for a pending cell.
"""

import json
import os
import sys
from functools import reduce

# Standalone-runnable for the preview refresh (``python figure_builders.py --refresh-preview <dir>``):
# add the recipe + sisyphus dirs to sys.path so the module's own sisyphus import resolves.
# A normal import already has ``__package__`` set and skips this.
# Mirrors table_data.py.
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
# Curated internal-silence example seqs.
_PICK = [4, 5, 13]
# The alignment kinds available as figure parts (panels):
# (panel label, data-name prefix, per-token-score HDF, model label, tokenizer result key or None).
_KINDS = [
    (
        "Gradient",
        "whisper-large-buckeye-segA",
        "whisper-large-v3-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc",
        "Whisper-large",
        None,
    ),
    (
        "Cross-attention",
        "whisper-large-crossattn-buckeye-segA",
        "baseline-whisper-large-v3-crossattn-auto-buckeye-segA-5h",
        "Whisper-large cross-att",
        "whisper-large-v3-model",
    ),
    (
        "Gradient",
        "fastconformer-ctc-buckeye-segA",
        "fastconformer-stream-ctc-buckeye-segA-5h-L2_grad-pertoken",
        "FastConformer-CTC",
        None,
    ),
    (
        "Gradient",
        "voxtral-buckeye-segA",
        "voxtral-charlevlogmel-buckeye-segA-5h-L1_grad-pertoken",
        "Voxtral",
        None,
    ),
    (
        "Self-attention",
        "voxtral-selfattn-buckeye-segA",
        "baseline-voxtral-selfattn-buckeye-segA-5h",
        "Voxtral self-att",
        None,  # voxtral tekken tokenizer needs the overlay (not in the render env); labels blank
    ),
]
_MANIFEST_SUFFIX = ".fig-manifest.json"
_FIG_SUFFIX = ".fig.json"


def _part_job(results, ds, hdf_reg, model, seq, tokenizer_reg):
    hdf = results.get(f"{hdf_reg}.hdf")
    if hdf is None:
        return None
    return WriteFigureDataJob(
        grad_score_hdf=hdf,
        grad_score_key="data",
        dataset_dir=ds,
        dataset_key="test",
        dataset_offset_factors=1,  # segA gold start/stop in samples
        align_opts=_ALIGN_OPTS,
        seq_idx=seq,
        audio_energy_pow=0.5,
        blank_silence_energy_scale=1.0,
        boundary_source="word_detail",
        model=model,
        family="AED",
        dataset="buckeye-segA",
        tokenizer_dir=results.get(tokenizer_reg) if tokenizer_reg else None,
    )


def _figures(results, ds):
    """Yield ``(figure_name, parts)`` with ``parts = [(label, data_name, job)]``.

    Default: one single-part figure per (kind, seq), named by its part, so the names match the data.
    To combine, yield several parts under one figure_name instead -- e.g. for a gradient-vs-attention
    figure replace the inner loop with one ``yield f"compare-seq{seq}", [grad_part, att_part]``.
    ``label`` is the panel's y-axis title; None falls back to the part's own model name.
    """
    for seq in _PICK:
        for label, prefix, hdf_reg, model, tok_reg in _KINDS:
            job = _part_job(results, ds, hdf_reg, model, seq, tok_reg)
            if job is not None:
                name = f"{prefix}-seq{seq}"
                yield name, [(None, name, job)]  # single-part: label None -> the part's model name


def build_figures(results):
    """Register every figure part as a Sis output so its job runs (deduped across figures)."""
    ds = results.get("buckeye-segA-5h-dataset")
    if ds is None:
        return
    seen = set()
    for _name, parts in _figures(results, ds):
        for _label, data_name, job in parts:
            if data_name not in seen:
                seen.add(data_name)
                tk.register_output(f"figures-data/{data_name}", job.out_dir)


def build_figures_preview(results):
    """Write one manifest per figure to ``output/figures-data-preview/``, pruning renamed/removed ones,
    then resolve once. The figure analog of build_preview_tables; see the module docstring."""
    ds = results.get("buckeye-segA-5h-dataset")
    if ds is None:
        return
    figures = {}
    for name, parts in _figures(results, ds):
        figures[name] = {
            "name": name,
            "parts": [
                {"label": label, "name": data_name, "out_dir": os.path.abspath(str(job.out_dir))}
                for label, data_name, job in parts
            ],
        }
    out_dir = "output/figures-data-preview"
    os.makedirs(out_dir, exist_ok=True)
    # Drop manifests of renamed/removed figures.
    for fn in os.listdir(out_dir):
        if fn.endswith(_MANIFEST_SUFFIX) and fn[: -len(_MANIFEST_SUFFIX)] not in figures:
            os.remove(os.path.join(out_dir, fn))
    for name, fig in figures.items():
        with open(os.path.join(out_dir, name + _MANIFEST_SUFFIX), "w") as f:
            json.dump(fig, f)
    refresh_preview(out_dir)


def refresh_preview(manifest_dir):
    """Rebuild every figure's resolved structure + ready-part symlinks from the manifests and current
    disk state, with no manager restart:
    a finished part becomes ``ready`` plus a symlink to its data dir,
    a pending one stays in the structure flagged not-ready (the renderer draws a placeholder for it)."""
    manifests = sorted(fn for fn in os.listdir(manifest_dir) if fn.endswith(_MANIFEST_SUFFIX))
    # Clear the previously-resolved entries (the *.fig.json structures and the part symlinks);
    # keep only the manifests, which build_figures_preview owns.
    for fn in os.listdir(manifest_dir):
        if not fn.endswith(_MANIFEST_SUFFIX):
            os.remove(os.path.join(manifest_dir, fn))
    n_ready = n_pending = 0
    for mf in manifests:
        with open(os.path.join(manifest_dir, mf)) as f:
            fig = json.load(f)
        parts_out = []
        for part in fig["parts"]:
            ready = os.path.exists(os.path.join(part["out_dir"], "fig.json"))
            parts_out.append({"label": part["label"], "name": part["name"], "ready": ready})
            if ready:
                link = os.path.join(manifest_dir, part["name"])
                if not os.path.islink(link):
                    os.symlink(part["out_dir"], link)
                n_ready += 1
            else:
                n_pending += 1
        with open(os.path.join(manifest_dir, fig["name"] + _FIG_SUFFIX), "w") as f:
            json.dump({"name": fig["name"], "parts": parts_out}, f)
    print(f"refreshed figure preview in {manifest_dir}: {n_ready} ready, {n_pending} pending")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="resolve the figure-data preview from current disk state")
    parser.add_argument("--refresh-preview", required=True, metavar="DIR", help="the output/figures-data-preview dir")
    refresh_preview(parser.parse_args().refresh_preview)
