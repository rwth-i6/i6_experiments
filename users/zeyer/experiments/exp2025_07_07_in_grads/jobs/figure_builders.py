"""Auto-generated grad-align figure DATA dumps (manifest + blobs), wired into the Sis graph.

``build_figures(results)`` is called at ``py()`` end with the same captured-outputs dict the table
and plot builders use. It emits one :class:`WriteFigureDataJob` per (model/tokenization, seq):
the four cells of the Whisper-large grad/cross-att x char/subword grid, on a curated subset of
internal-silence Buckeye-segA examples (a real pause between words shows the alignment placing the
silence). Each dumps ``figures-data/<name>/`` (``fig.json`` + ``blobs/*.npy``); a local renderer
turns these into the paper PDFs.
"""

from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.write_figure_data import WriteFigureDataJob

_ALIGN_OPTS = {"apply_softmax_over_time": True, "blank_score": -5}
# Internal-silence example seqs only (a real pause between words). 10 and 17 have no such pause.
_PICK = [8, 12, 18, 24]


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
        tk.register_output(f"figures-data/{name}-seq{si}", job.out_dir)


def build_figures(results):
    ds = results.get("buckeye-segA-5h-dataset")
    if ds is None:
        return
    # Whisper-large grad/cross-att x char/subword grid (internal-silence examples).
    # gradient-saliency variants
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
        "whisper-large-subword-buckeye-segA",
        "whisper-large-v3-logmel-buckeye-segA-5h-L2_grad-pertoken-subword",
        "Whisper-large",
        seqs=_PICK,
        tokenizer_reg="whisper-large-v3-model",
    )
    # cross-attention variants (the same DP, on the attention signal instead of the gradient)
    _emit(
        results,
        ds,
        "whisper-large-crossattn-buckeye-segA",
        "baseline-whisper-large-v3-crossattn-auto-buckeye-segA-5h",
        "Whisper-large cross-att",
        seqs=_PICK,
        tokenizer_reg="whisper-large-v3-model",
    )
    _emit(
        results,
        ds,
        "whisper-large-crossattn-char-buckeye-segA",
        "baseline-whisper-large-v3-crossattn-charlev-buckeye-segA-5h",
        "Whisper-large cross-att",
        seqs=_PICK,
    )
