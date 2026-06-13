"""Auto-generated grad-score figures, wired into the Sis graph.

``build_plots(results)`` is called at the end of the main recipe with the same
``{registered-output-name: Variable/Path}`` dict the table builder uses (captured by the ``reg``
wrapper during ``py()``). For each picked example it looks up that model's per-token grad HDF
(registered as ``<name>.hdf``) and the TIMIT dataset (``timit-dataset``), and emits one
:class:`PlotGradAlignJob` (the actual DP emission: log-softmax-over-time per token + silence strip),
registered under ``output/figures/``. The paper ``\\includegraphics`` the PDF.

Three groups: (1) one example per model family on the SAME TIMIT-test utterance (seq 3); (2) the
silence-lever diff (same Whisper-char grads, sil1.0 vs sil0.0 -> the silence strip lights up or not);
(3) the char-vs-subword diff (Whisper-base, same TIMIT-val utterance -> finer rows + tighter peaks
for char). Plus a deliberately weaker case (Parakeet TDT). All en0.5 unless noted.
"""

from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.plot_grad_align import PlotGradAlignJob

_SEQ = 3  # same utterance across families -> directly comparable


def _fig(hdf, title, *, key="test", seq=_SEQ, boundary="word_detail", sil=1.0, en=0.5):
    return dict(hdf=hdf, title=title, key=key, seq=seq, boundary=boundary, sil=sil, en=en)


_FIGURES = [
    # (1) one per model family (same TIMIT-test utterance) ---------------------------------------
    _fig("wav2vec2ctc-fproj_out-timit-test-L2_grad-pertoken", "Wav2Vec2-CTC grad-align (TIMIT-test)"),
    _fig(
        "w2v-phoneme-timit-test-L2_grad-perphone",
        "Phoneme-CTC grad-align, per-phone (TIMIT-test)",
        boundary="phonetic_detail",
    ),
    _fig("whisper-base-logmel-timit-test-L2_grad-pertoken-charlev-spc", "Whisper-base AED, char-level (TIMIT-test)"),
    _fig(
        "parakeet-rnnt-1.1b-logmel-timit-test-L2_grad-pertoken", "Parakeet RNN-T (transducer) grad-align (TIMIT-test)"
    ),
    _fig("voxtral-charlevlogmel-timit-test-L1_grad-pertoken", "Voxtral speech-LLM, char-level (TIMIT-test)"),
    _fig(
        "canary-qwen-charlev-spc-logmel-st15-timit-test-L1_grad-pertoken",
        "Canary-Qwen speech-LLM, char-level (TIMIT-test)",
    ),
    _fig("phi4mm-timit-test-L2_e_grad-pertoken-charlev-spc", "Phi-4-MM speech-LLM, char-level (TIMIT-test)"),
    _fig(
        "parakeet-tdt-0.6b-v2-logmel-timit-test-L2_grad-pertoken",
        "Parakeet TDT grad-align -- a harder case (TIMIT-test)",
    ),
    # (2) silence-lever diff: same Whisper-base char grads, sil1.0 (family fig above) vs sil0.0 ---
    _fig(
        "whisper-base-logmel-timit-test-L2_grad-pertoken-charlev-spc",
        "Whisper-base char, NO silence row (sil0.0) -- contrast the silence strip",
        sil=0.0,
    ),
    # (3) char-vs-subword diff: Whisper-base, same TIMIT-val utterance ----------------------------
    _fig(
        "whisper-base-logmel-timit-val-L2_grad-pertoken-charlev-spc",
        "Whisper-base CHAR-level grad-align (TIMIT-val)",
        key="val",
    ),
    _fig(
        "whisper-base-logmel-timit-val-L2_grad-pertoken", "Whisper-base SUBWORD-level grad-align (TIMIT-val)", key="val"
    ),
]


def build_plots(results):
    timit = results.get("timit-dataset")
    if timit is None:
        return
    for f in _FIGURES:
        hdf = results.get(f"{f['hdf']}.hdf")
        if hdf is None:
            continue
        job = PlotGradAlignJob(
            grad_score_hdf=hdf,
            grad_score_key="data",
            dataset_dir=timit,
            dataset_key=f["key"],
            dataset_offset_factors=1,  # _DATASET_OFFSET_FACTORS["timit"]
            align_opts={"apply_softmax_over_time": True, "blank_score": -5},
            seq_idx=f["seq"],
            audio_energy_pow=f["en"],
            blank_silence_energy_scale=f["sil"],
            boundary_source=f["boundary"],
            title=f["title"],
        )
        name = f"{f['hdf']}-seq{f['seq']}-en{f['en']}-sil{f['sil']}"
        tk.register_output(f"figures/{name}.png", job.out_png)
        tk.register_output(f"figures/{name}.pdf", job.out_pdf)
