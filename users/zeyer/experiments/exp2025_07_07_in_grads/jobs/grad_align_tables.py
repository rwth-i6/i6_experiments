"""Resolve the paper's result tables to ``data.json`` (numbers only), wired into the Sis graph.

``build_tables(results)`` is called at the end of the main recipe.
``results`` is the ``{registered-output-name: Variable}`` dict
captured by the module-level ``reg`` wrapper during ``py()``
(sis_graph is NOT introspectable mid-py()).
For each paper table it looks up the relevant metric outputs
and emits one :class:`WriteTableDataJob`, registered under ``output/tables-data/``.

This is the *data* half of the table pipeline:
only the numbers, which genuinely depend on upstream jobs, live here.
All presentation -- captions, column headers, units, layout, col-align --
lives in authored spec files in the paper repo (``tables-spec/<name>.spec.json`` + ``<name>.caption.tex``),
and is applied by the local ``scripts/render_tables.py``.
So a caption or header tweak is a one-second local re-render, never a manager restart;
the manager only runs when an actual number changes.

Lookup is by registered output NAME (this recipe's, never stale on-disk aliases).
A "<base>-wbe.txt" output IS the scalar out_wbe Variable;
its ``.creator`` is the metric job (whose ``out_metrics`` holds the full dict incl. acc@collar).
So WBE works for every aligner; acc needs out_metrics.
"""

import sys
from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data import WriteTableDataJob

_RESULTS = {}


def build_tables(results):
    global _RESULTS
    _RESULTS = results
    _compare_table()  # ground-truth forced-mode only (separate hyp table below)
    _compare_table(with_hyp=True)  # variant: hyp-mode columns merged onto the grad rows
    _hyp_table()
    _owsm_layer_table()  # OWSM-CTC grad-align per inter-CTC emit block (side table)
    _phi4_prompt_table()  # Phi-4-MM grad-align vs the spliced instruction (side table)
    _cost_table()  # forward vs batched grad-backward per model (T5 cost)
    _streaming_offset_table()  # streaming start/end signed offset (grad vs native viterbi)
    _time_stretch_table()  # length robustness (grad vs MMS-FA, vocoder vs resample)
    _word_length_table()  # word-duration accuracy (signed word-width error) per model
    _matched_tok_table()  # matched-tokenization: grad vs cross-attn x char/subword (whisper)
    _ablation_table()  # T3a grad-score reduction (cross-model)
    _attribution_table()  # T3b attribution method (cross-model, fast models)
    _alignopts_table()  # T3b align-opts (grad vs cross-attn, same DP)


def _emit(name, columns, rows):
    # `source` records this builder (file :: function) for a provenance comment in the rendered table;
    # the per-cell `src` (the registered output name) is what actually pins each number's origin.
    source = f"{__name__.replace('.', '/')}.py :: {sys._getframe(1).f_code.co_name}"
    job = WriteTableDataJob(columns=columns, rows=rows, source=source)
    tk.register_output(f"tables-data/{name}.data.json", job.out_data)


# Cell resolve-specs: var (+ optional dict key),
# plus `src` = the Sisyphus registered output name the number comes from.
# units/format are applied by the render layer;
# `src` becomes the per-row provenance comment.
def _wbe(base):
    # The "<base>-wbe.txt" output is the scalar out_wbe (raw seconds).
    name = f"{base}-wbe.txt"
    return {"var": _RESULTS.get(name), "src": name}


def _metrics_var(base):
    # The metrics dict behind a "<base>-wbe.txt": creator.out_metrics,
    # or out_word_metrics for the CTC forced-align / phoneme baselines
    # (which name their aggregate dict differently).
    v = _RESULTS.get(f"{base}-wbe.txt")
    j = getattr(v, "creator", None) if v is not None else None
    m = getattr(j, "out_metrics", None) if j is not None else None
    if m is None and j is not None:
        m = getattr(j, "out_word_metrics", None)
    return m


def _metric(base, key):
    # Value comes from the metrics dict,
    # but the registered output that pins it is the "<base>-wbe.txt".
    return {"var": _metrics_var(base), "key": key, "src": f"{base}-wbe.txt"}


def _hyp(name, key):
    out = f"hyp-align/{name}-metrics.txt"
    return {"var": _RESULTS.get(out), "key": key, "src": out}


# ----------------------------------------------------------------------------------------
# Streaming signed-offset: per streaming model, grad vs native viterbi, SIGNED start/end boundary
#     offset (pred - ref, ms; + = late). Native chunked viterbi starts late / ends early (emission
#     delay); grad ~0. Data = start_signed_mean / end_signed_mean (align_metrics out_metrics).
# ----------------------------------------------------------------------------------------
def _streaming_offset_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}")

    MODELS = [
        (
            "Emformer (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("Posteriors", f"baseline-fastconformer-stream-ctc-{S}", f"baseline-fastconformer-stream-ctc-{T}"),
            ],
        ),
        (
            "FastConformer-RNN-T (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
    ]

    columns = ["method", "t_start", "t_end", "s_start", "s_end"]
    rows = []
    for gi, (model, methods) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, (mlabel, sa, ti) in enumerate(methods):
            if mi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": model if mi == 0 else "",
                    "cells": {
                        "method": mlabel,
                        "t_start": _metric(ti, "start_signed_mean"),
                        "t_end": _metric(ti, "end_signed_mean"),
                        "s_start": _metric(sa, "start_signed_mean"),
                        "s_end": _metric(sa, "end_signed_mean"),
                    },
                }
            )
    _emit("streaming-offset", columns, rows)


# ----------------------------------------------------------------------------------------
# Length robustness: grad-align (wav2vec2) vs MMS-FA forced-align under audio time-stretch,
#     vocoder (librosa.effects.time_stretch, pitch-preserving) vs resample (pitch-shifted),
#     at 1.0/1.5/2.0/3.0x on TIMIT-val. grad-align tolerates mild stretch but degrades sharply
#     at 2-3x (resample worst); forced-align re-runs viterbi so it stays robust.
# ----------------------------------------------------------------------------------------
def _time_stretch_table():
    GRAD0 = "align/wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    def _grad(ts, method):
        if ts == "1.0":
            return GRAD0
        sfx = f"-ts{ts}" + ("-resample" if method == "resample" else "")
        return f"align/wav2vec2ctc-fproj_out-timit-val-L2_grad-pertoken{sfx}-asotTrue-bs-5-en0.5-sil1.0"

    def _mms(ts, method):
        if ts == "1.0":
            return "baseline-mms_fa-timit-val"
        return "baseline-mms_fa-timit-val" + f"-ts{ts}" + ("-resample" if method == "resample" else "")

    MODELS = [("grad-align", _grad), ("MMS-FA", _mms)]
    TS = ["1.0", "1.5", "2.0", "3.0"]
    KEYS = ["ts10", "ts15", "ts20", "ts30"]

    columns = ["method"] + KEYS
    rows = []
    for gi, (model, basefn) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, method in enumerate(("vocoder", "resample")):
            if mi > 0:
                rows.append({"cline": True})
            cells = {"method": method}
            for k, ts in zip(KEYS, TS):
                cells[k] = _wbe(basefn(ts, method))
            rows.append({"label": model if mi == 0 else "", "cells": cells})
    _emit("time-stretch", columns, rows)


# ----------------------------------------------------------------------------------------
# Word-length (duration) accuracy, full breakdown (Buckeye-segA): WBE (mean abs boundary error),
#     signed start/end offset (pred - ref; + = late), and word-WIDTH signed error (= end - start;
#     0 = correct duration). The CTC/forced-align posteriors bias start LATE + end EARLY -> the word
#     shrinks from both ends, so |width| ~ WBE; grad shifts start+end the SAME direction (a positional
#     lead, not a width change), so width is far below WBE -> grad is more accurate on word duration.
#     TIMIT-test shows the same pattern, milder (grad's lead is largest on spontaneous Buckeye).
# ----------------------------------------------------------------------------------------
def _word_length_table():
    S = "buckeye-segA-5h"
    MODELS = [
        (
            "Wav2Vec2-CTC",
            [
                ("Gradients", f"align/wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("Posteriors", f"baseline-mms_fa-{S}"),
            ],
        ),
        (
            "Phoneme-CTC",
            [
                (
                    "Gradients",
                    f"align/phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                ),
                ("Posteriors", f"baseline-phoneme-fa-{S}-word"),
            ],
        ),
        (
            "Nvidia CTC",
            [
                ("Gradients", f"align/parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("Posteriors", f"baseline-parakeet-ctc-1.1b-{S}"),
            ],
        ),
        (
            "OWSM-CTC",
            [
                ("Gradients", f"align/owsm-ctc-v4-1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("Posteriors", f"baseline-owsm-ctc-v4-1b-{S}"),
            ],
        ),
        (
            "Whisper-base",
            [
                ("Gradients", f"align/whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("Cross-att.", f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0"),
            ],
        ),
    ]

    columns = ["method", "wbe", "start", "end", "width"]
    rows = []
    for gi, (model, methods) in enumerate(MODELS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for mi, (mlabel, base) in enumerate(methods):
            if mi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": model if mi == 0 else "",
                    "cells": {
                        "method": mlabel,
                        "wbe": _metric(base, "wbe"),
                        "start": _metric(base, "start_signed_mean"),
                        "end": _metric(base, "end_signed_mean"),
                        "width": _metric(base, "width_signed_err"),
                    },
                }
            )
    _emit("word-length", columns, rows)


# ----------------------------------------------------------------------------------------
# T2 fairness 2x2: whisper grad vs cross-attn DTW, char vs subword. Method-consistent -- both use the
#     PLAIN cross-attn forced align (no head auto-selection), so the only variable per column is the
#     tokenization. Grad wins every cell; char helps grad; char collapses cross-attn on spontaneous Buckeye.
# ----------------------------------------------------------------------------------------
def _matched_tok_table():
    DATASETS = [("TIMIT", "timit-test"), ("Buckeye", "buckeye-segA-5h")]
    columns = ["method", "c_wbe", "c_a50", "s_wbe", "s_a50"]
    rows = []
    for di, (dlabel, ds) in enumerate(DATASETS):
        if di > 0:
            rows.append({"label": None, "cells": {}})
        gc = f"align/whisper-base-logmel-{ds}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"
        gs = f"align/whisper-base-logmel-{ds}-L2_grad-pertoken-subword-asotTrue-bs-5-en0.5-sil1.0"
        xc = f"baseline-whisper-crossattn-charlev-{ds}"
        xs = f"baseline-whisper-crossattn-{ds}"
        rows.append(
            {
                "label": dlabel,
                "cells": {
                    "method": "Gradients",
                    "c_wbe": _wbe(gc),
                    "c_a50": _metric(gc, "acc_50ms"),
                    "s_wbe": _wbe(gs),
                    "s_a50": _metric(gs, "acc_50ms"),
                },
            }
        )
        rows.append({"cline": True})
        rows.append(
            {
                "label": "",
                "cells": {
                    "method": "Cross-att.",
                    "c_wbe": _wbe(xc),
                    "c_a50": _metric(xc, "acc_50ms"),
                    "s_wbe": _wbe(xs),
                    "s_a50": _metric(xs, "acc_50ms"),
                },
            }
        )
    _emit("matched-tokenization", columns, rows)


# ----------------------------------------------------------------------------------------
# T3a grad-score reduction ablation (cross-model, Buckeye-segA, word-topology fixed): the per-token
#     reduction L0.5 / L1 / L2 / L2_e across one representative model per family. A few ms apart -> the
#     reduction choice is not critical; the saliency signal, not the norm, carries the alignment.
#     ("dot"/sum reductions are excluded: signed, they break the time-softmax scoring.)
# ----------------------------------------------------------------------------------------
def _ablation_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2-CTC", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
        ("Parakeet RNN-T", "parakeet-rnnt-1.1b-logmel"),
        ("Emformer (streaming)", "emformer-rnnt-prefix-logmel"),
        ("Voxtral", "voxtral-charlevlogmel"),
    ]
    REDS = ["L0.5_grad", "L1_grad", "L2_grad", "L2_e_grad"]
    columns = list(REDS)
    rows = [
        {
            "label": mlabel,
            "cells": {rk: _wbe(f"align/{mpre}-abl-{S}-{rk}-pertoken-{SFX}") for rk in REDS},
        }
        for mlabel, mpre in MODELS
    ]
    _emit("ablation", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b attribution-method ablation (cross-model, Buckeye-segA, word-topology fixed): plain L2 input-grad
#     vs SmoothGrad / VarGrad / Integrated-Gradients / Expected-Gradients, on the two fast models. The
#     single-pass gradient is already a good attribution -> the multi-pass methods are not worth it
#     (the slow families are omitted for the same cost reason).
# ----------------------------------------------------------------------------------------
def _attribution_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2-CTC", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
    ]
    METHODS = [
        "L2_grad",
        "L2-smoothgrad-std0.01-n8",
        "L2-vargrad-std0.02-n16",
        "L2-integrated-grad-ig16",
        "L2-expected-grad-eg16-bstd0.05",
    ]
    columns = list(METHODS)
    rows = [
        {
            "label": mlabel,
            "cells": {mk: _wbe(f"align/{mpre}-abl-{S}-{mk}-pertoken-{SFX}") for mk in METHODS},
        }
        for mlabel, mpre in MODELS
    ]
    _emit("attribution", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b align-OPTS (DP decoding) on Buckeye-segA: the SAME DP applied to whisper grad (char) and whisper
#     cross-attn (auto heads) -- different signal, same decoder. word-topology / silence / DTW corners.
#     Mostly consistent across signals; silence modeling helps grad but not attention (the attention
#     matrix already has ~no mass in silence). The openai whisper-DTW is the apply-log-off corner.
# ----------------------------------------------------------------------------------------
def _alignopts_table():
    GP = "align/whisper-base-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc-asotTrue-bs-5"
    AP = "align/baseline-whisper-base-crossattn-auto-buckeye-segA-5h-asotTrue-bs-5"
    # (row label, grad suffix, cross-attn suffix)
    ROWS = [
        ("default (CTC-topo)", "-en0.5-sil1.0", "-en0.5-sil1.0"),
        ("word-topology", "-en0.5-sil1.0-wordtopo", "-en0.5-sil1.0-wordtopo"),
        ("silence off", "-en0.5", ""),
        ("DTW (no blank)", "-en0.5-sil1.0-dtw", "-en0.5-sil1.0-dtw"),
        ("whisper-DTW (openai)", "-en0.5-sil1.0-wdtw", "-en0.5-sil1.0-wdtw"),
    ]
    columns = ["g_wbe", "g_a50", "a_wbe", "a_a50"]
    rows = []
    for lbl, gs, as_ in ROWS:
        gb, ab = GP + gs, AP + as_
        rows.append(
            {
                "label": lbl,
                "cells": {
                    "g_wbe": _wbe(gb),
                    "g_a50": _metric(gb, "acc_50ms"),
                    "a_wbe": _wbe(ab),
                    "a_a50": _metric(ab, "acc_50ms"),
                },
            }
        )
    _emit("alignopts", columns, rows)


# ----------------------------------------------------------------------------------------
# T1: per-model grad-align vs the model's own alternative aligner (same model, different method).
#     Rows grouped by model; columns = align method + per-dataset metrics (TIMIT, Buckeye).
# ----------------------------------------------------------------------------------------
def _compare_table(with_hyp=False):
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):  # grad-align align/ output bases (segA, TIMIT)
        return (f"align/{stem_sa}", f"align/{stem_ti}" if stem_ti else None)

    # (Model, [(align-method label, segA base, TIMIT base), ...])  -- grad first, then its alternative.
    # The method label names the alignment SIGNAL: grad / posteriors (CTC, transducer, GMM-HMM) /
    # cross-attn weights / self-attn weights.
    MODELS = [
        (
            "Wav2Vec2-CTC",
            [
                (
                    "Gradients",
                    *g(
                        f"wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"wav2vec2ctc-fproj_out-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("Posteriors", f"baseline-mms_fa-{S}", f"baseline-mms_fa-{T}"),
            ],
        ),
        (
            "Phoneme-CTC",
            [
                (
                    "Gradients",
                    *g(
                        f"phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                        f"phoneme-vitouphy-prefixfwd-{T}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-phoneme-fa-{S}-word",
                    "align/w2v-phoneme-timit-test-ctc-forced-align-word",
                ),
            ],
        ),
        (
            "Nvidia CTC",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-ctc-1.1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("Posteriors", f"baseline-parakeet-ctc-1.1b-{S}", f"baseline-parakeet-ctc-1.1b-{T}"),
            ],
        ),
        (
            # OWSM-CTC: general graphemic CTC, our least-favourable model for grad-align (still runs,
            # just imprecise). Shown with its BEST emit block (6); per-block detail in tab:owsm-per-layer.
            "OWSM-CTC",
            [
                (
                    "Gradients",
                    *g(
                        f"owsm-ctc-v4-1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"owsm-ctc-v4-1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                # posteriors = CTC forced-align of the model's emission; best/natural at the final block.
                ("Posteriors", f"baseline-owsm-ctc-v4-1b-{S}", f"baseline-owsm-ctc-v4-1b-{T}"),
            ],
        ),
        (
            "Whisper-base",
            [
                (
                    "Gradients",
                    *g(
                        f"whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-base-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Cross-att.",
                    f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-whisper-base-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Whisper-large-v3",
            [
                (
                    "Gradients",
                    *g(
                        f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-large-v3-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Cross-att.",
                    f"baseline-whisper-large-v3-crossattn-{S}",
                    f"baseline-whisper-large-v3-crossattn-{T}",
                ),
            ],
        ),
        (
            "Parakeet RNN-T",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Parakeet TDT",
            [
                (
                    "Gradients",
                    *g(
                        f"parakeet-tdt-0.6b-v2-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-tdt-0.6b-v2-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{S}",
                    f"baseline-parakeet-tdt-0.6b-v2-native-viterbi-{T}",
                ),
            ],
        ),
        (
            # Emformer RNN-T: the streaming transducer (vs offline Parakeet). grad resists the
            # streaming emission delay that the native forced alignment carries.
            "Emformer (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-ctc-{S}",
                    f"baseline-fastconformer-stream-ctc-{T}",
                ),
            ],
        ),
        (
            "FastConformer-RNN-T (streaming)",
            [
                (
                    "Gradients",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Voxtral",
            [
                (
                    "Gradients",
                    *g(
                        f"voxtral-charlevlogmel-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"voxtral-charlevlogmel-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-voxtral-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-voxtral-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Phi-4-MM",
            [
                (
                    "Gradients",
                    *g(
                        f"phi4mm-{S}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"phi4mm-{T}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-phi4mm-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-phi4mm-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Canary-Qwen",
            [
                (
                    "Gradients",
                    *g(
                        f"canary-qwen-charlev-spc-logmel-st15-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"canary-qwen-charlev-spc-logmel-st15-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "Self-att.",
                    f"align/baseline-canary-qwen-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-canary-qwen-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
    ]
    # Dedicated-aligner ceiling (model-agnostic), shown for reference.
    REFERENCE = [
        ("MFA (GMM-HMM)", [("Posteriors", f"baseline-mfa-{S}", f"baseline-mfa-{T}")]),
    ]

    # Hyp-mode (each model aligns its OWN recognition, Buckeye only) -> grad rows only, where it exists.
    HYP = {
        "Whisper-base": f"whisper-base-charlev-{S}-grad",
        "Voxtral": f"voxtral-charlevlogmel-{S}-grad",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-grad",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-grad",
        "Parakeet RNN-T": f"parakeet-rnnt-1.1b-logmel-{S}-grad",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-grad",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-grad",
        "Nvidia CTC": f"parakeet-ctc-1.1b-{S}-grad",
        "OWSM-CTC": f"owsm-ctc-v4-1b-{S}-grad",
        "FastConformer-CTC (streaming)": f"fastconformer-stream-ctc-{S}-grad",
        "FastConformer-RNN-T (streaming)": f"fastconformer-stream-rnnt-{S}-grad",
    }
    # Models with no word-level own-recognition -> hyp-mode is structurally n/a (not "unrun"):
    # MMS_FA (Wav2Vec2-CTC) + Phoneme-CTC emit no word boundaries; MFA is a forced-aligner, not a recognizer.
    HYP_NA = {"Wav2Vec2-CTC", "Phoneme-CTC", "MFA (GMM-HMM)"}

    # Rows grouped by model family; the leftmost (row-label) column is the family Type, shown once
    # per family; Model is the second column, shown once per model; double rule between families.
    TYPE = {
        "Wav2Vec2-CTC": "CTC",
        "Phoneme-CTC": "CTC",
        "Nvidia CTC": "CTC",
        "OWSM-CTC": "CTC",
        "FastConformer-CTC (streaming)": "CTC",
        "Whisper-base": "AED",
        "Whisper-large-v3": "AED",
        "Parakeet RNN-T": "Transd.",
        "Parakeet TDT": "Transd.",
        "Emformer (streaming)": "Transd.",
        "FastConformer-RNN-T (streaming)": "Transd.",
        "Voxtral": "Sp. LLM",
        "Phi-4-MM": "Sp. LLM",
        "Canary-Qwen": "Sp. LLM",
    }
    TYPE_ORDER = ["CTC", "AED", "Transd.", "Sp. LLM"]
    by_type = {t: [] for t in TYPE_ORDER}
    for model, methods in MODELS:
        by_type[TYPE[model]].append((model, methods))
    blocks = [(t, by_type[t]) for t in TYPE_ORDER]
    blocks.append(("Ref.", REFERENCE))  # MFA dedicated-aligner reference, its own block

    columns = ["model", "method", "t_wbe", "t_a50", "t_a100", "s_wbe", "s_a50", "s_a100"]
    if with_hyp:
        columns += ["h_mwbe", "h_f50"]

    rows = []
    for bi, (typ, fam_models) in enumerate(blocks):
        if bi > 0:
            rows.append({"hline2": True})  # double rule between families
        for mi2, (model, methods) in enumerate(fam_models):
            if mi2 > 0:
                rows.append({"label": None, "cells": {}})  # \hline between models in a family
            for mj, (mlabel, sa, ti) in enumerate(methods):
                if mj > 0:
                    rows.append({"cline": True})  # \cline between a model's own methods
                cells = {
                    "model": model if mj == 0 else "",
                    "method": mlabel,
                    "t_wbe": _wbe(ti) if ti else None,
                    "t_a50": _metric(ti, "acc_50ms") if ti else None,
                    "t_a100": _metric(ti, "acc_100ms") if ti else None,
                    "s_wbe": _wbe(sa) if sa else None,
                    "s_a50": _metric(sa, "acc_50ms") if sa else None,
                    "s_a100": _metric(sa, "acc_100ms") if sa else None,
                }
                if with_hyp:
                    hn = HYP.get(model) if mj == 0 else None  # hyp only on the grad row
                    if hn:
                        cells["h_mwbe"] = _hyp(hn, "matched_wbe")
                        cells["h_f50"] = _hyp(hn, "f1_50ms")
                    elif mj == 0 and model in HYP_NA:
                        # structurally no word-level own-recognition -> mark, don't leave empty
                        cells["h_mwbe"] = "n/a"
                        cells["h_f50"] = "n/a"
                label = typ if (mi2 == 0 and mj == 0) else ""
                rows.append({"label": label, "cells": cells})
    _emit("per-model-" + ("merged" if with_hyp else "methods"), columns, rows)


# ----------------------------------------------------------------------------------------
# T6: hypothesis-mode -- each model aligns its OWN recognition. Identity-gated F1 + matched-WBE.
# ----------------------------------------------------------------------------------------
def _hyp_table():
    S = "buckeye-segA-5h"

    def _row(label, name):
        return {
            "label": label,
            "cells": {
                "mwbe": _hyp(name, "matched_wbe"),
                "f50": _hyp(name, "f1_50ms"),
                "f100": _hyp(name, "f1_100ms"),
                "f200": _hyp(name, "f1_200ms"),
                "rm": _hyp(name, "match_rate_ref"),
            },
        }

    grad = [
        ("Whisper-base", f"whisper-base-charlev-{S}-grad"),
        ("Whisper-large-v3", f"whisper-large-v3-charlev-{S}-grad"),
        ("Voxtral", f"voxtral-charlevlogmel-{S}-grad"),
        ("Phi-4-MM", f"phi4mm-charlev-spc-{S}-grad"),
        ("Canary-Qwen", f"canary-qwen-charlev-spc-logmel-st15-{S}-grad"),
        ("Parakeet RNN-T", f"parakeet-rnnt-1.1b-logmel-{S}-grad"),
        ("Parakeet TDT", f"parakeet-tdt-0.6b-v2-logmel-{S}-grad"),
    ]
    baselines = [
        ("Whisper cross-attn DTW", f"whisper-crossattn-{S}"),
        ("CrisperWhisper (official)", f"crisperwhisper-official-{S}"),
    ]
    columns = ["mwbe", "f50", "f100", "f200", "rm"]
    rows = [_row(*m) for m in grad]
    rows.append({"label": None, "cells": {}})
    rows += [_row(*b) for b in baselines]
    _emit("hyp", columns, rows)


# ----------------------------------------------------------------------------------------
# T7: OWSM-CTC grad-align emitted from each inter-CTC self-conditioning block (side table).
# ----------------------------------------------------------------------------------------
def _owsm_layer_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def _tag(layer):
        # blocks 6/12/15/21 carry an `lyr{N}-` tag; block 27 is the default final-emission base.
        return "" if layer == 27 else f"lyr{layer}-"

    def gbase(layer, ds):  # grad-align output base
        return f"align/owsm-ctc-v4-1b-{_tag(layer)}{ds}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    def fbase(layer, ds):  # forced-align (posteriors) baseline base
        return f"baseline-owsm-ctc-v4-1b-{_tag(layer)}{ds}"

    columns = ["t_grad", "t_fa", "s_grad", "s_fa"]
    rows = []
    for layer in [6, 12, 15, 21, 27]:
        rows.append(
            {
                "label": f"{layer} (final)" if layer == 27 else f"{layer}",
                "cells": {
                    "t_grad": _wbe(gbase(layer, T)),
                    "t_fa": _wbe(fbase(layer, T)),
                    "s_grad": _wbe(gbase(layer, S)),
                    "s_fa": _wbe(fbase(layer, S)),
                },
            }
        )
    _emit("owsm-per-layer", columns, rows)


# ----------------------------------------------------------------------------------------
# T8: Speech-LLM prompt-splice sensitivity (Phi-4-MM): grad-align WBE vs the instruction.
# ----------------------------------------------------------------------------------------
def _phi4_prompt_table():
    def base(tag, level):
        return f"align/phi4mm-prompt-{tag}-{level}-timit-test-L2_e_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    columns = ["sub", "char"]
    prompts = [
        ("default", "into text (neutral)"),
        ("verbatim", "verbatim, exactly as spoken"),
        ("word", "exactly, word for word"),
        ("char", "at the character level"),
    ]
    rows = []
    for tag, label in prompts:
        rows.append(
            {
                "label": label,
                "cells": {
                    "sub": "n/a" if tag == "char" else _wbe(base(tag, "subword")),
                    "char": _wbe(base(tag, "char")),
                },
            }
        )
    _emit("phi4-prompt", columns, rows)


# ----------------------------------------------------------------------------------------
# T5: cost -- forward vs batched grad-backward, per model (same GPU).
# ----------------------------------------------------------------------------------------
def _cost_table():
    m = _RESULTS.get("cost-benchmark-metrics.txt")

    def cell(name, metric):
        return {"var": m, "key": f"{name}|{metric}", "src": "cost-benchmark-metrics.txt"}

    columns = ["fwd", "bwd", "tot"]
    models = [
        "Wav2Vec2-CTC",
        "Nvidia CTC",
        "OWSM-CTC",
        "Whisper-base",
        "Parakeet RNN-T",
        "Parakeet TDT",
        "Voxtral",
        "Phi-4-MM",
        "Canary-Qwen",
    ]
    rows = [
        {
            "label": name,
            "cells": {
                "fwd": cell(name, "fwd_ms_per_s"),
                "bwd": cell(name, "bwd_ms_per_s"),
                "tot": cell(name, "total_ms_per_s"),
            },
        }
        for name in models
    ]
    _emit("cost", columns, rows)
