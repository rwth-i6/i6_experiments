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
_CAPTURE = None  # when a list, _emit captures (name, columns, rows, source) -- preview mode

# Streaming model display names carry a forced LaTeX line break
# before "(streaming)",
# so the long name wraps onto two lines;
# render_tables wraps any line-break-bearing label/cell in a makecell.
_M_EMFORMER = "Emformer\\\\(streaming)"
_M_FC_CTC = "FastConformer-CTC\\\\(streaming)"
_M_FC_RNNT = "FastConformer-RNN-T\\\\(streaming)"


def build_tables(results):
    global _RESULTS
    _RESULTS = results
    _compare_table()  # ground-truth forced-mode only (separate hyp table below)
    _compare_table(with_hyp=True)  # variant: hyp-mode columns merged onto the grad rows
    _hyp_table()
    _owsm_layer_table()  # OWSM grad-align per inter-CTC emit block (side table)
    _prompt_splice_table()  # generalized prompt-splice: 3 LLMs x grad + self-attn (Buckeye-segA)
    _streaming_offset_table()  # streaming start/end signed offset (grad vs native viterbi)
    _time_stretch_table()  # length robustness (grad vs MMS-FA, vocoder vs resample)
    _word_length_table()  # word-duration accuracy (signed word-width error) per model
    _matched_tok_table()  # matched-tokenization: grad vs cross-attn x char/subword (whisper)
    _ablation_table()  # T3a grad-score reduction (cross-model)
    _attribution_table()  # T3b attribution method (cross-model, fast models)
    _alignopts_silence_table()  # T3b-i align-opts: silence/topology axis (grad-only silence fix)
    _alignopts_dtw_table()  # T3b-ii align-opts: apply-log/DTW equivalence (grad vs cross-attn)


def _emit(name, columns, rows):
    # `source` records this builder (file :: function) for a provenance comment in the rendered table;
    # the per-cell `src` (the registered output name) is what actually pins each number's origin.
    source = f"{__name__.replace('.', '/')}.py :: {sys._getframe(1).f_code.co_name}"
    if _CAPTURE is not None:  # preview mode: capture the definition instead of emitting a job
        _CAPTURE.append((name, columns, rows, source))
        return
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
# Streaming signed-offset: per streaming model, grad vs native viterbi,
#     SIGNED start/end boundary offset (pred - ref, ms; + = late).
#     Native chunked viterbi starts late / ends early (emission delay); grad ~0.
#     Data = start_signed_mean / end_signed_mean (align_metrics out_metrics).
# ----------------------------------------------------------------------------------------
def _streaming_offset_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}")

    MODELS = [
        (
            _M_EMFORMER,
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
            _M_FC_CTC,
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
            _M_FC_RNNT,
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
# Length robustness redesigned (Buckeye-segA, vocoder time-stretch, factors 0.5..3.0).
#     The same mechanism, opposite sign across the model's grad-resolution:
#     wav2vec2-CTC grad (fine grid) only degrades with stretch (robustness limit),
#     Voxtral grad (coarse projected-embedding grid) IMPROVES at mild stretch (upsamples the grid),
#     MMS-FA forced-align re-runs Viterbi on the stretched audio, so it stays flat (reference).
# ----------------------------------------------------------------------------------------
def _time_stretch_table():
    S = "buckeye-segA-5h"
    AO = "asotTrue-bs-5-en0.5-sil1.0"
    FACTORS = [
        ("0.5", "f05"),
        ("0.75", "f075"),
        ("1.0", "f10"),
        ("1.2", "f12"),
        ("1.5", "f15"),
        ("2.0", "f20"),
        ("3.0", "f30"),
    ]
    # (model label, base-name builder from the factor string).
    MODELS = [
        ("Wav2Vec2 (grad, fine)", lambda ts: f"align/wav2vec2ctc-fproj_out-{S}-L2_grad-pertoken-ts{ts}-{AO}"),
        ("Voxtral (grad, coarse)", lambda ts: f"align/voxtral-transcribe-{S}-L2_e_grad-pertoken-ts{ts}-{AO}"),
        ("MMS-FA (forced)", lambda ts: f"baseline-mms_fa-{S}-ts{ts}"),
    ]
    columns = [k for _, k in FACTORS]
    rows = [
        {
            "label": mlabel,
            "cells": {k: _wbe(basefn(ts)) for ts, k in FACTORS},
        }
        for mlabel, basefn in MODELS
    ]
    _emit("time-stretch", columns, rows)


# ----------------------------------------------------------------------------------------
# Word-length (duration) accuracy, full breakdown (Buckeye-segA):
#     WBE (mean abs boundary error),
#     signed start/end offset (pred - ref; + = late),
#     and word-WIDTH signed error (= end - start; 0 = correct duration).
#     The CTC/forced-align posteriors bias start LATE + end EARLY
#     -> the word shrinks from both ends, so |width| ~ WBE;
#     grad shifts start+end the SAME direction (a positional lead, not a width change),
#     so width is far below WBE -> grad is more accurate on word duration.
#     TIMIT-test shows the same pattern, milder (grad's lead is largest on spontaneous Buckeye).
# ----------------------------------------------------------------------------------------
def _word_length_table():
    S = "buckeye-segA-5h"
    MODELS = [
        (
            "Wav2Vec2",
            [
                ("Gradients", f"align/wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("Posteriors", f"baseline-mms_fa-{S}"),
            ],
        ),
        (
            "Phoneme",
            [
                (
                    "Gradients",
                    f"align/phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                ),
                ("Posteriors", f"baseline-phoneme-fa-{S}-word"),
            ],
        ),
        (
            "Nvidia",
            [
                ("Gradients", f"align/parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"),
                ("Posteriors", f"baseline-parakeet-ctc-1.1b-{S}"),
            ],
        ),
        (
            "OWSM",
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
# T2 fairness 2x2: whisper grad vs cross-attn DTW, char vs subword.
#     Method-consistent -- both use the PLAIN cross-attn forced align (no head auto-selection),
#     so the only variable per column is the tokenization.
#     Grad wins every cell; char helps grad; char collapses cross-attn on spontaneous Buckeye.
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
# T3a grad-score ablation (cross-model, Buckeye-segA, word-topology fixed).
#     Two orthogonal axes:
#     the feature-axis reduction (L0.5 / L1 / L2 / dot = signed sum)
#     and mult_grad_by_inputs (plain gradient vs gradient x input, the "x input" group).
#     A few ms apart -> neither choice is critical; the saliency signal, not the norm, carries it.
#     The signed "dot" scores align with apply_log off (a signed score has no meaningful log).
# ----------------------------------------------------------------------------------------
def _ablation_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2", "wav2vec2ctc-fproj_out-prefixfwd"),
        ("Nvidia", "parakeet-ctc-1.1b-prefixfwd"),
        ("Whisper-base", "whisper-base-logmel-charlev-spc"),
        ("Parakeet", "parakeet-rnnt-1.1b-logmel"),
        (_M_EMFORMER, "emformer-rnnt-prefix-logmel"),
        ("Voxtral", "voxtral-charlevlogmel"),
    ]
    # Signed "dot"/"dot_e" scores align with apply_log off (alFalse); same word-topology DP otherwise.
    DOT_SFX = "nsabsmeanS-cs1e-05_None-asotTrue-bs-6"
    # (column key, align suffix). Plain-gradient group then the gradient-x-input ("_e") group.
    COLS = [
        ("L0.5_grad", SFX),
        ("L1_grad", SFX),
        ("L2_grad", SFX),
        ("dot_grad", DOT_SFX),
        ("L2_e_grad", SFX),
        ("dot_e_grad", DOT_SFX),
    ]
    columns = [c for c, _ in COLS]
    # The signed-sum ("dot") DP collapses to a degenerate alignment on some families (CTC/transducer):
    # WBE far above any real result, opt-independent (the no-log signed DP fails; verified at the HDF level).
    # Don't hardcode which -- point every cell at its WBE; the renderer flags WBE > broken_above as "broken".
    rows = [
        {
            "label": mlabel,
            "cells": {c: _wbe(f"align/{mpre}-abl-{S}-{c}-pertoken-{sfx}") for c, sfx in COLS},
        }
        for mlabel, mpre in MODELS
    ]
    _emit("grad-score-ablation", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b attribution-method ablation (cross-model, Buckeye-segA, word-topology fixed):
#     plain L2 input-grad vs SmoothGrad / VarGrad / Integrated-Gradients / Expected-Gradients,
#     on the two fast models.
#     The single-pass gradient is already a good attribution
#     -> the multi-pass methods are not worth it
#     (the slow families are omitted for the same cost reason).
# ----------------------------------------------------------------------------------------
def _attribution_table():
    S = "buckeye-segA-5h"
    SFX = "asotTrue-bs-5-en0.5-sil1.0-wordtopo"
    MODELS = [
        ("Wav2Vec2", "wav2vec2ctc-fproj_out-prefixfwd"),
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
# T3b-i align-OPTS, silence & topology (Buckeye-segA, whisper grad-char).
#     Grouped "Blank scoring" {Type | Opts} + Topology;
#     grad-only (attention has ~no mass in silence).
# ----------------------------------------------------------------------------------------
def _alignopts_silence_table():
    GP = "align/whisper-base-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc-asotTrue-bs-5"
    # Blank-scoring scheme (Type + its parameter Opts) x topology;
    # energy token-weighting (en0.5) held constant.
    # constant = the fixed blank_score gamma baseline (no energy silence);
    # energy = energy-aware silence blank; z-score = grad z-score blank.
    # (type, opts, topology, grad suffix)
    # (type, opts, topology, grad suffix). Topology is an orthogonal axis, shown as its own column.
    ROWS = [
        ("constant", "$\\gamma = -5$", "CTC", "-en0.5"),
        ("energy", "$s = 1$", "CTC", "-en0.5-sil1.0"),
        ("energy", "$s = 2$", "CTC", "-en0.5-sil2.0"),
        ("z-score", "$\\kappa = 1$", "CTC", "-en0.5-zsk1.0"),
        ("energy", "$s = 1$", "word", "-en0.5-sil1.0-wordtopo"),
    ]
    columns = ["type", "opts", "topo", "g_wbe", "g_a50"]
    rows = []
    for typ, opts, topo, gs in ROWS:
        gb = GP + gs
        rows.append(
            {
                "label": "",
                "cells": {
                    "type": typ,
                    "opts": opts,
                    "topo": topo,
                    "g_wbe": _wbe(gb),
                    "g_a50": _metric(gb, "acc_50ms"),
                },
            }
        )
    _emit("alignopts-silence", columns, rows)


# ----------------------------------------------------------------------------------------
# T3b-ii align-OPTS, apply-log / DTW equivalence
#     (Buckeye-segA, whisper grad-char vs cross-attn).
#     Boolean option columns first (checkmark = on), then a short note, then the numbers.
#     Blank state off = a monotonic DTW;
#     additionally log off = the exact OpenAI Whisper timestamp DTW (bottom row),
#     where grad and cross-attn converge.
# ----------------------------------------------------------------------------------------
def _alignopts_dtw_table():
    GP = "align/whisper-base-logmel-buckeye-segA-5h-L2_grad-pertoken-charlev-spc-asotTrue-bs-5"
    AP = "align/baseline-whisper-base-crossattn-auto-buckeye-segA-5h-asotTrue-bs-5"
    ck, cx = "\\checkmark", "$\\times$"  # boolean option cell: on vs off
    # (apply-log, blank state, note, grad suffix, cross-attn suffix). Option columns first, note last.
    ROWS = [
        (ck, ck, "our standard", "-en0.5-sil1.0", "-en0.5-sil1.0"),
        (cx, ck, "", "-alFalse-en0.5-sil1.0", "-alFalse-en0.5-sil1.0"),
        (ck, cx, "", "-en0.5-sil1.0-dtw", "-en0.5-sil1.0-dtw"),
        (cx, cx, "OpenAI Whisper", "-en0.5-sil1.0-wdtw", "-en0.5-sil1.0-wdtw"),
    ]
    columns = ["applylog", "blank", "note", "g_wbe", "g_a50", "a_wbe", "a_a50"]
    rows = []
    for applylog, blank, note, gs, as_ in ROWS:
        gb, ab = GP + gs, AP + as_
        rows.append(
            {
                "label": "",
                "cells": {
                    "applylog": applylog,
                    "blank": blank,
                    "note": note,
                    "g_wbe": _wbe(gb),
                    "g_a50": _metric(gb, "acc_50ms"),
                    "a_wbe": _wbe(ab),
                    "a_a50": _metric(ab, "acc_50ms"),
                },
            }
        )
    _emit("alignopts-dtw", columns, rows)


# ----------------------------------------------------------------------------------------
# T1: per-model grad-align vs the model's own alternative aligner (same model, different method).
#     Rows grouped by model; columns = align method + per-dataset metrics (TIMIT, Buckeye).
# ----------------------------------------------------------------------------------------
def _compare_table(with_hyp=False):
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):  # grad-align align/ output bases (segA, TIMIT)
        return (f"align/{stem_sa}", f"align/{stem_ti}" if stem_ti else None)

    # (Model, [(align-method label, segA base, TIMIT base), ...])  -- grad first, then its alternative.
    # The method label names the alignment SIGNAL:
    # grad / posteriors (CTC, transducer, GMM-HMM) / cross-attn weights / self-attn weights.
    MODELS = [
        (
            "Wav2Vec2",
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
            "Phoneme",
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
            "Nvidia",
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
            # OWSM: general graphemic CTC, our least-favourable model for grad-align (still runs,
            # just imprecise). Shown with its BEST emit block (6); per-block detail in tab:owsm-per-layer.
            "OWSM",
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
            "Parakeet",
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
            _M_EMFORMER,
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
            _M_FC_CTC,
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
            _M_FC_RNNT,
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
        ("MFA", [("Likelihoods", f"baseline-mfa-{S}", f"baseline-mfa-{T}")]),
    ]

    # Hyp-mode (each model aligns its OWN recognition, Buckeye only) -> grad rows only, where it exists.
    HYP = {
        "Whisper-base": f"whisper-base-charlev-{S}-grad",
        "Voxtral": f"voxtral-charlevlogmel-{S}-grad",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-grad",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-grad",
        "Parakeet": f"parakeet-rnnt-1.1b-logmel-{S}-grad",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-grad",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-grad",
        "Nvidia": f"parakeet-ctc-1.1b-{S}-grad",
        "OWSM": f"owsm-ctc-v4-1b-{S}-grad",
        _M_FC_CTC: f"fastconformer-stream-ctc-{S}-grad",
        _M_FC_RNNT: f"fastconformer-stream-rnnt-{S}-grad",
        _M_EMFORMER: f"emformer-rnnt-prefix-logmel-{S}-grad",
    }
    # Models with no word-level own-recognition -> hyp-mode is structurally n/a (not "unrun"):
    # MMS_FA (Wav2Vec2) + Phoneme emit no word boundaries;
    # MFA is a forced-aligner, not a recognizer.
    HYP_NA = {"Wav2Vec2", "Phoneme", "MFA"}

    # Native aligner in hyp-mode (own recognition), shown on the model's ALTERNATIVE row:
    # cross-attn (whisper), self-attn (LLMs), native viterbi (transducers), CTC forced-align (CTCs).
    # whisper-base keeps its original whisper-crossattn alias; the CTC entries fill once their
    # forced-align job emits a boundary HDF (pending), so they render empty until then.
    NATIVE_HYP = {
        "Whisper-base": f"whisper-crossattn-{S}",
        "Whisper-large-v3": f"whisper-large-v3-charlev-{S}-native",
        "Voxtral": f"voxtral-charlevlogmel-{S}-native",
        "Phi-4-MM": f"phi4mm-charlev-spc-{S}-native",
        "Canary-Qwen": f"canary-qwen-charlev-spc-logmel-st15-{S}-native",
        "Parakeet": f"parakeet-rnnt-1.1b-logmel-{S}-native",
        "Parakeet TDT": f"parakeet-tdt-0.6b-v2-logmel-{S}-native",
        _M_EMFORMER: f"emformer-rnnt-prefix-logmel-{S}-native",
        _M_FC_RNNT: f"fastconformer-stream-rnnt-{S}-native",
        "Nvidia": f"parakeet-ctc-1.1b-{S}-native",
        "OWSM": f"owsm-ctc-v4-1b-{S}-native",
        _M_FC_CTC: f"fastconformer-stream-ctc-{S}-native",
    }

    # Rows grouped by model family; the leftmost (row-label) column is the family Type, shown once
    # per family; Model is the second column, shown once per model; double rule between families.
    TYPE = {
        "Wav2Vec2": "CTC",
        "Phoneme": "CTC",
        "Nvidia": "CTC",
        "OWSM": "CTC",
        _M_FC_CTC: "CTC",
        "Whisper-base": "AED",
        "Whisper-large-v3": "AED",
        "Parakeet": "Transd.",
        "Parakeet TDT": "Transd.",
        _M_EMFORMER: "Transd.",
        _M_FC_RNNT: "Transd.",
        "Voxtral": "Sp. LLM",
        "Phi-4-MM": "Sp. LLM",
        "Canary-Qwen": "Sp. LLM",
    }
    TYPE_ORDER = ["CTC", "AED", "Transd.", "Sp. LLM"]
    by_type = {t: [] for t in TYPE_ORDER}
    for model, methods in MODELS:
        by_type[TYPE[model]].append((model, methods))
    blocks = [(t, by_type[t]) for t in TYPE_ORDER]
    blocks.append(("GMM-HMM", REFERENCE))  # MFA dedicated-aligner reference (GMM-HMM), its own block

    columns = ["model", "method", "t_wbe", "t_a50", "t_a100", "s_wbe", "s_a50", "s_a100"]
    if with_hyp:
        columns += ["h_mwbe", "h_f50"]

    _FC_DISP = _M_EMFORMER.replace("Emformer", "FastConformer")
    # Display names drop the redundant family postfix (the Type column already shows CTC/Transd./...),
    # keeping an architecture suffix only where same-family siblings would otherwise collide
    # (Parakeet RNN-T vs Parakeet TDT; the two FastConformer heads).
    # The dict KEY stays the full name (TYPE/HYP/NATIVE_HYP are keyed by it); only the shown label changes.
    DISPLAY = {"Parakeet": "Parakeet RNN-T", _M_FC_CTC: _FC_DISP, _M_FC_RNNT: _FC_DISP}
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
                    "model": DISPLAY.get(model, model) if mj == 0 else "",
                    "method": mlabel,
                    "t_wbe": _wbe(ti) if ti else None,
                    "t_a50": _metric(ti, "acc_50ms") if ti else None,
                    "t_a100": _metric(ti, "acc_100ms") if ti else None,
                    "s_wbe": _wbe(sa) if sa else None,
                    "s_a50": _metric(sa, "acc_50ms") if sa else None,
                    "s_a100": _metric(sa, "acc_100ms") if sa else None,
                }
                if with_hyp:
                    # grad hyp on the gradients row (mj==0); the model's native-aligner hyp on its
                    # alternative row (mj>0) -- each method aligns the model's OWN recognition.
                    hn = HYP.get(model) if mj == 0 else NATIVE_HYP.get(model)
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
        ("Parakeet", f"parakeet-rnnt-1.1b-logmel-{S}-grad"),
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
# T7: OWSM grad-align emitted from each inter-CTC self-conditioning block (side table).
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
# T8b: prompt-splice sensitivity generalized across the speech LLMs, grad AND self-attn.
#     Rows = the 4 spliced instructions;
#     columns grouped per model into Grad / Self-att. (char-level, Buckeye-segA).
#     Shows prompt-robustness holds across models and across the grad-vs-attention signal.
# ----------------------------------------------------------------------------------------
def _prompt_splice_table():
    S = "buckeye-segA-5h"
    AO = "asotTrue-bs-5-en0.5-sil1.0"
    MODELS = [("phi4mm", "Phi-4-MM"), ("voxtral", "Voxtral"), ("canary-qwen", "Canary-Qwen")]
    PROMPTS = [
        ("default", "neutral (into text)"),
        ("verbatim", "verbatim, as spoken"),
        ("word", "exactly, word for word"),
        ("char", "at the character level"),
    ]
    columns = []
    for mk, _ in MODELS:
        columns += [f"{mk}_g", f"{mk}_a"]
    rows = []
    for tag, label in PROMPTS:
        cells = {}
        for mk, _ in MODELS:
            cells[f"{mk}_g"] = _wbe(f"align/{mk}-promptsplice-{tag}-char-{S}-grad-pertoken-{AO}")
            cells[f"{mk}_a"] = _wbe(f"align/baseline-{mk}-promptsplice-{tag}-selfattn-{S}-{AO}")
        rows.append({"label": label, "cells": cells})
    _emit("prompt-splice", columns, rows)


def build_preview_tables(results):
    """Write per-table preview MANIFESTS to output/tables-data-preview/<name>.manifest.pkl.

    Each gzip-pickles (columns, rows, source); the rows carry the real cell Variables, so the manifest
    holds everything needed to resolve the numbers later. Written once per config load (this runs in py()).
    The numbers are NOT resolved here -- jobs/table_data.py --refresh-preview output/tables-data-preview
    re-resolves them from current disk state any time, with no manager restart
    (scripts/sync_tables_preview.sh runs that refresher before rendering), so a pending cell never blocks."""
    import os
    from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data import (
        write_preview_manifest,
    )

    global _CAPTURE
    _CAPTURE = []
    try:
        build_tables(results)  # same builders, but _emit captures the defs instead of emitting jobs
        defs = list(_CAPTURE)
    finally:
        _CAPTURE = None

    out_dir = "output/tables-data-preview"
    os.makedirs(out_dir, exist_ok=True)
    # Drop snapshots of renamed/removed tables so the preview dir holds only the current set.
    current = {name for name, *_ in defs}
    for fn in os.listdir(out_dir):
        if fn.endswith(".manifest.pkl") and fn[: -len(".manifest.pkl")] not in current:
            os.remove(os.path.join(out_dir, fn))
        elif fn.endswith(".data.json") and fn[: -len(".data.json")] not in current:
            os.remove(os.path.join(out_dir, fn))
    for name, columns, rows, source in defs:
        write_preview_manifest(name, columns, rows, source, out_dir)
