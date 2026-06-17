"""Auto-generated LaTeX result tables, wired into the Sis graph.

``build_tables(results)`` is called at the end of the main recipe. ``results`` is the
``{registered-output-name: Variable}`` dict captured by the module-level ``reg`` wrapper during
``py()`` (sis_graph is NOT introspectable mid-py()). For each paper table it looks up the relevant
metric outputs and emits one :class:`WriteLatexTableJob`, registered under ``output/tables/``.

Lookup is by registered output NAME (this recipe's, never stale on-disk aliases). A "<base>-wbe.txt"
output IS the scalar out_wbe Variable; its ``.creator`` is the metric job (whose ``out_metrics``
holds the full dict incl. acc@collar). So WBE works for every aligner; acc needs out_metrics.
"""

from sisyphus import tk
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.latex_table import WriteLatexTableJob

_RESULTS = {}


def build_tables(results):
    global _RESULTS
    _RESULTS = results
    _headline_table()  # T1 breadth: grad WBE + acc@50, main models + cross-attn DTW baseline
    _compare_table()  # ground-truth forced-mode only (separate hyp table below)
    _compare_table(with_hyp=True)  # variant: hyp-mode columns merged onto the grad rows
    _hyp_table()
    _owsm_layer_table()  # OWSM-CTC grad-align per inter-CTC emit block (side table)
    _phi4_prompt_table()  # Phi-4-MM grad-align vs the spliced instruction (side table)
    _cost_table()  # forward vs batched grad-backward per model (T5 cost)
    _streaming_offset_table()  # streaming start/end signed offset (grad vs native viterbi)
    _fairness_table()  # T2 fairness 2x2 (grad vs cross-attn x char/subword)


# ----------------------------------------------------------------------------------------
# Streaming signed-offset: per streaming model, grad vs native viterbi, SIGNED start/end boundary
#     offset (pred - ref, ms; + = late). Native chunked viterbi starts late / ends early (emission
#     delay); grad ~0. Data = start_signed_mean / end_signed_mean (align_metrics out_metrics).
# ----------------------------------------------------------------------------------------
def _streaming_offset_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def _off(base, key):
        v = _RESULTS.get(f"{base}-wbe.txt")
        j = getattr(v, "creator", None) if v is not None else None
        m = getattr(j, "out_metrics", None) if j is not None else None
        if m is None and j is not None:
            # CTC forced-align baseline (ParakeetCtcForcedAlignJob) names its metrics dict out_word_metrics.
            m = getattr(j, "out_word_metrics", None)
        return {"var": m, "key": key, "mul": 1000.0, "fmt": "{:+.0f}"}

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}")

    MODELS = [
        (
            "Emformer RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "native viterbi",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("native viterbi", f"baseline-fastconformer-stream-ctc-{S}", f"baseline-fastconformer-stream-ctc-{T}"),
            ],
        ),
        (
            "FastConformer-RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "native viterbi",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
    ]

    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "t_start", "header": "start \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_end", "header": "end \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "s_start", "header": "start \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_end", "header": "end \\\\ {[ms]}", "group": "Buckeye"},
    ]

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
                        "t_start": _off(ti, "start_signed_mean"),
                        "t_end": _off(ti, "end_signed_mean"),
                        "s_start": _off(sa, "start_signed_mean"),
                        "s_end": _off(sa, "end_signed_mean"),
                    },
                }
            )

    caption = (
        "Signed word-boundary offset (predicted minus reference, ms; positive = late) for the streaming "
        "models, grad-align vs the model's native chunked-viterbi alignment, on TIMIT-test and Buckeye-segA. "
        "The native streaming alignment is systematically late on both boundaries (limited right-context "
        "emission delay; the start is delayed more than the end), whereas grad-align is far closer to "
        "unbiased (tens vs hundreds of ms). This signed view complements the absolute WBE: the "
        "native streaming viterbi's large WBE is largely a systematic emission-delay bias."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:streaming-offset",
        label_header="Model",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/streaming-offset.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T2 fairness 2x2: whisper grad vs cross-attn DTW, char vs subword. Method-consistent -- both use the
#     PLAIN cross-attn forced align (no head auto-selection), so the only variable per column is the
#     tokenization. Grad wins every cell; char helps grad; char collapses cross-attn on spontaneous Buckeye.
# ----------------------------------------------------------------------------------------
def _fairness_table():
    def gA(stem):
        return f"align/{stem}"

    DATASETS = [("TIMIT", "timit-test"), ("Buckeye", "buckeye-segA-5h")]
    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "c_wbe", "header": "WBE \\\\ {[ms]}", "group": "char"},
        {"key": "c_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "char"},
        {"key": "s_wbe", "header": "WBE \\\\ {[ms]}", "group": "subword"},
        {"key": "s_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "subword"},
    ]
    rows = []
    for di, (dlabel, ds) in enumerate(DATASETS):
        if di > 0:
            rows.append({"label": None, "cells": {}})
        gc = gA(f"whisper-base-logmel-{ds}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0")
        gs = gA(f"whisper-base-logmel-{ds}-L2_grad-pertoken-subword-asotTrue-bs-5-en0.5-sil1.0")
        xc = f"baseline-whisper-crossattn-charlev-{ds}"
        xs = f"baseline-whisper-crossattn-{ds}"
        rows.append(
            {
                "label": dlabel,
                "cells": {
                    "method": "grad",
                    "c_wbe": _wbe(gc),
                    "c_a50": _acc(gc, "acc_50ms"),
                    "s_wbe": _wbe(gs),
                    "s_a50": _acc(gs, "acc_50ms"),
                },
            }
        )
        rows.append({"cline": True})
        rows.append(
            {
                "label": "",
                "cells": {
                    "method": "cross-attn",
                    "c_wbe": _wbe(xc),
                    "c_a50": _acc(xc, "acc_50ms"),
                    "s_wbe": _wbe(xs),
                    "s_a50": _acc(xs, "acc_50ms"),
                },
            }
        )

    caption = (
        "Fairness 2x2 on whisper-base: gradient alignment vs cross-attention DTW, at character-level and "
        "subword tokenization (TIMIT-test, Buckeye-segA; WBE and acc@50). Both methods use the same plain "
        "cross-attention forced alignment (no head auto-selection), so the only variable per column pair is "
        "the tokenization. Grad-align wins every cell. Character targets help grad on both datasets; they "
        "help cross-attention on clean TIMIT but collapse it on spontaneous Buckeye (264.6 ms), where char "
        "tokens exceed the 80 ms attention grid. (The headline table uses the stronger head-auto-selected "
        "cross-attention variant for its subword baseline.)"
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:fairness",
        label_header="Data",
        col_align="|l|l|r|r|r|r|",
    )
    tk.register_output("tables/fairness.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T3 ablation (whisper-char, Buckeye-segA): grad-score reduction, silence scale, apply_log on/off.
#     grad-score variants are ~2 ms apart; apply_log-off and silence-off are the only large moves.
# ----------------------------------------------------------------------------------------
def _ablation_table():
    P = "align/whisper-base-logmel-buckeye-segA-5h"
    GROUPS = [
        (
            "grad-score",
            [
                ("L1", f"{P}-L1_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("L2 (default)", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("L2_e", f"{P}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
            ],
        ),
        (
            "silence scale",
            [
                ("1.0 (default)", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("0 (off)", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5"),
                ("2.0", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil2.0"),
                ("z-score k=1", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-zsk1.0"),
            ],
        ),
        (
            "apply-log",
            [
                ("on (default)", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0"),
                ("off", f"{P}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-alFalse-en0.5-sil1.0"),
            ],
        ),
    ]
    cols = [
        {"key": "setting", "header": "setting"},
        {"key": "wbe", "header": "WBE \\\\ {[ms]}"},
        {"key": "a50", "header": "acc@50 \\\\ {[\\%]}"},
    ]
    rows = []
    for gi, (axis, variants) in enumerate(GROUPS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for vi, (vlabel, base) in enumerate(variants):
            if vi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": axis if vi == 0 else "",
                    "cells": {"setting": vlabel, "wbe": _wbe(base), "a50": _acc(base, "acc_50ms")},
                }
            )
    caption = (
        "Ablation of the alignment levers on whisper-base char-level grad-align, Buckeye-segA. The grad-score "
        "reduction (L1 / L2 / L2_e) moves the result by ~2 ms; the silence scale and z-score variants are "
        "similarly small around the default. Only turning the silence row off, and especially turning "
        "apply-log off (the openai-DTW score), degrade markedly -- so the defaults are robust."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:ablation",
        label_header="Lever",
        col_align="|l|l|r|r|",
    )
    tk.register_output("tables/ablation.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T3b attribution axis (whisper-char, TIMIT-val): plain input-gradient vs SmoothGrad / VarGrad /
#     Integrated-Gradients / Expected-Gradients sweeps. IG converges back to plain grad; the noise
#     methods barely move it -- the single-pass gradient is already the good attribution.
# ----------------------------------------------------------------------------------------
def _attribution_table():
    P = "align/whisper-base-logmel-timit-val"
    ao = "asotTrue-bs-5-en0.5-sil1.0"
    GROUPS = [
        ("plain grad (L2)", [("--", f"{P}-L2_grad-pertoken-charlev-spc-{ao}")]),
        (
            "SmoothGrad",
            [
                (f"std {s}", f"{P}-L2_grad-pertoken-charlev-spc-smoothgrad-std{s}-n8-{ao}")
                for s in ("0.005", "0.01", "0.03")
            ],
        ),
        (
            "VarGrad",
            [(f"std {s}", f"{P}-L2-vargrad-pertoken-charlev-spc-std{s}-n16-{ao}") for s in ("0.01", "0.02", "0.04")],
        ),
        (
            "Integrated-Grad",
            [(f"{n} steps", f"{P}-L2-integrated-grad-pertoken-charlev-spc-ig{n}-{ao}") for n in ("8", "16", "32")],
        ),
        (
            "Expected-Grad",
            [
                (f"bstd {s}", f"{P}-L2-expected-grad-pertoken-charlev-spc-eg16-bstd{s}-{ao}")
                for s in ("0.01", "0.05", "0.1")
            ],
        ),
    ]
    cols = [
        {"key": "param", "header": "param"},
        {"key": "wbe", "header": "WBE \\\\ {[ms]}"},
        {"key": "a50", "header": "acc@50 \\\\ {[\\%]}"},
    ]
    rows = []
    for gi, (method, variants) in enumerate(GROUPS):
        if gi > 0:
            rows.append({"label": None, "cells": {}})
        for vi, (vlabel, base) in enumerate(variants):
            if vi > 0:
                rows.append({"cline": True})
            rows.append(
                {
                    "label": method if vi == 0 else "",
                    "cells": {"param": vlabel, "wbe": _wbe(base), "a50": _acc(base, "acc_50ms")},
                }
            )
    caption = (
        "Attribution-method ablation on whisper-base char-level grad-align, TIMIT-val. The plain single-pass "
        "input-gradient (L2) is compared to SmoothGrad, VarGrad, Integrated-Gradients (step sweep) and "
        "Expected-Gradients (baseline-std sweep). Integrated-Gradients converges back to the plain gradient as "
        "steps grow, and the noise-averaging variants barely move it -- the single-pass gradient is already a "
        "good attribution for alignment, so the costlier attribution methods are not worth their extra passes."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:attribution",
        label_header="Method",
        col_align="|l|l|r|r|",
    )
    tk.register_output("tables/attribution.tex", job.out_tex)


def _wbe(base):
    # The "<base>-wbe.txt" output is the scalar out_wbe (seconds); -> ms.
    return {"var": _RESULTS.get(f"{base}-wbe.txt"), "mul": 1000.0, "fmt": "{:.1f}"}


def _acc(base, key):
    v = _RESULTS.get(f"{base}-wbe.txt")
    j = getattr(v, "creator", None) if v is not None else None
    m = getattr(j, "out_metrics", None) if j is not None else None
    if m is None and j is not None:
        # ForcedAlignPhonemeBaselineJob names its (same aggregate_corpus) metrics dict out_word_metrics.
        m = getattr(j, "out_word_metrics", None)
    return {"var": m, "key": key, "mul": 100.0, "fmt": "{:.1f}"}


# ----------------------------------------------------------------------------------------
# T1 headline (breadth): grad-align WBE + acc@50 across the main models + the whisper cross-attn
#     DTW baseline. Grad bases == _compare_table grad rows (best per-model config), so the two agree.
# ----------------------------------------------------------------------------------------
def _headline_table():
    S, T = "buckeye-segA-5h", "timit-test"

    def g(stem_sa, stem_ti):
        return (f"align/{stem_sa}", f"align/{stem_ti}" if stem_ti else None)

    # (Model label, segA grad base, TIMIT grad base) -- mirror _compare_table's grad rows.
    MODELS = [
        (
            "Wav2Vec2-CTC",
            *g(
                f"wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                f"wav2vec2ctc-fproj_out-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Phoneme-CTC (G2P)",
            *g(f"phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0", None),
        ),
        (
            "Whisper-base",
            *g(
                f"whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                f"whisper-base-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Whisper-large-v3",
            *g(
                f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                f"whisper-large-v3-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Parakeet RNN-T",
            *g(
                f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Parakeet TDT",
            *g(
                f"parakeet-tdt-0.6b-v2-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                f"parakeet-tdt-0.6b-v2-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Voxtral",
            *g(
                f"voxtral-charlevlogmel-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                f"voxtral-charlevlogmel-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Phi-4-MM",
            *g(
                f"phi4mm-{S}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                f"phi4mm-{T}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
        (
            "Canary-Qwen",
            *g(
                f"canary-qwen-charlev-spc-logmel-st15-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                f"canary-qwen-charlev-spc-logmel-st15-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
            ),
        ),
    ]
    BASELINE = (
        "Whisper cross-attn DTW",
        f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0",
        f"align/baseline-whisper-base-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil1.0",
    )

    cols = [
        {"key": "t_wbe", "header": "WBE \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "TIMIT"},
        {"key": "s_wbe", "header": "WBE \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "Buckeye"},
    ]

    def row(label, sa, ti):
        return {
            "label": label,
            "cells": {
                "t_wbe": _wbe(ti) if ti else None,
                "t_a50": _acc(ti, "acc_50ms") if ti else None,
                "s_wbe": _wbe(sa) if sa else None,
                "s_a50": _acc(sa, "acc_50ms") if sa else None,
            },
        }

    rows = [row(m, sa, ti) for (m, sa, ti) in MODELS]
    rows.append({"label": None, "cells": {}})  # \\hline before the baseline
    rows.append(row(*BASELINE))

    caption = (
        "Headline breadth: gradient-based word alignment across ASR families on TIMIT-test and "
        "Buckeye-segA (word-boundary error and acc@50ms tolerance accuracy), best per-model config. "
        "The whisper cross-attention DTW baseline is shown for reference. Grad-align runs on every "
        "family; the per-model table breaks each model out against its own native / attention aligner."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:headline",
        label_header="Model",
        col_align="|l|r|r|r|r|",
    )
    tk.register_output("tables/headline.tex", job.out_tex)


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
                    "grad",
                    *g(
                        f"wav2vec2ctc-fproj_out-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"wav2vec2ctc-fproj_out-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("posteriors", f"baseline-mms_fa-{S}", f"baseline-mms_fa-{T}"),
            ],
        ),
        (
            "Phoneme-CTC",
            [
                (
                    "grad",
                    *g(
                        f"phoneme-vitouphy-prefixfwd-{S}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                        f"phoneme-vitouphy-prefixfwd-{T}-L2_grad-pertoken-g2pword-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-phoneme-fa-{S}-word",
                    "align/w2v-phoneme-timit-test-ctc-forced-align-word",
                ),
            ],
        ),
        (
            "Nvidia CTC",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-ctc-1.1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-ctc-1.1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                ("posteriors", f"baseline-parakeet-ctc-1.1b-{S}", f"baseline-parakeet-ctc-1.1b-{T}"),
            ],
        ),
        (
            # OWSM-CTC: general graphemic CTC, our least-favourable model for grad-align (still runs,
            # just imprecise). Shown with its BEST emit block (6); per-block detail in tab:owsm-per-layer.
            "OWSM-CTC",
            [
                (
                    "grad",
                    *g(
                        f"owsm-ctc-v4-1b-prefixfwd-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"owsm-ctc-v4-1b-prefixfwd-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                # posteriors = CTC forced-align of the model's emission; best/natural at the final block.
                ("posteriors", f"baseline-owsm-ctc-v4-1b-{S}", f"baseline-owsm-ctc-v4-1b-{T}"),
            ],
        ),
        (
            "Whisper-base",
            [
                (
                    "grad",
                    *g(
                        f"whisper-base-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-base-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "cross-attn weights",
                    f"align/baseline-whisper-base-crossattn-auto-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-whisper-base-crossattn-auto-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Whisper-large-v3",
            [
                (
                    "grad",
                    *g(
                        f"whisper-large-v3-logmel-{S}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"whisper-large-v3-logmel-{T}-L2_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "cross-attn weights",
                    f"baseline-whisper-large-v3-crossattn-{S}",
                    f"baseline-whisper-large-v3-crossattn-{T}",
                ),
            ],
        ),
        (
            "Parakeet RNN-T",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-rnnt-1.1b-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-rnnt-1.1b-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{S}",
                    f"baseline-parakeet-rnnt-1.1b-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Parakeet TDT",
            [
                (
                    "grad",
                    *g(
                        f"parakeet-tdt-0.6b-v2-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"parakeet-tdt-0.6b-v2-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
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
                    "grad",
                    *g(
                        f"emformer-rnnt-prefix-logmel-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"emformer-rnnt-prefix-logmel-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-emformer-rnnt-native-viterbi-{S}",
                    f"baseline-emformer-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "FastConformer-CTC (stream)",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-ctc-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-ctc-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-fastconformer-stream-ctc-{S}",
                    f"baseline-fastconformer-stream-ctc-{T}",
                ),
            ],
        ),
        (
            "FastConformer-RNN-T (stream)",
            [
                (
                    "grad",
                    *g(
                        f"fastconformer-stream-rnnt-{S}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"fastconformer-stream-rnnt-{T}-L2_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "posteriors",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{S}",
                    f"baseline-fastconformer-stream-rnnt-native-viterbi-{T}",
                ),
            ],
        ),
        (
            "Voxtral",
            [
                (
                    "grad",
                    *g(
                        f"voxtral-charlevlogmel-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"voxtral-charlevlogmel-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-voxtral-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-voxtral-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Phi-4-MM",
            [
                (
                    "grad",
                    *g(
                        f"phi4mm-{S}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                        f"phi4mm-{T}-L2_e_grad-pertoken-charlev-spc-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-phi4mm-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-phi4mm-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
        (
            "Canary-Qwen",
            [
                (
                    "grad",
                    *g(
                        f"canary-qwen-charlev-spc-logmel-st15-{S}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                        f"canary-qwen-charlev-spc-logmel-st15-{T}-L1_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0",
                    ),
                ),
                (
                    "self-attn weights",
                    f"align/baseline-canary-qwen-selfattn-{S}-asotTrue-bs-5-en0.5-sil1.0",
                    f"align/baseline-canary-qwen-selfattn-{T}-asotTrue-bs-5-en0.5-sil1.0",
                ),
            ],
        ),
    ]
    # Dedicated-aligner ceiling (model-agnostic), shown for reference.
    REFERENCE = [
        ("MFA (GMM-HMM)", [("posteriors", f"baseline-mfa-{S}", f"baseline-mfa-{T}")]),
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
    }
    # Models with no word-level own-recognition -> hyp-mode is structurally n/a (not "unrun"):
    # MMS_FA (Wav2Vec2-CTC) + Phoneme-CTC emit no word boundaries; MFA is a forced-aligner, not a recognizer.
    HYP_NA = {"Wav2Vec2-CTC", "Phoneme-CTC", "MFA (GMM-HMM)"}

    def _hm(name, key, mul, fmt):
        return {"var": _RESULTS.get(f"hyp-align/{name}-metrics.txt"), "key": key, "mul": mul, "fmt": fmt}

    # Two-level header: TIMIT / Buckeye groups, each with WBE + acc@{50,100}.
    cols = [
        {"key": "method", "header": "Align \\\\ method"},
        {"key": "t_wbe", "header": "WBE \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "TIMIT"},
        {"key": "t_a100", "header": "acc@100 \\\\ {[\\%]}", "group": "TIMIT"},
        {"key": "s_wbe", "header": "WBE \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_a50", "header": "acc@50 \\\\ {[\\%]}", "group": "Buckeye"},
        {"key": "s_a100", "header": "acc@100 \\\\ {[\\%]}", "group": "Buckeye"},
    ]
    if with_hyp:
        # hyp-mode (own-recognition) columns: matched-WBE + F1@50, on grad rows only.
        cols += [
            {"key": "h_mwbe", "header": "m-WBE \\\\ {[ms]}", "group": "Buckeye (hyp)"},
            {"key": "h_f50", "header": "F1@50", "group": "Buckeye (hyp)"},
        ]

    def _model_rows(groups, lead_hline):
        rows = []
        for gi, (model, methods) in enumerate(groups):
            if gi > 0 or lead_hline:
                rows.append({"label": None, "cells": {}})  # \hline between models
            for mi, (mlabel, sa, ti) in enumerate(methods):
                if mi > 0:
                    rows.append({"cline": True})  # \cline between a model's own methods
                cells = {
                    "method": mlabel,
                    "t_wbe": _wbe(ti) if ti else None,
                    "t_a50": _acc(ti, "acc_50ms") if ti else None,
                    "t_a100": _acc(ti, "acc_100ms") if ti else None,
                    "s_wbe": _wbe(sa) if sa else None,
                    "s_a50": _acc(sa, "acc_50ms") if sa else None,
                    "s_a100": _acc(sa, "acc_100ms") if sa else None,
                }
                if with_hyp:
                    hn = HYP.get(model) if mi == 0 else None  # hyp only on the grad row
                    if hn:
                        cells["h_mwbe"] = _hm(hn, "matched_wbe", 1000.0, "{:.1f}")
                        cells["h_f50"] = _hm(hn, "f1_50ms", 1.0, "{:.3f}")
                    elif mi == 0 and model in HYP_NA:
                        # structurally no word-level own-recognition -> mark, don't leave empty
                        cells["h_mwbe"] = "n/a"
                        cells["h_f50"] = "n/a"
                    else:
                        cells["h_mwbe"] = None
                        cells["h_f50"] = None
                rows.append({"label": model if mi == 0 else "", "cells": cells})
        return rows

    rows = _model_rows(MODELS, False) + _model_rows(REFERENCE, True)

    caption = (
        "Per-model alignment quality: gradient-based alignment vs. the model's own native / "
        "attention aligner, on TIMIT-test and Buckeye (segment-A 5h subset). "
        "The attention aligners (cross-/self-attn weights) use each model's native subword "
        "tokenization -- their best setting per align method, as the coarse attention grid cannot resolve "
        "char-level targets; grad-align and the CTC / transducer aligners use the same per-token "
        "units within each model. MFA (GMM-HMM) is a dedicated-aligner reference."
    )
    if with_hyp:
        caption += (
            " The rightmost group adds hypothesis-mode metrics (each model aligns its OWN "
            "recognition; matched-WBE and identity-gated F1@50ms), shown on the grad row. "
            "n/a = no word-level own-recognition (MMS_FA / Phoneme-CTC emit no word boundaries; "
            "MFA is a forced-aligner)."
        )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:per-model-methods" + ("-hyp" if with_hyp else ""),
        label_header="Model",
        col_align="|l|l|r|r|r|r|r|r|" + ("r|r|" if with_hyp else ""),
    )
    tk.register_output("tables/per-model-" + ("merged" if with_hyp else "methods") + ".tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T6: hypothesis-mode -- each model aligns its OWN recognition. Identity-gated F1 + matched-WBE.
# ----------------------------------------------------------------------------------------
def _hyp_table():
    S = "buckeye-segA-5h"

    def _m(name, key, mul, fmt="{:.1f}"):
        return {"var": _RESULTS.get(f"hyp-align/{name}-metrics.txt"), "key": key, "mul": mul, "fmt": fmt}

    def _row(label, name):
        return {
            "label": label,
            "cells": {
                "mwbe": _m(name, "matched_wbe", 1000.0),
                "f50": _m(name, "f1_50ms", 1.0, "{:.3f}"),
                "f100": _m(name, "f1_100ms", 1.0, "{:.3f}"),
                "f200": _m(name, "f1_200ms", 1.0, "{:.3f}"),
                "rm": _m(name, "match_rate_ref", 100.0),
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
    cols = [
        {"key": "mwbe", "header": "matched-WBE [ms]"},
        {"key": "f50", "header": "F1@50"},
        {"key": "f100", "header": "F1@100"},
        {"key": "f200", "header": "F1@200"},
        {"key": "rm", "header": "ref-match [\\%]"},
    ]
    rows = [_row(*m) for m in grad]
    rows.append({"label": None, "cells": {}})
    rows += [_row(*b) for b in baselines]

    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=(
            "Hypothesis-mode alignment on Buckeye-segA: each model aligns its OWN recognition. "
            "Identity-gated F1 at collar and Levenshtein-matched WBE against the reference."
        ),
        label="tab:hyp",
        label_header="Model",
        col_align="|l|r|r|r|r|r|",
    )
    tk.register_output("tables/hyp.tex", job.out_tex)


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

    cols = [
        {"key": "layer", "header": "Emit \\\\ block"},
        {"key": "t_grad", "header": "grad \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "t_fa", "header": "posteriors \\\\ {[ms]}", "group": "TIMIT"},
        {"key": "s_grad", "header": "grad \\\\ {[ms]}", "group": "Buckeye"},
        {"key": "s_fa", "header": "posteriors \\\\ {[ms]}", "group": "Buckeye"},
    ]
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
    caption = (
        "OWSM-CTC gradient alignment emitted from each inter-CTC self-conditioning block "
        "(6/12/15/21) vs the final block (27), on TIMIT-test and Buckeye-segA. The CTC emission "
        "forced-aligns well at every block ($\\sim$142\\,ms), but grad-align is weak at all of them "
        "-- worse than the 153\\,ms uniform-split trivial baseline, the earliest block 6 the least weak. "
        "The method still runs; this is simply our least-favourable model (see discussion)."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:owsm-per-layer",
        label_header="OWSM-CTC",
        col_align="|l|r|r|r|r|",
    )
    tk.register_output("tables/owsm-per-layer.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T8: Speech-LLM prompt-splice sensitivity (Phi-4-MM): grad-align WBE vs the instruction.
# ----------------------------------------------------------------------------------------
def _phi4_prompt_table():
    def base(tag, level):
        return f"align/phi4mm-prompt-{tag}-{level}-timit-test-L2_e_grad-pertoken-asotTrue-bs-5-en0.5-sil1.0"

    cols = [
        {"key": "sub", "header": "subword [ms]"},
        {"key": "char", "header": "char [ms]"},
    ]
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
    caption = (
        "Prompt sensitivity of gradient alignment on a speech LLM (Phi-4-MM), TIMIT-test, forced "
        "ground-truth transcription -- only the text instruction spliced around the audio is varied. "
        "Word-boundary error for subword and character-level targets. The last row instructs "
        "character-level output, paired only with character-level targets."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:phi4-prompt",
        label_header="Instruction",
        col_align="|l|r|r|",
    )
    tk.register_output("tables/phi4-prompt.tex", job.out_tex)


# ----------------------------------------------------------------------------------------
# T5: cost -- forward vs batched grad-backward, per model (same GPU).
# ----------------------------------------------------------------------------------------
def _cost_table():
    m = _RESULTS.get("cost-benchmark-metrics.txt")

    def cell(name, metric):
        return {"var": m, "key": f"{name}|{metric}", "fmt": "{:.0f}", "missing": "(*)"}

    cols = [
        {"key": "fwd", "header": "forward [ms/s]"},
        {"key": "bwd", "header": "+grad backward [ms/s]"},
        {"key": "tot", "header": "grad total [ms/s]"},
    ]
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
    caption = (
        "Cost of gradient alignment, per model, on a single GPU (TIMIT-test, 40 seqs): "
        "wall-clock for the forward pass vs the batched per-token grad backward, in ms per second "
        "of audio. The native methods (forced-align / attention-DTW) consume the same forward and "
        "skip the backward, so the forward column is their cost; gradient alignment adds the backward "
        "(fast batched variant). The align/DP step (shared, ~identical across methods) is excluded. "
        "(*) = batched backward unavailable for this model (vmap-incompatible attention kernel)."
    )
    job = WriteLatexTableJob(
        columns=cols,
        rows=rows,
        caption=caption,
        label="tab:cost",
        label_header="Model",
        col_align="|l|r|r|r|",
    )
    tk.register_output("tables/cost.tex", job.out_tex)
