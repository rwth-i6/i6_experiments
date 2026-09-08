"""Listen to a training run: the model's own replies, as audio and text, across its whole length.

The question this exists for. A17 established that recall really collapses -- held-out loss falls
3.5x while the in-loop probe goes 18.4% -> 2.3% in the same run -- but not *which* failure that is,
and the two call for opposite fixes:

  * **Forgetting.** The model still answers in fluent, well-formed, on-topic English and simply no
    longer knows the fact. The corpus is the lever.
  * **Incoherence.** The output itself degrades -- empty replies, looping, token salad -- and
    accuracy falls because nothing intelligible comes out. The trainer / LR / regulariser is.

No A-series run could answer it, because the probe generated the model's full reply, scored it to a
single 0/1 bit, and threw the text away. It now keeps them (``moshi_family.knowledge_probe``), and
this job turns what a run kept into one self-contained page.

**Why probe resolution, not checkpoints.** ``attach_knowledge_evals`` already re-runs inference on
saved checkpoints, but ``save_every`` is 500 and a17's collapse happened entirely between steps 250
and 450 -- inside a single checkpoint interval, invisible to any checkpoint-based ladder. The in-loop
probe fires every 50 steps on the SAME questions, so the same 8 questions are captured ~12 times
across the run and the change is audible as it happens.

Output is ONE ``index.html`` with the audio embedded as base64 ogg-opus, so it needs no server, no
sibling files and no network: open it, or publish it as-is. mini_task -- login-node, no GPU, and it
re-runs no training.
"""

import base64
import json
import os
import tempfile

from sisyphus import Job, Task, tk

#: ``sphn.write_opus(filename, data, sample_rate)`` -- no bitrate knob, and it resamples to Opus's
#: native 48 kHz itself, so there is nothing to pre-resample. Measured on the login node: a 10 s
#: reply encodes to ~65 KB, i.e. ~87 KB of base64, so a full 8-question x 12-step grid embeds in
#: ~8 MB against the ~92 MB the raw wavs would take.

#: Hard ceiling on the finished page. Kept under the 16 MB artifact limit with room for the markup.
MAX_PAGE_BYTES = 12 * 1024 * 1024


def _read_transcripts(run_dir: str) -> list:
    """Probe-step records from ``probe_transcripts.jsonl``, oldest first, deduped by step.

    A preempted run RESUMES and re-probes steps it already logged, so the file is append-ordered,
    not step-ordered, and a step can appear twice. The later record wins (it is the one produced by
    the run that actually continued), and the result is sorted by step -- the same hazard
    ``TrainMetricsPlot`` handles for ``metrics.train.jsonl``.
    """
    path = os.path.join(run_dir, "probe_transcripts.jsonl")
    if not os.path.exists(path):
        return []
    by_step = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a live writer must not lose the whole file
            if "step" in rec:
                by_step[int(rec["step"])] = rec
    return [by_step[s] for s in sorted(by_step)]


def _audio_steps(run_dir: str) -> dict:
    """``{step: dir}`` for every ``probe_audio/step_<N>/`` the run wrote."""
    root = os.path.join(run_dir, "probe_audio")
    if not os.path.isdir(root):
        return {}
    out = {}
    for name in os.listdir(root):
        if not name.startswith("step_"):
            continue
        try:
            step = int(name[len("step_") :])
        except ValueError:
            continue
        d = os.path.join(root, name)
        if os.path.isdir(d):
            out[step] = d
    return out


def _pick(values: list, k: int) -> list:
    """Evenly-spaced subset of ``values``, ALWAYS including the first and last.

    The endpoints are the comparison -- base-ish model versus collapsed model -- so a naive
    ``values[::stride]`` that drops the final step would remove the single most informative clip.
    """
    if k <= 0 or len(values) <= k:
        return list(values)
    if k == 1:
        return [values[-1]]
    idx = sorted({round(i * (len(values) - 1) / (k - 1)) for i in range(k)})
    return [values[i] for i in idx]


def _encode_clip(wav_path: str, clip_seconds: float) -> str | None:
    """Load an exchange wav, trim, encode to ogg-opus, return base64. None on any failure.

    **Channels are preserved.** The probe writes stereo -- the question on channel 0, the reply on
    channel 1, sample-aligned -- and that pairing is the whole point of listening: it is what shows
    whether the model answered late, talked over the question, or answered something else. Folding
    to mono here (the first version of this function took ``data[0]``, i.e. the question alone, and
    would have silently dropped every reply) throws away the half the page exists to present.

    Best effort by design: this is a listening convenience assembled from artifacts a finished run
    already produced, and one unreadable clip must not cost the whole page.
    """
    try:
        import numpy as np
        import sphn

        data, sr = sphn.read(wav_path)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]
        if clip_seconds > 0:
            data = data[:, : int(clip_seconds * sr)]
        if data.size == 0:
            return None
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.opus")
            sphn.write_opus(p, data if data.shape[0] > 1 else data[0], sr)
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"[listen] could not encode {wav_path}: {e!r}", flush=True)
        return None


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


#: Coherence fields rendered in the per-step table, in order: (json key, column label, format).
COHERENCE_COLUMNS = (
    ("words_mean", "words", "{:.1f}"),
    ("empty_frac", "empty", "{:.0%}"),
    ("distinct_word_ratio", "distinct", "{:.2f}"),
    ("looping_frac", "looping", "{:.0%}"),
    ("chars_per_word", "chars/word", "{:.2f}"),
)


class ProbeListeningReport(Job):
    """One self-contained HTML page per run: reply audio + transcripts + coherence, step by step.

    ``run_dirs`` maps a label to a ``SpeechFinetune.out_rundir``. Depending on the run dir (not a
    file inside it) makes this fire exactly when the run finishes.

    Degrades honestly rather than failing: a run trained before the probe kept anything renders with
    an explicit note saying so, instead of an empty page that reads like a broken job.
    """

    def __init__(
        self,
        *,
        run_dirs: dict,
        questions: int = 8,
        max_steps_shown: int = 12,
        clip_seconds: float = 10.0,
        title: str = "",
    ):
        self.run_dirs = run_dirs  # {label: tk.Path}; paths in VALUE position (see CLAUDE.md)
        self.questions = questions
        self.max_steps_shown = max_steps_shown
        self.clip_seconds = clip_seconds
        self.title = title
        self.out_html = self.output_path("index.html")
        self.out_summary = self.output_path("coherence.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        summary = {}
        parts = []
        for label, path in sorted(self.run_dirs.items()):
            run_dir = path.get_path() if hasattr(path, "get_path") else str(path)
            html, coh = self._one_run(label, run_dir)
            parts.append(html)
            summary[label] = coh

        title = self.title or "Probe replies across training"
        page = _PAGE.format(title=_esc(title), body="\n".join(parts))
        if len(page.encode()) > MAX_PAGE_BYTES:
            # Never silently ship a page too big to open/publish; say what to lower instead.
            raise AssertionError(
                f"page is {len(page.encode()) / 1e6:.1f} MB, over the {MAX_PAGE_BYTES / 1e6:.0f} MB "
                f"ceiling. Lower questions ({self.questions}), max_steps_shown "
                f"({self.max_steps_shown}) or clip_seconds ({self.clip_seconds})."
            )
        with open(self.out_html.get_path(), "w") as f:
            f.write(page)
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[listen] wrote {len(page.encode()) / 1e6:.2f} MB page", flush=True)

    # -- one run -------------------------------------------------------------------------------
    def _one_run(self, label: str, run_dir: str):
        recs = _read_transcripts(run_dir)
        audio = _audio_steps(run_dir)
        coh = {r["step"]: r.get("coherence", {}) for r in recs}
        acc = {r["step"]: r.get("accuracy") for r in recs}

        head = [f"<section><h2>{_esc(label)}</h2>"]
        head.append(
            '<p class="prov">Model: <b>ours</b> &mdash; a finetune we trained, not a released '
            "checkpoint. Replies are the model&rsquo;s own text stream (inner monologue), which is "
            "what it says; they are not an ASR transcription of the audio.<br>"
            "Audio is <b>stereo and sample-aligned</b>: <b>left</b> is the question the model "
            "heard, <b>right</b> is what it said back, on one timeline. Listen on headphones "
            "&mdash; overlap, hesitation and answering before the question ends are audible only "
            "in the pairing.</p>"
        )
        if not recs and not audio:
            head.append(
                '<p class="warn">This run recorded no probe transcripts and no probe audio &mdash; '
                "it trained before the probe kept them. Nothing here is missing by accident; there "
                "is nothing to show. Set <code>Probe(audio_n=8)</code> on the arm to capture it.</p>"
                "</section>"
            )
            return "\n".join(head), {}

        # coherence-vs-step table: the numeric answer, before anyone listens to anything.
        head.append("<h3>Coherence vs step</h3>")
        head.append(
            '<p class="note">If accuracy falls while these stay flat, the model still speaks and '
            "has <b>forgotten</b>. If they degrade with it, the output itself is breaking.</p>"
        )
        head.append('<table class="coh"><tr><th>step</th><th>probe acc</th>')
        for _, lab, _f in COHERENCE_COLUMNS:
            head.append(f"<th>{_esc(lab)}</th>")
        head.append("</tr>")
        for step in sorted(coh):
            a = acc.get(step)
            head.append(f"<tr><td>{step}</td><td>{('%.1f%%' % (100 * a)) if a is not None else '&mdash;'}</td>")
            c = coh[step] or {}
            for key, _lab, fmt in COHERENCE_COLUMNS:
                v = c.get(key)
                head.append(f"<td>{fmt.format(v) if isinstance(v, (int, float)) else '&mdash;'}</td>")
            head.append("</tr>")
        head.append("</table>")

        if not audio:
            head.append(
                '<p class="warn">No probe audio was written for this run (transcripts only). The '
                "text below still answers the coherence question; add <code>audio_n</code> to hear "
                "it.</p>"
            )

        # steps and questions to render
        steps = sorted(set(coh) | set(audio))
        shown_steps = _pick(steps, self.max_steps_shown)
        n_q = 0
        for r in recs:
            n_q = max(n_q, len(r.get("rows", [])))
        if audio:
            for d in audio.values():
                n_q = max(n_q, len([f for f in os.listdir(d) if f.endswith(".wav")]))
        q_idx = list(range(min(self.questions, n_q)))

        rows_by_step = {r["step"]: r.get("rows", []) for r in recs}
        body = []
        for q in q_idx:
            gold = question = ""
            for step in shown_steps:
                rr = rows_by_step.get(step) or []
                if q < len(rr):
                    gold = rr[q].get("answer", "")
                    question = rr[q].get("question", "")
                    break
            body.append(f'<div class="q"><h3>Q{q}</h3>')
            if question:
                body.append(f'<p class="ask">{_esc(question)}</p>')
            body.append(f'<p class="gold">gold: <b>{_esc(gold)}</b></p>')
            for step in shown_steps:
                rr = rows_by_step.get(step) or []
                row = rr[q] if q < len(rr) else {}
                ok = row.get("binary_correct")
                mark = "&#10003;" if ok == 1 else ("&#10007;" if ok == 0 else "&mdash;")
                cls = "ok" if ok == 1 else ("no" if ok == 0 else "")
                body.append(f'<div class="step"><span class="s">step {step}</span> ')
                body.append(f'<span class="mark {cls}">{mark}</span>')
                b64 = None
                d = audio.get(step)
                if d:
                    wav = os.path.join(d, f"{q:03d}.wav")
                    if os.path.exists(wav):
                        b64 = _encode_clip(wav, self.clip_seconds)
                if b64:
                    body.append(f'<audio controls preload="none" src="data:audio/ogg;base64,{b64}"></audio>')
                mono = row.get("monologue", "")
                body.append(f'<div class="mono">{_esc(mono) if mono else "<i>(no text)</i>"}</div>')
                body.append("</div>")
            body.append("</div>")
        head.extend(body)
        head.append("</section>")
        return "\n".join(head), {str(k): v for k, v in coh.items()}


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:980px;padding:24px;color:#111}}
h1{{font-size:20px}} h2{{font-size:17px;margin-top:32px;border-bottom:1px solid #ddd}}
h3{{font-size:14px;margin:18px 0 6px}}
.prov,.note{{color:#555;font-size:13px}}
.warn{{background:#fff6e0;border-left:3px solid #e0a800;padding:8px 12px}}
table.coh{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}}
table.coh th,table.coh td{{border:1px solid #ddd;padding:3px 9px;text-align:right}}
.q{{border-top:1px solid #eee;padding-top:10px;margin-top:18px}}
.ask{{margin:2px 0;color:#333}}
.gold{{margin:2px 0;color:#333}}
.step{{margin:8px 0;padding-left:10px;border-left:2px solid #eee}}
.s{{display:inline-block;min-width:76px;color:#666;font-variant-numeric:tabular-nums}}
.mark.ok{{color:#178a3a}} .mark.no{{color:#b3261e}}
audio{{vertical-align:middle;height:28px;margin-left:8px}}
.mono{{margin:4px 0 0 76px;color:#111;white-space:pre-wrap}}
</style></head><body>
<h1>{title}</h1>
<p class="note">Each question is played at several points in the run, same question every time.
Read down a question to hear what changed.</p>
{body}
</body></html>
"""


def probe_listening_py(name: str, run_dirs: dict, **kwargs):
    """Register a listening report under ``output/listening/<name>/``. Returns the job."""
    job = ProbeListeningReport(run_dirs=run_dirs, title=kwargs.pop("title", name), **kwargs)
    tk.register_output(f"listening/{name}/index.html", job.out_html)
    tk.register_output(f"listening/{name}/coherence.json", job.out_summary)
    return job
