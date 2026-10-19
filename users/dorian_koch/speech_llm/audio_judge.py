"""An audio LLM as the ONLY grader, against our text graders (2026-10-19).

User: "Can we do a test run in how well nemotron agrees with our text graders on gradings?" ... "in a setup i dont want
two concurrent graders runnin, so ideally only nemotron" -- i.e. one audio model in place of Whisper + the text judge.
Prototype: ``analysis/nemotron_judge_agreement.py`` (its prep stage decoded every spj2 conversation; its vLLM start
failed only on the hand-written sbatch's environment).

Two populations with audio AND a text grader's verdict:

* knowledge (``knowledge``: ``{tag: {"replies": moshi_output, "transcription": WhisperTranscription.out_json, "grades":
  LLMGrading.out_eval}}``; reference = the benchmark's gemma verdict on the Whisper transcript, ``binary_correct``): the
  same seeded ``n_knowledge`` question indices for every tag (paired). Audio = the model's own channel, cut as
  ``bench_audio_export.py`` (last speech + 1.5 s, at least question end + 3 s; the track is time-aligned to the 2 s
  lead-in + question, whose length comes from ``question_audio``).
* self-play (``selfplay``: ``{label: {"conversations": SelfPlayTranscribe.out_dir, "grades": SelfPlayJudgeV2.out_rows}}``;
  reference = gpt-oss-120b on the two-sided Whisper transcript, rows of that label): every conversation, both channels
  decoded from the stored Mimi codes with ``moshi_family.personaplex.selfplay_decode`` (the transcribe job's decoder).

Conditions, one server of ``judge_model`` (``common.vllm_server``), sampling as the reference graders:
  K_text  the benchmark's ``GRADING_PROMPT_TEMPLATE`` with the Whisper transcript -- the judge-model effect alone
  K_audio the same prompt, the transcript replaced by the attached reply audio (+ a ``transcription`` field)
  S_text  the spj2 rubric with the same Whisper transcript as gpt-oss -- the judge-model effect alone
  S_audio the same rubric, its transcript sentences swapped for two attached time-aligned recordings (A, then B) and no
          word counts (an audio-only setup has no ASR); the rubric's "Ignore recognition errors, ..., audio quality"
          becomes "Ignore disfluencies" + unintelligible speech counts as not said (user: ignoring audio quality hides
          exactly the audio-side hacks we want to catch), plus a per-side ``speech_a``/``speech_b`` 1-5 score (not in
          the reference, so reported as per-label means only)
Prompt texts (these + the two reference prompts) are hashed via ``prompt_digest``. Labels are not (``label_free_hash``).

Outputs: ``rows.jsonl`` (one line per call: condition, item, reference, reply, error) and ``summary.json``: knowledge per
tag accuracy under each grader, agreement and Cohen's kappa vs gemma, audio vs text; self-play per-metric Pearson vs
gpt-oss, within-scenario Spearman of ``overall`` (all labels x replicates of a scenario), per-label means.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from sisyphus import Job, Task, tk

from .common import worker_env
from .label_free_hash import label_free_hash, values_canon

NEMOTRON_OMNI = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
LEAD, TAIL, MIN_AFTER_Q = 2.0, 1.5, 3.0
CONDITIONS = ("K_text", "K_audio", "S_text", "S_audio")

AUDIO_RESPONSE = (
    "The assistant's spoken response is the attached audio recording of the assistant's own channel (it may open "
    "with a short greeting spoken before the question was asked). Listen to what it actually says."
)
AUDIO_JSON = (
    'Return ONLY JSON: {"transcription": "what the assistant says", "binary": 0 or 1, "quality": 1-5, '
    '"reasoning": "brief explanation"}'
)
SPJ2_ASR_SENTENCE = (
    "The transcript comes from automatic speech recognition of each agent's own audio channel: expect recognition "
    "errors, missing punctuation and fragments. Each line is one utterance, in time order, with its start time in "
    "seconds; the agents may overlap in time."
)
SPJ2_AUDIO_SENTENCE = (
    "You hear the conversation as two attached, time-aligned recordings, one per agent's own audio channel: the first "
    "recording is agent A, the second is agent B. Both start at 0 s; the agents may overlap in time."
)
SPJ2_RUBRIC = "spj2_judge_v1"
SPJ2_IGNORE = "Ignore recognition errors, disfluencies, audio quality, politeness and factual accuracy."
AUDIO_IGNORE = (
    "Ignore disfluencies, politeness and factual accuracy. Judge what you hear, not what the agents were meant to say: "
    "speech you cannot make out (garbled, distorted, mumbled, too fast, or buried in noise) counts as not said."
)
SPEECH_METRICS = ("speech_a", "speech_b")
SPEECH_RUBRIC = """speech_a: how A's speech sounds -- clear, intelligible words at a natural pace and loudness, a stable voice.
speech_b: the same for B.
  5 = clear and natural throughout. 4 = clear, with brief glitches. 3 = understandable, but noticeably garbled, distorted, rushed or unnatural in places. 2 = mostly hard to understand. 1 = unintelligible, noise, or (almost) no speech.

"""


def prompt_digest() -> str:
    """Hash of every prompt text the job sends: ours and the two reference graders' (an edit re-runs the job)."""
    from .knowledge_benchmark import GRADING_PROMPT_TEMPLATE
    from .selfplay_bench import SPJ2_RUBRICS

    parts = [AUDIO_RESPONSE, AUDIO_JSON, SPJ2_ASR_SENTENCE, SPJ2_AUDIO_SENTENCE, GRADING_PROMPT_TEMPLATE]
    parts += [SPJ2_IGNORE, AUDIO_IGNORE, SPEECH_RUBRIC]
    parts.append(SPJ2_RUBRICS[SPJ2_RUBRIC])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def knowledge_audio_prompt(text_prompt: str) -> str:
    """The benchmark prompt with its transcript line replaced by the attached audio."""
    out = text_prompt.replace("The assistant's response (transcribed via ASR): {AUDIO}", AUDIO_RESPONSE)
    out = out.rsplit("Return ONLY JSON", 1)[0] + AUDIO_JSON
    assert "{AUDIO}" not in out and "transcribed via ASR" not in out, "GRADING_PROMPT_TEMPLATE wording changed"
    return out


def selfplay_audio_prompt(rendered: str, n_a, n_b) -> str:
    """The rendered spj2 message (transcript = ``{AUDIO}``) turned into the audio variant."""
    assert SPJ2_ASR_SENTENCE in rendered, "the spj2 rubric's transcript sentence changed"
    out = rendered.replace(SPJ2_ASR_SENTENCE, SPJ2_AUDIO_SENTENCE)
    out = out.replace(f" Words recognised: A {n_a}, B {n_b}.", "")
    out = out.replace("TRANSCRIPT:\n{AUDIO}", "AUDIO: the two attached recordings (A first, B second).")
    assert SPJ2_IGNORE in out and out.count("Answer with JSON only.") == 1, (
        "the spj2 rubric's ignore/answer lines changed"
    )
    out = out.replace("Rate four things", "Rate six things").replace(SPJ2_IGNORE, AUDIO_IGNORE)
    out = out.replace("Answer with JSON only.", SPEECH_RUBRIC + "Answer with JSON only.")
    speech = ", ".join(f'"{m}": {{"reason": "...", "score": 1-5}}' for m in SPEECH_METRICS)
    assert out.rstrip().endswith("}"), "the spj2 JSON example changed"
    out = out.rstrip()[:-1] + ", " + speech + "}"
    assert "{AUDIO}" not in out and "Words recognised" not in out and "recognition errors" not in out, "spj2 wording"
    assert "audio quality" not in out, "the audio judge must not be told to ignore audio quality"
    return out


def parse_audio_sp_reply(raw: str) -> dict:
    """The spj2 metrics (``selfplay_bench.parse_judge_reply``) + ``SPEECH_METRICS``, same validation."""
    from .selfplay_bench import SPJ2_REASON_CHARS, parse_judge_reply

    out = parse_judge_reply(raw)
    d = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    for m in SPEECH_METRICS:
        item = d.get(m) if isinstance(d, dict) else None
        s = item.get("score") if isinstance(item, dict) else None
        if isinstance(s, bool) or not isinstance(s, int) or s not in (1, 2, 3, 4, 5):
            raise ValueError(f"{m}: score {s!r}")
        reason = " ".join(str(item.get("reason") or "").split())
        if not reason:
            raise ValueError(f"{m}: empty reason")
        out[m] = {"score": s, "reason": reason[:SPJ2_REASON_CHARS]}
    return out


def last_speech(y, sr, hop=0.02) -> float:
    import numpy as np

    n = int(sr * hop)
    m = len(y) // n
    db = 20 * np.log10(np.sqrt(np.mean(y[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12))
    act = np.flatnonzero(db > max(-45.0, np.percentile(db, 95) - 30.0))
    return (act[-1] + 1) * hop if len(act) else 0.0


def kappa(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


class AudioJudgeAgreement(Job):
    """See the module docstring. ``venv_python_path``: the moshi_family venv (the Mimi decode)."""

    def __init__(
        self,
        *,
        knowledge: dict,
        selfplay: dict,
        question_audio: tk.Path,
        venv_python_path: tk.Path,
        judge_model: str = NEMOTRON_OMNI,
        n_knowledge: int = 300,
        seed: int = 20261019,
    ):
        assert knowledge or selfplay, "nothing to judge"
        for tag, v in knowledge.items():
            assert set(v) == {"replies", "transcription", "grades"}, (tag, sorted(v))
        for label, v in selfplay.items():
            assert set(v) == {"conversations", "grades"}, (label, sorted(v))
        self.knowledge = knowledge
        self.selfplay = selfplay
        self.question_audio = question_audio
        self.venv_python_path = venv_python_path
        self.judge_model = judge_model
        self.n_knowledge = int(n_knowledge)
        self.seed = int(seed)
        self.out_rows = self.output_path("rows.jsonl")
        self.out_summary = self.output_path("summary.json")
        from .common import vllm_gpu_mem_gb

        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 96, "time": 4, "gpu_mem_gb": vllm_gpu_mem_gb(judge_model, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """Labels (the dict keys) are not hashed (``label_free_hash``); the prompt texts are (``prompt_digest``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {**a, "knowledge": values_canon(a["knowledge"]), "selfplay": values_canon(a["selfplay"])},
            base_hash=lambda a: super(AudioJudgeAgreement, cls).hash({**a, "prompt_digest": prompt_digest()}),
            labels=lambda a: (sorted(a["knowledge"]), sorted(a["selfplay"])),
        )

    def tasks(self):
        yield Task("prep", rqmt={"gpu": 1, "cpu": 4, "mem": 48, "time": 2})
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def info(self):
        n_sp = len(self.selfplay)
        return f"{len(self.knowledge)} knowledge tags x {self.n_knowledge}, {n_sp} self-play labels, {self.judge_model}"

    # ------------------------------------------------------------------ prep: audio on disk (work/wav, deleted at the end)
    def know_indices(self) -> list[int]:
        import numpy as np

        return sorted(np.random.default_rng(self.seed).choice(1000, self.n_knowledge, replace=False).tolist())

    def prep(self):
        import numpy as np
        import soundfile as sf
        from datasets import load_from_disk

        from .clip_codec import audio_columns, decode_row, numpy_rows

        wav = Path("wav").absolute()
        for label, v in self.selfplay.items():
            d = wav / "sp" / hashlib.md5(label.encode()).hexdigest()[:12]
            if (d / "energy.jsonl").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            cmd = [self.venv_python_path.get(), "-m", "moshi_family.personaplex.selfplay_decode"]
            cmd += ["--data", v["conversations"].get_path(), "--out_dir", str(d)]
            subprocess.run(cmd, env=worker_env(offline=True), check=True)
            print(f"[audio-judge] decoded self-play {label}", flush=True)
        tts = numpy_rows(load_from_disk(self.question_audio.get_path()))
        t_idx = {int(i): j for j, i in enumerate(tts["index"])}
        for tag, v in self.knowledge.items():
            d = wav / "know" / hashlib.md5(tag.encode()).hexdigest()[:12]
            d.mkdir(parents=True, exist_ok=True)
            ds = load_from_disk(v["replies"].get_path())
            ds = numpy_rows(ds.select_columns(["index", *audio_columns(ds.column_names), "sampling_rate"]))
            idx = {int(i): j for j, i in enumerate(ds["index"])}
            for i in self.know_indices():
                p = d / f"q{i}.wav"
                if p.exists():
                    continue
                r, q = ds[idx[i]], tts[t_idx[i]]
                y, sr = decode_row(r).reshape(-1), int(r["sampling_rate"])
                q_end = LEAD + len(decode_row(q).reshape(-1)) / int(q["sampling_rate"])
                end = min(len(y) / sr, max(last_speech(y, sr) + TAIL, q_end + MIN_AFTER_Q))
                sf.write(str(p), np.clip(y[: int(end * sr)], -1, 1), sr, subtype="PCM_16")
            print(f"[audio-judge] cut {self.n_knowledge} knowledge replies of {tag}", flush=True)

    # ------------------------------------------------------------------ run: the judge
    def items(self) -> list[dict]:
        from . import knowledge_benchmark as kb
        from . import selfplay_bench as sb

        wav = Path("wav").absolute()
        items = []
        for tag, v in self.knowledge.items():
            t = [json.loads(line) for line in open(v["transcription"].get_path())]
            e = [json.loads(line) for line in open(v["grades"].get_path())]
            d = wav / "know" / hashlib.md5(tag.encode()).hexdigest()[:12]
            for i in self.know_indices():
                assert t[i]["question"] == e[i]["question"] and int(t[i]["index"]) == i, (tag, i)
                aliases = t[i]["aliases"] if isinstance(t[i]["aliases"], list) else json.loads(t[i]["aliases"])
                base = dict(pop="know", tag=tag, i=i, ref=int(e[i]["binary_correct"]), whisper=t[i]["transcription"])
                fields = dict(question=t[i]["question"], answer=t[i]["answer"], aliases=", ".join(aliases))
                text = kb.GRADING_PROMPT_TEMPLATE.format(**fields, transcription=t[i]["transcription"])
                audio = knowledge_audio_prompt(kb.GRADING_PROMPT_TEMPLATE.format(**fields, transcription="{AUDIO}"))
                items.append({**base, "cond": "K_text", "msg": text, "audio": []})
                items.append({**base, "cond": "K_audio", "msg": audio, "audio": [str(d / f"q{i}.wav")]})
        for label, v in self.selfplay.items():
            d = wav / "sp" / hashlib.md5(label.encode()).hexdigest()[:12]
            for r in map(json.loads, open(v["grades"].get_path())):
                if r["label"] != label or not r.get("judged"):
                    continue
                ref = {m: r["judged"][m]["score"] for m in sb.SPJ2_METRICS}
                base = dict(pop="sp", tag=label, i=r["idx"], id=r["id"], scenario=r["scenario"], ref=ref)
                kw = dict(prompt_a=r["prompt_a"], prompt_b=r["prompt_b"], duration=r["dur"], n_a=r["n_a"], n_b=r["n_b"])
                text = sb.render_judge_message(SPJ2_RUBRIC, **kw, transcript=r["transcript"])
                audio = selfplay_audio_prompt(
                    sb.render_judge_message(SPJ2_RUBRIC, **kw, transcript="{AUDIO}"), r["n_a"], r["n_b"]
                )
                paths = [str(d / f"{r['idx']}_a.wav"), str(d / f"{r['idx']}_b.wav")]
                items.append({**base, "cond": "S_text", "msg": text, "audio": []})
                items.append({**base, "cond": "S_audio", "msg": audio, "audio": paths})
            assert any(it["pop"] == "sp" and it["tag"] == label for it in items), f"no judged rows of {label}"
        missing = [a for it in items for a in it["audio"] if not os.path.exists(a)]
        assert not missing, f"{len(missing)} audio files missing, e.g. {missing[:2]}"
        return items

    def run(self):
        from openai import OpenAI

        from . import knowledge_benchmark as kb
        from . import selfplay_bench as sb
        from .common import json_schema_format, map_concurrent, vllm_server

        items = self.items()
        counts = ", ".join(f"{c} {sum(it['cond'] == c for it in items)}" for c in CONDITIONS)
        print(f"[audio-judge] {len(items)} calls ({counts}) to {self.judge_model}", flush=True)
        sp_schema = json_schema_format(sb.SPJ2_JSON_SCHEMA, "spj2")
        spa_schema = json_schema_format(sb.rubric_json_schema(sb.SPJ2_METRICS + SPEECH_METRICS), "spj2_audio")
        model = self.judge_model

        def call(client, it):
            content = [{"type": "text", "text": it["msg"]}]
            content += [{"type": "audio_url", "audio_url": {"url": Path(a).as_uri()}} for a in it["audio"]]
            if it["pop"] == "know":
                kw = dict(**kb.JUDGE_SAMPLING, response_format={"type": "json_object"})
                parse = kb.parse_judge_reply
            else:
                seed = int(hashlib.md5((it["msg"] + it["cond"]).encode()).hexdigest(), 16) % (2**31)
                audio = it["cond"] == "S_audio"
                kw = dict(temperature=0.3, seed=seed, **(spa_schema if audio else sp_schema))
                parse = parse_audio_sp_reply if audio else sb.parse_judge_reply
            err = ""
            for _ in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": content}], max_tokens=6144, **kw
                    )
                    return {"reply": parse(resp.choices[0].message.content), "error": ""}
                except Exception as e:  # noqa: BLE001 -- retried like the reference graders
                    err = f"{type(e).__name__}: {e}"[:300]
            return {"reply": None, "error": err}

        with vllm_server(model) as url:
            client = OpenAI(api_key="EMPTY", base_url=url, timeout=600)
            res = map_concurrent(
                lambda it: call(client, it), items, concurrency=64, progress_path=os.path.abspath("progress.json")
            )
        failed = {c: sum(1 for it, r in zip(items, res) if it["cond"] == c and r["reply"] is None) for c in CONDITIONS}
        print(f"[audio-judge] failed calls: {failed}", flush=True)
        for it, r in [(it, r) for it, r in zip(items, res) if r["reply"] is None][:5]:
            print(f"  FAILED {it['cond']} {it['tag']} {it['i']}: {r['error']}", flush=True)
        if sum(failed.values()) > 0.05 * len(items):
            raise RuntimeError(f"audio judge failed on {sum(failed.values())}/{len(items)} calls (> 5%)")
        with open(self.out_rows.get_path(), "w") as f:
            for it, r in zip(items, res):
                f.write(json.dumps({k: v for k, v in {**it, **r}.items() if k != "msg"}) + "\n")
        summary = {"judge_model": model, "prompt_digest": prompt_digest(), "failed": failed, **summarize(items, res)}
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=1)
        print(json.dumps(summary, indent=1), flush=True)
        shutil.rmtree("wav", ignore_errors=True)  # scratch: ~5 GB of decoded audio; prep re-creates it on a rerun


def summarize(items: list[dict], res: list[dict]) -> dict:
    import numpy as np
    from scipy.stats import spearmanr

    from . import selfplay_bench as sb

    rows = [{**it, **r} for it, r in zip(items, res)]
    know = {}
    for tag in sorted({r["tag"] for r in rows if r["pop"] == "know"}):
        by = {}
        for r in rows:
            if r["pop"] == "know" and r["tag"] == tag and r["reply"] is not None:
                by.setdefault(r["i"], {"ref": r["ref"]})[r["cond"]] = int(r["reply"]["binary"])
        both = [v for v in by.values() if "K_text" in v and "K_audio" in v]
        g, t, a = (np.array([v[k] for v in both]) for k in ("ref", "K_text", "K_audio"))
        know[tag] = {
            "n": len(both),
            "acc": {
                "gemma_on_whisper": float(g.mean()),
                "judge_on_whisper": float(t.mean()),
                "judge_on_audio": float(a.mean()),
            },
            "agree_vs_gemma": {"text": float((t == g).mean()), "audio": float((a == g).mean())},
            "kappa_vs_gemma": {"text": kappa(t, g), "audio": kappa(a, g)},
            "audio_vs_text": {"agree": float((a == t).mean()), "kappa": kappa(a, t)},
        }
    conv = {}
    for r in rows:
        if r["pop"] == "sp" and r["reply"] is not None:
            scores = {m: r["reply"][m]["score"] for m in sb.SPJ2_METRICS + SPEECH_METRICS if m in r["reply"]}
            conv.setdefault((r["tag"], r["id"]), {"ref": r["ref"], "scenario": r["scenario"]})[r["cond"]] = scores
    full = [v for v in conv.values() if "S_text" in v and "S_audio" in v]

    def ov(x):
        return sb.derived_scores(x)["overall"]

    sp = {"n": len(full), "per_metric_pearson": {}, "within_scenario_spearman_overall": {}, "label_means_overall": {}}
    if full:
        for m in sb.SPJ2_METRICS:
            ref = np.array([v["ref"][m] for v in full], float)
            sp["per_metric_pearson"][m] = {c: float(np.corrcoef(ref, [v[c][m] for v in full])[0, 1]) for c in ("S_text", "S_audio")}  # fmt: skip
        for c in ("S_text", "S_audio"):
            per = []
            for scn in sorted({v["scenario"] for v in full}):
                vs = [v for v in full if v["scenario"] == scn]
                x, y = [ov(v["ref"]) for v in vs], [ov(v[c]) for v in vs]
                if np.std(x) and np.std(y):
                    per.append(float(spearmanr(x, y).correlation))
            sp["within_scenario_spearman_overall"][c] = float(np.mean(per)) if per else None
        for label in sorted({k[0] for k in conv}):
            vs = [v for (lb, _), v in conv.items() if lb == label and "S_text" in v and "S_audio" in v]
            sp["label_means_overall"][label] = {
                "n": len(vs),
                "gpt_oss_on_whisper": float(np.mean([ov(v["ref"]) for v in vs])),
                "judge_on_whisper": float(np.mean([ov(v["S_text"]) for v in vs])),
                "judge_on_audio": float(np.mean([ov(v["S_audio"]) for v in vs])),
                "speech_on_audio": float(np.mean([v["S_audio"][m] for v in vs for m in SPEECH_METRICS])),
            }
    return {"knowledge": know, "selfplay": sp}
