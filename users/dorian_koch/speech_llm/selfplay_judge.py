"""LLM judge of dialogue COHERENCE for self-play conversations (and the real windows they came from).

User, 2026-09-23, on p2/p3 self-play: "it's not really a coherent dialogue in the sense that they are not
really talking to each other in a way that makes sense, like persona plex does". The turn-taking statistics
(``analysis/selfplay_stats.py``) measure WHEN each side speaks; this job measures whether what they say is a
conversation: each conversation's two-sided transcript (the per-channel ASR words of ``SelfPlayTranscribe``,
grouped into utterances and interleaved by time) is rated 1-5 with a one-line reason by an LLM served with
vLLM, under a FIXED rubric whose content is hashed (``rubric_digest``), like ``persona_prompts.PROMPT_SPECS``.

The real windows are the ceiling: the self-play INPUT rows (``AttachPersonaPrompts(with_other_side=True)``)
carry both channels' corpus ASR words (``alignments`` / ``alignments_other``, window-relative), so the same
transcript builder and rubric apply to the real audio of the same windows (the ``selfplay_timeline.py`` rule).
Real windows are 20-60 s, self-play recordings 60 s; the transcript states its duration.
"""

from __future__ import annotations

import hashlib
import json
import os

from sisyphus import Job, Task, tk

from .common import json_schema_format, map_concurrent, vllm_gpu_mem_gb, vllm_server
from .label_free_hash import hash_bytes, label_free_hash, values_canon  # noqa: F401

#: Versioned rubrics. The CONTENT is hashed into every job (``rubric_digest``): editing one re-runs.
COHERENCE_RUBRICS = {
    "coherence_v1": {
        "instruction": """You are rating a transcript of a spoken conversation between two speakers, A and B. The transcript comes from automatic speech recognition of each speaker's own audio channel: expect recognition errors, missing punctuation and fragments. Each line is one utterance, in time order, with its start time in seconds; the speakers may overlap in time.

Rate how COHERENT this is as a dialogue: are the two speakers actually talking TO EACH OTHER -- answering what was asked, following up on what the other just said, building on a shared topic -- so that the exchange makes sense as a conversation between two people?

Ignore recognition errors, disfluencies, audio quality, politeness, and whether statements are true. Judge only whether the two sides respond to each other in a way that makes sense.

5 = coherent dialogue: both speakers consistently respond to each other and the topic develops naturally.
4 = mostly coherent: they respond to each other, with brief non-sequiturs or moments of talking past each other.
3 = partly coherent: some real exchange, but frequent non-sequiturs, ignored questions, or parallel monologues.
2 = barely coherent: the speakers mostly talk past each other; only occasional responses to the other.
1 = not a dialogue: no meaningful exchange (one or both speakers silent or nearly silent, or the speech is unrelated or incoherent).

Answer with JSON: {{"reason": "<one line>", "score": <1-5>}}

Duration: {duration:.0f} s. Words recognised: A {n_a}, B {n_b}.

TRANSCRIPT:
{transcript}""",
    },
}

_JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "minLength": 1},
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["reason", "score"],
    "additionalProperties": False,
}

#: The transcript of a conversation with no recognised word on either channel (still rated).
EMPTY_TRANSCRIPT = "(no words were recognised on either channel)"

#: Requests in flight against the vLLM server (run-side, not hashed; persona_prompts measured 192 for
#: gpt-oss-120b on one H100).
JUDGE_CONCURRENCY = 192


def rubric_digest(name: str) -> str:
    """Content hash of a rubric: hashed into every job using it, so an edited rubric re-runs."""
    return hashlib.sha256(json.dumps(COHERENCE_RUBRICS[name], sort_keys=True).encode()).hexdigest()[:16]


#: ``SelfPlayCoherenceJudge`` hashed no digest until 2026-10-29: a rubric at its pinned content keeps that hash, an
#: edited one (or a new rubric) adds its digest. Never update a pin; an edit is meant to move the hash.
PINNED_RUBRIC_DIGESTS = {"coherence_v1": "1512cf772a9ab49b"}


def _with_rubric_digest(a: dict) -> dict:
    digest = rubric_digest(a["rubric"])
    return a if digest == PINNED_RUBRIC_DIGESTS.get(a["rubric"]) else {**a, "rubric_digest": digest}


def two_sided_transcript(words_a: list[dict], words_b: list[dict], dur: float, gap: float = 1.5) -> str:
    """Both channels as utterances (one channel's consecutive words < ``gap`` s apart), ordered by onset,
    ``[12.3] A: ...`` per line. Words with an onset outside ``[0, dur)`` are dropped (the real rows' first
    word can start before the window). Same grouping rule as ``persona_prompts._dialogue_text``."""
    utts = []
    for who, ws in (("A", words_a), ("B", words_b)):
        cur = None
        for w in sorted((w for w in ws if 0 <= float(w["start"]) < dur), key=lambda w: float(w["start"])):
            s, e = float(w["start"]), float(w["end"])
            if cur and s - cur["end"] < gap:
                cur["words"].append(str(w["text"]))
                cur["end"] = e
            else:
                cur = {"who": who, "start": s, "end": e, "words": [str(w["text"])]}
                utts.append(cur)
    utts.sort(key=lambda u: (u["start"], u["who"]))
    if not utts:
        return EMPTY_TRANSCRIPT
    return "\n".join(f"[{u['start']:.1f}] {u['who']}: {' '.join(u['words'])}" for u in utts)


def bootstrap_ci(xs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile bootstrap interval of the mean (seeded)."""
    import numpy as np

    if len(xs) < 2:
        return None
    a = np.asarray(xs, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


#: Coherence judge versions (``summary.json["judge_version"]``; a summary without the field is v1):
#:   1 -- ``guided_json=True`` sent ``extra_body={"guided_json": ...}``, which vLLM >= 0.19 IGNORES: unconstrained
#:        replies, kept valid by the parser + 5 retries. Every coherence number up to 2026-10-12.
#:   2 -- (2026-10-12) ``guided_json=True`` enforces the schema (``common.json_schema_format``); same rubric,
#:        sampling and seeds. Hashed only when != 1 (v1 jobs keep their hash); ``guided_json=False`` = no constraint
#:        in either version.
COHERENCE_JUDGE_VERSION = 2


class SelfPlayCoherenceJudge(Job):
    """Rate every conversation of several self-play runs (and optionally their real windows) for coherence.

    ``runs``: ``{label: SelfPlayTranscribe dataset}``. ``real_data``: a self-play INPUT dataset; its rows
    whose ``id`` is one of the runs' JRE conversations are rated as label ``real`` (custom prompt pairs
    have no real counterpart). One job per group of runs, so the model loads once.

    Outputs: ``rows.jsonl`` (one line per conversation: label, idx = row index in its run, id, score,
    reason, word counts, transcript) and ``summary.json`` (per label: n, failed, mean, sd, 95% bootstrap
    CI, score histogram; and the same over the first 32 rows, the clean batch of the runs that predate
    the 2026-09-22 reset fix). More than 2% failed calls is fatal.
    """

    def __init__(
        self,
        *,
        runs: dict,
        real_data: tk.Path | None = None,
        rubric: str = "coherence_v1",
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.3,
        guided_json: bool = True,
        judge_version: int = COHERENCE_JUDGE_VERSION,
    ):
        assert rubric in COHERENCE_RUBRICS, rubric
        assert judge_version in (1, 2), judge_version
        assert runs and "real" not in runs, "'real' is the label of real_data"
        self.runs = {str(k): v for k, v in sorted(runs.items())}
        self.real_data = real_data
        self.rubric = rubric
        self.rubric_digest = rubric_digest(rubric)  # hashed: an edited rubric is a different job
        self.llm_name = llm_name
        self.temperature = float(temperature)
        self.guided_json = bool(guided_json)
        self.judge_version = int(judge_version)
        self.out_rows = self.output_path("rows.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    __sis_hash_exclude__ = {"judge_version": 1}  # v1 jobs keep their pre-2026-10-12 hash

    @classmethod
    def hash(cls, parsed_args):
        """Run labels are not hashed for a new job (``label_free_hash``)."""
        return label_free_hash(
            cls,
            parsed_args,
            canon=lambda a: {**a, "runs": values_canon(a["runs"])},
            base_hash=lambda a: super(SelfPlayCoherenceJudge, cls).hash(_with_rubric_digest(a)),
            labels=lambda a: sorted(a["runs"]),
        )

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def _items(self) -> list[dict]:
        from datasets import load_from_disk

        items, jre_ids = [], set()
        for label, path in self.runs.items():
            ds = load_from_disk(path.get_path()).select_columns(["id", "n_frames", "frame_rate", "words_a", "words_b"])
            for i, r in enumerate(ds):
                dur = int(r["n_frames"]) / float(r["frame_rate"])
                wa, wb = json.loads(r["words_a"]), json.loads(r["words_b"])
                items.append({"label": label, "idx": i, "id": r["id"], "dur": dur, "wa": wa, "wb": wb})
                if not str(r["id"]).startswith("custom:"):
                    jre_ids.add(r["id"])
        if self.real_data is not None:
            ds = load_from_disk(self.real_data.get_path())
            keep = [i for i, x in enumerate(ds["id"]) if x in jre_ids]
            sub = ds.select(keep).select_columns(["id", "duration", "alignments", "alignments_other"])
            for i, r in enumerate(sub):
                items.append(
                    {
                        "label": "real",
                        "idx": i,
                        "id": r["id"],
                        "dur": float(r["duration"]),
                        "wa": r["alignments"] or [],
                        "wb": r["alignments_other"] or [],
                    }
                )
            if len(keep) != len(jre_ids):
                raise RuntimeError(f"real_data holds {len(keep)} of the runs' {len(jre_ids)} JRE windows")
        for it in items:
            it["n_a"] = sum(0 <= float(w["start"]) < it["dur"] for w in it["wa"])
            it["n_b"] = sum(0 <= float(w["start"]) < it["dur"] for w in it["wb"])
            it["transcript"] = two_sided_transcript(it["wa"], it["wb"], it["dur"])
            del it["wa"], it["wb"]
        return items

    def run(self):
        import numpy as np
        from openai import OpenAI

        items = self._items()
        rubric = COHERENCE_RUBRICS[self.rubric]["instruction"]
        llm_name, temp = self.llm_name, self.temperature
        # v1's guided_json was ignored by vLLM: nothing constrained decoding
        constraint = (
            json_schema_format(_JUDGE_JSON_SCHEMA, "coherence") if self.guided_json and self.judge_version >= 2 else {}
        )
        print(
            f"[coherence] {len(items)} conversations over {sorted({it['label'] for it in items})}, "
            f"judge v{self.judge_version} ({'schema enforced' if constraint else 'unconstrained'})",
            flush=True,
        )

        def judge(client, it):
            msg = rubric.format(duration=it["dur"], n_a=it["n_a"], n_b=it["n_b"], transcript=it["transcript"])
            # Seeded by the TRANSCRIPT (not the label), so the same conversation gets the same draw anywhere.
            seed = int(hashlib.md5(it["transcript"].encode()).hexdigest(), 16) % (2**31)
            err = ""
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=llm_name,
                        messages=[{"role": "user", "content": msg}],
                        seed=(seed + attempt) % (2**31),
                        temperature=temp,
                        max_tokens=4096,  # GPT-OSS reasons before it answers
                        **constraint,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                    score = int(parsed["score"])
                    if score not in (1, 2, 3, 4, 5):
                        raise ValueError(f"score {score}")
                    return {"score": score, "reason": " ".join(str(parsed["reason"]).split()), "error": ""}
                except Exception as e:  # API error, empty or unparsable reply: retry
                    err = f"{type(e).__name__}: {e}"[:200]
            return {"score": None, "reason": None, "error": err}

        with vllm_server(llm_name) as url:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            outs = map_concurrent(
                lambda it: judge(client, it),
                items,
                concurrency=JUDGE_CONCURRENCY,
                progress_path=os.path.abspath("progress.json"),
            )
        failed = [(it, o) for it, o in zip(items, outs) if o["score"] is None]
        print(f"[coherence] {len(failed)}/{len(items)} calls failed", flush=True)
        for it, o in failed[:5]:
            print(f"  FAILED {it['label']} #{it['idx']} {it['id']}: {o['error']}", flush=True)
        if len(failed) > 0.02 * len(items):
            raise RuntimeError(f"coherence judging failed on {len(failed)}/{len(items)} calls (> 2%)")

        with open(self.out_rows.get_path(), "w") as f:
            for it, o in zip(items, outs):
                f.write(json.dumps({**it, **o}) + "\n")

        def stats(scores):
            s = [x for x in scores if x is not None]
            if not s:
                return {"n": 0}
            return {
                "n": len(s),
                "mean": float(np.mean(s)),
                "sd": float(np.std(s, ddof=1)) if len(s) > 1 else None,
                "ci95": bootstrap_ci(s),
                "hist": {str(k): s.count(k) for k in (1, 2, 3, 4, 5)},
            }

        summary = {
            "rubric": self.rubric,
            "rubric_digest": self.rubric_digest,
            "llm_name": llm_name,
            "judge_version": self.judge_version,
            "labels": {},
        }
        for label in sorted({it["label"] for it in items}):
            rows = [(it, o) for it, o in zip(items, outs) if it["label"] == label]
            summary["labels"][label] = {
                **stats([o["score"] for _, o in rows]),
                "failed": sum(o["score"] is None for _, o in rows),
                "first32": stats([o["score"] for it, o in rows if it["idx"] < 32]),
                "mute_conv_frac": float(np.mean([it["n_a"] == 0 or it["n_b"] == 0 for it, _ in rows])),
            }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        for label, s in summary["labels"].items():
            ci = s.get("ci95")
            ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""
            print(
                f"[coherence] {label:28s} n={s.get('n')} mean={s.get('mean', float('nan')):.2f} {ci_s} hist={s.get('hist')}",
                flush=True,
            )
