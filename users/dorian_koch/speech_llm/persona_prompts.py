"""Persona (system) prompts + voice prompts for PersonaPlex training on podcast windows.

PersonaPlex conditions on a HYBRID system prompt: a voice sample and a role text. For the released
model, NVIDIA annotated 1,217 h of Fisher with GPT-OSS-120B prompts "at varying detail levels ... to
balance generalization capability with instruction-following precision" (arXiv 2602.06053 v1,
Appendix A). The paper publishes only three example OUTPUTS -- the annotation instruction is not
public -- all shaped "You enjoy having a good conversation." + optional topic sentence + optional
second-person persona facts. This module copies that shape and that model, with our own instruction
(``PROMPT_SPECS``, reproduced verbatim in projects/2026-01-speech-llm/paper_recipes.md).

Four jobs, each keyed on the training row ``id`` written by ``PodcastCodesTrainData``
(``<item_id>@<channel>#w<k>``):

  PodcastWindowContext   CPU. Rebuilds every training window's transcript and speaker identity.
  PersonaPromptGen       GPU (vLLM). Writes the prompts, LONG format: one row per (id, level).
  VoicePromptCodes       CPU. Voice-prompt mimi codes per (episode, diarized speaker).
  AttachPersonaPrompts   CPU. Training rows + ``context`` + ``voice_codes`` (the loader's columns).

Speaker profiles (ablation ab_epad1_persona, 2026-09-24): ``PersonaProfileGen`` (GPU, vLLM) writes ONE profile per
(episode, diarized speaker) -- who the speaker is, no window content -- and ``AttachPersonaProfiles`` (CPU) swaps
it into an attached dataset's prompt lists, nothing else.

Storage is long format on purpose: a new level, spec, LLM or input view is new ROWS in a new job's
output, never a schema change, and prompt sets from any number of jobs simply concatenate.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import map_concurrent, vllm_gpu_mem_gb, vllm_server
from .podcast_ingest import WINDOW_SELECT_KEYS, _moshi_pythonpath, select_windows

#: The paper's Minimal level, verbatim, and the opener every LLM level starts with.
PERSONA_MINIMAL = "You enjoy having a good conversation."

#: A copied phrase of this many words or more between a prompt and the transcript it describes fails
#: the row (retried, then counted). The prompts are meant to say WHAT is talked about, not the words.
MAX_SHARED_NGRAM = 5

_VIEW_DESCRIPTIONS = {
    "assistant": (
        "the transcript of everything THE ASSISTANT says in one excerpt of a real two-person "
        "conversation. The other person's words are not shown."
    ),
    "dialogue": (
        "a transcript of one excerpt of a real two-person conversation. Lines starting with "
        "'Assistant:' are THE ASSISTANT; lines starting with 'Other:' are the other person."
    ),
}

#: Versioned prompt specs. The spec's CONTENT is hashed into every job that uses it (see
#: ``spec_digest``), so editing an instruction yields new jobs instead of silently reusing old output.
PROMPT_SPECS = {
    "persona_v1": {
        "levels": ["minimal", "general", "topic", "detailed"],
        "llm_levels": ["general", "topic", "detailed"],
        "instruction": """You write system prompts for a speech-to-speech conversational AI. The AI will be trained to play ONE participant ("the assistant") in real conversations. Below is {view_description}

Write three prompts that tell the assistant what to talk about in this conversation, at three levels of detail. Each prompt is appended to the fixed opening sentence "You enjoy having a good conversation." -- do NOT repeat that sentence.

- general: ONE short sentence naming only the broad theme, in a few words. Example: "Talk about comedy and life on the road."
- topic: ONE sentence of the form "Have a <tone> conversation about <concrete topics>." naming the tone and the concrete topics the assistant discusses. Example: "Have a casual discussion about eating at home versus dining out."
- detailed: TWO to FOUR sentences: the tone and topics, the specific points or opinions the assistant brings up, and second-person facts about the assistant that the conversation reveals (background, work, experiences, likes and dislikes). Example: "Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings."

Rules for all three:
- Write in the second person ("You ...") and describe ONLY the assistant.
- Describe WHAT is talked about, never HOW it is worded. Do not quote. Do not copy any phrase of four or more consecutive words from the transcript.
- Do not mention a podcast, a show, a host, a transcript, an excerpt, or these instructions.
- Never name the assistant. Name another real person only if the assistant talks about them.
- State only what the transcript supports.

Answer with JSON: {{"general": "...", "topic": "...", "detailed": "..."}}

TRANSCRIPT:
{transcript}""",
    },
}

_PROMPT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "general": {"type": "string", "minLength": 1},
        "topic": {"type": "string", "minLength": 1},
        "detailed": {"type": "string", "minLength": 1},
    },
    "required": ["general", "topic", "detailed"],
    "additionalProperties": False,
}


def spec_digest(name: str) -> str:
    """Content hash of a prompt spec: hashed into every job using it, so an edited spec re-runs."""
    return hashlib.sha256(json.dumps(PROMPT_SPECS[name], sort_keys=True).encode()).hexdigest()[:16]


def _words(text: str) -> list[str]:
    return ["".join(c for c in w.lower() if c.isalnum()) for w in text.split()]


def max_shared_ngram(prompt: str, transcript: str, cap: int = 16) -> int:
    """Length of the longest run of consecutive words the prompt shares with the transcript.

    Word-level, case- and punctuation-insensitive. Capped at ``cap`` (anything that long is a copy).
    """
    p = [w for w in _words(prompt) if w]
    t = [w for w in _words(transcript) if w]
    best = 0
    for n in range(1, min(cap, len(p)) + 1):
        grams = {tuple(t[i : i + n]) for i in range(len(t) - n + 1)}
        if any(tuple(p[i : i + n]) in grams for i in range(len(p) - n + 1)):
            best = n
        else:
            break
    return best


def _seeded_unit(seed: int, key: str) -> float:
    """Deterministic value in [0, 1) from (seed, key). Never ``random``: re-runs must be identical."""
    d = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2**64


def choose_level(level_weights: dict, seed: int, row_id: str) -> str:
    """Pick one level for a row by seeded hash, proportional to ``level_weights`` (sorted by name)."""
    items = sorted((k, float(v)) for k, v in level_weights.items() if float(v) > 0)
    assert items, f"no positive level weight in {level_weights}"
    u = _seeded_unit(seed, row_id) * sum(v for _, v in items)
    acc = 0.0
    for k, v in items:
        acc += v
        if u < acc:
            return k
    return items[-1][0]


def _parts(d) -> list[str]:
    root = d.get_path() if hasattr(d, "get_path") else str(d)
    pd = os.path.join(root, "parts")
    return sorted(os.path.join(pd, n) for n in os.listdir(pd) if not n.endswith(".tmp"))


# ---------------------------------------------------------------------------------------------
# 1a. window context
# ---------------------------------------------------------------------------------------------
class PodcastWindowContext(Job):
    """Per training window: its transcript (both views) and the diarized label of its assistant.

    Recomputes each window EXACTLY as ``PodcastCodesTrainData`` did -- the same ``select_windows``
    rule, the same seeded channel choice, the same frame-aligned bounds and onset-based word
    selection -- and keeps the ids present in the training corpus. Every training id must be
    reproduced; one that is not means this job and the converter disagree, and a prompt would be
    attached to the wrong audio, so that is an error, not a count.

    ``assistant_label``: the words' own ``speaker`` field is the channel's EPISODE-level anchor and is
    wrong for about a quarter of dialogue rows (audio_datasets.md, "Label trap"), so the label is
    taken from the turns instead, by word onsets inside each diarized speaker's turns.

    ``label_rule`` (2026-09-22): ``"joint"`` (default) labels the TWO channels together -- two
    different speakers, the pair with the most onsets in total (:func:`_joint_labels`); a channel
    left without any speaker of its own gets None (no voice -> not swappable). ``"per_channel"``
    (every job before 2026-09-22, pinned by p0/p1) labels each channel alone (:func:`_label_for`),
    so the quiet channel's backchannels, which fall inside the other speaker's turns, took the
    other speaker's label: 1,150 / 12,492 p1 swap rows gave both sides the same voice.

    ``with_other_side=True`` also writes the OTHER channel as a training side: ``user_alignments``
    (its words in exactly the converter's alignment format: onset-selected, re-based to the window
    start, ``speaker="assistant"`` because that is its role when the loader swaps channels),
    ``user_label`` (same turns-based rule) and ``n_user_words``.
    """

    __sis_hash_exclude__ = {"with_other_side": False, "label_rule": "per_channel"}

    def __init__(
        self,
        *,
        train_data: tk.Path,
        dialogue_shard_dirs: list[tk.Path],
        window_select: dict,
        assistant_channel: str = "random",
        selection_seed: int = 0,
        with_other_side: bool = False,
        label_rule: str = "joint",
    ):
        assert assistant_channel in ("a", "b", "random"), assistant_channel
        assert label_rule in ("joint", "per_channel"), label_rule
        assert set(window_select) <= set(WINDOW_SELECT_KEYS), window_select
        self.train_data = train_data
        self.dialogue_shard_dirs = list(dialogue_shard_dirs)
        self.window_select = {k: window_select[k] for k in WINDOW_SELECT_KEYS if k in window_select}
        self.assistant_channel = assistant_channel
        self.selection_seed = int(selection_seed)
        self.with_other_side = bool(with_other_side)
        self.label_rule = label_rule
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 16, "time": 2})

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk

        want_ids = set(load_from_disk(self.train_data.get_path()).select_columns(["id"])["id"])
        print(f"[context] {len(want_ids)} training ids to reproduce", flush=True)
        cols = [
            "item_id",
            "episode_id",
            "duration_sec",
            "frame_rate",
            "n_frames",
            "words_a",
            "words_b",
            "turns_json",
            "meta",
        ]
        rows = []
        for d in self.dialogue_shard_dirs:
            for p in _parts(d):
                for r in load_from_disk(p).select_columns(cols):
                    if self.assistant_channel == "random":
                        d8 = hashlib.sha256(f"{self.selection_seed}:{r['item_id']}".encode()).digest()
                        ch = "a" if d8[0] % 2 == 0 else "b"
                    else:
                        ch = self.assistant_channel
                    other = "b" if ch == "a" else "a"
                    turns = json.loads(r["turns_json"])
                    wins = select_windows(
                        turns, float(r["duration_sec"]), **self.window_select, seed_key=str(r["item_id"])
                    )
                    if not wins:
                        continue
                    fr, n_fr = float(r["frame_rate"]), int(r["n_frames"])
                    words = {c: json.loads(r[f"words_{c}"] or "[]") for c in ("a", "b")}
                    title = (json.loads(r["meta"] or "{}").get("title") or "")[:120]
                    for wi, (a, b) in enumerate(wins):
                        rid = f"{r['item_id']}@{ch}#w{wi}"
                        if rid not in want_ids:
                            continue
                        f0 = int(np.floor(a * fr))
                        f1 = min(n_fr, int(np.ceil(b * fr)))
                        t0, t1 = f0 / fr, f1 / fr

                        def sel(ws):
                            return [w for w in ws if (w["start"] >= t0 or f0 == 0) and w["start"] < t1]

                        wa, wu = sel(words[ch]), sel(words[other])
                        if getattr(self, "label_rule", "per_channel") == "joint":
                            label_a, label_u = _joint_labels(wa, wu, turns)
                        else:
                            label_a, label_u = _label_for(wa, turns), _label_for(wu, turns)
                        extra = {}
                        if getattr(self, "with_other_side", False):
                            extra = {
                                "user_alignments": [
                                    {
                                        "text": w["text"],
                                        "start": float(w["start"]) - t0,
                                        "end": min(float(w["end"]), t1) - t0,
                                        "speaker": "assistant",
                                    }
                                    for w in wu
                                ],
                                "user_label": label_u,
                                "n_user_words": len(wu),
                            }
                        rows.append(
                            {
                                **extra,
                                "id": rid,
                                "item_id": r["item_id"],
                                "episode_id": r["episode_id"],
                                "title": title,
                                "channel": ch,
                                "t0": t0,
                                "t1": t1,
                                "assistant_text": " ".join(w["text"] for w in wa),
                                "user_text": " ".join(w["text"] for w in wu),
                                "dialogue_text": _dialogue_text(wa, wu),
                                "assistant_label": label_a,
                                "n_assistant_words": len(wa),
                            }
                        )
        got = {r["id"] for r in rows}
        missing = want_ids - got
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(want_ids)} training ids were not reproduced (e.g. "
                f"{sorted(missing)[:3]}): this job's window/channel rule differs from the converter's, "
                "so prompts would be attached to the wrong windows."
            )
        n_nolabel = sum(r["assistant_label"] is None for r in rows)
        msg = f"[context] {len(rows)} windows ({getattr(self, 'label_rule', 'per_channel')} labels); {n_nolabel} without an assistant label"
        if getattr(self, "with_other_side", False):
            n_same = sum(r["user_label"] is not None and r["user_label"] == r["assistant_label"] for r in rows)
            msg += f", {sum(r['user_label'] is None for r in rows)} without a user label, {n_same} with both the same"
        print(msg, flush=True)
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())


def _dialogue_text(wa: list[dict], wu: list[dict], gap: float = 1.5) -> str:
    """Both channels as labelled utterances (one channel's words < ``gap`` s apart), ordered by onset."""
    utts = []
    for who, ws in (("Assistant", wa), ("Other", wu)):
        cur = None
        for w in sorted(ws, key=lambda w: w["start"]):
            if cur and w["start"] - cur["end"] < gap:
                cur["words"].append(w["text"])
                cur["end"] = w["end"]
            else:
                cur = {"who": who, "start": w["start"], "end": w["end"], "words": [w["text"]]}
                utts.append(cur)
    utts.sort(key=lambda u: u["start"])
    return "\n".join(f"{u['who']}: {' '.join(u['words'])}" for u in utts)


def _onset_counts(words: list[dict], turns: list[dict]) -> dict[str, int]:
    """Per diarized speaker: how many of these word onsets fall inside that speaker's turns."""
    counts = {}
    for w in words:
        for t in turns:
            if t["start"] <= w["start"] <= t["end"]:
                counts[t["speaker"]] = counts.get(t["speaker"], 0) + 1
    return counts


def _label_for(words: list[dict], turns: list[dict]) -> str | None:
    """The diarized speaker whose turns contain most of these word onsets (None if none do).

    One channel ALONE -- ``label_rule="per_channel"``. Two channels labelled this way can get the same
    speaker (the quiet side's backchannels sit inside the other's turns); see :func:`_joint_labels`.
    """
    counts = _onset_counts(words, turns)
    return max(sorted(counts), key=lambda s: counts[s]) if counts else None


def _joint_labels(words_a: list[dict], words_b: list[dict], turns: list[dict]) -> tuple[str | None, str | None]:
    """Diarized labels for two channels at once: two DIFFERENT speakers, most onsets in total.

    Each channel may only take a speaker that holds at least one of its own onsets, or None. Among
    the pairs with different speakers, the highest total onset count wins; ties prefer labelling both
    channels, then the sorted speaker order (deterministic). When the per-channel choices already
    differ, this is exactly that pair (it maximises each term).
    """
    ca, cb = _onset_counts(words_a, turns), _onset_counts(words_b, turns)
    best, best_key = (None, None), (-1, -1)
    for sa in [*sorted(ca), None]:
        for sb in [*sorted(cb), None]:
            if sa is not None and sa == sb:
                continue
            key = (ca.get(sa, 0) + cb.get(sb, 0), (sa is not None) + (sb is not None))
            if key > best_key:
                best, best_key = (sa, sb), key
    return best


# ---------------------------------------------------------------------------------------------
# 1b. LLM prompt generation
# ---------------------------------------------------------------------------------------------
#: Requests PersonaPromptGen keeps in flight against its vLLM server (see common.map_concurrent for
#: the measurement). Run-side (not hashed): it changes speed, not output (each call is seeded by its
#: row id and side).
PROMPT_GEN_CONCURRENCY = 192


class PersonaPromptGen(Job):
    """Persona prompts for training windows at several detail levels, by an LLM served with vLLM.

    Modelled on ``HfToDialogue`` (vLLM server, ``datasets.map`` concurrency, md5-seeded per-row
    sampling, retries, >1 % failures is fatal) with two differences that matter:

      * the spec's CONTENT is hashed (``spec_digest``), so editing an instruction re-runs;
      * a VERBATIM guard: a prompt sharing a run of ``MAX_SHARED_NGRAM`` or more words with the
        transcript it describes is retried and, if it persists, fails the row.

    ``input_view``: ``assistant`` (only the assistant's words) or ``dialogue`` (both channels,
    labelled). ``sample_n``: a seeded subset for review before a full run (None = every window).
    ``sides``: whose prompts to write, in ONE job (one model load). ``assistant`` = the window's
    assistant channel; ``other`` = the OTHER channel, described with the same assistant-view
    instruction from that speaker's own words only (to the LLM, that speaker is "the assistant").
    Both sides give each channel of a window its own prompt: for training with the channels swapped,
    or for two concurrent models at inference, one per channel. ``other`` requires
    ``input_view="assistant"``.

    Output (long format, one row per (id, side, level)): ``id, prompt_set, level, text, llm_name,
    input_view, spec, max_ngram_overlap``, with ``prompt_set = f"{spec}|{llm_name}|{input_view}"`` for
    the assistant side and that plus ``|other`` for the other side (``prompt_set_for(side)``).
    The ``minimal`` level is the constant ``PERSONA_MINIMAL`` and is written too, so a set is complete.
    """

    def __init__(
        self,
        *,
        context_data: tk.Path,
        input_view: str,
        spec: str = "persona_v1",
        llm_name: str = "openai/gpt-oss-120b",
        sample_n: int | None = None,
        sample_seed: int = 0,
        shard: int | None = None,
        num_shards: int | None = None,
        guided_json: bool = True,
        temperature: float = 0.6,
        sides: tuple = ("assistant",),
    ):
        sides = tuple(sides)
        assert input_view in _VIEW_DESCRIPTIONS, input_view
        assert sides and set(sides) <= {"assistant", "other"} and len(set(sides)) == len(sides), sides
        assert "other" not in sides or input_view == "assistant", "side 'other' is defined for the assistant view"
        assert spec in PROMPT_SPECS, spec
        self.context_data = context_data
        self.input_view = input_view
        self.spec = spec
        self.spec_digest = spec_digest(spec)  # hashed: an edited spec is a different job
        self.llm_name = llm_name
        self.sample_n = sample_n
        self.sample_seed = int(sample_seed)
        self.shard = shard
        self.num_shards = num_shards
        self.guided_json = bool(guided_json)
        self.temperature = float(temperature)
        self.sides = sides
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 6, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    __sis_hash_exclude__ = {"sides": ("assistant",)}

    def prompt_set_for(self, side: str) -> str:
        base = f"{self.spec}|{self.llm_name}|{self.input_view}"
        return base if side == "assistant" else f"{base}|{side}"

    @property
    def prompt_set(self) -> str:
        """The assistant side's set (the one every single-side consumer reads)."""
        return self.prompt_set_for("assistant")

    def completed_fraction(self):
        import glob

        from sisyphus import global_settings as gs

        files = glob.glob(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress_*.json"))
        done, total = 0, None
        for f in files:
            try:
                d = json.load(open(f))
                done += d.get("done", 0)
                total = total or d.get("total")
            except (OSError, ValueError):
                pass
        return min(done / total, 1.0) if total else None

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk
        from openai import OpenAI

        ds = load_from_disk(self.context_data.get_path())
        if self.sample_n is not None and self.sample_n < len(ds):
            idx = np.sort(np.random.default_rng(self.sample_seed).choice(len(ds), self.sample_n, replace=False))
            ds = ds.select(idx.tolist())
        if self.shard is not None and self.num_shards is not None:
            ds = ds.shard(num_shards=self.num_shards, index=self.shard)
        spec = PROMPT_SPECS[self.spec]
        total = len(ds)
        work_dir = os.getcwd()
        view, llm_name, guided, temp = self.input_view, self.llm_name, self.guided_json, self.temperature
        sides = tuple(getattr(self, "sides", ("assistant",)))  # jobs pickled before `sides` existed
        text_col = {
            s: ({"assistant": "assistant_text", "other": "user_text"}[s] if view == "assistant" else "dialogue_text")
            for s in sides
        }
        n_calls = total * len(sides)
        print(
            f"[prompts] {total} windows x sides {sides} = {n_calls} calls, view={view!r}, guided_json={guided}",
            flush=True,
        )

        with vllm_server(llm_name) as url:

            def gen_side(client, row, side):
                transcript = row[text_col[side]]
                msg = spec["instruction"].format(view_description=_VIEW_DESCRIPTIONS[view], transcript=transcript)
                # The assistant side keeps the single-side seed (id only), so it reproduces the review.
                key = row["id"] if side == "assistant" else f"{row['id']}|{side}"
                seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)
                out, err = None, ""
                for attempt in range(5):
                    try:
                        kw = {"extra_body": {"guided_json": _PROMPT_JSON_SCHEMA}} if guided else {}
                        resp = client.chat.completions.create(
                            model=llm_name,
                            messages=[{"role": "user", "content": msg}],
                            seed=(seed + attempt) % (2**31),
                            temperature=temp,
                            max_tokens=4096,  # GPT-OSS spends tokens on reasoning before the answer
                            **kw,
                        )
                        raw = (resp.choices[0].message.content or "").strip()
                        if raw.startswith("```"):
                            raw = raw.strip("`").split("\n", 1)[-1]
                        parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
                        cand = {k: " ".join(str(parsed[k]).split()) for k in spec["llm_levels"]}
                    except Exception as e:  # API error, empty or unparsable reply: retry
                        err = f"{type(e).__name__}: {e}"[:200]
                        continue
                    overlaps = {k: max_shared_ngram(v, transcript) for k, v in cand.items()}
                    if max(overlaps.values()) >= MAX_SHARED_NGRAM:
                        err = f"verbatim overlap {overlaps}"
                        continue
                    out = (cand, overlaps)
                    break
                if out is None:
                    return {"texts": None, "overlaps": None, "error": err}
                return {"texts": json.dumps(out[0]), "overlaps": json.dumps(out[1]), "error": ""}

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            calls = [(row, s) for row in ds for s in sides]
            outs = map_concurrent(
                lambda rs: gen_side(client, rs[0], rs[1]),
                calls,
                concurrency=PROMPT_GEN_CONCURRENCY,
                progress_path=os.path.join(work_dir, "progress_0.json"),
            )

        per = [(row["id"], s, v) for (row, s), v in zip(calls, outs)]
        failed = [(rid, s, v) for rid, s, v in per if v["texts"] is None]
        print(f"[prompts] {len(failed)}/{n_calls} (window, side) calls failed", flush=True)
        for rid, s, v in failed[:5]:
            print(f"  FAILED {rid} [{s}]: {v['error']}", flush=True)
        if len(failed) > 0.01 * n_calls:
            raise RuntimeError(f"prompt generation failed on {len(failed)}/{n_calls} calls (> 1%); see the log")

        out = {
            k: [] for k in ("id", "prompt_set", "level", "text", "llm_name", "input_view", "spec", "max_ngram_overlap")
        }

        def add(rid, side, level, text, ov):
            for k, v in (
                ("id", rid),
                ("prompt_set", self.prompt_set_for(side)),
                ("level", level),
                ("text", text),
                ("llm_name", llm_name),
                ("input_view", view),
                ("spec", self.spec),
                ("max_ngram_overlap", ov),
            ):
                out[k].append(v)

        lens = {s: {k: [] for k in spec["llm_levels"]} for s in sides}
        for rid, s, v in per:
            if v["texts"] is None:
                continue
            texts, ovs = json.loads(v["texts"]), json.loads(v["overlaps"])
            add(rid, s, "minimal", PERSONA_MINIMAL, 0)
            for k in spec["llm_levels"]:
                add(rid, s, k, f"{PERSONA_MINIMAL} {texts[k]}", int(ovs[k]))
                lens[s][k].append(len(texts[k].split()))
        Dataset.from_dict(out).save_to_disk(self.out_dir.get_path())
        summary = {
            "prompt_sets": {s: self.prompt_set_for(s) for s in sides},
            "spec_digest": self.spec_digest,
            "rows": total,
            "failed": {s: sum(1 for _, fs, _ in failed if fs == s) for s in sides},
            "words_per_level_median": {
                s: {k: float(np.median(v)) if v else None for k, v in lens[s].items()} for s in sides
            },
        }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[prompts] {summary}", flush=True)


# ---------------------------------------------------------------------------------------------
# 1c. voice-prompt codes
# ---------------------------------------------------------------------------------------------
class VoicePromptCodes(Job):
    """Voice-prompt mimi codes per (episode, diarized speaker) from the stored ``speaker_ref_audio``.

    Runs ``moshi_family.persona_voice_codes`` in the moshi_family venv, which rebuilds each prompt
    with PersonaPlex's OWN inference functions (resample, -24 LUFS, frame iteration, streaming mimi
    encode), trimmed to ``voice_sec`` (4.0 s = the length of the voices PersonaPlex ships).
    """

    def __init__(self, *, episode_shard_dirs: list[tk.Path], venv_python_path: tk.Path, voice_sec: float = 4.0):
        self.episode_shard_dirs = list(episode_shard_dirs)
        self.venv_python_path = venv_python_path
        self.voice_sec = float(voice_sec)
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 8, "mem": 32, "time": 4})

    def run(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        parts_dirs = [os.path.join(d.get_path(), "parts") for d in self.episode_shard_dirs]
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.persona_voice_codes",
            "--out",
            self.out_dir.get_path(),
            "--voice_sec",
            str(self.voice_sec),
        ] + parts_dirs
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


# ---------------------------------------------------------------------------------------------
# 1d. attach prompts + voices to the training rows
# ---------------------------------------------------------------------------------------------
class AttachPersonaPrompts(Job):
    """Training rows + ``context`` (persona text) + ``voice_codes``: what the PersonaPlex loader reads.

    Per row, ONE level is chosen by seeded hash of the row id, proportional to ``level_weights``.
    With ``per_read=True`` the choice moves to TRAINING time instead: ``context`` holds every level's
    prompt as a list (level order sorted by name, ``context_level`` the matching names) and the
    PersonaPlex loader draws one uniformly each time it reads the row, so a row seen in several epochs
    can get a different level each time. Uniform, so the weights must all be equal; a row missing any
    of the levels counts as ``noprompt``.
    ``minimal`` needs no prompt set (it is the constant ``PERSONA_MINIMAL``), so an all-minimal corpus
    does not wait on an LLM job. The voice is the row's assistant speaker's clip from
    ``VoicePromptCodes`` (matched on episode + diarized label from ``PodcastWindowContext``).

    Rows with no prompt (for a non-minimal level) or no voice clip are dropped and COUNTED; more than
    ``max_drop_frac`` of either is an error rather than a quietly smaller corpus.

    Extra columns for analysis: ``context_level``, ``context_set``, ``voice_label``, ``voice_n_frames``.

    ``with_other_side=True`` (needs ``per_read`` and a context built with ``with_other_side``) also
    stores the OTHER channel as a second training side, for the loader's ``swap_channels``:
    ``context_other`` (its levels' prompts from ``other_prompt_set``), ``voice_codes_other`` /
    ``voice_n_frames_other`` / ``voice_label_other`` (that speaker's clip), ``alignments_other`` and
    ``other_ok``. A row whose other side has fewer than ``other_min_words`` words (the converter's
    ``min_assistant_words``), or lacks a prompt or voice for it, is KEPT with ``other_ok=False`` and
    empty ``*_other`` columns, so the loader never swaps it; the count is logged.
    ``reject_same_voice`` (default; ``False`` = every job before 2026-09-22, pinned by p0/p1): a row
    whose two sides carry the SAME diarized label (hence the same voice prompt) is not swappable
    either (``other_samevoice``) -- a model told "you sound like X" while hearing X would learn to
    ignore its voice prompt. With a ``label_rule="joint"`` context this should never fire.
    """

    def __init__(
        self,
        *,
        train_data: tk.Path,
        context_data: tk.Path,
        voice_codes: tk.Path,
        level_weights: dict,
        prompt_data: list[tk.Path] | None = None,
        prompt_set: str | None = None,
        seed: int = 0,
        max_drop_frac: float = 0.05,
        per_read: bool = False,
        with_other_side: bool = False,
        other_prompt_set: str | None = None,
        other_min_words: int = 8,
        reject_same_voice: bool = True,
    ):
        if with_other_side:
            assert per_read, "with_other_side stores prompt lists; it needs per_read=True"
            assert other_prompt_set, "with_other_side needs other_prompt_set"
        levels = {k for k, v in level_weights.items() if float(v) > 0}
        assert levels, level_weights
        if per_read:
            w = {float(v) for v in level_weights.values() if float(v) > 0}
            assert len(w) == 1, f"per_read draws uniformly in the loader; weights must be equal: {level_weights}"
        if levels != {"minimal"}:
            assert prompt_data and prompt_set, "non-minimal levels need prompt_data + prompt_set"
        self.train_data = train_data
        self.context_data = context_data
        self.voice_codes = voice_codes
        self.level_weights = {k: float(v) for k, v in sorted(level_weights.items())}
        self.prompt_data = list(prompt_data or [])
        self.prompt_set = prompt_set
        self.seed = int(seed)
        self.max_drop_frac = float(max_drop_frac)
        self.per_read = bool(per_read)
        self.with_other_side = bool(with_other_side)
        self.other_prompt_set = other_prompt_set
        self.other_min_words = int(other_min_words)
        self.reject_same_voice = bool(reject_same_voice)
        self.out_dir = self.output_path("dataset", directory=True)

    __sis_hash_exclude__ = {
        "per_read": False,
        "with_other_side": False,
        "other_prompt_set": None,
        "other_min_words": 8,
        "reject_same_voice": False,
    }

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 24, "time": 2})

    def run(self):
        import numpy as np
        from datasets import Dataset, Sequence, Value, load_from_disk

        other_side = getattr(self, "with_other_side", False)
        ctx_cols = ["id", "episode_id", "assistant_label"]
        if other_side:
            ctx_cols += ["user_alignments", "user_label", "n_user_words"]
        ctx = {r["id"]: r for r in load_from_disk(self.context_data.get_path()).select_columns(ctx_cols)}
        voices = {}
        for r in load_from_disk(self.voice_codes.get_path()):
            voices[(r["episode_id"], r["label"])] = (np.asarray(r["codes"], dtype=np.int16), int(r["n_frames"]))
        prompts, other_prompts = {}, {}
        for p in self.prompt_data:
            for r in load_from_disk(p.get_path()):
                if r["prompt_set"] == self.prompt_set:
                    prompts.setdefault(r["id"], {})[r["level"]] = r["text"]
                elif other_side and r["prompt_set"] == self.other_prompt_set:
                    other_prompts.setdefault(r["id"], {})[r["level"]] = r["text"]
        train = load_from_disk(self.train_data.get_path())
        print(
            f"[attach] {len(train)} rows, {len(voices)} voice clips, {len(prompts)} prompted ids, "
            f"levels={self.level_weights}",
            flush=True,
        )

        n = {"in": 0, "out": 0, "noprompt": 0, "novoice": 0}
        levels_used = {}
        feats = train.features.copy()
        feats["context"] = Sequence(Value("string")) if self.per_read else Value("string")
        feats["context_level"] = Sequence(Value("string")) if self.per_read else Value("string")
        read_levels = sorted(k for k, v in self.level_weights.items() if v > 0)

        def text_of(rid, level):
            if level == "minimal":
                return PERSONA_MINIMAL, "builtin"
            return prompts.get(rid, {}).get(level), self.prompt_set

        feats["context_set"] = Value("string")
        feats["voice_codes"] = Sequence(Value("int16"))
        feats["voice_n_frames"] = Value("int32")
        feats["voice_label"] = Value("string")
        if other_side:
            feats["context_other"] = Sequence(Value("string"))
            feats["voice_codes_other"] = Sequence(Value("int16"))
            feats["voice_n_frames_other"] = Value("int32")
            feats["voice_label_other"] = Value("string")
            feats["alignments_other"] = train.features["alignments"]
            feats["other_ok"] = Value("bool")
            n.update({"other_ok": 0, "other_short": 0, "other_noprompt": 0, "other_novoice": 0, "other_samevoice": 0})

        def other_cols(rid, c):
            """The other side's columns, or empty ones with other_ok=False (and the reason counted)."""
            empty = {
                "context_other": [],
                "voice_codes_other": np.zeros(0, dtype=np.int16),
                "voice_n_frames_other": 0,
                "voice_label_other": c["user_label"],
                "alignments_other": [],
                "other_ok": False,
            }
            if c["n_user_words"] < self.other_min_words:
                n["other_short"] += 1
                return empty
            if getattr(self, "reject_same_voice", False) and c["user_label"] == c["assistant_label"]:
                n["other_samevoice"] += 1
                return empty
            texts = [PERSONA_MINIMAL if lv == "minimal" else other_prompts.get(rid, {}).get(lv) for lv in read_levels]
            if any(t is None for t in texts):
                n["other_noprompt"] += 1
                return empty
            ov = voices.get((c["episode_id"], c["user_label"]))
            if ov is None:
                n["other_novoice"] += 1
                return empty
            n["other_ok"] += 1
            return {
                "context_other": texts,
                "voice_codes_other": ov[0],
                "voice_n_frames_other": ov[1],
                "voice_label_other": c["user_label"],
                "alignments_other": c["user_alignments"],
                "other_ok": True,
            }

        def rows():
            for r in train:
                n["in"] += 1
                c = ctx[r["id"]]  # KeyError = the context job does not cover this corpus: a wiring bug
                if self.per_read:
                    level = read_levels
                    text = [text_of(r["id"], lv)[0] for lv in read_levels]
                    pset = self.prompt_set if any(lv != "minimal" for lv in read_levels) else "builtin"
                    if any(t is None for t in text):
                        n["noprompt"] += 1
                        continue
                else:
                    level = choose_level(self.level_weights, self.seed, r["id"])
                    text, pset = text_of(r["id"], level)
                    if text is None:
                        n["noprompt"] += 1
                        continue
                v = voices.get((c["episode_id"], c["assistant_label"]))
                if v is None:
                    n["novoice"] += 1
                    continue
                for lv in level if self.per_read else [level]:
                    levels_used[lv] = levels_used.get(lv, 0) + 1
                n["out"] += 1
                yield {
                    **r,
                    "context": text,
                    "context_level": level,
                    "context_set": pset,
                    "voice_codes": v[0],
                    "voice_n_frames": v[1],
                    "voice_label": c["assistant_label"],
                    **(other_cols(r["id"], c) if other_side else {}),
                }

        out = Dataset.from_generator(rows, features=feats, cache_dir=os.path.abspath("hf_gen_cache"))
        print(f"[attach] {n}; levels used {levels_used}", flush=True)
        for k in ("noprompt", "novoice"):
            if n[k] > self.max_drop_frac * n["in"]:
                raise RuntimeError(f"{n[k]}/{n['in']} rows dropped for {k} (> {self.max_drop_frac:.0%}): {n}")
        out.save_to_disk(self.out_dir.get_path())


# ---------------------------------------------------------------------------------------------
# 2. speaker PROFILES (ablation ab_epad1_persona, 2026-09-24)
# ---------------------------------------------------------------------------------------------
#: Versioned speaker-profile specs (hashed by content, like ``PROMPT_SPECS``). Unlike ``persona_v1``, which
#: summarises the assistant's words of ONE window (the model learns to recite its prompt: 28-33% of a topic
#: prompt's content words are said in that window, ~1% in another window of the same episode), a profile is
#: written ONCE per (episode, diarized speaker) from that speaker's words across the whole episode and says who
#: the speaker is, never what they say (user, 2026-09-24: "only have the system prompt describe the speaker,
#: not what the speaker says").
PROFILE_SPECS = {
    "persona_profile_v1": {
        "instruction": """You write system prompts for a speech-to-speech conversational AI. The AI will be trained to play ONE participant ("the assistant") in real conversations. Below are excerpts of everything THE ASSISTANT says across one long conversation, in order. The other person's words are not shown.

Write ONE short profile of the assistant as a SPEAKER: who they are and how they talk, not what they talk about. It is appended to the fixed opening sentence "You enjoy having a good conversation." -- do NOT repeat that sentence.

The profile is TWO or THREE sentences in the second person ("You ...") and covers only:
- their role in a conversation, e.g. whether they mostly ask questions and steer, or mostly answer, tell stories and explain;
- their speaking style, e.g. casual or formal, blunt, joking, enthusiastic, measured, uses slang or swears;
- their background or expertise in broad terms only, as a profession or field, e.g. "You are a comedian" or "You have a background in medicine".

Strict rules:
- Do NOT mention any topic, subject, claim, opinion, story, anecdote, event, product, place, date, number or name from the conversation. The profile must fit EVERY part of this conversation equally well, so it must say nothing about what is discussed.
- Background and expertise only as a broad profession or field, never a specific experience, achievement or fact that the conversation reveals.
- Do not quote. Do not copy any phrase of four or more consecutive words from the excerpts.
- Do not mention a podcast, a show, a host, a guest, an interview, a transcript, an excerpt, or these instructions.
- Never name the assistant or anyone else.

Answer with JSON: {{"profile": "..."}}

EXCERPTS:
{transcript}""",
    },
}

_PROFILE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"profile": {"type": "string", "minLength": 1}},
    "required": ["profile"],
    "additionalProperties": False,
}


def profile_spec_digest(name: str) -> str:
    """Content hash of a profile spec (hashed into ``PersonaProfileGen``)."""
    return hashlib.sha256(json.dumps(PROFILE_SPECS[name], sort_keys=True).encode()).hexdigest()[:16]


def _item_order(item_id: str) -> tuple:
    """Dialogue rows of an episode in order: ``<episode>#<row index>``."""
    head, _, tail = str(item_id).rpartition("#")
    return (head, int(tail)) if tail.isdigit() else (str(item_id), -1)


def speaker_excerpts(ctx_rows, max_words: int) -> dict:
    """``{(episode_id, label): (excerpt text, n_words_total, n_windows_total, n_windows_used)}`` from a window
    context built ``with_other_side=True``: each window contributes its assistant channel's words to
    ``(episode, assistant_label)`` and its other channel's words to ``(episode, user_label)``, in dialogue order.

    Over ``max_words``, the windows are thinned to the largest EVENLY SPACED subset that fits (so the excerpts
    still span the whole episode), never truncated mid-window. One line per window."""
    import numpy as np

    per = {}
    for r in ctx_rows:
        for lab, text in ((r["assistant_label"], r["assistant_text"]), (r.get("user_label"), r.get("user_text"))):
            if lab is None or not (text or "").strip():
                continue
            per.setdefault((r["episode_id"], lab), []).append((_item_order(r["item_id"]), float(r["t0"]), text.strip()))
    out = {}
    for key, wins in per.items():
        wins.sort(key=lambda w: (w[0], w[1]))
        lens = [len(w[2].split()) for w in wins]
        total = sum(lens)
        n = len(wins)

        def pick(m):
            return sorted({int(round(x)) for x in np.linspace(0, n - 1, m)}) if m > 0 else []

        lo, hi = 1, n  # largest m whose evenly spaced subset fits (always keep >= 1 window)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sum(lens[i] for i in pick(mid)) <= max_words:
                lo = mid
            else:
                hi = mid - 1
        idx = pick(lo)
        out[key] = ("\n".join(wins[i][2] for i in idx), total, n, len(idx))
    return out


class PersonaProfileGen(Job):
    """One speaker PROFILE per (episode, diarized speaker) by an LLM served with vLLM (ablation
    ``ab_epad1_persona``, 2026-09-24): role, speaking style and broad background, explicitly no topics, claims
    or events (``PROFILE_SPECS``), written from that speaker's words across ALL windows of the episode in
    ``context_data`` (a ``PodcastWindowContext(with_other_side=True)``; thinned evenly to ``max_words``). The
    instruction text is hashed (``hash``).

    The same key as ``VoicePromptCodes`` and ``AttachPersonaPrompts`` use for the voice (episode + the
    context's diarized label), so every window of one speaker gets the SAME profile. Same machinery as
    ``PersonaPromptGen``: seeded per key, retries, verbatim guard (a shared run of ``MAX_SHARED_NGRAM`` words
    with the excerpts fails the attempt), > 1% failed keys is fatal.

    Output: ``episode_id, label, text`` (``PERSONA_MINIMAL`` + profile), ``profile``, ``n_words``, ``n_windows``,
    ``n_windows_used``, ``max_ngram_overlap``, ``spec``, ``llm_name``; ``summary.json``.

    ``episodes``: profile only these episodes (all their windows, so each profile is what the full run would
    write for that key); None = every episode (not hashed then, so existing jobs keep their hash).
    """

    def __init__(
        self,
        *,
        context_data: tk.Path,
        spec: str = "persona_profile_v1",
        llm_name: str = "openai/gpt-oss-120b",
        max_words: int = 3000,
        temperature: float = 0.6,
        episodes: list[str] | None = None,
    ):
        assert spec in PROFILE_SPECS, spec
        self.context_data = context_data
        self.episodes = sorted(set(episodes)) if episodes else None
        self.spec = spec
        self.spec_digest = profile_spec_digest(spec)
        self.llm_name = llm_name
        self.max_words = int(max_words)
        self.temperature = float(temperature)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The spec's CONTENT is part of the hash (Sisyphus hashes only ``__init__`` arguments, so an attribute
        alone would not be): an edited instruction is a different job."""
        d = {**parsed_args, "spec_digest": profile_spec_digest(parsed_args["spec"])}
        if d.get("episodes") is None:
            d.pop("episodes", None)
        return super(PersonaProfileGen, cls).hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @property
    def prompt_set(self) -> str:
        return f"{self.spec}|{self.llm_name}|episode_speaker"

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress_0.json")))
            return min(d["done"] / d["total"], 1.0) if d.get("total") else None
        except (OSError, ValueError, KeyError):
            return None

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk
        from openai import OpenAI

        cols = ["id", "item_id", "episode_id", "t0", "assistant_label", "assistant_text", "user_label", "user_text"]
        ctx = load_from_disk(self.context_data.get_path()).select_columns(cols)
        if self.episodes:
            keep = set(self.episodes)
            eps = ctx["episode_id"]
            missing = keep - set(eps)
            if missing:
                raise RuntimeError(f"{len(missing)} requested episodes not in the context: {sorted(missing)[:5]}")
            ctx = ctx.select([i for i, e in enumerate(eps) if e in keep])
            print(f"[profiles] restricted to {len(keep)} episodes: {len(ctx)} windows", flush=True)
        exc = speaker_excerpts(ctx, self.max_words)
        keys = sorted(exc)
        spec, llm_name, temp = PROFILE_SPECS[self.spec], self.llm_name, self.temperature
        print(
            f"[profiles] {len(ctx)} windows -> {len(keys)} (episode, speaker) keys; words/key median "
            f"{np.median([exc[k][1] for k in keys]):.0f}, thinned to <= {self.max_words} for "
            f"{sum(exc[k][3] < exc[k][2] for k in keys)} keys",
            flush=True,
        )

        with vllm_server(llm_name) as url:

            def gen(client, key):
                transcript = exc[key][0]
                msg = spec["instruction"].format(transcript=transcript)
                seed = int(hashlib.md5(f"{key[0]}|{key[1]}".encode()).hexdigest(), 16) % (2**31)
                err = ""
                for attempt in range(5):
                    try:
                        resp = client.chat.completions.create(
                            model=llm_name,
                            messages=[{"role": "user", "content": msg}],
                            seed=(seed + attempt) % (2**31),
                            temperature=temp,
                            max_tokens=4096,  # GPT-OSS reasons before answering
                            extra_body={"guided_json": _PROFILE_JSON_SCHEMA},
                        )
                        raw = (resp.choices[0].message.content or "").strip()
                        prof = " ".join(str(json.loads(raw[raw.find("{") : raw.rfind("}") + 1])["profile"]).split())
                    except Exception as e:  # API error, empty or unparsable reply: retry
                        err = f"{type(e).__name__}: {e}"[:200]
                        continue
                    ov = max_shared_ngram(prof, transcript)
                    if ov >= MAX_SHARED_NGRAM:
                        err = f"verbatim overlap {ov}"
                        continue
                    return {"profile": prof, "overlap": ov, "error": ""}
                return {"profile": None, "overlap": None, "error": err}

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            outs = map_concurrent(
                lambda k: gen(client, k),
                keys,
                concurrency=PROMPT_GEN_CONCURRENCY,
                progress_path=os.path.join(os.getcwd(), "progress_0.json"),
            )

        failed = [(k, o) for k, o in zip(keys, outs) if o["profile"] is None]
        print(f"[profiles] {len(failed)}/{len(keys)} keys failed", flush=True)
        for k, o in failed[:5]:
            print(f"  FAILED {k}: {o['error']}", flush=True)
        if len(failed) > 0.01 * len(keys):
            raise RuntimeError(f"profile generation failed on {len(failed)}/{len(keys)} keys (> 1%); see the log")
        rows = [
            {
                "episode_id": k[0],
                "label": k[1],
                "text": f"{PERSONA_MINIMAL} {o['profile']}",
                "profile": o["profile"],
                "n_words": exc[k][1],
                "n_windows": exc[k][2],
                "n_windows_used": exc[k][3],
                "max_ngram_overlap": int(o["overlap"]),
                "spec": self.spec,
                "llm_name": llm_name,
            }
            for k, o in zip(keys, outs)
            if o["profile"] is not None
        ]
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())
        lens = [len(r["profile"].split()) for r in rows]
        summary = {
            "prompt_set": self.prompt_set,
            "spec_digest": self.spec_digest,
            "keys": len(keys),
            "failed": len(failed),
            "profile_words_median": float(np.median(lens)) if lens else None,
            "profile_words_p90": float(np.percentile(lens, 90)) if lens else None,
            "examples": [r["text"] for r in rows[:5]],
        }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[profiles] {summary}", flush=True)


def profile_text(profile: str) -> str:
    """``PERSONA_MINIMAL`` + the profile, with any opener the LLM repeated despite the instruction removed (49 of 690
    ``persona_profile_v1`` replies on p3's corpus began with it; ``PersonaProfileGen``'s ``text`` keeps them raw)."""
    body = " ".join(profile.split())
    while body.startswith(PERSONA_MINIMAL):
        body = body[len(PERSONA_MINIMAL) :].strip()
    return f"{PERSONA_MINIMAL} {body}"


def profile_prompt_lists(row: dict, episode_id: str, profiles: dict, levels: tuple) -> tuple:
    """``(context, context_other)`` of one attached row with every non-minimal level replaced by the speaker's
    profile: ``PERSONA_MINIMAL`` for ``minimal``, ``profiles[(episode, voice label)]`` otherwise; ``None`` for a
    side whose profile is missing (the other side only when ``other_ok``)."""
    a = profiles.get((episode_id, row["voice_label"]))
    ctx = None if a is None else [PERSONA_MINIMAL if lv == "minimal" else a for lv in levels]
    oth = []
    if row.get("other_ok"):
        b = profiles.get((episode_id, row["voice_label_other"]))
        oth = None if b is None else [PERSONA_MINIMAL if lv == "minimal" else b for lv in levels]
    return ctx, oth


class AttachPersonaProfiles(Job):
    """An ``AttachPersonaPrompts(per_read=True, with_other_side=True)`` dataset with ONLY its prompt lists
    replaced: ``context`` / ``context_other`` become ``levels`` (``minimal`` = ``PERSONA_MINIMAL``, any other name =
    that side's ``PersonaProfileGen`` profile, keyed by the row's episode (from ``context_data``) + its voice
    label). Codes, alignments, voices, other side, row order: unchanged. The loader draws one list element
    uniformly per read, so ``("profile", "minimal", "profile")`` keeps p3's mix (``general, minimal, topic``,
    sorted by name) of 1/3 minimal and 2/3 an LLM prompt AT THE SAME LIST POSITIONS: with the same rows the
    loader's random stream is unchanged, so every read that drew the minimal prompt in ab_epad1 draws it here
    too, and every read that drew general/topic gets the profile (ablation ``ab_epad1_persona``).

    ``keep_all_rows=False`` (training): a row without its assistant profile is dropped, an ``other_ok`` row
    without the other side's profile becomes unswappable; counted, > ``max_drop_frac`` is an error.
    ``keep_all_rows=True`` (self-play inputs, which pick rows by index among ``other_ok``): every row stays; an
    ``other_ok`` row missing either profile is an error, a row that is not ``other_ok`` and lacks its profile
    keeps ``PERSONA_MINIMAL`` in the profile slots (never picked by self-play; counted).
    """

    #: 2 (2026-09-24, before any consumer): the profile text via ``profile_text`` (drops an opener the LLM repeated).
    __sis_version__ = 2

    def __init__(
        self,
        *,
        data: tk.Path,
        context_data: tk.Path,
        profiles: tk.Path,
        prompt_set: str,
        levels: tuple = ("profile", "minimal", "profile"),
        keep_all_rows: bool = False,
        max_drop_frac: float = 0.05,
    ):
        assert levels and "minimal" in levels and any(lv != "minimal" for lv in levels), levels
        self.data = data
        self.context_data = context_data
        self.profiles = profiles
        self.prompt_set = prompt_set
        self.levels = tuple(levels)
        self.keep_all_rows = bool(keep_all_rows)
        self.max_drop_frac = float(max_drop_frac)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 24, "time": 2})

    def run(self):
        from datasets import Dataset, load_from_disk

        ep_of = dict(
            zip(
                *[
                    load_from_disk(self.context_data.get_path()).select_columns(["id", "episode_id"])[c]
                    for c in ("id", "episode_id")
                ]
            )
        )
        profiles = {
            (r["episode_id"], r["label"]): profile_text(r["profile"]) for r in load_from_disk(self.profiles.get_path())
        }
        src = load_from_disk(self.data.get_path())
        assert "context_other" in src.column_names, "needs an AttachPersonaPrompts(with_other_side=True) dataset"
        levels = list(self.levels)
        n = {"in": 0, "out": 0, "noprofile": 0, "other_noprofile": 0, "kept_minimal": 0}

        def rows():
            for r in src:
                n["in"] += 1
                ctx, oth = profile_prompt_lists(r, ep_of[r["id"]], profiles, self.levels)
                if oth is None:
                    if self.keep_all_rows:
                        raise RuntimeError(f"{r['id']}: other_ok row without the other side's profile")
                    n["other_noprofile"] += 1
                    r = {
                        **r,
                        "other_ok": False,
                        "voice_codes_other": [],
                        "voice_n_frames_other": 0,
                        "alignments_other": [],
                    }
                    oth = []
                if ctx is None:
                    if not self.keep_all_rows:
                        n["noprofile"] += 1
                        continue
                    if r["other_ok"]:
                        raise RuntimeError(f"{r['id']}: other_ok row without its assistant profile")
                    n["kept_minimal"] += 1
                    ctx = [PERSONA_MINIMAL for _ in levels]
                n["out"] += 1
                yield {
                    **r,
                    "context": ctx,
                    "context_level": levels,
                    "context_set": self.prompt_set,
                    "context_other": oth,
                }

        out = Dataset.from_generator(rows, features=src.features, cache_dir=os.path.abspath("hf_gen_cache"))
        print(f"[attach-profiles] {n}; levels {levels}", flush=True)
        for k in ("noprofile", "other_noprofile"):
            if n[k] > self.max_drop_frac * n["in"]:
                raise RuntimeError(f"{n[k]}/{n['in']} rows hit {k} (> {self.max_drop_frac:.0%}): {n}")
        out.save_to_disk(self.out_dir.get_path())
        json.dump(
            {**n, "levels": levels, "prompt_set": self.prompt_set}, open(self.out_summary.get_path(), "w"), indent=2
        )


# ---------------------------------------------------------------------------------------------
# 3. short CONTENT prompts from a large chunk / from the first half (ablations ab_epad1_chunkprompt /
#    ab_epad1_firsthalf, 2026-09-25)
# ---------------------------------------------------------------------------------------------
#: One instruction for BOTH content specs, so the two ablations differ only in the transcript the LLM is shown (user,
#: 2026-09-25: keep content prompts, but condense much more; one short sentence a person would plausibly type). The
#: two style examples are PersonaPlex's published topic prompts (arXiv 2602.06053 v1 Appendix A; the repo README).
CONTENT_PROMPT_INSTRUCTION = """You write system prompts for a speech-to-speech conversational AI. The AI will be trained to play ONE participant ("the assistant") in real conversations. Below is a transcript of part of a real two-person conversation. Lines starting with "Assistant:" are THE ASSISTANT; lines starting with "Other:" are the other person.

Write ONE short sentence that tells the assistant what to talk about: the broad subject of this conversation from the assistant's side, the way a person would type it when asking for a conversation. It is appended to the fixed opening sentence "You enjoy having a good conversation." -- do NOT repeat that sentence. Examples of the style:
- "Have a casual discussion about eating at home versus dining out."
- "Have an empathetic discussion about the meaning of family amid uncertainty."

Rules:
- At most {max_words} words, ONE sentence.
- Name only the broad subject: one or two themes that fit the whole transcript. Leave out the individual points, claims, opinions, stories, examples, names, places and numbers.
- Describe only what the assistant talks about.
- Do not quote. Do not copy any phrase of four or more consecutive words from the transcript.
- Do not mention a podcast, a show, a host, a guest, an interview, a transcript, an excerpt, or these instructions.
- Never name the assistant or anyone else.

Answer with JSON: {{"prompt": "..."}}

TRANSCRIPT:
{transcript}"""

#: Versioned content specs (the whole dict entry is hashed into ``PersonaContentPromptGen``). ``source``:
#:   ``chunk``: the fixed ``chunk_sec`` block of EPISODE time holding the window's midpoint (extended to cover every
#:     window assigned to it), both speakers' words, one prompt per (episode, block, speaker) shared by all its
#:     windows. 600 s = a Fisher call: PersonaPlex's real-data stage has ONE prompt per call, 7,303 calls / 1,217 h
#:     = 10.0 min each, while its training crops are <= 163.84 s (paper_recipes.md).
#:   ``firsthalf``: the training window's own words with onset in its FIRST half (both speakers); nothing said in the
#:     second half reaches the LLM.
#: ``max_words``: the prompt body (after the opener) is at most this many words; longer replies are retried.
CONTENT_SPECS = {
    "content_chunk600_v1": {
        "instruction": CONTENT_PROMPT_INSTRUCTION,
        "source": "chunk",
        "chunk_sec": 600.0,
        "max_words": 15,
    },
    "content_firsthalf_v1": {
        "instruction": CONTENT_PROMPT_INSTRUCTION,
        "source": "firsthalf",
        "max_words": 15,
    },
    # v1's instruction and source at 3.5x the throughput (p6, 2026-09-25; user: prompt generation too expensive).
    # Measured on 500 v1 calls (analysis/ppx_promptgen_profile.py, test node H100): v1 as run 19.0 calls/s, 205
    # completion tokens/call; reasoning_effort low + 1024 in flight 67.2 calls/s, 85 tokens/call, 0 failed, words
    # p10/p50/p90 10/11/13 (v1 10/12/14); leak on the read-out's 200 rows own 15.5% [13.0, 17.9] (v1 19.4),
    # own_2nd_only 0.9% [0.4, 1.6] (v1 1.0). gpt-oss-20b at low effort was no faster (74.6 calls/s, 3x retries).
    "content_firsthalf_v2": {
        "instruction": CONTENT_PROMPT_INSTRUCTION,
        "source": "firsthalf",
        "max_words": 15,
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "concurrency": 1024,
    },
    # Several detail levels from ONE call per (window, side), all written from the window's FIRST HALF only (p6, user
    # 2026-09-25: "prompts with varying levels of detail again, all from the first half ... like the old persona_v1
    # did"). ``levels``: output level name -> (JSON key, word cap). fh_topic = v2's sentence (same rules, same cap);
    # fh_general = persona_v1's general (a few words); fh_detailed = persona_v1's detailed (points, opinions and
    # second-person facts), capped. v2's request settings. Cap + verbatim 5-gram per level; any level failing retries
    # the whole call (so a (row, side) has all levels or none).
    "content_firsthalf_v3": {
        "instruction": None,  # CONTENT_LEVELS_INSTRUCTION, set below (defined after this dict for readability)
        "source": "firsthalf",
        "levels": {"fh_general": ["general", 8], "fh_topic": ["topic", 15], "fh_detailed": ["detailed", 60]},
        "reasoning_effort": "low",
        "max_tokens": 1536,
        "concurrency": 1024,
        # 3 checked texts per reply (the 60-word one hits a verbatim 5-gram more often): 8 attempts instead of 5
        # (profile, 500 calls: 5 attempts left 1.2% failed calls, over the job's fatal 1%).
        "max_attempts": 8,
    },
}

#: content_firsthalf_v3's instruction: persona_v1's three-level shape with the content spec's rules (v2's sentence is
#: the topic level verbatim in its rules and examples).
CONTENT_LEVELS_INSTRUCTION = """You write system prompts for a speech-to-speech conversational AI. The AI will be trained to play ONE participant ("the assistant") in real conversations. Below is a transcript of part of a real two-person conversation. Lines starting with "Assistant:" are THE ASSISTANT; lines starting with "Other:" are the other person.

Write three prompts that tell the assistant what to talk about, at three levels of detail. Each prompt is appended to the fixed opening sentence "You enjoy having a good conversation." -- do NOT repeat that sentence.

- general: a few words naming only the broad theme, at most {max_words_general} words. Example: "Talk about food and cooking."
- topic: ONE short sentence, at most {max_words_topic} words: the broad subject of this conversation from the assistant's side, the way a person would type it when asking for a conversation. Name only one or two general themes that fit the whole transcript; leave out the individual points, claims, opinions, stories, examples, and every name (people, brands, products, platforms, places) and number. Examples of the style: "Have a casual discussion about eating at home versus dining out." / "Have an empathetic discussion about the meaning of family amid uncertainty."
- detailed: TWO to FOUR sentences, at most {max_words_detailed} words in total: the tone and the topics, the specific points or opinions the assistant brings up, and second-person facts about the assistant that the conversation reveals (background, work, experiences, likes and dislikes). Example: "Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings."

Rules for all three:
- Describe only what the assistant talks about; address the assistant as "you".
- Describe WHAT is talked about, never HOW it is worded. Do not quote. Paraphrase in your own words: never copy four or more consecutive words from the transcript.
- Do not mention a podcast, a show, a host, a guest, an interview, a transcript, an excerpt, or these instructions.
- Never name the assistant. Only the detailed prompt may name another real person, and only one the assistant talks about.
- State only what the transcript supports.

Answer with JSON: {{"general": "...", "topic": "...", "detailed": "..."}}

TRANSCRIPT:
{transcript}"""
CONTENT_SPECS["content_firsthalf_v3"]["instruction"] = CONTENT_LEVELS_INSTRUCTION

_CONTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {"prompt": {"type": "string", "minLength": 1}},
    "required": ["prompt"],
    "additionalProperties": False,
}


def content_levels(spec: dict) -> dict:
    """``{output level: (JSON key, word cap)}`` of a content spec: a multi-level spec's ``levels``, else ONE level
    named after the source (``firsthalf`` / ``chunk``) from the ``prompt`` key at ``max_words`` (v1 / v2)."""
    if spec.get("levels"):
        return {lv: (str(k), int(cap)) for lv, (k, cap) in spec["levels"].items()}
    return {spec["source"]: ("prompt", int(spec["max_words"]))}


def content_json_schema(spec: dict) -> dict:
    """The guided-JSON schema of a spec's reply (v1 / v2: ``_CONTENT_JSON_SCHEMA`` exactly)."""
    if not spec.get("levels"):
        return _CONTENT_JSON_SCHEMA
    keys = [k for k, _ in content_levels(spec).values()]
    return {
        "type": "object",
        "properties": {k: {"type": "string", "minLength": 1} for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def content_message(spec: dict, transcript: str) -> str:
    """The user message of one call (v1 / v2: ``instruction.format(max_words=, transcript=)`` exactly)."""
    if not spec.get("levels"):
        return spec["instruction"].format(max_words=spec["max_words"], transcript=transcript)
    caps = {f"max_words_{k}": cap for k, cap in content_levels(spec).values()}
    return spec["instruction"].format(transcript=transcript, **caps)


def content_parse_reply(spec: dict, raw: str, transcript: str) -> tuple:
    """``(bodies, error)``: ``{level: body}`` (opener stripped) when EVERY level passes ``content_reply_error`` at its
    cap, else ``({}, reason)``. Raises on an unparsable reply (the caller retries)."""
    d = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    bodies = {}
    for lv, (key, cap) in content_levels(spec).items():
        body = strip_opener(d[key])
        err = content_reply_error(body, transcript, cap)
        if err:
            return {}, f"{lv}: {err}"
        bodies[lv] = body
    return bodies, ""


def content_level_errors(spec: dict, raw: str, transcript: str) -> dict:
    """``{level: (body, error)}`` for EVERY level of a reply (``content_parse_reply``'s checks, none short-circuited;
    error "" = accepted). Raises on an unparsable reply."""
    d = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    out = {}
    for lv, (key, cap) in content_levels(spec).items():
        body = strip_opener(d[key])
        out[lv] = (body, content_reply_error(body, transcript, cap))
    return out


def content_call(
    client,
    spec: dict,
    llm_name: str,
    temp: float,
    max_tokens: int,
    effort_kw: dict,
    key: str,
    transcript: str,
    attempts: int | None = None,
    fallback_levels: tuple = (),
):
    """ONE content call as ``PersonaContentPromptGen.run`` makes it (seeded per call key, 5 attempts; an API error,
    an unparsable reply, or any level over its cap / with a verbatim 5-gram is retried). Returns ``{"bodies": {level:
    body} | None, "overlap": {level: max shared n-gram} | None, "error", "attempts", "completion_tokens", "partial"}``
    (``analysis/ppx_promptgen_profile.py`` calls this very function).

    ``fallback_levels`` (default none = all levels or nothing, as every job before 2026-09-25 ran): when EVERY attempt
    fails, the FIRST attempt in which all ``fallback_levels`` passed is kept with only the levels that passed in it
    (``partial`` = the full check's error then; "" otherwise). Deterministic (same seeds), and a call that succeeds
    in full within ``attempts`` returns exactly what it would without the fallback."""
    if not transcript.strip():
        return {
            "bodies": None,
            "overlap": None,
            "error": "empty transcript",
            "attempts": 0,
            "completion_tokens": 0,
            "partial": "",
        }
    msg = content_message(spec, transcript)
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)
    err, toks = "", 0
    fallback = None  # (bodies, attempt, error) of the first attempt whose fallback levels all passed
    attempts = int(attempts or spec.get("max_attempts", 5))  # v1 / v2: 5; a job's max_attempts overrides
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=llm_name,
                messages=[{"role": "user", "content": msg}],
                seed=(seed + attempt) % (2**31),
                temperature=temp,
                max_tokens=max_tokens,  # GPT-OSS reasons before answering
                extra_body={"guided_json": content_json_schema(spec)},
                **effort_kw,
            )
            usage = getattr(resp, "usage", None)
            toks += int(getattr(usage, "completion_tokens", 0) or 0)
            raw = (resp.choices[0].message.content or "").strip()
            bodies, err = content_parse_reply(spec, raw, transcript)
            if err and fallback_levels and fallback is None:
                per = content_level_errors(spec, raw, transcript)
                if all(not per[lv][1] for lv in fallback_levels):
                    fallback = ({lv: b for lv, (b, e) in per.items() if not e}, attempt + 1, err)
        except Exception as e:  # API error, empty or unparsable reply: retry
            err = f"{type(e).__name__}: {e}"[:200]
            continue
        if err:
            continue
        return {
            "bodies": bodies,
            "overlap": {lv: max_shared_ngram(b, transcript) for lv, b in bodies.items()},
            "error": "",
            "attempts": attempt + 1,
            "completion_tokens": toks,
            "partial": "",
        }
    if fallback is not None:
        bodies, n_att, why = fallback
        return {
            "bodies": bodies,
            "overlap": {lv: max_shared_ngram(b, transcript) for lv, b in bodies.items()},
            "error": "",
            "attempts": n_att,
            "completion_tokens": toks,
            "partial": why,
        }
    return {
        "bodies": None,
        "overlap": None,
        "error": err,
        "attempts": attempts,
        "completion_tokens": toks,
        "partial": "",
    }


def content_spec_digest(name: str) -> str:
    """Content hash of a content spec (hashed into ``PersonaContentPromptGen``)."""
    return hashlib.sha256(json.dumps(CONTENT_SPECS[name], sort_keys=True).encode()).hexdigest()[:16]


def strip_opener(text: str) -> str:
    """The prompt body: whitespace-normalised, any leading copies of ``PERSONA_MINIMAL`` removed."""
    body = " ".join(str(text).split())
    while body.startswith(PERSONA_MINIMAL):
        body = body[len(PERSONA_MINIMAL) :].strip()
    return body


def content_reply_error(body: str, transcript: str, max_words: int) -> str:
    """Why an LLM reply (opener already stripped) is rejected, or "" when it is accepted: empty, over the word cap,
    or a verbatim run of ``MAX_SHARED_NGRAM`` words shared with the transcript it was written from."""
    n = len(body.split())
    if n == 0:
        return "empty"
    if n > max_words:
        return f"{n} words > {max_words}"
    ov = max_shared_ngram(body, transcript)
    if ov >= MAX_SHARED_NGRAM:
        return f"verbatim overlap {ov}"
    return ""


def firsthalf_transcript(focus: list, other: list, duration: float) -> str:
    """Labelled dialogue of the words whose ONSET lies in the first half of a window (``start < duration / 2``,
    window-relative alignments as the training rows store them): ``focus`` is "Assistant", ``other`` is "Other"."""
    half = float(duration) / 2.0
    fa = [w for w in focus or [] if float(w["start"]) < half]
    fo = [w for w in other or [] if float(w["start"]) < half]
    return _dialogue_text(fa, fo)


def chunk_block(abs_mid: float, chunk_sec: float) -> int:
    """Index of the fixed ``chunk_sec`` block of episode time holding a window's midpoint."""
    return int(float(abs_mid) // float(chunk_sec))


def chunk_transcripts(windows: list, slice_rows: list, chunk_sec: float) -> tuple:
    """Per (episode, block, speaker): the labelled dialogue of that block of the episode, the speaker as "Assistant".

    ``windows``: dicts ``id, episode_id, abs_t0, abs_t1, labels`` (``labels`` = the diarized speakers that need a
    prompt for this window: its assistant, and its other side when swappable). ``slice_rows``: the episode's dialogue
    rows (``episode_id, span_start_sec, words_a, words_b, turns_json``; words/turns row-relative). Each row's two
    channels are labelled together (``_joint_labels`` over the whole row); a channel's words go to "Assistant" when its
    label is the key's speaker, else to "Other". The block span is ``[b * chunk_sec, (b + 1) * chunk_sec)`` widened
    to cover every window assigned to the block, so each window lies inside the text its prompt is written from.

    Returns ``(texts, key_of)``: ``{(episode, block, label): (transcript, span_start, span_end)}`` and
    ``{(window id, label): key}``."""
    spans, key_of = {}, {}
    for w in windows:
        b = chunk_block((w["abs_t0"] + w["abs_t1"]) / 2.0, chunk_sec)
        s0, s1 = spans.get((w["episode_id"], b), (b * chunk_sec, (b + 1) * chunk_sec))
        spans[(w["episode_id"], b)] = (min(s0, w["abs_t0"]), max(s1, w["abs_t1"]))
        for lab in w["labels"]:
            key_of[(w["id"], lab)] = (w["episode_id"], b, lab)
    by_ep = {}
    for r in slice_rows:
        by_ep.setdefault(r["episode_id"], []).append(r)
    labelled = {}  # episode -> [(label, abs word)] over all its rows

    def ep_words(ep):
        if ep not in labelled:
            out, seen = [], set()
            for r in by_ep.get(ep, []):
                s = float(r["span_start_sec"])
                wa, wb = json.loads(r["words_a"] or "[]"), json.loads(r["words_b"] or "[]")
                la, lb = _joint_labels(wa, wb, json.loads(r["turns_json"] or "[]"))
                for lab, ws in ((la, wa), (lb, wb)):
                    for x in ws:
                        a = round(s + float(x["start"]), 3)
                        if (lab, a, x["text"]) in seen:  # rows never overlap; this only guards a duplicate row
                            continue
                        seen.add((lab, a, x["text"]))
                        out.append((lab, {"text": x["text"], "start": a, "end": s + float(x["end"])}))
            labelled[ep] = out
        return labelled[ep]

    texts = {}
    for key in sorted(set(key_of.values()), key=lambda k: (k[0], k[1], str(k[2]))):
        ep, b, lab = key
        s0, s1 = spans[(ep, b)]
        inside = [(lb, w) for lb, w in ep_words(ep) if s0 <= w["start"] < s1]
        focus = [w for lb, w in inside if lb is not None and lb == lab]
        other = [w for lb, w in inside if lb is None or lb != lab]
        texts[key] = (_dialogue_text(focus, other), s0, s1)
    return texts, key_of


class PersonaContentPromptGen(Job):
    """Short CONTENT prompts (one sentence, <= ``max_words``) for training windows, by an LLM served with vLLM
    (ablations ``ab_epad1_chunkprompt`` / ``ab_epad1_firsthalf``, 2026-09-25). What the LLM is shown is the spec's
    ``source`` (``CONTENT_SPECS``): a ~10-min block of the episode around the window (``chunk``; one call per
    (episode, block, speaker)), or the window's first half (``firsthalf``; one call per (window, side)). Both see
    both speakers, the prompted speaker labelled "Assistant". The spec (instruction + source + caps) is hashed.

    ``data``: an ``AttachPersonaPrompts(per_read=True, with_other_side=True)`` corpus -- the rows to prompt, their
    alignments (firsthalf) and voice labels (chunk). ``context_data`` + ``dialogue_shard_dirs`` (chunk only): the
    corpus's ``PodcastWindowContext`` (window -> row + bounds) and its dialogue slices (the block's words).
    ``sides``: ``assistant`` = the row's assistant channel; ``other`` = its other channel (swappable rows only).
    ``ids``: prompt only these rows (self-play inputs, probe windows); None = every row. ``shard``/``num_shards``:
    a strided slice of the calls.

    Output (long format): ``id, side, level, text`` (``PERSONA_MINIMAL`` + body), ``body, n_words, key,
    max_ngram_overlap, spec, llm_name``; ``level`` = the spec's source name. Seeded per call key, retries, > 1% failed
    calls is fatal; ``summary.json`` has the word-count distribution and examples."""

    def __init__(
        self,
        *,
        data: tk.Path,
        spec: str,
        context_data: tk.Path | None = None,
        dialogue_shard_dirs: list | None = None,
        sides: tuple = ("assistant", "other"),
        ids: list | None = None,
        shard: int | None = None,
        num_shards: int | None = None,
        llm_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.6,
        max_attempts: int | None = None,
        fallback_levels: tuple | None = None,
    ):
        assert spec in CONTENT_SPECS, spec
        sides = tuple(sides)
        assert sides and set(sides) <= {"assistant", "other"} and len(set(sides)) == len(sides), sides
        if CONTENT_SPECS[spec]["source"] == "chunk":
            assert context_data is not None and dialogue_shard_dirs, "a chunk spec needs context_data + slices"
        self.data = data
        self.spec = spec
        self.context_data = context_data
        self.dialogue_shard_dirs = list(dialogue_shard_dirs or [])
        self.sides = sides
        self.ids = sorted(set(ids)) if ids else None
        self.shard = shard
        self.num_shards = num_shards
        self.llm_name = llm_name
        self.temperature = float(temperature)
        # Attempts per call, overriding the spec's (``None`` = the spec's; hash-neutral then). For rows that MUST all
        # get prompts (the 64 pinned self-play rows: the attach raises on a missing one; 2026-09-25 one of 128 v3
        # calls kept a verbatim 6-gram in fh_detailed through all 8 attempts).
        self.max_attempts = max_attempts
        # Levels a call may fall back to when every attempt fails the full check (``content_call``): the kept attempt
        # stores only its passing levels. ``None`` = all or nothing (hash-neutral). For the pinned self-play rows, which
        # read one level (2026-09-25: rss_1341116090418274#46@a#w0's other side kept "the Laugh Factory in Long Beach",
        # a verbatim 6-gram, in fh_detailed through all 24 attempts, so the row had no fh_topic either).
        if fallback_levels is not None:
            fallback_levels = tuple(fallback_levels)
            assert fallback_levels and set(fallback_levels) <= set(content_levels(CONTENT_SPECS[spec])), fallback_levels
        self.fallback_levels = fallback_levels
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 48, "time": 4, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name, 80)}

    @classmethod
    def hash(cls, parsed_args):
        """The spec's CONTENT is part of the hash (Sisyphus hashes only ``__init__`` arguments): an edited instruction,
        source or cap is a different job. ``max_attempts=None`` / ``fallback_levels=None`` are left out (every job
        declared before they existed)."""
        d = {**parsed_args, "spec_digest": content_spec_digest(parsed_args["spec"])}
        for k in ("max_attempts", "fallback_levels"):
            if d.get(k) is None:
                d.pop(k, None)
        return super(PersonaContentPromptGen, cls).hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @property
    def level(self) -> str:
        """The source name (single-level specs: also the output ``level``; a multi-level spec: see ``levels``)."""
        return CONTENT_SPECS[self.spec]["source"]

    @property
    def levels(self) -> tuple:
        """The output ``level`` values this job writes (v1 / v2: ``(source,)``; v3: its ``levels`` names)."""
        return tuple(content_levels(CONTENT_SPECS[self.spec]))

    @property
    def prompt_set(self) -> str:
        return f"{self.spec}|{self.llm_name}"

    def completed_fraction(self):
        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress_0.json")))
            return min(d["done"] / d["total"], 1.0) if d.get("total") else None
        except (OSError, ValueError, KeyError):
            return None

    def _calls(self) -> tuple:
        """``(calls, n_rows)``: ``calls`` = ``[(call key, transcript, [(row id, side), ...])]`` in a fixed order."""
        from datasets import load_from_disk

        spec = CONTENT_SPECS[self.spec]
        cols = ["id", "duration", "alignments", "alignments_other", "voice_label", "voice_label_other", "other_ok"]
        ds = load_from_disk(self.data.get_path()).select_columns(cols)
        if self.ids:
            want = set(self.ids)
            idx = [i for i, x in enumerate(ds["id"]) if x in want]
            if len(idx) != len(want):
                raise RuntimeError(f"{len(want) - len(idx)} of {len(want)} requested ids are not in the data")
            ds = ds.select(idx)
        rows = list(ds)

        def row_sides(r):
            return [s for s in self.sides if s == "assistant" or r["other_ok"]]

        if spec["source"] == "firsthalf":
            calls = []
            for r in rows:
                for s in row_sides(r):
                    focus, other = (
                        (r["alignments"], r["alignments_other"])
                        if s == "assistant"
                        else (r["alignments_other"], r["alignments"])
                    )
                    calls.append((f"{r['id']}|{s}", firsthalf_transcript(focus, other, r["duration"]), [(r["id"], s)]))
            return calls, len(rows)

        ctx = load_from_disk(self.context_data.get_path()).select_columns(["id", "item_id", "episode_id", "t0", "t1"])
        want = {r["id"] for r in rows}
        cinfo = {c["id"]: c for c in ctx if c["id"] in want}
        missing = want - set(cinfo)
        if missing:
            raise RuntimeError(f"{len(missing)} rows have no window context (e.g. {sorted(missing)[:3]})")
        eps = {c["episode_id"] for c in cinfo.values()}
        slice_rows, span_start = [], {}
        for d in self.dialogue_shard_dirs:
            for p in _parts(d):
                part = load_from_disk(p).select_columns(["item_id", "episode_id", "span_start_sec"])
                keep = [i for i, e in enumerate(part["episode_id"]) if e in eps]
                if not keep:
                    continue
                sub = load_from_disk(p).select_columns(
                    ["item_id", "episode_id", "span_start_sec", "words_a", "words_b", "turns_json"]
                )
                for r in sub.select(keep):
                    slice_rows.append(r)
                    span_start[r["item_id"]] = float(r["span_start_sec"])
        windows = []
        for r in rows:
            c = cinfo[r["id"]]
            s = span_start[c["item_id"]]  # KeyError = the slices do not match the context: a wiring bug
            labels = [r["voice_label"] if sd == "assistant" else r["voice_label_other"] for sd in row_sides(r)]
            windows.append(
                {
                    "id": r["id"],
                    "episode_id": c["episode_id"],
                    "abs_t0": s + float(c["t0"]),
                    "abs_t1": s + float(c["t1"]),
                    "labels": labels,
                    "sides": row_sides(r),
                }
            )
        texts, key_of = chunk_transcripts(windows, slice_rows, spec["chunk_sec"])
        members = {}
        for w in windows:
            for lab, sd in zip(w["labels"], w["sides"]):
                members.setdefault(key_of[(w["id"], lab)], []).append((w["id"], sd))
        calls = [
            (f"{k[0]}|{k[1]}|{k[2]}", texts[k][0], members[k])
            for k in sorted(members, key=lambda k: (k[0], k[1], str(k[2])))
        ]
        return calls, len(rows)

    def run(self):
        import numpy as np
        from datasets import Dataset
        from openai import OpenAI

        spec, llm_name, temp = CONTENT_SPECS[self.spec], self.llm_name, self.temperature
        # Request knobs (2026-09-25, p6's throughput work): absent from the v1 specs = the settings they ran with.
        max_tokens = int(spec.get("max_tokens", 4096))
        concurrency = int(spec.get("concurrency", PROMPT_GEN_CONCURRENCY))
        effort_kw = {"reasoning_effort": spec["reasoning_effort"]} if spec.get("reasoning_effort") else {}
        calls, n_rows = self._calls()
        if self.shard is not None and self.num_shards is not None:
            calls = calls[self.shard :: self.num_shards]
        tlens = [len(c[1].split()) for c in calls]
        print(
            f"[content] {self.spec} ({spec['source']}): {n_rows} rows -> {len(calls)} calls "
            f"({sum(len(c[2]) for c in calls)} (row, side) prompts); transcript words median "
            f"{np.median(tlens) if tlens else 0:.0f}, max {max(tlens) if tlens else 0}",
            flush=True,
        )

        with vllm_server(llm_name) as url:

            def gen(client, call):
                return content_call(
                    client,
                    spec,
                    llm_name,
                    temp,
                    max_tokens,
                    effort_kw,
                    call[0],
                    call[1],
                    getattr(self, "max_attempts", None),
                    getattr(self, "fallback_levels", None) or (),
                )

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url)
            outs = map_concurrent(
                lambda c: gen(client, c),
                calls,
                concurrency=concurrency,
                progress_path=os.path.join(os.getcwd(), "progress_0.json"),
            )

        failed = [(c[0], o["error"]) for c, o in zip(calls, outs) if o["bodies"] is None]
        print(f"[content] {len(failed)}/{len(calls)} calls failed", flush=True)
        for k, e in failed[:5]:
            print(f"  FAILED {k}: {e}", flush=True)
        partial = [(c[0], sorted(o["bodies"]), o["partial"]) for c, o in zip(calls, outs) if o.get("partial")]
        for k, lvs, e in partial:
            print(f"  PARTIAL {k}: kept {lvs} (full check: {e})", flush=True)
        if len(failed) > 0.01 * max(1, len(calls)):
            raise RuntimeError(f"content prompt generation failed on {len(failed)}/{len(calls)} calls (> 1%)")
        rows = []
        for (key, _, members), o in zip(calls, outs):
            if o["bodies"] is None:
                continue
            for rid, side in members:
                for lv, body in o["bodies"].items():
                    rows.append(
                        {
                            "id": rid,
                            "side": side,
                            "level": lv,
                            "text": f"{PERSONA_MINIMAL} {body}",
                            "body": body,
                            "n_words": len(body.split()),
                            "key": key,
                            "max_ngram_overlap": int(o["overlap"][lv]),
                            "spec": self.spec,
                            "llm_name": llm_name,
                        }
                    )
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())

        def words_stats(lens):
            return {
                "words": {
                    f"p{q}": float(np.percentile(lens, q)) if lens else None for q in (0, 10, 25, 50, 75, 90, 100)
                },
                "words_hist": {str(n): lens.count(n) for n in sorted(set(lens))},
            }

        by_level = {
            lv: [len(o["bodies"][lv].split()) for o in outs if o["bodies"] is not None and lv in o["bodies"]]
            for lv in self.levels
        }
        summary = {
            "spec": self.spec,
            "spec_digest": content_spec_digest(self.spec),
            "source": spec["source"],
            "rows": n_rows,
            "calls": len(calls),
            "failed": len(failed),
            **({"partial": [list(p) for p in partial]} if getattr(self, "fallback_levels", None) else {}),
            "prompts": len(rows),
            # single-level specs: the top-level word stats as before; multi-level: per level
            **(words_stats(by_level[self.level]) if len(self.levels) == 1 else {}),
            **({"levels": {lv: words_stats(v) for lv, v in by_level.items()}} if len(self.levels) > 1 else {}),
            "transcript_words_median": float(np.median(tlens)) if tlens else None,
            "examples": [r["text"] for r in rows[:: max(1, len(rows) // 8)][:8]],
        }
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[content] {summary}", flush=True)


def content_prompt_lists(row: dict, prompts: dict, levels: tuple) -> tuple:
    """``(context, context_other)`` of one attached row with every non-minimal level replaced by that side's content
    prompt OF THAT LEVEL: ``PERSONA_MINIMAL`` for ``minimal``, ``prompts[(id, side, level)]`` otherwise (the
    generator's ``level`` column: ``firsthalf`` / ``chunk`` for the single-level specs, ``fh_general`` / ``fh_topic``
    / ``fh_detailed`` for v3); ``None`` for a side missing ANY of its levels (the other side only when ``other_ok``)."""

    def side_list(side):
        out = []
        for lv in levels:
            t = PERSONA_MINIMAL if lv == "minimal" else prompts.get((row["id"], side, lv))
            if t is None:
                return None
            out.append(t)
        return out

    ctx = side_list("assistant")
    oth = side_list("other") if row.get("other_ok") else []
    return ctx, oth


class AttachContentPrompts(Job):
    """An ``AttachPersonaPrompts(per_read=True, with_other_side=True)`` dataset with ONLY its prompt lists replaced
    by ``PersonaContentPromptGen`` prompts (``prompts``: one or more of its outputs), per (row id, side); the exact
    counterpart of ``AttachPersonaProfiles`` (same list positions, so the loader's per-read draw of minimal vs the LLM
    level is unchanged: ``levels=("chunk", "minimal", "chunk")`` in place of p3's ``general, minimal, topic``).

    ``keep_all_rows=False`` (training): a row without its assistant prompt is dropped, an ``other_ok`` row without
    the other side's prompt becomes unswappable; counted, > ``max_drop_frac`` is an error.
    ``keep_all_rows=True`` (self-play inputs; self-play picks rows by index among ``other_ok``, so every row and
    flag stays): every id in ``require_ids`` must have its prompt(s), else an error; any other row keeps
    ``PERSONA_MINIMAL`` in the LLM slots (never picked; counted)."""

    def __init__(
        self,
        *,
        data: tk.Path,
        prompts: list,
        prompt_set: str,
        levels: tuple,
        keep_all_rows: bool = False,
        require_ids: list | None = None,
        max_drop_frac: float = 0.05,
    ):
        assert levels and "minimal" in levels and any(lv != "minimal" for lv in levels), levels
        assert not require_ids or keep_all_rows, "require_ids is the keep_all_rows (self-play) contract"
        self.data = data
        self.prompts = list(prompts)
        self.prompt_set = prompt_set
        self.levels = tuple(levels)
        self.keep_all_rows = bool(keep_all_rows)
        self.require_ids = sorted(set(require_ids)) if require_ids else None
        self.max_drop_frac = float(max_drop_frac)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 24, "time": 2})

    def run(self):
        from datasets import Dataset, load_from_disk

        prompts = {}
        for p in self.prompts:
            for r in load_from_disk(p.get_path()).select_columns(["id", "side", "level", "text"]):
                prompts[(r["id"], r["side"], r["level"])] = r["text"]
        have = {k[2] for k in prompts}
        want = {lv for lv in self.levels if lv != "minimal"}
        if prompts and not want <= have:
            raise RuntimeError(f"levels {sorted(want - have)} are in no prompt table (tables have {sorted(have)})")
        src = load_from_disk(self.data.get_path())
        assert "context_other" in src.column_names, "needs an AttachPersonaPrompts(with_other_side=True) dataset"
        levels = list(self.levels)
        required = set(self.require_ids or [])
        n = {"in": 0, "out": 0, "noprompt": 0, "other_noprompt": 0, "kept_minimal": 0, "required": 0}

        def rows():
            for r in src:
                n["in"] += 1
                ctx, oth = content_prompt_lists(r, prompts, self.levels)
                if r["id"] in required:
                    if ctx is None or oth is None:
                        raise RuntimeError(f"{r['id']}: a required (self-play) row without its content prompt(s)")
                    n["required"] += 1
                elif self.keep_all_rows and (ctx is None or oth is None):
                    n["kept_minimal"] += 1
                    ctx = [PERSONA_MINIMAL for _ in levels]
                    oth = [PERSONA_MINIMAL for _ in levels] if r["other_ok"] else []
                if oth is None:
                    n["other_noprompt"] += 1
                    r = {**r, "other_ok": False, "voice_codes_other": [], "voice_n_frames_other": 0}
                    r["alignments_other"] = []
                    oth = []
                if ctx is None:
                    n["noprompt"] += 1
                    continue
                n["out"] += 1
                yield {
                    **r,
                    "context": ctx,
                    "context_level": levels,
                    "context_set": self.prompt_set,
                    "context_other": oth,
                }

        out = Dataset.from_generator(rows, features=src.features, cache_dir=os.path.abspath("hf_gen_cache"))
        print(f"[attach-content] {n}; levels {levels}", flush=True)
        if n["required"] != len(required):
            raise RuntimeError(f"{len(required) - n['required']} required ids are not in the data")
        if not self.keep_all_rows:
            for k in ("noprompt", "other_noprompt"):
                if n[k] > self.max_drop_frac * n["in"]:
                    raise RuntimeError(f"{n[k]}/{n['in']} rows hit {k} (> {self.max_drop_frac:.0%}): {n}")
        out.save_to_disk(self.out_dir.get_path())
        json.dump(
            {**n, "levels": levels, "prompt_set": self.prompt_set}, open(self.out_summary.get_path(), "w"), indent=2
        )


# ---------------------------------------------------------------------------------------------
# 4. episode-level train / held-out split (p6, 2026-09-25)
# ---------------------------------------------------------------------------------------------
def episode_of(row_id: str) -> str:
    """The episode of a training row id (``<episode>#<row>@<channel>#w<k>``)."""
    return str(row_id).split("#", 1)[0]


def split_train_episodes(episodes, *, fraction: float, seed: int, heldout=()) -> set:
    """The TRAIN episodes: every episode not in ``heldout`` whose seeded unit value is ``< fraction``.

    Deterministic per episode (``_seeded_unit(seed, "split|" + episode)``, never ``random``), so the split of an
    episode does not depend on which other episodes the corpus has, and a larger ``fraction`` gives a SUPERSET of a
    smaller one's train episodes (half is nested in all). ``heldout``: episodes forced to the held-out side whatever
    their draw (the evaluation episodes)."""
    assert 0.0 < float(fraction) <= 1.0, fraction
    forced = set(heldout)
    return {e for e in set(episodes) if e not in forced and _seeded_unit(seed, f"split|{e}") < float(fraction)}


def window_spans(context_path: str, slice_dirs: list, ids=None) -> dict:
    """``{window id: (episode, abs t0, abs t1)}`` in EPISODE time: a ``PodcastWindowContext`` row's ``t0``/``t1`` are
    relative to its dialogue row, whose ``span_start_sec`` is in the corpus's dialogue slices. ``ids``: only these."""
    from datasets import load_from_disk

    ctx = load_from_disk(context_path).select_columns(["id", "item_id", "episode_id", "t0", "t1"])
    rows = [r for r in ctx if ids is None or r["id"] in ids]
    items = {r["item_id"] for r in rows}
    start = {}
    for d in slice_dirs:
        for p in _parts(d):
            part = load_from_disk(p).select_columns(["item_id", "span_start_sec"])
            for it, s in zip(part["item_id"], part["span_start_sec"]):
                if it in items:
                    start[it] = float(s)
    missing = items - set(start)
    if missing:
        raise RuntimeError(
            f"{len(missing)} dialogue rows of {context_path} not in its slices (e.g. {sorted(missing)[:3]})"
        )
    return {
        r["id"]: (r["episode_id"], start[r["item_id"]] + float(r["t0"]), start[r["item_id"]] + float(r["t1"]))
        for r in rows
    }


def overlapping_ids(spans: dict, exclude: list) -> set:
    """Ids of ``spans`` (``{id: (episode, t0, t1)}``) whose interval shares any time (> 0 s) with an ``exclude``
    interval ``(episode, t0, t1)`` of the same episode."""
    by_ep = {}
    for ep, a, b in exclude:
        by_ep.setdefault(ep, []).append((a, b))
    return {i for i, (ep, a, b) in spans.items() if any(a < y and x < b for x, y in by_ep.get(ep, ()))}


class EpisodeSplit(Job):
    """The TRAINING rows of an attached corpus after an EPISODE split and a held-out-window exclusion (p6, 2026-09-25).

    ``data``: any corpus whose rows carry ``id`` (and ``duration``). An episode trains when it is not in
    ``heldout_episodes`` and its seeded draw is below ``train_fraction`` (``split_train_episodes``; 1.0 = every
    episode but the forced ones). So no window of a held-out episode is ever trained on -- a window-level split
    would put the same speakers and conversation on both sides.

    ``exclude_windows`` (+ ``exclude_context`` / ``exclude_slice_dirs``, the corpus those windows come from, and
    ``window_context`` / ``slice_dirs``, this corpus's): a fixed evaluation set of windows from ANOTHER slicing of the
    same episodes (``PersonaHeldoutSet``). Every training row whose episode-time span overlaps one of them is dropped
    too, so that set's audio stays unseen (``window_spans`` + ``overlapping_ids``).

    Outputs: ``out_train`` (the kept rows, row order unchanged), ``out_episodes`` (json: train / held-out / forced
    episodes, dropped overlap ids) and ``out_summary`` (episodes, rows and hours per side, the overlap drop)."""

    def __init__(
        self,
        *,
        data: tk.Path,
        train_fraction: float,
        seed: int = 0,
        heldout_episodes: list | None = None,
        window_context: tk.Path | None = None,
        slice_dirs: list | None = None,
        exclude_windows: tk.Path | None = None,
        exclude_context: tk.Path | None = None,
        exclude_slice_dirs: list | None = None,
    ):
        assert 0.0 < float(train_fraction) <= 1.0, train_fraction
        ex = (window_context, slice_dirs, exclude_windows, exclude_context, exclude_slice_dirs)
        assert all(x is None for x in ex) or all(x is not None for x in ex), "the window exclusion needs all five"
        self.data = data
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)
        self.heldout_episodes = sorted(set(heldout_episodes or []))
        self.window_context = window_context
        self.slice_dirs = list(slice_dirs) if slice_dirs is not None else None
        self.exclude_windows = exclude_windows
        self.exclude_context = exclude_context
        self.exclude_slice_dirs = list(exclude_slice_dirs) if exclude_slice_dirs is not None else None
        self.out_train = self.output_path("train", directory=True)
        self.out_episodes = self.output_path("episodes.json")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 48, "time": 3})

    def run(self):
        from datasets import load_from_disk

        ds = load_from_disk(self.data.get_path())
        meta = ds.select_columns(["id", "duration"])
        ids, dur = meta["id"], meta["duration"]
        episodes = sorted({episode_of(r) for r in ids})
        forced = set(self.heldout_episodes)
        train_eps = split_train_episodes(episodes, fraction=self.train_fraction, seed=self.seed, heldout=forced)
        heldout_eps = set(episodes) - train_eps
        overlap, n_exclude = set(), 0
        if self.exclude_windows is not None:
            ex_ids = set(load_from_disk(self.exclude_windows.get_path()).select_columns(["id"])["id"])
            n_exclude = len(ex_ids)
            ex = window_spans(self.exclude_context.get_path(), self.exclude_slice_dirs, ids=ex_ids)
            if len(ex) != len(ex_ids):
                raise RuntimeError(f"{len(ex_ids) - len(ex)} excluded windows have no context row")
            ex_eps = {e for e, _, _ in ex.values()}
            cand = {r for r in ids if episode_of(r) in train_eps and episode_of(r) in ex_eps}
            overlap = overlapping_ids(
                window_spans(self.window_context.get_path(), self.slice_dirs, ids=cand), list(ex.values())
            )
        keep = [i for i, r in enumerate(ids) if episode_of(r) in train_eps and r not in overlap]
        print(
            f"[split] {len(ds)} rows / {len(episodes)} episodes -> train {len(keep)} rows / {len(train_eps)} episodes; "
            f"held out {len(heldout_eps)} episodes (forced {len(forced & set(episodes))} of {len(forced)}); "
            f"{len(overlap)} train rows dropped for overlapping {n_exclude} evaluation windows",
            flush=True,
        )
        ds.select(keep).save_to_disk(self.out_train.get_path())

        def side(idx):
            return {"rows": len(idx), "hours": round(sum(float(dur[i]) for i in idx) / 3600.0, 2)}

        summary = {
            "train_fraction": self.train_fraction,
            "seed": self.seed,
            "episodes": {
                "train": len(train_eps),
                "heldout": len(heldout_eps),
                "forced_in_corpus": len(forced & set(episodes)),
            },
            "train": side(keep),
            "heldout_episodes_rows": side([i for i, r in enumerate(ids) if episode_of(r) in heldout_eps]),
            "overlap_dropped": side([i for i, r in enumerate(ids) if r in overlap]),
            "exclude_windows": n_exclude,
        }
        json.dump(
            {
                "train": sorted(train_eps),
                "heldout": sorted(heldout_eps),
                "forced_heldout": sorted(forced),
                "overlap_dropped_ids": sorted(overlap),
            },
            open(self.out_episodes.get_path(), "w"),
            indent=1,
        )
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[split] {summary}", flush=True)


# ---------------------------------------------------------------------------------------------
# Episode-edge filter (user, 2026-09-28: "Hearing a lot of podcast wrap ups. Maybe we should exclude the first and
# last 2% of every episode? Add that as a filter for future runs")
# ---------------------------------------------------------------------------------------------
#: Intro / wrap-up marker phrases, for :class:`EpisodeEdgeRows`' SUMMARY only (they never decide a row, so they are not
#: part of the hash): whole-phrase, case-insensitive matches on a window's text (both sides), punctuation stripped.
EDGE_MARKERS = (
    "joe rogan experience",
    "welcome to",
    "we're live",
    "we're up",
    "thanks for having me",
    "thank you for having me",
    "thanks for coming",
    "thank you for coming",
    "thanks for doing this",
    "thank you for doing this",
    "my guest",
    "where can people",
    "tell everybody",
    "follow",
    "instagram",
    "twitter",
    "podcast",
    "sponsor",
    "brought to you by",
    "goodnight",
    "good night",
    "bye everybody",
    "see you soon",
    "see you next time",
)
#: Relative-position bins of the summary: fraction of the episode at the window MIDPOINT.
EDGE_BINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.90, 0.95, 0.97, 0.98, 0.99, 1.0)


def episode_durations(index) -> dict:
    """``{episode: seconds}`` from an EPISODE codes index (``PodcastCodesIndex`` over the ``PodcastEpisodeIngest``
    shards, e.g. ``output/podcast_episodes/jre_full/codes_index``; a directory holding ``index.jsonl`` or the file).
    One row per episode; ``duration_sec`` = the encoded episode's frames / frame rate, i.e. the time base every
    dialogue slice's ``span_start_sec`` is in (``podcast_slice_main``: an episode row's frame 0 is episode t=0)."""
    root = index.get_path() if hasattr(index, "get_path") else str(index)
    path = os.path.join(root, "index.jsonl") if os.path.isdir(root) else root
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["item_id"] != r["episode_id"]:
                raise RuntimeError(
                    f"{path}: not an EPISODE index (row {r['item_id']!r} is a slice of {r['episode_id']})"
                )
            if r["episode_id"] in out:
                raise RuntimeError(f"{path}: episode {r['episode_id']} twice")
            out[r["episode_id"]] = float(r["duration_sec"])
    return out


def episode_edge_side(t0: float, t1: float, duration: float, edge_frac: float):
    """``"head"`` if the window ``[t0, t1]`` (episode seconds) shares any time with the first ``edge_frac`` of an
    episode of ``duration`` seconds, ``"tail"`` if with the last, else None (a window touching both: ``"head"``).
    OVERLAP, not the midpoint: a sign-off is a few seconds anywhere in a 20-60 s window, so a window that only
    reaches into the edge zone still carries it."""
    e = float(edge_frac) * float(duration)
    if t0 < e:
        return "head"
    if t1 > float(duration) - e:
        return "tail"
    return None


def _marker_patterns(markers=EDGE_MARKERS) -> dict:
    import re

    return {m: re.compile(r"(?<![a-z'])" + re.escape(m) + r"(?![a-z'])") for m in markers}


def _norm_text(text: str) -> str:
    import re

    t = str(text or "").lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", t).split())


def edge_rows(ids, spans: dict, durations: dict, edge_frac: float) -> tuple:
    """``(keep, side)`` for the rows ``ids`` (a table's ``id`` column): ``keep`` = the row indices whose window lies
    wholly inside the middle ``1 - 2 * edge_frac`` of its episode, ascending; ``side`` = ``{row index: "head" /
    "tail"}`` for the others (:func:`episode_edge_side`). ``spans``: :func:`window_spans` of these ids; ``durations``:
    :func:`episode_durations`. Raises on a row without a span or an episode without a duration, and on a window that
    ends more than 1 s after its episode (the durations would not be this corpus's time base)."""
    assert 0.0 <= float(edge_frac) < 0.5, edge_frac
    keep, side = [], {}
    for i, rid in enumerate(ids):
        if rid not in spans:
            raise RuntimeError(f"row {i} ({rid}) has no window span (not in the window context)")
        ep, t0, t1 = spans[rid]
        if ep not in durations:
            raise RuntimeError(f"row {i} ({rid}): episode {ep} not in the episode index")
        d = durations[ep]
        if t1 > d + 1.0 or t0 < -1e-3:
            raise RuntimeError(f"row {i} ({rid}): window [{t0:.2f}, {t1:.2f}] s outside its episode (0, {d:.2f}) s")
        s = episode_edge_side(t0, t1, d, edge_frac)
        if s is None:
            keep.append(i)
        else:
            side[i] = s
    return keep, side


class EpisodeEdgeRows(Job):
    """The rows of a window corpus that lie in the MIDDLE of their episode: every row whose window shares any time with
    the first or last ``edge_frac`` of its episode's duration is dropped (user, 2026-09-28, listening to p5's
    self-play: "Hearing a lot of podcast wrap ups. Maybe we should exclude the first and last 2% of every episode?").
    Measured on p4's corpus before it became a default (backlog / rules.md "episode-edge filter").

    ``data``: the training corpus (rows carry ``id`` + ``duration``; any ``AttachPersonaPrompts``-style output).
    ``window_context`` + ``slice_dirs``: the ``PodcastWindowContext`` its ids come from and its dialogue slices
    (:func:`window_spans`: window id -> episode seconds). ``episode_index``: the EPISODE codes index of the same build
    (:func:`episode_durations`). Rows keep their table order; nothing is copied.

    Outputs: ``out_rows`` (``rows.npy``: int64 row indices to train on, ascending -- the launcher's ``train_rows``;
    ``train_data_common.load_row_subset``) and ``out_summary`` (rows / hours kept and dropped per side, episodes, rows
    per relative-position bin, and the share of windows whose text holds an ``EDGE_MARKERS`` phrase per bin -- the
    evidence for the cut, recomputed on every corpus it is applied to)."""

    def __init__(
        self,
        *,
        data: tk.Path,
        window_context: tk.Path,
        slice_dirs: list,
        episode_index: tk.Path,
        edge_frac: float = 0.02,
    ):
        assert 0.0 < float(edge_frac) < 0.5, edge_frac
        self.data = data
        self.window_context = window_context
        self.slice_dirs = list(slice_dirs)
        self.episode_index = episode_index
        self.edge_frac = float(edge_frac)
        self.out_rows = self.output_path("rows.npy")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        import numpy as np

        summary, keep = edge_rows_summary(
            self.data.get_path(),
            self.window_context.get_path(),
            self.slice_dirs,
            self.episode_index,
            self.edge_frac,
        )
        np.save(self.out_rows.get_path(), np.asarray(keep, dtype=np.int64))
        json.dump(summary, open(self.out_summary.get_path(), "w"), indent=2)
        print(f"[edge] {json.dumps({k: summary[k] for k in ('kept', 'dropped_head', 'dropped_tail')})}", flush=True)


def edge_rows_summary(data_path: str, context_path: str, slice_dirs: list, episode_index, edge_frac: float) -> tuple:
    """:class:`EpisodeEdgeRows`' worker: ``(summary, keep)``. Also the measurement behind the default."""
    from datasets import load_from_disk

    meta = load_from_disk(data_path).select_columns(["id", "duration"])
    ids, dur = meta["id"], meta["duration"]
    spans = window_spans(context_path, slice_dirs, ids=set(ids))
    durations = episode_durations(episode_index)
    keep, side = edge_rows(ids, spans, durations, edge_frac)

    ctx = load_from_disk(context_path).select_columns(["id", "assistant_text", "user_text"])
    want = set(ids)
    text = {r["id"]: _norm_text(f"{r['assistant_text']} {r['user_text']}") for r in ctx if r["id"] in want}
    pats = _marker_patterns()
    bins = list(EDGE_BINS)
    per_bin = [{"rows": 0, "hours": 0.0, "marker_rows": 0} for _ in bins[:-1]]
    marker_by_side = {s: {m: 0 for m in pats} for s in ("head", "middle", "tail")}
    for i, rid in enumerate(ids):
        ep, t0, t1 = spans[rid]
        mid = min(max(0.5 * (t0 + t1) / durations[ep], 0.0), 1.0 - 1e-12)
        b = max(k for k in range(len(bins) - 1) if bins[k] <= mid)
        hits = [m for m, p in pats.items() if p.search(text.get(rid, ""))]
        per_bin[b]["rows"] += 1
        per_bin[b]["hours"] += float(dur[i]) / 3600.0
        per_bin[b]["marker_rows"] += bool(hits)
        for m in hits:
            marker_by_side[side.get(i, "middle")][m] += 1

    def rows_hours(idx):
        return {"rows": len(idx), "hours": round(sum(float(dur[i]) for i in idx) / 3600.0, 2)}

    eps_all = {spans[r][0] for r in ids}
    ep_hours = sorted(durations[e] / 3600.0 for e in eps_all)
    summary = {
        "edge_frac": float(edge_frac),
        "rule": "drop a row whose window [t0, t1] (episode seconds) overlaps [0, f*D) or (D - f*D, D]",
        "rows": len(ids),
        "episodes": len(eps_all),
        "episode_hours": {
            "median": round(ep_hours[len(ep_hours) // 2], 3),
            "min": round(ep_hours[0], 3),
            "max": round(ep_hours[-1], 3),
        },
        "kept": rows_hours(keep),
        "dropped_head": rows_hours([i for i, s in side.items() if s == "head"]),
        "dropped_tail": rows_hours([i for i, s in side.items() if s == "tail"]),
        "episodes_losing_all_rows": len(eps_all - {spans[ids[i]][0] for i in keep}),
        "bins_by_midpoint": [
            {
                "from": bins[k],
                "to": bins[k + 1],
                "rows": v["rows"],
                "hours": round(v["hours"], 2),
                "marker_rows": v["marker_rows"],
                "marker_rate": round(v["marker_rows"] / v["rows"], 4) if v["rows"] else None,
            }
            for k, v in enumerate(per_bin)
        ],
        "markers": list(pats),
        "marker_rows_by_side": marker_by_side,
    }
    return summary, keep
