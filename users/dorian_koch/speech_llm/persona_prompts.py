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
    taken from the turns instead: the diarized speaker whose turns contain most of the assistant's
    word onsets in this window.

    ``with_other_side=True`` also writes the OTHER channel as a training side: ``user_alignments``
    (its words in exactly the converter's alignment format: onset-selected, re-based to the window
    start, ``speaker="assistant"`` because that is its role when the loader swaps channels),
    ``user_label`` (same turns-based rule) and ``n_user_words``.
    """

    __sis_hash_exclude__ = {"with_other_side": False}

    def __init__(
        self,
        *,
        train_data: tk.Path,
        dialogue_shard_dirs: list[tk.Path],
        window_select: dict,
        assistant_channel: str = "random",
        selection_seed: int = 0,
        with_other_side: bool = False,
    ):
        assert assistant_channel in ("a", "b", "random"), assistant_channel
        assert set(window_select) <= set(WINDOW_SELECT_KEYS), window_select
        self.train_data = train_data
        self.dialogue_shard_dirs = list(dialogue_shard_dirs)
        self.window_select = {k: window_select[k] for k in WINDOW_SELECT_KEYS if k in window_select}
        self.assistant_channel = assistant_channel
        self.selection_seed = int(selection_seed)
        self.with_other_side = bool(with_other_side)
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
                    wins = select_windows(turns, float(r["duration_sec"]), **self.window_select)
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
                                "user_label": _label_for(wu, turns),
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
                                "assistant_label": _label_for(wa, turns),
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
        print(f"[context] {len(rows)} windows; {n_nolabel} without an assistant label", flush=True)
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


def _label_for(words: list[dict], turns: list[dict]) -> str | None:
    """The diarized speaker whose turns contain most of these word onsets (None if none do)."""
    counts = {}
    for w in words:
        for t in turns:
            if t["start"] <= w["start"] <= t["end"]:
                counts[t["speaker"]] = counts.get(t["speaker"], 0) + 1
    return max(sorted(counts), key=lambda s: counts[s]) if counts else None


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
        self.out_dir = self.output_path("dataset", directory=True)

    __sis_hash_exclude__ = {
        "per_read": False,
        "with_other_side": False,
        "other_prompt_set": None,
        "other_min_words": 8,
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
            n.update({"other_ok": 0, "other_short": 0, "other_noprompt": 0, "other_novoice": 0})

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
