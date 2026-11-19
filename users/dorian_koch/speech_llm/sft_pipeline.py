"""The SFT dialogue corpus as Sisyphus jobs (``sft_dialogues`` writes, ``tts.ChatterboxInferenceV2`` speaks): one
independent chain per (scenario, language) cell, sharded, a shared voice stage, and a cheap check and mixing step.

  voices (shared):  SftVoiceTexts -> SftDesignVoices (Qwen3-TTS VoiceDesign, every language)  -> SftVoicePool
                                  -> SftParlerVoices (Parler-TTS, English user voices)       /
  cell x shard k:   SftSeeds(start=k*S, n=S) -> SftWriteDialogues -> SftCastVoices -> ChatterboxInferenceV2
                    -> SftShardStats (hard invariants raise); the writer and the TTS are GPU jobs over a group of
                    shards (``job_groups``: shards of one cell; one output per shard), so the model loads once
  cell:             HfMergeShards (links): the cell's dataset
  corpus:           SftDataReport (bounds -> flags, duplicates, voice reuse, the mix); ``mix_train_data``: the
                    training mix, ``[(cell dataset, weight)]`` from a scenario and a language weight

Seed ``k`` of a cell depends only on (scenario, language, seed, k) (``sft_dialogues.make_seeds``), so shard ``k`` is
the same job whatever the cell's size: a bigger cell adds shards. Changing one scenario (its
``SCENARIO_VERSIONS`` entry) or one cell's size re-hashes only that cell's jobs (its groups'); a weight reaches only
the report and the mix. Global inputs re-hash every cell on purpose: the writer (``WRITER_VERSION``, its settings), the TTS
settings, the voice pool, the source rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field

from sisyphus import Job, Task, tk

from . import sft_dialogues as sd

__all__ = [
    "SftVoiceTexts",
    "SftParlerVoices",
    "SftVoicePool",
    "SftCastVoices",
    "SftShardStats",
    "SftDataReport",
    "SftCorpus",
    "build_sft_corpus",
    "build_voice_pool",
    "shard_ranges",
    "job_groups",
    "mix_weights",
    "mix_train_data",
]

#: bump when the voice-text prompt or its checks change
VOICE_TEXTS_VERSION = 1
#: columns the TTS carries through from the cast dialogues (``id`` is the TTS's own)
TTS_KEEP = ["sft_id", "scenario", "lang", "prompt_lang", "policy", "level", "system_prompt", "user_facts", "seed",
            "dialogue", "tries"]  # fmt: skip

#: Sanity bounds per (scenario, language) cell; a value outside is a flag in the report (sink), not a failure.
#: Sources: the pilot (sft_data_plan.md "Quality so far": writer v11 p19a, 1,008 seeds, kept 63-90% per scenario;
#: Qwen3-TTS on 204 dialogues: mean CER 1.8-4.3% per language, turns redone 4.6-7.4%, still over 10% after three
#: tries 3.0-6.8%; ~95 s of speech per dialogue); seeds ask for 4-21 turns.
SFT_BOUNDS = {
    "kept_rate_min": 0.45,  # pilot minimum 0.63 (tutor), less ~2 binomial SD at 32 seeds
    "cer_mean_max": 0.08,  # 2x the worst pilot language mean (en 4.3%)
    "over_max": 0.14,  # segments over max_err after the retries: 2x the worst pilot share (6.8%)
    "redone_max": 0.15,  # segments synthesized more than once: 2x the worst pilot share (7.4%)
    "dur_mean_s": [30.0, 300.0],  # pilot mean ~95 s; help is short (4-8 turns), longtask long (14-20)
    "turns_mean": [4.0, 22.0],  # what the seeds ask for, +1 for the parity fix
    "silent_turns_max": 0,  # a turn whose audio region is below SILENT_DBFS: a TTS failure
    "script_bad_max": 0.02,  # segments whose letters are mostly not their language's script
    "own_lang_min": 0.6,  # speech seconds in the user's language, single-language cells (not tutor / switch)
    "timbre_max": 0.25,  # speech seconds spoken timbre-only (another language than the voice), not tutor
    "dup_opening_max": 0.05,  # dialogues of a cell whose first user turn repeats another's exactly
    "voice_reuse_max": 4.0,  # a voice's uses over the mean of its pool (judged once the mean is >= 2)
    "parler_sd": 3.0,  # the Parler share of English user voices within this many binomial SDs of the target
    "mix_hours_ratio": [0.5, 2.0],  # a cell's share of training audio over its weight
}
SILENT_DBFS = -50.0  # turns are levelled to -23 dBFS speech RMS (ChatterboxInferenceV2 loudness_dbfs)


# ---------------------------------------------------------------------------------------------------------------
# structure: shards and the mix


def shard_ranges(n: int, size: int) -> list[tuple[int, int]]:
    """``(start, count)`` per shard of a cell of ``n`` seeds: fixed ``size`` from 0, so growing ``n`` keeps every full
    shard (only a partial last one changes)."""
    assert n >= 0 and size > 0
    return [(s, min(size, n - s)) for s in range(0, n, size)]


def mix_weights(cells, scenario_weights: dict, lang_weights: dict) -> dict:
    """``{(scenario, lang): share}``: scenario weight x language weight, normalised over ``cells``; zero-weight cells
    are left out."""
    w = {c: float(scenario_weights[c[0]]) * float(lang_weights[c[1]]) for c in cells}
    assert all(v >= 0 for v in w.values()), w
    z = sum(w.values())
    assert z > 0, "every mix weight is zero"
    return {c: v / z for c, v in sorted(w.items()) if v > 0}


def mix_train_data(cell_hf: dict, scenario_weights: dict, lang_weights: dict) -> list:
    """The training mix ``[(cell dataset, weight)]`` (``finetune_adapters.train_data_specs``: the loader draws a cell
    with probability proportional to its weight)."""
    w = mix_weights(cell_hf, scenario_weights, lang_weights)
    return [(cell_hf[c], w[c]) for c in sorted(w)]


# ---------------------------------------------------------------------------------------------------------------
# voices


VT_SYSTEM = "You write short things people say out loud, as native speakers say them."
VT_PROMPT = """Write one thing this person might say out loud in an everyday conversation (to a friend, on the \
phone, or to a voice assistant), in {language}, as a native speaker living in {place} says it.
WHO: {persona}
- {length}, one to three sentences, natural spoken register (a filler or a restart is fine).
- About their own life, plans, opinions or a small problem.
- No personal names, no digits (numbers as words), no abbreviations, no words from another language, no quotation \
marks, brackets or symbols.
Answer as JSON: {{"text": "..."}}"""
VT_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def voice_text_problem(text: str, lang: str) -> str | None:
    """Why ``text`` is no good voice-prompt text in ``lang`` (None = fine): ~10 s of speech (20-38 words; 40-70
    characters for Chinese), plain spoken text in the language's own script."""
    if re.search(r"[\d<>\"“”„«»()\[\]{}/*#@&%+=_|~]", text):
        return "digits or symbols"
    n = len(re.sub(r"\W", "", text)) if lang == "zh" else len(text.split())
    lo, hi = (40, 70) if lang == "zh" else (20, 38)
    if not lo <= n <= hi:
        return "length"
    if lang in ("zh", "ru") and re.search(r"[A-Za-z]", text):
        return "Latin letters"
    if not sd._foreign_ok(text, lang) or not sd._script_ok(text, lang):
        return "script"
    return None


def write_voice_texts(personas: list[dict], langs: list[str], n: int, seed: int, url: str, model: str,
                      temperature: float = 1.0, workers: int = 64, max_tries: int = 3) -> tuple[list[dict], dict]:  # fmt: skip
    """``n`` texts per language, each for a random persona (``k``-th: ``Random(f"vt/{lang}/{seed}/{k}")``) in a
    random native locale; a text failing :func:`voice_text_problem` is drawn again (up to ``max_tries``)."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from .common import json_schema_format

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "nothing"), base_url=url, timeout=600)
    stats = {"asked": 0, "kept": 0, "reasons": {}}

    def one(item):
        lang, k = item
        rng = random.Random(f"vt/{lang}/{seed}/{k}")
        per = personas[rng.randrange(len(personas))]
        length = "40 to 65 Chinese characters" if lang == "zh" else "20 to 35 words"
        msg = VT_PROMPT.format(language=sd.LANGS[lang], place=rng.choice(sd.LOCALES[lang]),
                               persona=sd._persona_text_clean(per["text"]), length=length)  # fmt: skip
        why = None
        for t in range(max_tries):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=400, seed=t,
                    messages=[{"role": "system", "content": VT_SYSTEM}, {"role": "user", "content": msg}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    **json_schema_format(VT_SCHEMA, "voice_text"))  # fmt: skip
                text = " ".join(json.loads(r.choices[0].message.content)["text"].split())
                why = voice_text_problem(text, lang)
            except Exception as e:  # malformed output or a request error: drawn again
                text, why = "", f"error: {type(e).__name__}"
            if why is None:
                return {"id": f"{lang}_{k}", "lang": lang, "text": text, "persona_id": per["id"]}
            stats["reasons"][why] = stats["reasons"].get(why, 0) + 1
        return None

    items = [(lg, k) for lg in langs for k in range(n)]
    stats["asked"] = len(items)
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, items))
    rows, seen = [], set()
    for r in res:
        if r is not None and (r["lang"], r["text"]) not in seen:
            seen.add((r["lang"], r["text"]))
            rows.append(r)
    stats["kept"] = len(rows)
    stats["kept_by_lang"] = {lg: sum(r["lang"] == lg for r in rows) for lg in langs}
    return rows, stats


class SftVoiceTexts(Job):
    """Voice-prompt texts (:func:`write_voice_texts`): per language ``n`` things a person might say, ~10 s each,
    written by ``model`` for a random persona (``personas``: ``SftSourceRows`` output) -> ``texts.jsonl``
    (``{"id", "lang", "text", "persona_id"}``), ``stats.json``. Independent of the dialogues, so the shared voice
    stage does not depend on any cell. Fails when a language keeps less than 90% of ``n``."""

    def __init__(self, *, personas: tk.Path, langs: list[str], n: int, seed: int = 0,
                 model: str = "google/gemma-4-31B-it", vllm_args: list[str] | None = None,
                 temperature: float = 1.0):  # fmt: skip
        self.personas, self.langs, self.n, self.seed = personas, list(langs), n, seed
        self.model, self.vllm_args, self.temperature = model, vllm_args, temperature
        self.out_texts = self.output_path("texts.jsonl")
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 1}

    @classmethod
    def hash(cls, parsed_args):
        return super().hash({**parsed_args, "version": VOICE_TEXTS_VERSION})

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server

        personas = [json.loads(x) for x in open(self.personas.get(), encoding="utf-8")]
        with vllm_server(self.model, max_model_len=8192, extra_args=self.vllm_args) as url:
            rows, stats = write_voice_texts(personas, self.langs, self.n, self.seed, url, self.model,
                                            self.temperature)  # fmt: skip
        with open(self.out_stats.get(), "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        short = {lg: k for lg, k in stats["kept_by_lang"].items() if k < 0.9 * self.n}
        if short:
            raise RuntimeError(f"SftVoiceTexts: too few texts {short} of {self.n}; reasons {stats['reasons']}")
        with open(self.out_texts.get(), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[SftVoiceTexts] {stats}", flush=True)


#: Parler-TTS descriptions in the words of its training labels (the attributes it controls); English only, so its
#: voices serve only English users
PARLER_EXPR = ["a slightly monotone", "a slightly expressive and animated", "an expressive and animated"]
PARLER_SPEED = ["a slightly slow speed", "a moderate speed", "a slightly fast speed", "a fast speed"]
PARLER_PITCH = ["a low pitch", "a slightly low pitch", "a moderate pitch", "a slightly high pitch", "a high pitch"]
PARLER_QUALITY = [
    "The recording is of very high quality, with the speaker's voice sounding clear and very close up.",
    "The recording is clear, with the speaker's voice sounding close up.",
    "The recording is slightly noisy, with the speaker's voice sounding close up.",
    "The recording is slightly noisy and slightly distant-sounding.",
]


def parler_jobs(texts: list[str], n_per_gender: int, seed: int) -> list[dict]:
    """``sft_parler_voices.py``'s clips: per gender ``n_per_gender`` distinct descriptions, each with an English text
    (shuffled, in rotation) and its own sampling seed; checked and filed as ``en/user/<gender>/parler_<id>``."""
    tx = sorted(set(texts))
    assert tx, "no English voice-prompt texts"
    random.Random(f"parler/{seed}/texts").shuffle(tx)
    out, k = [], 0
    for g in ("female", "male"):
        rng, seen = random.Random(f"parler/{g}/{seed}"), set()
        while len(seen) < n_per_gender:
            d = (f"A {g} speaker delivers {rng.choice(PARLER_EXPR)} speech with {rng.choice(PARLER_SPEED)} and "
                 f"{rng.choice(PARLER_PITCH)}. {rng.choice(PARLER_QUALITY)}")  # fmt: skip
            if d in seen:
                continue
            seen.add(d)
            out.append({"id": f"en_user_{g}_{k}", "name": f"parler_en_user_{g}_{k}", "lang": "en", "role": "user",
                        "gender": g, "instruct": d, "text": tx[k % len(tx)], "seed": seed * 100003 + k})  # fmt: skip
            k += 1
    return out


class SftParlerVoices(Job):
    """English user voices from Parler-TTS large v1 (:func:`parler_jobs`, ``n_per_gender`` per gender; texts from
    ``SftVoiceTexts``), kept when Qwen3-ASR reads them back (``sft_voice_design.py check``) -> ``out_dir``
    ``en/user/<gender>/parler_*.wav`` + transcript. Its recordings are noisier than VoiceDesign's (pilot: quiet
    frames 25-29 dB under speech vs 52 dB), the variance it is there for."""

    def __init__(self, *, parler_python: tk.AbstractPath, annotate_python: tk.AbstractPath, texts: tk.Path,
                 n_per_gender: int = 30, seed: int = 1, asr_model: str = "Qwen/Qwen3-ASR-0.6B"):  # fmt: skip
        self.parler_python, self.annotate_python, self.texts = parler_python, annotate_python, texts
        self.n_per_gender, self.seed, self.asr_model = n_per_gender, seed, asr_model
        self.out_work = self.output_path("work", directory=True)
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 2})

    def run(self):
        import shutil
        import subprocess

        w, here = self.out_work.get(), os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(f"{w}/jobs.json"):
            texts = sd.voice_prompt_texts(self.texts.get(), ["en"])["en"]
            with open(f"{w}/jobs.json", "w", encoding="utf-8") as f:
                json.dump(parler_jobs(texts, self.n_per_gender, self.seed), f, ensure_ascii=False, indent=1)
        subprocess.run([self.parler_python.get(), f"{here}/sft_parler_voices.py", w], check=True)
        subprocess.run([self.annotate_python.get(), f"{here}/sft_voice_design.py", "check", w, self.asr_model],
                       check=True)  # fmt: skip
        if os.path.isdir(self.out_dir.get()):
            shutil.rmtree(self.out_dir.get())
        shutil.copytree(f"{w}/voices", self.out_dir.get())


class SftVoicePool(Job):
    """The one voice directory the TTS reads (``<lang>/<role>/<gender>/<name>.wav`` + ``.txt``): ``sources``' voice
    directories joined (hard links, no copies) -> ``out_dir``; ``out_index`` (``pool.json``): ``{"<lang>/<role>/
    <gender>": {"<source>": [names]}}``, the source being a name's prefix before ``_`` (``vd``, ``parler``)."""

    def __init__(self, *, sources: list[tk.Path]):
        self.sources = sources
        self.out_dir = self.output_path("voices", directory=True)
        self.out_index = self.output_path("pool.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        index = {}
        for src in self.sources:
            for root, _, files in os.walk(src.get()):
                rel = os.path.relpath(root, src.get())
                for fn in sorted(files):
                    stem, ext = os.path.splitext(fn)
                    if ext not in (".wav", ".txt"):
                        continue
                    os.makedirs(os.path.join(self.out_dir.get(), rel), exist_ok=True)
                    dst = os.path.join(self.out_dir.get(), rel, fn)
                    if os.path.exists(dst):
                        raise RuntimeError(f"SftVoicePool: two sources hold {rel}/{fn}")
                    os.link(os.path.join(root, fn), dst)
                    if ext == ".wav":
                        if not os.path.exists(os.path.join(root, stem + ".txt")):
                            raise RuntimeError(f"SftVoicePool: {rel}/{fn} has no transcript")
                        index.setdefault(rel, {}).setdefault(stem.split("_")[0], []).append(stem)
        with open(self.out_index.get(), "w") as f:
            json.dump({k: {s: sorted(v) for s, v in sorted(d.items())} for k, d in sorted(index.items())}, f, indent=1)
        print(
            f"[SftVoicePool] { {k: {s: len(v) for s, v in d.items()} for k, d in sorted(index.items())} }", flush=True
        )


def build_voice_pool(*, personas: tk.Path, env: dict, langs: list[str], n_assistant: int, n_user: int,
                     n_parler: int, seed: int = 1, region: float = 0.5, writer_model: str = "google/gemma-4-31B-it",
                     vllm_args: list[str] | None = None, **_unused) -> SftVoicePool:  # fmt: skip
    """The shared voice stage: texts, VoiceDesign voices for every language, Parler-TTS English user voices,
    joined. ``n_*``: voices per language and gender (Parler: English users only); ``**_unused``: the voice spec's
    draw settings (``parler_share``), read by the cast."""
    n_texts = 2 * (n_assistant + n_user + n_parler)
    texts = SftVoiceTexts(personas=personas, langs=langs, n=n_texts, seed=seed, model=writer_model,
                          vllm_args=vllm_args)  # fmt: skip
    design = sd.SftDesignVoices(tts_python=env["qwen3_tts"], annotate_python=env["annotate"], texts=texts.out_texts,
                                langs=langs, n_assistant=n_assistant, n_user=n_user, seed=seed, region=region)  # fmt: skip
    sources = [design.out_dir]
    if n_parler:
        sources.append(SftParlerVoices(parler_python=env["parler"], annotate_python=env["annotate"],
                                       texts=texts.out_texts, n_per_gender=n_parler, seed=seed).out_dir)  # fmt: skip
    return SftVoicePool(sources=sources)


# ---------------------------------------------------------------------------------------------------------------
# per shard: voices drawn, stats


def cast_voices(rows: list[dict], index: dict, parler_share: float) -> list[dict]:
    """Each written dialogue's ``voices`` (``<lang>/<role>/<gender>/rng_v``, ``sft_dialogues.to_tts_dialogue``) made
    concrete from the pool ``index`` (``SftVoicePool``): per dialogue and role its own random stream (dialogue id +
    role), so a draw depends on nothing else; a pool with Parler voices (English users) takes one with probability
    ``parler_share``. Adds ``sft_id`` (the writer's id; the TTS names rows itself)."""
    out = []
    for r in rows:
        d = json.loads(r["dialogue"])
        voices = {}
        for role in sorted(d["voices"]):
            lang, rdir, gender = d["voices"][role].split("/")[:3]
            pools = index[f"{lang}/{rdir}/{gender}"]
            rng = random.Random(f"{r['id']}/{role}/voice")
            src = "parler" if pools.get("parler") and rng.random() < parler_share else "vd"
            voices[role] = f"{lang}/{rdir}/{gender}/{rng.choice(pools[src])}"
        if len(set(voices.values())) != len(voices):
            raise RuntimeError(f"cast_voices: {r['id']} got one voice for two speakers: {voices}")
        out.append({**r, "sft_id": r["id"], "dialogue": json.dumps({**d, "voices": voices}, ensure_ascii=False)})
    return out


class SftCastVoices(Job):
    """:func:`cast_voices` over one shard's written dialogues (``written``: ``SftWriteDialogues.out_hf``) with the
    voice pool (``pool``: ``SftVoicePool.out_index``) -> ``out_hf``, the TTS input."""

    def __init__(self, *, written: tk.Path, pool: tk.Path, parler_share: float):
        self.written, self.pool, self.parler_share = written, pool, parler_share
        self.out_hf = self.output_path("out_hf", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from datasets import Dataset, load_from_disk

        rows = cast_voices(list(load_from_disk(self.written.get())), json.load(open(self.pool.get())),
                           self.parler_share)  # fmt: skip
        if not rows:
            raise RuntimeError("SftCastVoices: the writer kept no dialogue in this shard")
        Dataset.from_list(rows).save_to_disk(self.out_hf.get())


def _cols(x) -> dict:
    """A ``Sequence`` of structs as a dict of lists (Arrow may hand it over as a list of dicts)."""
    if isinstance(x, list):
        return {k: [e[k] for e in x] for k in (x[0] if x else {})}
    return x or {}


def _norm_text(x: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", re.sub(r"</?[a-z]{2}>|<cut/>", " ", x.lower())).split())


def _digest(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]


def dialogue_stats(row: dict, max_err: dict) -> tuple[dict, list[str]]:
    """One spoken dialogue's numbers and its hard problems. ``row``: a TTS output row with ``tracks`` ``{speaker:
    float array}`` and ``sr`` (decoded) and the kept columns."""
    import numpy as np

    sid, problems = row.get("sft_id"), []
    turns, segs, vo = _cols(row["turns"]), _cols(row["segments"]), _cols(row["voices"])
    tracks, sr = row["tracks"], row["sr"]
    voice = {s: os.path.splitext(v)[0] for s, v in zip(vo.get("speaker", []), vo.get("voice", []))}  # pool names
    if set(tracks) != {"user", "assistant"}:
        problems.append(f"{sid}: tracks {sorted(tracks)}")
    if len({len(w) for w in tracks.values()}) > 1:
        problems.append(f"{sid}: track lengths differ {[len(w) for w in tracks.values()]}")
    if any(len(w) < sr for w in tracks.values()):
        problems.append(f"{sid}: a track under 1 s")
    if len(set(voice.values())) < len(voice) or set(voice) != set(tracks):
        problems.append(f"{sid}: voices {voice}")
    n = len(turns.get("speaker", []))
    if n < 2:
        problems.append(f"{sid}: {n} turns")
    silent, empty = 0, 0
    for k in range(n):
        sp, a, b = turns["speaker"][k], turns["start_time"][k], turns["end_time"][k]
        if b - a < 0.05:
            empty += 1
            continue
        w = tracks.get(sp)
        if w is None:
            continue
        x = w[int(a * sr) : int(b * sr)]
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
        silent += 20 * math.log10(rms + 1e-12) < SILENT_DBFS
    dur = max((len(w) for w in tracks.values()), default=0) / sr
    lang_s, timbre_s, script_bad, over, redone, errs = {}, 0.0, 0, 0, 0, []
    for k in range(len(segs.get("lang", []))):
        lg, txt = segs["lang"][k], segs["text"][k]
        s = max(0.0, float(segs["end"][k]) - float(segs["start"][k]))
        lang_s[lg] = lang_s.get(lg, 0.0) + s
        sp = turns["speaker"][segs["turn"][k]] if 0 <= segs["turn"][k] < n else None
        if sp in voice and voice[sp].split("/")[0] != lg:
            timbre_s += s
        letters = [c for c in txt if c.isalpha()]
        if len(letters) >= 3 and lg in sd._SCRIPT:
            script_bad += sum(bool(sd._SCRIPT[lg].match(c)) for c in letters) < 0.8 * len(letters)
        e = segs["err"][k]
        if e is not None and not math.isnan(e):
            errs.append(float(e))
            over += e > max_err.get(lg, max_err["*"])
        redone += (segs["attempts"][k] or 1) > 1
    texts = [_norm_text(t) for t in turns.get("text", [])]
    first_user = next((t for t, sp in zip(texts, turns.get("speaker", [])) if sp == "user"), "")
    return {
        "sft_id": sid, "dur": round(dur, 2), "turns": n, "interrupted": int(sum(turns.get("interrupted") or [])),
        "silent_turns": int(silent), "empty_turns": empty, "segments": len(errs), "over": int(over),
        "redone": int(redone), "err_mean": round(float(np.mean(errs)), 4) if errs else None,
        "err_max": round(max(errs), 4) if errs else None, "script_bad": int(script_bad),
        "lang_s": {k: round(v, 2) for k, v in sorted(lang_s.items())}, "timbre_s": round(timbre_s, 2),
        "policy": row.get("policy"), "voices": voice, "hash": _digest("\n".join(texts)),
        "opening": _digest(first_user),
    }, problems  # fmt: skip


def _tts_rows(path: str):
    """A TTS output's rows with ``tracks`` decoded (FLAC bytes read from Arrow, no ``datasets`` decode)."""
    import soundfile as sf
    from datasets import load_from_disk

    ds = load_from_disk(path).with_format("arrow")
    for i in range(0, len(ds), 16):
        for r in ds[i : i + 16].to_pylist():
            sa = _cols(r.pop("speaker_audio"))
            tracks, sr = {}, None
            for sp, a in zip(sa.get("speaker", []), sa.get("audio", [])):
                w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                tracks[sp] = w if w.ndim == 1 else w.mean(1)
            yield {**r, "tracks": tracks, "sr": sr or 24000}


def shard_stats(cell: dict, seed_ids: list[str], write_stats: dict, cast_rows: list[dict], tts_rows,
                max_err: dict) -> tuple[dict, list[str]]:  # fmt: skip
    """One shard's numbers (``dialogues``: per dialogue, :func:`dialogue_stats`) and its hard problems: a dialogue
    of another cell, ids that do not match from seeds to cast to speech, broken audio."""
    problems = []
    cast_ids = [r["id"] for r in cast_rows]
    if len(set(cast_ids)) != len(cast_ids):
        problems.append("duplicate ids in the cast")
    if not set(cast_ids) <= set(seed_ids):
        problems.append(f"cast ids not among the seeds: {sorted(set(cast_ids) - set(seed_ids))[:5]}")
    wrong = [r["id"] for r in cast_rows if (r["scenario"], r["lang"]) != (cell["scenario"], cell["lang"])]
    if wrong:
        problems.append(f"dialogues of another cell: {wrong[:5]}")
    dialogues = []
    for r in tts_rows:
        st, pr = dialogue_stats(r, max_err)
        dialogues.append(st)
        problems += pr
    tts_ids = [d["sft_id"] for d in dialogues]
    if sorted(tts_ids) != sorted(cast_ids):
        problems.append(f"speech rows {len(tts_ids)} vs cast rows {len(cast_ids)} (ids differ)")
    return {"cell": cell, "seeds": len(seed_ids), "kept": len(cast_ids), "spoken": len(dialogues),
            "drop_reasons": write_stats.get("reasons", {}), "writer_wall_s": write_stats.get("wall_s"),
            "dialogues": dialogues}, problems  # fmt: skip


class SftShardStats(Job):
    """:func:`shard_stats` for one shard -> ``stats.json``; raises after writing it on any hard problem (a dialogue
    of another cell, ids lost between seeds, cast and speech, a missing / empty / unequal track, one voice for both
    speakers). Bounds are judged across shards by ``SftDataReport``."""

    def __init__(self, *, cell: dict, seeds: tk.Path, write_stats: tk.Path, cast: tk.Path, tts: tk.Path,
                 max_err: dict):  # fmt: skip
        self.cell, self.seeds, self.write_stats, self.cast, self.tts = cell, seeds, write_stats, cast, tts
        self.max_err = max_err
        self.out_stats = self.output_path("stats.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import load_from_disk

        seed_ids = [json.loads(x)["id"] for x in open(self.seeds.get(), encoding="utf-8")]
        cast = list(load_from_disk(self.cast.get()).select_columns(["id", "scenario", "lang"]))
        st, problems = shard_stats(self.cell, seed_ids, json.load(open(self.write_stats.get())), cast,
                                   _tts_rows(self.tts.get()), self.max_err)  # fmt: skip
        st["problems"] = problems
        with open(self.out_stats.get(), "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)
        print(f"[SftShardStats] {self.cell}: kept {st['kept']}/{st['seeds']}, spoken {st['spoken']}", flush=True)
        if problems:
            raise RuntimeError(f"SftShardStats {self.cell}: {len(problems)} problems: {problems[:10]}")


# ---------------------------------------------------------------------------------------------------------------
# the corpus report


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def corpus_report(name: str, shards: list[dict], mix: list[dict], scenario_weights: dict, lang_weights: dict,
                  pool: dict, parler_share: float, bounds: dict) -> tuple[dict, list[str]]:  # fmt: skip
    """Per cell and overall numbers, ``flags`` (a bound broken) and the hard problems: a dialogue repeated across
    shards (the same text twice is a seeding bug), a mix whose weights are not scenario x language weights, a mixed
    cell without dialogues. ``shards``: ``SftShardStats`` outputs loaded; ``mix``: ``[{"scenario", "lang",
    "weight"}]``."""
    b, flags, problems = bounds, [], []
    cells: dict = {}
    for st in shards:
        c = cells.setdefault(
            (st["cell"]["scenario"], st["cell"]["lang"]), {"shards": 0, "seeds": 0, "kept": 0, "d": []}
        )
        c["shards"] += 1
        c["seeds"] += st["seeds"]
        c["kept"] += st["kept"]
        c["d"] += st["dialogues"]

    def flag(cell, check, value, bound):
        flags.append({"cell": cell, "check": check, "value": value, "bound": bound})

    out_cells = {}
    for (sc, lg), c in sorted(cells.items()):
        key, d = f"{sc}/{lg}", c["d"]
        segs = sum(x["segments"] for x in d) or 1
        tot_s = sum(sum(x["lang_s"].values()) for x in d) or 1e-9
        single = [x for x in d if x["policy"] in ("same", "cross")]
        own = sum(x["lang_s"].get(lg, 0.0) for x in single) / (sum(sum(x["lang_s"].values()) for x in single) or 1e-9)
        errs = [x["err_mean"] for x in d if x["err_mean"] is not None]
        openings = [x["opening"] for x in d]
        r = {
            "shards": c["shards"], "seeds": c["seeds"], "kept": c["kept"],
            "kept_rate": round(c["kept"] / max(1, c["seeds"]), 3), "hours": round(sum(x["dur"] for x in d) / 3600, 3),
            "dur_mean_s": round(sum(x["dur"] for x in d) / max(1, len(d)), 1),
            "turns_mean": round(sum(x["turns"] for x in d) / max(1, len(d)), 2),
            "interrupted_per_dialogue": round(sum(x["interrupted"] for x in d) / max(1, len(d)), 2),
            "cer_mean": round(sum(errs) / len(errs), 4) if errs else None,
            "cer_p90_dialogue_max": _q([x["err_max"] for x in d if x["err_max"] is not None], 0.9),
            "over": round(sum(x["over"] for x in d) / segs, 4), "redone": round(sum(x["redone"] for x in d) / segs, 4),
            "script_bad": round(sum(x["script_bad"] for x in d) / segs, 4),
            "silent_turns": sum(x["silent_turns"] for x in d), "empty_turns": sum(x["empty_turns"] for x in d),
            "own_lang_share": round(own, 3) if single else None,
            "timbre_share": round(sum(x["timbre_s"] for x in d) / tot_s, 3),
            "lang_share": {k: round(sum(x["lang_s"].get(k, 0.0) for x in d) / tot_s, 3) for k in sd.LANGS},
            "dup_opening": round(1 - len(set(openings)) / max(1, len(openings)), 3),
        }  # fmt: skip
        out_cells[key] = r
        if r["kept_rate"] < b["kept_rate_min"]:
            flag(key, "kept_rate", r["kept_rate"], b["kept_rate_min"])
        if r["cer_mean"] is not None and r["cer_mean"] > b["cer_mean_max"]:
            flag(key, "cer_mean", r["cer_mean"], b["cer_mean_max"])
        for k in ("over", "redone", "script_bad"):
            if r[k] > b[f"{k}_max"]:
                flag(key, k, r[k], b[f"{k}_max"])
        for k in ("dur_mean_s", "turns_mean"):
            if d and not b[k][0] <= r[k] <= b[k][1]:
                flag(key, k, r[k], b[k])
        if r["silent_turns"] > b["silent_turns_max"]:
            flag(key, "silent_turns", r["silent_turns"], b["silent_turns_max"])
        if sc != "tutor" and r["own_lang_share"] is not None and r["own_lang_share"] < b["own_lang_min"]:
            flag(key, "own_lang_share", r["own_lang_share"], b["own_lang_min"])
        if sc != "tutor" and r["timbre_share"] > b["timbre_max"]:
            flag(key, "timbre_share", r["timbre_share"], b["timbre_max"])
        if sc == "tutor" and d and r["timbre_share"] == 0:
            flag(key, "timbre_share", 0, "> 0 in tutoring (the target language is spoken)")
        if r["dup_opening"] > b["dup_opening_max"]:
            flag(key, "dup_opening", r["dup_opening"], b["dup_opening_max"])

    every = [x for c in cells.values() for x in c["d"]]
    by_hash: dict = {}
    for x in every:
        by_hash.setdefault(x["hash"], []).append(x["sft_id"])
    dups = [v for v in by_hash.values() if len(v) > 1]
    if dups:
        problems.append(f"{len(dups)} dialogue texts occur more than once: {dups[:5]}")

    uses: dict = {}
    for x in every:
        for v in x["voices"].values():
            uses[v] = uses.get(v, 0) + 1
    voices = {}
    for key, srcs in sorted(pool.items()):
        names = [f"{key}/{s}" for v in srcs.values() for s in v]
        u = [uses.get(n, 0) for n in names]
        mean = sum(u) / max(1, len(u))
        voices[key] = {"pool": len(names), "used": sum(x > 0 for x in u), "uses": sum(u), "max_uses": max(u, default=0)}
        if mean >= 2 and max(u) > b["voice_reuse_max"] * mean:
            flag(key, "voice_reuse", round(max(u) / mean, 2), b["voice_reuse_max"])
    en_users = [x["voices"].get("user", "") for x in every if x["voices"].get("user", "").startswith("en/user/")]
    k_parler = sum(os.path.basename(v).startswith("parler_") for v in en_users)
    parler = {"en_user_dialogues": len(en_users), "parler": k_parler, "share": round(k_parler / max(1, len(en_users)), 3),
              "target": parler_share}  # fmt: skip
    sd_ = math.sqrt(max(1e-9, len(en_users) * parler_share * (1 - parler_share)))
    if en_users and abs(k_parler - parler_share * len(en_users)) > b["parler_sd"] * sd_:
        flag("en/user", "parler_share", parler["share"], f"{parler_share} +- {b['parler_sd']} SD")
    for x in every:
        if not x["voices"].get("user", "").startswith("en/") and "parler_" in x["voices"].get("user", ""):
            problems.append(f"{x['sft_id']}: a Parler voice outside English")

    want = mix_weights([(m["scenario"], m["lang"]) for m in mix], scenario_weights, lang_weights)
    got = {(m["scenario"], m["lang"]): m["weight"] for m in mix}
    if abs(sum(got.values()) - 1) > 1e-6 or any(abs(got[c] - want.get(c, -1)) > 1e-9 for c in got):
        problems.append(f"the mix weights are not the scenario x language weights: {got} vs {want}")
    hours_w = {c: got[c] * out_cells.get(f"{c[0]}/{c[1]}", {}).get("dur_mean_s", 0.0) for c in got}
    z = sum(hours_w.values()) or 1e-9
    mix_rows = []
    for c in sorted(got):
        key = f"{c[0]}/{c[1]}"
        if out_cells.get(key, {}).get("kept", 0) == 0:
            problems.append(f"{key} is in the mix but has no dialogues")
        ratio = (hours_w[c] / z) / got[c]
        mix_rows.append({"cell": key, "weight": round(got[c], 5), "hours_share": round(hours_w[c] / z, 5),
                         "ratio": round(ratio, 3)})  # fmt: skip
        if not b["mix_hours_ratio"][0] <= ratio <= b["mix_hours_ratio"][1]:
            flag(key, "mix_hours_ratio", round(ratio, 3), b["mix_hours_ratio"])

    totals = {"cells": len(cells), "shards": sum(c["shards"] for c in cells.values()),
              "seeds": sum(c["seeds"] for c in cells.values()), "kept": sum(c["kept"] for c in cells.values()),
              "hours": round(sum(x["dur"] for x in every) / 3600, 2),
              "cer_mean": round(sum(x["err_mean"] for x in every if x["err_mean"] is not None)
                                / max(1, sum(x["err_mean"] is not None for x in every)), 4),
              "over": round(sum(x["over"] for x in every) / max(1, sum(x["segments"] for x in every)), 4)}  # fmt: skip
    totals["kept_rate"] = round(totals["kept"] / max(1, totals["seeds"]), 3)
    report = {"sft_report": name, "totals": totals, "n_flags": len(flags), "flags": flags, "problems": problems,
              "cells": out_cells, "mix": mix_rows, "voices": voices, "parler": parler,
              "duplicates": {"exact_text": dups[:20]}, "bounds": bounds}  # fmt: skip
    return report, problems


class SftDataReport(Job):
    """:func:`corpus_report` over every shard's stats -> ``summary.json`` (its ``flags``: a broken bound, read in the
    result sink); raises after writing it on a hard problem. ``shards``: ``[{"scenario", "lang", "shard",
    "stats"}]``; ``mix``: ``[{"scenario", "lang", "weight"}]``; ``pool``: ``SftVoicePool.out_index``."""

    def __init__(self, *, name: str, shards: list[dict], mix: list[dict], scenario_weights: dict, lang_weights: dict,
                 pool: tk.Path, parler_share: float, bounds: dict | None = None):  # fmt: skip
        self.name, self.shards, self.mix = name, shards, mix
        self.scenario_weights, self.lang_weights = scenario_weights, lang_weights
        self.pool, self.parler_share = pool, parler_share
        self.bounds = dict(SFT_BOUNDS if bounds is None else bounds)
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        shards = [json.load(open(s["stats"].get(), encoding="utf-8")) for s in self.shards]
        rep, problems = corpus_report(self.name, shards, self.mix, self.scenario_weights, self.lang_weights,
                                      json.load(open(self.pool.get())), self.parler_share, self.bounds)  # fmt: skip
        rep["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.out_summary.get(), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"[SftDataReport] {rep['totals']}; {rep['n_flags']} flags: {rep['flags'][:10]}", flush=True)
        if problems:
            raise RuntimeError(f"SftDataReport: {len(problems)} problems: {problems[:10]}")


# ---------------------------------------------------------------------------------------------------------------
# the corpus


@dataclass
class SftCorpus:
    pool: SftVoicePool
    #: (scenario, lang) -> per shard ``{"seeds", "write", "cast", "tts", "stats"}`` jobs and ``part`` (its index in
    #: the group's writer / TTS outputs)
    shards: dict = field(default_factory=dict)
    #: (scenario, lang) -> the cell's dataset (its shards linked)
    cells: dict = field(default_factory=dict)
    report: SftDataReport | None = None
    #: ``[(cell dataset, weight)]``
    train_data: list = field(default_factory=list)


def job_groups(shards: list[tuple[str, str, int]], group: str = "cell", per_job: int = 1) -> list[list]:
    """Which shards ``(scenario, lang, k)`` one GPU job (writer, TTS) processes. ``"cell"``: shards ``k`` with the same
    ``k // per_job`` of one cell, so a job never mixes cells and growing a cell keeps its full groups (the full run:
    ``per_job=1``). ``"language"``: every shard of a language in one job, across scenarios (test runs only: a change
    in one cell re-runs its language's whole group). A shard's seeds and outputs do not depend on its group."""
    assert group in ("cell", "language") and per_job >= 1, (group, per_job)
    out: dict = {}
    for sc, lg, k in shards:
        key = (sc, lg, k // per_job) if group == "cell" else lg
        out.setdefault(key, []).append((sc, lg, k))
    return list(out.values())


def build_sft_corpus(*, name: str, rows: dict, env: dict, cell_seeds: int, shard_seeds: int, seed: int,
                     writer: dict, tts: dict, voices: dict, scenario_weights: dict, lang_weights: dict,
                     group: str = "cell", shards_per_job: int = 1, cell_sizes: dict | None = None,
                     scenarios: list[str] | None = None, langs: list[str] | None = None) -> SftCorpus:  # fmt: skip
    """Every (scenario, language) cell: ``cell_seeds`` seeds (``cell_sizes`` overrides per ``(scenario, lang)``) in
    shards of ``shard_seeds``, each its own seeds -> cast -> stats chain, written and spoken in GPU jobs over
    :func:`job_groups` (``group``, ``shards_per_job``; one output per shard); the shared voice pool (``voices``:
    :func:`build_voice_pool`'s arguments + ``parler_share``); the report and the mix from the weights.

    ``rows``: ``{(source, lang): SftSourceRows.out_rows}`` (personas: ``("personas", "en")`` serves every language);
    ``env``: interpreters ``chatterbox``, ``annotate``, ``parakeet``, ``qwen3_tts``, ``parler`` and ``ffmpeg``;
    ``writer`` / ``tts``: ``SftWriteDialogues`` / ``ChatterboxInferenceV2`` settings."""
    from ..jobs.hf import HfMergeShards
    from .tts import ChatterboxInferenceV2

    scenarios, langs = list(scenarios or sd.SCENARIOS), list(langs or sd.LANGS)
    assert set(scenarios) <= set(scenario_weights) and set(langs) <= set(lang_weights), "a cell without a weight"
    pool = build_voice_pool(personas=rows[("personas", "en")], env=env, langs=langs, **voices)
    out = SftCorpus(pool=pool)
    max_err = dict(tts.get("max_err") or {"*": 0.1})
    seeds = {}  # (scenario, lang, k) -> SftSeeds
    for sc in scenarios:
        for lg in langs:
            src = {s: rows[(s, lg)] for s in ("wikipedia", "wildchat") if (s, lg) in rows}
            src["personas"] = rows[("personas", "en")]
            for start, n in shard_ranges((cell_sizes or {}).get((sc, lg), cell_seeds), shard_seeds):
                seeds[(sc, lg, start // shard_seeds)] = sd.SftSeeds(scenario=sc, lang=lg, n=n, start=start, seed=seed,
                                                                    sources=src)  # fmt: skip
    shard_jobs = {}
    for grp in job_groups(list(seeds), group, shards_per_job):
        write = sd.SftWriteDialogues(seeds=[seeds[s].out_seeds for s in grp], per_input=True, **writer)
        casts = [SftCastVoices(written=write.out_hfs[i], pool=pool.out_index, parler_share=voices["parler_share"])
                 for i in range(len(grp))]  # fmt: skip
        speech = ChatterboxInferenceV2(venv_python_path=env["chatterbox"], annotate_python_path=env["annotate"],
                                       parakeet_python_path=env["parakeet"], qwen3_tts_python_path=env["qwen3_tts"],
                                       ffmpeg_path=env["ffmpeg"], in_hfs=[c.out_hf for c in casts],
                                       in_prefixes=[f"{sc}_{lg}_{k:03d}_" for sc, lg, k in grp],
                                       speaker_dir=pool.out_dir, keep_columns=TTS_KEEP, **tts)  # fmt: skip
        for i, (sc, lg, k) in enumerate(grp):
            stats = SftShardStats(cell={"scenario": sc, "lang": lg, "shard": k}, seeds=seeds[(sc, lg, k)].out_seeds,
                                  write_stats=write.out_stats_parts[i], cast=casts[i].out_hf, tts=speech.out_hfs[i],
                                  max_err=max_err)  # fmt: skip
            shard_jobs[(sc, lg, k)] = {"seeds": seeds[(sc, lg, k)], "write": write, "cast": casts[i], "tts": speech,
                                       "part": i, "stats": stats}  # fmt: skip
    for (sc, lg, k), j in sorted(shard_jobs.items()):
        out.shards.setdefault((sc, lg), []).append(j)
    for c, js in out.shards.items():
        out.cells[c] = HfMergeShards(shard_paths=[j["tts"].out_hfs[j["part"]] for j in js]).out_hf
    w = mix_weights(out.cells, scenario_weights, lang_weights)
    out.train_data = mix_train_data(out.cells, scenario_weights, lang_weights)
    out.report = SftDataReport(
        name=name,
        shards=[{"scenario": sc, "lang": lg, "shard": k, "stats": j["stats"].out_stats}
                for (sc, lg), js in sorted(out.shards.items()) for k, j in enumerate(js)],
        mix=[{"scenario": c[0], "lang": c[1], "weight": v} for c, v in sorted(w.items())],
        scenario_weights=dict(scenario_weights), lang_weights=dict(lang_weights), pool=pool.out_index,
        parler_share=voices["parler_share"])  # fmt: skip
    return out
