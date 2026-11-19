"""Guard: the SFT corpus pipeline's structure (``sft_pipeline``) and its checks, against the real module:

  * cells are independent: bumping one scenario's version, growing one cell or adding a scenario re-hashes only those
    cells' jobs (seeds, writer, cast, TTS, stats, merge); a mix weight moves no cell job, only the report and the
    training mix; a new tag prompt (``TAG_VERSION``) moves every cell that draws requests, and only those;
  * shards are deterministic: shard ``k`` of a 32-seed cell is the same job and the same seeds as shard ``k`` of a
    64-seed one (``shard_ranges``, ``make_seeds(start=)``);
  * the mix weights apply: a cell's weight is its scenario's x its language's, normalised; zero drops the cell; the
    report refuses a mix that is not that;
  * voices are drawn per dialogue (the same voices in any row order), Parler voices go to English users at the set
    share and nowhere else, never one voice for both speakers;
  * grouping shards into GPU jobs (``job_groups``) changes no shard's seeds; the full-run setting (``"cell"``) never
    puts two cells in one job; a grouped writer hands every shard exactly its own seeds (``write_groups``) and a
    grouped TTS gives every shard the rows and ids it would have alone (``concat_group_inputs`` /
    ``split_group_output``, with a stand-in for the GPU worker);
  * a language's own sources (German: LMArena requests, Nemotron-Personas-Germany) move exactly that language's cells
    and tag job, not the voice pool; its tag job reads every request file, its seeds its own personas;
  * the stats catch oddities: a silent turn, unequal or empty tracks, a dialogue of another cell, lost ids, a dialogue
    repeated across shards, a Parler voice outside English.

Each block is also run against a mutation that must make it fail (``MUTANTS``).

Run from the setup root (~10 s):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_sft_pipeline.py
"""

import json
import math
import os
import random
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import sft_dialogues as sd  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import sft_pipeline as sp  # noqa: E402

SC_W = {sc: 1.0 for sc in sd.SCENARIOS}
LG_W = {lg: 1.0 for lg in sd.LANGS}


def _rows():
    return {(s, lg): tk.Path(f"/tmp/sft_guard/{s}_{lg}.jsonl") for s in ("wikipedia", "wildchat") for lg in sd.LANGS} | {
        ("personas", "en"): tk.Path("/tmp/sft_guard/personas_en.jsonl")}  # fmt: skip


ENV = {
    k: tk.Path(f"/tmp/sft_guard/{k}") for k in ("chatterbox", "annotate", "parakeet", "qwen3_tts", "parler", "ffmpeg")
}
VOICES = dict(n_assistant=4, n_user=6, n_parler=3, parler_share=0.3)


def _build(**over):
    kw = dict(name="guard", rows=_rows(), env=ENV, cell_seeds=32, shard_seeds=16, seed=100,
              writer=dict(model="m", review=1), tts=dict(tts_model="qwen3", code_switch="segments"),
              voices=VOICES, scenario_weights=SC_W, lang_weights=LG_W, tagger=dict(model="m"),
              fallback={"de": "en"})  # fmt: skip
    kw.update(over)
    return sp.build_sft_corpus(**kw)


def _cell_ids(corpus) -> dict:
    """(scenario, lang) -> the ids of every job of the cell (per shard, in order) and its merge."""
    out = {}
    for c, js in corpus.shards.items():
        out[c] = [tuple(j[k]._sis_id() for k in ("seeds", "write", "cast", "tts", "stats")) + (j["part"],) for j in js]
        out[c].append(corpus.cells[c].creator._sis_id())
    return out


def _seed_ids(corpus) -> dict:
    return {(c, k): j["seeds"]._sis_id() for c, js in corpus.shards.items() for k, j in enumerate(js)}


def _cells_per_job(corpus) -> dict:
    """GPU job id (writer, TTS) -> the cells whose shards it processes."""
    out: dict = {}
    for c, js in corpus.shards.items():
        for j in js:
            for k in ("write", "tts"):
                out.setdefault(j[k]._sis_id(), set()).add(c)
    return out


def check_grouping() -> list[str]:
    bad = []
    cell = _build()
    lang = _build(group="language")
    pair = _build(shards_per_job=2)
    if _seed_ids(lang) != _seed_ids(cell) or _seed_ids(pair) != _seed_ids(cell):
        bad.append("grouping changed a shard's seeds job")
    for name, c in (("cell", cell), ("cell, 2 per job", pair)):
        mixed = {j: cs for j, cs in _cells_per_job(c).items() if len(cs) > 1}
        if mixed:
            bad.append(f"{name}: a GPU job holds several cells: {sorted(mixed.values(), key=str)[:2]}")
    n = len({j for j, cs in _cells_per_job(lang).items() if len(cs) == len(sd.SCENARIOS)})
    if n != 2 * len(sd.LANGS):  # one writer + one TTS job per language
        bad.append(f"language grouping: {n} GPU jobs hold a whole language, want {2 * len(sd.LANGS)}")
    # a grouped writer: each shard written from exactly its own seeds
    lists = [[{"id": f"s{i}_{k}"} for k in range(3 + i)] for i in range(3)]

    def fake(seeds, workers):
        return [{"id": x["id"], "text": x["id"][::-1]} for x in seeds], {"n": len(seeds)}

    together = sd.write_groups(lists, fake, 64)
    alone = [sd.write_groups([x], fake, 64)[0] for x in lists]
    if together != alone:
        bad.append("a grouped writer's shard output differs from the shard written alone")
    bad += _check_tts_groups()
    return bad


def _check_tts_groups() -> list[str]:
    """``ChatterboxInferenceV2(in_hfs=...)``'s concat and split around a stand-in worker (the worker's contract: rows
    in input order, ids ``dialogue_<i>``, the keep columns passed through as strings)."""
    import shutil
    import tempfile

    from datasets import Dataset, load_from_disk

    from i6_experiments.users.dorian_koch.speech_llm import tts

    tmp = tempfile.mkdtemp(prefix="sft_guard_")
    try:
        parts = []
        for i in range(3):
            parts.append(f"{tmp}/in{i}")
            Dataset.from_list([{"id": f"help_en_{10 * i + j}", "sft_id": f"help_en_{10 * i + j}", "dialogue": f"d{i}{j}"}
                               for j in range(2 + i)]).save_to_disk(parts[-1])  # fmt: skip

        def run(idx, tag):
            gin = tts.concat_group_inputs([parts[i] for i in idx], f"{tmp}/{tag}_in")
            rows = [{"id": f"dialogue_{n}", "turns": f"spoken {r['dialogue']}", "sft_id": r["sft_id"],
                     tts.GROUP_PART: r[tts.GROUP_PART]} for n, r in enumerate(load_from_disk(gin))]  # fmt: skip
            Dataset.from_list(rows).save_to_disk(f"{tmp}/{tag}_out")
            outs = [f"{tmp}/{tag}_o{i}" for i in idx]
            tts.split_group_output(f"{tmp}/{tag}_out", [f"p{i}_" for i in idx], outs)
            return {i: [dict(r) for r in load_from_disk(o)] for i, o in zip(idx, outs)}

        together = run([0, 1, 2], "g")
        alone = {}
        for i in range(3):
            alone.update(run([i], f"a{i}"))
        bad = []
        if together != alone:
            bad.append("a grouped TTS shard differs from the shard spoken alone (rows or ids)")
        if [r["id"] for r in together[2]] != [f"p2_dialogue_{k}" for k in range(4)]:
            bad.append(f"grouped TTS ids {[r['id'] for r in together[2]]}")
        if [r["sft_id"] for r in together[1]] != ["help_en_10", "help_en_11", "help_en_12"]:
            bad.append("a grouped TTS shard holds another shard's rows")
        return bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_independence() -> list[str]:
    bad = []
    base = _build()
    a = _cell_ids(base)
    if len(a) != len(sd.SCENARIOS) * len(sd.LANGS) or any(len(v) != 3 for v in a.values()):
        bad.append(f"expected 32 cells of 2 shards + merge, got {len(a)} cells")
    old = dict(sd.SCENARIO_VERSIONS)
    try:
        sd.SCENARIO_VERSIONS["tutor"] += 1
        b = _cell_ids(_build())
    finally:
        sd.SCENARIO_VERSIONS.clear()
        sd.SCENARIO_VERSIONS.update(old)
    moved = sorted(c for c in a if a[c] != b[c])
    if moved != sorted(c for c in a if c[0] == "tutor"):
        bad.append(f"a tutor version bump moved {moved}")
    tv = sp.TAG_VERSION
    try:
        sp.TAG_VERSION += 1
        t = _cell_ids(_build())
    finally:
        sp.TAG_VERSION = tv
    moved = sorted(c for c in a if a[c] != t[c])
    if moved != sorted(c for c in a if c[0] in sd.REQUEST_SCENARIOS):
        bad.append(f"a TAG_VERSION bump moved {len(moved)} cells, want every request-drawing cell (not grounded)")
    g = _cell_ids(_build(cell_sizes={("tutor", "de"): 64}))
    moved = sorted(c for c in a if a[c] != g[c])
    if moved != [("tutor", "de")]:
        bad.append(f"growing tutor/de moved {moved}")
    elif g[("tutor", "de")][:2] != a[("tutor", "de")][:2] or len(g[("tutor", "de")]) != 5:
        bad.append("growing tutor/de 32 -> 64 changed its first two shards (or did not add two)")
    w = _build(scenario_weights={**SC_W, "tutor": 3.0})
    if _cell_ids(w) != a:
        bad.append("a mix weight moved a cell job")
    if w.report._sis_id() == base.report._sis_id():
        bad.append("a mix weight did not reach the report")
    sub = _cell_ids(_build(scenarios=["help", "service"]))
    if any(sub[c] != a[c] for c in sub):
        bad.append("building fewer scenarios moved the remaining cells")
    return bad


def check_sources() -> list[str]:
    bad = []
    a = _build()
    extra = {(s_, "de"): tk.Path(f"/tmp/sft_guard/{s_}_de.jsonl") for s_ in ("lmarena", "personas")}
    b = _build(rows={**_rows(), **extra})
    ca, cb = _cell_ids(a), _cell_ids(b)
    moved = sorted(c for c in ca if ca[c] != cb[c])
    if moved != sorted(c for c in ca if c[1] == "de"):
        bad.append(f"German sources moved {len(moved)} cells, want the {len(sd.SCENARIOS)} German ones: {moved[:4]}")
    tags = sorted(lg for lg in a.tags if a.tags[lg]._sis_id() != b.tags[lg]._sis_id())
    if tags != ["de"]:
        bad.append(f"German sources moved the tag jobs of {tags}")
    want = [_rows()[("wildchat", "de")], extra[("lmarena", "de")]]
    if b.tags["de"].rows != want or b.tags["en"].rows != _rows()[("wildchat", "en")]:
        bad.append("the German tag job does not read WildChat + LMArena (or English reads more than WildChat)")
    if a.pool._sis_id() != b.pool._sis_id():
        bad.append("German sources moved the voice pool")
    for (sc, lg), js in b.shards.items():
        per = js[0]["seeds"].sources.get("personas")
        if per != (extra[("personas", "de")] if lg == "de" else _rows()[("personas", "en")]):
            bad.append(f"{sc}/{lg} seeds read the personas {per}")
            break
    return bad


def _fake_rows():
    rng = random.Random(0)
    per = [
        {"id": f"p{i}", "text": f"A {20 + i % 50}-year-old teacher who likes walking and cooking {i}."}
        for i in range(50)
    ]
    flags = {"needs_document": False, "needs_business_facts": False, "unsafe": False, "not_speakable": False}
    out = {"personas": per}
    for lg in sd.LANGS:
        out[f"wikipedia_{lg}"] = [{"id": f"w{i}", "title": f"T{i}", "text": f"passage {i} " * 30} for i in range(50)]
        out[f"tagged_{lg}"] = [{"id": f"c{lg}{i}", "lang": lg, "text": f"how do I cook rice number {rng.randrange(999)}",
                                "gist": f"cooking rice {i}", "flags": flags,
                                "weights": {k: rng.choice([0.0, 0.5, 1.0]) for k in sd.SCENARIOS}} for i in range(50)]  # fmt: skip
    return out


def check_shards() -> list[str]:
    bad = []
    if sp.shard_ranges(64, 16)[:2] != sp.shard_ranges(32, 16) or sp.shard_ranges(40, 16)[-1] != (32, 8):
        bad.append(f"shard_ranges {sp.shard_ranges(64, 16)} / {sp.shard_ranges(40, 16)}")
    fr = _fake_rows()
    for sc in sd.SCENARIOS:
        for lg in ("en", "zh"):
            rows = {"personas": fr["personas"], "wikipedia": fr[f"wikipedia_{lg}"], "tagged": fr[f"tagged_{lg}"]}
            whole = sd.make_seeds(sc, lg, 12, 100, rows)
            parts = sd.make_seeds(sc, lg, 6, 100, rows) + sd.make_seeds(sc, lg, 6, 100, rows, start=6)
            if json.dumps(whole, sort_keys=True) != json.dumps(parts, sort_keys=True):
                bad.append(f"{sc}/{lg}: seeds of two shards differ from the whole cell's")
                break
            if len({s["id"] for s in whole}) != 12:
                bad.append(f"{sc}/{lg}: seed ids not unique")
    return bad


def check_mix() -> list[str]:
    bad = []
    cells = [(sc, lg) for sc in sd.SCENARIOS for lg in sd.LANGS]
    w = sp.mix_weights(cells, SC_W, LG_W)
    if any(abs(v - 1 / 32) > 1e-12 for v in w.values()) or len(w) != 32:
        bad.append(f"equal weights do not give 1/32 each: {sorted(set(w.values()))}")
    w = sp.mix_weights(cells, {**SC_W, "tutor": 3.0, "help": 0.0}, {**LG_W, "zh": 2.0})
    z = (6 + 3) * (3 + 2)  # scenario sum x language sum
    if ("help", "en") in w or abs(w[("tutor", "zh")] - 6 / z) > 1e-12 or abs(w[("chitchat", "en")] - 1 / z) > 1e-12:
        bad.append(
            f"weights not scenario x language: tutor/zh {w.get(('tutor', 'zh'))}, help/en {w.get(('help', 'en'))}"
        )
    paths = {c: tk.Path(f"/tmp/{c[0]}_{c[1]}") for c in cells}
    td = sp.mix_train_data(paths, {**SC_W, "tutor": 3.0}, LG_W)
    if len(td) != 32 or any(p is not paths[c] for (p, _), c in zip(td, sorted(paths))):
        bad.append("mix_train_data does not pair each cell's own dataset with its weight")
    # the report refuses a mix that is not the weights'
    st = _shard("tutor", "en", 0, [_stats_row("tutor_en_0")])
    mix = [{"scenario": "tutor", "lang": "en", "weight": 0.5}, {"scenario": "help", "lang": "en", "weight": 0.5}]
    _, problems = sp.corpus_report("g", [st, _shard("help", "en", 0, [_stats_row("help_en_0")])], mix,
                                   {**SC_W, "tutor": 3.0}, LG_W, {}, 0.3, sp.SFT_BOUNDS)  # fmt: skip
    if not any("mix weights" in p for p in problems):
        bad.append("the report took a mix that is not the scenario x language weights")
    return bad


INDEX = {f"{lg}/{r}/{g}": {"vd": [f"vd_{lg}_{r}_{g}_{k}" for k in range(5)]} for lg in sd.LANGS
         for r in ("assistant", "user") for g in ("female", "male")}  # fmt: skip
for _g in ("female", "male"):
    INDEX[f"en/user/{_g}"]["parler"] = [f"parler_en_user_{_g}_{k}" for k in range(3)]


def _written(n):
    out = []
    for i in range(n):
        lg = list(sd.LANGS)[i % 4]
        g = ("female", "male")[i % 2]
        d = {
            "lang": lg,
            "turns": [],
            "voices": {"assistant": f"{lg}/assistant/{g}/rng_v", "user": f"{lg}/user/{g}/rng_v"},
        }
        out.append({"id": f"help_{lg}_{i}", "scenario": "help", "lang": lg, "dialogue": json.dumps(d)})
    return out


def check_cast() -> list[str]:
    bad = []
    rows = _written(4000)
    a = {r["sft_id"]: r["dialogue"] for r in sp.cast_voices(rows, INDEX, 0.3)}
    b = {r["sft_id"]: r["dialogue"] for r in sp.cast_voices(rows[::-1], INDEX, 0.3)}
    if a != b:
        bad.append("voice draws depend on the row order")
    users = [json.loads(d)["voices"]["user"] for d in a.values()]
    en = [u for u in users if u.startswith("en/")]
    k = sum("/parler_" in u for u in en)
    if abs(k - 0.3 * len(en)) > 3 * math.sqrt(len(en) * 0.21):
        bad.append(f"Parler share of English users {k}/{len(en)}, want 0.3")
    if any("/parler_" in u for u in users if not u.startswith("en/")):
        bad.append("a Parler voice outside English")
    if any(len(set(json.loads(d)["voices"].values())) < 2 for d in a.values()):
        bad.append("one voice for both speakers")
    return bad


def _turns(spk, times, langs=None):
    return {"speaker": spk, "start_time": [t[0] for t in times], "end_time": [t[1] for t in times],
            "text": [f"turn {i} words" for i in range(len(spk))], "lang": langs or ["en"] * len(spk),
            "interrupted": [False] * len(spk)}  # fmt: skip


def _tts_row(sid, silent_turn=None, short_user=False, text_seed=0, text_id=None):
    sr = 16000
    rng = np.random.default_rng(text_seed)
    tr = {sp_: np.zeros(4 * sr, np.float32) for sp_ in ("user", "assistant")}
    times = [(0.0, 1.0), (1.2, 2.2), (2.4, 3.4)]
    spk = ["user", "assistant", "user"]
    for k, (a, b) in enumerate(times):
        if k != silent_turn:
            tr[spk[k]][int(a * sr) : int(b * sr)] = 0.07 * rng.standard_normal(int((b - a) * sr))
    if short_user:
        tr["user"] = tr["user"][: sr * 2]
    turns = _turns(spk, times)
    turns["text"] = [f"{t} {text_id or sid}" for t in turns["text"]]
    segs = {"turn": [0, 1, 2], "lang": ["en"] * 3, "text": turns["text"], "start": [t[0] for t in times],
            "end": [t[1] for t in times], "err": [0.0, 0.02, 0.5], "attempts": [1, 1, 3]}  # fmt: skip
    return {"sft_id": sid, "policy": "same", "turns": turns, "segments": segs, "tracks": tr, "sr": sr,
            "voices": {"speaker": ["user", "assistant"], "voice": ["en/user/female/vd_1", "en/assistant/male/vd_2"]}}  # fmt: skip


def _stats_row(sid, **kw):
    return sp.dialogue_stats(_tts_row(sid, **kw), {"*": 0.1})[0]


def _shard(sc, lg, k, dialogues, seeds=None):
    return {"cell": {"scenario": sc, "lang": lg, "shard": k}, "seeds": seeds or len(dialogues), "kept": len(dialogues),
            "spoken": len(dialogues), "dialogues": dialogues}  # fmt: skip


def check_stats() -> list[str]:
    bad = []
    cell = {"scenario": "help", "lang": "en", "shard": 0}
    ok_rows = [_tts_row(f"help_en_{i}", text_seed=i) for i in range(3)]
    cast = [{"id": f"help_en_{i}", "scenario": "help", "lang": "en"} for i in range(3)]
    ids = [f"help_en_{i}" for i in range(4)]
    st, pr = sp.shard_stats(cell, ids, {}, cast, ok_rows, {"*": 0.1})
    if pr or st["dialogues"][0]["over"] != 1 or st["dialogues"][0]["redone"] != 1 or st["dialogues"][0]["silent_turns"]:
        bad.append(f"a clean shard: problems {pr}, first dialogue {st['dialogues'][0]}")
    _, pr = sp.shard_stats(cell, ids, {}, cast, [ok_rows[0], ok_rows[1], _tts_row("help_en_2", short_user=True)],
                           {"*": 0.1})  # fmt: skip
    if not any("track lengths" in p for p in pr):
        bad.append("unequal tracks not caught")
    _, pr = sp.shard_stats(cell, ids, {}, cast[:2] + [{"id": "help_de_2", "scenario": "help", "lang": "de"}],
                           ok_rows, {"*": 0.1})  # fmt: skip
    if not any("another cell" in p for p in pr) or not any("ids differ" in p for p in pr):
        bad.append(f"a dialogue of another cell / lost ids not caught: {pr}")
    empty = _tts_row("help_en_2")
    empty["tracks"] = {"user": np.zeros(0, np.float32), "assistant": np.zeros(0, np.float32)}
    _, pr = sp.shard_stats(cell, ids, {}, cast, ok_rows[:2] + [empty], {"*": 0.1})
    if not any("under 1 s" in p for p in pr):
        bad.append("an empty track not caught")
    sil = _stats_row("help_en_9", silent_turn=1)
    if sil["silent_turns"] != 1:
        bad.append(f"a silent turn counted {sil['silent_turns']}")
    mix = [{"scenario": "help", "lang": "en", "weight": 1.0}]
    rep, pr = sp.corpus_report("g", [_shard("help", "en", 0, [_stats_row("a"), sil])], mix, SC_W, LG_W, {}, 0.3,
                               sp.SFT_BOUNDS)  # fmt: skip
    if not any(f["check"] == "silent_turns" for f in rep["flags"]):
        bad.append("the report did not flag a silent turn")
    d1 = _stats_row("help_en_1", text_seed=1)
    d2 = _stats_row("help_en_17", text_seed=1, text_id="help_en_1")  # the same text in another shard
    _, pr = sp.corpus_report("g", [_shard("help", "en", 0, [d1]), _shard("help", "en", 1, [d2])], mix, SC_W, LG_W,
                             {}, 0.3, sp.SFT_BOUNDS)  # fmt: skip
    if not any("more than once" in p for p in pr):
        bad.append("a dialogue repeated across shards not caught")
    de = dict(_stats_row("help_de_1"), voices={"user": "de/user/male/parler_x", "assistant": "de/assistant/male/vd_1"})
    _, pr = sp.corpus_report("g", [_shard("help", "de", 0, [de])], [{"scenario": "help", "lang": "de", "weight": 1.0}],
                             SC_W, LG_W, {}, 0.3, sp.SFT_BOUNDS)  # fmt: skip
    if not any("Parler voice outside English" in p for p in pr):
        bad.append("a Parler voice outside English not caught")
    low = _shard("help", "en", 0, [_stats_row("a")], seeds=10)
    rep, _ = sp.corpus_report("g", [low], mix, SC_W, LG_W, {}, 0.3, sp.SFT_BOUNDS)
    if not any(f["check"] == "kept_rate" for f in rep["flags"]):
        bad.append("a kept rate of 0.1 not flagged")
    return bad


class _Patch:
    def __init__(self, obj, **over):
        self.obj, self.over, self.old = obj, over, {}

    def __enter__(self):
        for k, v in self.over.items():
            self.old[k] = getattr(self.obj, k)
            setattr(self.obj, k, v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            setattr(self.obj, k, v)


_make_seeds = sd.make_seeds
_cast = sp.cast_voices


def _global_version_hash(cls, parsed_args):  # mutant: every scenario's version in every seed job
    return sd.Job.hash.__func__(cls, {**parsed_args, "v": sorted(sd.SCENARIO_VERSIONS.items())})


def _shared_stream_cast(rows, index, share):  # mutant: one random stream over the shard
    rng = random.Random(0)
    out = []
    for r in rows:
        d = json.loads(r["dialogue"])
        v = {}
        for role in sorted(d["voices"]):
            key = "/".join(d["voices"][role].split("/")[:3])
            pools = index[key]
            src = "parler" if pools.get("parler") and rng.random() < share else "vd"
            v[role] = f"{key}/{rng.choice(pools[src])}"
        out.append({**r, "sft_id": r["id"], "dialogue": json.dumps({**d, "voices": v})})
    return out


_job_groups = sp.job_groups


def _split_ids_across(path, prefixes, outs):  # mutant: ids counted across the whole group
    from datasets import load_from_disk

    from i6_experiments.users.dorian_koch.speech_llm import tts

    ds = load_from_disk(path)
    part = [int(x) for x in ds[tts.GROUP_PART]]
    for i, (pre, out) in enumerate(zip(prefixes, outs)):
        idx = [j for j, p in enumerate(part) if p == i]
        sub = ds.select(idx).remove_columns(["id", tts.GROUP_PART])
        sub.add_column("id", [f"{pre}dialogue_{j}" for j in idx]).save_to_disk(out)


MUTANTS = {
    "seeds keyed by the shard's own index": (check_shards, sd,
                                             dict(make_seeds=lambda sc, lg, n, seed, rows, start=0: _make_seeds(sc, lg, n, seed, rows))),
    "one shard per cell": (check_independence, sp, dict(shard_ranges=lambda n, size: [(0, n)])),
    "one version for all scenarios": (check_independence, sd.SftSeeds, dict(hash=classmethod(_global_version_hash))),
    "mix ignores the weights": (check_mix, sp, dict(mix_weights=lambda cells, a, b: {c: 1 / len(list(cells)) for c in cells})),
    "voices from one stream per shard": (check_cast, sp, dict(cast_voices=_shared_stream_cast)),
    "cell grouping by language": (check_grouping, sp,
                                  dict(job_groups=lambda sh, group="cell", per_job=1: _job_groups(sh, "language", per_job))),
    "every shard written from the group's seeds": (check_grouping, sd,
                                                   dict(write_groups=lambda lists, fn, w: [fn(sum(lists, []), w) for _ in lists])),
    "TTS ids counted across the group": (check_grouping, __import__("i6_experiments.users.dorian_koch.speech_llm.tts",
                                                                    fromlist=["tts"]), dict(split_group_output=_split_ids_across)),
    "German users on the English personas": (check_sources, sp, dict(persona_rows=lambda rows, lg: rows[("personas", "en")])),
    "German pool without LMArena": (check_sources, sp, dict(request_rows=lambda rows, lg: rows[("wildchat", lg)])),
    "silence never counted": (check_stats, sp, dict(SILENT_DBFS=-400.0)),
    "no duplicate detection": (check_stats, sp, dict(_digest=lambda x: os.urandom(8).hex())),
}  # fmt: skip

if __name__ == "__main__":
    fails = (
        check_independence()
        + check_sources()
        + check_grouping()
        + check_shards()
        + check_mix()
        + check_cast()
        + check_stats()
    )
    for f in fails:
        print("FAIL", f)
    for name, (fn, obj, over) in MUTANTS.items():
        with _Patch(obj, **over):
            caught = bool(fn())
        if not caught:
            fails.append(f"mutant {name!r} was not caught")
            print("FAIL", fails[-1])
        else:
            print(f"ok: mutant {name!r} caught")
    print(f"check_sft_pipeline: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
