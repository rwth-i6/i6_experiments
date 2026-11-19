"""Guard: the SFT seeds' request draw from tagged WildChat (German: + LMArena) requests (``sft_dialogues.cell_requests`` / ``draw_request``
/ ``make_seeds``, ``sft_pipeline.tag_row``), against the real module:

  * deterministic: the same pool gives the same seeds, in any row order of the pool; shards of a cell give the seeds
    of the whole cell;
  * each cell draws on its own: one cell's seeds do not depend on another cell being drawn or on its size;
  * ``scenario_smoothing``: at 0 a request rated 0 for the scenario is never drawn, at 0.01 at the expected rate;
  * a request with weight 0 for the scenario (smoothing 0), unsafe or unspeakable is never drawn; no request repeats in a cell
    before every eligible one was used; higher weights come earlier on average;
  * German continues into the fallback (English) pool after its own; service takes requests of weight >= 0.3 only and
    then a matter from the persona;
  * German's pool is WildChat + LMArena: both load into one pool with their ``source`` (an LMArena text WildChat has
    is dropped), keep it through tagging, and a German cell draws from both, each seed naming its request's source;
    LMArena rows are first user turns only (no later battle, other language, code or image; the answers never read);
    a German persona loses its name;
  * the retired content lists (service domains, chit-chat topics, long tasks, interests) are gone and no brief names
    them; the tagger clamps weights, marks ``_NSFW`` requests unsafe without asking, and a bumped ``TAG_VERSION``
    re-hashes the tag job.

Each block is also run against a mutation that must make it fail (``MUTANTS``).

Run from the setup root (~3 s):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_sft_seed_draw.py
"""

import json
import os
import random
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import sft_dialogues as sd  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import sft_pipeline as sp  # noqa: E402

FLAGS = {"needs_document": False, "needs_business_facts": False, "unsafe": False, "not_speakable": False}


def _req(i, lang, w, **flags):
    return {"id": f"{lang}{i}", "lang": lang, "text": f"How do I plan request number {i} in {lang}?",
            "gist": f"planning thing {i} ({lang})", "weights": {k: w.get(k, 0.5) for k in sd.SCENARIOS},
            "flags": {**FLAGS, **flags}}  # fmt: skip


def _pool(lang, n, seed=0):
    rng = random.Random(f"{lang}/{seed}")
    return [_req(i, lang, {k: rng.choice([0.0, 0.2, 0.5, 1.0]) for k in sd.SCENARIOS}) for i in range(n)]


def _rows(lang="en", n=40, fallback=None):
    per = [{"id": f"p{k}", "text": f"A {30 + k}-year-old nurse who likes cycling {k}."} for k in range(9)]
    wiki = [{"id": f"k{k}", "title": f"T{k}", "text": "A passage. " * 20} for k in range(5)]
    r = {"personas": per, "wikipedia": wiki, "tagged": _pool(lang, n)}
    if fallback:
        r["tagged_fallback"] = _pool(fallback, 30)
    return r


def check_determinism() -> list[str]:
    bad = []
    rows = _rows()
    for sc in sd.REQUEST_SCENARIOS:
        a = sd.make_seeds(sc, "en", 12, 7, rows)
        shuffled = dict(rows, tagged=list(reversed(rows["tagged"])))
        b = sd.make_seeds(sc, "en", 6, 7, shuffled) + sd.make_seeds(sc, "en", 6, 7, shuffled, start=6)
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            bad.append(f"{sc}: seeds depend on the pool's row order or on the shard split")
        c = sd.make_seeds(sc, "en", 12, 7, rows)
        if json.dumps(a) != json.dumps(c):
            bad.append(f"{sc}: two draws differ")
    # one cell's seeds do not depend on another cell's draw or size
    h0 = sd.make_seeds("help", "en", 8, 7, rows)
    sd.make_seeds("service", "en", 30, 7, rows)
    sd.make_seeds("longtask", "en", 3, 7, rows)
    if json.dumps(h0) != json.dumps(sd.make_seeds("help", "en", 8, 7, rows)):
        bad.append("help's seeds moved after drawing other cells")
    return bad


def check_draw() -> list[str]:
    bad = []
    pool = _pool("en", 60) + [_req(900, "en", {"help": 1.0}, unsafe=True), _req(901, "en", {"help": 1.0},
                                                                                 not_speakable=True)]  # fmt: skip
    rows = {"tagged": pool}
    seq = sd.cell_requests("help", "en", 3, rows, smoothing=0.0)
    elig = [r for r in pool if sd.request_eligible(r) and r["weights"]["help"] > 0]
    if sorted(r["id"] for r in seq) != sorted(r["id"] for r in elig):
        bad.append(f"the order is not every eligible request with weight > 0 once ({len(seq)} vs {len(elig)})")
    drawn = [sd.draw_request(seq, "help", k)["id"] for k in range(len(elig) + 5)]
    if len(set(drawn[: len(elig)])) != len(elig):
        bad.append("a request repeats before every eligible one was used")
    if drawn[len(elig) :] != drawn[:5]:
        bad.append("the second pass does not repeat the first pass's order")
    if any(r["weights"]["help"] == 0 or not sd.request_eligible(r) for r in seq):
        bad.append("a weight-0, unsafe or unspeakable request was drawn")
    # higher weights first on average (over many keys)
    hi, lo = 0, 0
    for key in range(200):
        s2 = sd.request_order(pool, "help", f"k{key}")
        top = s2[: len(s2) // 3]
        hi += sum(r["weights"]["help"] == 1.0 for r in top)
        lo += sum(r["weights"]["help"] == 0.2 for r in top)
    if hi <= 2 * lo:
        bad.append(f"weight 1.0 requests are not drawn early ({hi} vs {lo} of weight 0.2 in the first third)")
    return bad


def check_smoothing() -> list[str]:
    """``scenario_smoothing``: at 0 a request rated 0 for the scenario never comes; at 0.01 it comes first at the rate
    Efraimidis-Spirakis gives it, p'(zero) / (p'(zero) + p'(other)), over many cell keys."""
    bad = []
    zero = _req(0, "en", {k: float(k == "chitchat") for k in sd.SCENARIOS})
    one = _req(1, "en", {k: float(k == "help") for k in sd.SCENARIOS})
    if any(r["id"] == "en0" for r in sd.request_order([zero, one], "help", "k", smoothing=0.0)):
        bad.append("smoothing 0: a request rated 0 for the scenario was drawn")
    n = 20000
    hits = sum(sd.request_order([zero, one], "help", f"k{i}", smoothing=0.01)[0]["id"] == "en0" for i in range(n))
    pz, po = 0.01 / len(sd.SCENARIOS), 0.99 + 0.01 / len(sd.SCENARIOS)  # p'(help | zero), p'(help | one)
    exp = n * pz / (pz + po)
    if not 0.5 * exp <= hits <= 1.6 * exp:
        bad.append(f"smoothing 0.01: a zero-rated request came first {hits} of {n} times, expected ~{exp:.0f}")
    if abs(sum(sd.request_prob(one, k, 0.01) for k in sd.SCENARIOS) - 1) > 1e-9:
        bad.append("p'(scenario | request) does not sum to 1")
    return bad


def check_floors() -> list[str]:
    bad = []
    rows = _rows("de", 6, fallback="en")
    rows["tagged"][0]["weights"]["help"] = 0.0  # a German request rated 0 for help: drawn only by smoothing
    seq = sd.cell_requests("help", "de", 1, rows)
    rated = [(r["lang"], r["weights"]["help"] > 0) for r in seq]
    first_en = next((i for i, (lg, _) in enumerate(rated) if lg == "en"), len(rated))
    last_en_rated = max((i for i, x in enumerate(rated) if x == ("en", True)), default=-1)
    late_de = [i for i, (lg, _) in enumerate(rated) if lg == "de" and i > first_en]
    if (any(x == ("de", True) for x in rated[first_en:]) or any(i < last_en_rated for i in late_de)
            or ("de", True) not in rated or ("en", True) not in rated or not late_de):  # fmt: skip
        bad.append(f"German does not continue into the English pool after its own rated requests: {rated[:12]}")
    n_de = sum(1 for lg, ok in rated if lg == "de" and ok)
    seeds = sd.make_seeds("help", "de", n_de + 3, 1, rows)
    if [s["request"]["lang"] for s in seeds][n_de:] != ["en"] * 3:
        bad.append("German seeds past the own rated requests do not take English requests")
    # service: weight >= SERVICE_MIN_WEIGHT only, then the persona
    pool = [_req(i, "en", {"service": w}) for i, w in enumerate([1.0, 0.5, 0.3, 0.2, 0.0, 0.29])]
    seeds = sd.make_seeds("service", "en", 6, 1, {**_rows(), "tagged": pool})
    got = [s.get("request", {}).get("id") for s in seeds]
    if sorted(x for x in got if x) != ["en0", "en1", "en2"] or got[3:] != [None] * 3:
        bad.append(f"service draws {got}: want the three requests of weight >= 0.3, then persona matters")
    if not all(s["matter_from"] == "persona" and "WHO THE USER IS" in s["brief"] for s in seeds[3:]):
        bad.append("service past its suitable requests does not take the matter from the persona")
    return bad


def check_german_sources() -> list[str]:
    import tempfile

    import pandas as pd

    bad = []
    with tempfile.TemporaryDirectory() as d:
        wc, ar = Path(d, "wc.jsonl"), Path(d, "ar.jsonl")
        wc.write_text(
            "".join(json.dumps({"id": f"w{i}", "text": f"Wie plane ich Anfrage {i}?"}) + "\n" for i in range(20))
        )
        ar.write_text("".join(json.dumps({"id": f"lmarena:a{i}", "text": t, "source": "lmarena"}) + "\n" for i, t in
                              enumerate([f"Was koche ich heute, Variante {i}?" for i in range(20)] + ["Wie plane ich Anfrage 3?"])))  # fmt: skip
        loaded = sp.load_request_rows([str(wc), str(ar)])
    got = {src: sum(r["source"] == src for r in loaded) for src in ("wildchat", "lmarena")}
    if got != {"wildchat": 20, "lmarena": 20} or len(loaded) != 40:
        bad.append(f"the German pool loads {got} of {len(loaded)} rows, want 20 + 20 (one LMArena duplicate dropped)")
    rng = random.Random(5)
    tagged = [sp.tag_row(r, "de", {"gist": f"thing {r['id']}", "weights": {k: rng.choice([0.2, 0.5, 1.0]) for k in
                                   sd.SCENARIOS}, "flags": dict(FLAGS)}) for r in loaded]  # fmt: skip
    if [t["source"] for t in tagged] != [r["source"] for r in loaded]:
        bad.append("tagging loses the request's source")
    rows = {**_rows("de", 1, fallback="en"), "tagged": tagged}
    for sc in sd.REQUEST_SCENARIOS:
        src = {s_["request"].get("source") for s_ in sd.make_seeds(sc, "de", 16, 3, rows) if s_.get("request")}
        if src != {"wildchat", "lmarena"}:
            bad.append(f"{sc}: a German cell drew requests from {src}, want WildChat and LMArena")
    user = lambda t: {"role": "user", "content": [{"type": "text", "text": t, "image": None}]}  # noqa: E731
    ans = {"role": "assistant", "content": [{"type": "text", "text": "ANSWER TEXT"}]}
    first = "Was kann ich heute Abend mit Freunden kochen?"
    a140 = pd.DataFrame([
        {"id": "a", "language": "de", "is_code": False, "evaluation_order": 1, "conversation_a": [user(first), ans]},
        {"id": "b", "language": "de", "is_code": False, "evaluation_order": 2, "conversation_a": [user("Und was trinke ich dazu am besten?"), ans]},
        {"id": "c", "language": "en", "is_code": False, "evaluation_order": 1, "conversation_a": [user("What can I cook tonight with friends?"), ans]},
        {"id": "d", "language": "de", "is_code": True, "evaluation_order": 1, "conversation_a": [user("Schreib mir eine Funktion zum Sortieren."), ans]},
        {"id": "e", "language": "de", "is_code": False, "evaluation_order": 1, "conversation_a": [
            {"role": "user", "content": [{"type": "image", "text": None, "image": "x"}, {"type": "text", "text": "Was ist auf dem Bild zu sehen?"}]}, ans]},
    ])  # fmt: skip
    a100 = pd.DataFrame([{"question_id": "q", "language": "German", "is_code": False,
                          "conversation_a": [{"role": "user", "content": "Wie wird das Wetter morgen in Berlin?"}, ans]}])  # fmt: skip
    got = sd._lmarena_rows(a140, "de") + sd._lmarena_rows(a100, "de")
    if [(r["id"], r["source"]) for r in got] != [("lmarena:a", "lmarena"), ("lmarena:q", "lmarena")] or got[0][
        "text"
    ] != first:
        bad.append(f"LMArena rows {[(r['id'], r['text'][:30]) for r in got]}: want the first turns of a and q only")
    per = pd.DataFrame([
        {"uuid": "u", "persona": "Mario von Lojewski ist ein bescheidener Pragmatiker.", "age": 58,
         "occupation": "Gärtner", "employment_status": "Erwerbstätig", "kreis": "Kassel", "bundesland": "Hessen"},
        {"uuid": "v", "persona": "Ein ruhiger Handwerker namens Nico Schmelzer, der gern liest.", "age": 70,
         "occupation": "Nicht zutreffend", "employment_status": "Ruhestand", "kreis": "Cham", "bundesland": "Bayern"},
    ])  # fmt: skip
    t, u = (sd._persona_text_clean(r["text"]) for r in sd._personas_rows(per, "de"))
    if "Mario" in t or "Lojewski" in t or not t.startswith("Diese Person ist") or "Hessen" not in t:
        bad.append(f"a German persona reads {t!r}: want the name gone, the place kept")
    if "Nico" in u or "Schmelzer" in u or "Nicht zutreffend" in u or "im Ruhestand" not in u:
        bad.append(f"a German persona reads {u!r}: want the name gone and the status for no occupation")
    return bad


def check_retired() -> list[str]:
    bad = []
    for name in ("SERVICE_DOMAINS", "CHITCHAT_TOPICS", "LONG_TASKS", "INTERESTS", "_clean_request"):
        if hasattr(sd, name):
            bad.append(f"sft_dialogues.{name} still exists")
    old = ["a table reservation", "plans for the weekend", "planning a three-day city trip", "birdwatching",
           "a restaurant's reservation"]  # fmt: skip
    rows = _rows()
    for sc in sd.SCENARIOS:
        for s in sd.make_seeds(sc, "en", 10, 3, rows):
            if any(o in s["brief"] for o in old):
                bad.append(f"{sc}: a retired list entry in the brief")
            if sc != "grounded" and s.get("matter_from") != "persona":
                req = next(r for r in rows["tagged"] if r["id"] == s["request"]["id"])
                if req["gist"] not in s["brief"] and repr(req["text"]) not in s["brief"]:
                    bad.append(f"{sc}: neither the drawn request nor its gist is in the brief")
                    break
    return bad


def check_tagger() -> list[str]:
    from sisyphus import job as sis_job
    from sisyphus import tk

    bad = []
    r = sp.tag_row({"id": "x", "text": "hi"}, "en", {"gist": " a  gist ", "weights": {**{k: 0.5 for k in sd.SCENARIOS},
                   "tutor": 3.0, "help": -1}, "flags": dict(FLAGS)})  # fmt: skip
    if r["weights"]["tutor"] != 1.0 or r["weights"]["help"] != 0.0 or r["gist"] != "a gist":
        bad.append(f"tag_row does not clamp / tidy: {r['weights']['tutor']}, {r['weights']['help']}, {r['gist']!r}")
    u = sp.tag_row({"id": "y", "text": "x"}, "de", None)
    if not u["flags"]["unsafe"] or sd.request_eligible(u):
        bad.append("an _NSFW request is not unsafe")
    if "FOREIGN-LANGUAGE" not in sp.TAG_DEFS["tutor"] or "programming" not in sp.TAG_DEFS["tutor"]:
        bad.append("the tutor type is not defined as language learning only")
    sis_job.created_jobs.clear()
    a = sp.SftTagRequests(rows=tk.Path("/tmp/r"), lang="en")._sis_id()
    old = sp.TAG_VERSION
    try:
        sp.TAG_VERSION += 1
        sis_job.created_jobs.clear()
        b = sp.SftTagRequests(rows=tk.Path("/tmp/r"), lang="en")._sis_id()
    finally:
        sp.TAG_VERSION = old
    if a == b:
        bad.append("a TAG_VERSION bump did not re-hash SftTagRequests")
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


_order, _cell, _prob, _ref, _load = (
    sd.request_order,
    sd.cell_requests,
    sd.request_prob,
    sd.request_ref,
    sp.load_request_rows,
)


def _uniform_order(tagged, scenario, key, min_w=0.0, smoothing=0.0):  # mutant: weights ignored (zero ones included)
    rng = random.Random(key)
    out = [r for r in tagged if sd.request_eligible(r)]
    rng.shuffle(out)
    return out


def _with_replacement(seq, scenario, k):  # mutant: an independent draw per seed
    return random.Random(k).choice(seq) if seq else None


_shared = {"n": 0}


def _global_assignment(scenario, lang, seed, rows, smoothing=0.01):  # mutant: requests handed out across cells
    seq = _cell(scenario, lang, seed, rows, smoothing)
    _shared["n"] += 1
    return seq[_shared["n"] % max(1, len(seq)) :] + seq[: _shared["n"] % max(1, len(seq))]


MUTANTS = {
    "weights ignored": (check_draw, sd, dict(request_order=_uniform_order)),
    "draw with replacement": (check_draw, sd, dict(draw_request=_with_replacement)),
    "requests assigned across cells": (check_determinism, sd, dict(cell_requests=_global_assignment)),
    "row order matters": (check_determinism, sd, dict(request_order=lambda t, s, k, m=0.0, sm=0.0: [
        r for r in t if sd.request_eligible(r) and r["weights"][s] > 0 and r["weights"][s] >= m])),
    "German waits for its zero-rated tail": (check_floors, sd, dict(cell_requests=lambda sc, lg, seed, rows, sm=0.01: (
        sd.request_order(rows.get("tagged") or [], sc, f"draw/{sc}/{lg}/{seed}", 0.0, sm)
        + sd.request_order(rows.get("tagged_fallback") or [], sc, f"draw/{sc}/{lg}/{seed}/fallback", 0.0, sm)))),
    "no German fallback": (check_floors, sd, dict(cell_requests=lambda sc, lg, seed, rows, sm=0.01: _cell(
        sc, lg, seed, {k: v for k, v in rows.items() if k != "tagged_fallback"}, sm))),
    "service without a floor": (check_floors, sd, dict(SERVICE_MIN_WEIGHT=0.0)),
    "no smoothing": (check_smoothing, sd, dict(request_prob=lambda r, s, sm: _prob(r, s, 0.0))),
    "German pool loads its first file only": (check_german_sources, sp, dict(load_request_rows=lambda paths: _load(paths[:1]))),
    "seeds without the request's source": (check_german_sources, sd, dict(request_ref=lambda req: {
        k: v for k, v in _ref(req).items() if k != "source"})),
    "German persona keeps its name": (check_german_sources, sd, dict(_PERSONA_NAME_DE=__import__("re").compile("^$"))),
    "German persona keeps a name given in passing": (check_german_sources, sd, dict(_PERSONA_NAMENS_DE=__import__("re").compile("^$"))),
    "tutor = any tutoring": (check_tagger, sp, dict(TAG_DEFS={**sp.TAG_DEFS, "tutor": "a tutor in any subject"})),
}  # fmt: skip

if __name__ == "__main__":
    fails = (
        check_determinism()
        + check_draw()
        + check_smoothing()
        + check_floors()
        + check_german_sources()
        + check_retired()
        + check_tagger()
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
    print(f"check_sft_seed_draw: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
