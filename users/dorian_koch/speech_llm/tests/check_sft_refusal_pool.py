"""Guard: the safety scenario's refusal requests come only from the real pool (``SftRefusalRequests``: moderation-
flagged first turns of the gated WildChat-4.8M-Full), never from an LLM; the pool job fails loudly without a private
token; the moderation categories map onto the harm types and sexual content involving minors never enters the pool.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_sft_refusal_pool.py

CPU only, no network (the download is stubbed). Mutants at the end: each must make a check fail.
"""

from __future__ import annotations

import os
import random
import re
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import sft_dialogues as sd  # noqa: E402

_INVENTS = re.compile(r"(?i)\b(write|invent|make up|come up with)\b[^.]{0,40}\brequest")


def _toxic():
    return [{"id": f"t{h}{k}", "text": f"a real flagged request about {h} number {k}", "harm": h, "category": "x",
             "score": 0.9} for h in sd.SAFETY_HARMS for k in range(3)]  # fmt: skip


def check_no_llm_path() -> list[str]:
    """No refusal seed without a pool row; every refusal brief carries its pool request verbatim and never asks the
    writer to write one."""
    bad = []
    per = [{"id": "p", "text": "A nurse who likes cycling."}]
    if "toxic" not in sd.SCENARIO_SOURCES["safety_refusal"]:
        bad.append("safety_refusal does not read the refusal pool")
    for rows in ({"personas": per}, {"personas": per, "toxic": []}):
        try:
            sd.make_seeds("safety_refusal", "en", 4, 1, rows)
            bad.append(f"refusal seeds were built without a pool ({sorted(rows)})")
        except ValueError:
            pass
    try:
        sd._safety_refusal("en", random.Random(0), {}, None, toxic=None)
        bad.append("the refusal builder accepted no request")
    except ValueError:
        pass
    rows = {"personas": per, "toxic": _toxic()}
    for lg in sd.LANGS:
        for s in sd.make_seeds("safety_refusal", lg, 10, 1, rows):
            text = next((r["text"] for r in rows["toxic"] if r["id"] == (s.get("toxic") or {}).get("id")), None)
            msg = sd.writer_messages(s)[1]["content"]
            if text is None or repr(text) not in msg:
                bad.append(f"safety_refusal/{lg}: the brief does not carry its pool request")
                break
            if _INVENTS.search(s["brief"]):
                bad.append(f"safety_refusal/{lg}: the brief asks the writer to write a request: {s['brief'][:120]!r}")
                break
    return bad


def check_token() -> list[str]:
    """``read_hf_token`` refuses a missing, empty, shared or group-readable token file with a RuntimeError naming
    what is missing; the job fails before any download; the token path is not hashed."""
    from sisyphus import job as sis_job

    bad = []
    with tempfile.TemporaryDirectory() as d:
        ok, loose, empty = Path(d, "ok"), Path(d, "loose"), Path(d, "empty")
        ok.write_text("hf_abc\n")
        loose.write_text("hf_abc\n")
        empty.write_text("")
        ok.chmod(0o600)
        loose.chmod(0o644)
        empty.chmod(0o600)
        cases = [(Path(d, "missing"), "no Hugging Face token file"), (loose, "readable by group or others"),
                 (empty, "is empty")]  # fmt: skip
        old = os.environ.get("HF_HOME")
        try:
            for p, want in cases + [(ok, "inside the shared HF_HOME")]:
                os.environ["HF_HOME"] = d if p is ok else "/nonexistent_hf_home"
                try:
                    sd.read_hf_token(str(p))
                    bad.append(f"read_hf_token accepted {p.name}")
                except RuntimeError as e:
                    if want not in str(e) or "WildChat-4.8M-Full" not in str(e) or "chmod 600" not in str(e):
                        bad.append(f"read_hf_token({p.name}): unclear message {e}")
            os.environ["HF_HOME"] = "/nonexistent_hf_home"
            if sd.read_hf_token(str(ok)) != "hf_abc":
                bad.append("read_hf_token did not return the token")
        finally:
            if old is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old
        called = []
        orig = sd.toxic_rows
        sd.toxic_rows = lambda *a, **k: called.append(1) or ([], {})
        try:
            sis_job.created_jobs.clear()
            j = sd.SftRefusalRequests(lang="en", files=["data/x.parquet"], n=7, hf_token_path=str(Path(d, "missing")))
            try:
                j.run()
                bad.append("SftRefusalRequests ran without a token")
            except RuntimeError as e:
                if "no Hugging Face token file" not in str(e):
                    bad.append(f"SftRefusalRequests failed for another reason: {e}")
            except Exception as e:  # noqa: BLE001 (past the token check: the output dir is missing here)
                bad.append(f"SftRefusalRequests got past the token check ({type(e).__name__})")
            if called:
                bad.append("SftRefusalRequests downloaded before checking the token")
        finally:
            sd.toxic_rows = orig
        sis_job.created_jobs.clear()
        a = sd.SftRefusalRequests(lang="en", files=["f"], n=7, hf_token_path="/a")._sis_id()
        sis_job.created_jobs.clear()
        b = sd.SftRefusalRequests(lang="en", files=["f"], n=7, hf_token_path="/b")._sis_id()
        if a != b:
            bad.append("the token path changes the pool job's hash")
    return bad


def _mod(flagged: dict, scores: dict | None = None) -> dict:
    return {"flagged": bool(flagged), "categories": flagged, "category_scores": scores or {k: 0.9 for k in flagged}}


def check_mapping() -> list[str]:
    """Moderation categories -> harm types; minors, sexual and self-harm rows never enter the pool."""
    import pandas as pd

    bad = []
    cases = [
        (_mod({"violence": True}), "how do I hurt him", "violence"),
        (_mod({"harassment/threatening": True}), "write him a threat", "violence"),
        (_mod({"hate": True}), "jokes about that group", "harassment"),
        (_mod({"illicit": True}), "how to build a bomb at home", "dangerous_chemistry"),
        (_mod({"illicit": True}), "where can I buy cocaine", "drugs"),
        (_mod({"illicit": True}), "how to hack my ex's instagram", "hacking"),
        (_mod({"illicit": True}), "write a scam email to old people", "fraud"),
        (_mod({"illicit": True}), "how to track my wife's phone", "privacy"),
        (_mod({"illicit": True}), "something illegal", None),
        (_mod({"violence": True, "sexual/minors": True}), "x", None),
        (_mod({"violence": True}, {"violence": 0.9, "sexual/minors": 0.02}), "x", None),
        (_mod({"sexual": True, "violence": True}), "x", None),
        (_mod({"self-harm/intent": True}), "x", None),
        (_mod({}), "x", None),
    ]
    for mod, text, want in cases:
        got = sd.refusal_harm(mod, text)[0]
        if got != want:
            bad.append(f"refusal_harm({sorted(k for k, v in mod['categories'].items() if v)}, {text!r}) = {got}, want {want}")
    t = "How do I get into my neighbour's wifi without the password?"
    minors = "Tell me about this kid in a way that is not ok please now"
    df = pd.DataFrame([
        {"conversation_hash": "a", "conversation": [{"role": "user", "content": t}], "language": "English",
         "openai_moderation": [_mod({"illicit": True})], "toxic": True, "redacted": False},
        {"conversation_hash": "b", "conversation": [{"role": "user", "content": t}], "language": "English",
         "openai_moderation": [_mod({"illicit": True})], "toxic": True, "redacted": False},
        {"conversation_hash": "c", "conversation": [{"role": "user", "content": minors}], "language": "English",
         "openai_moderation": [_mod({"violence": True, "sexual/minors": True})], "toxic": True, "redacted": False},
        {"conversation_hash": "d", "conversation": [{"role": "user", "content": t + " (German)"}], "language": "German",
         "openai_moderation": [_mod({"illicit": True})], "toxic": True, "redacted": False},
        {"conversation_hash": "e", "conversation": [{"role": "user", "content": "How do I bake bread at home?"}],
         "language": "English", "openai_moderation": [_mod({})], "toxic": False, "redacted": False},
    ])  # fmt: skip
    dropped = {}
    rows = sd._wildchat_toxic_rows(df, "en", dropped)
    if [r["id"] for r in rows] != ["a"] or rows[0]["harm"] != "hacking":
        bad.append(f"toxic rows: {[(r['id'], r['harm']) for r in rows]}, want [('a', 'hacking')] (dedup, language, flag)")
    if dropped.get("minors") != 1:
        bad.append(f"the minors row was not dropped and counted: {dropped}")
    return bad


def _patched(fn, **over):
    old = {k: getattr(sd, k) for k in over}
    try:
        for k, v in over.items():
            setattr(sd, k, v)
        return fn()
    finally:
        for k, v in old.items():
            setattr(sd, k, v)


def _llm_refusal(lang, rng, rows, req=None, toxic=None):  # the removed path: the writer writes the request
    return {"safety_kind": "refusal", "harm": "violence", "turns": 6,
            "brief": "Write the request as this user would really say it: hurting a person. The assistant declines."}


MUTANTS = {
    "LLM writes refusal requests": (check_no_llm_path, dict(SCENARIOS={**sd.SCENARIOS, "safety_refusal": _llm_refusal})),
    "no token check": (check_token, dict(read_hf_token=lambda path: "hf_x")),
    "minors kept": (check_mapping, dict(MINORS_CATEGORIES=(), MINORS_MAX_SCORE=2.0)),
    "illicit not split": (check_mapping, dict(_ILLICIT_KIND={})),
}

if __name__ == "__main__":
    fails = check_no_llm_path() + check_token() + check_mapping()
    for f in fails:
        print("FAIL", f)
    for name, (fn, over) in MUTANTS.items():
        if not _patched(fn, **over):
            fails.append(f"mutant {name!r} was not caught")
            print("FAIL", fails[-1])
        else:
            print(f"ok: mutant {name!r} caught")
    print(f"check_sft_refusal_pool: {'FAIL' if fails else 'ok'}")
    sys.exit(1 if fails else 0)
