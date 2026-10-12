"""Guard: every vLLM judge/generator in this package ENFORCES its JSON schema (2026-10-12, duplication audit item 6).

WHY. vLLM removed the ``guided_*`` request fields; 0.19.1 (setup .venv) and 0.30.0 (``vllm_venv_v030``) accept
``extra_body={"guided_json": ...}`` with HTTP 200 and decode UNCONSTRAINED (probe on login23-g-1, Qwen2.5-1.5B on
both versions and gpt-oss-20b on 0.30: 0/6 schema-valid replies with guided_json, byte-identical to sending nothing;
6/6 with ``response_format`` json_schema, gpt-oss reasoning unchanged). So the spj2 / coherence / persona-prompt /
safety / dialogue / scenario judges ran unconstrained until 2026-10-12. ``common.json_schema_format`` is the one
request form now.

  1. STATIC. No ``guided_json`` / ``guided_regex`` / ``guided_choice`` / ``guided_grammar`` /
     ``guided_decoding_backend`` / ``guided_whitespace_pattern`` string KEY in a dict literal anywhere in the i6
     ``speech_llm`` package or the full_duplex package (AST over every .py, tests excluded). CAN FAIL: the scanner
     flags a mutant source that sends ``extra_body={"guided_json": S}``.
  2. HELPER. ``json_schema_format(S, n)`` == ``{"response_format": {"type": "json_schema", "json_schema": {"name": n,
     "schema": S}}}`` (the training self-play judge's form, ``moshi_family/rl/selfplay_reward.py``, also checked
     by AST: it must pass ``response_format`` with type ``json_schema``).
  3. COHERENCE JOB. the real ``SelfPlayCoherenceJudge.run`` with a stub server/client: v2 (default) sends the
     schema as response_format and no extra_body; ``judge_version=1`` and ``guided_json=False`` send no constraint;
     summary records ``judge_version``. CAN FAIL: a mutant helper that returns the old extra_body form is caught.
  4. SAFETY LABELS. ``row_filters.label_safety`` sends ``_SAFETY_JSON_SCHEMA`` as response_format.
  5. HASH. ``judge_version=1`` hashes exactly like the pre-change job (``__sis_hash_exclude__``) for both judges, and
     the default (v2) hashes differently.
  (spj2's ``SelfPlayJudgeV2.run`` request form: ``check_selfplay_judge_v2.py`` [7].)

Recipe venv, login node, ~30 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_json_constraint.py
"""

import ast
import contextlib
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from i6_experiments.users.dorian_koch.speech_llm import common  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import row_filters as rf  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import selfplay_bench as sb  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm import selfplay_judge as sj  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAIL.append(msg)


REMOVED = {
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
    "guided_whitespace_pattern",
}


def removed_keys(src: str) -> list:
    """(line, key) of every dict-literal string key in REMOVED (``extra_body={"guided_json": ...}``) and every
    ``guided_*=`` keyword argument."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value in REMOVED:
                    out.append((k.lineno, k.value))
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in REMOVED:
                    out.append((kw.value.lineno, kw.arg))
    return out


print("[1] static: no removed guided_* request field")
pkg = os.path.dirname(os.path.dirname(os.path.abspath(sb.__file__)))  # .../speech_llm
fd = os.path.realpath("recipe/speech_llm/full_duplex")
files = [
    f
    for root in (pkg, fd)
    for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
    if "/tests/" not in f and "/.venv/" not in f
]
hits = []
for f in files:
    try:
        src = open(f).read()
        hits += [(f, ln, k) for ln, k in removed_keys(src)]
    except (SyntaxError, UnicodeDecodeError):
        pass
for h in hits:
    print("     ", h)
check(len(files) > 100 and not hits, f"{len(files)} files scanned, {len(hits)} removed-field uses")
mutant = 'client.chat.completions.create(model=m, extra_body={"guided_json": S})\n'
check(removed_keys(mutant) == [(1, "guided_json")], "CAN FAIL: the scanner flags the old extra_body form")

print("[2] helper")
S = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
check(
    common.json_schema_format(S, "n")
    == {"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": S}}},
    "json_schema_format builds response_format json_schema",
)
rw = open(os.path.join(fd, "moshi_family", "rl", "selfplay_reward.py")).read()
forms = [
    n
    for n in ast.walk(ast.parse(rw))
    if isinstance(n, ast.keyword) and n.arg == "response_format" and "json_schema" in ast.unparse(n.value)
]
check(len(forms) == 1, "the training self-play judge sends the same response_format json_schema form")

print("[3] SelfPlayCoherenceJudge.run (stub vLLM + client)")
from datasets import Dataset  # noqa: E402


class P:
    def __init__(self, p):
        self.p = p

    def get_path(self):
        return self.p

    get = get_path


tmp = tempfile.mkdtemp(prefix="check_json_constraint_")
rows = [
    {
        "id": f"jre_{i}",
        "n_frames": 750,
        "frame_rate": 12.5,
        "words_a": json.dumps([{"text": "hi", "start": 1.0, "end": 1.2}]),
        "words_b": json.dumps([{"text": "hello", "start": 2.0, "end": 2.3}]),
    }
    for i in range(4)
]
dpath = os.path.join(tmp, "asr")
Dataset.from_list(rows).save_to_disk(dpath)
SEEN = []


class StubClient:
    def __init__(self, **_):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        SEEN.append(kw)
        msg = type("M", (), {"content": json.dumps({"reason": "ok", "score": 4})})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


@contextlib.contextmanager
def stub_server(_name, *a, **k):
    yield "http://stub"


import openai  # noqa: E402

openai.OpenAI = StubClient
sj.vllm_server = stub_server


def run_coherence(**kw):
    SEEN.clear()
    j = sj.SelfPlayCoherenceJudge(runs={"m": P(dpath)}, **kw)
    j.out_rows, j.out_summary = P(os.path.join(tmp, "rows.jsonl")), P(os.path.join(tmp, "summary.json"))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        j.run()
    finally:
        os.chdir(cwd)
    return list(SEEN), json.load(open(os.path.join(tmp, "summary.json")))


def enforced(calls, schema):
    return bool(calls) and all(
        "extra_body" not in c and c.get("response_format", {}).get("json_schema", {}).get("schema") == schema
        for c in calls
    )


calls, summ = run_coherence()
check(enforced(calls, sj._JUDGE_JSON_SCHEMA), f"v2: all {len(calls)} calls send the schema as response_format")
check(summ.get("judge_version") == 2, "v2: summary records judge_version 2")
calls, summ = run_coherence(judge_version=1)
check(all("response_format" not in c and "extra_body" not in c for c in calls), "v1: no constraint sent")
check(summ.get("judge_version") == 1, "v1: summary records judge_version 1")
calls, _ = run_coherence(guided_json=False)
check(all("response_format" not in c and "extra_body" not in c for c in calls), "guided_json=False: no constraint")
real = sj.json_schema_format
sj.json_schema_format = lambda schema, name="reply": {"extra_body": {"guided_json": schema}}
calls, _ = run_coherence()
check(not enforced(calls, sj._JUDGE_JSON_SCHEMA), "CAN FAIL: a mutant helper (old extra_body form) is caught")
sj.json_schema_format = real

print("[4] row_filters.label_safety")
SEEN.clear()
rf.OpenAI = StubClient  # label_safety imports OpenAI inside; patched module-level openai covers it
items = [{"transcript": "a b c", "id": "x"}]
try:
    rf.label_safety(items, "http://stub", "m", sorted(rf.SAFETY_RUBRICS)[0], concurrency=1)
except Exception as e:  # the stub's reply is not a safety label: only the request form matters here
    print("     (reply parse:", type(e).__name__, ")")
check(
    enforced(SEEN, rf._SAFETY_JSON_SCHEMA),
    f"label_safety sends the safety schema as response_format ({len(SEEN)} calls)",
)

print("[5] hash")
for cls, args in (
    (sb.SelfPlayJudgeV2, {"runs": {"m": "x"}, "rubric": "spj2_judge_v1", "llm_name": "l", "temperature": 0.3}),
    (
        sj.SelfPlayCoherenceJudge,
        {
            "runs": {"m": "x"},
            "real_data": None,
            "rubric": "coherence_v1",
            "llm_name": "l",
            "temperature": 0.3,
            "guided_json": True,
        },
    ),
):
    old = cls.hash(args)
    check(cls.hash({**args, "judge_version": 1}) == old, f"{cls.__name__}: judge_version=1 keeps the old hash")
    check(cls.hash({**args, "judge_version": 2}) != old, f"{cls.__name__}: v2 is a new hash")

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED")
sys.exit(1 if FAIL else 0)
