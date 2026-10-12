"""Guard: ONE word rule for reply length and ``spoke`` -- the GRPO trainer's ``count_words`` (``[A-Za-z0-9']+``).

Why (duplication audit item 7f, user 2026-10-12: "so everything just uses >=1"): eval ``spoke`` counted whitespace
tokens (``evalstats.n_words`` / ``read_benchmarks.py``), the RL length term counts ``[A-Za-z0-9']+`` runs
(``moshi_family/rl/knowledge_reward.count_words``). Over 244,526 saved replies, 142 had >= 1 whitespace token but 0
regex words -- all punctuation ('... ... ...', '--') or non-Latin-only Whisper output ('ご視聴ありがとうございました',
'Продолжение следует...'), none judged correct -- so the ledger called them "spoke". Now ``evalstats.n_words`` is a
stdlib COPY of the trainer's rule (the login-node analysis code cannot import the trainer), and every analysis caller
goes through ``evalstats.spoke`` / ``n_words`` at >= 1 word.

  [1] parity: ``knowledge_reward.count_words`` (imported for real), ``evalstats.n_words`` and
      ``instruct_prompts.n_words`` (prompt filter, AST-extracted: importing it needs sisyphus) give the same count on
      a battery of verbatim saved replies + English edge cases; the trainer's pattern is the canonical one.
  [2] real callers: a sandbox setup (copies of ``read_benchmarks.py`` + ``analysis/*.py`` + a fixture
      ``output/benchmark/llm_judge/n9/<tag>``) runs ``read_benchmarks.py`` and ``analysis/cond_acc.py`` as
      subprocesses; both must print spoke 44.4% / acc|spoke 50.0% (whitespace rule: 88.9% / 25.0%).
  [3] no second definition: AST scan of ``analysis/*.py`` + ``read_benchmarks.py``: no ``len(<reply>.split())``,
      no word-regex ``re.*`` call, no ``def n_words/spoke/count_words`` outside evalstats, and a script using a
      ``spoke`` name imports evalstats (``SPOKE_OK`` lists the deliberate non-reply uses).

Out of scope on purpose (different measures, kept): echo/WER ``words`` (lowercase + strip, a word SEQUENCE for WER),
MRQA's ``spoke`` (upstream evaluator: non-empty text), the in-loop probe's ``coherence_stats`` (a logged time series),
corpus / prompt / self-play-rate word counts.

CAN FAIL: ``--selftest`` mutates the sandbox copies (never the real files) and requires a flag for each: evalstats
back to ``str.split``; evalstats regex drifts to ``[A-Za-z0-9]+``; ``read_benchmarks.conditioned`` counts its own
``len(reply.split())``; ``cond_acc.py`` gets a private ``len(i["reply"].split())``.

Run from the setup root, no GPU, ~5 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_spoke_words.py [--selftest]
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
FD = SETUP / "recipe" / "speech_llm" / "full_duplex"
I6 = SETUP / "recipe" / "i6_experiments" / "users" / "dorian_koch"
ANALYSIS = SETUP / "analysis"
TAG = "zz_check_spoke_words"

# Verbatim saved replies (output/benchmark/*/eval_results, 2026-10-12) + English edge cases.
BATTERY = [
    "",
    "   ",
    "... ... ...",
    "--",
    "---.---------------",
    ", ,",
    "ご視聴ありがとうございました",
    "Продолжение следует...",
    "Стив Волзняк",
    "霍尔姆斯女华生的作者是柯南道尔这本书可是侦探小说的经典之作呢",
    "Yeah.",
    "Roger Waters.",
    "Sesame seeds.",
    "It’s Paris.",
    "It's Paris.",
    "9-5",
    "Champs-Élysées,",
    "Malmö.",
    "the 1960s, I think -- maybe '68?",
]
# (reply, correct): 4 of 9 have a word under the rule, 2 of those correct.
FIXTURE = [
    ("", 0),
    ("... ... ...", 0),
    ("--", 0),
    ("ご視聴ありがとうございました", 0),
    ("Продолжение следует...", 0),
    ("Yes.", 0),
    ("Roger Waters.", 1),
    ("9-5", 0),
    ("It’s Paris.", 1),
]
WANT = ("44.4%", "50.0%")  # whitespace rule would give 88.9% / 25.0%
REPLY_HINT = re.compile(r"reply|transcri|response|pred|hyp|\brep\b")
#: scripts whose `spoke` is not a reply-word rule: selfplay_timeline = first-word ONSET timing of self-play audio.
SPOKE_OK = {"selfplay_timeline.py"}


def load_evalstats(path):
    spec = importlib.util.spec_from_file_location(f"evalstats_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ast_word_pattern(path, name="_WORD"):
    for n in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            c = n.value
            if isinstance(c, ast.Call) and c.args and isinstance(c.args[0], ast.Constant):
                return c.args[0].value
    raise AssertionError(f"{path}: no {name} = re.compile(...)")


def check_parity(es_path):
    sys.path.insert(0, str(FD))
    from moshi_family.rl import knowledge_reward as kr

    es = load_evalstats(es_path)
    fails = []
    if kr._WORD.pattern != r"[A-Za-z0-9']+":
        fails.append(f"trainer pattern changed to {kr._WORD.pattern!r}: re-measure the ledger before following it")
    ip = re.compile(ast_word_pattern(I6 / "speech_llm" / "instruct_prompts.py"))
    for r in BATTERY:
        a, b, c = kr.count_words(r), es.n_words(r), len(ip.findall(r))
        if not (a == b == c):
            fails.append(f"{r!r}: trainer {a}, evalstats {b}, instruct_prompts {c}")
    if es.SPOKE_MIN_WORDS != 1:
        fails.append(f"evalstats.SPOKE_MIN_WORDS = {es.SPOKE_MIN_WORDS} (user: >= 1)")
    return fails


def make_sandbox(root):
    shutil.copy(SETUP / "read_benchmarks.py", root / "read_benchmarks.py")
    shutil.copytree(ANALYSIS, root / "analysis", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), symlinks=True)
    d = root / "output" / "benchmark" / "llm_judge" / f"n{len(FIXTURE)}" / TAG
    d.mkdir(parents=True)
    n = len(FIXTURE)
    k = sum(c for _, c in FIXTURE)
    (d / "summary").write_text(
        json.dumps({"overall": {"accuracy": k / n, "n": n, "avg_quality": 1.0}, "grader": {"name": "llm_judge"}})
    )
    with open(d / "eval_results", "w") as f:
        for i, (rep, c) in enumerate(FIXTURE):
            f.write(json.dumps({"question": f"q{i}", "transcription": rep, "binary_correct": c}) + "\n")


def check_callers(root):
    fails = []
    py = sys.executable
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([py, "read_benchmarks.py", "-q", TAG], cwd=root, capture_output=True, text=True, env=env)
    row = [ln for ln in out.stdout.splitlines() if ln.split()[:1] in ([TAG], [f"n{len(FIXTURE)}/{TAG}"])]
    if out.returncode or len(row) != 1:
        fails.append(f"read_benchmarks.py: rc {out.returncode}, no row\n{out.stdout[-800:]}{out.stderr[-800:]}")
    else:
        got = re.findall(r"\d+\.\d%", row[0])  # acc, spoke, acc|spoke
        if got[-2:] != list(WANT):
            fails.append(f"read_benchmarks.py spoke/acc|spoke {got[-2:]} != {list(WANT)}: {row[0]!r}")
    out = subprocess.run(
        [py, "analysis/cond_acc.py", f"n{len(FIXTURE)}/{TAG}"], cwd=root, capture_output=True, text=True, env=env
    )
    row = [ln for ln in out.stdout.splitlines() if ln.split()[:1] in ([TAG], [f"n{len(FIXTURE)}/{TAG}"])]
    if out.returncode or len(row) != 1:
        fails.append(f"cond_acc.py: rc {out.returncode}\n{out.stdout[-800:]}{out.stderr[-800:]}")
    else:
        got = re.findall(r"\d+\.\d%", row[0])  # headline, spoke, acc|spoke
        if got[1:3] != list(WANT):
            fails.append(f"cond_acc.py spoke/acc|spoke {got[1:3]} != {list(WANT)}: {row[0]!r}")
    return fails


def scan(files):
    fails = []
    for f in files:
        name = f.name
        tree = ast.parse(f.read_text())
        imports_es = any(
            (isinstance(n, ast.Import) and any(a.name == "evalstats" for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module == "evalstats")
            for n in ast.walk(tree)
        )
        ids = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                ids.add(n.id)
            elif isinstance(n, ast.Attribute):
                ids.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                ids.add(n.value)
            if name == "evalstats.py":
                continue
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "len" and n.args:
                a = n.args[0]
                if (
                    isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Attribute)
                    and a.func.attr == "split"
                    and not a.args
                ):
                    src = ast.unparse(a.func.value)
                    if REPLY_HINT.search(src):
                        fails.append(f"{name}:{n.lineno}: len({src}.split()) -- use evalstats.n_words")
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "re"
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and "A-Za-z0-9" in str(n.args[0].value)
            ):
                fails.append(f"{name}:{n.lineno}: private word regex -- use evalstats.n_words")
            if isinstance(n, ast.FunctionDef) and n.name in ("n_words", "spoke", "count_words", "_n_words"):
                fails.append(f"{name}:{n.lineno}: def {n.name} -- the one copy is evalstats")
        uses = [i for i in ids if re.fullmatch(r"(n_)?spoke(_frac|_share)?|acc_spoke|frac_spoke", i)]
        if uses and not imports_es and name not in SPOKE_OK:
            fails.append(f"{name}: uses {sorted(uses)} without importing evalstats")
    return fails


def run_all(root):
    fails = check_parity(root / "analysis" / "evalstats.py")
    fails += check_callers(root)
    fails += scan(sorted((root / "analysis").glob("*.py")) + [root / "read_benchmarks.py"])
    return fails


def mutate(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, f"mutation anchor moved in {path.name}: {old!r}"
    path.write_text(s.replace(old, new))


def main():
    selftest = "--selftest" in sys.argv
    with tempfile.TemporaryDirectory(prefix="check_spoke_") as td:
        root = Path(td) / "base"
        root.mkdir()
        make_sandbox(root)
        fails = run_all(root)
    print(f"[1-3] real tree: {'PASS' if not fails else 'FAIL'}")
    for f in fails:
        print("   ", f)
    if not selftest:
        return 1 if fails else 0
    mutants = [
        (
            "evalstats -> str.split",
            "analysis/evalstats.py",
            'return len(_WORD.findall(reply or ""))',
            'return len((reply or "").split())',
        ),
        (
            "evalstats regex drift",
            "analysis/evalstats.py",
            '_WORD = re.compile(r"[A-Za-z0-9\']+")',
            '_WORD = re.compile(r"[A-Za-z0-9]+")',
        ),
        (
            "read_benchmarks private split",
            "read_benchmarks.py",
            "if es.spoke(reply, MIN_WORDS):",
            "if len(reply.split()) >= MIN_WORDS:",
        ),
        (
            "cond_acc private split",
            "analysis/cond_acc.py",
            'w = [es.n_words(i["reply"]) for i in items]',
            'w = [len(i["reply"].split()) for i in items]',
        ),
    ]
    bad = 0
    for label, rel, old, new in mutants:
        with tempfile.TemporaryDirectory(prefix="check_spoke_m_") as td:
            root = Path(td) / "m"
            root.mkdir()
            make_sandbox(root)
            mutate(root / rel, old, new)
            got = run_all(root)
        print(
            f"  mutant {label:<32} {'CAUGHT' if got else 'MISSED'} ({len(got)} flags; first: {got[0][:110] if got else '-'})"
        )
        bad += not got
    print(f"selftest: {len(mutants) - bad}/{len(mutants)} mutants caught")
    return 1 if (fails or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
