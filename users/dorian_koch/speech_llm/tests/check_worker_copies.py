"""Guard: helpers deliberately COPIED between standalone worker scripts stay identical.

Why (duplication audit item 8, 2026-10-12): ``podcast_duplex_main.py`` (JRE separation, DuplexChat venv) and
``podcast_episode_main.py`` (episode diarisation, DuplexChat venv) run as standalone scripts in a foreign venv
(torch 2.11) that must not import the ``moshi_family`` package, so they cannot share a module and each carries its
own ``ffmpeg_to_wav`` / ``download`` / ``xcorr``. ``ffmpeg_to_wav`` is the TIME BASE of every diarised span (its own
docstring: "Keep the flags identical to podcast_duplex_main.ffmpeg_to_wav"), ``xcorr`` is the only check that
separation happened. Today: ``ffmpeg_to_wav`` and ``download`` are AST-identical (docstrings and comments aside),
``xcorr`` is written differently but computes the same float64 zero-lag Pearson correlation.

  [1] each AST pair is identical (docstrings stripped);
  [2] each numeric pair agrees on random and edge inputs (constant channel, different lengths, a scaled copy).

CAN FAIL: the self-test changes one ffmpeg flag in one copy (``-ar`` -> ``-r``) and drops the mean subtraction from
one ``xcorr``; both must be flagged.

Run from the setup root, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_worker_copies.py
"""

import ast
import sys
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
MF = SETUP / "recipe" / "speech_llm" / "full_duplex" / "moshi_family"
A, B = "podcast_duplex_main.py", "podcast_episode_main.py"
AST_PAIRS = [(A, B, "ffmpeg_to_wav"), (A, B, "download")]
NUMERIC_PAIRS = [(A, B, "xcorr")]


def _fn(src: str, name: str) -> ast.FunctionDef:
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    return fn


def _load(src: str, name: str):
    """The function alone, compiled from its source (the scripts' top level needs nothing but numpy for these)."""
    ns = {"np": np}
    exec(compile(ast.Module(body=[_fn(src, name)], type_ignores=[]), name, "exec"), ns)
    return ns[name]


def _inputs():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(48_000).astype(np.float32)
    y = rng.standard_normal(47_000).astype(np.float32)
    yield x, y
    yield x, 0.5 * x + 0.1 * rng.standard_normal(len(x)).astype(np.float32)  # correlated
    yield x, np.full_like(x, 0.3)  # constant channel -> 0
    yield x + 3.0, -2.0 * x  # offset + scaled copy -> -1
    yield np.zeros(10, np.float32), np.zeros(10, np.float32)


def check(srcs: dict[str, str]) -> list[str]:
    bad = []
    for a, b, name in AST_PAIRS:
        if ast.dump(_fn(srcs[a], name)) != ast.dump(_fn(srcs[b], name)):
            bad.append(f"{name}: {a} and {b} differ -- the copies must stay identical (standalone workers)")
    for a, b, name in NUMERIC_PAIRS:
        fa, fb = _load(srcs[a], name), _load(srcs[b], name)
        for k, (x, y) in enumerate(_inputs()):
            va, vb = fa(x.copy(), y.copy()), fb(x.copy(), y.copy())
            if not np.isclose(va, vb, rtol=0, atol=1e-12):
                bad.append(f"{name}: input {k}: {a} -> {va!r}, {b} -> {vb!r}")
    return bad


MUTATIONS = [
    (B, '            "-ar",\n', '            "-r",\n', "one ffmpeg_to_wav copy changes a flag", "ffmpeg_to_wav"),
    (A, "    x, y = x - x.mean(), y - y.mean()\n", "", "one xcorr copy drops the mean subtraction", "xcorr: input"),
]


def main():
    srcs = {f: (MF / f).read_text() for f in (A, B)}
    bad = check(srcs)
    failed = bool(bad)
    print(
        f"[{'ok' if not bad else 'FAIL'}] {len(AST_PAIRS)} AST pairs identical, {len(NUMERIC_PAIRS)} numeric pair(s) agree"
    )
    for b in bad:
        print(f"   - {b}")
    for f, old, new, what, expect in MUTATIONS:
        assert srcs[f].count(old) >= 1, f"self-test: {f} no longer contains {old!r} -- update {what!r}"
        got = [g for g in check({**srcs, f: srcs[f].replace(old, new, 1)}) if expect in g]
        failed |= not got
        print(f"[{'ok' if got else 'FAIL'}] can fail: {what:44s} -> {(got or ['NOT flagged'])[0][:90]}")
    if failed:
        sys.exit(1)
    print("\nthe standalone workers' copied helpers agree")


if __name__ == "__main__":
    main()
