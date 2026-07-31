"""Guard: a tokenizer subclass method that overrides nothing and is called by nobody is a bug.

``DuplexTokenizerBase`` is a template-method base: subclasses customise the pipeline by overriding
named hooks (``interleave_text``, ``build_codes``, ...). Python will happily let a subclass define
``_interleave_text`` instead -- a leading underscore, a typo, a hook the base later renamed -- and
nothing complains. The subclass looks customised, the base's default silently runs instead, and the
run produces a plausible loss curve for the wrong configuration.

That is not hypothetical. ``AudexTextMoshiTokenizer`` defined ``_interleave_text`` while every
caller used ``interleave_text`` (found 2026-07-31), so Audex Stage-1 tokenized its inner monologue
with moshiko's SentencePiece instead of Audex's tokenizer -- defeating the entire purpose of the
class, exactly as its own docstring described it.

The invariant: every method a tokenizer subclass defines must either **override** something in its
MRO, or be **referenced** somewhere in the package. A method that does neither is unreachable, and
unreachable customisation is indistinguishable from a silently-broken override.

Uses AST over the sources rather than importing, so it stays a cheap login-node check that does not
need torch/transformers/sisyphus loadable.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_tokenizer_overrides.py
"""

import ast
import os
import re
import sys

PKG = os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex", "moshi_family")

# Every module that defines a duplex tokenizer, plus the base they all derive from.
SOURCES = [
    "train_data_common.py",
    "moshi_train_data.py",
    "moshirag_train_data.py",
    os.path.join("personaplex", "train_data.py"),
    os.path.join("audex", "train_data.py"),
]
ROOT_BASE = "DuplexTokenizerBase"


def load_classes():
    """Map class name -> (base names, {method name: lineno}, source file), across the tokenizer modules."""
    classes, missing = {}, []
    for rel in SOURCES:
        path = os.path.join(PKG, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        tree = ast.parse(open(path).read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            bases += [b.attr for b in node.bases if isinstance(b, ast.Attribute)]
            methods = {m.name: m.lineno for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
            classes[node.name] = (bases, methods, rel)
    assert not missing, f"tokenizer sources missing -- update SOURCES: {missing}"
    assert ROOT_BASE in classes, f"{ROOT_BASE} not found; did train_data_common.py move?"
    return classes


def inherited_methods(name, classes, _seen=None):
    """All method names reachable through this class's ancestors (excluding the class itself)."""
    _seen = _seen or set()
    out = set()
    bases, _, _ = classes.get(name, ([], {}, None))
    for base in bases:
        if base in _seen:
            continue
        _seen.add(base)
        if base in classes:
            out |= set(classes[base][1])
            out |= inherited_methods(base, classes, _seen)
    return out


def derives_from_root(name, classes, _seen=None):
    _seen = _seen or set()
    if name == ROOT_BASE:
        return True
    if name in _seen:
        return False
    _seen.add(name)
    bases, _, _ = classes.get(name, ([], {}, None))
    return any(derives_from_root(b, classes, _seen) for b in bases)


def package_sources():
    """Every .py under moshi_family, so 'is this name referenced anywhere' is answered package-wide."""
    blobs = {}
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                blobs[p] = open(p, encoding="utf-8", errors="replace").read()
    return blobs


def referenced_outside(method, defining_file, blobs):
    """True if `method` is used as an attribute/call anywhere other than its own `def` line."""
    pattern = re.compile(rf"(?:\.|\b){re.escape(method)}\s*\(")
    for path, src in blobs.items():
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ")) and defining_file in path:
                continue  # the definition itself is not a use
            if pattern.search(line):
                return True
    return False


classes = load_classes()
blobs = package_sources()
base_methods = set(classes[ROOT_BASE][1])

subclasses = sorted(n for n in classes if n != ROOT_BASE and derives_from_root(n, classes))
assert subclasses, "no DuplexTokenizerBase subclasses found -- this guard would check nothing"

problems = []
for name in subclasses:
    _, methods, rel = classes[name]
    ancestors = inherited_methods(name, classes)
    for method, lineno in sorted(methods.items()):
        if method.startswith("__") and method.endswith("__"):
            continue
        if method in ancestors:
            continue  # a genuine override
        if referenced_outside(method, rel, blobs):
            continue  # not an override, but genuinely called
        near = sorted(b for b in ancestors | base_methods if b.lstrip("_") == method.lstrip("_"))
        hint = f" -- did you mean to override {near[0]!r}?" if near else ""
        problems.append(f"{rel}:{lineno} {name}.{method} overrides nothing and is never called{hint}")
    print(f"[ok] {name:26s} {len(methods):2d} methods, {len(ancestors)} inherited  ({rel})")

assert not problems, "unreachable tokenizer methods -- these look like customisation but never run:\n  " + "\n  ".join(
    problems
)

print(f"\nall {len(subclasses)} tokenizer subclasses have reachable overrides")
