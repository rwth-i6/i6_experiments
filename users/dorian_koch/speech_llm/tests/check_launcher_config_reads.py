"""Guard: every knob a recipe renders must be read by its launcher, unconditionally.

Two failure modes live at the recipe -> launcher seam, and both produce a run whose *results do not
match its declared config* -- which is worse than a crash, because the number looks legitimate.

**1. Rendered but never read.** A typo, or a knob wired on the render side only. `report_unread_config`
catches this at runtime, but only after the model is built and only on the code path that actually
ran. This check catches it statically, for every launcher, in a second.

**2. Read after the check, or only on one branch.** The subtler one, and the reason this check exists at all.
`report_unread_config` is *fatal* by design, so a launcher that reads `init_checkpoint` only inside
its `stage == "stage1"` branch will abort every `stage0` run -- even though the config is perfectly
valid and the key is genuinely irrelevant to that stage. Adopting the strict guard therefore forces
an invariant: **`main()` reads every knob up front, into a local, before branching.** That is a good
shape independently (the knob block documents the schema in one place), but it is not self-enforcing
-- the next person to add a branch-local `cfg.get()` reintroduces the abort, and only for the branch
they did not test. So assert it here.

Ordering is the same bug wearing a different hat: `report_unread_config` can only see the reads that
have already happened, so a `cfg.get(...)` sitting in the argument list of the `run_training(...)`
call *below* it counts as unread and aborts the run. That is precisely how the audex launcher failed
on its first real submission after adopting the strict guard (`sample_every`), with every static check
green -- the read was at the top level of `main()`, just late. So both rules are enforced here.

Both audex and rl violated (2) before 2026-08-01: they carried private `_load_config` copies and so
had never been subject to the strict guard at all.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_launcher_config_reads.py
"""

import ast
import builtins
import re
import sys
from pathlib import Path

# Walk up the UNRESOLVED path: recipe/i6_experiments and recipe/speech_llm are symlinks into the
# projects/ repos, so .resolve() lands inside one of those repos and never sees the setup root.
SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
LIB = SETUP / "recipe" / "speech_llm" / "full_duplex"
RECIPES = SETUP / "recipe" / "i6_experiments" / "users" / "dorian_koch" / "speech_llm"

#: launcher file -> (recipe file, name of the function rendering its YAML).
#: The fork launchers under ``i6_experiments/`` are the live A/B references and are checked too.
PAIRS = {
    LIB / "moshi_family/moshi_finetune_launcher.py": ("finetune.py", "_render_moshi_lib_config"),
    LIB / "moshi_family/moshirag_finetune_launcher.py": ("finetune.py", "_render_moshirag_lib_config"),
    LIB / "moshi_family/personaplex/finetune_launcher.py": ("finetune.py", "_render_personaplex_config"),
    LIB / "moshi_family/audex/finetune_launcher.py": ("audex_finetune.py", "_render_audex_config"),
    LIB / "moshi_family/rl/launcher.py": ("rl_finetune.py", "_render_rl_config"),
    RECIPES / "moshi_finetune_launcher.py": ("finetune.py", "_render_moshi_finetune_config"),
    RECIPES / "personaplex_finetune_launcher.py": ("finetune.py", "_render_personaplex_config"),
}

#: Launchers that must read their whole config before branching. Every launcher whose recipe calls
#: ``report_unread_config`` belongs here -- the strict guard makes a branch-local read fatal.
UNCONDITIONAL = {p for p in PAIRS if "full_duplex" in str(p)}

#: Launchers that hand the rendered YAML to a *third-party* parser instead of reading ``cfg`` keys
#: themselves -- this one is a thin wrapper around the vendored kyutai moshi-finetune fork and ends in
#: ``fire.Fire(train_module.train)``, which binds the YAML to the fork's own dataclasses. Scanning for
#: ``cfg.get(...)`` therefore finds nothing, and "renders 20, reads 0" is correct rather than a bug.
#: The exemption is checked, not assumed: an exempt launcher that starts reading ``cfg`` itself fails
#: below, so converting one to our style cannot silently keep the free pass.
DELEGATED = {
    RECIPES / "moshi_finetune_launcher.py",
}

#: A YAML key at the start of a line in a rendered config.
KEY_RE = re.compile(r"^([a-z_][a-z_0-9]*):", re.MULTILINE)

# Every adapter must appear in PAIRS -- a new launcher silently skipping this check is the whole
# failure mode we are guarding against.
declared = set()
for recipe in RECIPES.glob("*.py"):
    for m in re.finditer(r'launcher_module="([^"]+)"', recipe.read_text()):
        declared.add(m.group(1))
assert declared, f"no launcher_module= found under {RECIPES} -- has the layout moved?"
covered = {p.stem for p in PAIRS} | {f"{p.parent.name}.{p.stem}" for p in PAIRS}
missing = {d for d in declared if not any(d.endswith(c) for c in covered)}
assert not missing, f"launcher(s) with no entry in PAIRS: {sorted(missing)}"
print(f"[ok] all {len(declared)} declared launcher modules are covered")


def rendered_keys(recipe_file: str, fn_name: str) -> set:
    """The YAML keys a render function emits, read off its f-string template."""
    tree = ast.parse((RECIPES / recipe_file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            text = ""
            for js in ast.walk(node):
                if isinstance(js, ast.JoinedStr):
                    for part in js.values:
                        # Interpolations become a placeholder so they cannot look like a key.
                        text += part.value if isinstance(part, ast.Constant) else "?"
            assert text, f"{fn_name} has no f-string template"
            return set(KEY_RE.findall(text))
    raise AssertionError(f"{fn_name} not found in {recipe_file}")


def read_keys(path: Path) -> tuple[set, set, set]:
    """Config keys main() reads -> (all, read inside a branch/closure, read after the strict check)."""
    tree = ast.parse(path.read_text())
    main = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    assert main is not None, f"{path.name} has no main()"

    # Line of the report_unread_config(...) call: every read must be strictly above it, because the
    # tracking dict only knows about reads that have already run when the check fires.
    check_line = min(
        (
            n.lineno
            for n in ast.walk(main)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "report_unread_config"
        ),
        default=None,
    )

    all_keys, conditional, late = set(), set(), set()

    def visit(node, guarded: bool):
        for child in ast.iter_child_nodes(node):
            key = config_key(child)
            if key is not None:
                all_keys.add(key)
                if guarded:
                    conditional.add(key)
                if check_line is not None and child.lineno > check_line:
                    late.add(key)
            # A read inside a branch, a loop, a try, or a closure is not guaranteed to happen.
            nested = guarded or isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.FunctionDef, ast.IfExp))
            visit(child, nested)

    visit(main, False)
    return all_keys, conditional, late


def config_key(node) -> str | None:
    """``cfg.get("k", ...)`` or ``cfg["k"]`` -> ``"k"``, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cfg"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ):
        return node.args[0].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "cfg"
        and isinstance(node.slice, ast.Constant)
    ):
        return node.slice.value
    return None


failures = []
for launcher, (recipe_file, fn_name) in sorted(PAIRS.items()):
    emitted = rendered_keys(recipe_file, fn_name)
    read, conditional, late = read_keys(launcher)
    name = launcher.name if launcher.parent.name == "speech_llm" else f"{launcher.parent.name}/{launcher.name}"

    if launcher in DELEGATED:
        if read:
            failures.append(
                f"{name}: listed in DELEGATED (config assumed handled by the vendored fork's own "
                f"parser) but it now reads {sorted(read)} from cfg itself -- drop it from DELEGATED "
                f"so its keys are actually checked"
            )
        print(f"[--] {name:<42} {len(emitted)} rendered, parsed by the fork (exempt)")
        continue

    unread = sorted(emitted - read)
    if unread:
        failures.append(
            f"{name}: {recipe_file}::{fn_name} renders {unread} but the launcher never reads them "
            f"-- the recipe asks for something that does not happen"
        )

    if launcher in UNCONDITIONAL:
        after = sorted(emitted & late)
        if after:
            failures.append(
                f"{name}: reads {after} BELOW its report_unread_config() call, which therefore sees "
                f"them as unread and aborts the run. Move the cfg.get(...) above the check (usually "
                f"into the knob block at the top of main())."
            )
        branchy = sorted(emitted & conditional)
        if branchy:
            failures.append(
                f"{name}: reads {branchy} only inside a branch/closure. report_unread_config() is "
                f"fatal, so any run taking the other path aborts on a valid config. Hoist the "
                f"cfg.get(...) into the knob block at the top of main()."
            )

    print(f"[ok] {name:<42} {len(emitted)} rendered, {len(read)} read")


# --- use-before-assignment -------------------------------------------------------------------
#
# The third way the knob block goes wrong, and the meanest (2026-09-07). `full_finetuning` was read
# at the top, but the assert protecting full-FT runs referenced `kl_weight`/`l2sp_weight` which were
# only assigned 19 lines LOWER. Python evaluates `A and B` lazily, so on every LoRA run
# (`full_finetuning` False) the undefined names were never touched and the module looked fine --
# every static check here was green, and the four A16 LoRA arms would have run. The single run shape
# the assert exists to protect died on `UnboundLocalError` two minutes in, after 17 days queued.
#
# Neither of the two rules above can see this: the read IS unconditional and IS above the strict
# check. What is wrong is purely the order of two lines. So check that too, for every function in
# every launcher: no local may be LOADED above its first assignment.


def _scope_names(fn):
    """(first_assign_line, first_load_line) for locals of ``fn``, ignoring nested scopes.

    Nested ``def``/``lambda``/comprehensions are separate scopes whose loads are deferred to call
    time, so a load inside one says nothing about ordering here; they are not descended into.
    """
    assigned, loaded, skip = {}, {}, set()

    def visit(node, top=False):
        if not top and isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            return  # separate scope
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            skip.update(node.names)
        if isinstance(node, ast.ExceptHandler) and node.name:
            assigned.setdefault(node.name, node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                assigned.setdefault((a.asname or a.name).split(".")[0], node.lineno)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            loaded.setdefault(node.target.id, node.lineno)  # x += 1 reads x first
        if isinstance(node, ast.Name):
            book = assigned if isinstance(node.ctx, ast.Store) else loaded
            book.setdefault(node.id, node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs:
        assigned.setdefault(a.arg, fn.lineno)
    for a in (fn.args.vararg, fn.args.kwarg):
        if a:
            assigned.setdefault(a.arg, fn.lineno)
    visit(fn, top=True)
    for name in skip:
        assigned.pop(name, None)
        loaded.pop(name, None)
    return assigned, loaded


def use_before_assignment(tree, module_names):
    """[(function, name, load_line, assign_line)] for every local loaded above its assignment."""
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned, loaded = _scope_names(fn)
        for name, load_line in sorted(loaded.items(), key=lambda kv: kv[1]):
            if name in module_names or name in dir(builtins):
                continue  # a global/builtin of the same name is legal and common
            assign_line = assigned.get(name)
            if assign_line is not None and load_line < assign_line:
                out.append((fn.name, name, load_line, assign_line))
    return out


def module_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    names.add(sub.id)
    return names


#: The bug exactly as it shipped, plus its fix. Asserted below, so this check cannot rot into a
#: no-op: if the detector stops working, BAD stops being flagged and the self-test fails.
_BAD = """
def main(cfg):
    full_finetuning = bool(cfg.get("full_finetuning", False))
    assert not (full_finetuning and (kl_weight > 0 or l2sp_weight > 0)), "..."
    kl_weight = float(cfg.get("kl_weight", 0.0))
    l2sp_weight = float(cfg.get("l2sp_weight", 0.0))
"""
_GOOD = """
def main(cfg):
    full_finetuning = bool(cfg.get("full_finetuning", False))
    kl_weight = float(cfg.get("kl_weight", 0.0))
    l2sp_weight = float(cfg.get("l2sp_weight", 0.0))
    assert not (full_finetuning and (kl_weight > 0 or l2sp_weight > 0)), "..."
"""

_bad = use_before_assignment(ast.parse(_BAD), set())
assert {n for _, n, _, _ in _bad} == {"kl_weight", "l2sp_weight"}, (
    f"self-test: the detector no longer catches the 2026-09-07 bug it was written for: {_bad}"
)
assert not use_before_assignment(ast.parse(_GOOD), set()), "self-test: false positive on valid code"

for launcher in PAIRS:
    tree = ast.parse(launcher.read_text())
    for fn_name, name, load_line, assign_line in use_before_assignment(tree, module_level_names(tree)):
        failures.append(
            f"{launcher.name}::{fn_name}: reads local '{name}' at line {load_line} but only assigns "
            f"it at line {assign_line} -- UnboundLocalError whenever that line is reached. Note a "
            f"short-circuiting `and`/`or` can hide this on every run shape but one."
        )

print(f"[ok] {'use-before-assignment':<42} {len(PAIRS)} launchers scanned")


# --- the two defaults for one key must agree (backlog C6, 2026-09-15) ---------------------------
#
# Every knob is written TWICE: the renderer emits `hp.get("k", D1)` and the launcher reads
# `cfg.get("k", D2)`. ~25 constants are duplicated this way. Today the launcher's default is dead
# code -- the renderer always emits the key, and `report_unread_config` is fatal on an unread one --
# which is exactly what makes a divergence silent and survivable until it is not.
#
# It had already happened: `sample_every` was 100 in the renderer and 0 in the launcher. Nothing
# failed, because every rendered config carried the key. But a config written BEFORE a key exists
# falls through to the launcher default on a resume, and then the two numbers are not academic --
# 7 lib configs predating `sample_every` do exactly that.
#
# So: same key, same default, checked statically in both files.


def _const(node):
    """A literal default, or the sentinel `_NOT_LITERAL` for anything computed.

    `"null"` from the renderer is normalised to `None`: the template emits the YAML *token*, which
    the launcher's parser turns back into `None`, so the two agree even though the Python literals
    differ. Normalising is right here and an exemption would not be -- the values really are equal
    once the config round-trips.
    """
    if isinstance(node, ast.Constant):
        return None if node.value == "null" else node.value
    if isinstance(node, (ast.List, ast.Tuple)) and not node.elts:
        return ()
    return _NOT_LITERAL


_NOT_LITERAL = object()


def _get_defaults(tree, obj_name: str) -> dict:
    """{key: default} for every `<obj_name>.get("key", <literal>)` in the tree.

    A key read more than once with DIFFERENT literal defaults is itself a bug, so it is recorded
    as a conflict rather than silently taking the last one.
    """
    out, conflict = {}, set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != obj_name or len(node.args) != 2:
            continue
        k = node.args[0]
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            continue
        d = _const(node.args[1])
        if d is _NOT_LITERAL:
            continue
        if k.value in out and out[k.value] != d:
            conflict.add(k.value)
        out[k.value] = d
    for k in conflict:
        out.pop(k, None)
    return out


#: Keys whose two defaults may legitimately differ, each with the reason. Deliberately empty --
#: an entry here is a claim that a silent divergence is FINE for that key, which needs an argument.
DEFAULT_MISMATCH_OK: dict = {}

_render_tree = ast.parse((RECIPES / "finetune.py").read_text())
mismatches = 0
for launcher, (recipe_file, fn_name) in sorted(PAIRS.items()):
    render_fn = next(
        (
            n
            for n in ast.walk(ast.parse((RECIPES / recipe_file).read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == fn_name
        ),
        None,
    )
    if render_fn is None:
        continue
    rendered = _get_defaults(render_fn, "hp")
    read = _get_defaults(ast.parse(launcher.read_text()), "cfg")
    shared = sorted(set(rendered) & set(read))
    for k in shared:
        if k in DEFAULT_MISMATCH_OK:
            continue
        if rendered[k] != read[k]:
            mismatches += 1
            failures.append(
                f"{launcher.name}: default for '{k}' is {rendered[k]!r} in {fn_name} but "
                f"{read[k]!r} in the launcher. Harmless only while every rendered config carries "
                f"the key -- a config written before the key existed takes the launcher's value on "
                f"a resume, and the two silently disagree."
            )
    print(f"[ok] {launcher.name:<42} {len(shared):>2} shared defaults compared")

# Non-vacuous: the comparison must be capable of finding something.
assert any(
    _get_defaults(
        next(
            n
            for n in ast.walk(ast.parse((RECIPES / rf).read_text()))
            if isinstance(n, ast.FunctionDef) and n.name == fn
        ),
        "hp",
    )
    for rf, fn in PAIRS.values()
), "no rendered defaults were extracted at all -- the AST reader has drifted"

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(PAIRS)} launchers consume their rendered config, unconditionally")
