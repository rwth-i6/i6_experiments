"""Guard: every knob a recipe renders must be read by its launcher, unconditionally.

Two failure modes live at the recipe -> launcher seam, and both produce a run whose *results do not
match its declared config* -- which is worse than a crash, because the number looks legitimate.

**1. Rendered but never read.** A typo, or a knob wired on the render side only. `report_unread_config`
catches this at runtime, but only after the model is built and only on the code path that actually
ran. This check catches it statically, for every launcher, in a second.

**2. Read, but only on one branch.** The subtler one, and the reason this check exists at all.
`report_unread_config` is *fatal* by design, so a launcher that reads `init_checkpoint` only inside
its `stage == "stage1"` branch will abort every `stage0` run -- even though the config is perfectly
valid and the key is genuinely irrelevant to that stage. Adopting the strict guard therefore forces
an invariant: **`main()` reads every knob up front, into a local, before branching.** That is a good
shape independently (the knob block documents the schema in one place), but it is not self-enforcing
-- the next person to add a branch-local `cfg.get()` reintroduces the abort, and only for the branch
they did not test. So assert it here.

Both audex and rl violated (2) before 2026-08-01: they carried private `_load_config` copies and so
had never been subject to the strict guard at all.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_launcher_config_reads.py
"""

import ast
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


def read_keys(path: Path) -> tuple[set, set]:
    """Config keys the launcher reads -> (all keys, keys read inside a branch or nested function)."""
    tree = ast.parse(path.read_text())
    main = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    assert main is not None, f"{path.name} has no main()"

    all_keys, conditional = set(), set()

    def visit(node, guarded: bool):
        for child in ast.iter_child_nodes(node):
            key = config_key(child)
            if key is not None:
                all_keys.add(key)
                if guarded:
                    conditional.add(key)
            # A read inside a branch, a loop, a try, or a closure is not guaranteed to happen.
            nested = guarded or isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.FunctionDef, ast.IfExp))
            visit(child, nested)

    visit(main, False)
    return all_keys, conditional


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
    read, conditional = read_keys(launcher)
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
        branchy = sorted(emitted & conditional)
        if branchy:
            failures.append(
                f"{name}: reads {branchy} only inside a branch/closure. report_unread_config() is "
                f"fatal, so any run taking the other path aborts on a valid config. Hoist the "
                f"cfg.get(...) into the knob block at the top of main()."
            )

    print(f"[ok] {name:<42} {len(emitted)} rendered, {len(read)} read")

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"\nall {len(PAIRS)} launchers consume their rendered config, unconditionally")
