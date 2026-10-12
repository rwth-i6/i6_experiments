"""Guard: the knowledge benchmark's 2 s lead-in is ONE number -- every copy equals ``offline_cli.DEFAULT_LEAD_IN_S``.

Why (duplication audit item 8, 2026-10-12): "2 s of user-channel silence before the question" was spelled ``2.0`` in
14 places -- the offline drivers' ``--lead_in_s`` defaults, the base-Moshi / KAME / probe engines, the recipe's
``SpeechInference.lead_in_s``, ``track_persona_depth``'s depth base, and RL's TRAINING layout
(``rl/knowledge_segments.LEAD_IN_S``, ``ppx_policy`` / ``moshi_policy`` defaults). They agreed. A drifted copy would
either move every knowledge number (eval side) or silently train on a different input layout than the benchmark
measures (RL side). Now the package copies reference ``moshi_family.offline_cli.DEFAULT_LEAD_IN_S`` and the recipe
copies ``speech_inference.DEFAULT_LEAD_IN_S``; the few that must stay literal (isolated venvs, fork-era drivers,
RL files that running arms import) are listed in ``LITERAL_OK``.

Scanned (AST, no imports): every ``lead_in_s`` parameter default, every ``add_argument("--lead_in_s", default=)``,
every module-level ``*LEAD_IN_S`` constant, in ``speech_llm/full_duplex`` (vendored ``models/`` / ``third_party``
and tests excluded) and ``i6_experiments/users/dorian_koch`` (``speech_llm/`` + the top-level driver scripts).
Names are resolved through module constants and ``from ... import`` (relative and absolute).

  [1] every site in ``KNOWLEDGE_SITES`` resolves to the canonical value;
  [2] a registered site not in ``LITERAL_OK`` references a constant (no new literal copy of the number);
  [3] every OTHER site is clip mode (0.0: FDB / upstream "user speaks first") or None -- a new site at any other
      value is an unregistered copy;
  [4] no registered site has vanished (renamed? moved? update the registry).

CAN FAIL: the self-test applies each mutation to the REAL source in memory and requires a flag: RL's training
lead-in drifts to 1.5; ``moshi_engine`` goes back to a literal 2.0; the canonical value changes; a clip-mode default
(``personaplex/engine.reply``) becomes 2.0.

Run from the setup root, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_lead_in_default.py
"""

import ast
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
RECIPE = SETUP / "recipe"
FD = RECIPE / "speech_llm" / "full_duplex"
I6 = RECIPE / "i6_experiments" / "users" / "dorian_koch"
CANON = ("speech_llm/full_duplex/moshi_family/offline_cli.py", "DEFAULT_LEAD_IN_S")

#: Every copy of the knowledge lead-in. Key = "<path under recipe/>::<site>" (site = function qualname, a module
#: constant, or "--lead_in_s" for an argparse default).
KNOWLEDGE_SITES = {
    "speech_llm/full_duplex/moshi_family/offline_cli.py::DEFAULT_LEAD_IN_S",
    "speech_llm/full_duplex/moshi_family/offline_cli.py::--lead_in_s",
    "speech_llm/full_duplex/moshi_family/moshi_engine.py::run_pairs",
    "speech_llm/full_duplex/moshi_family/kame_engine.py::run_pairs",
    "speech_llm/full_duplex/moshi_family/knowledge_probe.py::run_knowledge_probe",
    "speech_llm/full_duplex/moshi_family/rl/knowledge_segments.py::LEAD_IN_S",
    "speech_llm/full_duplex/moshi_family/rl/ppx_policy.py::PpxRolloutGenerator.__init__",
    "speech_llm/full_duplex/moshi_family/rl/moshi_policy.py::MoshiRolloutGenerator.__init__",
    "speech_llm/full_duplex/flmaudio/offline_inference.py::--lead_in_s",
    "speech_llm/full_duplex/sis_recipe/doriank/track_persona_depth.py::BASE_LEAD_IN_S",
    "i6_experiments/users/dorian_koch/speech_llm/speech_inference.py::DEFAULT_LEAD_IN_S",
    "i6_experiments/users/dorian_koch/speech_llm/speech_inference.py::SpeechInference.__init__",
    "i6_experiments/users/dorian_koch/moshi_offline_inference.py::--lead_in_s",
    "i6_experiments/users/dorian_koch/personaplex_offline_inference.py::--lead_in_s",
}
#: Registered sites that may stay a literal, and why.
LITERAL_OK = {
    "speech_llm/full_duplex/moshi_family/offline_cli.py::DEFAULT_LEAD_IN_S": "the canonical definition",
    "i6_experiments/users/dorian_koch/speech_llm/speech_inference.py::DEFAULT_LEAD_IN_S": "the recipe-side copy: "
    "the manager does not import moshi_family (its __init__ pulls in torch models)",
    "speech_llm/full_duplex/flmaudio/offline_inference.py::--lead_in_s": "runs in flm_audio_venv; importing "
    "moshi_family there would load our model package into FLM's venv",
    "i6_experiments/users/dorian_koch/moshi_offline_inference.py::--lead_in_s": "fork-era driver, frozen",
    "i6_experiments/users/dorian_koch/personaplex_offline_inference.py::--lead_in_s": "fork-era driver, frozen",
    # RL: imported by running arms; pointed at the constant on branch audit8-rl (merge after the arms finish)
    "speech_llm/full_duplex/moshi_family/rl/knowledge_segments.py::LEAD_IN_S": "RL, see branch audit8-rl",
    "speech_llm/full_duplex/moshi_family/rl/ppx_policy.py::PpxRolloutGenerator.__init__": "RL, branch audit8-rl",
    "speech_llm/full_duplex/moshi_family/rl/moshi_policy.py::MoshiRolloutGenerator.__init__": "RL, branch audit8-rl",
}
CLIP_MODE = (0.0, None)


def _files() -> list[Path]:
    out = []
    for p in FD.rglob("*.py"):
        parts = set(p.relative_to(FD).parts)
        if parts & {"models", "third_party", "tests", "__pycache__"}:
            continue
        out.append(p)
    out += [p for p in (I6 / "speech_llm").glob("*.py")] + list(I6.glob("*.py"))
    return sorted(out)


def _rel(p: Path) -> str:
    return str(p.relative_to(RECIPE))


class Module:
    def __init__(self, rel: str, src: str):
        self.rel, self.tree = rel, ast.parse(src)
        self.consts: dict[str, ast.AST] = {}
        self.imports: dict[str, tuple[str, str]] = {}  # local name -> (module file rel, its name)
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.consts[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.consts[node.target.id] = node.value
            elif isinstance(node, ast.ImportFrom):
                target = self._module_file(node)
                for a in node.names:
                    if target:
                        self.imports[a.asname or a.name] = (target, a.name)

    def _module_file(self, node: ast.ImportFrom) -> str | None:
        mod = (node.module or "").split(".") if node.module else []
        if node.level:
            base = Path(self.rel).parent
            for _ in range(node.level - 1):
                base = base.parent
            cand = [base.joinpath(*mod).with_suffix(".py")] if mod else []
        else:
            cand = [Path(*mod).with_suffix(".py"), Path("speech_llm/full_duplex", *mod).with_suffix(".py")]
        return next((str(c) for c in cand if (RECIPE / c).is_file()), None)

    def sites(self) -> dict[str, ast.AST | None]:
        out: dict[str, ast.AST | None] = {}
        for name, v in self.consts.items():
            if name == "LEAD_IN_S" or name.endswith("_LEAD_IN_S"):
                out[name] = v

        def visit(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = f"{prefix}{ch.name}"
                    a = ch.args
                    pos = a.posonlyargs + a.args
                    pairs = list(zip(pos[len(pos) - len(a.defaults) :], a.defaults)) + list(
                        zip(a.kwonlyargs, a.kw_defaults)
                    )
                    for arg, d in pairs:
                        if arg.arg == "lead_in_s":
                            out[qn] = d
                    visit(ch, qn + ".")
                elif isinstance(ch, ast.ClassDef):
                    visit(ch, f"{prefix}{ch.name}.")
                else:
                    visit(ch, prefix)

        visit(self.tree, "")
        for n in ast.walk(self.tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value == "--lead_in_s"
            ):
                d = next((k.value for k in n.keywords if k.arg == "default"), None)
                out["--lead_in_s"] = d
        return out


def resolve(mods: dict[str, Module], rel: str, node, depth=0):
    """-> (value, is_literal). Raises on anything it cannot resolve statically."""
    if node is None:
        return None, True
    if isinstance(node, ast.Constant):
        return (float(node.value) if node.value is not None else None), True
    if isinstance(node, ast.Name) and depth < 8:
        m = mods[rel]
        if node.id in m.consts:
            return resolve(mods, rel, m.consts[node.id], depth + 1)[0], False
        if node.id in m.imports:
            src_rel, name = m.imports[node.id]
            if src_rel not in mods:
                mods[src_rel] = Module(src_rel, (RECIPE / src_rel).read_text())
            return resolve(mods, src_rel, mods[src_rel].consts[name], depth + 1)[0], False
    if isinstance(node, ast.BinOp) and depth < 8:  # e.g. BASE + depth -- not a default we track
        raise ValueError(f"unresolvable expression {ast.unparse(node)!r}")
    raise ValueError(f"unresolvable {ast.unparse(node)!r}")


def check(overrides: dict[str, str] | None = None) -> tuple[list[str], int]:
    """-> (problems, number of sites). ``overrides``: {path under recipe/: source} (the self-test's mutations)."""
    overrides = overrides or {}
    mods = {}
    for p in _files():
        r = _rel(p)
        mods[r] = Module(r, overrides.get(r, p.read_text()))
    canon_mod = mods[CANON[0]]
    canon = resolve(mods, CANON[0], canon_mod.consts[CANON[1]])[0]
    problems, seen = [], set()
    for r in list(mods):
        for site, node in mods[r].sites().items():
            key = f"{r}::{site}"
            try:
                val, literal = resolve(mods, r, node)
            except (ValueError, KeyError) as e:
                problems.append(f"{key}: {e}")
                continue
            if key in KNOWLEDGE_SITES:
                seen.add(key)
                if val != canon:
                    problems.append(f"{key}: lead-in {val} != DEFAULT_LEAD_IN_S {canon} -- a drifted copy")
                elif literal and key not in LITERAL_OK:
                    problems.append(f"{key}: a literal {val} -- reference DEFAULT_LEAD_IN_S instead")
            elif val not in CLIP_MODE:
                problems.append(
                    f"{key}: lead-in default {val} is neither clip mode (0 / None) nor a registered knowledge site "
                    f"-- if it is the knowledge lead-in, reference DEFAULT_LEAD_IN_S and add it to KNOWLEDGE_SITES"
                )
    for key in sorted(KNOWLEDGE_SITES - seen):
        problems.append(f"{key}: registered site not found (renamed or moved? update KNOWLEDGE_SITES)")
    return problems, len(seen)


#: (path under recipe/, old, new, what, expected substring)
MUTATIONS = [
    (
        "speech_llm/full_duplex/moshi_family/rl/knowledge_segments.py",
        ("LEAD_IN_S = 2.0", "LEAD_IN_S = DEFAULT_LEAD_IN_S"),  # main / after branch audit8-rl
        "LEAD_IN_S = 1.5",
        "RL training lead-in drifts from the benchmark's",
        "rl/knowledge_segments.py::LEAD_IN_S: lead-in 1.5",
    ),
    (
        "speech_llm/full_duplex/moshi_family/moshi_engine.py",
        "lead_in_s: float = DEFAULT_LEAD_IN_S,",
        "lead_in_s: float = 2.0,",
        "moshi_engine back to a literal copy",
        "moshi_engine.py::run_pairs: a literal",
    ),
    (
        "speech_llm/full_duplex/moshi_family/offline_cli.py",
        "DEFAULT_LEAD_IN_S = 2.0",
        "DEFAULT_LEAD_IN_S = 2.5",
        "the canonical value changes, the literal copies do not",
        "flmaudio/offline_inference.py::--lead_in_s: lead-in 2.0 != DEFAULT_LEAD_IN_S 2.5",
    ),
    (
        "speech_llm/full_duplex/moshi_family/personaplex/engine.py",
        "    lead_in_s: float = 0.0,",
        "    lead_in_s: float = 2.0,",
        "an unregistered new copy (a clip-mode default set to 2.0)",
        "nor a registered knowledge site",
    ),
]


def main():
    problems, n = check()
    failed = bool(problems)
    print(
        f"[{'ok' if not problems else 'FAIL'}] {n}/{len(KNOWLEDGE_SITES)} knowledge lead-in sites equal the canonical value; "
        f"every other lead_in_s default is clip mode"
    )
    for p in problems:
        print(f"   - {p}")
    for rel, old, new, what, expect in MUTATIONS:
        src = (RECIPE / rel).read_text()
        old = next((o for o in (old if isinstance(old, tuple) else (old,)) if src.count(o) == 1), old)
        assert isinstance(old, str), f"self-test: {rel} contains none of {old!r} exactly once -- update {what!r}"
        got = [p for p in check({rel: src.replace(old, new, 1)})[0] if expect in p]
        ok = bool(got)
        failed |= not ok
        print(f"[{'ok' if ok else 'FAIL'}] can fail: {what:58s} -> {(got or ['NOT flagged'])[0][-110:]}")
    if failed:
        sys.exit(1)
    print("\nthe knowledge lead-in is one number")


if __name__ == "__main__":
    main()
