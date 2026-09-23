"""Guard: every training launcher implements -- or explicitly opts out of -- each common setup piece.

Why (code-map §9, the 2026-09-23 code-sharing review): the five launchers (base Moshi, PersonaPlex,
MoshiRAG, Audex, RL) share ``train_loop`` but each re-implements the SETUP around it, each with a
different subset. A fix landed in one launcher and silently not in the others, four times over:

  * RL stepped AdamW on bf16 LoRA at lr 2e-7 -- no fp32 master weights (bug 1);
  * Audex's replay/curriculum branch called ``moshi_loss`` without ``text_pad_weight`` (bug 2);
  * PersonaPlex wrapped the LM in DDP and then called ``core.forward_train`` -- no grad sync (bug 3);
  * the 2026-09-16 resume fix (continue the data stream) was in the Moshi launcher only (bug 4);
  * the training fast path was first wired into one launcher only.

Each launcher is checked for each CONCERN below; it must implement it or name it in a module-level
``SETUP_OPT_OUTS = {concern: reason}``. An opt-out of something it implements is stale and fails too.

  fp32_master_weights   a ``TrainConfig``/``RLTrainerConfig`` call passes ``fp32_master_weights=``
                        (anything but a literal False)
  resume_data_position  a loader call passes ``skip_batches=<name bound to start_step * grad_accum>``,
                        AFTER the checkpoint scan and before the training loop starts
  ddp                   a ``DistributedDataParallel`` wrapper is built and ``loss_step`` calls it; may
                        be opted out of only by a launcher that never reads ``WORLD_SIZE``
  loss_norm             reads the ``loss_norm`` key and every duplex-loss call passes ``weighted_mean=``
  train_perf            calls ``train_perf.apply_train_perf`` (the training fast path)

and, with no opt-out: ``loss_step`` never calls a ``.forward*()`` method (it bypasses DDP's reducer
and FSDP's all-gather), and **every call site of one ``*_loss`` function passes the same keywords**
-- the rule that would have caught bug 2 (two ``moshi_loss`` calls, one without the knob) -- and (2026-09-23,
the PersonaPlex FSDP port) a launcher that shards its model (``shard_for_full_ft``) names its master dtype
explicitly (``master_dtype=``: float32 = fp32 master weights; None = bf16 sharded params, whose AdamW updates
round away at lr ~1e-6) and passes ``sharded_callbacks=`` to its ``TrainConfig`` (a rank-0-only save gathers
alone and deadlocks). That rule
covers the training objective, i.e. calls inside ``loss_step``; eval/diagnostic losses elsewhere may
use fixed yardstick weights on purpose (base Moshi's held-out losses run at the default pad weight).

Launchers are discovered as ``moshi_family/**/*launcher.py``, so a new one is checked the day it lands.

CAN FAIL: the self-test re-runs every rule on an in-memory mutation of the REAL launcher source that
re-introduces the bug the rule exists for (each old string asserted present), and requires a flag.

Run from the setup root, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_launcher_setup.py
"""

import ast
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
MOSHI_FAMILY = SETUP / "recipe" / "speech_llm" / "full_duplex" / "moshi_family"

CONCERNS = ("fp32_master_weights", "resume_data_position", "ddp", "loss_norm", "train_perf")
#: The duplex CE losses (text + audio streams); ``loss_norm`` is about their denominator.
DUPLEX_LOSSES = {"moshi_loss", "personaplex_loss", "moshirag_loss", "paper_weight_loss", "duplex_ce_loss"}
RESUME_CALLS = {"resume_from_checkpoint", "restore_trainer_state"}
EXPECTED_AT_LEAST = {
    "moshi_finetune_launcher.py",
    "moshirag_finetune_launcher.py",
    "personaplex/finetune_launcher.py",
    "audex/finetune_launcher.py",
    "rl/launcher.py",
}


def _name(func) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _kw(call: ast.Call) -> dict:
    return {k.arg: k.value for k in call.keywords if k.arg is not None}


def _reads_key(tree, key: str) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and _name(n.func) == "get" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and a.value == key:
                return True
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value == key:
            return True
    return False


def _is_steps_times_accum(node) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            names = {getattr(n.left, "id", None), getattr(n.right, "id", None)}
            if names == {"start_step", "grad_accum"}:
                return True
    return False


def analyse(src: str) -> tuple[dict, list[str], dict]:
    """-> (implemented {concern: bool}, problems that no opt-out excuses, SETUP_OPT_OUTS)."""
    tree = ast.parse(src)
    problems: list[str] = []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    opt_outs = None
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
        if isinstance(target, ast.Name) and target.id == "SETUP_OPT_OUTS":
            opt_outs = ast.literal_eval(node.value)
    if opt_outs is None:
        problems.append("no module-level SETUP_OPT_OUTS (declare {} if it implements everything)")
        opt_outs = {}
    for k, why in opt_outs.items():
        if k not in CONCERNS:
            problems.append(f"SETUP_OPT_OUTS names unknown concern {k!r} (known: {CONCERNS})")
        if not (isinstance(why, str) and why.strip()):
            problems.append(f"SETUP_OPT_OUTS[{k!r}] gives no reason")

    impl = {}
    # fp32_master_weights
    impl["fp32_master_weights"] = any(
        "fp32_master_weights" in _kw(c)
        and not (
            isinstance(_kw(c)["fp32_master_weights"], ast.Constant) and _kw(c)["fp32_master_weights"].value is False
        )
        for c in calls
    )

    # resume_data_position
    skip_names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and _is_steps_times_accum(n.value):
            skip_names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    resume_line = min((c.lineno for c in calls if _name(c.func) in RESUME_CALLS), default=None)
    train_line = min((c.lineno for c in calls if _name(c.func) == "run_training"), default=None)
    seek_lines = [
        c.lineno
        for c in calls
        if isinstance(_kw(c).get("skip_batches"), ast.Name) and _kw(c)["skip_batches"].id in skip_names
    ]
    impl["resume_data_position"] = bool(seek_lines)
    for ln in seek_lines:
        if resume_line is None or ln < resume_line:
            problems.append(f"line {ln}: the loader seeks BEFORE the checkpoint scan -- start_step is not known yet")
        if train_line is not None and ln > train_line:
            problems.append(f"line {ln}: the loader seeks after the training loop started")

    # ddp + the forward rule
    ddp_names = set()
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and _name(n.value.func) == "DistributedDataParallel"
        ):
            ddp_names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    loss_steps = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "loss_step"]
    ls_calls = [c for fn in loss_steps for c in ast.walk(fn) if isinstance(c, ast.Call)]
    for c in ls_calls:
        if isinstance(c.func, ast.Attribute) and c.func.attr.startswith("forward"):
            problems.append(
                f"line {c.lineno}: loss_step calls .{c.func.attr}() -- that bypasses DDP's reducer (no grad "
                f"sync) and FSDP's all-gather; call the (wrapped) module instead"
            )
    calls_wrapper = any(isinstance(c.func, ast.Name) and c.func.id in ddp_names for c in ls_calls)
    impl["ddp"] = bool(ddp_names) and calls_wrapper
    reads_world = any(isinstance(n, ast.Constant) and n.value == "WORLD_SIZE" for n in ast.walk(tree))
    if ddp_names and loss_steps and not calls_wrapper:
        problems.append(f"builds DistributedDataParallel ({sorted(ddp_names)}) but loss_step never calls it")
    if reads_world and not impl["ddp"]:
        problems.append("reads WORLD_SIZE (can run multi-process) but does not train through a DDP wrapper")

    # loss_norm
    duplex = [c for c in calls if _name(c.func) in DUPLEX_LOSSES]
    reads_norm = _reads_key(tree, "loss_norm")
    impl["loss_norm"] = reads_norm and bool(duplex) and all("weighted_mean" in _kw(c) for c in duplex)
    if reads_norm and not impl["loss_norm"]:
        missing = [c.lineno for c in duplex if "weighted_mean" not in _kw(c)]
        problems.append(f"reads loss_norm but the duplex loss call(s) at line(s) {missing} ignore it")

    # train_perf
    impl["train_perf"] = any(_name(c.func) == "apply_train_perf" for c in calls)

    # sharded (FSDP2) full finetune: an explicit master dtype, and collective callbacks
    shard_calls = [c for c in calls if _name(c.func) == "shard_for_full_ft"]
    for c in shard_calls:
        if "master_dtype" not in _kw(c):
            problems.append(
                f"line {c.lineno}: shard_for_full_ft(...) without an explicit master_dtype= -- say whether the "
                f"sharded params are fp32 masters (bf16 AdamW updates round away at lr ~1e-6)"
            )
    tcfg = [c for c in calls if _name(c.func) == "TrainConfig"]
    if shard_calls and not any(
        "sharded_callbacks" in _kw(c)
        and not (isinstance(_kw(c)["sharded_callbacks"], ast.Constant) and _kw(c)["sharded_callbacks"].value is False)
        for c in tcfg
    ):
        problems.append(
            "shards the model (shard_for_full_ft) but no TrainConfig gets sharded_callbacks= -- the save gather "
            "would run on rank 0 alone and deadlock"
        )

    # every call site of one *_loss function inside loss_step (the TRAINING objective) passes the same
    # keywords. Eval/diagnostic losses elsewhere may use fixed yardstick weights on purpose.
    by_fn: dict[str, list[ast.Call]] = {}
    for c in ls_calls:
        if _name(c.func).endswith("_loss"):
            by_fn.setdefault(_name(c.func), []).append(c)
    for fn, cs in by_fn.items():
        for c in cs:
            if any(k.arg is None for k in c.keywords):
                problems.append(f"line {c.lineno}: {fn}(**...) -- spell the loss keywords out so they can be compared")
        sets = {c.lineno: set(_kw(c)) for c in cs}
        union = set().union(*sets.values())
        for ln, s in sets.items():
            if s != union:
                problems.append(
                    f"line {ln}: {fn}(...) lacks {sorted(union - s)} that its other call site(s) pass -- that "
                    f"branch trains at the function's default"
                )
    return impl, problems, opt_outs


def verdict(src: str) -> list[str]:
    impl, problems, opt_outs = analyse(src)
    out = list(problems)
    for c in CONCERNS:
        if impl[c] and c in opt_outs:
            out.append(f"{c}: implemented AND opted out -- the opt-out is stale, remove it")
        elif not impl[c] and c not in opt_outs:
            out.append(f"{c}: neither implemented nor listed in SETUP_OPT_OUTS")
    return out


#: (launcher, old, new, what it re-introduces). Each must make verdict() non-empty.
MUTATIONS = [
    (
        "rl/launcher.py",
        "            fp32_master_weights=fp32_master_weights,\n",
        "",
        "bug 1: RL without fp32 master weights",
        "fp32_master_weights: neither",
    ),
    (
        "audex/finetune_launcher.py",
        "                text_pad_weight=full_stage1_text_pad_weight,\n",
        "",
        "bug 2: the full_stage1 moshi_loss call drops text_pad_weight",
        "lacks ['text_pad_weight']",
    ),
    (
        "personaplex/finetune_launcher.py",
        "            out = model(codes)\n",
        "            out = core.forward_train(codes)\n",
        "bug 3: PersonaPlex loss_step calls the unwrapped core",
        "calls .forward_train()",
    ),
    (
        "moshirag_finetune_launcher.py",
        "loader = _make_loader(skip_batches=_skip)",
        "loader = _make_loader()",
        "bug 4: MoshiRAG restarts its data stream on resume",
        "resume_data_position: neither",
    ),
    (
        "moshirag_finetune_launcher.py",
        ', weighted_mean=loss_norm == "weight")',
        ")",
        "bug 5: MoshiRAG reads loss_norm and ignores it",
        "reads loss_norm but",
    ),
    (
        "moshi_finetune_launcher.py",
        "apply_train_perf(",
        "_not_the_fast_path(",
        "the fast path missing from a launcher",
        "train_perf: neither",
    ),
    (
        "rl/launcher.py",
        "SETUP_OPT_OUTS: dict[str, str] = {\n",
        'SETUP_OPT_OUTS: dict[str, str] = {\n    "fp32_master_weights": "stale",\n',
        "a stale opt-out of an implemented concern",
        "opt-out is stale",
    ),
    (
        "personaplex/finetune_launcher.py",
        "SETUP_OPT_OUTS: dict[str, str] = {}\n",
        "",
        "a launcher without SETUP_OPT_OUTS",
        "no module-level SETUP_OPT_OUTS",
    ),
    (
        "personaplex/finetune_launcher.py",
        ", master_dtype=torch.float32)",
        ")",
        "PersonaPlex FSDP without fp32 master weights",
        "without an explicit master_dtype",
    ),
    (
        "personaplex/finetune_launcher.py",
        "            sharded_callbacks=fsdp,\n",
        "",
        "PersonaPlex FSDP with rank-0-only callbacks",
        "no TrainConfig gets sharded_callbacks",
    ),
]


def main():
    launchers = sorted(p for p in MOSHI_FAMILY.glob("**/*launcher.py") if "__pycache__" not in p.parts)
    rel = {str(p.relative_to(MOSHI_FAMILY)): p for p in launchers}
    missing = EXPECTED_AT_LEAST - set(rel)
    assert not missing, f"launchers not found (moved? renamed?): {sorted(missing)}"
    failed = False
    for r, p in rel.items():
        issues = verdict(p.read_text())
        impl, _, opt_outs = analyse(p.read_text())
        summary = ", ".join(f"{c}={'yes' if impl[c] else 'opt-out'}" for c in CONCERNS)
        if issues:
            failed = True
            print(f"FAIL {r}")
            for i in issues:
                print(f"       - {i}")
        else:
            print(f"[ok] {r:36s} {summary}")
    print()
    for r, old, new, what, expect in MUTATIONS:
        src = rel[r].read_text()
        assert old in src, f"self-test: {r} no longer contains {old!r} -- update the mutation for {what!r}"
        flagged = [f for f in verdict(src.replace(old, new, 1)) if expect in f]
        if flagged:
            print(f"[ok] can fail: {what:58s} -> {flagged[0][:90]}")
        else:
            failed = True
            print(f"FAIL self-test: {what} ({r}) is NOT flagged")
    if failed:
        sys.exit(1)
    print(f"\nall {len(rel)} launchers implement or explicitly opt out of {len(CONCERNS)} setup concerns")


if __name__ == "__main__":
    main()
