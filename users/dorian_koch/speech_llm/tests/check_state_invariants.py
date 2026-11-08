"""Guard: small state / invariant checks of the training, ingest and analysis code (each drives the real function).

  [1] ``train_loop.check_resume_world`` refuses a sidecar of another world size, accepts an old one without the key;
      the base-Moshi full-FT sidecar records ``world`` and its resume calls the check.
  [2] ``train_data_common.dialogue_frames``: the loss mask must be a contiguous suffix (the per-frame targets go to
      the last ``n`` frames).
  [3] ``collate_codes`` refuses a batch of mixed arities (the layout is read off sample 0).
  [4] ``UserTextStream.save``: a kill while writing leaves no stream that ``exists()``; a complete save loads.
  [5] ``evalstats.drop_relogged``: steps a resumed run logged again replace the earlier records.
  [6] ``podcast_ingest_main.next_part_index``: one past the highest part, never the count.
  [7] PersonaPlex training's prompt spacer = the inference spacer; the launcher refuses another one.
  [8] ``policies._policy_applies`` refuses an unknown epoch.
  [9] ``LMGen.load_voice_prompt_embeddings`` drops a stored-codes frame source (``selfplay.set_voice_codes``).

CAN FAIL: each check has a converse that the pre-fix behaviour fails (the count rule in [6], a mask with a hole in
[2], a mixed batch in [3], a kill before the last file in [4], a re-logged step in [5]).

Run from the setup root in moshi_family_venv (CPU, ~20 s):
    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/run_guards.py --only check_state_invariants
"""

import importlib.util
import json
import os
import pathlib
import re
import sys
import tempfile
import types

SETUP = os.getcwd()
LIB = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex")
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(LIB, "analysis_tools"))

import torch  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


print("[1] resume world size")
from moshi_family import train_loop  # noqa: E402

world = train_loop._world_size()
check("same world resumes", not raises(lambda: train_loop.check_resume_world({"world": world}, "x")))
check("an old sidecar (no key) resumes", not raises(lambda: train_loop.check_resume_world({}, "x")))
check(
    "another world is refused", raises(lambda: train_loop.check_resume_world({"world": world + 1}, "x"), RuntimeError)
)
src = open(os.path.join(LIB, "moshi_family", "moshi_finetune_launcher.py")).read()
check('the full-FT sidecar records "world"', '"world": world,' in src)
check("the full-FT resume calls check_resume_world", "check_resume_world(state, trainer_state_path(cdir))" in src)

print("[2] the loss mask is a contiguous suffix")
from moshi_family.train_data_common import USER_TEXT_IGNORE, collate_codes, dialogue_frames  # noqa: E402


def mask(bits):
    return torch.tensor(bits, dtype=torch.bool)


check("prompt zeros then ones -> n", dialogue_frames(mask([0, 0, 1, 1, 1])) == 3)
check("all ones -> T", dialogue_frames(mask([1, 1, 1])) == 3)
check(
    "a masked frame inside the dialogue is refused", raises(lambda: dialogue_frames(mask([0, 1, 0, 1, 1])), ValueError)
)
check("ones before the prompt are refused", raises(lambda: dialogue_frames(mask([1, 0, 0, 1])), ValueError))

print("[3] collate_codes: one arity per batch")
K = 17


def item(T, extra=None):
    out = (torch.zeros(K, T, dtype=torch.long), torch.ones(T, dtype=torch.bool))
    return out if extra is None else (*out, extra)


w = item(5, torch.full((5, 1), 0.3))
check("a uniform batch collates", len(collate_codes([w, item(4, torch.ones(4, 1))])) == 3)
check("a mixed batch (weights dropped) is refused", raises(lambda: collate_codes([item(5), w]), AssertionError))
u = item(5, torch.full((5,), USER_TEXT_IGNORE, dtype=torch.long))
check("a mixed long-extra batch is refused", raises(lambda: collate_codes([u, item(5)]), AssertionError))

print("[4] UserTextStream.save is atomic")
from moshi_family import user_text_stream as uts  # noqa: E402

dim = 8
s = uts.UserTextStream(torch.nn.Linear(dim, 11, bias=False), dim)
lm = types.SimpleNamespace(dim=dim, text_emb=torch.nn.Embedding(11, dim))
with tempfile.TemporaryDirectory() as d:
    real = pathlib.Path.write_text

    def dies_on_meta(self, *a, **k):  # a kill while the meta file is written
        if self.name.startswith(uts.META_FILE):
            raise KeyboardInterrupt("killed")
        return real(self, *a, **k)

    pathlib.Path.write_text = dies_on_meta
    try:
        raises(lambda: s.save(d, delay_frames=2, text_kwargs={}), KeyboardInterrupt)
    finally:
        pathlib.Path.write_text = real
    check("CAN FAIL: a kill while writing the meta leaves no stream that exists()", not uts.UserTextStream.exists(d))
    s.save(d, delay_frames=2, text_kwargs={"onset_floor": True})
    check("a complete save exists()", uts.UserTextStream.exists(d))
    check("no .tmp left behind", not [n for n in os.listdir(d) if n.endswith(".tmp")], os.listdir(d))
    _, meta = uts.UserTextStream.load(d, lm, device="cpu")
    check("and loads", meta["delay_frames"] == 2 and meta["text_kwargs"] == {"onset_floor": True}, meta)

print("[5] evalstats drops re-logged steps")
import evalstats  # noqa: E402

rows = [{"step": 1}, {"step": 2}, {"kind": "sample", "step": 2}, {"step": 3}, {"step": 2, "r": 1}, {"step": 3, "r": 1}]
got = evalstats.drop_relogged(rows)
steps = [(r["step"], r.get("r")) for r in got if "kind" not in r]
check("the resumed run's records replace the re-logged ones", steps == [(1, None), (2, 1), (3, 1)], steps)
check("another kind is untouched", [r for r in got if r.get("kind") == "sample"] == [{"kind": "sample", "step": 2}])
check("a monotone stream is unchanged", evalstats.drop_relogged(rows[:4]) == rows[:4])

print("[6] ingest part numbering")
from moshi_family.podcast_ingest_main import next_part_index  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    check("an empty dir starts at 0", next_part_index(d) == 0)
    for n in ("part_00000", "part_00002", "part_00003.tmp"):
        os.makedirs(os.path.join(d, n))
    finished = [n for n in os.listdir(d) if not n.endswith(".tmp")]
    check("one past the highest (a .tmp included)", next_part_index(d) == 4, next_part_index(d))
    check("CAN FAIL: the count would reuse part_00002", f"part_{len(finished):05d}" in os.listdir(d))

print("[7] the prompt spacer")
from moshi_family.personaplex import train_data as ptd  # noqa: E402
from moshi_family.personaplex.finetune_launcher import train_loader_args  # noqa: E402

for f in ("engine.py", "offline.py"):
    txt = open(os.path.join(LIB, "moshi_family", "personaplex", f)).read()
    m = re.findall(r"audio_silence_frame_cnt=int\(([\d.]+) \* (?:self\.)?mimi\.frame_rate\)", txt)
    check(f"{f} primes int(0.5 * 12.5) spacer frames", m == ["0.5"], m)
check(
    "= INFERENCE_SILENCE_FRAMES",
    ptd.INFERENCE_SILENCE_FRAMES == int(0.5 * 12.5) == ptd.PersonaPlexDataConfig().silence_frames,
)
check(
    "the launcher refuses another spacer",
    raises(lambda: train_loader_args({"train_data": "x", "silence_frames": 5}), ValueError),
)

print("[8] policy dates")
spec = importlib.util.spec_from_file_location("policies", os.path.join(LIB, "sis_recipe", "doriank", "policies.py"))
pol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pol)
check("a known epoch compares", pol._policy_applies(pol.POLICY_LATEST, pol.POLICY_2026_09_14))
check("CAN FAIL: an unknown epoch would apply every rule", "rl2-2026-11-06" > max(pol.KNOWN_POLICIES))
check(
    "... and is refused", raises(lambda: pol._policy_applies("rl2-2026-11-06", pol.POLICY_2026_09_14), AssertionError)
)

print("[9] a loaded voice replaces stored codes")
from moshi_family.personaplex.models.lm import LMGen  # noqa: E402

with tempfile.TemporaryDirectory() as d:
    pt = os.path.join(d, "v.pt")
    torch.save({"embeddings": torch.zeros(3, 1, 1, 4), "cache": torch.zeros(2)}, pt)
    gen = types.SimpleNamespace(lm_model=types.SimpleNamespace(device="cpu"))
    gen._encode_voice_prompt_frames = lambda _mimi: iter(())  # what set_voice_codes installs
    LMGen.load_voice_prompt_embeddings(gen, pt)
    check("the override is gone", "_encode_voice_prompt_frames" not in vars(gen))

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
