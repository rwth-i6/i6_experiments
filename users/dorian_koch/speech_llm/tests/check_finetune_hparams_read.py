"""Guard: every hashed ``SpeechFinetune.hparams`` key is read by its adapter's renderer (or by the scheduler).

A key the renderer never reads still re-hashes the arm and configures nothing: "the knob that does nothing"
(``MOSHIRAG_LIB`` renders only ``loss_norm`` of the hparams ``train()`` emits; a base-Moshi arm ignores PersonaPlex-only
keys). RL has ``check_knobs_rendered`` for this; the SFT renderers had nothing. ``finetune.unread_hparams`` renders a
job's config with a dict that records reads. Checks:
  [1] the graph: no SpeechFinetune carries an unread key, except the arms already finished with one (``KNOWN``, frozen:
      never add to it; an arm there trained without that knob);
  [2] can fail: a base-Moshi arm with a PersonaPlex-only key is flagged; a key the renderer reads is not.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_finetune_hparams_read.py
"""

import logging
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.finetune import (  # noqa: E402
    MOSHI_LIB_ADAPTER,
    SpeechFinetune,
    unread_hparams,
)

#: job id (Class.hash) -> its unread keys, when this guard was written. Frozen.
KNOWN: dict[str, tuple] = {}

print("[2] can fail")
data = tk.Path("/dev/null/hparams_read_guard", hash_overwrite="hparams_read_guard")
probe = SpeechFinetune(
    train_data=data, adapter=MOSHI_LIB_ADAPTER, venv_python_path=data, max_steps=10, lora_rank=128,
    hparams={"lr": 1e-6, "per_gpu_batch": 2, "swap_channels": True},
)  # fmt: skip
got = unread_hparams(probe, MOSHI_LIB_ADAPTER)
assert got == ["swap_channels"], got
print(f"  ok   a base-Moshi arm with a PersonaPlex-only key is flagged ({got}); lr and per_gpu_batch are read")

print("[1] the graph")
config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
bad, n, by_adapter = {}, 0, {}
for j in tk.sis_graph.jobs():
    if not isinstance(j, SpeechFinetune):
        continue
    n += 1
    keys = tuple(unread_hparams(j, j.adapter))
    if not keys:
        continue
    jid = f"{type(j).__name__}.{os.path.basename(j._sis_path()).split('.')[-1]}"
    by_adapter.setdefault(j.adapter.name, set()).update(keys)
    if KNOWN.get(jid) != keys:
        bad[jid] = (j.adapter.name, keys, j.get_aliases() if hasattr(j, "get_aliases") else None)
for name, keys in sorted(by_adapter.items()):
    print(f"  {name}: unread keys seen {sorted(keys)}")
for jid, (ad, keys, al) in sorted(bad.items()):
    print(f"  FAIL {jid} [{ad}] {list(keys)} {sorted(al or [])[:1]}")
if bad:
    raise SystemExit(
        f"check_finetune_hparams_read: {len(bad)} of {n} SpeechFinetune job(s) carry hparams no renderer reads"
    )
print(f"check_finetune_hparams_read: OK ({n} SpeechFinetune jobs, {len(KNOWN)} known)")
