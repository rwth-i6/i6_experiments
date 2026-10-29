"""Guard: data-job knobs that are hashed but did not reach the worker (2026-10-29 audit) stay harmless.

  [1] ``HfToDialogue.seed_offset`` is hashed but unused unless ``seed_offset_effective``: that flag is not hashed at
      its default (every existing corpus keeps its id), moves the hash when set, and ``run`` then passes the offset to
      the generator (source check on the ``map`` call);
  [2] ``PodcastMimiIngest.download_workers`` (how fast, not what) does not reach the hash;
  [3] ``SeamlessSegmentEncodeJob`` refuses a ``sample_rate`` its encoder never receives.
Can fail: [1] with the flag always hashed or never forwarded, [2] with ``download_workers`` hashed, [3] without the
assert.

Run from the setup root: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_data_job_knobs.py``
"""

import inspect
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.hf_to_dialogue import HfToDialogue  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import PodcastMimiIngest  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.seamless_jobs import SeamlessSegmentEncodeJob  # noqa: E402

p = tk.Path("/dev/null/knob_guard", hash_overwrite="knob_guard")
kw = dict(dataset_split_path=p, llm_name="m", shard=1, num_shards=4, seed_offset=1)
a, b, c = (
    HfToDialogue(**kw),
    HfToDialogue(**kw, seed_offset_effective=False),
    HfToDialogue(**kw, seed_offset_effective=True),
)
assert a._sis_id() == b._sis_id(), "seed_offset_effective=False moved the hash"
assert a._sis_id() != c._sis_id(), "seed_offset_effective=True kept the hash"
src = inspect.getsource(HfToDialogue.run)
assert 'fn_kwargs = {"seed_offset": self.seed_offset} if self.seed_offset_effective else None' in src, src[:200]
assert "dataset.map(gen_fn, num_proc=32, fn_kwargs=fn_kwargs)" in src
print("[1] seed_offset: hashed only when effective, then forwarded to the generator")

ik = dict(venv_python_path=p, index_dir=p, shard_idx=0)
assert PodcastMimiIngest(**ik)._sis_id() == PodcastMimiIngest(**ik, download_workers=32)._sis_id()
print("[2] PodcastMimiIngest.download_workers does not reach the hash")

try:
    SeamlessSegmentEncodeJob(venv_python_path=p, audio_dir=p, sample_rate=16000)
    raise SystemExit("SeamlessSegmentEncodeJob accepted sample_rate=16000")
except AssertionError:
    pass
SeamlessSegmentEncodeJob(venv_python_path=p, audio_dir=p)
print("[3] SeamlessSegmentEncodeJob refuses a sample_rate it never forwards")
print("check_data_job_knobs: OK")
