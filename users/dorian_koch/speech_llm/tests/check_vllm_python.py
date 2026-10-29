"""Guard: every vLLM server runs from the vLLM 0.30 squashfs image, never silently from another interpreter.

Until 2026-10-27 ``common.vllm_python`` looked for a probe marker under the job's /hpcwork root (no ``output/`` there)
and fell back to the setup ``.venv``'s vLLM 0.19 for every job; Gemma 4 was also kept on 0.19 by name, and a failed
image boot retried on 0.19. Run from the setup root.
"""

import os, sys, tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
from i6_experiments.users.dorian_koch.speech_llm import common

IMAGE = os.path.join(os.getcwd(), common._VLLM_VENV_JOB, "output", "venv", "bin", "python")
assert os.path.exists(IMAGE), IMAGE
os.environ.pop("VLLM_PYTHON", None)
here = os.getcwd()

# a real job's cwd: inside the work dir under /hpcwork, whose root holds only work/
job_cwd = os.path.realpath(os.path.join(common._VLLM_VENV_JOB, ".."))
for cwd in (job_cwd, tempfile.mkdtemp(dir="/tmp")):  # ... and a copied job dir outside the setup (test node)
    os.chdir(cwd)
    for model in ("openai/gpt-oss-120b", "google/gemma-4-31B-it", None):
        got = common.vllm_python(model)
        assert os.path.realpath(got) == os.path.realpath(IMAGE), (cwd, model, got)
    print(f"[ok] image chosen from cwd {cwd}")
os.chdir(here)

# no image -> loud failure, not another interpreter
real = common._VLLM_VENV_JOB
common._VLLM_VENV_JOB = "work/no/such/CreateVenv.job"
try:
    got = common.vllm_python("openai/gpt-oss-120b")
except RuntimeError as e:
    print("[ok] missing image raises:", str(e)[:70], "...")
else:
    raise SystemExit(f"FAIL: missing image silently gave {got}")
finally:
    common._VLLM_VENV_JOB = real

# Gemma 4 thinking is off server-side
args = common._VLLM_MODEL_ARGS["google/gemma-4-31B-it"]
i = args.index("--default-chat-template-kwargs")
assert args[i + 1] == '{"enable_thinking": false}', args
print("[ok] gemma-4 server default enable_thinking=false")

# the recipe builds the image under the same name this module looks it up by (one constant, no second literal)
import re  # noqa: E402

venvs_src = open("recipe/speech_llm/full_duplex/sis_recipe/doriank/venvs.py").read()
assert not re.search(r"[\"']vllm_venv_v\d", venvs_src), "venvs.py spells the vLLM image name: use VLLM_VENV_HASH"
assert common._VLLM_VENV_JOB.endswith("CreateVenv." + common.VLLM_VENV_HASH), common._VLLM_VENV_JOB
print("[ok] vllm_venv and vllm_python share VLLM_VENV_HASH")
print("\nALL CHECKS PASSED")
