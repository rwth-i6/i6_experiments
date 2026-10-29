"""Guard: ``HeadTranscriptQuestions`` re-hashes when its script-written hypotheses file changes, and refuses a file
whose content differs from the declared hash.

The file (``analysis/ppx_user_text_head_eval.py``'s ``per_unit.jsonl``) is rewritten in place by a script, so its path
alone cannot tell two contents apart. Checks: [1] the pinned content hash keeps the hash of a job declared without
one; another content hash moves it. [2] ``run`` on a file whose sha256 differs from the declared one raises; on the
declared content it writes the questions. Can fail: [1] with the digest always popped, [2] without the run check.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_head_questions.py
"""

import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
logging.disable(logging.WARNING)

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.head_questions import (  # noqa: E402
    HeadTranscriptQuestions,
    file_sha256,
)

tmp = tempfile.mkdtemp(prefix="check_head_questions_")
hyps, qs = os.path.join(tmp, "per_unit.jsonl"), os.path.join(tmp, "questions.jsonl")
with open(hyps, "w") as f:
    f.write(json.dumps({"step": 7, "pop": "knowledge", "unit": "q0", "hyp": "who who wrote it"}) + "\n")
with open(qs, "w") as f:
    f.write(json.dumps({"index": 0, "question": "Who wrote it?"}) + "\n")

kw = dict(hypotheses=tk.Path(hyps), questions=tk.Path(qs), step=7)
plain = HeadTranscriptQuestions(**kw)
pinned = HeadTranscriptQuestions(**kw, hypotheses_sha256=HeadTranscriptQuestions.HYPOTHESES_PINNED_SHA256)
other = HeadTranscriptQuestions(**kw, hypotheses_sha256="0" * 64)
assert plain._sis_id() == pinned._sis_id(), "the pinned content hash moved the job hash"
assert plain._sis_id() != other._sis_id(), "another content hash keeps the job hash"
print("[1] the pinned content keeps the hash; another content moves it")

job = HeadTranscriptQuestions(**kw, hypotheses_sha256=file_sha256(hyps))
job.out_questions = tk.Path(os.path.join(tmp, "out_questions.jsonl"))
job.out_stats = tk.Path(os.path.join(tmp, "out_stats.json"))
job.run()
assert json.loads(open(job.out_questions.get_path()).readline())["question"] == "who wrote it"
bad = HeadTranscriptQuestions(**kw, hypotheses_sha256="0" * 64)
bad.out_questions, bad.out_stats = job.out_questions, job.out_stats
try:
    bad.run()
    raise SystemExit("run() accepted a hypotheses file whose sha256 differs from the declared one")
except RuntimeError:
    pass
print("[2] run checks the declared content hash")
print("check_head_questions: OK")
