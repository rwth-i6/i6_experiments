"""Guard: workers whose resume test is "the file exists" write that file atomically (temp name, then ``os.replace``).

A kill mid-write left a truncated file under its final name, and the resume then took it as done:
``seamless_download_main`` (a half-written FLAC with a header-only duration flowed into the RL codes) and Fisher's
``FisherSphToWav`` (a short ``B.wav``, zero-padded later). Source check on both writers (2026-10-29); can fail: the
pre-fix seamless writer is flagged.

Run from the setup root: ``python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_atomic_writes.py``
"""

import os
import re

SITES = {
    "recipe/speech_llm/full_duplex/moshi_family/rl/seamless_download_main.py": [
        r'str\(flac\) \+ "\.tmp"',
        r"os\.replace\(str\(flac\) \+ \"\.tmp\", flac\)",
        r"os\.replace\(tj,",
    ],
    "recipe/i6_experiments/users/dorian_koch/speech_llm/fisher_prep.py": [
        r'dst\.with_suffix\("\.tmp\.wav"\)',
        r"os\.replace\(tmp, dst\)",
    ],
}


def missing(src: str, patterns) -> list:
    return [p for p in patterns if not re.search(p, src)]


bad = {path: m for path, pats in SITES.items() if (m := missing(open(path).read(), pats))}
old = 'sf.write(\n    str(out / "audio" / f"{file_id}.flac"),\n'
assert missing(old, SITES["recipe/speech_llm/full_duplex/moshi_family/rl/seamless_download_main.py"]), "cannot fail"
print("  ok   can fail: the pre-fix seamless writer is flagged")
for path, m in bad.items():
    print(f"  FAIL {os.path.basename(path)}: no atomic write ({m})")
if bad:
    raise SystemExit("check_atomic_writes: a resumable writer writes its final file in place")
print("check_atomic_writes: OK")
