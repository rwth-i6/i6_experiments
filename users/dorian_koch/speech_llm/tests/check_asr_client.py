"""Guard: ``podcast_episode_main.AsrClient`` never returns another request's transcript and never waits forever.

The client speaks a one-line-JSON protocol with a long-lived ASR server. Before 2026-10-29: one stray line on the
server's stdout desynchronised it for good (every later call read the previous request's response, and when the audio
was already loaded, returned the previous channel's words as this one's); a hung server blocked to walltime; a dead
server failed every later episode after its GPU work. With fake servers (plain python, no GPU): [1] a junk line
raises and kills the server, and the next call starts a fresh one; [2] a response naming another ``out`` raises;
[3] no response within ``timeout_s`` raises; [4] a normal round trip returns the transcript. Can fail: [2] without
the out check (the mismatched response would be returned).

Run from the setup root: ``CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_asr_client.py``
"""

import importlib.util
import os
import sys
import tempfile

PATH = os.path.join(os.getcwd(), "recipe/speech_llm/full_duplex/moshi_family/podcast_episode_main.py")
spec = importlib.util.spec_from_file_location("pem", PATH)
pem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pem)
tmp = tempfile.mkdtemp(prefix="check_asr_client_")

SERVER = """
import json, sys, time
mode = sys.argv[sys.argv.index("--backend") + 1]
for line in sys.stdin:
    req = json.loads(line)
    if mode == "junk":
        sys.stdout.write("loading weights...\\n")
    if mode == "hang":
        time.sleep(60)
    out = req["out"] + ".other" if mode == "mismatch" else req["out"]
    with open(req["out"], "w") as f:
        json.dump({"words": [req["wav"]]}, f)
    sys.stdout.write(json.dumps({"ok": True, "out": out}) + "\\n")
    sys.stdout.flush()
"""
srv = os.path.join(tmp, "fake_server.py")
open(srv, "w").write(SERVER)


def client(mode, timeout_s=10.0):
    return pem.AsrClient(sys.executable, srv, backend=mode, model="m", batch_size=1, timeout_s=timeout_s)


def raises(c, wav):
    try:
        c.transcribe(wav, os.path.join(tmp, wav + ".json"))
        return False
    except pem.AsrError:
        return True


c = client("junk")
assert raises(c, "a"), "a junk line was accepted"
assert c.proc.poll() is not None, "the desynchronised server was left running"
assert raises(c, "b") and c.restarts == 1, "the next call did not start a fresh server"
c.close()
print("[1] a junk line raises and kills the server; the next call restarts it")
c = client("mismatch")
assert raises(c, "a"), "a response for another request was returned"
c.close()
print("[2] a response naming another out raises")
c = client("hang", timeout_s=1.0)
assert raises(c, "a"), "a hung server was waited on"
c.close()
print("[3] no response within timeout_s raises")
c = client("ok")
assert c.transcribe("a", os.path.join(tmp, "a.json")) == {"words": ["a"]}
c.close()
print("[4] a normal round trip returns the transcript")
print("check_asr_client: OK")
