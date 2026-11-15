"""Guard: the live server serves any model family through the core contract (``moshi_family.live.cores``).

The server must not know which model it serves: frame length, voices and record fields come from the core. Checks on
the real ``LiveServer.handle_chat`` (fake cores, aiohttp test client, CPU) and the real family modules:

  1. FRAME LENGTH FROM THE CORE. A 133 ms core that yields its frame 0 together with frame 1 (Haru) and an 80 ms
     core with a 2-frame initial delay (Moshi) both record ``frame_s`` = their own frame length, and every text
     piece's ``out_frame`` is its position in the model's audio track (the fake cores name each piece after the
     track frame it belongs to). CAN FAIL: the model step index (``frame``) is NOT that position for either core.
  2. VOICES. An unknown ``jre:`` id and a voice the core refuses (``check_voice``) close the socket with the reason
     before any prompt runs; a known one starts the conversation.
  3. MAXIMUM LENGTH. A core that sets ``session_done`` ends the conversation; the record says ``ended_by``.
  4. PAGE. ``pt_voices=False`` drops NVIDIA's ``.pt`` voices from the page's voice list and keeps ours: every
     anchor on NVIDIA's real bundle found exactly once, and the rewritten list expression evaluates (node) to the
     JRE ids only. With ``pt_voices=True`` the list keeps them.
  5. FAMILIES. Every ``FAMILIES`` module imports and has ``add_live_args`` / ``build_live_core``; each parser accepts
     its own flags, and ``--family`` selects the flag set (``--guidance_scale`` is Haru's, not PersonaPlex's).
  6. THE CORE IS ONE LANE OF AN ADAPTER (``cores.SessionCore``, every family's core). A session that emits a frame
     before it hears (Haru) gives that frame together with the first reply, in track order; a session with an
     initial delay (Moshi) gives nothing until it emits; a session without pieces gets its text from
     ``adapter.piece`` (no text for PAD); ``info`` / ``done`` reach the server's record fields; every conversation
     is a new session. CAN FAIL: a core that drops the frames of ``start``.

moshi_family venv, CPU, seconds::

    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'CUDA_VISIBLE_DEVICES= PYTHONPATH=recipe/speech_llm/full_duplex:recipe \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_live_cores.py'
"""

import argparse
import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from moshi_family.live import cores
from moshi_family.live import server as ls

FAILS: list[str] = []
DIST = "/hpcwork/p0023999/common_hf_home/hub/models--nvidia--personaplex-7b-v1/snapshots/*/dist"


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class FakeSession:
    """A seam session of one lane: ``lead`` frames from ``start``, none for the first ``delay`` steps, ``done`` after
    ``max_steps``. Frame ``k`` of its track is filled with the value ``k`` and has text id ``k`` (id 3 = PAD)."""

    def __init__(self, adapter):
        self.a, self.k, self.steps = adapter, 0, 0
        self.info = {"family": "fake", "opened": adapter.opened}

    def _frames(self, n):
        from moshi_family.duplex import Frames

        fs = self.a.spec.frame_samples
        ids = np.arange(self.k, self.k + n, dtype=np.int64)[None]
        audio = np.repeat(ids.astype(np.float32), fs, axis=1)
        self.k += n
        pieces = [[f"<{i}>" for i in ids[0]]] if self.a.pieces else []
        return Frames(audio=audio, text_ids=ids, pieces=pieces)

    def start(self, voice=None, role=None, clips=None):
        return self._frames(self.a.lead)

    def step(self, user_pcm):
        assert user_pcm.shape == (1, self.a.spec.frame_samples), user_pcm.shape
        self.steps += 1
        return self._frames(0 if self.steps <= self.a.delay else 1)

    @property
    def done(self):
        return self.a.max_steps is not None and self.steps >= self.a.max_steps


class FakeAdapter:
    def __init__(self, lead=0, delay=0, pieces=False, max_steps=None):
        import types

        self.spec = types.SimpleNamespace(frame_samples=8, frame_s=8 / 24000, sample_rate=24000)
        self.lead, self.delay, self.pieces, self.max_steps, self.opened = lead, delay, pieces, max_steps, 0

    def open(self, batch=1):
        assert batch == 1
        self.opened += 1
        return FakeSession(self)

    def piece(self, token):
        return "" if token == 3 else f"p{token}"


def check_session_core():
    x = np.zeros(8, np.float32)

    def track(core, steps):
        core.begin_sync("v", "role")
        return [[(int(a[0]), len(a), t) for a, t in core.step_pcm(x)] for _ in range(steps)]

    haru = track(cores.SessionCore(FakeAdapter(lead=1, pieces=True)), 3)
    check(
        haru == [[(0, 8, "<0>"), (1, 8, "<1>")], [(2, 8, "<2>")], [(3, 8, "<3>")]],
        f"6a a session that speaks first: frame 0 comes with the first reply, in track order ({haru[0]})",
    )
    moshi = track(cores.SessionCore(FakeAdapter(delay=2)), 6)
    check(
        moshi[:2] == [[], []] and moshi[2:] == [[(0, 8, "p0")], [(1, 8, "p1")], [(2, 8, "p2")], [(3, 8, None)]],
        f"6b an initial delay gives nothing; text comes from adapter.piece, none for PAD ({moshi[2:]})",
    )
    core = cores.SessionCore(FakeAdapter(max_steps=2), default_voice="d", pt_voices=False)
    done = []
    for _ in range(2):  # two conversations
        core.begin_sync(None, None)
        done.append([core.session_done] + [(core.step_pcm(x), core.session_done)[1] for _ in range(2)])
    check(
        done == [[False, False, True]] * 2 and core.session_info == {"family": "fake", "opened": 2},
        f"6c done and info are the session's, and every conversation is a new session ({done}, {core.session_info})",
    )
    check(
        (core.frame_size, core.default_voice, core.pt_voices, core.jre_voices) == (8, "d", False, None),
        "6d frame size from the adapter's spec; the page's voice defaults from the family",
    )

    class DropsStart(cores.SessionCore):
        def begin_sync(self, voice, text_prompt):
            super().begin_sync(voice, text_prompt)
            self._first = None

    bad = track(DropsStart(FakeAdapter(lead=1, pieces=True)), 3)
    check(bad != haru, "6e CAN FAIL: a core that drops the frames of start loses frame 0")


class FakeCore:
    """``lead`` outputs on the first step (Haru: 2), none for the first ``delay`` steps (Moshi: its initial delay).
    Each output carries a text piece naming its index in the model's audio track."""

    def __init__(self, frame_size: int, lead: int = 1, delay: int = 0, max_steps: int | None = None):
        self.frame_size = frame_size
        self.lead, self.delay, self.max_steps = lead, delay, max_steps
        self.jre_voices = {"jre00": np.zeros((8, 4), np.int64)}
        self.voice_index = None
        self.session_done = False
        self.session_info = {}
        self.begun: list = []

    def check_voice(self, voice):
        if voice and voice.endswith(".pt"):
            raise ValueError(f"{voice} is a voice embedding; this model needs a clip")

    def begin_sync(self, voice, text_prompt):
        self.begun.append(voice)
        self.steps = self.n_out = 0
        self.session_done = False
        self.session_info = {"family": "fake"}

    def step_pcm(self, chunk):
        assert len(chunk) == self.frame_size, (len(chunk), self.frame_size)
        self.steps += 1
        n = 0 if self.steps <= self.delay else (self.lead if self.n_out == 0 else 1)
        out = []
        for _ in range(n):
            out.append((np.zeros(self.frame_size, np.float32), f"<{self.n_out}>"))
            self.n_out += 1
        if self.max_steps and self.steps >= self.max_steps:
            self.session_done = True
        return out


async def chat(td: str, core, voice: str | None, seconds: float = 1.6):
    import sphn
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    server = ls.LiveServer(
        core, idle_min=5, max_hours=1, label="fake", jitter="off", record_dir=os.path.join(td, "rec")
    )
    app = web.Application()
    app.router.add_get("/api/chat", server.handle_chat)
    closed = None
    async with TestClient(TestServer(app)) as cl:
        ws = await cl.ws_connect("/api/chat", params=({"voice_prompt": voice} if voice else {}))
        msg = await ws.receive()
        if msg.data != b"\x00":  # refused before the handshake: the close message carries the reason
            closed = (msg.extra or "") if isinstance(msg.extra, str) else str(msg.data)
            return server, closed, None
        w = sphn.OpusStreamWriter(24000)
        for _ in range(int(seconds / 0.02)):
            pkt = w.append_pcm(np.zeros(480, np.float32))
            if pkt:
                try:
                    await ws.send_bytes(b"\x01" + pkt)
                except Exception:
                    break
            await asyncio.sleep(0.02)
            if ws.closed:
                break
        await ws.close()
        for _ in range(80):
            if server.sessions:
                break
            await asyncio.sleep(0.1)
    recs = sorted(glob.glob(os.path.join(td, "rec", "*.json")))
    return server, closed, (json.load(open(recs[-1])) if recs else None)


def stamps_ok(doc: dict, key: str) -> bool:
    return bool(doc["text"]) and all(t["text"] == f"<{t[key]}>" for t in doc["text"])


def node_eval(expr: str):
    node = shutil.which("node")
    if node is None:
        return None
    r = subprocess.run([node, "-e", f"console.log(JSON.stringify({expr}))"], capture_output=True, text=True, timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 else f"node failed: {r.stderr[:200]}"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        for name, core in (
            ("133 ms, frame 0 with frame 1", FakeCore(3200, lead=2)),
            ("80 ms, 2-frame delay", FakeCore(1920, delay=2)),
        ):
            d = os.path.join(td, name[:3])
            _, _, doc = asyncio.run(chat(d, core, "jre:jre00"))
            ok = doc is not None and abs(doc["frame_s"] - core.frame_size / 24000) < 1e-9 and core.steps >= 5
            check(ok, f"1a {name}: frame_s {doc and round(doc['frame_s'], 4)} from the core, {core.steps} steps")
            check(
                doc is not None and stamps_ok(doc, "out_frame"),
                f"1b {name}: out_frame = the piece's frame in the model track",
            )
            check(
                doc is not None and not stamps_ok(doc, "frame"),
                f"1c CAN FAIL {name}: the model step index is not that frame",
            )

        core = FakeCore(1920)
        _, closed, _ = asyncio.run(chat(os.path.join(td, "v1"), core, "jre:nope"))
        check(
            closed is not None and "unknown JRE voice" in closed and not core.begun,
            f"2a unknown jre: id refused: {closed!r}",
        )
        _, closed, _ = asyncio.run(chat(os.path.join(td, "v2"), core, "NATF2.pt"))
        check(
            closed is not None and "voice embedding" in closed and not core.begun, f"2b check_voice refusal: {closed!r}"
        )
        _, closed, doc = asyncio.run(chat(os.path.join(td, "v3"), core, "jre:jre00", 0.6))
        check(closed is None and core.begun == ["jre:jre00"], "2c a known voice starts the conversation")

        core = FakeCore(1920, max_steps=6)
        server, _, doc = asyncio.run(chat(os.path.join(td, "m"), core, None, 3.0))
        rec = server.sessions[-1] if server.sessions else {}
        check(
            rec.get("ended_by") == "model maximum length" and rec.get("frames") == 6,
            f"3 session_done ends it: {rec.get('ended_by')!r} after {rec.get('frames')} frames",
        )

    bank = [{"id": "jre00", "name": "A"}, {"id": "jre01", "name": "B"}]
    dists = glob.glob(DIST)
    if dists:
        src = open(glob.glob(dists[0] + "/assets/index-*.js")[0]).read()
        for pt in (False, True):
            _, reps = ls._ui_extras(bank, None, "jre:jre00", None, None, pt)
            found = all(src.count(old) == n for old, _, n in reps)
            new = next(new for old, new, _ in reps if old == '"VARM4.pt"]')
            got = node_eval('["NATF0.pt", ' + new)
            want = ["jre:jre00", "jre:jre01"] if not pt else ["NATF0.pt", "VARM4.pt", "jre:jre00", "jre:jre01"]
            if got is None:
                got = want if (".filter(" in new) == (not pt) else "no node; list expression unexpected"
            check(found and got == want, f"4 pt_voices={pt}: {len(reps)} anchors found once each, voice list -> {got}")
    else:
        print("SKIP 4 (no NVIDIA dist in the HF cache)")

    for fam in sorted(cores.FAMILIES):
        m = cores.family_module(fam)
        p = argparse.ArgumentParser()
        ok = callable(getattr(m, "add_live_args", None)) and callable(getattr(m, "build_live_core", None))
        if ok:
            m.add_live_args(p)
            ok = p.parse_args([]).hf_repo is not None
        check(ok, f"5a family {fam}: module {m.__name__} has the two hooks and parses its defaults")
    try:
        with open(os.devnull, "w") as dn:
            old, sys.stderr = sys.stderr, dn
            try:
                cores.family_module("personaplex").add_live_args(p := argparse.ArgumentParser())
                p.parse_args(["--guidance_scale", "1.5"])
            finally:
                sys.stderr = old
        check(False, "5b --guidance_scale is not a PersonaPlex flag")
    except SystemExit:
        check(True, "5b --guidance_scale is not a PersonaPlex flag")
    cores.family_module("haru").add_live_args(p := argparse.ArgumentParser())
    check(p.parse_args(["--guidance_scale", "1.2"]).guidance_scale == 1.2, "5c --guidance_scale is a Haru flag")
    check_session_core()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
