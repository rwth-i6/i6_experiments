"""Guard: the live server's Doerg benchmark recording (2026-09-25).

The Doerg benchmark is a team member's own conversations with our models, recorded by the live server
(``hpc-talk.py --record``, always on under ``hpc-talk-discord.py``) and listed one line per conversation
in an append-only manifest. A conversation lost at the end of a session (``--stop``, ``--max_hours``) or
a manifest line pointing at missing files would silently shrink it. Checks, on the REAL
``LiveServer.handle_chat`` (fake model core, aiohttp test client, Opus via sphn):

  1. a finished conversation writes: the stereo file (input as fed | model), the mono raw input (as
     received) with ~ the audio sent, the json (session record, recv_log, per-frame arrays of one length
     == frames, kind string of r/u), and ONE manifest line carrying ``--session_meta`` + those paths;
  2. a server stop while a conversation is OPEN (the stop hooks ``shutdown`` calls, the client still
     connected) ends it cleanly: recorded, ``ended_by == "server stop"``, manifest line written;
  3. CAN FAIL: with the stop hooks removed (the pre-2026-09-25 behaviour, ``os._exit`` straight away),
     the open conversation is NOT recorded within the same wait;
  4. without ``doerg_manifest`` nothing is appended (hpc-talk without ``--record`` stays manifest-free);
  5. (2026-09-28) a scripted client's ``POST /api/client_info`` lands in ITS conversation's record as
     ``client`` and its ``via`` replaces the session's in the manifest line; a report posted during an
     earlier conversation does not leak into the next one.

moshi_family venv (torch, sphn, aiohttp, soundfile), CPU, a few seconds::

    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && PYTHONPATH=recipe/speech_llm/full_duplex:recipe \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_doerg_record.py'
"""

import asyncio
import json
import os
import sys
import tempfile

import numpy as np

from moshi_family.personaplex import live_server as ls

FAILS: list[str] = []
SR = 24000


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


class FakeCore:
    frame_size = 1920

    def begin_sync(self, voice, text_prompt):
        pass

    def step_pcm(self, chunk):
        return [(np.full(1920, 0.1, np.float32), "hi")]


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


async def conversation(
    td: str, manifest: str | None, stop_mid: str | None, client_info: dict | None = None, server=None
) -> tuple[ls.LiveServer, bool]:
    """One conversation of 3 s of audio. ``stop_mid``: None = the client closes; "hooks" = server stop
    while open; "mutant" = the same with the stop hooks removed. Returns (server, recorded within 5 s)."""
    import sphn
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    server = server or ls.LiveServer(
        FakeCore(),
        idle_min=5,
        max_hours=1,
        label="fake@1",
        record_dir=os.path.join(td, "rec"),
        ffmpeg=None,
        doerg_manifest=manifest,
        session_meta={"run": "fake", "step": 1, "via": "check"},
    )
    app = web.Application()
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_post("/api/client_info", server.handle_client_info)
    n_before = len(server.sessions)
    async with TestClient(TestServer(app)) as cl:
        ws = await cl.ws_connect("/api/chat?voice_prompt=NATF2.pt&text_prompt=hello")
        msg = await ws.receive()
        assert msg.data == b"\x00", msg
        w = sphn.OpusStreamWriter(SR)

        async def drain():  # a browser keeps reading (and so answers the server's close)
            async for _ in ws:
                pass

        reader = asyncio.create_task(drain())
        pcm = tone(3.0)
        for i in range(0, pcm.size, 480):
            pkt = w.append_pcm(pcm[i : i + 480])
            if pkt:
                await ws.send_bytes(b"\x01" + pkt)
            await asyncio.sleep(0.004)
        await asyncio.sleep(0.5)
        if client_info is not None:
            r = await cl.post("/api/client_info", data=json.dumps(client_info))
            assert r.status == 200, r.status
        if stop_mid is None:
            await ws.close()
        else:
            server.stopping = True
            if stop_mid == "mutant":
                server.stop_hooks.clear()
            for hook in list(server.stop_hooks):
                hook()
        recorded = False
        for _ in range(50):
            if len(server.sessions) > n_before:
                recorded = True
                break
            await asyncio.sleep(0.1)
        if not ws.closed:
            await ws.close()
        reader.cancel()
        for _ in range(50):  # let the handler finish before the test server goes away
            if server.active == 0:
                break
            await asyncio.sleep(0.1)
    return server, recorded


def lines(path: str) -> list[dict]:
    return [json.loads(x) for x in open(path)] if os.path.exists(path) else []


def main() -> int:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as td:
        man = os.path.join(td, "doerg", "sessions.jsonl")
        server, rec_ok = asyncio.run(conversation(td, man, None))
        check(rec_ok, "1a a closed conversation is recorded")
        L = lines(man)
        check(len(L) == 1, f"1b one manifest line ({len(L)})")
        if L:
            e = L[0]
            check(
                e.get("benchmark") == "doerg"
                and e.get("run") == "fake"
                and e.get("step") == 1
                and e.get("via") == "check",
                "1c the line carries --session_meta",
            )
            check(
                e.get("label") == "fake@1" and e.get("voice") == "NATF2.pt" and e.get("text_prompt") == "hello",
                "1d label, voice and prompt",
            )
            paths = [e.get("recording"), e.get("recording_raw_input"), e.get("recording_json")]
            check(all(p and os.path.exists(p) for p in paths), f"1e the recording paths exist: {paths}")
            if all(p and os.path.exists(p) for p in paths):
                st, sr = sf.read(e["recording"])
                raw, _ = sf.read(e["recording_raw_input"])
                check(st.ndim == 2 and st.shape[1] == 2 and sr == SR, "1f stereo at 24 kHz")
                check(abs(raw.size / SR - 3.0) < 0.2, f"1g raw input ~ the 3.0 s sent ({raw.size / SR:.2f} s)")
                doc = json.load(open(e["recording_json"]))
                pf = doc.get("per_frame", {})
                n = doc["session"]["frames"]
                check(
                    n > 0
                    and len(pf.get("kind", "")) == n
                    and len(pf.get("compute_ms", [])) == n
                    and set(pf.get("kind", "")) <= {"r", "u"},
                    f"1h per-frame arrays have one entry per frame ({n})",
                )
                check(len(doc.get("recv_log", {}).get("rows", [])) > 10, "1i packet arrival log present")
                check(len(doc.get("text", [])) == n, "1j text pieces by frame")

        man2 = os.path.join(td, "doerg2", "sessions.jsonl")
        server, rec_ok = asyncio.run(conversation(td, man2, "hooks"))
        L = lines(man2)
        check(rec_ok and len(L) == 1, "2a a server stop during an open conversation records it + manifest line")
        check(bool(L) and L[0]["record"].get("ended_by") == "server stop", "2b marked ended_by = server stop")

        man3 = os.path.join(td, "doerg3", "sessions.jsonl")
        server, rec_ok = asyncio.run(conversation(td, man3, "mutant"))
        check(not rec_ok, "3 CAN FAIL: without the stop hooks the open conversation is not recorded in time")

        server, rec_ok = asyncio.run(conversation(os.path.join(td, "x"), None, None))
        check(rec_ok and not os.path.exists(os.path.join(td, "x", "doerg")), "4 no manifest without doerg_manifest")

        man5 = os.path.join(td, "doerg5", "sessions.jsonl")
        info = {"via": "discord-voice", "discord_user_id": "123", "bridge": {"added_latency_ms": 250.0}}
        server, rec_ok = asyncio.run(conversation(td, man5, None, client_info=info))
        server, rec_ok2 = asyncio.run(conversation(td, man5, None, server=server))
        L = lines(man5)
        check(rec_ok and rec_ok2 and len(L) == 2, f"5a two conversations, two lines ({len(L)})")
        if len(L) == 2:
            check(L[0]["record"].get("client") == info, "5b client_info lands in its conversation's record")
            check(L[0].get("via") == "discord-voice", "5c its via replaces the session's in the manifest line")
            check(
                "client" not in L[1]["record"] and L[1].get("via") == "check",
                "5d the next conversation carries no stale client_info",
            )
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
