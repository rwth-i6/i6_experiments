"""Guard: random voices of the live-mic server (``voice=random`` in !talk / !voice, 2026-09-28).

A random voice must be REPLAYABLE: the Doerg record has to name the exact stored clip, and the same spec + seed
must give the same clip. Checks on the real ``VoiceIndex`` and the real ``LiveServer.handle_chat`` (fake core,
aiohttp test client):

  1. a draw is a pure function of (pool, seed): same seed -> same clip, 40 seeds -> several clips, ``@seed`` in
     the spec wins over the argument, every drawn clip is in its pool; unknown pool raises; no seed -> a seed is
     chosen AND recorded (redrawing with it gives the same clip);
  2. ``clip_codes`` returns the clip trimmed to its ``n_frames`` (int64), ``LiveCore.jre_codes`` routes ``clip:``;
  3. a chat with ``voice_prompt=random:jre@7`` primes the core with the drawn clip id, and the session record +
     manifest line carry ``voice`` = that clip, ``voice_draw`` (spec with seed) and ``voice_info`` (episode, label,
     source, row); ``/api/voice_draw`` previews the SAME clip;
  4. the page patch lists a drawn ``clip:`` voice (``extra_voices``) and preselects it: every anchor on NVIDIA's
     real bundle found exactly once;
  5. CAN FAIL: a draw that ignores the seed (``random.choice``) fails check 1.

moshi_family venv, CPU, seconds::

    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'CUDA_VISIBLE_DEVICES= PYTHONPATH=recipe/speech_llm/full_duplex:recipe \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_live_voice_draw.py'
"""

import asyncio
import glob
import json
import os
import random
import sys
import tempfile

import numpy as np

from moshi_family.personaplex import live_server as ls
from moshi_family.personaplex import voice_index as vx

FAILS: list[str] = []
DIST = "/hpcwork/p0023999/common_hf_home/hub/models--nvidia--personaplex-7b-v1/snapshots/*/dist"


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def tiny_index(td: str) -> str:
    meta, codes = [], np.zeros((5, 8, 50), np.int16)
    for k, (pool, eid, lab, n) in enumerate(
        [
            ("jre", "e1", "SPEAKER_00", 50),
            ("jre", "e1", "SPEAKER_01", 34),
            ("jre", "e2", "SPEAKER_00", 50),
            ("jre", "e3", "SPEAKER_02", 42),
            ("know", None, "v6_tts", 50),
        ]
    ):
        codes[k, :, :n] = k + 1
        cid = f"clip:{pool}/{eid}/{lab}" if eid else "clip:know/v6_tts"
        meta.append(
            {
                "id": cid,
                "pool": pool,
                "source": "/x/" + pool,
                "row": k,
                "episode_id": eid,
                "label": lab,
                "title": f"#{k} - Guest",
                "split": "train",
                "n_frames": n,
            }
        )
    p = os.path.join(td, "voice_index.npz")
    np.savez(p, codes=codes, meta=np.array(json.dumps(meta)))
    return p


def draws_ok(vi) -> bool:
    a = [vi.draw("random:jre", s)["voice"] for s in range(40)]
    b = [vi.draw("random:jre", s)["voice"] for s in range(40)]
    return a == b and len(set(a)) > 1 and set(a) <= set(vi.pools["jre"])


class FakeCore:
    frame_size = 1920

    def __init__(self, vi):
        self.voice_index = vi
        self.voices = []

    def begin_sync(self, voice, text_prompt):
        self.voices.append(voice)

    def step_pcm(self, chunk):
        return [(np.zeros(1920, np.float32), None)]


async def chat(td, vi, voice):
    import sphn
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    core = FakeCore(vi)
    man = os.path.join(td, "m", "sessions.jsonl")
    server = ls.LiveServer(
        core,
        idle_min=5,
        max_hours=1,
        label="fake",
        record_dir=os.path.join(td, "rec"),
        doerg_manifest=man,
        session_meta={"via": "check"},
    )
    app = web.Application()
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_get("/api/voice_draw", server.handle_voice_draw)
    async with TestClient(TestServer(app)) as cl:
        r = await cl.get("/api/voice_draw", params={"spec": voice})
        preview = await r.json()
        ws = await cl.ws_connect("/api/chat", params={"voice_prompt": voice, "text_prompt": "hi"})
        msg = await ws.receive()
        assert msg.data == b"\x00", msg
        w = sphn.OpusStreamWriter(24000)
        for _ in range(25):
            pkt = w.append_pcm(np.zeros(480, np.float32))
            if pkt:
                await ws.send_bytes(b"\x01" + pkt)
            await asyncio.sleep(0.02)
        await ws.close()
        for _ in range(50):
            if server.sessions:
                break
            await asyncio.sleep(0.1)
    lines = [json.loads(x) for x in open(man)] if os.path.exists(man) else []
    return core, server, preview, lines


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        vi = vx.VoiceIndex(tiny_index(td), nat_voices=["NATF2.pt", "VARM0.pt"])
        check(draws_ok(vi), "1a same seed -> same clip; 40 seeds -> several clips, all in the pool")
        check(vi.draw("random:jre@5", 99)["voice"] == vi.draw("random:jre", 5)["voice"], "1b @seed in the spec wins")
        d = vi.draw("random:know")
        check(d["voice"] == "clip:know/v6_tts" and d["pool_size"] == 1, "1c know pool")
        check(vi.draw("random:nat", 3)["voice"] in ("NATF2.pt", "VARM0.pt"), "1d nat pool = the .pt voices")
        d = vi.draw("random")
        check(
            d["spec"].startswith("random:jre@") and vi.draw(d["spec"])["voice"] == d["voice"],
            "1e no seed: one is chosen, recorded in the spec, and replays",
        )
        try:
            vi.draw("random:nope")
            check(False, "1f unknown pool raises")
        except ValueError:
            check(True, "1f unknown pool raises")
        c = vi.clip_codes("clip:jre/e1/SPEAKER_01")
        check(c.shape == (8, 34) and c.dtype == np.int64 and (c == 2).all(), "2a clip codes trimmed to n_frames")
        core = ls.LiveCore(object(), object(), None, None, "cpu", voice_index=vi)
        check(
            np.array_equal(core.jre_codes("clip:jre/e2/SPEAKER_00"), vi.clip_codes("clip:jre/e2/SPEAKER_00")),
            "2b LiveCore.jre_codes routes clip: voices to the index",
        )

        fc, server, preview, lines = asyncio.run(chat(td, vi, "random:jre@7"))
        want = vi.draw("random:jre", 7)["voice"]
        rec = server.sessions[-1] if server.sessions else {}
        check(fc.voices == [want], f"3a the core is primed with the drawn clip ({fc.voices} vs {want})")
        check(
            rec.get("voice") == want and rec.get("voice_draw", {}).get("spec") == "random:jre@7",
            "3b record: voice = the clip, voice_draw.spec with the seed",
        )
        vinfo = rec.get("voice_info") or {}
        check(
            vinfo.get("episode_id") and vinfo.get("label") and vinfo.get("source") and "row" in vinfo,
            "3c record: voice_info names corpus, row, episode, speaker",
        )
        check(
            bool(lines) and lines[-1].get("voice") == want and lines[-1]["record"].get("voice_draw"),
            "3d manifest line: voice + voice_draw",
        )
        check(preview.get("voice") == want, "3e /api/voice_draw previews the same clip")

        dists = glob.glob(DIST)
        if dists:
            src = open(glob.glob(dists[0] + "/assets/index-*.js")[0]).read()
            _, reps = ls._ui_extras(None, None, want, "hi", {want: "Drawn · x"})
            check(
                all(src.count(old) == n for old, _, n in reps) and any(want in new for _, new, _ in reps),
                f"4 page patch: drawn clip listed + preselected, {len(reps)} anchors found once each",
            )
        else:
            print("SKIP 4 (no NVIDIA dist in the HF cache)")

        orig = vx.VoiceIndex.draw

        def mutant(self, spec, seed=None):
            d = orig(self, spec, seed)
            d["voice"] = random.choice(self.pools[d["pool"]])
            return d

        vx.VoiceIndex.draw = mutant
        ok = not draws_ok(vi)
        vx.VoiceIndex.draw = orig
        check(ok, "5 CAN FAIL: an unseeded draw fails check 1a")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
