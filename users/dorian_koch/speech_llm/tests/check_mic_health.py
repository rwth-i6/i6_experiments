"""Guard: the live page's mic-health badge and the server's own view of the input level.

2026-09-25: the user's first live sessions failed because their mic clipped hard, and nothing on the page
said so. ``moshi_family/personaplex/mic_health.js`` (served by the patched client) measures the raw mic
samples before NVIDIA's Opus encoder and shows Clipping! / Too quiet / OK / No speech; the server records
``input_clip_share`` (decoded input at ``|x| >= 0.99``), serves the last 2 s at ``/api/stats`` ``live`` and
keeps the page's ``/api/client_mic`` report as ``client_mic``. Checks:

  1. JS CORE (node runs the real ``mic_health.js``; 48 kHz synthetic speech-like bursts fed in 128-sample
     worklet blocks): -24 dBFS speech -> OK and never Clipping!; the same +30 dB and clipped at +-1 ->
     Clipping!; -25 dB -> Too quiet; silence -> No speech; isolated full-scale single-sample spikes -> never
     Clipping! (the run rule); a run of 3 split across two worklet blocks counts; speech hard-limited at
     0.97 (what Chrome's AGC makes of a clipping mic) -> Clipping!.
  2. CAN FAIL. With ``clipRun = 1`` the spikes read Clipping!; with ``clipLevel = 0.99`` the 0.97-limited
     speech reads OK (why the page uses 0.95); with ``quietRmsDbfs = -60`` the quiet speech reads OK.
  3. SERVER, REAL CALLER. ``LiveServer.handle_chat`` (fake model core, aiohttp test client) receives Opus
     of clean vs clipped speech: the session record's ``input_clip_share`` is 0 vs > 1 %; ``/api/stats``
     ``live`` is present during the session with the clip share of the last 2 s; a ``/api/client_mic``
     POST lands in the record as ``client_mic``.
  4. PATCH. ``_patched_client`` on NVIDIA's real ``dist`` (HF cache) hooks ``MicHealth.attach`` into the
     Recorder's ``start()`` exactly once, serves the script byte-identical from ``<head>``; a bundle without
     the anchor raises instead of serving a page without the meter.

moshi_family venv (torch, sphn, aiohttp), CPU, ~10 s; needs ``node`` (login node: /usr/bin/node)::

    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && PYTHONPATH=recipe/speech_llm/full_duplex:recipe \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mic_health.py'
"""

import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from moshi_family.personaplex import live_server as ls

FAILS: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


# ---------------------------------------------------------------- 1 + 2: the JS core under node
JS_DRIVER = r"""
const mh = require(process.argv[2]);
const SR = 48000;
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
// speech-like: 300 ms bursts of a 150-900 Hz gliding tone + noise, 200 ms pauses; RMS of bursts ~= target
function speech(seconds, rmsDbfs) {
  const x = new Float32Array(Math.round(seconds * SR));
  const a = Math.pow(10, rmsDbfs / 20) * Math.SQRT2;
  let ph = 0;
  for (let i = 0; i < x.length; i++) {
    const t = i / SR, inBurst = (t % 0.5) < 0.3;
    const f = 150 + 750 * ((t * 1.7) % 1);
    ph += 2 * Math.PI * f / SR;
    x[i] = inBurst ? a * (0.9 * Math.sin(ph) + 0.1 * (rnd() * 2 - 1)) : 1e-4 * (rnd() * 2 - 1);
  }
  return x;
}
const scale = (x, db, lim) => x.map((v) => Math.max(-lim, Math.min(lim, v * Math.pow(10, db / 20))));
function spikes(seconds) {  // quiet speech with a single full-scale sample every 10 ms
  const x = speech(seconds, -30);
  for (let i = 0; i < x.length; i += 480) x[i] = 1.0;
  return x;
}
function run(x, cfg) {
  const c = Object.assign({}, mh.CFG, cfg || {});
  const push = mh.makeAccumulator(Math.round(SR * c.frameMs / 1000), c.clipLevel, c.clipRun);
  const w = new mh.HealthWindow(c);
  const states = {};
  let last = null;
  for (let i = 0; i < x.length; i += 128) {
    const fr = push(x.subarray(i, i + 128));
    if (fr.length) { last = w.push(fr, SR); if (w.t > 3) states[last.state] = (states[last.state] || 0) + 1; }
  }
  return { final: last.state, states: states, activeDbfs: last.activeDbfs, clipShare: last.clipShare, summary: w.summary() };
}
// a run of exactly 3 clipped samples straddling two 128-sample blocks
const acc = mh.makeAccumulator(960, 0.95, 3);
const b1 = new Float32Array(128), b2 = new Float32Array(832);
b1[126] = 1; b1[127] = 1; b2[0] = 1;
const split = acc(b1).length + acc(b2)[0].clip;

const clean = speech(12, -24);
const out = {
  cfg: mh.CFG,
  clean: run(clean),
  clipped: run(scale(clean, 30, 1.0)),
  limited: run(scale(clean, 30, 0.97)),
  quiet: run(scale(clean, -25, 1.0)),
  silence: run(new Float32Array(12 * SR)),
  spikes: run(spikes(12)),
  split_run_clip: split,
  // can-fail mutants
  spikes_run1: run(spikes(12), { clipRun: 1 }),
  limited_099: run(scale(clean, 30, 0.97), { clipLevel: 0.99 }),
  quiet_thr60: run(scale(clean, -25, 1.0), { quietRmsDbfs: -60 }),
};
console.log(JSON.stringify(out));
"""


def js_checks() -> None:
    node = shutil.which("node")
    check(node is not None, "node is available (login node: /usr/bin/node)")
    if node is None:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(JS_DRIVER)
    try:
        r = subprocess.run([node, f.name, str(ls.MIC_HEALTH_JS)], capture_output=True, text=True, timeout=120)
    finally:
        os.unlink(f.name)
    check(r.returncode == 0, f"node ran mic_health.js ({r.stderr.strip()[:300]})")
    if r.returncode:
        return
    o = json.loads(r.stdout)
    print("   thresholds:", json.dumps(o["cfg"]))
    for k in ("clean", "clipped", "limited", "quiet", "silence", "spikes", "spikes_run1", "limited_099", "quiet_thr60"):
        v = o[k]
        print(f"   {k:12s} final={v['final']:5s} states(after 3 s)={v['states']} active={v['activeDbfs']}")
    check(
        o["clean"]["final"] == "ok" and "clip" not in o["clean"]["states"], "1a -24 dBFS speech -> OK, never Clipping!"
    )
    check(
        o["clipped"]["final"] == "clip" and set(o["clipped"]["states"]) == {"clip"},
        "1b +30 dB clipped -> Clipping! throughout",
    )
    check(
        o["quiet"]["final"] == "quiet" and set(o["quiet"]["states"]) == {"quiet"}, "1c -25 dB -> Too quiet throughout"
    )
    check(o["silence"]["final"] == "idle", "1d silence -> No speech")
    check("clip" not in o["spikes"]["states"], "1e single-sample full-scale spikes -> never Clipping!")
    check(
        o["split_run_clip"] == 3, f"1f a 3-sample run split across worklet blocks counts 3 (got {o['split_run_clip']})"
    )
    check(o["limited"]["final"] == "clip", "1g speech hard-limited at 0.97 (Chrome AGC on a clipping mic) -> Clipping!")
    check("clip" in o["spikes_run1"]["states"], "2a CAN FAIL: clipRun=1 flags the spikes")
    check(o["limited_099"]["final"] != "clip", "2b CAN FAIL: clipLevel=0.99 misses the 0.97-limited clipping")
    check(o["quiet_thr60"]["final"] == "ok", "2c CAN FAIL: quietRmsDbfs=-60 calls the quiet speech OK")


# ---------------------------------------------------------------- 3: the server, real caller
def speech24(seconds: float, rms_dbfs: float, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f = 150 + 750 * ((t * 1.7) % 1)
    ph = np.cumsum(2 * np.pi * f / sr)
    burst = (t % 0.5) < 0.3
    rng = np.random.default_rng(0)
    x = 10 ** (rms_dbfs / 20) * np.sqrt(2) * (0.9 * np.sin(ph) + 0.1 * rng.uniform(-1, 1, t.size))
    return np.where(burst, x, 1e-4 * rng.uniform(-1, 1, t.size)).astype(np.float32)


class FakeCore:
    frame_size = 1920

    def begin_sync(self, voice, text_prompt):
        pass

    def step_pcm(self, chunk):
        return [(np.zeros(1920, np.float32), None)]


async def one_session(pcm: np.ndarray, post: dict | None) -> tuple[dict, dict | None]:
    import sphn
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    server = ls.LiveServer(FakeCore(), idle_min=5, max_hours=1, label="fake")
    app = web.Application()
    app.router.add_get("/api/chat", server.handle_chat)
    app.router.add_get("/api/stats", server.handle_stats)
    app.router.add_post("/api/client_mic", server.handle_client_mic)
    async with TestClient(TestServer(app)) as cl:
        ws = await cl.ws_connect("/api/chat")
        msg = await ws.receive()
        assert msg.data == b"\x00", msg
        w = sphn.OpusStreamWriter(24000)
        live = None
        for i in range(0, pcm.size, 480):  # 20 ms packets in real time-ish
            pkt = w.append_pcm(pcm[i : i + 480])
            if pkt:
                await ws.send_bytes(b"\x01" + pkt)
            await asyncio.sleep(0.004)
            if i == 480 * 150:
                live = (await (await cl.get("/api/stats")).json()).get("live")
                if post is not None:
                    await cl.post("/api/client_mic", data=json.dumps(post))
        await asyncio.sleep(0.5)
        await ws.close()
        for _ in range(50):
            if server.sessions:
                break
            await asyncio.sleep(0.1)
    return server.sessions[-1], live


def server_checks() -> None:
    clean = speech24(4.0, -24)
    clipped = np.clip(clean * 10 ** (30 / 20), -1, 1).astype(np.float32)
    rec_c, live_c = asyncio.run(
        one_session(clean, {"summary": {"clip_share": 0}, "encoded_stream": {"settings": {"autoGainControl": True}}})
    )
    rec_x, live_x = asyncio.run(one_session(clipped, None))
    print(
        f"   clean:   input_clip_share={rec_c.get('input_clip_share')} peak={rec_c.get('input_peak_dbfs')} live={live_c}"
    )
    print(
        f"   clipped: input_clip_share={rec_x.get('input_clip_share')} peak={rec_x.get('input_peak_dbfs')} live={live_x}"
    )
    check(rec_c.get("input_clip_share") == 0.0, "3a clean speech after Opus: input_clip_share == 0")
    check((rec_x.get("input_clip_share") or 0) > 0.01, "3b clipped speech after Opus: input_clip_share > 1 %")
    check(
        live_c is not None and live_c["clip_share"] == 0 and live_c["seconds"] >= 1.9,
        "3c /api/stats live (clean) covers ~2 s, no clipping",
    )
    check(live_x is not None and live_x["clip_share"] > 0.01, "3d /api/stats live (clipped) shows the clipping")
    check(
        (rec_c.get("client_mic") or {}).get("encoded_stream") == {"settings": {"autoGainControl": True}}
        and "client_mic" not in rec_x,
        "3e /api/client_mic lands in its own session's record only",
    )


# ---------------------------------------------------------------- 4: the client patch on NVIDIA's real dist
def patch_checks() -> None:
    hub = os.environ.get("HF_HUB_CACHE", "/hpcwork/p0023999/common_hf_home/hub")
    tgz = sorted(glob.glob(f"{hub}/models--nvidia--personaplex-7b-v1/snapshots/*/dist.tgz"))
    check(bool(tgz), f"NVIDIA dist.tgz in the HF cache ({hub})")
    if not tgz:
        return
    import tarfile

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tgz[-1]) as t:
            t.extractall(td)
        out = ls._patched_client(os.path.join(td, "dist"), 80, 600, 20)
        html = open(os.path.join(out, "index.html")).read()
        mh = glob.glob(os.path.join(out, "assets", "mic-health-*.js"))
        check(len(mh) == 1 and open(mh[0]).read() == ls.MIC_HEALTH_JS.read_text(), "4a script served byte-identical")
        name = os.path.basename(mh[0]) if mh else "?"
        head = html.split("</head>")[0]
        check(
            f'<script src="/assets/{name}"></script>' in head, "4b classic <script> in <head> (runs before the bundle)"
        )
        idx = [p for p in glob.glob(os.path.join(out, "assets", "index-*.js")) if "-j80-" in p]
        src = open(idx[0]).read() if idx else ""
        hook = ls._MIC_HOOK[1]
        i = src.find(hook)
        start = src.rfind("_.prototype.start = function", 0, i)
        stop = src.find("_.prototype.stop = function", max(start, 0))
        check(src.count(hook) == 1 and 0 <= start < i < stop, "4c attach() is called once, inside Recorder.start")
        # a bundle without the anchor must raise
        bad = os.path.join(td, "bad")
        shutil.copytree(os.path.join(td, "dist"), os.path.join(bad, "dist"))
        b_idx = [p for p in glob.glob(os.path.join(bad, "dist", "assets", "index-*.js"))][0]
        s = open(b_idx).read().replace(ls._MIC_HOOK[0], "b.sourceNode.connect(b.recordingGainNode) , ")
        open(b_idx, "w").write(s)
        try:
            ls._patched_client(os.path.join(bad, "dist"), 80, 600, 20)
            raised = False
        except RuntimeError as e:
            raised = "mic-health hook" in str(e)
        check(raised, "4d CAN FAIL: a bundle without the Recorder anchor raises")


def main() -> int:
    js_checks()
    server_checks()
    patch_checks()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
