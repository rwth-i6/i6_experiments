"""Guard: in every multi-clip inference engine, clip N on a reused instance == clip N on a fresh one.

The bug class (PersonaPlex, 2026-08-01 -> 09-22): an engine loads its model ONCE and streams many
clips through it. Any per-clip state the between-clip reset misses -- a streaming-detached LM's KV
cache and positions, an HF ``DynamicCache``, a pending conditioning tensor -- leaks from one clip
into the next. Nothing fails; clip 1 is correct and every later clip is quietly conditioned on its
predecessors (p0/p1 knowledge fell from ~0.2 in the first batch group to 0.00 by group 6).
``check_ppx_reset.py`` covers PersonaPlex; this covers the other engines, each through the REAL loop
its offline driver calls:

  * base Moshi -- ``moshi_engine.run_pairs`` (batched; reset = ``mimi/lm_gen.reset_streaming()``
    per batch, which reaches the detached LM through ``_LMGenState.reset_callback``);
  * Audex      -- ``audex.engine.AudexLMGen`` driven by the same ``moshi_engine.run_pairs`` (its
    driver's call); its own reset is the trunk ``DynamicCache`` (``reset_streaming_cache``), which the
    streaming machinery cannot reach because the HF trunk is a plain ``nn.Module``;
  * KAME       -- ``kame_engine.run_pairs`` (B=1, per-clip oracle injection);
  * MoshiRAG   -- ``moshirag.engine.run_pairs`` -> ``MoshiRagModel.run_clip`` (B=1 lockstep pump:
    ``BatchRunner.run_step`` resets on the first frame, the local STT resets in ``start_up``, and a
    retrieval's pending streaming-sum rows are cleared by the reset callback).

For each engine: [a] clips 2..N of one multi-clip call equal each clip run alone on a FRESH instance
(tokens: text stream + every audio codebook, recovered exactly from the stub codec; MoshiRAG also the
STT transcript, trigger step and reference of the trace); [b] CAN FAIL -- with that engine's reset
disabled, [a] must NOT hold. Mutations: Moshi/KAME the LM reset hook removed (as in
``check_ppx_reset``); Audex ``reset_streaming_cache`` skipped; MoshiRAG each of the RAG LM hook, the
STT LM hook and the pending-streaming-sum clear, one at a time.

Tiny random LMs, float64, greedy, CPU. Mimi is a stateless stub codec (codes are a pure function of
the frame's samples; decoding writes the audio codes into the PCM), so a difference can only come
from engine/LM state, and the ARC encoder call is a deterministic stub of the network request.

moshi_family venv, CPU, ~1 min (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_engine_reset.py'
"""

import asyncio
import contextlib
import copy
import importlib
import io
import os
import sys
import tempfile
import types
import wave
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import sphn  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
SR, FR = 24000, 12.5
FRAME = int(SR / FR)
TMP = Path(tempfile.mkdtemp(prefix="check_engine_reset_"))
FAILURES: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


@contextlib.contextmanager
def quiet():
    # The engines' token printers (run_inference.RawPrinter) write to stdout/stderr; display only.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


# ------------------------------------------------------------------------------------------------
# Stubs: a stateless codec and a tokenizer. Neither can carry state across clips, so any clip-2
# difference below comes from the engine or the LM.
# ------------------------------------------------------------------------------------------------
class StubMimi:
    sample_rate = SR
    frame_rate = FR
    channels = 1
    cardinality = 2048

    def __init__(self, num_codebooks=8):
        self.num_codebooks = num_codebooks
        self._streaming_state = None

    def set_num_codebooks(self, n):
        self.num_codebooks = n

    def streaming_forever(self, _b):
        pass

    def reset_streaming(self, reset_mask=None):
        pass

    def set_exec_mask(self, _m):
        pass

    def set_profile(self, _p):
        pass

    def _stop_streaming(self):
        pass

    def encode(self, pcm):
        B, _, n = pcm.shape
        f = pcm.reshape(B, 1, n // FRAME, FRAME).double()
        # A pure function of the frame's samples, a different code per codebook.
        s = (f * 1000.0).sum(-1).abs()  # [B, 1, T]
        k = torch.arange(self.num_codebooks, dtype=torch.float64).view(1, -1, 1)
        return torch.floor(s * (k + 1.3) + 17.0 * k).long() % 2048

    def decode(self, codes):
        # Write each audio code into its own sample of the frame, so the PCM carries every code
        # exactly (also through the int16 wav MoshiRAG writes: 2048 levels << 32767).
        B, K, T = codes.shape
        out = torch.zeros(B, 1, T * FRAME, dtype=torch.float32)
        for t in range(T):
            out[:, 0, t * FRAME : t * FRAME + K] = codes[:, :, t].float().clamp(min=0) / 2048.0
        return out


class StubTok:
    def __init__(self, n=100):
        self.n = n

    def id_to_piece(self, i):
        return f"▁t{int(i)}"

    def eos_id(self):
        return -1

    def get_piece_size(self):
        return self.n

    def encode(self, text):
        return [5 + (ord(c) % 90) for c in text]


def write_inputs(n, seed, tag):
    r = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        dur = FRAME * int(r.integers(6, 12))
        p = TMP / f"{tag}_in{i}.wav"
        sphn.write_wav(str(p), (0.3 * r.standard_normal(dur)).astype(np.float32), SR)
        paths.append(str(p))
    return paths


def small_kw(loaders, **extra):
    kw = dict(loaders._lm_kwargs)
    kw.update(
        dim=64,
        num_heads=4,
        num_layers=2,
        hidden_scale=2.0,
        text_card=100,
        context=256,
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=1,
    )
    kw.update(extra)
    return kw


# ------------------------------------------------------------------------------------------------
# base Moshi / Audex: moshi_engine.run_pairs
# ------------------------------------------------------------------------------------------------
from moshi_family import moshi_engine  # noqa: E402
from moshi_family.client_utils import RawPrinter  # noqa: E402
from moshi_family.models import loaders as mloaders  # noqa: E402
from moshi_family.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.run_inference import InferenceState  # noqa: E402

torch.manual_seed(0)
MOSHI_LM = LMModel(device="cpu", dtype=torch.float64, **small_kw(mloaders)).eval()
KAME_LM = LMModel(device="cpu", dtype=torch.float64, use_oracle=True, **small_kw(mloaders)).eval()


class NoHookLMGen(LMGen):
    """The PersonaPlex bug, re-created on the canonical LMGen: resets never reach the detached LM."""

    def _init_streaming_state(self, batch_size):
        state = super()._init_streaming_state(batch_size)
        state.reset_callback = None
        return state


def moshi_family_model(lm, lm_gen_cls, batch_size=2):
    """Exactly MoshiModel/AudexModel minus the checkpoint load: `.state` is the real InferenceState."""
    m = types.SimpleNamespace(batch_size=batch_size, sample_rate=SR, device="cpu")
    m.state = _inference_state(lm, lm_gen_cls, batch_size)
    # The token printer binds the real stdout/stderr at import time, so redirecting cannot mute it.
    m.state.printer = RawPrinter(io.StringIO(), io.StringIO())
    return m


def _inference_state(lm, lm_gen_cls, batch_size):
    return InferenceState(
        types.SimpleNamespace(model_type="moshi"),
        StubMimi(),
        StubTok(lm.text_card),
        copy.deepcopy(lm),
        batch_size,
        1.0,
        "cpu",
        lm_gen_cls=lm_gen_cls,
        use_sampling=False,
    )


def moshi_outputs(model, ins):
    with quiet():
        res = moshi_engine.run_pairs(
            model,
            [(p, None) for p in ins],
            lead_in_s=0.16,
            capture_s=0.8,
            keep_audio=len(ins),
            max_truncated_frac=1.0,
        )
    return [(mono, a["assistant"]) for mono, a in zip(res.monologues, res.audio)]


def same_outputs(a, b):
    return all(x[0] == y[0] and np.array_equal(np.asarray(x[1]), np.asarray(y[1])) for x, y in zip(a, b, strict=True))


def reused_equals_fresh(make, run, ins, group):
    """``ins`` in one call on one instance vs each ``group``-sized chunk after the first on a fresh one."""
    reused = run(make(), ins)
    fresh = []
    for s in range(group, len(ins), group):
        fresh += run(make(), ins[s : s + group])
    first_alone = run(make(), ins[:group])
    return same_outputs(reused[group:], fresh), same_outputs(reused[:group], first_alone), reused


def check_moshi():
    print("\n== base Moshi: moshi_engine.run_pairs (batch 2, 3 batches)")
    ins = write_inputs(6, 1, "moshi")
    ok, first_ok, reused = reused_equals_fresh(lambda: moshi_family_model(MOSHI_LM, LMGen), moshi_outputs, ins, 2)
    check(first_ok, "batch 1 on the reused instance == batch 1 alone (the comparison is deterministic)")
    check(ok, "batches 2-3 of one run_pairs call == each batch on a fresh instance")
    check(len({r[0] for r in reused}) == len(reused), "the six clips' monologues all differ (outputs are not constant)")
    bad, _, _ = reused_equals_fresh(lambda: moshi_family_model(MOSHI_LM, NoHookLMGen), moshi_outputs, ins, 2)
    check(not bad, "CAN FAIL: without the LM reset hook batches 2-3 differ from fresh")


def audex_lm():
    """A tiny AudexDuplexLM: the real class over a tiny random Nemotron-dense trunk (the vendored
    ``modeling_nemotron_dense.py`` from the HF cache; no weights read)."""
    import moshi_family.audex.models.lm as alm
    from moshi_family.audex.backbone import AudexBackbone

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        "nvidia/Nemotron-Labs-Audex-2B",
        allow_patterns=["checkpoint_folder_full/modeling_nemotron_dense.py"],
    )
    snap = os.path.join(root, "checkpoint_folder_full")
    sys.path.insert(0, snap)
    nd = importlib.import_module("modeling_nemotron_dense")
    cfg = nd.NemotronDenseConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    torch.manual_seed(1)
    bb = AudexBackbone(nd.NemotronDenseForCausalLM(cfg).double(), cfg)
    real_cfg = alm.audex_duplex_lm_config
    alm.audex_duplex_lm_config = lambda V: small_kw(mloaders, text_card=V)
    try:
        lm = alm.AudexDuplexLM(bb)
    finally:
        alm.audex_duplex_lm_config = real_cfg
    return lm.to(dtype=torch.float64).eval()


def check_audex():
    print("\n== Audex: AudexLMGen through moshi_engine.run_pairs (its driver's loop; batch 2)")
    from moshi_family.audex.engine import AudexLMGen

    class NoCacheResetAudexLMGen(AudexLMGen):
        """AudexLMGen with its own reset (the trunk DynamicCache) removed; the LMGen reset stays."""

        def reset_streaming(self, *args, **kwargs):
            return LMGen.reset_streaming(self, *args, **kwargs)

    lm = audex_lm()
    ins = write_inputs(4, 2, "audex")
    ok, first_ok, _ = reused_equals_fresh(lambda: moshi_family_model(lm, AudexLMGen), moshi_outputs, ins, 2)
    check(first_ok, "batch 1 on the reused instance == batch 1 alone")
    check(ok, "batch 2 of one run_pairs call == batch 2 on a fresh instance")
    bad, _, _ = reused_equals_fresh(lambda: moshi_family_model(lm, NoCacheResetAudexLMGen), moshi_outputs, ins, 2)
    check(not bad, "CAN FAIL: without reset_streaming_cache (trunk DynamicCache kept) batch 2 differs")


# ------------------------------------------------------------------------------------------------
# KAME: kame_engine.run_pairs (B=1, oracle per clip)
# ------------------------------------------------------------------------------------------------
def check_kame():
    print("\n== KAME: kame_engine.run_pairs (B=1, per-clip oracle)")
    from moshi_family import kame_engine

    def make(cls=LMGen):
        m = kame_engine.KameModel.__new__(kame_engine.KameModel)
        base = moshi_family_model(KAME_LM, cls, batch_size=1)
        m.state, m.lm = base.state, base.state.lm_gen.lm_model
        m.text_tokenizer = m.state.text_tokenizer
        m.sample_rate, m.frame_rate, m.device = SR, FR, "cpu"
        return m

    oracles = ["paris is the capital", "forty two", "the blue whale"]

    def run(model, ins):
        got = []
        real_run = model.state.run

        def recording_run(pcms):
            out = real_run(pcms)
            got.append((out[0][0].tolist(), out[0][1].numpy().copy()))
            return out

        model.state.run = recording_run
        texts = [oracles[int(Path(p).stem.split("in")[-1])] for p in ins]
        with quiet():
            kame_engine.run_pairs(
                model,
                [(p, str(TMP / f"kame_out_{i}.wav"), t) for i, (p, t) in enumerate(zip(ins, texts))],
                lead_in_s=0.16,
                capture_s=0.8,
            )
        return got

    ins = write_inputs(3, 3, "kame")
    ok, first_ok, _ = reused_equals_fresh(make, run, ins, 1)
    check(first_ok, "clip 1 on the reused instance == clip 1 alone")
    check(ok, "clips 2-3 of one run_pairs call == each clip on a fresh instance")
    bad, _, _ = reused_equals_fresh(lambda: make(NoHookLMGen), run, ins, 1)
    check(not bad, "CAN FAIL: without the LM reset hook clips 2-3 differ from fresh")


# ------------------------------------------------------------------------------------------------
# MoshiRAG: moshirag.engine.run_pairs -> MoshiRagModel.run_clip
# ------------------------------------------------------------------------------------------------
def check_moshirag():
    print("\n== MoshiRAG: moshirag.engine.run_pairs -> run_clip (B=1 lockstep pump, local STT, retrieval)")
    from moshi_family.moshirag import engine as reng
    from moshi_family.moshirag.inference_utils import utils as rutils
    from moshi_family.moshirag.inference_utils.audio_processor import AudioProcessor
    from moshi_family.moshirag.inference_utils.batch_runner import BatchRunner
    from moshi_family.moshirag.models import loaders as rloaders
    from moshi_family.moshirag.models.lm import LMGen as RLMGen
    from moshi_family.moshirag.models.lm import LMModel as RLMModel
    from moshi_family.moshirag.stt.local_stt import LocalSpeechToText

    kw = small_kw(rloaders)
    kw["conditioners"] = {"first_speaker": copy.deepcopy(rloaders._lm_kwargs["conditioners"]["first_speaker"])}
    kw["fuser"] = {"sum": [], "streaming_sum": [], "prepend": ["first_speaker"], "cross": []}
    kw["condition_provider"] = rloaders.get_conditioner_provider(kw["dim"], "cpu", kw)
    kw["fuser"] = rloaders.get_condition_fuser(kw)
    del kw["conditioners"]
    torch.manual_seed(2)
    rag_lm = RLMModel(device="cpu", dtype=torch.float64, **kw).eval()
    stt_kw = small_kw(rloaders, extra_heads_num_heads=3)
    stt_kw.pop("conditioners")
    stt_kw.pop("fuser")
    torch.manual_seed(3)
    stt_lm = RLMModel(device="cpu", dtype=torch.float64, **stt_kw).eval()

    async def fake_encoder(text, encoder_url="", timeout=30.0):
        # The ARC encoder service, stubbed deterministically: 40 rows, so a late retrieval leaves
        # pending streaming-sum rows at the end of a clip (the state the reset callback must clear).
        g = torch.Generator().manual_seed(sum(map(ord, text or "")) % 10_000)
        return 0.5 * torch.randn(1, 40, kw["dim"], generator=g)

    rutils.get_conditioning_remote_async = fake_encoder

    def make(disable=(), rag_token_id=None):
        """``disable`` names resets to break: "lm" (the RAG LM's reset hook), "stt" (the STT LM's),
        "pending" (keep the hook, but let pending streaming-sum rows survive it)."""
        m = reng.MoshiRagModel.__new__(reng.MoshiRagModel)
        lm = copy.deepcopy(rag_lm)
        cond = rutils.get_condition_tensors("moshi", lm, 1, 1.0, reference_text=None, first_speaker="user")
        lm_gen = RLMGen(lm, condition_tensors=cond, force_streaming_sum=True, use_sampling=False)
        mimi = StubMimi(lm.num_codebooks - 1)
        stt = LocalSpeechToText.__new__(LocalSpeechToText)
        stt.vad_callback, stt._device = None, torch.device("cpu")
        stt.text_tokenizer = StubTok()
        stt._lm_gen = RLMGen(copy.deepcopy(stt_lm), use_sampling=False)
        stt._prime_cap = max(stt._lm_gen.lm_model.delays)
        stt.mimi = copy.deepcopy(mimi)
        stt.mimi.streaming_forever(1)
        stt._lm_gen.streaming_forever(1)
        stt._out_queue, stt._lock = asyncio.Queue(), asyncio.Lock()
        stt.shutdown_complete = asyncio.Event()
        stt.shutdown_complete.set()
        stt.sent_samples, stt._pending, stt._frame_id = 0, np.zeros(0, np.float32), 0
        stt._playhead_s, stt._vad_step_count = 0.0, 0
        m.lm_gen, m.text_tokenizer, m.device = lm_gen, StubTok(lm.text_card), "cpu"
        m.runner = BatchRunner(mimi, lm_gen, "cpu", batch_size=1)
        if "lm" in disable:
            m.lm_gen._streaming_state.reset_callback = None
        if "stt" in disable:
            stt._lm_gen._streaming_state.reset_callback = None
        if "pending" in disable:
            # Pending rows are cleared TWICE per clip: run_clip reassigns the list before the first
            # frame, and the reset callback clears reset slots. Defeat both, or the other masks it.
            st = m.lm_gen._streaming_state
            real_cb = st.reset_callback
            sticky = st.__dict__.pop("pending_streaming_sums")

            class _StickyPending(type(st)):
                pending_streaming_sums = property(lambda self: sticky, lambda self, _v: None)

            st.__class__ = _StickyPending

            def keep_pending(mask, real_cb=real_cb):
                saved = list(sticky)
                real_cb(mask)
                sticky[:] = saved

            st.reset_callback = keep_pending
        m.mimi, m.frame_size, m.sample_rate, m.frame_rate = mimi, m.runner.frame_size, SR, FR
        m.rag_token_id = lm.rag_token_id if rag_token_id is None else rag_token_id
        m.audio_processor = AudioProcessor(power_threshold=-65)
        m.stt, m.stt_wait_steps = stt, int(0.5 * FR)
        m.vad_window_size, m.vad_threshold, m.init_active_speaker = 4, 0.5, "user"
        m.reference_generator, m.encoder_url, m.rag_timeout, m.max_reference_tokens = None, "stub", 10.0, 64
        m.runner.warmup()
        return m

    leftover = []

    def run(model, ins):
        outs = [str(TMP / f"rag_out_{Path(p).stem}.wav") for p in ins]
        with quiet():
            asyncio.run(
                reng.run_pairs(
                    model,
                    list(zip(ins, outs)),
                    lead_in_s=0.16,
                    capture_s=0.8,
                    tail_s=0.0,
                    gt_reference_for={p: f"reference for {Path(p).stem}" for p in ins},
                )
            )
        leftover.append(sum(p.shape[0] for p in model.lm_gen._streaming_state.pending_streaming_sums if p is not None))
        got = []
        for o in outs:
            import json

            tr = json.loads(Path(o).with_suffix(".json").read_text())
            with wave.open(o) as wf:
                pcm = wf.readframes(wf.getnframes())
            got.append(
                (
                    (tuple(tr["model_text"]), tuple(tr["user_text"]), tr["rag_trigger_step"], tr["reference_text"]),
                    np.frombuffer(pcm, np.int16),
                )
            )
        return got

    ins = write_inputs(3, 4, "rag")
    # Make retrieval actually fire: use as <ret> the text token the tiny model emits most often.
    probe = run(make(), ins[:1])[0][0][0]
    ids = [int(t[2:]) for t in probe if t.startswith("▁t")]
    rag_id = max(set(ids), key=ids.count) if ids else None
    leftover.clear()
    ok, first_ok, reused = reused_equals_fresh(lambda: make(rag_token_id=rag_id), run, ins, 1)
    fired = [r[0][2] for r in reused]
    print(f"  (retrieval fired at steps {fired} with <ret>={rag_id}; pending rows left after each run: {leftover})")
    check(all(s >= 0 for s in fired), "retrieval fires in every clip (the streaming-sum path is exercised)")
    check(min(leftover) > 0, "a clip ends with unconsumed reference rows pending (the state a reset must clear)")
    check(first_ok, "clip 1 on the reused instance == clip 1 alone")
    check(ok, "clips 2-3 of one run_pairs call == each clip on a fresh instance (text, STT, trace, wav)")
    for what, why in (
        ("lm", "without the RAG LM's reset hook (KV cache kept)"),
        ("stt", "without the local STT LM's reset hook (user transcript / retrieval context leaks)"),
        ("pending", "with a retrieval's leftover pending streaming-sum rows surviving both per-clip clears"),
    ):
        bad, _, _ = reused_equals_fresh(lambda what=what: make(disable=(what,), rag_token_id=rag_id), run, ins, 1)
        check(not bad, f"CAN FAIL: {why} clips 2-3 differ from fresh")


def main():
    for fn in (check_moshi, check_audex, check_kame, check_moshirag):
        fn()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
