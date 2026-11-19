"""``ChatterboxInferenceV2``'s worker: dialogues in any Chatterbox language, turns that switch language, an ASR check
that regenerates bad segments, and word timings, in one pass on one GPU.

Run by the job under the Chatterbox venv (``chatterbox_inference.py``'s sibling files). Per chunk of dialogues:

1. **Plan** (``plan_dialogue``): each turn's language segments (``chatterbox_v2_core.turn_segments``), the TTS model
   per dialogue (``tts_model: "qwen3"``: Qwen3-TTS for every dialogue, ``qwen3_tts_worker.py`` in its own venv, each
   turn one item in its main language (``code_switch: "single_pass"``); else Turbo when every segment is English and
   ``english_model`` is ``turbo``, else Multilingual V3 for the whole dialogue, so a voice never changes model
   mid-dialogue), the voice per speaker, one seed per segment, the
   pause before each turn drawn by who speaks next (``gaps``: the user's varies widely, the assistant's little).
2. **Synthesize + check** (``chatterbox_v2_core.synthesize_checked``): every segment batched through
   ``chatterbox_fast`` / ``chatterbox_fast_mtl``, transcribed by its language's ``check`` backend, scored against its
   text after normalisation (``speech_text_norm``: numbers, symbols, punctuation); segments above ``max_err`` are
   synthesized again with a new seed, up to ``max_attempts`` in all, and so are segments whose speaking rate is
   outside the speaker role's ``pace`` band; the best attempt is kept.
3. **Word timings** by each language's ``words`` backend, per segment.
4. **Render** (``chatterbox_v2_core.render``): every segment's edge silence trimmed, a turn's segments joined with
   ``segment_gap`` between, turns placed
   on the timeline as ``chatterbox_inference`` does, one track per speaker.

Annotators run as worker processes on the same GPU: ``parakeet_worker.py`` (``speech_llm``, NeMo venv) and
``speech_annotate_worker.py`` (annotate venv: Qwen3-ASR / -ForcedAligner and the scorer).

Input: ``--in_hf`` with a ``dialogue`` column (JSON): a list of turns ``{"speaker", "text" | "segments", "lang"?,
"exaggeration"?, "pre_silence"?}`` (``<cut/>`` in a text: the next turn interrupts there,
``chatterbox_v2_core.CUT``) or a dict ``{"turns", "lang"?, "voices"?: {speaker: voice name}}``. Output (HF):
``id``, ``speaker_audio``, ``turns`` (``text`` without language tags, ``lang``), ``segments`` (lang, text, times, error,
attempts, ASR text), ``words`` (turn, segment, speaker, lang, text, start, end on the dialogue timeline), ``voices``,
``tts_model``, and ``--keep_columns``. ``--stats_json``: ``{"raw", "summary"}`` (``chatterbox_v2_core.stats_*``): per
language segments, audio, error rate, regeneration shares; wall times.
"""

import argparse
import json
import os
import pickle
import random
import struct
import subprocess
import time

import numpy as np
import torch
from datasets import Audio, Dataset, Features, Sequence, Value, load_from_disk

import chatterbox_fast
import chatterbox_fast_mtl
import chatterbox_inference as v1
import chatterbox_v2_core as core
from clip_codec import wav_to_flac

print("Imports successful", flush=True)

FEATURES = Features(
    {
        "id": Value("string"),
        "speaker_audio": Sequence({"speaker": Value("string"), "audio": Audio()}),
        "turns": Sequence(
            {
                "speaker": Value("string"),
                "start_time": Value("float32"),
                "end_time": Value("float32"),
                "text": Value("string"),
                "lang": Value("string"),
                "interrupted": Value("bool"),
            }
        ),
        "segments": Sequence(
            {
                "turn": Value("int32"),
                "lang": Value("string"),
                "text": Value("string"),
                "start": Value("float32"),
                "end": Value("float32"),
                "err": Value("float32"),
                "wer": Value("float32"),
                "cer": Value("float32"),
                "attempts": Value("int32"),
                "asr_text": Value("string"),
                "rate": Value("float32"),
            }
        ),
        "words": Sequence(
            {
                "turn": Value("int32"),
                "segment": Value("int32"),
                "speaker": Value("string"),
                "lang": Value("string"),
                "text": Value("string"),
                "start": Value("float32"),
                "end": Value("float32"),
            }
        ),
        "voices": Sequence({"speaker": Value("string"), "voice": Value("string"), "exaggeration": Value("float32")}),
        "tts_model": Value("string"),
    }
)


class WorkerProc:
    """A worker process speaking the ``asr_worker`` protocol (8-byte length + pickle; first message ready)."""

    def __init__(self, argv: list[str], name: str):
        self.name = name
        env = {
            **os.environ,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }  # less fragmentation on a shared card
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        if self._read() != {"ready": True}:
            raise RuntimeError(f"{name}: the worker did not start")

    def _read(self):
        head = self.p.stdout.read(8)
        if len(head) < 8:
            raise RuntimeError(f"{self.name}: the worker exited (code {self.p.poll()})")
        (n,) = struct.unpack("<Q", head)
        return pickle.loads(self.p.stdout.read(n))

    def call(self, method: str, **kw):
        b = pickle.dumps({"method": method, **kw}, protocol=5)
        self.p.stdin.write(struct.pack("<Q", len(b)) + b)
        self.p.stdin.flush()
        r = self._read()
        if "error" in r:
            raise RuntimeError(f"{self.name}.{method}: {r['error']}")
        return r["result"]

    def call_split(self, method: str, lists: tuple[str, ...], **kw):
        """``call`` over list arguments ``lists`` (same length); on a CUDA out-of-memory error in the worker, the two
        halves one after the other (down to one clip), so a long batch slows down instead of failing the job."""
        try:
            return self.call(method, **kw)
        except RuntimeError as e:
            n = len(kw[lists[0]])
            if "OutOfMemory" not in str(e) and "out of memory" not in str(e) or n <= 1:
                raise
            print(f"[v2] {self.name}.{method}: out of memory on {n} clips, splitting", flush=True)
            h = n // 2
            a = self.call_split(method, lists, **{**kw, **{k: kw[k][:h] for k in lists}})
            return a + self.call_split(method, lists, **{**kw, **{k: kw[k][h:] for k in lists}})


#: clips per ASR / aligner forward: small, so the annotators stay at a few GB beside the TTS (at 32 Parakeet alone held
#: 28 GB on long LLM-written turns, 2026-11-17)
ASR_BATCH = 8


class Annotators:
    """The per-language ``check`` / ``words`` backends (``chatterbox_v2_core.DEFAULT_ANNOTATORS``) over the worker
    processes; the scorer (annotate worker) always runs."""

    def __init__(self, cfg: dict, parakeet: tuple[str, str] | None, annotate: tuple[str, str], qwen3: dict,
                 metric: str = "cer"):  # fmt: skip
        core.check_annotators(cfg)
        assert metric in ("cer", "wer"), metric
        self.cfg, self.metric = cfg, metric
        used = {b for roles in cfg.values() for b in roles.values()}
        self.parakeet = None
        if "parakeet" in used:
            assert parakeet, "a parakeet annotator needs --parakeet_python and --parakeet_worker"
            spec = {
                "backend": "parakeet",
                "model": "nvidia/parakeet-tdt-0.6b-v3",
                "device": "cuda",
                "python": parakeet[0],
            }
            self.parakeet = WorkerProc([parakeet[0], parakeet[1], json.dumps(spec)], "parakeet")
        checks_q = any(r["check"] == "qwen3" for r in cfg.values())
        words_q = any(r["words"] == "qwen3" for r in cfg.values())
        spec = {"asr_model": qwen3["asr"] if checks_q else None, "aligner_model": qwen3["aligner"] if words_q else None,
                "device": "cuda", "batch_size": ASR_BATCH}  # fmt: skip
        self.annotate = WorkerProc([annotate[0], annotate[1], json.dumps(spec)], "annotate")

    def backend(self, lang: str, role: str) -> str:
        b = core.per_lang(self.cfg, lang)[role]
        ok = core.BACKEND_LANGS[b]
        if ok is not None and lang not in ok:
            raise RuntimeError(f"{role} backend {b} does not support {lang!r}; set annotators[{lang!r}]")
        return b

    def check(self, wavs, sr: int, refs: list[str], lang: str, min_tokens: int = 0) -> list[dict]:
        """Per clip ``{"err", "asr_text", "words", "exempt"}`` (``words``: Parakeet's, reused for timings, else None;
        ``exempt``: fewer than ``min_tokens`` reference tokens, never regenerated)."""
        b = self.backend(lang, "check")
        pcms = [np.asarray(w, np.float32) for w in wavs]
        words = [None] * len(pcms)
        if b == "none":
            return [{"err": 0.0, "wer": 0.0, "cer": 0.0, "asr_text": "", "words": None, "exempt": True} for _ in pcms]
        if b == "parakeet":
            words = self.parakeet.call_split("transcribe_words_many", ("pcms",), pcms=pcms, sr=sr, batch_size=ASR_BATCH)
            hyps = [" ".join(w["text"] for w in ws) for ws in words]
        else:
            hyps = self.annotate.call_split("transcribe_many", ("pcms",), pcms=pcms, sr=sr, language=lang)
        scores = self.annotate.call("score_many", refs=refs, hyps=hyps, language=lang)
        return [{"err": s[self.metric], "wer": s["wer"], "cer": s["cer"], "asr_text": h, "words": w,
                 "exempt": s["n_ref"] < min_tokens} for s, h, w in zip(scores, hyps, words)]  # fmt: skip

    def words(self, wavs, sr: int, refs: list[str], lang: str, cached: list) -> list[list[dict]]:
        b = self.backend(lang, "words")
        if b == "none":
            return [[] for _ in wavs]
        pcms = [np.asarray(w, np.float32) for w in wavs]
        if b == "parakeet":
            if self.backend(lang, "check") == "parakeet" and all(c is not None for c in cached):
                return cached
            return self.parakeet.call_split("transcribe_words_many", ("pcms",), pcms=pcms, sr=sr, batch_size=ASR_BATCH)
        return self.annotate.call_split("align_many", ("pcms", "texts"), pcms=pcms, sr=sr, texts=refs, language=lang)


class Tts:
    """Turbo and Multilingual V3, each loaded on first use; conditionals cached per (model, voice file, exaggeration).
    ``q3``: (python, worker script, spec) of the Qwen3-TTS worker (model kind ``q3``), started on first use."""

    #: the batched decoders' sizes per model (slots, S3Gen batch); halved for the rest of the run after an out-of-memory
    SIZES = {"turbo": (96, 32), "mtl": (64, 16)}

    def __init__(self, device: str, mtl_decode: dict | None = None, q3: tuple[str, str, dict] | None = None):
        self.device, self.models, self.conds, self.q3 = device, {}, {}, q3
        self.mtl_decode = mtl_decode or {}  # ContinuousMtlT3 sampling (cfg_weight, temperature, ...)
        self.sizes = dict(self.SIZES)

    def model(self, kind: str):
        if kind not in self.models:
            t0 = time.time()
            if kind == "q3":
                assert self.q3, "tts_model qwen3 needs --qwen3_tts_python and --qwen3_tts_worker"
                py, worker, spec = self.q3
                m = WorkerProc([py, worker, json.dumps(spec)], "qwen3_tts")
            elif kind == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                m = ChatterboxTurboTTS.from_pretrained(device=self.device)
                chatterbox_fast.speed_s3gen(m)
            else:
                m = chatterbox_fast_mtl.load_mtl(self.device, "v3")
                chatterbox_fast_mtl.speed_s3gen(m)
            self.models[kind] = m
            print(f"TTS: loaded {kind} in {time.time() - t0:.0f} s", flush=True)
        return self.models[kind]

    @property
    def sr(self) -> int:
        return 24000  # S3GEN_SR, both models (asserted in synth)

    def cond(self, kind: str, path: str, exaggeration: float):
        key = (kind, path, exaggeration)
        if key not in self.conds:
            m = self.model(kind)
            m.prepare_conditionals(
                path, exaggeration=exaggeration, **({"norm_loudness": True} if kind == "turbo" else {})
            )
            self.conds[key], m.conds = m.conds, None
        return self.conds[key]

    def synth(self, items: list[dict]) -> list:
        out = [None] * len(items)
        # Multilingual items whose language differs from their voice prompt's may get their own guidance weight
        # (``cross_lang_cfg``; upstream's README: such output "may inherit the accent of the reference clip's language.
        # To mitigate this, set cfg_weight to 0"): one decoder pass per (model, guidance) group
        groups = sorted({(it["model"], it.get("cfg")) for it in items}, key=lambda g: (g[0], g[1] is not None, g[1] or 0))
        for kind, cfg_w in groups:
            idx = [i for i, it in enumerate(items) if it["model"] == kind and it.get("cfg") == cfg_w]
            if not idx:
                continue
            m = self.model(kind)
            if kind == "q3":
                wavs = m.call_split("synth_many", ("texts", "languages", "voices", "seeds"),
                                    texts=[items[i].get("tts_text") or items[i]["text"] for i in idx],
                                    languages=[items[i].get("tts_lang") or items[i]["lang"] for i in idx],
                                    voices=[items[i]["voice"] for i in idx],
                                    seeds=[items[i]["seed"] for i in idx])  # fmt: skip
                for i, w in zip(idx, wavs):  # raw float32 bytes (the worker's numpy is not ours)
                    out[i] = np.frombuffer(w, "<f4").astype(np.float32)
                continue
            assert m.sr == self.sr
            batch = [{"text": items[i].get("tts_text") or items[i]["text"], "lang": items[i]["lang"], "seed": items[i]["seed"],
                      "spans": items[i].get("tts_spans") or items[i].get("spans"), "span_tags": items[i].get("span_tags", True),
                      "cond": self.cond(kind, items[i]["voice"], items[i]["exaggeration"])} for i in idx]  # fmt: skip
            wavs = self._synth(kind, m, batch, None if cfg_w is None else {**self.mtl_decode, "cfg_weight": cfg_w})
            for i, w in zip(idx, wavs):
                out[i] = np.asarray(w, np.float32)
            torch.cuda.empty_cache()  # the decoder's static KV cache is sized per call; give it back to the other model
        return out

    def _synth(self, kind, m, batch, decode_kw=None):
        """One model's batched synthesis; on CUDA out of memory, again with half the slots and S3Gen batch (kept for
        the rest of the run), down to 4 slots."""
        import gc

        while True:
            slots, s3 = self.sizes[kind]
            try:
                if kind == "turbo":
                    return chatterbox_fast.synthesize(m, batch, slots=slots, s3_batch=s3, s3_pool=slots)
                return chatterbox_fast_mtl.synthesize(m, batch, slots=slots, s3_batch=s3, s3_pool=slots,
                                                      decode_kw=decode_kw or self.mtl_decode)  # fmt: skip
            except torch.OutOfMemoryError:
                if slots <= 4:
                    raise
                self.sizes[kind] = (slots // 2, max(1, s3 // 2))
                print(f"[v2] {kind}: CUDA out of memory at {slots} slots / S3Gen batch {s3}; retrying with "
                      f"{self.sizes[kind]}", flush=True)  # fmt: skip
                gc.collect()
                torch.cuda.empty_cache()


def plan_dialogue(raw, diag_id, cfg: dict, speaker_dir: str) -> dict:
    """Segments, model, voices, seeds and gaps of one dialogue (deterministic in ``diag_id``)."""
    spec = raw if isinstance(raw, dict) else {"turns": raw}
    turns, lang, voice_names = spec["turns"], spec.get("lang") or cfg["default_lang"], spec.get("voices") or {}
    assert isinstance(turns, list) and turns, f"dialogue {diag_id}: no turns"
    random.seed(v1.SEED ^ v1.seed_for_dialogue(diag_id))  # speaker draws and gaps, as chatterbox_inference
    segs = []
    for t in turns:
        s = core.turn_segments(t, lang)
        if cfg["code_switch"] == "single_pass" and len(s) > 1:
            s = [{"lang": core.main_lang(s), "text": core.join_segments([x["text"] for x in s]), "mixed": True}]
        if cfg["code_switch"] in ("mixed", "mixed_lead") and len(s) > 1:
            # one T3 pass across languages (chatterbox_fast_mtl.text_tokens ``spans``); a turn too long for one item
            # stays in segments
            main, whole = core.main_lang(s), core.join_segments([x["text"] for x in s])
            if len(whole) <= core.per_lang(core.MAX_ITEM_CHARS, main):
                segs.append([{"lang": main, "text": whole, "spans": s}])
                continue
        assert s, f"dialogue {diag_id}: a turn without text: {t!r}"
        segs.append([{"lang": x["lang"], "text": p, **({"mixed": True} if x.get("mixed") else {})} for x in s
                     for p in core.split_long(x["text"], x["lang"])])  # fmt: skip
    langs = {x["lang"] for s in segs for x in s} | {sp["lang"] for s in segs for x in s for sp in x.get("spans", [])}
    if cfg.get("tts_model") == "qwen3":  # Qwen3-TTS: no per-span language control
        # single_pass: a mixed turn is one item in its main language (or no language token: ``qwen3_mixed_lang``
        # "auto"); segments: each language span its own item with its own language token
        assert cfg["code_switch"] in ("single_pass", "segments"), "tts_model qwen3: code_switch single_pass or segments"
        kind = "q3"
    else:
        kind = "turbo" if cfg["english_model"] == "turbo" and langs == {"en"} else "mtl"
    voices, used = {}, set()
    for t in turns:
        sp = t["speaker"]
        if sp not in voices:
            for _ in range(5):  # as chatterbox_inference: avoid giving two speakers one random voice
                p = v1.speaker_name_to_path(voice_names.get(sp, sp), speaker_dir)
                if p not in used:
                    break
            voices[sp] = p
            used.add(p)
    # exaggeration per speaker and dialogue: drawn from cfg["exaggeration"][speaker] (a [lo, hi] range or a number;
    # "*" = any other speaker) on its own random stream, so the voice and gap draws above stay as they were
    exr = random.Random(core.stable_seed(v1.SEED, diag_id, "exaggeration"))
    exag = {}
    for sp in voices:
        e = core.per_lang(cfg.get("exaggeration") or {"*": 0.5}, sp)
        exag[sp] = round(exr.uniform(*e), 3) if isinstance(e, (list, tuple)) else float(e)
    # a voice prompt's language = the first directory of its name in the voice directory (``<lang>/<role>/...``)
    vlang = {sp: os.path.relpath(p, speaker_dir).split(os.sep)[0] for sp, p in voices.items()}
    xcfg = cfg.get("cross_lang_cfg")
    items = []
    for ti, (t, s) in enumerate(zip(turns, segs)):
        for si, x in enumerate(s):
            items.append({"turn": ti, "segment": si, "text": x["text"], "lang": x["lang"], "model": kind,
                          **({"tts_lang": "auto"} if kind == "q3" and x.get("mixed")
                             and cfg.get("qwen3_mixed_lang") == "auto" else {}),
                          **({"spans": x["spans"], "span_tags": cfg["code_switch"] == "mixed"} if "spans" in x else {}),
                          "speaker": t["speaker"], "voice": voices[t["speaker"]],
                          "exaggeration": float(t.get("exaggeration", exag[t["speaker"]])),
                          "seed": core.stable_seed(v1.SEED, diag_id, ti, si),
                          "cfg": xcfg if (xcfg is not None and kind == "mtl" and x["lang"] != vlang[t["speaker"]]
                                          and vlang[t["speaker"]] in core.LANG_IDS) else None})  # fmt: skip
    # the pause AFTER turn k is the one before turn k + 1: drawn by who speaks next (cfg["gaps"]), own random stream
    grng = random.Random(core.stable_seed(v1.SEED, diag_id, "gaps"))
    gspec = cfg.get("gaps") or core.DEFAULT_GAPS
    gaps = [core.draw_gap(grng, core.per_lang(gspec, turns[k + 1]["speaker"])) if k + 1 < len(turns) else 0.0
            for k in range(len(turns))]  # fmt: skip
    ov = cfg.get("cut_overlap") or core.DEFAULT_CUT_OVERLAP
    for k, t in enumerate(turns[:-1]):  # an interrupted turn: the next one starts before it ends (render cuts it)
        if core.turn_cut_units(t, lang) is not None:
            gaps[k] = -grng.uniform(*ov)
    return {"id": diag_id, "lang": lang, "turns": turns, "segs": segs, "kind": kind, "voices": voices, "exaggeration": exag,
            "items": items, "gaps": gaps}  # fmt: skip


def process_chunk(chunk, cfg, tts: Tts, ann: Annotators, speaker_dir, id_prefix, keep_cols, stats: dict,
                  stress: "WorkerProc | None" = None) -> list[dict]:  # fmt: skip
    """``chunk`` = [(diag_id, dialogue, passthrough)] -> its output rows (``FEATURES`` + ``keep_cols``), audio inline
    as FLAC: nothing is written to disk per dialogue."""
    plans = [(plan_dialogue(d, i, cfg, speaker_dir), pt) for i, d, pt in chunk]
    items = [it for p, _ in plans for it in p["items"]]
    sr = tts.sr

    def check(wavs, batch):
        res = [None] * len(batch)
        for lg in sorted({it["lang"] for it in batch}):
            idx = [i for i, it in enumerate(batch) if it["lang"] == lg]
            rs = ann.check([wavs[i] for i in idx], sr, [batch[i]["text"] for i in idx], lg,
                           cfg.get("min_check_tokens", 0))  # fmt: skip
            for i, r in zip(idx, rs):
                res[i] = r
        for i, (w, it) in enumerate(zip(wavs, batch)):  # pace over the sounding span
            a, b = core.edge_trim(w, sr, lead=True, trail=True, margin=0.0)
            units = core.speech_units(it["text"], it["lang"])
            if it.get("spans"):  # units of one language would misjudge a mixed turn's pace
                res[i].update(core.pace_check(units, (b - a) / sr, None, None))
                continue
            band = (cfg.get("pace") or {}).get(it["speaker"]) or {}
            band = core.per_lang(band, it["lang"]) if (it["lang"] in band or "*" in band) else None
            cap = None
            if units < cfg.get("min_pace_units", core.DEFAULT_MIN_PACE_UNITS):  # too short for a rate: only a length cap
                band, cap = None, core.short_cap(units, it["lang"])
            res[i].update(core.pace_check(units, (b - a) / sr, band, cap))
        return res

    if stress is not None:  # Russian TTS input with stress marks; the check still scores against the plain text
        ru = [it for it in items if it["lang"] == "ru" and not it.get("spans")]
        for it, st in zip(ru, stress.call("stress_many", texts=[it["text"] for it in ru]) if ru else []):
            it["tts_text"] = st
        mixed = [(it, k) for it in items for k, sp in enumerate(it.get("spans") or []) if sp["lang"] == "ru"]
        if mixed:  # a mixed turn: its Russian spans stressed
            for it in {id(it): it for it, _ in mixed}.values():
                it["tts_spans"] = [dict(sp) for sp in it["spans"]]
            for (it, k), st in zip(mixed, stress.call("stress_many", texts=[it["spans"][k]["text"] for it, k in mixed])):
                it["tts_spans"][k]["text"] = st
    t0 = time.time()
    results = core.synthesize_checked(items, tts.synth, check, lambda it: core.per_lang(cfg["max_err"], it["lang"]),
                                      cfg["max_attempts"])  # fmt: skip
    t1 = time.time()
    for lg in sorted({it["lang"] for it in items}):
        idx = [i for i, it in enumerate(items) if it["lang"] == lg]
        ws = ann.words([results[i]["wav"] for i in idx], sr, [items[i]["text"] for i in idx], lg,
                       [results[i]["words"] for i in idx])  # fmt: skip
        for i, w in zip(idx, ws):
            results[i]["words"] = w
    t2 = time.time()
    k, rows = 0, []
    for p, pt in plans:
        n = len(p["items"])
        res = results[k : k + n]
        for it, r in zip(p["items"], res):
            core.stats_add(stats, it["lang"], r["err"], r["attempts"], len(r["wav"]) / sr,
                           core.per_lang(cfg["max_err"], it["lang"]), r.get("pace_ok"))  # fmt: skip
        k += n
        tracks, turn_rows, seg_rows, word_rows = core.render(p, res, sr, cfg["segment_gap"], cfg.get("loudness_dbfs", core.DEFAULT_LOUDNESS_DBFS))
        rows.append(make_row(p, tracks, sr, turn_rows, seg_rows, word_rows, speaker_dir, id_prefix, pt, keep_cols))
        stats["dialogues"] += 1
    for k, v in (("synth_check_s", t1 - t0), ("words_s", t2 - t1), ("write_s", time.time() - t2)):
        stats["time"][k] = stats["time"].get(k, 0.0) + v
    return rows


def _columns(rows: list[dict], feature) -> dict:
    """A list-of-structs column the way ``datasets`` takes it: one list per field (``Sequence({...})`` is a dict of
    lists in datasets 4)."""
    keys = feature.feature if hasattr(feature, "feature") else feature
    return {k: [r[k] for r in rows] for k in keys}


def _flac(w, sr: int) -> dict:
    """An ``Audio()`` value: the track as PCM_16 FLAC bytes (``clip_codec.wav_to_flac`` of its PCM_16 WAV, as
    ``chatterbox_inference`` stores tracks), under a ``.wav`` name as there."""
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.clip(w, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    flac = wav_to_flac(buf.getvalue())
    assert flac is not None
    return flac


def make_row(p, tracks, sr, turn_rows, seg_rows, word_rows, speaker_dir, id_prefix, pt, keep_cols) -> dict:
    sps = list(tracks)
    row = {
        "id": f"{id_prefix}dialogue_{p['id']}",
        "speaker_audio": {
            "speaker": sps,
            "audio": [{"bytes": _flac(tracks[sp], sr), "path": f"{sp}.wav"} for sp in sps],
        },
        "turns": _columns(turn_rows, FEATURES["turns"]),
        "segments": _columns(seg_rows, FEATURES["segments"]),
        "words": _columns(word_rows, FEATURES["words"]),
        "voices": {
            "speaker": list(p["voices"]),
            "voice": [os.path.relpath(v, speaker_dir) for v in p["voices"].values()],
            "exaggeration": [p["exaggeration"][sp] for sp in p["voices"]],
        },
        "tts_model": {"turbo": "chatterbox_turbo", "mtl": "chatterbox_multilingual_v3", "q3": "qwen3_tts_12hz_1.7b_base"}[p["kind"]],
    }
    for c in keep_cols:
        v = (pt or {}).get(c)
        row[c] = None if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
    return row


@torch.inference_mode()
def main():
    import torchcodec  # noqa: F401  (fail early: FFmpeg / NPP on LD_LIBRARY_PATH, see ChatterboxInference)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in_hf", required=True)
    ap.add_argument("--in_hf_shard", type=int)
    ap.add_argument("--in_hf_num_shards", type=int)
    ap.add_argument("--sub_shard", type=int)
    ap.add_argument("--sub_num_shards", type=int)
    ap.add_argument("--limit", type=int, help="First N dialogues only (tests).")
    ap.add_argument("--scratch_dir", required=True, help="Node-local scratch: the dataset builder's cache.")
    ap.add_argument("--out_hf", required=True)
    ap.add_argument("--stats_json", required=True)
    ap.add_argument("--speaker_directory", required=True)
    ap.add_argument("--speaker_alias", default="{}")
    ap.add_argument("--keep_columns", nargs="*", default=[])
    ap.add_argument("--id_prefix", default="")
    ap.add_argument("--config", required=True, help="JSON: default_lang, english_model, code_switch, annotators, "
                    "max_err, max_attempts, segment_gap, chunk, qwen3_asr_model, qwen3_aligner_model, exaggeration, mtl_decode, "
                    "pace, min_pace_units, loudness_dbfs, gaps, cut_overlap, ru_stress, cross_lang_cfg, tts_model, qwen3_slots, "
                    "qwen3_tts_model, qwen3_mixed_lang")  # fmt: skip
    ap.add_argument("--parakeet_python")
    ap.add_argument("--parakeet_worker")
    ap.add_argument("--annotate_python", required=True)
    ap.add_argument("--annotate_worker", required=True)
    ap.add_argument("--stress_python", help="Russian stress venv (ru_stress_worker.py); without it Russian is unstressed")
    ap.add_argument("--stress_worker")
    ap.add_argument("--qwen3_tts_python", help="Qwen3-TTS venv (tts_model qwen3)")
    ap.add_argument("--qwen3_tts_worker", help="qwen3_tts_worker.py")
    args = ap.parse_args()
    cfg = json.loads(args.config)
    print(f"Config: {json.dumps(cfg)}", flush=True)
    v1.SPEAKER_ALIAS = json.loads(args.speaker_alias)
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("high")
    # Leave room for the annotator processes on the same card: this process may use the card minus
    # ``annotator_reserve_gb``; past that, synthesis retries with smaller batches (Tts._synth) instead of failing.
    total = torch.cuda.get_device_properties(0).total_memory
    reserve = float(cfg.get("annotator_reserve_gb", 16)) * 2**30
    torch.cuda.set_per_process_memory_fraction(max(0.2, (total - reserve) / total))
    print(f"TTS memory cap: {(total - reserve) / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)

    ds = load_from_disk(args.in_hf)
    if args.in_hf_shard is not None and args.in_hf_num_shards is not None:
        ds = ds.shard(num_shards=args.in_hf_num_shards, index=args.in_hf_shard)
    id_base = 0
    if args.sub_shard is not None and args.sub_num_shards is not None:  # contiguous, as chatterbox_inference
        div, mod = divmod(len(ds), args.sub_num_shards)
        id_base = args.sub_shard * div + min(args.sub_shard, mod)
        ds = ds.select(range(id_base, id_base + div + (1 if args.sub_shard < mod else 0)))
        assert len(ds) > 0, "empty sub-shard: cap the fan-out at the row count"
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    total = len(ds)
    v1._write_progress(0, total)

    t_start = time.time()
    ann = Annotators(cfg["annotators"], (args.parakeet_python, args.parakeet_worker) if args.parakeet_python else None,
                     (args.annotate_python, args.annotate_worker),
                     {"asr": cfg["qwen3_asr_model"], "aligner": cfg["qwen3_aligner_model"]},
                     cfg.get("metric", "cer"))  # fmt: skip
    print(f"Annotators ready in {time.time() - t_start:.0f} s", flush=True)
    q3 = None
    if args.qwen3_tts_python:
        q3 = (args.qwen3_tts_python, args.qwen3_tts_worker, {"device": "cuda", "slots": cfg.get("qwen3_slots", 64),
                                                              "model": cfg.get("qwen3_tts_model")})  # fmt: skip
    tts, stats = Tts("cuda", cfg.get("mtl_decode"), q3), {"dialogues": 0, "time": {}, "lang": {}}
    stress = None
    if cfg.get("ru_stress"):
        assert args.stress_python and args.stress_worker, "ru_stress needs --stress_python and --stress_worker"
        stress = WorkerProc([args.stress_python, args.stress_worker], "ru_stress")
        # upstream's tokenizer tries its own stresser (absent here) on every Russian text and warns each time
        import logging

        logging.getLogger("chatterbox.models.tokenizers.tokenizer").setLevel(logging.ERROR)

    def gen():  # rows stream into the dataset builder's arrow file as each chunk finishes
        chunk, done = [], 0
        for i, ex in enumerate(ds, start=id_base):
            chunk.append((i, json.loads(ex["dialogue"]), {c: ex[c] for c in args.keep_columns if c in ex}))
            if len(chunk) >= cfg["chunk"] or done + len(chunk) == total:
                yield from process_chunk(chunk, cfg, tts, ann, args.speaker_directory, args.id_prefix,
                                         args.keep_columns, stats, stress)  # fmt: skip
                done += len(chunk)
                chunk.clear()
                v1._write_progress(done, total)
                print(f"[v2] {done}/{total} dialogues, {time.time() - t_start:.0f} s: "
                      f"{json.dumps(core.stats_summary(stats))}", flush=True)  # fmt: skip

    import shutil
    import tempfile
    import uuid

    feats = Features({**FEATURES, **{c: Value("string") for c in args.keep_columns}})
    os.makedirs(args.scratch_dir, exist_ok=True)
    cache = tempfile.mkdtemp(prefix="hf_gen_cache_", dir=args.scratch_dir)
    # an explicit fingerprint: the default hashes the generator, i.e. pickles the models and worker processes
    out = Dataset.from_generator(gen, features=feats, cache_dir=cache, fingerprint=uuid.uuid4().hex)
    assert len(out) == total, (len(out), total)
    t0 = time.time()
    out.save_to_disk(args.out_hf)
    del out
    shutil.rmtree(cache, ignore_errors=True)
    stats["time"]["total_s"] = time.time() - t_start
    stats["peak_tts_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)  # this process (not the annotators)
    s = core.stats_summary(stats)
    with open(args.stats_json, "w") as f:
        json.dump({"raw": stats, "summary": s, "peak_tts_gpu_gb": stats["peak_tts_gpu_gb"]}, f, indent=1)
    print(f"[v2] {total} dialogues saved in {time.time() - t0:.0f} s; {json.dumps(s)}", flush=True)


if __name__ == "__main__":
    main()
