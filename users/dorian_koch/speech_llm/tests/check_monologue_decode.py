"""Guard: every inner-monologue text decode goes through ONE decoder, ``moshi_family.text_trace.decode_monologue``.

Audit item 7a (2026-10-12). PersonaPlex's text-logit dump and RL rollouts dropped the four text specials (EPAD 0, BOS 1,
EOS 2, PAD 3), while the self-play eval (``personaplex/selfplay.decode_text``), the base-Moshi eval engine
(``moshi_engine.run_pairs``), the self-play RL transcript (``rl/ppx_selfplay.monologue_words``) and the live server each
had a copy dropping only 0 and 3 -- an id 1/2 would have been scored as the literal word ``<s>`` / ``</s>``. (None was
ever sampled: 0 of 63,176 saved self-play sides, 0 of 216,228 saved SpeechInference monologues, so no published number
moves.) Audex's adapter over its HF vocab keeps (0, 3) on purpose (``text_special_ids``): 1/2 are real tokens there.

[1] the shared decoder on the real Moshi SentencePiece model: specials interleaved with real pieces vanish, the words
    survive; [2] base Moshi: the REAL ``moshi_engine.run_pairs`` (stub model emitting chosen text ids) returns and writes
    (``<i>.txt``) exactly that decode; [3] Audex: the real ``_AudexTextTokenizer`` through the same ``run_pairs`` keeps
    its historical (0, 3) decode; [4] RL self-play: ``monologue_words`` joins to the same text; [5] AST: the self-play
    eval's ``text_a``/``text_b`` and the live server use the shared decoder / id set, and no ``(0, 3)`` filter is left
    in our code (vendored upstream files excepted); [6] CAN FAIL: the old (0, 3) filter in [2] and [4] is caught.

moshi_family venv, CPU, seconds:
    ./hpc-venv.py moshi_family_venv --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_monologue_decode.py'
"""

import ast
import contextlib
import glob
import io
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import sentencepiece as spm  # noqa: E402
import sphn  # noqa: E402
import torch  # noqa: E402

from moshi_family import moshi_engine  # noqa: E402
from moshi_family.rl import ppx_selfplay  # noqa: E402
from moshi_family.text_trace import TEXT_SPECIAL_IDS, decode_monologue  # noqa: E402

FAILURES: list[str] = []
SR = 24000
PKG = Path(moshi_engine.__file__).parent
#: upstream files kept verbatim (their (0, 3) display filters are the released code's)
VENDORED = ("personaplex/offline.py", "run_inference.py", "flmaudio/", "/models/", "third_party/", "/tests/")


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAILURES.append(msg)


def legacy_decode(tok, ids):
    """The removed copies (selfplay.decode_text, moshi_engine, monologue_words' skip set): only 0 and 3 dropped."""
    return "".join(tok.id_to_piece(int(i)) for i in ids if int(i) not in (0, 3)).replace("▁", " ").strip()


def moshi_tokenizer():
    hub = os.environ.get("HF_HUB_CACHE", "/hpcwork/p0023999/common_hf_home/hub")
    paths = sorted(glob.glob(f"{hub}/models--kyutai--moshiko-pytorch-bf16/snapshots/*/tokenizer_spm_32k_3.model"))
    assert paths, f"moshiko tokenizer not in {hub}"
    return spm.SentencePieceProcessor(model_file=paths[0])


def interleave(word_ids):
    """A frame stream as the model emits it: PAD runs, EPAD before words, and the two control ids 1 / 2."""
    out = [3, 3, 0]
    for k, t in enumerate(word_ids):
        out.append(int(t))
        out += [3, 1] if k % 3 == 0 else ([2, 0] if k % 3 == 1 else [3])
    return out + [3, 3]


def run_engine(tok, ids, out_wav):
    """moshi_engine.run_pairs on a stub model whose InferenceState emits ``ids`` as the text stream."""
    state = SimpleNamespace(
        text_tokenizer=tok,
        mimi=SimpleNamespace(reset_streaming=lambda: None),
        lm_gen=SimpleNamespace(reset_streaming=lambda: None),
        frame_size=1920,
        run=lambda pcms: [(torch.tensor(ids), torch.zeros(1, SR)) for _ in range(pcms.shape[0])],
    )
    model = SimpleNamespace(sample_rate=SR, batch_size=1, state=state, device="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        res = moshi_engine.run_pairs(model, [(str(IN_WAV), out_wav)], lead_in_s=0.1, capture_s=0.5)
    return res.monologues[0]


def ast_checks():
    sp_tree = ast.parse((PKG / "personaplex" / "selfplay.py").read_text())
    vals = {}
    for d in ast.walk(sp_tree):
        if isinstance(d, ast.Dict):
            for k, v in zip(d.keys, d.values):
                if isinstance(k, ast.Constant) and k.value in ("text_a", "text_b"):
                    vals[k.value] = ast.unparse(v)
    check(vals.get("text_a", "").startswith("decode_monologue(") and vals.get("text_b", "").startswith("decode_monologue("),
          f"[5] self-play eval text_a / text_b = decode_monologue(...): {vals}")  # fmt: skip
    live = (PKG / "personaplex" / "live_server.py").read_text()
    check("tid not in TEXT_SPECIAL_IDS" in live, "[5] live server skips TEXT_SPECIAL_IDS")
    hits = []
    for f in sorted(PKG.rglob("*.py")):
        rel = str(f.relative_to(PKG))
        if any(v in "/" + rel for v in VENDORED):
            continue
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
                for c in node.comparators:
                    elts = getattr(c, "elts", None) or []
                    if all(isinstance(e, ast.Constant) and type(e.value) is int for e in elts) and sorted(
                        e.value for e in elts
                    ) == [0, 3]:
                        hits.append(f"{rel}:{node.lineno}")
    check(not hits, f"[5] no (0, 3) text filter left outside vendored files: {hits}")


def main():
    global IN_WAV
    tmp = Path(tempfile.mkdtemp(prefix="check_monologue_decode_"))
    IN_WAV = tmp / "q.wav"
    sphn.write_wav(str(IN_WAV), np.zeros(SR // 2, dtype=np.float32), SR)
    tok = moshi_tokenizer()
    words = tok.encode("the capital of france is paris, i think")
    ids = interleave(words)
    assert {1, 2} <= set(ids)

    print("== [1] shared decoder, real moshiko SentencePiece")
    mono = decode_monologue(tok, ids)
    check(TEXT_SPECIAL_IDS == (0, 1, 2, 3), f"TEXT_SPECIAL_IDS = {TEXT_SPECIAL_IDS}")
    check(mono == tok.decode(words), f"decode == sentencepiece's decode of the word ids: {mono!r}")
    check("<s>" in legacy_decode(tok, ids), f"(the old filter prints the controls: {legacy_decode(tok, ids)!r})")

    print("== [2] base Moshi: moshi_engine.run_pairs")
    out_wav = tmp / "0.wav"
    got = run_engine(tok, ids, str(out_wav))
    check(got == mono, f"run_pairs monologue == decode_monologue: {got!r}")
    check(out_wav.with_suffix(".txt").read_text() == mono, "<i>.txt sidecar == decode_monologue")

    print("== [3] Audex: _AudexTextTokenizer through run_pairs keeps its (0, 3) decode")
    from moshi_family.audex.engine import _AudexTextTokenizer

    vocab = {1: "▁Hi", 2: "!", 5: "▁there", 7: "s"}
    hf = SimpleNamespace(eos_token_id=None, convert_ids_to_tokens=lambda i: vocab.get(int(i), f"<{i}>"))
    atok = _AudexTextTokenizer(hf)
    a_ids = [3, 0, 1, 2, 3, 5, 7, 3, 0]
    check(run_engine(atok, a_ids, None) == legacy_decode(atok, a_ids) == "Hi! theres",
          f"Audex monologue unchanged: {run_engine(atok, a_ids, None)!r}")  # fmt: skip

    print("== [4] RL self-play: ppx_selfplay.monologue_words")
    w = ppx_selfplay.monologue_words(tok, ids, 0, len(ids))
    joined = " ".join(x["text"] for x in w)
    check(joined == mono, f"monologue_words joined == decode_monologue: {joined!r}")

    print("== [5] wiring (AST)")
    ast_checks()

    print("== [6] CAN FAIL: the old (0, 3) filter")
    real = moshi_engine.decode_monologue
    moshi_engine.decode_monologue = legacy_decode
    try:
        bad = run_engine(tok, ids, None)
    finally:
        moshi_engine.decode_monologue = real
    check(bad != mono and "<s>" in bad, f"run_pairs with the legacy decode differs: {bad!r}")
    real_ids = ppx_selfplay.TEXT_SPECIAL_IDS
    ppx_selfplay.TEXT_SPECIAL_IDS = (0, 3)
    try:
        bad_w = " ".join(x["text"] for x in ppx_selfplay.monologue_words(tok, ids, 0, len(ids)))
    finally:
        ppx_selfplay.TEXT_SPECIAL_IDS = real_ids
    check(bad_w != mono, f"monologue_words with the legacy skip set differs: {bad_w!r}")

    if FAILURES:
        print(f"\ncheck_monologue_decode: {len(FAILURES)} FAILED")
        sys.exit(1)
    print("\ncheck_monologue_decode: all passed")


if __name__ == "__main__":
    main()
