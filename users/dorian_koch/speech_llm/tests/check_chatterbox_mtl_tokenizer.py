"""Guard (chatterbox venv, CPU, offline as our jobs run): Chatterbox Multilingual tokenizes Chinese through
``chatterbox_fast_mtl``'s loader fix, not into ``[UNK]``.

Upstream's ``MTLTokenizer`` looks its Cangjie table (``Cangjie5_TC.json``) up with ``cache_dir=<the checkpoint
snapshot dir>``; offline that lookup fails with only a log warning, every Chinese character becomes ``[UNK]`` and the
speech is gibberish (Whisper CER ~1.0 on 8 sentences, 2026-11-17). Asserted:

  * upstream's tokenizer, built as upstream builds it, offline: the table is empty and a Chinese sentence has
    ``[UNK]`` tokens (the bug this guard exists for; if upstream ever fixes it, this part reports so and passes);
  * ``chatterbox_fast_mtl.fix_cangjie`` on it: the table loads (> 50,000 entries) and the same sentence has none;
  * ``chatterbox_fast_mtl.text_tokens`` (the batched path's text path) for German and Chinese: start / stop text
    tokens, the language tag, no ``[UNK]``.

    HF_HUB_OFFLINE=1 <chatterbox venv python> recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_chatterbox_mtl_tokenizer.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HOME", "/hpcwork/p0023999/common_hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm"))

import chatterbox_fast_mtl as cm  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config  # noqa: E402
from chatterbox.models.tokenizers import MTLTokenizer  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

ZH = "今天天气不错，我们下午一起去公园散步吧。"
DE = "Am Wochenende fahren wir an die Ostsee."


def main():
    fails = []
    ckpt = Path(snapshot_download(cm.REPO, allow_patterns=["grapheme_mtl_merged_expanded_v1.json"]))
    tok = MTLTokenizer(str(ckpt / "grapheme_mtl_merged_expanded_v1.json"))
    unk = tok.tokenizer.token_to_id("[UNK]")

    def n_unk(t, text_lang):
        return sum(int(i) == unk for i in t.encode(text_lang[0], language_id=text_lang[1]))

    before = n_unk(tok, (ZH, "zh"))
    if tok.cangjie_converter.word2cj or before == 0:
        print(f"NOTE upstream's tokenizer now loads the Cangjie table offline ({before} [UNK]): the fix is a no-op")
    else:
        print(f"upstream tokenizer offline: empty Cangjie table, {before} [UNK] in {ZH!r} (the bug)")
    cm.fix_cangjie(tok)
    if len(tok.cangjie_converter.word2cj) < 50000:
        fails.append(f"fix_cangjie: only {len(tok.cangjie_converter.word2cj)} Cangjie entries")
    if n_unk(tok, (ZH, "zh")):
        fails.append(f"after fix_cangjie {n_unk(tok, (ZH, 'zh'))} [UNK] in the Chinese sentence")
    model = SimpleNamespace(tokenizer=tok, t3=SimpleNamespace(hp=T3Config.multilingual()))
    for text, lang in ((ZH, "zh"), (DE, "de")):
        t = cm.text_tokens(model, text, lang).tolist()
        hp = model.t3.hp
        tag = tok.tokenizer.token_to_id(f"[{lang}]")
        if t[0] != hp.start_text_token or t[-1] != hp.stop_text_token or tag not in t[:3] or unk in t:
            fails.append(f"text_tokens({lang}): {t[:6]}...{t[-3:]} (start/stop/[{lang}] tag/[UNK])")
    for f in fails:
        print("FAIL", f)
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
