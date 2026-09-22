"""Guard: ``interleave_text(word_overlap=...)`` against the REFERENCE interleaver (moshi-finetune).

WHY. When a word's onset falls inside the previous word's tokens (3.5% of JRE words), our historical
``overwrite`` writes the later word over the earlier one in place, and a later word shorter than the
overlap leaves the earlier word's tail AFTER it (0.55% of words). Kyutai's own interleaver
(``recipe/moshi_finetune/finetune/data/interleaver.py``) never does that: by default it DROPS the
unwritten tail (moshi-finetune trains with ``keep_and_shift=False``); with ``keep_and_shift=True`` it
SHIFTS the later word behind the earlier one. Driven here on the same words with the same tokenizer:
  1. ``drop``  == reference keep_and_shift=False, token for token (EPAD on, onset floor).
  2. ``shift`` == reference keep_and_shift=True.
  3. ``overwrite`` DIFFERS from the reference on the scrambling case -- so (1) is not vacuous.

moshi_family venv (sentencepiece), CPU, seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_word_overlap.py'
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "moshi_finetune"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))  # kyutai `moshi`, which the reference imports

import sentencepiece  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from finetune.data.interleaver import Interleaver  # noqa: E402  (the reference)
from moshi_family.train_data_common import TEXT_EPAD_ID, TEXT_PADDING_ID, DuplexTokenizerBase  # noqa: E402

sp = sentencepiece.SentencePieceProcessor(hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model"))
FR = 12.5
ours = types.SimpleNamespace(text_tok=sp, frame_rate=FR)

# Seconds chosen so frames collide: "extraordinarily" is several tokens; "a"/"I" start inside it
# (shorter than the overlap -> the scrambling case), "neighbourhoods" starts in the same frame as the
# word before it, and the tail is ordinary spaced speech.
words = [
    ("extraordinarily", 0.50, 0.95),
    ("a", 0.60, 0.70),
    ("unbelievable", 1.30, 1.80),
    ("I", 1.44, 1.50),
    ("neighbourhoods", 2.00, 2.40),
    ("everywhere", 2.02, 2.50),
    ("okay", 3.20, 3.40),
    ("then", 3.60, 3.80),
]
alignments = [(w, (s, e), "SPEAKER_MAIN") for w, s, e in words]
T = 60


def reference(keep_and_shift):
    il = Interleaver(sp, FR, TEXT_PADDING_ID, TEXT_EPAD_ID, -1, keep_and_shift=keep_and_shift, device="cpu")
    return il.prepare_item(alignments, T / FR).view(-1)[:T]


def mine(mode):
    return DuplexTokenizerBase.interleave_text(
        ours, alignments, T, emit_epad=True, onset_floor=True, word_overlap=mode
    ).view(-1)


assert any(len(sp.encode(w)) > 1 for w, *_ in words), "need multi-token words for a collision"
ref_drop, ref_shift = reference(False), reference(True)
assert torch.equal(mine("drop"), ref_drop), f"drop != reference\n{mine('drop').tolist()}\n{ref_drop.tolist()}"
print("[1] drop == reference Interleaver(keep_and_shift=False), token for token")
assert torch.equal(mine("shift"), ref_shift), f"shift != reference\n{mine('shift').tolist()}\n{ref_shift.tolist()}"
print("[2] shift == reference Interleaver(keep_and_shift=True)")
assert not torch.equal(mine("overwrite"), ref_drop), "guard is vacuous: overwrite matched the reference"
assert not torch.equal(ref_drop, ref_shift)
print("[3] overwrite (historical) differs from the reference on the scrambling case")
print("OK")
