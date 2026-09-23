"""Guard: the opt-in two-stage PersonaPlex text sampler draws what it claims, and is off by default.

``moshi_family.personaplex.text_sampling.two_stage_sample`` (2026-09-23, self-play ablations): the class
{EPAD, PAD, word} is drawn from the full softmax at T=1, then a word at temp / top-k among word tokens.
Checked on 200k draws from fixed logits: class frequencies = the T=1 softmax (not the sharpened one-stage
sampler's), the word conditional = softmax(logits/T) over the top-k WORD tokens, specials never drawn as a
word; and ``LMGen.process_transformer_output`` only uses it when ``text_sample_fn`` is set (the released
sampler otherwise). Fails if the class is drawn at the word temperature ([1]) or top-k includes the specials ([2]).

Run: CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_two_stage_sampler.py
"""

import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe/speech_llm/full_duplex"))

import torch  # noqa: E402

from moshi_family.personaplex.text_sampling import TEXT_EPAD_ID, TEXT_PAD_ID, two_stage_sample  # noqa: E402
from moshi_family.utils.sampling import sample_token  # noqa: E402

FAIL = []


def check(ok, msg, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {msg} {'' if ok else detail}")
    if not ok:
        FAIL.append(msg)


torch.manual_seed(0)
V, N, T, K = 60, 200_000, 0.7, 25
logits = torch.randn(V) * 2
logits[TEXT_PAD_ID], logits[TEXT_EPAD_ID] = 5.0, 3.0  # PAD the mode, EPAD a minority class
L = logits.expand(N, 1, 1, V).contiguous()
out = two_stage_sample(L, temp=T, top_k=K).reshape(-1)
check(out.shape == (N,), "shape [*] like sample_token", str(out.shape))
p1 = torch.softmax(logits, -1)
f = lambda i: float((out == i).float().mean())  # noqa: E731
print("[1] class frequencies = the full softmax at T=1")
for name, i in (("EPAD", TEXT_EPAD_ID), ("PAD", TEXT_PAD_ID)):
    check(abs(f(i) - float(p1[i])) < 0.005, f"P({name}) {f(i):.4f} vs softmax {float(p1[i]):.4f}")
one = sample_token(L, True, T, K).reshape(-1)
f_one = float((one == TEXT_EPAD_ID).float().mean())
check(
    f_one < 0.8 * f(TEXT_EPAD_ID),
    "the one-stage sampler draws EPAD clearly less (non-vacuous)",
    f"{f_one:.4f} vs {f(TEXT_EPAD_ID):.4f}",
)
print("[2] word conditional = softmax(logits/T) over the top-k word tokens")
w = logits.clone()
w[[TEXT_EPAD_ID, TEXT_PAD_ID]] = -float("inf")
top = torch.topk(w, K).indices
q = torch.zeros(V)
q[top] = torch.softmax(w[top] / T, -1)
words = out[(out != TEXT_EPAD_ID) & (out != TEXT_PAD_ID)]
emp = torch.bincount(words, minlength=V).float() / len(words)
check(float((emp - q).abs().max()) < 0.01, "word distribution matches", f"max dev {float((emp - q).abs().max()):.4f}")
check(bool(torch.isin(words, top).all()), "only top-k word tokens drawn")
print("[3] off by default: LMGen uses it only when text_sample_fn is set")
src = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/personaplex/models/lm.py").read_text()
check(
    'getattr(self, "text_sample_fn", None)' in src and "if text_sample_fn is not None and self.use_sampling" in src,
    "hook is opt-in",
)
print()
if FAIL:
    print(f"FAILED: {FAIL}")
    sys.exit(1)
print("all checks passed")
