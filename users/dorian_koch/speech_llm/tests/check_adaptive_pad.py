"""Per-example pad weighting must equalise rows, and must be a no-op when unset."""
import sys, os
sys.path.insert(0, "recipe"); sys.path.insert(0, "recipe/sisyphus"); sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")
import torch
from moshi_family.train_data_common import duplex_ce_loss, TEXT_PADDING_ID, TEXT_ROW

fails = []
def ck(c, l, e=""):
    print(f"  [{'ok' if c else 'FAIL'}] {l}{'  ' + e if e else ''}")
    if not c: fails.append(l)

B, T, DEP_Q, CARD, TCARD = 2, 100, 4, 32, 64
torch.manual_seed(0)
class O: pass

def build(fracs):
    """row b has fracs[b] padding -- a deliberately MIXED batch, like Fisher+QA."""
    codes = torch.randint(1, CARD, (B, 2 + DEP_Q, T))
    for b in range(B):
        row = torch.full((T,), 5000 % TCARD, dtype=torch.long)
        row[: int(fracs[b] * T)] = TEXT_PADDING_ID
        codes[b, TEXT_ROW] = row
    o = O()
    o.text_logits = torch.randn(B, 1, T, TCARD)
    o.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
    o.logits = torch.randn(B, DEP_Q, T, CARD)
    o.mask = torch.ones(B, DEP_Q, T, dtype=torch.bool)
    return o, codes

MASK = torch.ones(B, T, dtype=torch.bool)
# 1. unset == exactly the old behaviour
o, c = build([0.9, 0.5])
a = duplex_ce_loss(o, c, MASK, dep_q=DEP_Q, other_audio_weight=0.02, text_pad_weight=0.3)
b_ = duplex_ce_loss(o, c, MASK, dep_q=DEP_Q, other_audio_weight=0.02, text_pad_weight=0.3,
                    text_pad_target_share=None)
ck(torch.equal(a, b_), "text_pad_target_share=None is bit-identical to the fixed weight")

# 2. with it set, EACH ROW's pad share hits the target regardless of its padding fraction
import torch.nn.functional as F
def row_pad_share(o, c, share, b):
    tl = o.text_logits[:, 0].reshape(-1, TCARD)
    ce = F.cross_entropy(tl, c[:, TEXT_ROW].reshape(-1), reduction="none").view(B, T)
    tgt = c[:, TEXT_ROW]
    is_pad = tgt == TEXT_PADDING_ID
    n_pad = is_pad[b].sum().float(); n_word = (~is_pad[b]).sum().float()
    w = (share * n_word) / ((1 - share) * n_pad.clamp(min=1))
    w = w.clamp(1e-3, 1.0)
    pad_mass = (ce[b][is_pad[b]] * w).sum(); word_mass = ce[b][~is_pad[b]].sum()
    return float(pad_mass / (pad_mass + word_mass))

for share in (0.5, 0.3):
    o, c = build([0.9, 0.5])
    s0, s1 = row_pad_share(o, c, share, 0), row_pad_share(o, c, share, 1)
    # CE per position is ~equal in expectation, so count-balancing gives ~the target share
    ck(abs(s0 - s1) < 0.08, f"rows with 90% vs 50% padding land at the SAME share (target {share})",
       f"{s0:.3f} vs {s1:.3f}")

# 3. it must actually CHANGE the loss vs the fixed weight (non-vacuous)
o, c = build([0.9, 0.5])
fixed = duplex_ce_loss(o, c, MASK, dep_q=DEP_Q, other_audio_weight=0.02, text_pad_weight=0.3)
adapt = duplex_ce_loss(o, c, MASK, dep_q=DEP_Q, other_audio_weight=0.02, text_pad_weight=0.3,
                       text_pad_target_share=0.5)
ck(not torch.equal(fixed, adapt), "setting the share changes the loss", f"{float(fixed):.4f} vs {float(adapt):.4f}")

# 4. an all-padding row must not vanish entirely (floor), nor dominate
o, c = build([1.0, 0.5])
_, aux = duplex_ce_loss(o, c, MASK, dep_q=DEP_Q, other_audio_weight=0.02, text_pad_weight=0.3,
                        text_pad_target_share=0.5, return_components=True)
ck(aux["text_loss"] > 0, "an all-padding row still contributes (floored, not zeroed)")
print("\nFAILURES:", fails or "none")
sys.exit(1 if fails else 0)
