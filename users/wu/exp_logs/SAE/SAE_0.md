# SAE Phase 0 — Foundations (0a representation audit, 0b lexicon/phonemization)

## Approach

**1. §0a representation audit on frozen BEST-RQ (layer 6, 25 Hz).** Per-frame metrics
(purity, PNMI, oracle-map PER, H(phi|u), dedup ratio) over k-means unit streams scored against MFA
gold on 500 dev-clean utterances, plus supervised continuous probes on the same features; k-means and
the oracle map are fit on the train split only, scored framewise -> dedup -> drop-SIL -> S/I/D.

| K / probe | oracle-map PER | frame-acc | sub | ins | del |
|---|---|---|---|---|---|
| k-means 500 | 0.632 | 0.616 | 0.245 | 0.375 | 0.012 |
| k-means 4000 | 0.591 | 0.647 | 0.204 | ~0.37 | -- |
| + w2v-U preprocessing (VAD+pool+PCA) | 0.594 | -- | -- | -- | -- |
| linear probe (supervised) | 0.145 | 0.832 | -- | 0.056 | -- |
| MLP-256 probe | 0.140 | 0.841 | -- | -- | -- |

**2. Same audit on wav2vec2-Large-lv60 `hidden_states[15]`, rate-matched.** Identical metrics, same
500 utterances, wav2vec2 pooled x2 from 50 -> 25 Hz so the two encoders are compared at matched rate
and matched K (native 50 Hz over-segments and is insertion-inflated, not comparable).

| metric | BEST-RQ L6 | w2v2-L15 pool2 | w2v2 raw 50 Hz |
|---|---|---|---|
| oracle-PER K=500 | 0.632 | 0.602 | 0.763 |
| oracle-PER K=4000 | 0.591 | 0.530 | -- |
| linear probe PER | 0.145 | 0.127 | 0.200 |
| MLP-256 probe PER | 0.140 | 0.131 | -- |
| ins @ best K | ~0.37 | 0.363 | 0.669 |

**3. §0b lexicon + corpus phonemization.** Folded stress-free LibriSpeech lexicon -> phoneme
inventory; LM corpus phonemized to boundary-free T_phi with first-pron lookup and Sequitur G2P for
the OOV tail. Result: inventory = exactly the 39 ARPAbet phones + `[SILENCE]` + `[UNKNOWN]`,
identical to `repr_audit.ARPABET_39`; corpus 40,418,261 -> 39,630,169 lines (1.59 GB), base lexicon
covers 99.75 % of word tokens, residual unresolved types dropped rather than `[UNKNOWN]`-tagged
(~0.1 % of tokens).

## Conclusion

1. (1) **The discretization, not the representation, is the ceiling**: hard k-means with an *oracle*
   label map still reads 0.63 PER while a supervised linear map on the same features reads 0.145, and
   the error is insertion-dominated (0.375) with insertions stuck at ~0.37 for every K in 500..4000 —
   over-segmentation is a structural property of hard assignment + dedup, not a granularity problem.
2. (1) The decoder-independent statement of that loss is the **framewise gap** — unit id caps
   frame-acc at 0.616 (K=500) / 0.647 (K=4000) against 0.832 continuous; the headline "0.63 vs 0.145,
   three quarters of phone info discarded" compares two *decoders* and must not be quoted as an
   information content.
3. (1) §1a(i)'s oracle ceiling (0.59 even with w2v-U preprocessing) already fails the Phase-1 gate of
   dev-other PER <= 50 %, which closed §1a(i) as a bootstrap on a bound.
4. (2) The encoder swap buys a real but small margin at matched rate and K (~0.03-0.06 oracle-PER,
   0.018 linear probe) and the insertion floor is shared, so §0a alone cannot explain §1c's 0.75 ->
   0.17 GAN gap; only the GAN reveals it.
   *(An earlier write-up compared wav2vec2's best-K 0.530 to BEST-RQ's K=500 0.632 and called the edge
   ~0.10 — WRONG; matched-K it is 0.03-0.06.)*
5. (3) The lexicon inventory and the audit's phone set validate each other exactly, so unit-vs-phone
   scoring shares one symbol convention across the whole program.

## Catalog

Paths are workspace-relative. `W/` = `work/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/`.

| artifact | path |
|---|---|
| §0a metric core + gold builder | `recipe/i6_experiments/users/wu/experiments/ssl/analysis/repr_audit.py` |
| §0a real audit entry point (reproduces every number above) | `.../ssl/analysis/real_repr_probe.py --stage both` |
| §0a w2v2 audit job | `W/eval/W2v2ReprAuditJob.BFLCTjxcvrNE` |
| BEST-RQ frozen ckpt (sibling workspace, hash-frozen) | `/e/project1/spell/wu24/2026-06-17_ssl/work/i6_core/returnn/training/ReturnnTrainingJob.iDPxBJeb35l8/output/models/epoch.100.pt` |
| §0b lexicon / inventory | `output/sae/0b/{ls_lexicon_folded.xml.gz,phoneme_inventory.txt}` |
| §0b phonemized corpus T_phi | `work/i6_experiments/users/wu/experiments/posterior_hmm/data/phon_lm/TextToPhonemeJob.THKMON3k9LJQ` |

## Verifier feedback

**2026-07-14.** Re-ran the committed §0a entry point on the cached layer-6 features: oracle-PER 0.632
(sub .245 / ins .375 / del .012), frame-acc 0.616, linear probe 0.832/0.145, MLP-256 0.841/0.140 —
every digit reproduces. Protocol audited fair (identical scoring pipeline both arms, utterance-level
80/20 split, k-means and oracle map fit on train only).
- The "~3/4 of phone info discarded" framing compares decoders, not information contents — quote the
  framewise Bayes gap plus H(phi|u) instead (folded into conclusion 2).
- Gaps, none conclusion-threatening: the layer sweep skipped the plan's anchor layer 9; the `--vad`
  arm deletes rVAD-silence frames from the *gold reference* too, so 0.594 is mildly contaminated (the
  headline no-VAD 0.632 is unaffected); dev-clean only, replicate the two headline rows on dev-other.
- `load_state_dict(strict=False)` does not assert empty missing/unexpected despite the comment
  claiming an exact load.
