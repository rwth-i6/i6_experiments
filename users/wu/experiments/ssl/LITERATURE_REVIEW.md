# Text-Prior Injection for Syllable-Level Speech SSL - Literature Review

*Date: 2026-06-20. Scope: how (and where) to inject an unpaired-TEXT prior into our frozen-BEST-RQ-L9 + trainable-CIF + segment-level masked-prediction stack, to serve unsupervised ASR and a speech-LLM interface. Synthesizes 5 family surveys, 2 embedding deep-dives, and an adversarial critique. WER is treated only as a cheap proxy; the real target is unit->text mappability.*

## 0. TL;DR and recommendation

**The bet is directionally sound but misbadged, and not identifiable on its own.** Moving the text prior earlier and keeping it light IS the verified 2024-2025 frontier (REBORN shapes boundaries; JSTTI/SylCipher shape the target/segmenter; SylCipher reports up to 40% relative CER reduction at *syllable* rate, the closest published analogue to our exact setup). But three results constrain the strong form of the bet:

1. **Identifiability.** Matching unit inventory / unigram / rate to text is *necessary but not sufficient*: it is invariant to any code->phoneme relabeling, so it cannot pin which code means /k/. Only **directional, higher-order sequence statistics** (positional-unigram + skipgram = PUSM) break that permutation symmetry. SylCipher's recovery theorem needs PUSM + an entropy constraint + **invertible encoders** — not marginals alone, and its ablation shows PUSM gives the largest gains (44%/23% rel. over length-only).
2. **The mapping does not become trivial.** Even with the full phonotactic prior, wav2vec-U still needs the GAN *and* self-training (+25-40% rel.) and a 5-seed sweep. Honest re-scope: a light prior makes the existing PUSM/GAN mapper *easier/more stable/fewer-seed*, it does not *replace* it.
3. **The binding constraint is our TARGET, not the segmenter.** Our seg-diag already proved frozen-mean + 128-code is non-discriminative at coarse rate (gold boundaries beat uniform by ~1.5%; CIF captures ~none). A distribution/embedding prior atop a collapsing target violates the invertibility precondition the theory needs — there is nothing to bite on. Codebook-init priors wash out under training (DinoSR).

**Recommended sequence (effort/payoff in §6):**
1. **Fix the TARGET first, text-free** — replace masked-pred-over-frozen-128-kmeans with a Sylber/SylBoost segment-mean self-distillation target (EMA/staged-swap teacher, CIF supplies the segmentation), enlarge the analysis codebook. *High payoff, attacks root cause.*
2. **Rate + inventory prior (cheap text-light legs)** — set CIF quantity-loss target from the *syllabified*-text length distribution (Pyphen), refit codebook size = text syllable inventory (~1-2k) with `<OOV>` bucketing.
3. **PUSM skipgram/positional matching** as an auxiliary loss between deduped CIF codes and syllabified-text n-grams (non-adversarial) — this is the identifiability leg.
4. **Embedding variant done right** — masked-predict *into* a frozen phoneme/syllable-LM input-embedding space (phonotactic target geometry), keep separate directional in/out vectors, quotient out frequency before any cloud comparison.
5. **GW/MUSE cloud-alignment: diagnostic only, never a training objective.**

Use **syllabified** text (not G2P phonemes) for the 80/120 ms arms; phonemes are rate-incompatible there. Validate with a downstream PUSM/mapping metric + an oracle ceiling from the cached gold MFA alignments — **not CTC-WER alone**.

## 1. Our setup, goal, and the specific bet

**Setup.** Frozen 12-layer BEST-RQ Conformer (LS-960, no labels); layer-9 features at 25 Hz / 40 ms. A **trainable CIF segmenter** consumes those frozen features and emits variable-rate "syllable-ish" tokens (alpha-weighted convex segment means) with a quantity loss for length control; two arms ~80 ms (12.5 Hz) and ~120 ms (8.33 Hz). On top: a 9-layer "high" Conformer with BEST-RQ-style **masked prediction** against a **frozen 128-cluster k-means codebook** fit on frozen layer-9 *frames*. Downstream sanity only: CTC FT on LS-100 (base single-level BEST-RQ FT = 6.46/14.33 dev-clean/other).

**Goal (NOT supervised ASR).** (1) a starting point for **unsupervised ASR** (no paired speech-text); (2) maybe a better **tokenizer/interface for a speech LLM**. WER is a cheap proxy; the real metric is unit->text mappability.

**What we already measured (seg-diag vs gold MFA).** The masked-pred-over-frame-kmeans *target* does not reward good segmentation: gold phone boundaries beat uniform by only ~1.5% quant-error, and CIF captures ~none of it. **The binding constraint is the TARGET REPRESENTATION** (frozen-mean + 128 codes can't make coarse units discriminative), not the segmenter. Segmentation helps token purity most at *fine* (phone) rate and washes out at coarse rate with this target. At 40 ms a phoneme target ≈ a fixed rate-2 subsample; only syllable/word aggregation is genuinely new. Deduped CIF sequences sit near phoneme length-ratio ~1.0 but with ~114-125 active codes (vs 39 phonemes) and ~0.9 bits higher transition entropy.

**The bet.** Inject a **text prior already during SSL/segmentation**: shape the unit *inventory size*, the token *rate*, and the *phonotactic/n-gram structure* toward (phonemized/syllabified) unpaired text, so the downstream unit->text mapping becomes near-trivial. Prefer a **light** prior (inventory + rate + phonotactic LM reward / distribution matching / embedding shaping) over a full joint GAN; keep the representation otherwise text-free.

## 2. Taxonomy: where can a text prior enter?

| Injection point | Representative methods | Text signal used | Needs pairs? | Fit to us |
|---|---|---|---|---|
| **Unit->text MAPPING** (representation/segmentation stay text-free) | wav2vec-U, wav2vec-U 2.0, EURO, Liu2018/Chen2019, EODM, PUSM, GW/MUSE | n-gram phonotactic (GAN / explicit marginal / OT coupling) | no (phonemizer/syllabifier only) | Conservative default; literature *validates* this for sharp ASR. Caps out because our target is weak. |
| **SEGMENTATION** (text reward shapes boundaries) | REBORN (4-gram phoneme-LM perplexity RL reward), SylCipher soft-pooler (mild) | n-gram phonotactic reward + length term | no (phoneme/syll LM) | Direct analogue to CIF, but needs a working mapper first (chicken-and-egg) + RL variance. |
| **TARGET / REPRESENTATION** (text shapes the predicted space) | JSTTI, SylCipher (shared-encoder MLM), SpeechT5 (shared VQ), LAST, EODM | masked-LM / shared codebook / LM-reward / marginal match | mostly no (SpeechT5/LAST/SylCipher pair-free); JOIST/Maestro/USM **yes** | Reuses our masked-pred machinery; the embedding-based answer lives here. Must fix target first. |
| **REPRESENTATION+TARGET via PAIRED aligner** (strong but supervised) | Maestro, Maestro-U, JOIST, tts4pretrain, USM-MOST, PAST, TASTE, DyCAst | upsampled text embeddings + RNN-T/CTC/forced-align | **yes** | Upper bound to *approximate*, not adopt. Only the pair-free residue (fixed-rate upsampling, rate=text-length) transfers. |
| **LM / DECODING stage only** | TWIST, AudioPaLM | LM weight-init / shared vocab | no (TWIST) / yes (AudioPaLM) | Cheapest baseline: defer text to the LM. Does not make units text-mappable. |

## 3. Family reviews

### 3.1 Unsupervised ASR & the unit->text mapping

The one durable signal across this entire family is the **phonotactic structure of phonemized/syllabified unpaired text**: injected as a GAN discriminator over real phoneme n-grams (wav2vec-U/EURO), as a 4-gram-phoneme-LM perplexity *reward* (REBORN), or as direct n-gram/skipgram *marginal matching* (EODM, PUSM, SylCipher). Historically text entered **only at the unit->text mapping** stage; the clear 2024-2025 trend moves it **earlier** — into segmentation (REBORN) and into the target/representation (JSTTI, SylCipher). This validates the bet's *direction*. Two stable GAN-free routes exist: explicit n-gram/skipgram/positional distribution matching, and joint masked-LM on a shared speech-text codebook. GAN scores best PER but is unstable and fails to converge on Mandarin (where SylCipher succeeds). **Inventory matching** (set codebook size = text token inventory) is a recurring cheap, high-leverage trick. But distribution matching only constrains low-order marginals, so as a standalone target it is weak (EODM/PUSM TIMIT PER ~42 vs wav2vec-U 11.3) — it makes the *downstream mapping easier*, it does not replace it.

| Method | Year | Mechanism (1-line) | Inject point | Pairs? | Key number |
|---|---|---|---|---|---|
| wav2vec-U | 2021 | GAN: seg-mean feats -> phoneme dist vs phonemized text; +GP/smoothness/diversity | mapping | no | LS test-c/o 3.4/5.9 |
| wav2vec-U 2.0 | 2022 | end-to-end, stride-3 conv replaces segmentation; +MFCC-kmeans aux | mapping | no | test-other 6.3 (self-train) |
| REBORN | 2024 | RL boundary segmenter; reward = phoneme-LM perplexity drop + edit + len | **segmentation** | no | LS-100 test-c PER/WER 8.3/12.5 |
| EODM | 2019 | match predicted-phoneme n-gram marginals to text LM (no GAN) | target | no | TIMIT PER 41.6 |
| PUSM/ESPUM | 2023 | L1 match of positional-unigram + skipgram marginals (no GAN) | mapping | no | TIMIT PER ~43; seg-F1 88.4 |
| JSTTI | 2024 | shared-Transformer S2S+T2T masked infilling, word-level, G2P-free | target | no | word WER 20-23 |
| SylCipher | 2025 | **syllable**-rate: soft-pooler+VQ, shared MLM + PUSM, K=text inventory | target+seg | no | LS CER 35.9 unmatched / 17.5 self-train |
| EURO | 2022 | ESPnet wav2vec-U reimpl; WavLM > w2v2; layer choice matters | mapping | no | (toolkit) |

**Fit to us.** Best-fitting in order: (1) **SylCipher** as the overall blueprint — only published coarse/syllable-rate UASR, matches codebook size to text inventory, joint-MLM + PUSM, no GAN, cross-lingual. (2) **PUSM/EODM** as the concrete differentiable GAN-free phonotactic regularizer on deduped CIF codes. (3) **REBORN's phonotactic-LM reward** if we want to shape CIF *boundaries* — but it needs a serviceable mapper first. Caveat: all are capped by our weak frozen-mean+128 target and assume a roughly commensurate inventory/rate.

### 3.2 Joint speech+text / text-injection pretraining

The hard pattern: **every method that achieves a strong speech<->text binding buys it with paired data** — an RNN-T/forced-alignment aligner (Maestro, Maestro-U, JOIST, USM-MOST) or a supervised TLM/STM loss (SLAM). The good news for the bet: **duration/alignment precision is NOT the bottleneck** — JOIST shows fixed phoneme repetition ≈ forced-align upsampling, and Maestro-U shows uniform duration (45.0 CER) is within 2.3 of a learned duration model (42.7). What *is* load-bearing is cross-modal **consistency/shared-space binding** (Maestro-U: removing it collapses CER 42.7->58.7; SLAM: removing the paired alignment loss makes joint training *hurt*). **Exactly one** member binds without pairs: **SpeechT5's shared vector-quantization codebook + random speech/text code mix-up + diversity loss**, trained on unpaired speech and unpaired text — but pair-free binding is empirically **weak** (SpeechT5 w/o-joint 4.4->4.6; SLAM shared-trunk underperforms speech-only). Phonemes > word-pieces for injection.

| Method | Year | Binding mechanism | Pairs? | Pair-free lesson for us |
|---|---|---|---|---|
| MAESTRO | 2022 | RNN-T aligner + resample/refine + L_MM | yes | architecture only |
| Maestro-U | 2022 | cross-lingual paired aligner; consistency load-bearing | partial | uniform duration ≈ learned |
| JOIST | 2022 | joint RNN-T, param-free duration (fixed/random rep) | yes | drop the learned aligner; fixed repetition fine |
| SLAM/mSLAM | 2021/22 | shared Conformer + supervised TLM/STM | partial->yes | pure shared-trunk MLM can *hurt* |
| SpeechT5 | 2021/22 | **shared VQ codebook + random code mix-up + diversity** | **no** (binding) | the one importable pair-free primitive |
| tts4pretrain | 2021 | TTS intermediate features + seq loss | yes | least applicable |
| USM-MOST | 2023 | **BEST-RQ** + text via learned duration upsampler | yes | proves text composes with BEST-RQ |

**Fit to us.** Only **SpeechT5's shared-codebook binding** is directly usable: add a text branch (phonemized/syllabified text, upsampled to CIF rate by a *fixed/sampled* ratio — justified pair-free by JOIST + Maestro-U), quantize into the same/tied codebook the high-Conformer predicts into, masked-predict it, with code mix-up + diversity. This shapes inventory and rate with no pairs. Honest caveats: (a) pair-free shared-codebook binding is *weak*, so reinforce with a phonotactic/PUSM term; (b) our codebook is **frozen** frame-kmeans — SpeechT5's is learned, so consider a trainable text-side codebook or a learned text->code projection; (c) this family never touches the segmenter, so our rate/inventory-from-text twist is genuinely outside its playbook.

### 3.3 Syllable/segment tokenizers & fixing the TARGET

This family is the **direct fix for our root cause**. Every in-representation syllable tokenizer here is **text-free**, and they converge on one objective for a segmentation-rewarding target: **regress each frame to the SEGMENT-MEAN of a LEARNED/self-distilled representation** (SylBoost MSE-to-teacher-segment-mean; Sylber frame-wise MSE to averaged teacher embedding; speaker-disentangled L2-normalized MSE). A frozen-frame k-means target (our 128-code masked-pred, vanilla HuBERT) provably does *not* reward segmentation — its optimum is per-frame identity — which is exactly our seg-diag result. **Rate without gold lengths** is solved either by one penalty knob (DPDP's λ, our CIF quantity loss) or by thresholds (Sylber, ZeroSyl norm-prominence). **Frame norm** is an almost-free boundary signal. Collapse trap: self-distillation needs a stop-gradient teacher (Sylber uses a staged teacher *swap*, not EMA — correcting our earlier note). ZeroSyl is a deflationary reality check: getting syllable-*rate* segmentation is cheap; making units **text-mappable** is the hard part (our bet).

| Method | Year | Target / segmenter | Text? | Lesson for us |
|---|---|---|---|---|
| SyllableLM (LossPred+SylBoost) | 2024 | loss-affinity min-cut + regress to iteratively-sharpened segment-mean | no | the segmentation-rewarding target we lack |
| Sylber | 2024 | frame-wise MSE to teacher segment-mean; staged teacher swap | no | best template for fixing our target |
| SD-HuBERT | 2023 | sentence-level distillation; norm dips at boundaries (L11) | no | bootstrap/diagnostic only |
| Speaker-disentangled HuBERT | 2024 | frame-wise L2-norm MSE + speaker-perturb augmentation | no | cuts speaker-driven over-fragmentation |
| VG-HuBERT | 2022-23 | visual grounding; min-cut on self-sim | **image pairs** | skip (violates unpaired premise) |
| DP-Parse / DPDP / GradSeg | 2022-23 | frequency / duration-penalty / norm-gradient segmentation | no | rate by a single penalty; free boundary priors |
| ZeroSyl | 2026 | training-free norm-prominence peaks (WavLM) | no | syllable rate is cheap; mappability is the bet |
| HuBERT k-means | 2021 | frame-rate k-means target | no | ≈ our current broken target |

**Fit to us.** Replace masked-pred-over-frozen-frame-kmeans with a **Sylber/SylBoost segment-mean self-distillation target**, keeping the frozen encoder, the trainable CIF (it supplies A(i) via the convex means v_j we already compute), and the CIF quantity loss as the rate knob aimed at the text syllable rate (~4-5 Hz). Bootstrap the teacher from current CIF firing or a free frame-norm-prominence signal; add a stop-gradient/projector to avoid collapse; optionally add speaker-perturbation consistency to cut the 114-125-codes-vs-39-phonemes over-fragmentation. Keep a small codebook only for analysis (purity/PNMI). On the text prior, this family *cautions where*: fix the target text-free first, then use text mainly as inventory + rate + a *light* PUSM reward; keep heavy distribution-matching at the unit->text stage.

### 3.4 Distribution-matching / OT / LM-reward priors

This is the "light phonotactic-prior toolbox." Two axes matter: **(a) GAN vs non-adversarial** (wav2vec-U/Liu2018 need multi-seed sweeps and can fail to converge; EODM/PUSM/GW are explicit, stable, slightly worse final PER but better segmentation/stability), and **(b) n-gram order** — unigram/inventory matching is *gameable* (correct marginals, wrong sequences), and our measured +0.9-bit transition-entropy gap is a *2nd-order* discrepancy, so **skipgram/positional matching is the targeted tool**. Every method imposes a **length/rate-matching** constraint (smoothness, positional terms, explicit length loss, comparable inventory sizes) — which means our CIF quantity-loss target should be set from the text token-length distribution *for free*. The **embedding bridge is closed-form and offline**: SPPMI+SVD (Levy-Goldberg) turns co-occurrence counts of both CIF codes and the text inventory into comparable metric spaces, and **Gromov-Wasserstein** then aligns them using only intra-domain geometry (no shared space, no adversary). MWER/expected-risk is the template proving we *can* backprop a non-differentiable phonotactic-LM reward (the REBORN move).

| Method | Year | Mechanism | Adversarial? | Order | Fit |
|---|---|---|---|---|---|
| wav2vec-U GAN | 2021 | discriminator over phonemized text; GP/smoothness/diversity | yes | implicit n-gram | full GAN the bet avoids |
| EODM | 2019 | match predicted n-gram marginals to LM | no | ≤5-gram | cleanest non-GAN ancestor; weak |
| PUSM/ESPUM | 2023 | L1 positional-unigram + skipgram co-occurrence match | no | skipgram + positional | **the light prior we want** |
| JSTTI / SylCipher | 2024/25 | shared-encoder MLM (implicit) + explicit PUSM + length/entropy | no | n-gram + positional | SylCipher = our architecture, works |
| REBORN | 2024 | phoneme-LM perplexity RL reward on boundaries + length | no (RL) | 4-gram | shapes CIF boundaries; needs mapper |
| Gromov-Wasserstein | 2018 | match intra-domain distance matrices -> soft coupling | no | relational | **best embedding alignment**; diagnostic |
| SPPMI+SVD | 2014 | closed-form embeddings from co-occurrence | no | bigram-PMI | the enabling bridge |
| MWER / expected-risk | 2017-18 | policy-gradient over non-diff sequence reward | n/a | any | template for LM-reward backprop |

**Fit to us.** Priority: (1) **rate prior** — CIF quantity-loss target from the text length distribution; (2) **inventory prior** — refit codebook with size = phonemized/syllabified-text inventory + `<OOV>` bucketing; (3) **non-adversarial phonotactic prior** — PUSM skipgram/positional matching of deduped CIF code co-occurrence to text n-grams, directly attacking the +0.9-bit gap; (4) **best embedding instantiation** — SPPMI+SVD embeddings for both inventories, GW-align them for a near-trivial code->token map using only intra-domain geometry. Reserve REBORN-RL and GAN as later refinements. Caveat: skipgram-match and GW assume comparable inventory size/rate, so (1)-(2) must precede (3)-(4).

### 3.5 Discrete tokens for speech LLMs

Two meanings of "semantic/text-aligned" must not be conflated: **(a) SSL-content alignment** (AudioLM, SpeechTokenizer, Mimi, LM-SPT — distill toward a frozen SSL/ASR feature space, *no transcripts*) vs **(b) literal text/phonetic alignment** (PAST, TASTE, DyCAst — supervise with transcripts/phonemes/CTC, *requires pairs*). Our no-pair constraint rules out (b) as-is. Consensus: linguistic **content lives in a clean low-rate semantic stream**, acoustic detail in RVQ — validating our single-stream segment-level direction. Every semantic-distillation codec reports tension between the semantic objective and reconstruction; **Mimi's fix (a parallel dedicated semantic VQ, isolated from reconstruction)** is the cleanest — our masked-pred-over-128-codes is reconstruction-flavored and weak. **Sylber-SLM** shows coarse units help the LM only with a **large vocabulary (~20k)** — plausibly *why* our 128 codes washed out at coarse rate. **LAST** is the existence proof that a **text prior can be injected into the tokenizer with NO pairs** — purely as gradients from a *frozen text/LM next-token objective* (sWUGGY 71.8 vs 63.5; WER 6.08 vs 6.83).

| Method | Year | Mechanism | Pairs? | Lesson |
|---|---|---|---|---|
| AudioLM / AudioPaLM | 2022/23 | semantic (k-means SSL) vs acoustic split; LLM vocab | no / yes | framing; interface |
| TWIST | 2023 | SpeechLM warm-started from text-LM; plain HuBERT units | no | defer text to LM init (cheap baseline) |
| SpeechTokenizer / Mimi | 2023/24 | RVQ-1 distilled from HuBERT/WavLM; **isolate semantic VQ** | no | dedicated isolated semantic objective |
| LAST | 2024 | frozen text-LM next-token gradients shape tokenizer | **no** | **pair-free text prior into the tokenizer** |
| LM-SPT | 2025 | distill via resynthesis into Whisper-encoder space | no | embedding-target option (heavier) |
| PAST | 2025 | CTC + phoneme-classification aux heads | **yes** | supervised upper bound to approximate |
| TASTE / DyCAst | 2025/26 | one token per ASR text token / char-aligned boundaries | **yes** | pair-free residue = rate-from-text-length |
| Sylber-SLM | 2025 | syllable units (4-5 Hz) + k-means 20k | no | coarse units need LARGE vocab |

**Fit to us.** Best fit: **LAST-style frozen-LM reward** — attach a frozen text/phoneme/syllable LM on top of deduped CIF codes and backprop its next-token NLL into the CIF segmenter + code assignment; no transcripts, representation stays text-free, far lighter than a GAN. Supporting moves: **fix the target** (Mimi/SpeechTokenizer/LM-SPT: dedicated isolated semantic distillation + much larger codebook, per Sylber-SLM's 20k), and a **rate/length prior** (pair-free slice of TASTE/DyCAst). The strongest text-alignment results all use pairs and are the supervised upper bound to *approximate*, not adopt — no paper yet proves a fully pair-free inventory+rate+phonotactic prior makes the mapping near-trivial, so treat the bet as plausible-but-unvalidated and prototype LAST-reward + a stronger target as the cheap test.

## 4. SPECIAL: do text embeddings encode frequency & n-gram structure?

### 4.1 Frequency — STRONGLY encoded, HIGHLY recoverable (double-edged)

Token frequency is among the most robustly encoded and easily extracted signals in any embedding space. Two independent channels carry it: **(1) NORM** — L2 norm grows ~linearly with log frequency (Schakel 2015); the sharper result is that *squared* norm encodes the KL information gain of a token's co-occurrence vs the unigram, with frequency a confound (Oyama 2023). **(2) DOMINANT DIRECTIONS** — the top 1-3 PCA directions of the embedding matrix encode unigram frequency; All-but-the-Top removes mean + top D~d/100 directions because they *are* the frequency subspace (Mu & Viswanath 2018). In contextual models, representation degeneration produces an anisotropic cone with frequency recoverable from top PCs (Gao 2019; Rajaee 2021).

**Implication for us.** With only ~39 phonemes (or ~1-2k syllables), frequency is captured by **literally 1-3 numbers** (norm + top-2 PCA). So matching it is *trivial* — but it **contaminates the geometry**: naive embedding distribution-matching risks **matching the frequency cone, not phonotactic structure**, unless frequency is explicitly quotiented out (center + ABTT) *before* claiming any phonotactic match.

### 4.2 N-gram / phonotactic — PARTIALLY encoded, recoverable only approximately and not fully linearly

Levy & Goldberg prove SGNS implicitly factorizes shifted PMI: ⟨word_vec, context_vec⟩ ≈ PMI(w,c) + const. With window 1, contexts *are* adjacent tokens, so the bilinear form **linearly reconstructs bigram/transition PMI**. Three hard caveats: **(i)** this needs the **separate input and output (context) matrices** — directional P(b|a) lives in ⟨in_a, out_b⟩; a single *tied* embedding keeps only an undirected similarity proxy and loses direction. **(ii)** the d-dim truncation is a rank-d approximation; for 39 phonemes the 39×39 PMI matrix is reconstructed well even at d≈16-32, but PMI is a noisy estimator for rare bigrams. **(iii)** skip-gram bags the window, so genuine **trigram+ phonotactics is NOT linearly recoverable** from static vectors — it needs a sequence model / n-gram LM. The phoneme-specific evidence (Silfverberg 2018) is sobering: unsupervised distributional phoneme embeddings correlate with articulatory feature space only **modestly** (PPMI+SVD Pearson r ≈ 0.17-0.36; word2vec weaker; weakly-*supervised* RNN reaches 0.28-0.46). So an embedding-space prior aligned to text is a **soft prior, not a tight bijection**, weakest exactly where order/feature detail matters. (Standardized yardstick: PWESuite, Zouhar 2024.)

**Implication for us.** The embedding prior should target **inventory size + unigram rate + first-order transition PMI** (keep separate directional in/out vectors), and lean on an **explicit n-gram/LM reward** (or the mapping stage) for everything higher-order.

### 4.3 The optimal embedding-based prior-injection method (ranked, with the winner + identifiability caveat)

1. **WINNER — Predict masked tokens INTO a frozen phoneme/syllable-LM embedding space (target-side), + PUSM on the sequence side (identifiability).** Swap the frozen 128-code k-means target for a target whose *geometry is phonotactic* — the input-embedding table of a small frozen phoneme/syllable LM (or a phoneme2vec space), with masked **classification into LM-anchored codes** (not raw regression, which collapses). This fixes the binding constraint seg-diag identified and makes units inherit phoneme/syllable-like equivalence geometry text-free (the LAST/LM-SPT lever). Then pass deduped CIF sequences and syllabified text through a **shared encoder with weighted MLM + positional-unigram + skip-bigram matching + an entropy/length constraint** — this **pins the correspondence** (SylCipher Theorem 1 recovers the true map to zero KL). *A makes the units worth pinning; B pins them.* This is the SylCipher recipe specialized to us.
2. **Distribution matching through a shared encoder (PUSM / SylCipher MLM)** — the identifiability mechanism on its own; strong precedent, stable, no adversary. (Folded into the winner.)
3. **Codebook init/anchor from phoneme embeddings** — a **1-line ablation, never the mechanism**: init washes out under training (DinoSR shows different inits converge to the same perplexity), and it adds no representational capacity.
4. **Contrastive alignment (InfoNCE/WACO)** — only a **second-stage refiner** once a rough unit->phoneme hypothesis exists; it needs positive pairs, so cold-start collapses to marginal-matching.
5. **MUSE / Procrustes / CSLS / Gromov-Wasserstein cloud-alignment** — **offline DIAGNOSTIC only.** Cloud-alignment assumes near-isometry, which fails even across similar languages (Søgaard 2018); our spaces (114-125 active codes, +0.9-bit entropy vs 39 phonemes) are more distant and non-bijective, so an adversarial/OT map will match cloud *shape*, not identity. Use GW once to check "is the coupling between deduped units and syllables low-entropy?" as a go/no-go gate.

**The identifiability caveat (load-bearing).** Embedding-cloud alignment and inventory/unigram/rate matching only constrain the **marginal** and assume near-isometry. They are **permutation-invariant**: they cannot distinguish code-A=/k/ from code-A=/t/. Only **higher-order, directional sequence statistics** (positional + skipgram phonotactics, or directional bigram-PMI via *separate* in/out vectors) break that symmetry — which is exactly why wav2vec-U/REBORN/SylCipher yield working UASR while speech-vs-text MUSE only worked under matched distributions and conceded non-unique maps.

## 5. Critical assessment of the bet

**Verdict: directionally sound but misbadged, and not identifiable on its own.** The frontier (REBORN, JSTTI, SylCipher) confirms moving the text prior earlier is real, not speculative. But the strong form breaks on three points already summarized in §0: identifiability (marginal/rate/inventory matching is permutation-invariant — necessary but not sufficient; PUSM/skipgram is the load-bearing leg, and SylCipher's theorem additionally requires **invertible encoders**), the mapping does **not** become trivial (wav2vec-U still needs GAN + self-training + a 5-seed sweep), and the binding constraint is **our target**, not the segmenter (a prior atop a non-discriminative collapsing target violates the invertibility precondition; init priors wash out).

**Failure modes to guard against:**

- **Permutation / label-swap non-identifiability** — inventory+rate+unigram matching admits a whole orbit of permuted optima. *Fix:* directional skipgram/positional (PUSM) or directional bigram-PMI.
- **Isomorphism assumption fails** for MUSE/Procrustes/CSLS/GW — even similar-language spaces aren't isomorphic (Søgaard 2018); ours are more distant and non-bijective. *Fix:* diagnostic only.
- **Frequency-cone degeneracy** — with ~39-2048 symbols, frequency is 1-3 dims; naive embedding matching matches the cone, not phonotactics. *Fix:* center + All-but-the-Top before any cloud comparison.
- **Target collapse / invertibility violation** — frozen-mean+128-code is many-to-one and non-discriminative at coarse rate; self-distillation also admits the trivial all-frames-equal solution. *Fix:* fix the target first; stop-gradient/staged-swap teacher.
- **Init-only priors wash out** (DinoSR). *Fix:* ablation only.
- **GAN/RL instability re-imported** if the "light" prior quietly grows a discriminator or REINFORCE reward (5-seed sweeps; REBORN needs a working mapper to even compute its reward). *Fix:* prefer non-adversarial PUSM.
- **Rate/phoneme conflict** — shaping toward a 39-phoneme inventory and phoneme phonotactics contradicts the 80/120 ms *syllable* rate (phonemes need ~25-50 Hz). *Fix:* use rule-based **syllabified** text (Pyphen, ~1-2k) for our arms; reserve G2P phonemes for a hypothetical ≥25 Hz fine arm.
- **Low-order matching is gameable** (EODM/PUSM cap at n≤3-5). *Fix:* skipgram is targeted but still needs a downstream mapper for sharp ASR.

**What the literature implies is necessary-but-not-sufficient.** Necessary: a discriminative/invertible target (our binding constraint); a commensurate inventory and rate; *some* text statistic for the prior to act on. Not sufficient alone: marginal/inventory/rate matching (permutation-invariant), embedding-cloud alignment (isometry fails), codebook init (washes out), or any single low-order match. The honest re-scope of the bet: **light pair-free priors on inventory/rate/structure to make the EXISTING unit->text mapping easier and more stable, not to replace it** — validated against mappability, not WER.

## 6. Recommendation for our project

Ranked action list (sequenced — each step is a precondition for the next):

| # | Action | Effort | Payoff | Why |
|---|---|---|---|---|
| 1 | **Fix the TARGET, text-free.** Replace masked-pred-over-frozen-128-kmeans with a Sylber/SylBoost **segment-mean self-distillation** target on an EMA/staged-swap teacher of the high-Conformer; CIF supplies A(i) via the convex means we already compute; add a collapse guard (stop-grad/projector); **enlarge the analysis codebook** (~thousands, per Sylber-SLM's 20k). | medium | **HIGH** | Attacks the measured root cause; satisfies the invertibility/discriminativeness precondition every text prior needs; improves WER proxy + purity independent of any text work. |
| 2 | **Rate + inventory prior.** Set CIF quantity-loss target from the **syllabified**-text (Pyphen) syllable-count distribution; refit codebook size = text syllable inventory (~1-2k) with long-tail `<OOV>` bucketing. | low | MEDIUM | Pair-free residue of TASTE/DyCAst + SylCipher's K=text-inventory trick; makes the spaces commensurate so identifiability is even possible. Necessary scaffolding, not sufficient alone. |
| 3 | **PUSM skipgram/positional matching** as an auxiliary, non-adversarial loss between deduped CIF code co-occurrence and syllabified-text n-grams, added *after* 1-2. | medium | MEDIUM-HIGH (conditional) | The identifiability leg: skipgram/positional breaks the permutation symmetry marginals cannot; targets the measured +0.9-bit 2nd-order gap; stable, fits RETURNN. |
| 4 | **Embedding variant done right.** Masked-predict **into a frozen phoneme/syllable-LM input-embedding space** (classification into LM-anchored codes, not regression); keep **separate directional in/out vectors** so bigram-PMI is recoverable; quotient out frequency (center + ABTT) before any cloud comparison. | medium-high | MEDIUM | Strongest single embedding lever (LAST/LM-SPT), fixes the weak-target problem too; identifiability still only soft, so pair with #3. Combines with #1 (it *is* a target-side fix). |
| 5 | **Downstream PUSM/wav2vec-U mapping head as the EVALUATION harness.** Measure unit->text mappability (PUSM loss / CER / seed-stability) + an **oracle ceiling from the cached gold MFA alignments** — not just CTC-WER. | medium | MEDIUM | The only way to validate/falsify the bet against its stated goal. |

**Diagnostics (cheap, gate the above):** run **Gromov-Wasserstein** offline (SPPMI+SVD on cached deduped code sequences vs syllabified-text co-occurrence) as a go/no-go check of whether the spaces are even alignable before investing in #3-#4.

**What NOT to do:**

- Do **not** run a full joint wav2vec-U/JSTTI GAN inside SSL as the "light" prior — it's the exact instability the bet wants to avoid (5-seed sweeps + unsupervised model selection; fails on Mandarin).
- Do **not** rely on codebook **initialization** from phoneme embeddings as the mechanism — it washes out (DinoSR). Keep it a 1-line ablation.
- Do **not** use MUSE/Procrustes/CSLS or GW as a *training* objective to pin identity — isometry fails for distant non-bijective spaces. Diagnostic only.
- Do **not** match only unigram/inventory/rate marginals and expect a trivial mapping — permutation-invariant. Add directional skipgram/positional (PUSM).
- Do **not** shape units toward a **39-phoneme** inventory at the 80/120 ms **syllable** arms — rate-incompatible. Use syllabified text; reserve G2P for a ≥25 Hz fine arm (and our decode env already flags phoneme FT as a headache).
- Do **not** add any text prior **before** fixing the frozen-mean+128-code target — seg-diag proves there's almost nothing for the prior to bite on.
- Do **not** compare raw embedding clouds without quotienting out frequency (center + ABTT) — you'll match the frequency cone and mistake it for phonotactics.
- Do **not** use a single **tied** embedding for the phonotactic prior — directional P(b|a) lives in ⟨in_a, out_b⟩.
- Do **not** claim success on **CTC-WER alone** — validate with a downstream mapping metric + oracle ceiling.

## 7. References (grouped by family)

**Unsupervised ASR & unit->text mapping**
- Unsupervised Speech Recognition (wav2vec-U) - Baevski et al. 2021 - arXiv:2105.11084 (NeurIPS 2021)
- Towards End-to-end Unsupervised Speech Recognition (wav2vec-U 2.0) - Liu et al. 2022 - arXiv:2204.02492
- REBORN: Reinforcement-Learned Boundary Segmentation with Iterative Training for Unsupervised ASR - Yeh/Tseng et al. 2024 - arXiv:2402.03988 (NeurIPS 2024)
- Unsupervised Speech Recognition via Segmental Empirical Output Distribution Matching (EODM) - Yeh et al. 2019 - arXiv:1812.09323 (ICLR 2019)
- Completely Unsupervised Phoneme Recognition by Adversarially Learning Mapping... - Liu et al. 2018 - arXiv:1804.00316 (Interspeech 2018)
- Completely Unsupervised Speech Recognition By A GAN Harmonized With Iteratively Refined HMMs - Chen et al. 2019 - arXiv:1904.04100 (Interspeech 2019)
- Unsupervised Speech Recognition with N-Skipgram and Positional Unigram Matching (PUSM/ESPUM) - Wang 2023 - arXiv:2310.02382 (ICASSP 2024)
- Towards Unsupervised Speech Recognition Without Pronunciation Models (JSTTI) - Wang & Hasegawa-Johnson 2024 - arXiv:2406.08380
- Towards Unsupervised Speech Recognition at the Syllable-Level (SylCipher) - Wang et al. 2025 - arXiv:2510.03639
- EURO: ESPnet Unsupervised ASR Open-source Toolkit - Gao et al. 2022 - arXiv:2211.17196 (ICASSP 2023)

**Joint speech+text / text-injection pretraining**
- MAESTRO: Matched Speech Text Representations through Modality Matching - Chen et al. 2022 - arXiv:2204.03409 (Interspeech 2022)
- Maestro-U: Leveraging joint speech-text representation learning for zero supervised speech ASR - Chen et al. 2022 - arXiv:2210.10027 (SLT 2022)
- JOIST: A Joint Speech and Text Streaming Model For ASR - Sainath et al. 2022 - arXiv:2210.07353 (SLT 2022)
- SLAM: A Unified Encoder for Speech and Language Modeling... - Bapna et al. 2021 - arXiv:2110.10329
- mSLAM: Massively Multilingual Joint Pre-Training for Speech and Text - Bapna et al. 2022 - arXiv:2202.01374
- SpeechT5: Unified-Modal Encoder-Decoder Pre-Training... - Ao et al. 2021 - arXiv:2110.07205 (ACL 2022)
- Injecting Text in Self-Supervised Speech Pretraining (tts4pretrain) - Chen et al. 2021 - arXiv:2108.12226 (ASRU 2021)
- Google USM: Scaling ASR Beyond 100 Languages (MOST) - Zhang et al. 2023 - arXiv:2303.01037

**Syllable/segment tokenizers & fixing the target**
- SyllableLM: Learning Coarse Semantic Units for Speech Language Models - Baade et al. 2024 - arXiv:2410.04029
- Sylber: Syllabic Embedding Representation of Speech from Raw Audio - Cho et al. 2024 - arXiv:2410.07168 (ICLR 2025)
- SD-HuBERT: Sentence-Level Self-Distillation Induces Syllabic Organization in HuBERT - Cho et al. 2023 - arXiv:2310.10803 (ICASSP 2024)
- Self-Supervised Syllable Discovery Based on Speaker-Disentangled HuBERT - Komatsu & Shinozaki 2024 - arXiv:2409.10103 (SLT 2024)
- Word Discovery in Visually Grounded, Self-Supervised Speech Models (VG-HuBERT) - Peng & Harwath 2022 - arXiv:2203.15081
- Syllable Discovery and Cross-Lingual Generalization in a Visually Grounded... Model - Peng et al. 2023 - arXiv:2305.11435 (Interspeech 2023)
- DP-Parse: Finding Word Boundaries from Raw Speech with an Instance Lexicon - Algayres et al. 2022 - TACL doi:10.1162/tacl_a_00505
- Word Segmentation on Discovered Phone Units with Dynamic Programming... (DPDP) - Kamper 2022 - arXiv:2202.11929 (IEEE/ACM TASLP)
- GradSeg: unsupervised word segmentation with pretrained deep features - Fuchs & Hoshen 2023 (ICASSP 2023)
- HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units - Hsu et al. 2021 - arXiv:2106.07447
- ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling - 2026 - arXiv:2602.15537 (author list unverified)
- Normalized Cuts and Image Segmentation - Shi & Malik 2000

**Distribution-matching / OT / LM-reward priors & embeddings**
- Neural Word Embedding as Implicit Matrix Factorization - Levy & Goldberg 2014 - NeurIPS 2014 (papers.nips.cc/paper/5477)
- Gromov-Wasserstein Alignment of Word Embedding Spaces - Alvarez-Melis & Jaakkola 2018 - EMNLP (arXiv:1809.00013 / ACL D18-1214)
- Gromov-Wasserstein Averaging of Kernel and Distance Matrices - Peyré et al. 2016 - ICML
- Word Translation Without Parallel Data (MUSE) - Conneau et al. 2017 - arXiv:1710.04087
- On the Limitations of Unsupervised Bilingual Dictionary Induction - Søgaard et al. 2018 - ACL P18-1072 (arXiv:1805.03620)
- Are All Good Word Vector Spaces Isomorphic? - Vulić et al. 2020 - arXiv:2004.04070
- Unsupervised Cross-Modal Alignment of Speech and Text Embedding Spaces - Chung et al. 2018 - arXiv:1805.07467
- Minimum Word Error Rate Training for Attention-based Seq2Seq Models - Prabhavalkar et al. 2018 - arXiv:1712.01818
- Optimizing Expected Word Error Rate via Sampling for Speech Recognition - Shannon 2017 - arXiv:1706.02776
- All-but-the-Top: Simple and Effective Postprocessing for Word Representations - Mu & Viswanath 2018 - arXiv:1702.01417 (ICLR 2018)
- Norm of Word Embedding Encodes Information Gain - Oyama, Yokoi & Shimodaira 2023 - arXiv:2212.09663 (EMNLP 2023)
- Controlled Experiments for Word Embeddings - Schakel & Wilson 2015 - arXiv:1510.02675
- Representation Degeneration Problem in Training NLG Models - Gao et al. 2019 - ICLR 2019 (OpenReview ByxY8CNtvr)
- An Isotropy Analysis in the Multilingual BERT Embedding Space - Rajaee & Pilehvar 2021 - arXiv:2110.04504
- Sound Analogies with Phoneme Embeddings - Silfverberg, Mao & Hulden 2018 - SCiL 2018 (ACL W18-0314)
- PWESuite: Phonetic Word Embeddings and Tasks They Facilitate - Zouhar et al. 2024 - arXiv:2304.02541 (LREC-COLING 2024)
- Word Embeddings as Statistical Estimators - PMC12711318 (author/venue partly unverified)

**Discrete tokens for speech LLMs**
- AudioLM: a Language Modeling Approach to Audio Generation - Borsos et al. 2022 - arXiv:2209.03143
- AudioPaLM: A Large Language Model That Can Speak and Listen - Rubenstein et al. 2023 - arXiv:2306.12925
- Textually Pretrained Speech Language Models (TWIST) - Hassid et al. 2023 - arXiv:2305.13009
- SpeechTokenizer: Unified Speech Tokenizer for Speech LLMs - Zhang et al. 2023 - arXiv:2308.16692
- Moshi: a speech-text foundation model for real-time dialogue (Mimi codec) - Défossez et al. 2024 - arXiv:2410.00037
- LAST: Language Model Aware Speech Tokenization - Turetzky & Adi 2024 - arXiv:2409.03701
- LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization - 2025 - arXiv:2506.16738
- PAST: Phonetic-Acoustic Speech Tokenizer - 2025 - arXiv:2505.14470
- TASTE: Text-Aligned Speech Tokenization and Embedding... - 2025 - arXiv:2504.07053
- TASLA: Text-Aligned Speech Tokens with Multiple Layer-Aggregation - 2025 - arXiv:2510.14934
- Scaling Spoken Language Models with Syllabic Speech Tokenization (Sylber-SLM) - 2025 - arXiv:2509.26634
- Beyond Fixed Frames: Dynamic Character-Aligned Speech Tokenization (DyCAst) - 2026 - arXiv:2601.23174 (numbers unverified)
- DinoSR - Liu et al. 2023 - arXiv:2305.10005
- DM-Codec - 2024 - arXiv:2410.15017
- Phoneme-Level BERT (PL-BERT) - 2023 - PMC10417533; PnG BERT - Jia et al. 2021 - arXiv:2103.15060 (arxiv id unverified)
