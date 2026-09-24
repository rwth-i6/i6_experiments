# Literature: GPU lexical-prefix-tree search with a word n-gram LM -- marginalisation vs beam

Date 2026-09-21. Question from `SAE_4A_lexlat.md` State: at C = 1024 lexical contexts, m_max ~ 3,800
candidates/frame, O = 51 band offsets, float64, B ~ 114-120, ~770 padded frames, one GH200 96 GiB,
the measured lexlat step is 802-912 s against a 21 s/step budget. Does the published work show a
GPU-efficient design for (a) pruned MARGINALISATION with gradients over trie x word-LM, and
(b) max-plus BEAM search over the same space?

Every number below was read from the full text (or, for k2, from the source) at the URL given.
Items marked NOT READ were not reached before the hand-back and carry no weight here.

---

## 1. What each source actually does, at what scale

### Braun, Luitjens, Leary, Kaldewey, Povey -- "GPU-Accelerated Viterbi Exact Lattice Decoder for
Batched Online and Offline Speech Recognition", ICASSP 2020. https://arxiv.org/pdf/1910.10032
Read in full.
- Batched over utterances by "lanes" (active batch) and "channels" (parked streams); context switch
  ~5 us per batch. One thread per outgoing ARC in a load-balanced expand.
- Operating point: beam = 15, lattice-beam = 8, **max-active = 10,000 tokens per utterance per
  frame**; adaptive beam set FROM the candidate scores after expansion (histogram pruning).
- Graph: LibriSpeech HCLG, lexicon x word trigram, 192.6 MB (pruned 3e-10) up to 8,724 MB (unpruned
  trigram). FST memory model `M_fst = 12|Q| + 8|E| + 4|E_E|`, about 1/3 of the on-disk FST size.
- Decoder memory is closed-form and **independent of graph size and beam**:
  `M_state = 64*alpha*n_c + 544*alpha*n_l + 1024*n_l`; alpha = 10,000, n_c = 5,000, n_l = 500 ->
  5.5 GB; a single stream -> 5.8 MB.
- Throughput (decoder only, offline, one V100): **xRTF 9,031 on LibriSpeech test-clean, 4,392 on
  test-other**; ASPiRE 769 / 650. Multi-GPU scaling 85% on 8x V100.
- Accuracy is equivalent to Kaldi CPU (WER +-0.02%, lattice density 4.22 vs 4.19).
- Notably: the LARGER (unpruned, 8.7 GB) trigram decodes FASTER (9,162 xRTF) than the pruned one,
  "likely due to reduced perplexity during decoding yielding extra pruning".
- "During the discovery stage, we had to create and consider (typically an order of magnitude) more
  tokens than the ones we ultimately keep."
- **No gradients.** Output is a lattice, Viterbi/max-plus throughout.
- Per-utterance-frame cost (derived; the Kaldi LibriSpeech chain model runs at a 3x-subsampled
  33.3 frames/s): 9,031 xRTF x 33.3 fps = ~3.0e5 utterance-frames/s -> **~3.3 us per utterance-frame**
  (test-clean), ~6.8 us (test-other).

### Chen, Luitjens, Xu, Wang, Povey, Khudanpur -- "A GPU-based WFST Decoder with Exact Lattice
Generation", Interspeech 2018. https://arxiv.org/pdf/1804.03243 Read in full.
- Token recombination as a **64-bit atomicMin per ARC** (cost in the high bits, arc index in the
  low bits) -- no precision loss, no critical section, one array slot per arc so no write conflict.
- Dynamic load balancing: a 32-thread cooperative group per token, new tokens dispatched by
  atomicAdd. Static load balancing (prefix sum over arc counts) is the alternative; dynamic wins.
- Switchboard, 30k-vocab trigram HCLG (13 / 62 / 196 / 258 MB), beam = 14, V100.
- **RTF 0.0035 (1-best) and 0.0080 (with exact lattice) with 8-sequence MPS**; single-sequence
  0.011 / 0.028. CPU baseline 0.16 / 0.27.
- "Our current implementation decodes at most 11 GB WFST in a 12 GB TITAN GPU."
- 10 ms (unsubsampled) frame rate is **3x slower** than the subsampled model: "frame rate reduction
  is crucial even in GPU decoders" -- the time axis is serial.
- Derived: RTF 0.0035 at 33.3 fps -> ~1e5 utterance-frames/s -> **~100 us per utterance-frame**,
  i.e. Braun 2020 is ~30x faster than this.

### Galvez, Kaldewey -- "GPU-Accelerated WFST Beam Search Decoder for CTC-based Speech Recognition",
arXiv 2311.04996 (NVIDIA Riva). https://arxiv.org/pdf/2311.04996 Read in full.
- Same decoder family as Braun 2020, retargeted to CTC via a T o L o G graph. Batch size 200,
  A100-80GB, LibriSpeech `3-gram.pruned.3e-7`, **beam width 17.0, max-active 10,000**.
- Two changes worth naming: **CUDA graphs** remove kernel-launch overhead that had grown to "up to
  20% of each iteration"; **double-buffered D2H** removes a further ~25% stall.
- End-to-end pipeline RTFx (feature extraction + acoustic model + beam search), Conformer CTC Large:
  **2,261 (test-clean) / 2,205 (test-other)** vs Flashlight on 16 CPU cores 488 / 465. Small model:
  4,320 vs 646. Flashlight beam size 2,500 (its units are non-blank hypotheses, not WFST states).
- Flashlight scales 4x on 16 cores, not 16x. Before GPU decoding, beam search was ~90% of pipeline
  runtime with the AM already on GPU.
- Credits the WER edge to **LM weights pushed left in the WFST**, "able to inform search before a
  word is fully seen, an optimization not possible in dynamic decoders".
- No gradients.

### Bataev et al. -- "NGPU-LM: GPU-Accelerated N-Gram Language Model for Context-Biasing in Greedy
ASR Decoding", Interspeech 2025. https://arxiv.org/pdf/2505.22857 Read in full.
- The n-gram LM is a suffix-tree stored as flat CSR tensors (arc tokens / weights / to-states,
  per-state arc ranges, per-state backoff weight and backoff target). A batched query is
  `state[B] -> (token_weights[B, V], next_states[B, V])`: fill the arcs present at the state, then
  iterate the backoff chain; the loop length is bounded by the LM order. Final weights are
  precomputed at load time by traversing backoff transitions.
- 10-gram token-level (1,024 BPE) LMs, **under 100 MB on GPU** (1.5 GB for the unpruned SPGI one).
- CUDA-graph compatible by construction (fixed shapes, no data-dependent control flow).
- **Under 7% RTFx overhead over greedy** at batch 32 on an A6000: RNN-T 996 -> 973, CTC 3,311 ->
  3,108. For contrast, CPU-KenLM beam = 4 costs 11x (RNN-T 996 -> 79).
- No lexicon and no trie over words; this is a token-level LM query structure.

### Grigoryan, Bataev et al. -- "Pushing the Limits of Beam Search Decoding for Transducer-based ASR
Models", Interspeech 2025. https://arxiv.org/pdf/2506.00185 Skimmed (structure + table captions),
not line-by-line.
- `BatchedBeamHyps`: audio- AND beam-level batching, a **trie-like transcript store built from two
  3D tensors** (`transcripts`, `transcripts_ptrs`) so hypotheses share prefixes, plus incremental
  hashes for O(1) recombination. CUDA graphs throughout. Batch 32, beam 6.
- ALSD++ with CUDA graphs comes within ~15% of greedy RTFx on SPGI; decoder-only beam RTFx is
  1.7-1.9x the full-model figure at batch 32.

### "FlexCTC: GPU-powered CTC Beam Decoding with advanced Contextual Abilities", arXiv 2508.07315.
https://arxiv.org/html/2508.07315v1 Read in full (method, setup, all four tables, both figures).
- Fully vectorised CTC beam search: a single loop over time; the batch and the beam are both tensor
  axes. Per step it forms `logp [B, K, |V|]`, adds the NGPU-LM score, does a **flat TopK over
  K x |V|**, applies a relative threshold theta = 12, then recombines by hash. CUDA graphs.
- A5000, **float32**, batch 32, FastConformer CTC Large (1,024 BPE), 6-gram subword LM.
- Full-pipeline RTFx on SPGI: greedy 2,797; FlexCTC beam 4 **2,085**; beam 16 **1,928**; Flashlight
  (CPU) beam 4 1,071, beam 16 454; PyCTCDecode beam 16 909; **CUDA WFST (word 4-gram, max = 10k)
  832** at WER 4.40 vs FlexCTC 4.38.
- Scaling: beam 128 at batch 32 costs ~2.5x beam 4 (RTFx ~2,200 -> ~880) and buys 0.65 WER absolute.
  At batch 128 the beam-4 penalty over greedy is only **10.3%** (21.4% at beam 16).
- No gradients. No lexicon trie (subword LM only); the paper notes word-level LMs (PyCTCDecode,
  CUDA WFST) do badly in low-resource domains.

### Ondel, Lam-Yee-Mui, Kocour, Corro, Burget -- "GPU-Accelerated Forward-Backward Algorithm with
Application to Lattice-Free MMI", ICASSP 2022 / arXiv 2112.00709. https://arxiv.org/pdf/2112.00709
Read in full. **This is the closest published match to our computation.**
- Forward-backward written as sparse matrix products in the **log-semifield**:
  `alpha_n = v_n o (T^T alpha_{n-1})`, `beta_n = T (beta_{n+1} o v_n)`. Batching is a
  **block-diagonal T** over the batch, with a phony self-loop end state for padding.
- Benchmark shape matches ours almost exactly: **batch 128, 700 frames per sequence**.
  Denominator graph = 3-gram phonotactic LM over 42 phones with a 2-state HMM topology:
  **3,022 states, 50,984 arcs**. Numerator (largest WSJ alignment graph): 454 states, 1,036 arcs.
- RTX 2080 Ti timings for the forward-backward AND its gradient (the LF-MMI posteriors):
  | impl | device | leaky-HMM | numerator | denominator |
  |---|---|---|---|---|
  | PyChain | CPU | no | 5.696 s | 421.4 s |
  | PyChain | CPU | yes | 1.212 s | 27.6 s |
  | PyChain | GPU | no | 0.093 s | 5.862 s |
  | PyChain | GPU | yes | 0.248 s | 5.449 s |
  | proposed | GPU | no | **0.058 s** | **1.04 s** |
- On GPU the leaky-HMM approximation buys nothing (it is slower on the numerator).
- Per-step training overhead on WSJ: LF-MMI loss+grad 0.220 s vs 4.08 s for PyChain, against 0.180 s
  for the whole NN forward+backward.
- Derived rates: 1.04 s / (128 x 700) = **11.6 us per utterance-frame**; 50,984 arcs x 700 x 128 /
  1.04 s = **~4.4e9 arc-updates/s** in the log-semiring on a 2080 Ti.
- Exact, unpruned, no beam. The paper notes the tropical semiring would drop in trivially.

### Wu, Variani, Bagby, Riley -- "LAST: Scalable Lattice-Based Speech Modelling in JAX",
arXiv 2304.13134. https://arxiv.org/pdf/2304.13134 Read in full.
- Shape is structurally ours: GNAT with a FULLNGRAM context dependency, **context size 2 over
  V = 32, C = 1,057 context states**, T = 1,024 frames, batch 16. Complete lattice is O(T C) states
  and O(T C V) arcs: **~1e6 states and ~33e6 arcs PER INPUT**.
- Two techniques, both stated as necessities:
  1. **On-the-fly arc weights.** "Storing all O(TCV) arc weights quickly become impractical";
     LAST computes them one frame at a time inside the topological sweep. Explicitly contrasts
     k2 and GTN, which "store all the arc weights of a WFSA in memory at once".
  2. **Hand-written forward-backward instead of autodiff.** Under the LOG semiring, naive AD needs
     O(|E| + |Q|) space because it cannot cancel `exp alpha[p]`; FB needs O(|Q|). Under the TROPICAL
     semiring the VJP of max is just the argmax index, so O(|Q|) either way.
- Benchmark (training = gradient to decoder params and encoder frames):
  | | TPUv3 | V100 |
  |---|---|---|
  | forward-backward | 0.92 s / 217 MB | **2.42 s / 278 MB** |
  | autodiff + remat | 1.12 s / 219 MB | 2.40 s / 278 MB |
  | autodiff, no remat | OOM / >=102 GB | **OOM / >=68.5 GB** |
  | inference (shortest path) | 0.41 s / 171 MB | 0.30 s / 203 MB |
  | **k2**, training | | **1.69 s / 11.9 GB** |
  | **k2**, inference | | 0.70 s / 9.6 GB |
- "Even with optimizations such as DENSEFSAVEC and gradient rematerialization, our k2 implementation
  is not able to process batch size 16 input without running out of memory" -- split into 5+5+6.
- k2 is faster than LAST only because it computes arc weights 16 frames at a time; at 1 or 2 frames
  k2's training time rises to 4.76 s / 2.59 s.
- Derived: 2.42 s / (16 x 1024) = **148 us per utterance-frame** at ~32k arcs per frame per utterance
  (the arc weight is an MLP output, so this is not comparable arc-for-arc with Ondel's flat adds).

### k2 `intersect_dense_pruned` -- read from source, not from a paper (no paper reports its cost).
`https://raw.githubusercontent.com/k2-fsa/k2/master/k2/python/k2/fsa_algo.py` (lines 380-440, 667-760)
and `https://raw.githubusercontent.com/k2-fsa/k2/master/k2/csrc/intersect_dense_pruned.cu`.
- `a_fsas` are "the decoding graphs, one per sequence. E.g. might just be a linear sequence of phones,
  or might be something more complicated. Must have either the same Dim0() as b_fsas, or
  **Dim0()==1 in which case the graph is shared**." So a single HLG (lexicon o word n-gram) is a
  legal `a_fsas`, shared across the batch -- exactly our trie x word-LM space.
- It is wrapped in `_IntersectDensePrunedFunction.apply(...)` with `unused_scores_a` /
  `unused_scores_b` passed "solely for back propagation": **gradients flow to both the NN output and
  the graph weights.** The marginalisation itself is then done downstream on the pruned lattice.
- The SEARCH is max-plus: `AtomicMax(&(kept_states_data[...].forward_loglike), ...)` -- a Viterbi
  beam. `output_beam` then prunes the output lattice "similar to lattice-beam in Kaldi".
- **Dynamic beam, per utterance, per frame** (lines ~706-762): start at `search_beam`; if active
  states <= max_active and >= min_active, relax toward the default (`0.8*b + 0.2*default`); if below
  min_active, `b *= 1.25`; if above max_active, `b *= 0.8`. `max_active` "determines the hash size".
  Work is sized as `max_active_ * num_fsas * T_`.
- icefall `icefall/mmi.py::_compute_mmi_loss_pruned`: **search_beam = 20, output_beam = 8,
  min_active_states = 30, max_active_states = 10,000**, with the in-code note that these "are not
  tuned", and that the pruned path "uses the least amount of memory, but the loss is not exact due to
  pruning". `intersect_dense` (exact) defaults: `max_states = 15,000,000`, `max_arcs = 1,073,741,824`.
- **No published per-frame timing for `intersect_dense_pruned` was found.** The only k2 timing in the
  read literature is LAST's, and that is the UNPRUNED `intersect_dense` path.

### Hannun, Pratap, Kahn, Hsu -- "Differentiable Weighted Finite-State Transducers" (GTN),
arXiv 2010.01003. https://arxiv.org/pdf/2010.01003 Skimmed for the cost table only.
- GTN parallelises over the batch on CPU; there is no GPU backend for the autograd operations.
- IAM epoch times with a bigram transition graph: letters 544 s unpruned -> 249 / 202 s at pruning
  thresholds 0 / 10; **1,000 word pieces 17,939 s unpruned -> 683 / 204 s**. That 88x from pruning
  alone is the only GTN-side datum relevant here.

### Hannun et al. -- "Parallel Composition of Weighted Finite-State Transducers",
arXiv 2110.02848. https://arxiv.org/pdf/2110.02848 Read in full.
- SIMD-style GPU composition, one thread per ARC PAIR, epsilon supported; atomics only for the
  arc-count prefix sums. V100 32 GB vs single-threaded Xeon E5-2698.
- The speech benchmark IS a lexicon: LibriSpeech lexicon, 200,000 word->phoneme entries, 69 phonemes,
  composed (as a closure, so with epsilons) with a **linear emissions graph of 251 nodes x 69 arcs**
  (10 s of audio at 40 ms frames). At **32,000 words the GPU is >10x the CPU**; at 1,000 words they
  are comparable. Random graphs: 30x at 8,192 nodes.
- Both implementations "ultimately scale **quadratically** in the number of nodes".
- This is EAGER, UNPRUNED, full composition -- the graph build only, not the forward-backward, and
  not batched over utterances.

### Ondel Yang, Raissi, Kocour, Riera, Corro -- "Fast and General Automatic Differentiation for
Finite-State Methods", arXiv 2602.12300. https://arxiv.org/pdf/2602.12300 Skimmed.
- Semiring-agnostic VJPs ("morphism trick"); handles "graphs with millions of states and hundreds of
  millions of transitions"; orders of magnitude faster than generic AD (Zygote).
- **"Computation on GPU is currently not supported but planned for future releases."** Negative
  result for us: the most general differentiable-FST tooling as of 2026 is CPU-only.

### NOT READ (no weight given to these below)
- Ortmanns, Ney, Eiden, Coenen, "Look-ahead techniques for fast beam search" (1997/2000). The RWTH
  PDF at `www-i6.informatik.rwth-aachen.de/publications/download/210/` fetched as undecodable
  compressed streams. From search-result summary only, therefore **UNVERIFIED**: LM look-ahead +
  phoneme look-ahead on 20k-word NAB'94 with a bigram reduce the search space by **~30x** at equal
  word accuracy. I could not verify the state counts, beam thresholds, RTF or look-ahead table
  memory. Treat the 30x as hearsay until the paper is read.
- Klejch, Wallington, Bell (Interspeech 2022), Baum-Welch on composed transducers. NOT READ in this
  task. (`SAE_4A_lexlat.md` already cites its curriculum finding from an earlier round; I add nothing.)
- Kim et al. / Hannun et al. GPU CTC prefix beam search. NOT READ.
- Flashlight/wav2letter lexicon beam decoder as a paper. NOT READ; only its operating points as
  reported by others (beam size 2,500 in Galvez 2311.04996; beam 4/16 in FlexCTC; 4x scaling on
  16 cores in Galvez). The trie-smearing detail (partial-word LM score approximated by the recursive
  max/logadd unigram smear up the trie) comes from the wav2letter wiki, not a paper -- UNVERIFIED.
- Zhang & Ney / word-conditioned tree search as primary sources. NOT READ.

---

## 2. Answers

### (1) Is a GPU pruned forward-backward WITH gradients over trie x word-LM plausible at our scale?

**In kind yes, at our arc count no -- and the literature reports no per-frame cost for it.**

The tooling exists and is autograd-complete. `k2.intersect_dense_pruned` accepts a single shared
decoding graph (`Dim0()==1`), which is how icefall passes an HLG, and it is an autograd Function
whose gradient reaches the acoustic scores; the marginalisation is then a log-semiring sum over the
pruned lattice it returns. But note WHAT is published: the pruned intersection searches with
**AtomicMax** under a **dynamic beam that targets 10,000 active states per utterance per frame**
(icefall `_compute_mmi_loss_pruned`: search_beam 20, output_beam 8, min 30, max 10,000), and the
resulting loss is documented as "not exact due to pruning". The published recipe for pruned
marginalisation with gradients is therefore *Viterbi-beam-pruned lattice, then log-sum*, not a
sum-pruned context lattice of fixed width.

Per-frame costs the literature does supply, for exact FB with gradients:
- **11.6 us per utterance-frame** at 3,022 states / 50,984 arcs (batch 128, 700 frames, RTX 2080 Ti),
  i.e. ~4.4e9 log-semiring arc-updates/s (arXiv 2112.00709, Table 1).
- **148 us per utterance-frame** at ~1e6 states / 33e6 arcs per utterance (batch 16, T = 1024, V100,
  278 MB with on-the-fly arc weights) (arXiv 2304.13134, Table 3).
Neither is beam-limited, so neither gives a cost "at beam 32-1024 hypotheses". **No source read here
reports a per-frame cost for pruned FB with gradients at any beam size.** That question is open.

The extrapolation that does survive: our per-utterance-frame arc budget is
C x m_max x O ~ 1024 x 3,800 x 51 ~ **2.0e8**, against 50,984 for the Ondel denominator and ~32,000
per frame for the LAST lattice -- **~4,000x and ~6,000x more arcs per frame respectively**. At
Ondel's measured 4.4e9 arc-updates/s scaled ~4-5x for a GH200 in fp32, 84,000 utterance-frames x
2.0e8 arcs ~ 1.7e13 arc-updates lands at several hundred seconds per step. **Our 850 s is roughly
what the best published GPU log-semiring arc rate predicts for our arc count.** The 40x gap is a
state-space problem, not a kernel-efficiency problem -- which confirms the orchestrator's prior
estimate recorded in `SAE_4A_lexlat.md` State, and means a re-chunking or a better kernel cannot
close it. Reaching 21 s requires the per-utterance-frame arc count down by ~40x, to ~5e6.

For scale calibration against the published pruning widths: our C x O = 52,224 surviving states per
frame is already 5x k2's and Braun's max_active of 10,000, and our pre-pruning candidate count
C x m_max ~ 3.9e6 is ~400x it. Nothing in the read literature runs a differentiable lattice of that
width per frame.

### (2) What would beam (max-plus) search cost instead?

**One to two orders of magnitude INSIDE our per-step budget, with no gradients.**

Our budget is 21 s for 120 x ~700 = 84,000 utterance-frames = **250 us per utterance-frame**.
Measured GPU beam search over a genuine lexicon x word-trigram HCLG:
- Braun 2020 (V100, max-active 10,000, beam 15, LibriSpeech HCLG, exact lattice output):
  **~3.3 us per utterance-frame** (test-clean, 9,031 xRTF) to **~6.8 us** (test-other) --
  **~40-75x inside our budget on 2017 hardware**, with bounded memory (5.8 MB for a single stream;
  5.5 GB for 5,000 parked channels).
- Chen 2018 (V100, beam 14, Switchboard 30k trigram, 8-way MPS): ~100 us per utterance-frame for
  1-best, ~230 us with the exact lattice -- i.e. the weaker of the two GPU decoders is still at or
  just inside our budget.
- Galvez 2023 (A100, batch 200, max-active 10,000, TLG): end-to-end RTFx 2,261 for a 120M-parameter
  Conformer CTC including feature extraction and the acoustic model; the decoder is a small share.
- FlexCTC 2025 gives the beam-size scaling for a lexicon-free but LM-fused GPU beam: beam 128 costs
  2.5x beam 4 at batch 32 (A5000, fp32), and at batch 128 beam 4 costs only 10.3% over greedy.
  So beams in the 32-1024 range are affordable when the beam is a tensor axis rather than a loop.

Caveats that decide whether this transfers: (i) all four return a lattice or a 1-best, **not
posteriors** -- there is no gradient, which is why this only supports a hard-EM / self-training use;
(ii) the tropical VJP is cheap in principle (LAST 4.2.3: storing argmax indices is O(|Q|)), but no
read source reports a batched GPU max-plus pass over a lexicon graph *with* a backward; (iii) their
frame rates are 25-33 Hz against our stride-3 frames, and Chen 2018 measured that running at the
unsubsampled 10 ms rate costs **3x** because the time axis is serial; (iv) their graphs are static
HCLGs compiled offline, 192 MB to 11 GB on device.

### (3) Which techniques do the published systems rely on that our design lacks?

1. **Arc-level (candidate-level) pruning after expansion, not context-level pruning before it.**
   Braun expands one thread per outgoing arc, *then* sets the adaptive beam from the candidate scores
   and admits survivors; Chen recombines per arc with a 64-bit atomicMin. Our design ranks C = 1024
   CONTEXTS by forward mass and then expands each one's full trie candidate set, so the arc count is
   C x m_max (3.9e6/frame) and is never bounded. This is the single largest structural difference.
2. **A dynamic beam that targets a fixed active-state count.** k2 multiplies its per-utterance beam
   by 1.25 or 0.8 each frame to hold active states between 30 and 10,000; Braun sets the beam from
   max-active per batch. Our C is a fixed constant chosen offline (E0 re-declared it 4096), so easy
   frames pay the same as hard ones and the width cannot adapt to the acoustic confidence -- which
   is precisely the effect Braun exploits when the *unpruned* trigram decodes *faster*.
3. **LM look-ahead / weight pushing into the trie.** Galvez attributes the CUDA WFST decoder's WER
   advantage to LM weights "pushed left in the WFST graph... able to inform search before a word is
   fully seen, an optimization not possible in dynamic decoders". Our trie nodes carry no look-ahead
   estimate of the best reachable word-LM cost, so mid-word prefixes are ranked on an incomplete
   score and the escape reservation (C_esc = 64) is doing by fiat what look-ahead does by score.
   (Ortmanns & Ney's ~30x search-space reduction from LM + phoneme look-ahead is the classic citation
   here but is UNVERIFIED -- see NOT READ.)
4. **float32 or narrower.** FlexCTC states float32 explicitly; k2 and the Braun/Chen line are fp32;
   Chen packs cost+arc index into a uint64 for the atomic. We run float64, which doubles the bytes
   moved on what is a gather/reduce-bound kernel. Ondel's 4.4e9 arc-updates/s is an fp32 figure.
5. **A compact CSR LM answering batched full-vocabulary queries.** NGPU-LM's
   `state[B] -> (weights[B,V], next_states[B,V])` with a bounded backoff loop costs <7% over greedy
   and fits in <100 MB. We hold the CSR trigram but query it per (context, candidate) inside the DP.
6. **On-the-fly arc weights and an O(|Q|) hand-written backward.** LAST computes arc weights one
   frame at a time and refuses to materialise O(|E|); autodiff without rematerialisation OOMs at
   >=68.5 GB on its lattice, and k2 -- which does materialise arc weights -- needs 11.9 GB where LAST
   needs 278 MB. Our backward already avoids autograd, but it holds five per-frame
   [B, K, m_max, O] fp64 tensors (64 GiB at B = 114, m_max = 7,416, per
   `reports/debug_lexlat_e1_oom_2026-09-21.md`), which is the O(|E|) materialisation LAST identifies
   as the thing to avoid.
7. **Topological/temporal batching of arc-weight computation.** k2 beats LAST purely by computing
   16 frames of arc weights at once instead of 1 (1.69 s vs 2.42 s; at 1 frame k2 would take 4.76 s).
8. **Kernel-launch amortisation (CUDA graphs) and double-buffered D2H.** Worth 20% and 25% of a
   decoder iteration respectively (Galvez). Irrelevant at an 850 s step, but it is why every 2023+
   GPU decoder is CUDA-graph-captured, and it constrains any design to fixed shapes.
9. **Batching over utterances with bounded, graph-size-independent state** (Braun's closed-form
   `M_state`). Our memory scales with C x m_max x O x B and has already OOMed at C = 4096.

### What would settle the open part

A timed `k2.intersect_dense_pruned` with an actual HLG (our lexicon composed with the word trigram)
as the shared `a_fsas`, at `max_active_states` swept over 1k-10k and our batch and frame count,
followed by `k2.get_tot_scores(..., log_semiring=True)` for the gradient. That is the exact published
mechanism for pruned marginalisation with gradients over trie x word-LM, and **its per-frame cost has
never been published**. Until it is measured, "pruned FB with gradients at beam 32-1024 over a
lexicon graph costs X us/frame" is not answerable from the literature.

---

## Sources
- https://arxiv.org/pdf/1910.10032 (Braun et al., ICASSP 2020)
- https://arxiv.org/pdf/1804.03243 (Chen et al., Interspeech 2018)
- https://arxiv.org/pdf/2311.04996 (Galvez & Kaldewey, NVIDIA Riva)
- https://arxiv.org/pdf/2505.22857 (Bataev et al., NGPU-LM, Interspeech 2025)
- https://arxiv.org/pdf/2506.00185 (Grigoryan & Bataev, Interspeech 2025)
- https://arxiv.org/html/2508.07315v1 (FlexCTC, 2025)
- https://arxiv.org/pdf/2112.00709 (Ondel et al., ICASSP 2022)
- https://arxiv.org/pdf/2304.13134 (Wu, Variani, Bagby, Riley, LAST)
- https://arxiv.org/pdf/2010.01003 (Hannun et al., GTN)
- https://arxiv.org/pdf/2110.02848 (Hannun et al., parallel composition)
- https://arxiv.org/pdf/2602.12300 (Ondel Yang et al., 2026)
- https://raw.githubusercontent.com/k2-fsa/k2/master/k2/python/k2/fsa_algo.py
- https://raw.githubusercontent.com/k2-fsa/k2/master/k2/csrc/intersect_dense_pruned.cu
- https://raw.githubusercontent.com/k2-fsa/icefall/master/icefall/mmi.py
