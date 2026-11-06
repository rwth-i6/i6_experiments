# Tree-search blank-penalty approach

For the implemented optimizer's line-by-line explanation and design tradeoffs,
see [`TREE_NBEST_BLANK_PENALTY_WALKTHROUGH.md`](TREE_NBEST_BLANK_PENALTY_WALKTHROUGH.md).

## Current plan

1. Collect N-best hypotheses using the matching joint L2 ADC 2prec/6range
   configuration, initially at additive post-softmax penalty 1.44. Optionally
   collect at a few nearby anchors to broaden the candidate pool. Each hypothesis
   uses its own prediction history; references are used to count word errors.
2. Use the existing exported terminal traceback cost. Keep collection penalty,
   scorer scale, hardware settings, and run identity with each observation.
3. Infer each complete RNA candidate's nonblank model-token count from the
   lexicon. Then `blank_count = encoder_frames - nonblank_token_count`.
4. For the currently supported unit scorer scale, calculate
   `cost(b) = exported_cost + (b - collection_penalty) * blank_count`.
5. Find each utterance's lower-cost candidate switches, then sweep their union
   to find intervals minimizing total corpus word errors. Use a reference-free
   rule for score ties. Keep utterances whose reference is absent from the pool.
6. Report candidate-oracle WER and coverage. Test a selected interior penalty
   with fresh tree decoding against the original anchor. Expand the candidate
   pool if pruning or missing hypotheses materially changes the result.

No digital soft alignment, forced alignment, or monotonic Viterbi rescoring is
required for this version. Fresh decoding retains active readout noise. The
analytical calculation holds collected scores fixed; it does not reproduce new
noise or penalty-dependent pruning. Pooling observations from separate noisy
runs can favor lucky reads and should be reported as a candidate-pool heuristic,
not an estimate of a candidate's mean score.

## What is available through the current interfaces?

Checked the installed module in the recognition container on 2026-10-22:

```text
/work/smt4/azim.javed/sisyphus_work/i6_core/tools/compile/
CMakeJob.qMRpN7b4MOKN/output/install/librasr.cpython-310-x86_64-linux-gnu.so
```

Runtime inspection exposes these search methods:

```text
enter_segment, finish_segment, put_feature, put_features,
get_current_best_traceback, get_common_prefix,
get_current_n_best_list, recognize_segment, recognize_segment_n_best
```

There is no additional Python lattice, token-path, beam-export, or aligner
interface in this installed module. This is a statement about this build, not
all RASR versions.

| Interface | What it provides | Limitation for this task |
|---|---|---|
| `recognize_segment` / `get_current_best_traceback` | Original best traceback, with timing when preserved by the configured decoder | Only one winning path; a word span can hide internal emissions |
| `get_common_prefix` | Stable shared prefix | Not a complete set of alternatives |
| `recognize_segment_n_best` / `get_current_n_best_list` | N-best tracebacks reconstructed from a word lattice | Saved tree results have zero timestamps and word-level entries |
| C++ `getCurrentBestWordLattice` | Word lattice containing alternatives | Not bound to Python here; word arcs do not expose every internal model transition |
| C++ `getCurrentBestLatticeTrace` | Best lattice trace structure | Not bound to Python here; still based on word-end/blank traces in the tree decoder |

The two Python N-best methods use the same extraction code; changing between
them does not recover missing timing. Likewise, streaming calls to the best
traceback do not expose all final candidate token paths.

## Evidence from the saved tree probe

The earlier `tree_nbest_probe/bp0/nbest.jsonl` contains three utterances and 60
candidates. All 1,534 N-best traceback items have zero timestamps. Explicit
`[BLANK]` entries exist, but neither their count nor their zero durations is a
reliable total blank count.

The same searches' direct-best results have nonzero timestamps. For `MART`:

```text
Encoder frames:                  29
Lexicon nonblank model tokens:     3
Total blanks:                    26
Explicit blank-span durations:   19
Word span:                       frames 14 through 24
```

Seven blanks are inside the word span. This is why even the presence of timed
blank entries is not sufficient for direct counting in the word-level tree
traceback.

In contrast, a token-level traceback such as `P@@`, `AL@@`, `E@@`, `T@@`, `'S`
with complete blank spans can support direct duration counting. The field name
`lemma` alone does not identify whether entries are words or model tokens.

These earlier results establish interface behavior only: they used the old ADC
setting and must not be reused as current-hardware scoring observations. The
new ADC-specific `bp1.44/nbest.jsonl` was empty at inspection.

## Cost export limitation

The earlier top-N-best/direct cost differences were zero for all three
utterances. We proceed using the existing terminal costs as an approximation.

Source inspection shows a more authoritative total is stored on each N-best
root selection arc, including epsilon/final contributions. The current Python
export instead accumulates labeled-arc costs. Those totals can agree, but
agreement is not established for every alternative. Candidate-dependent missing
contributions can shift the estimated crossing points.

If better cost export becomes necessary, exposing the existing root-arc total
is a smaller change than recording full model-token paths. Merely exposing a
word lattice would not recover internal token timings already omitted by tree
trace construction.

## Source references

Inspected RASR checkout:

```text
/work/smt4/azim.javed/sisyphus_work/i6_core/tools/git/
CloneGitRepositoryJob.MBfQffJ6OgEy/output/rasr
```

- `src/Tools/LibRASR/Search.cc:84-138`: Python search bindings.
- `src/Python/Search.cc:93-113,194-283`: original versus reconstructed tracebacks.
- `src/Search/SearchV2.hh:80-90`: internal lattice and traceback interfaces.
- `src/Search/TreeTimesyncBeamSearch/TreeTimesyncBeamSearch.cc:63-110`: within-word
  transitions retain the previous trace; word-end transitions create trace items.
- The same file, lines 491-512: best traceback and word-lattice accessors.
- `src/Flf/NBest.cc:906,1041,1292`: full totals on root selection arcs and their
  use by the native N-best writer.

## Implementation status

The collection probe and the cost-line/corpus-error optimizer now exist. The
optimizer is [`tree_nbest_blank_penalty.py`](tree_nbest_blank_penalty.py), with
Sisyphus wiring in `config/tree_nbest_debug.py`. It supports compatible
multi-anchor pools and configured LM/ILM contributions. The statistical
rescoring validator is separate and optional; fresh decoding is still needed
to validate a selected penalty.
