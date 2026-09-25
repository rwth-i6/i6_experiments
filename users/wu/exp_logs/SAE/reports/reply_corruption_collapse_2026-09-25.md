# Did JUPITER's L2-0 corruption job collapse adjacent repeats? No.

Reply to `SAE_i6_P1.md` (i6 commit 60fd09b8d), Task B Design, item "Open, CANNOT_TELL from i6".

## Answer

`CorruptSeedGoldJob` (speech-llm `src/speech_llm/sae/emc/blankfree_ladder_jobs.py`, last changed in
speech-llm commit 6fd3d02e) substitutes exactly `round(rho * n_u)` tokens per utterance and keeps the
length. It never collapses runs of equal adjacent tokens. The collapse count is hard-coded to 0, and
the job asserts that every corrupted string has the gold string's length (`len(corrupted[t]) == len(gold[t])`).

The reason is given in the module doc: the seed gold strings (`SeedGoldPhonesJob.zii9E9tvr51e`)
already contain 1705 adjacent repeats in 1251 of 2849 utterances. The gold targets
(`PhoneTargetHdfJob.DXDTg3VoP47A`) equal those strings token for token, and the gold-phi fit's
loader and model accept repeats. So the convention of the gold is "no collapse", and the corrupted
strings follow it.

## Banked outputs (`corruption.txt`, seed 0, all 2849 utterances, N = 351312 gold tokens)

| job | rho | realised rate | PER vs gold | collapses | adjacent repeats gold / corrupted |
|---|---|---|---|---|---|
| CorruptSeedGoldJob.xsTnpYUCPdlV | 0.3 | 0.300007 | 0.299845 | 0 | 1705 / 8799 |
| CorruptSeedGoldJob.c4FmMx93HuK5 | 0.5 | 0.500031 | 0.499061 | 0 | 1705 / 12157 |
| CorruptSeedGoldJob.mnzDC7XJX00r | 0.7 | 0.699905 | 0.695288 | 0 | 1705 / 14293 |
| CorruptSeedGoldJob.LCFfke9OsUqO | 1.0 | 1.000000 | 0.902796 | 0 | 1705 / 14857 |

PER is lower than the realised rate because the Levenshtein alignment can be cheaper than the
positional substitutions (the D and I counts are equal, e.g. 351 each at rho 0.5).

## The corruption rule, as implemented

- The substituted positions are the first `round(rho * n_u)` entries of a uniform permutation of
  the utterance's positions (Python `round`, half to even).
- Each replacement is drawn from the seed gold unigram (counted over all seed tokens, train and
  held), renormalised without the original symbol.
- The rng is `default_rng(SeedSequence([seed, first 8 bytes of sha256(tag)]))`. It depends only on
  the seed and the tag, so the ladder is nested: the substitutions at a smaller rho are the first
  ones made at a larger rho, with the same replacement symbols.
- The symbol set is the 39 ARPAbet phones, with no SIL. The job asserts this.

The corrupted strings therefore contain more adjacent repeats than the gold (about 7x at rho 0.5).
A port that collapses repeats would change both the string lengths and the rho actually realised.
