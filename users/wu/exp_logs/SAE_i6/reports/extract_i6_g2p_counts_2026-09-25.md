# i6 g2p / phone-corpus counts (2026-09-25)

## ApplyG2PModelJob (i6)
Job dir: /work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_core/g2p/apply/ApplyG2PModelJob.3eJqzOadjOqw
(also reachable via ./work/i6_core/g2p/apply/ApplyG2PModelJob.3eJqzOadjOqw)

16 task chunks (run.1..run.16), all present with finished.run.N markers and finished.split_word_list.1, finished.merge.1, finished.filter.1.

Per-chunk word-list input (words.NN) and g2p.lexicon.N output line counts (all match, none empty):
chunk 1: words.01=49384 lines, g2p.lexicon.1=49384 lines, 1845532 bytes
chunk 2: words.02=48659, g2p.lexicon.2=48659, 1786591 bytes
chunk 3: words.03=46610, g2p.lexicon.3=46610, 1791182 bytes
chunk 4: words.04=46469, g2p.lexicon.4=46469, 1793251 bytes
chunk 5: words.05=49432, g2p.lexicon.5=49432, 1831730 bytes
chunk 6: words.06=48947, g2p.lexicon.6=48947, 1800297 bytes
chunk 7: words.07=47883, g2p.lexicon.7=47883, 1840059 bytes
chunk 8: words.08=49483, g2p.lexicon.8=49483, 1829119 bytes
chunk 9: words.09=49138, g2p.lexicon.9=49138, 1815889 bytes
chunk 10: words.10=49470, g2p.lexicon.10=49470, 1822781 bytes
chunk 11: words.11=47805, g2p.lexicon.11=47805, 1805576 bytes
chunk 12: words.12=46622, g2p.lexicon.12=46622, 1805198 bytes
chunk 13: words.13=48616, g2p.lexicon.13=48616, 1789595 bytes
chunk 14: words.14=47965, g2p.lexicon.14=47965, 1791159 bytes
chunk 15: words.15=46623, g2p.lexicon.15=46623, 1794563 bytes
chunk 16: words.16=50567, g2p.lexicon.16=50567, 1820036 bytes
No empty chunk. Sum of chunk lexicon lines (wc -l work/g2p.lexicon.* total): 773673.

Merged output: output/g2p.lexicon, `wc -l` = 773672 (file has no trailing newline on last line, so line count undercounts by 1; last byte before EOF is a non-newline-terminated entry "Z Z AH Z AH Z AH Z AH Z" — i.e. true entry count = 773673, matching the chunk sum and the claimed JUPITER-expected 773,673).
output/g2p.untranslated: 299 bytes, contains only 16 "stack usage: N" lines (one per chunk), no word entries — i.e. 0 untranslated words recorded there.

## Band check (DITCHLIKE..RIVAW) in merged i6 lexicon
awk over output/g2p.lexicon, uppercased first field, DITCHLIKE <= w <= RIVAW:
count = 388779 (non-zero; band IS present, first entries include DITCHLIKE, DITCHLING, DITCHMOOR, DITCHOU, DITCHSIDE ...).
(Claimed JUPITER number for comparison: 388,780 missing in that same band — not judged here.)

## PhonemizeWithSilJob (i6) — equivalent of PhonemizeWithSilJob
Job dir: /work/asr4/hwu/setups/u/hwu/setups/librispeech-960/2026-09-24-unsupervised/i6_experiments/users/wu/experiments/unsupervised_asr/lm/phone_text/PhonemizeWithSilJob.NpoY1pGJWNUJ
Single task (finished.run.1 only; not chunked/parallelized in this graph — no per-chunk breakdown applicable).

output/stats.txt (verbatim):
lines_in=40418261
lines_out=40418258
dropped_oov=3
sil_prob=0.5
surround=True
seed=0
tokens=3342048858
sil_tokens=462273856
sil_token_rate=0.138320

No total non-SIL token count or word-count field is recorded in this stats.txt (only combined tokens and sil_tokens are given; non-SIL tokens would be tokens - sil_tokens = 3342048858 - 462273856 = 2879774... NOT computed here per instructions — reporting verbatim fields only). No "banked words" field present.
output/text.phn.gz: 1964052493 bytes (line count not separately verified beyond stats.txt's lines_out=40418258).

## Multi-task / chunked jobs check (bounded, maxdepth 5 under work/i6_core, work/i6_experiments)
Only one job in the current graph at that search depth has finished.run.N markers with N>1: ApplyG2PModelJob.3eJqzOadjOqw (16 chunks, detailed above, none empty). No other chunked job found within the search bound (searches deeper than maxdepth 5 were not attempted due to timeout risk from self-referential "work" subdirectories inside job dirs).
