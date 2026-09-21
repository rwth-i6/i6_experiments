# Descriptive read: what the soft/sf pack's dev-other decodes actually look like (fresh context, 2026-09-21)

Read-only, no gate. Question: the scorer arms (`soft_20`, `soft_20_s1`, `sf_20`, null `softshuf_20`)
are no better than the controls (`ctrl_20`, `ctrl_20_s1`) on PER and no closer on n-gram JS; what do
the strings look like, and does the error *pattern* differ?

## 0. Provenance and comparability

Six `BlankfreeGreedyPerJob` output dirs under
`/e/project1/spell/wu24/2026-07-13_unsupervised/work/speech_llm/sae/emc/blankfree_eval_jobs/`.
The arm->hash map in the dispatch is confirmed from each job's own `info` ALIAS line (not from a typed list):

| arm | hash | ALIAS | posteriors input |
|---|---|---|---|
| sf_20 | eQVw2XrcOWXg | sae/4a/blankfree/soft/sf_20/ep10/dev-other/per | ReturnnForwardJobV2.ZV8ROhl6N0jE |
| soft_20 | ZzEhEJ2ghmA7 | sae/4a/blankfree/soft/soft_20/ep10/dev-other/per | ReturnnForwardJobV2.5wpWXkTFjlrF |
| soft_20_s1 | 59MMKl1PFa6G | sae/4a/blankfree/soft/soft_20_s1/ep10/dev-other/per | ReturnnForwardJobV2.WDIpb3Nzussg |
| softshuf_20 | aP10M9b5T6bF | sae/4a/blankfree/soft/softshuf_20/ep10/dev-other/per | ReturnnForwardJobV2.C4LXm5pTyXwi |
| ctrl_20 | vhscomRFFJIJ | sae/4a/blankfree/prepro/ctrl_20/ep10/dev-other/per | ReturnnForwardJobV2.FpifZ1yGpWO6 |
| ctrl_20_s1 | mUXW2ACkKzDa | sae/4a/blankfree/prepro/ctrl_20_s1/ep10/dev-other/per | ReturnnForwardJobV2.NoS7IEeokfKQ |

All six jobs are the same job class, read the same gold (`GoldPhonesJob.ZGSp0hxyd2YP`) and the same
feature/length HDFs (`BlankfreeVadHdfJob.SAjz8y1cT06g`); the only differing input is the posteriors dump.
Same split (dev-other), same 2864 utterances, same denominator N = 177275 in every `per.txt`,
same kept epoch 10, same greedy blank-free decode. The comparison is therefore like-for-like; the
controls come from a different config file (`config_sae_4a_prepro_pack_v1.py`) than the scorer arms
(`config_sae_4a_soft_pack_v1.py`), whose docstring states the four soft arms are "identical to the
prepro pack's `ctrl_20` in everything but the named delta -- same bed streams, same priorshuf prior,
same frozen quantizer, same schedules, same kept checkpoints", with `soft_20_s1` on the `ctrl_20_s1`
seed. I did not re-verify the training-side identity claim from the training job hashes; it is taken
from that docstring.

Gold is SIL-free (39 symbols, 177275 tokens, 2864 utts, one utterance `1651-136854-0012` has 0 gold
phones). `greedy_phones.json` is `greedy_raw.json` minus SIL (checked: raw_emitted - SIL count =
phones_emitted for sf_20: 178253 - 8391 = 169862). Both sides of every comparison below are
SIL-free phone strings, so the normalisation matches. `per.json` and `decode_stats.json` are
byte-identical in all six jobs; there is **no confusion matrix** in them (`symbols` is just the
40-symbol inventory), so every alignment below is my own.

## 1. Method and internal check

All numbers below were recomputed from `greedy_phones.json` + `gold.json` with stdlib
`/usr/bin/python3` (3.9.25), scripts in `/tmp/claude-34349/.../scratchpad/{analyze,stats,paired,examples,ex2,ex3}.py`.
Alignment = standard Levenshtein DP per utterance with unit sub/del/ins cost and backtrace,
tie-break order substitution > insertion > deletion; "aligned pair" = a match or substitution slot.

My alignment reproduces every banked count exactly:

| arm | banked S/D/I (per.txt) | my S/D/I | banked PER | my PER |
|---|---|---|---|---|
| sf_20 | 131285 / 17791 / 10378 | 131285 / 17791 / 10378 | 0.899473 | 0.899473 |
| soft_20 | 128522 / 18985 / 8307 | 128522 / 18985 / 8307 | 0.878940 | 0.878940 |
| soft_20_s1 | 128321 / 18787 / 10690 | 128321 / 18787 / 10690 | 0.890131 | 0.890131 |
| softshuf_20 | 127255 / 19329 / 9588 | 127255 / 19329 / 9588 | 0.880959 | 0.880959 |
| ctrl_20 | 125787 / 18493 / 9750 | 125787 / 18493 / 9750 | 0.868876 | 0.868876 |
| ctrl_20_s1 | 127974 / 18482 / 9582 | 127974 / 18482 / 9582 | 0.880203 | 0.880203 |

The headline PERs in the dispatch (ctrl_20 0.869, soft_20 0.879, sf_20 0.899) are confirmed from
`per.txt` and independently re-derived. Per-utterance PER below = (S+D+I)/len(gold) from this same
alignment (the one empty-gold utterance is given per-utt PER 0; its insertions still enter the
corpus totals, which is why the totals match).

## 2. Twelve utterances side by side

Selection: 6 by `random.Random(20260921).sample(sorted(utt_ids), 6)`; 6 by largest
|per-utt PER(soft_20) - per-utt PER(ctrl_20)|. Strings verbatim.


### SIX RANDOM UTTERANCES (random.Random(20260921).sample(sorted(ids),6))
```
[4153-185072-0003]  gold_len=106  PER ctrl_20=0.821 soft_20=0.840 sf_20=0.887
  GOLD    : DH IY IH N D IY AH N Z W ER S UW N AE N S ER D B AY DH IY AH M EH R IH K AH N AO F IH S ER DH AE T DH AH W AA M P AH M W AH Z K AO R JH AH L IY AE K S EH P T IH D AH N D DH AE T AH K AH N T IH N Y UW AH N S AH V P IY S W AH Z AA R D AH N T L IY W IH SH T F AO R
  ctrl_20 : HH ER AA T AH S T IY W IY M N T AA N T M AH D R AY ER S EH N D AH T TH AA L AH IY AH V HH UW ER W AY T AW T HH W IY TH IH N AH V AA L M AE L AH D HH IY T TH Y UW S IH AH T AH T IY AA JH IH N W IY AA N AH T AH V W EY L Z K EY
  soft_20 : AA R IY M EY TH P W AE N ER TH R AH P N L IH NG AY AO R F OW B AH D M EY Z R AH M EH N D Y AA R W AH M N ER Z W S N AH N EY L OW F AH N T JH EY HH AW Z AA Y UW L IH N AH N EY L F W EY P R AO S N AH W EH R AH L EY L AA OW W EH P S T D
  sf_20   : W AY AH S T IY N TH Y UW AO R AH OW D DH IH S P IH S D UW AY G S T S IY Z DH IH M AA JH IY NG HH AY D EH K Y UW NG R OW Z HH Y UW JH P R IH T IY SH IH S P AH S IY D AW D DH HH W AY S K R IH S T SH IY D DH EY K R AH Y UW JH DH IH S T D IY Y AH P M IH L Z
[1255-90407-0001]  gold_len=66  PER ctrl_20=0.894 soft_20=0.879 sf_20=0.909
  GOLD    : B AH T EH Z DH AH R EY N G EY V N AA T DH AH L IY S T S AY N AH V S EH S EY SH AH N HH IY AH B Z ER V D AY TH IH NG K W IY SH AE L HH AE V T AH G OW B AE K N EH V ER
  ctrl_20 : HH AY B IY ER AO N T IH K JH EH N D ER T AE N Z TH M AE N JH M AE M AE N AH T Y UW S M AE N Z HH AY OW T W UW AH JH UH P IH R IH L D HH W EH L AH
  soft_20 : Y UW L EH R B AH TH IH T AH M R DH AY Z R K AH S Z AE N AH L AO AE N IY AE N AH N Z HH UW F IH N AH L M HH AY S ER M UW AE UH HH AO AA IH T ER IH NG AH DH IY M D AY
  sf_20   : HH W AY ER EH AH AE N D IH K AE N D EH R AH P AO R IH NG K AO AA R IH T HH UW UH JH IH K D HH UW M AH N Y UW ZH IY F K B R AH V R AH N S HH W AH S IY
[3660-172182-0028]  gold_len=64  PER ctrl_20=0.891 soft_20=0.875 sf_20=0.859
  GOLD    : AE N D HH IY G R IY T S DH IY W EH L AE Z AH N AH NG K AH L SH UH D G R IY T HH IH Z N EH F Y UW AE N D EH Z AH V AE S AH L SH UH D G R IY T HH IH Z L AO R D
  ctrl_20 : HH IY Y UW EH N D Z Y UW W AY N D HH B IY S T AA T AH V M AH NG D EH N B IY EH K B OY HH IY N T HH IY S K AE L AH V M AH NG D EH N F B IY M AE N D
  soft_20 : HH AW HH UW IH T AH P R UW W UH D HH EH R AO L IY M D AE N JH IH T AH L EH DH IY M ER AW L Z HH AO R OW K AH N D AE N JH IH T AH L EH K T AH L OY
  sf_20   : HH AW CH UW R AH D P AY N Y UW AH N Z HH ER EY DH AH S IY T OW D R AH S ER JH AH M OW Z HH AW N D HH ER JH K M AH P IY AO OW R AH S ER JH R IH L D
[1630-141772-0020]  gold_len=15  PER ctrl_20=1.000 soft_20=0.867 sf_20=1.133
  GOLD    : DH AH S T R EH CH ER Z M UW V D AO N
  ctrl_20 : HH ER M AE N R AH IY R EH N AH CH T
  soft_20 : AA R AE S T N P S B ER M T AW L Z
  sf_20   : HH W EH P R AE AH D T IY JH TH G OW K S EY N Z
[1255-90413-0006]  gold_len=40  PER ctrl_20=0.900 soft_20=0.850 sf_20=0.850
  GOLD    : HH IY D IH T ER M AH N D T UW K AO L IH M IY D IY AH T L IY AO N DH AH N UW IH N K AA R N EY SH AH N
  ctrl_20 : HH Y UW P R IH N AH T TH Z P R IH N S EH N AH D AH V HH CH T ER T EH N OY TH SH AA T IH S T EH N L AH T
  soft_20 : HH UW IH NG IH N AH EY Z AA R IH N AH L B AH EY K OW D AW AA R DH SH ER D AA IY TH IH N AH L AH N Z
  sf_20   : HH UW B R IH G IY D B R IH K AA AH T D IY Z HH EY D EH G OW Z DH AH D R IH L IH T
[1701-141760-0019]  gold_len=38  PER ctrl_20=0.842 soft_20=0.816 sf_20=0.868
  GOLD    : Y EH S AH N D AY HH AE V S AH M M AH N IY AH N D AH L EH T ER T IH G IH V Y UW HH IY AE D IH D
  ctrl_20 : HH OW L Z S B AY ER M T EH L AH S T CH AE L AH P R IH NG K OW Y UW AA L AH
  soft_20 : SH IY AW L AY HH AO AE N TH B IY L D F L OW K IY M EH R IH T IY M HH UW F JH
  sf_20   : HH ER AH P EY UW F ER AO R AH G AH IY SH EY K AA R AH S T B R AH K OW F UW SH IH S IY
```

### SIX WITH LARGEST |per-utt PER(soft_20) - per-utt PER(ctrl_20)|
```
[5543-27761-0085]  gold_len=10  PER ctrl_20=2.900 soft_20=2.200 sf_20=2.700
  GOLD    : AH W AY T W IH S P ER D
  ctrl_20 : HH CH W AY N D EH S T AH N D AH R AH OY TH W AY L Z R V D AH S AH L AH Z R IH AH V
  soft_20 : HH OW W ER Z B IY B IY M P M D W IY P S D EY K IY OW K AH P M D
  sf_20   : HH AW Y UW IH N D AA G AA D TH R IY Z HH Y UW P IY T D AA R IH TH R IY Z
[4515-11057-0053]  gold_len=3  PER ctrl_20=1.667 soft_20=1.000 sf_20=1.667
  GOLD    : Y EH S
  ctrl_20 : HH W R L Z
  soft_20 : IH SH IY
  sf_20   : HH W AY N TH
[1630-73710-0004]  gold_len=5  PER ctrl_20=1.200 soft_20=1.800 sf_20=1.800
  GOLD    : M AY N OW N
  ctrl_20 : HH R N T AA N OY T
  soft_20 : B AH L Z TH R ER Z AY
  sf_20   : HH W IH N D DH AH V D Z
[1686-142278-0068]  gold_len=2  PER ctrl_20=2.000 soft_20=2.500 sf_20=2.500
  GOLD    : N OW
  ctrl_20 : HH R EH N OY
  soft_20 : AA DH ER D AY
  sf_20   : HH W AH N Z SH
[4323-55228-0038]  gold_len=13  PER ctrl_20=1.077 soft_20=1.538 sf_20=1.154
  GOLD    : F AY V D EY Z IH N D IY D S ER
  ctrl_20 : K AE N R IH N IY S T R IH N D M AE N
  soft_20 : S T AH L Z M IH NG AH P F TH IH NG AH L M AE N AH D
  sf_20   : M IH NG N K R IH N JH EY D R AH N D AO R IH L
[2506-169427-0006]  gold_len=12  PER ctrl_20=1.000 soft_20=1.417 sf_20=1.417
  GOLD    : AH L AE S P UW R Y AO R IH K
  ctrl_20 : HH S T AE L Z TH R IH EY R EY AH D Z
  soft_20 : AA IY L K L IY L IY AA IH S T L NG AH L IY M
  sf_20   : HH W AA R AH N TH W IH SH W IH L SH IY Z N S
```

The six largest-|delta| utterances are all 2-13 gold phones long: the criterion selects tiny
denominators, so the table below adds the same criterion restricted to gold length >= 40, where the
strings are representative of the corpus.

### SUPPLEMENT: same criterion restricted to gold_len>=40
```
[700-122866-0032]  gold_len=41  PER ctrl_20=0.780 soft_20=0.976 sf_20=0.756
  GOLD    : IY CH G ER L HH AE Z T AH R IY D HH ER S T AO R IY AW T L AW D AE N D DH EH N W IY T AO K IH T OW V ER
  ctrl_20 : HH AA L D Z IH N F IY M P AO N F ER UW M EY N AA N D AE N D SH IY T W UW IH L S AA N AH V
  soft_20 : HH AH P IH T AH D HH IY EH AA R B AH L UW R AA AE S AH D R V Z K CH V CH AW Y UW W UW N AH M F ER M D
  sf_20   : HH AH D TH R IH N F ER B R AH S CH ER D AO R IH NG DH AE D R AE TH D DH EY AY G UW R IH T AH V IY
[6267-53049-0001]  gold_len=72  PER ctrl_20=0.986 soft_20=0.792 sf_20=1.000
  GOLD    : P AH N EH L AH P IY HH AE Z S T AH D IY D S OW HH AA R D AO L W IH N T ER AE N D SH IY HH AE Z AH N T G AO N EH N IY W EH R TH AO T DH AH OW L D ER S IH S T ER W IH S T F AH L IY
  ctrl_20 : HH S T AH D AH V F UH IY D M R AE L AH V TH M AE N OY DH AE N D HH AY D W AY N T AH V HH IY T Y UW F UH IY AH T R IH CH T SH AA T AH V W AY N D HH AE N OY D HH ER AA N AH V M AE AH Z AH V TH W AY L Z K AH
  soft_20 : IH N L AH OW M D HH EH L S CH AH L EY L AA AE N ER D HH AH L Z HH AH L W TH M D HH AW AE N D HH IY JH P EY Z IH T L Z R IY L D W UW S T AH L AA R AH L D AE N EH S L D W P S L
  sf_20   : HH W AA S T S AH S IY Z F ER AO R AH N S T N D AO R AH V Z F IH L D HH SH Y UW AH N W IY HH AW Y UW F ER JH IY D R AH N D DH AH S T Y UW L Z HH W IH AE UH D HH W AY DH AH N S T AO R AH P IY Z HH Y UW AH P M AA IY
[1255-74899-0016]  gold_len=58  PER ctrl_20=0.948 soft_20=0.776 sf_20=0.879
  GOLD    : W IH TH HH ER DH EH R W AH Z AH R IY L W IH SH DH AH T DH AH P OW L Z M AY T B IY JH OY N D T AH G EH DH ER B AY HH ER F Y UW CH ER HH AH Z B AH N D
  ctrl_20 : HH W AY IY F ER HH Y UW W IY S AO N D W AY Z TH Y UW D ER R IH N D IY EH N D R IH G R AE N T TH IH P R IH L AH V R AY F ER K OW N D AH F UH L IY AY T
  soft_20 : W IY EH HH UW Y EH R OW B AH L W IY Y UW AA R IH N AH L P B AH Z IH T AH Z AA R IH T IY L D IH NG AY HH UW S SH N IY S EY Z
  sf_20   : HH Y UW M F ER IY HH W AY Y UW JH K R AH N Y UW TH AY D EH R AH N TH AE N D AA R IH NG D B R AH S IY R UW F ER M OW D T F AH JH R IY
[1650-157641-0009]  gold_len=42  PER ctrl_20=0.833 soft_20=1.000 sf_20=1.024
  GOLD    : AY B IH L IY V IH N M AY HH AA R T DH AE T W IH M AH N ER JH EH L AH S AH V IH T AE Z AH V AH R AY V AH L
  ctrl_20 : HH AY R IH G AE N AH S T AY DH AE N D Y UW ER W AY T AH T EY R AE L AH Z AH JH AH D HH B IY M AE L AH S AO N AH V
  soft_20 : HH AY IH NG K AH M F AO B AY HH AH L Z Y UW AO R W IY G EY L F IH N IY AH L EY P AO R L EY Z L HH S N JH L AO R OW B AH M D
  sf_20   : HH UW AA R AH N K EY D UW F IH L D AY EH R Y AH S IY EY UH R AH S IY JH P EY K UH D HH AW ER AO R AE EY K AA R IH NG K IY Z
[8254-84205-0017]  gold_len=45  PER ctrl_20=0.800 soft_20=0.956 sf_20=0.933
  GOLD    : W EH L DH EY SH OW D DH AH M S EH L V Z T IH M IY AY D IH N T W AA N T DH AH M S EH D G R IH G Z D R AY L IY
  ctrl_20 : HH W AY Y UW M AE N D T M AE N Z P EH V HH AY AH T W AY D AW T TH AE NG D IH D Z IY AE N AH V
  soft_20 : UH Y UW AE N ER Y UW TH AE N UH P AA R B OW D HH AY L JH EY Z W IY Z Y Z N JH IH T IY M P T AH L OW D
  sf_20   : Y UW N AY AO R AH V S AY D R AH N TH B AA G OW Z HH UW S AH D Y UW D AY Z AO R AE R AH S TH R IH NG S IY Z
[3663-172528-0054]  gold_len=59  PER ctrl_20=0.780 soft_20=0.932 sf_20=0.915
  GOLD    : AY L IH F T AH D M AY F UH T AE N D L EH T DH AH W AO T ER R AH N AW T DH EH N W EH N AY HH AE D M AW N T AH D W IY M EY D HH EY S T F ER R OW M
  ctrl_20 : HH AY AE L K AH D R AY K AE L D HH IY T AE N D ER W EY N AH AO T AA N D HH AW T HH W AY T B AY D ER D R EH N T AH D HH W UW R EH D F AE N D Z K EY S AO N T
  soft_20 : HH AY K IY M EY B AY S T IY T L M HH AW K IY Z AA R W AH M EY B IY L V Z L Y UW Z W IY L AY AA D JH Z B V Z M EY W UW B AH L AA HH AH S AA T UW B ER Z
  sf_20   : HH UW R AH S IY D UW M AH N D HH AW EY R AE N D EH R UW IH S IY T AH S AE N D HH W AY N Z HH Y UW S UW Z UW F ER D AE D S IY D HH Y UW IH N D F IH N P B R AH V Z
```


**What the strings look like.** Every arm emits fluent-looking English-phonotactic nonsense at
roughly the right length and the right phone inventory (all 39 gold phones used by every arm except
ctrl_20, which never emits ZH (38 of 39); no arm emits a symbol outside the gold inventory). The decodes are word-shaped (`M AE N D`, `R EH S T`,
`EY SH AH N`, `HH AE V`) but bear no systematic relation to the reference: in the long examples above
the hypothesis shares essentially no content with the gold beyond short accidental runs. There is no
degenerate mode (no repeated loops, no collapse to one phone, no empty output; shortest hypothesis 3-5 phones: min length per arm is
sf_20 5, soft_20 3, soft_20_s1 4, softshuf_20 4, ctrl_20 4, ctrl_20_s1 4). Utterance onsets are the most visible artefact: the decoder starts nearly every utterance
with HH.

### 2a. Utterance-initial phone (share of the 2863 non-empty-gold utterances)

```
GOLD         DH 15.3% HH 10.9% AY 8.6% IH 7.5% W 7.0%
sf_20        HH 83.4% W 5.6% AW 1.8% AO 1.5% Y 1.4%
soft_20      HH 31.2% Y 10.2% W 8.4% IH 8.3% AA 6.2%
soft_20_s1   AH 14.8% AA 14.1% EY 13.0% HH 10.1% W 7.4%
softshuf_20  HH 77.6% AE 8.1% P 4.6% W 1.4% AY 1.1%
ctrl_20      HH 90.7% M 3.0% Y 0.8% IH 0.8% ER 0.8%
ctrl_20_s1   HH 81.1% UW 2.5% ER 2.3% W 2.0% R 1.7%

```

### 2b. Utterance-initial bigram (top 3)

```
GOLD         DH+AH 6.1% HH+IY 4.9% IH+T 4.1%
sf_20        HH+W 31.7% HH+UW 13.9% HH+AW 13.0%
soft_20      Y+UW 7.9% HH+UW 7.8% HH+AY 5.6%
soft_20_s1   AH+N 7.8% AA+R 7.0% HH+W 5.6%
softshuf_20  HH+AE 25.5% HH+AH 9.3% HH+AY 7.5%
ctrl_20      HH+AY 12.4% HH+IY 10.8% HH+W 9.5%
ctrl_20_s1   HH+UW 29.5% HH+ER 10.1% HH+IH 6.4%

```

Computed as `Counter(hyp[u][0])` over the 2863 utterances with non-empty gold. Gold starts with DH
15.3% / HH 10.9%. ctrl_20 starts with HH in 90.7% of utterances, ctrl_20_s1 81.1%, sf_20 83.4%,
softshuf_20 77.6% -- but soft_20 only 31.2% and soft_20_s1 only 10.1%. The onset artefact is
present in both controls and is *not* something the scorer removes consistently: the two soft seeds
sit at opposite ends of the whole range (31.2% vs 10.1%), i.e. this statistic's seed spread is larger
than any arm-vs-control gap.

## 3. Length, repeats, diversity, unigrams, 4-grams

All counts over the full 2864-utterance dev-other decode; "gold" row computed the same way on
`gold.json["dev-other"]`.

```
GOLD: tokens=177275  distinct_phones=39  adjacent_repeat_frac=0.00567 (989/174412)  distinct4g=54049  distinct4g_per10k=3048.9
GOLD has SIL: False

=== 2. LENGTH / REPEAT / DIVERSITY ===
arm            hyptok len_rat rep_frac distinct_ph      d4g   d4g/10k   hasSIL
sf_20          169862  0.9582   0.0002       39     51787   3048.8    False
soft_20        166597  0.9398   0.0003       39     55191   3312.8    False
soft_20_s1     169178  0.9543   0.0004       39     52696   3114.8    False
softshuf_20    167534  0.9451   0.0005       39     52847   3154.4    False
ctrl_20        168532  0.9507   0.0001       38     46331   2749.1    False
ctrl_20_s1     168375  0.9498   0.0003       39     53004   3148.0    False

```

`len_rat` = sum(hyp tokens)/177275. `rep_frac` = adjacent identical phones / (tokens - utts).
`d4g/10k` = distinct hypothesis 4-gram types per 10000 hypothesis tokens.

**Length**: every arm is short of the reference by 4-6%: sf_20 0.958, soft_20 0.940, soft_20_s1
0.954, softshuf_20 0.945, ctrl_20 0.951, ctrl_20_s1 0.950. The soft arms straddle the controls; the
control seed spread (0.951 vs 0.950) is small but soft_20 (0.940) is the shortest arm and soft_20_s1
(0.954) the second longest, so the soft seed spread (0.014) exceeds the soft-vs-ctrl gap (0.011).

**Adjacent repeats**: an artefact, not a model property. In `greedy_raw.json` there are **zero**
adjacent repeats in every arm (sf_20 0/175389, ctrl_20 0/173446, etc.) -- the blank-free greedy decode
collapses runs by construction. The 17-90 repeats visible in the SIL-stripped strings are exactly
the joins created by removing a SIL between two identical phones. Gold has 989 repeats (0.57% of
tokens), which no arm can produce: a structural floor of ~0.56 PER points shared by all six arms.

**4-gram diversity**: ctrl_20 is the *least* diverse arm (2749 distinct 4-grams per 10k tokens)
and soft_20 the most (3313); gold is 3049. But ctrl_20_s1 is at 3148, inside the soft range, so the
control seed spread (2749-3148, 399) is as large as the soft-vs-control difference. Diversity does
not separate the conditions.

### 3a. Unigram over- and under-production (hyp rate minus gold rate, percentage points of tokens)

```
sf_20        OVER : R +5.00(15522 vs 7341) UW +2.14(6046 vs 2511) JH +1.67(3565 vs 753) S +1.65(11156 vs 8720) Y +1.45(3781 vs 1378)
sf_20        UNDER: T -3.42(6096 vs 12424) N -2.44(8021 vs 12700) L -2.14(2971 vs 6894) DH -1.65(2393 vs 5427) EH -1.28(3119 vs 5533)
soft_20      OVER : L +2.65(10901 vs 6894) UW +1.73(5243 vs 2511) TH +1.16(2905 vs 1029) P +1.12(4856 vs 3182) IY +1.02(8217 vs 6934)
soft_20      UNDER: T -3.31(6163 vs 12424) DH -2.03(1716 vs 5427) S -1.62(5490 vs 8720) AH -1.60(13220 vs 16900) N -1.40(9606 vs 12700)
soft_20_s1   OVER : AH +2.63(20570 vs 16900) ER +1.47(7009 vs 4744) AY +1.36(5557 vs 3416) AA +1.33(4758 vs 2624) UW +1.05(4173 vs 2511)
soft_20_s1   UNDER: T -2.20(8127 vs 12424) AE -1.67(1978 vs 5025) S -1.62(5574 vs 8720) EH -1.56(2649 vs 5533) D -1.38(6414 vs 9176)
softshuf_20  OVER : R +3.52(12827 vs 7341) AH +2.92(20867 vs 16900) K +1.31(6521 vs 4585) JH +1.31(2898 vs 753) L +1.26(8621 vs 6894)
softshuf_20  UNDER: D -2.88(3844 vs 9176) S -2.30(4380 vs 8720) DH -2.26(1337 vs 5427) N -1.95(8734 vs 12700) IH -1.65(7935 vs 11319)
ctrl_20      OVER : AE +2.69(9314 vs 5025) N +2.38(16087 vs 12700) V +1.29(5462 vs 3466) HH +1.24(6211 vs 4340) T +0.85(13244 vs 12424)
ctrl_20      UNDER: DH -2.60(785 vs 5427) AH -1.75(13122 vs 16900) S -1.04(6529 vs 8720) F -0.95(1532 vs 3298) B -0.78(1643 vs 3108)
ctrl_20_s1   OVER : UW +1.88(5552 vs 2511) EH +1.22(7305 vs 5533) S +1.20(10305 vs 8720) F +1.14(5055 vs 3298) Y +1.11(3170 vs 1378)
ctrl_20_s1   UNDER: DH -2.17(1495 vs 5427) T -2.09(8288 vs 12424) IY -1.71(3706 vs 6934) N -1.45(9627 vs 12700) V -1.11(1430 vs 3466)

```

Each arm has a *different* set of favourite phones (sf_20 floods R +5.0pp; ctrl_20 floods AE +2.7pp
and N +2.4pp; soft_20 floods L +2.7pp; ctrl_20_s1 floods UW +1.9pp), and the two control seeds
disagree with each other as strongly as any arm disagrees with a control (ctrl_20 over-produces N
+2.38pp while ctrl_20_s1 *under*-produces N -1.45pp). The one thing every arm shares: DH and T are
under-produced by every single arm (DH -1.65 to -2.60pp in 5 of 6 arms).

Unigram Jensen-Shannon divergence to gold (natural log, JS = 0.5 KL(p||m) + 0.5 KL(q||m)):
sf_20 0.03000, soft_20 0.02064, soft_20_s1 0.01811, softshuf_20 0.02555, **ctrl_20 0.01739**,
ctrl_20_s1 0.02014. The control is the closest arm at n=1, and the soft range (0.018-0.021) brackets
the control range (0.017-0.020). Consistent with the banked "no closer by n-gram JS" premise.

### 3b. Top 15 hypothesis 4-grams, with their gold count

```
-- sf_20 (total 4gram tokens 161270, 51787 distinct)
   R AH S T                  630  0.391%  gold_count=34
   AO R AH N                 601  0.373%  gold_count=22
   R AH S IY                 569  0.353%  gold_count=16
   AO R AH S                 496  0.308%  gold_count=10
   R AH N D                  408  0.253%  gold_count=27
   R IH S IY                 359  0.223%  gold_count=10
   D AO R AH                 351  0.218%  gold_count=16
   R IH S T                  336  0.208%  gold_count=11
   R AE N D                  323  0.200%  gold_count=32
   R IH N D                  287  0.178%  gold_count=3
   D R AH N                  267  0.166%  gold_count=22
   D R AH S                  266  0.165%  gold_count=2
   R AH N S                  250  0.155%  gold_count=22
   AH S T IY                 238  0.148%  gold_count=5
   R AH N Z                  222  0.138%  gold_count=3
   token-share of hyp 4grams that exist in gold: 37.41% ; type-share: 17.09%
-- soft_20 (total 4gram tokens 158005, 55191 distinct)
   IH N AH L                 653  0.413%  gold_count=12
   IH T AH L                 430  0.272%  gold_count=111
   AE N AH L                 349  0.221%  gold_count=3
   IH N AH M                 336  0.213%  gold_count=15
   R IH N AH                 307  0.194%  gold_count=2
   IH NG AH L                250  0.158%  gold_count=6
   R IH T AH                 239  0.151%  gold_count=12
   AE S T AH                 199  0.126%  gold_count=12
   AA R IH NG                197  0.125%  gold_count=0
   N AH L D                  194  0.123%  gold_count=1
   D IH N AH                 184  0.116%  gold_count=22
   AE N AH M                 182  0.115%  gold_count=24
   R AE N AH                 174  0.110%  gold_count=6
   AA R IH N                 170  0.108%  gold_count=2
   S T AH L                  162  0.103%  gold_count=7
   token-share of hyp 4grams that exist in gold: 36.12% ; type-share: 17.78%
-- soft_20_s1 (total 4gram tokens 160586, 52696 distinct)
   EY SH AH N                831  0.517%  gold_count=143
   AH B AH N                 633  0.394%  gold_count=2
   AH N T ER                 619  0.385%  gold_count=12
   AH N T IH                 615  0.383%  gold_count=81
   AH N S IH                 601  0.374%  gold_count=33
   HH W AH N                 459  0.286%  gold_count=29
   AH N L IY                 444  0.276%  gold_count=37
   SH AH N T                 310  0.193%  gold_count=33
   IH NG AH N                281  0.175%  gold_count=28
   W AH N T                  280  0.174%  gold_count=14
   R AH N T                  276  0.172%  gold_count=39
   AH N V ER                 272  0.169%  gold_count=4
   K AH N T                  257  0.160%  gold_count=67
   AH N T UW                 250  0.156%  gold_count=28
   L IY AH N                 237  0.148%  gold_count=25
   token-share of hyp 4grams that exist in gold: 42.39% ; type-share: 19.73%
-- softshuf_20 (total 4gram tokens 158942, 52847 distinct)
   R IH K AH                 261  0.164%  gold_count=14
   AH L AH T                 253  0.159%  gold_count=7
   AH Z AH M                 219  0.138%  gold_count=2
   R IH N T                  219  0.138%  gold_count=3
   EH Z AH M                 201  0.126%  gold_count=4
   EH Z AH T                 198  0.125%  gold_count=4
   HH AE V AH                184  0.116%  gold_count=24
   R IY N T                  176  0.111%  gold_count=1
   R EH Z AH                 175  0.110%  gold_count=30
   HH AE V R                 174  0.109%  gold_count=11
   AH T ER HH                172  0.108%  gold_count=0
   R IY Z AH                 171  0.108%  gold_count=23
   R EH N T                  171  0.108%  gold_count=1
   Z AH L AH                 168  0.106%  gold_count=3
   T HH AE V                 166  0.104%  gold_count=31
   token-share of hyp 4grams that exist in gold: 41.39% ; type-share: 19.91%
-- ctrl_20 (total 4gram tokens 159940, 46331 distinct)
   M AE N D                  557  0.348%  gold_count=62
   R IH N D                  507  0.317%  gold_count=3
   M AE N T                  497  0.311%  gold_count=9
   R IH N T                  418  0.261%  gold_count=3
   AE N AH V                 349  0.218%  gold_count=6
   N T AH V                  314  0.196%  gold_count=38
   M AE N AH                 304  0.190%  gold_count=21
   T M AE N                  292  0.183%  gold_count=10
   AE L AH V                 287  0.179%  gold_count=0
   AE N T AH                 281  0.176%  gold_count=14
   R IH N AH                 272  0.170%  gold_count=2
   M AE L AH                 258  0.161%  gold_count=2
   IH L AH V                 253  0.158%  gold_count=9
   K AE N AH                 253  0.158%  gold_count=3
   K AE N T                  250  0.156%  gold_count=24
   token-share of hyp 4grams that exist in gold: 46.74% ; type-share: 21.57%
-- ctrl_20_s1 (total 4gram tokens 159783, 53004 distinct)
   R EH S T                  448  0.280%  gold_count=72
   IH S T AH                 352  0.220%  gold_count=48
   EH S T AH                 346  0.217%  gold_count=27
   AH M IH S                 271  0.170%  gold_count=11
   AH M IH N                 252  0.158%  gold_count=35
   R EH S AH                 252  0.158%  gold_count=6
   R EH N IY                 239  0.150%  gold_count=4
   IH S AH M                 214  0.134%  gold_count=3
   M IH S T                  202  0.126%  gold_count=91
   S T Y UW                  193  0.121%  gold_count=2
   R EH K T                  189  0.118%  gold_count=6
   S T AH M                  186  0.116%  gold_count=18
   S T AH D                  172  0.108%  gold_count=22
   M IH S AH                 172  0.108%  gold_count=0
   IH K AH M                 170  0.106%  gold_count=18
   token-share of hyp 4grams that exist in gold: 37.61% ; type-share: 19.38%

```

Every arm has a small set of high-frequency invented "words" that are 10-100x over-represented
relative to gold (`R AH S T` 630 vs gold 34 in sf_20; `M AE N D` 557 vs 62 in ctrl_20; `IH N AH L`
653 vs 12 in soft_20). Only 36-47% of hypothesis 4-gram tokens exist anywhere in gold's dev-other
4-gram set; the ranking is ctrl_20 46.7% > soft_20_s1 42.4% > softshuf_20 41.4% > ctrl_20_s1 37.6% >
sf_20 37.4% > soft_20 36.1% -- again with the two controls at the two ends of the range.

## 4. Consistency of the code

From the Levenshtein alignment's match+substitution pairs (~158k of 177275 gold tokens per arm;
deletions and insertions are excluded, as they have no counterpart). MI and entropies in nats.
NMI_arith = 2I/(H(g)+H(h)) (the sklearn default normalisation); NMI_sqrt = I/sqrt(H(g)H(h)).

```
arm             aligned  H(gold)   H(hyp)       MI NMI_arith NMI_sqrt
sf_20            159484   3.3431   3.3831   0.2227   0.0662   0.0662
soft_20          158290   3.3421   3.3910   0.2830   0.0841   0.0841
soft_20_s1       158488   3.3401   3.3347   0.2218   0.0665   0.0665
softshuf_20      157946   3.3422   3.2815   0.2338   0.0706   0.0706
ctrl_20          158782   3.3445   3.2946   0.3102   0.0934   0.0934
ctrl_20_s1       158793   3.3431   3.3696   0.2676   0.0797   0.0797

```

**The control is the most consistent code, not the scorer arms.** NMI ranking:
ctrl_20 0.0934 > soft_20 0.0841 > ctrl_20_s1 0.0797 > softshuf_20 0.0706 > soft_20_s1 0.0665 ~
sf_20 0.0662. The seed spreads are ctrl 0.0934-0.0797 (0.0137) and soft 0.0841-0.0665 (0.0176);
every scorer arm lies inside or below the control band, and the two soft seeds differ from each
other more than soft_20 differs from ctrl_20_s1. H(gold) ~ 3.34 nats throughout, so MI accounts for
only 6.6-9.3% of the available entropy in all six arms.

### 4a. Per-gold-phone modal hypothesis phone and its share

```
gold   goldN | sf_20          | soft_20        | soft_20_s1     | softshuf_20    | ctrl_20        | ctrl_20_s1    
AH     16900 | AH  0.33*      | AH  0.29*      | AH  0.45*      | AH  0.49*      | AH  0.36*      | AH  0.43*     
N      12700 | N   0.23*      | N   0.24*      | N   0.34*      | N   0.23*      | N   0.40*      | N   0.27*     
T      12424 | T   0.14*      | T   0.16*      | T   0.23*      | T   0.27*      | T   0.27*      | T   0.23*     
IH     11319 | IH  0.19*      | IH  0.27*      | IH  0.25*      | IH  0.21*      | IH  0.26*      | IH  0.28*     
D       9176 | D   0.35*      | D   0.18*      | D   0.25*      | AH  0.09       | D   0.30*      | D   0.29*     
S       8720 | S   0.25*      | S   0.21*      | S   0.14*      | S   0.14*      | S   0.16*      | S   0.31*     
R       7341 | R   0.43*      | R   0.21*      | R   0.18*      | R   0.36*      | R   0.22*      | R   0.15*     
IY      6934 | IY  0.21*      | IY  0.16*      | IY  0.14*      | IY  0.19*      | IY  0.14*      | IY  0.10*     
L       6894 | R   0.12       | L   0.39*      | L   0.18*      | L   0.24*      | L   0.18*      | L   0.13*     
EH      5533 | AH  0.12       | IY  0.11       | AH  0.13       | EH  0.19*      | AE  0.11       | EH  0.25*     
DH      5427 | R   0.12       | Y   0.09       | DH  0.11*      | R   0.09       | N   0.07       | ER  0.10      
M       5368 | G   0.11       | M   0.13*      | M   0.09*      | AH  0.11       | T   0.18       | M   0.20*     
AE      5025 | AE  0.18*      | AE  0.12*      | AH  0.15       | AE  0.16*      | AE  0.26*      | AE  0.13*     
Z       4886 | JH  0.09       | Z   0.10*      | K   0.11       | Z   0.12*      | Z   0.22*      | CH  0.08      
ER      4744 | ER  0.11*      | D   0.10       | ER  0.20*      | ER  0.16*      | ER  0.12*      | AH  0.11      
K       4585 | R   0.14       | N   0.09       | K   0.17*      | K   0.23*      | IH  0.09       | K   0.15*     
HH      4340 | HH  0.21*      | HH  0.44*      | AY  0.10       | HH  0.24*      | HH  0.23*      | HH  0.21*     
W       4098 | Y   0.15       | W   0.54*      | W   0.19*      | W   0.09*      | W   0.56*      | AY  0.12      
V       3466 | S   0.08       | M   0.09       | P   0.08       | V   0.11*      | V   0.19*      | T   0.09      
AY      3416 | UW  0.09       | AY  0.29*      | AY  0.12*      | AH  0.12       | AY  0.33*      | N   0.09      
F       3298 | M   0.09       | S   0.13       | AH  0.11       | F   0.12*      | K   0.18       | F   0.15*     
P       3182 | P   0.14*      | P   0.15*      | N   0.10       | R   0.10       | IH  0.12       | S   0.09      
B       3108 | R   0.13       | NG  0.12       | B   0.19*      | R   0.14       | R   0.14       | B   0.11*     
EY      2900 | IH  0.11       | AH  0.13       | EY  0.13*      | AH  0.12       | N   0.15       | S   0.11      
AO      2646 | AO  0.15*      | AH  0.12       | AH  0.12       | AH  0.09       | EY  0.08       | AO  0.15*     
AA      2624 | AA  0.10*      | AH  0.12       | AA  0.15*      | AH  0.14       | AA  0.11*      | IH  0.10      
UW      2511 | UW  0.15*      | UW  0.11*      | UW  0.11*      | UW  0.20*      | N   0.11       | UW  0.13*     
OW      2407 | AH  0.15       | ER  0.15       | OW  0.11*      | AH  0.12       | N   0.13       | N   0.12      
NG      1832 | T   0.11       | NG  0.11*      | IY  0.12       | T   0.16       | N   0.14       | V   0.11      
G       1704 | R   0.14       | T   0.12       | AH  0.12       | IY  0.10       | IH  0.11       | IH  0.12      
SH      1400 | R   0.14       | N   0.18       | AH  0.11       | R   0.09       | L   0.15       | S   0.15      
Y       1378 | Y   0.11*      | SH  0.17       | AE  0.08       | AH  0.17       | OW  0.12       | Y   0.10*     
AW      1119 | AE  0.15       | V   0.13       | AH  0.15       | N   0.14       | N   0.16       | IH  0.15      
TH      1029 | M   0.11       | S   0.11       | G   0.10       | R   0.10       | N   0.09       | T   0.11      
UH       937 | UW  0.09       | IH  0.07       | AH  0.15       | AH  0.09       | IH  0.09       | AY  0.12      
CH       921 | R   0.13       | N   0.13       | N   0.11       | AH  0.11       | Z   0.10       | D   0.11      
JH       753 | R   0.13       | N   0.10       | N   0.07       | JH  0.10*      | R   0.11       | D   0.13      
OY       162 | NG  0.11       | AH  0.16       | AH  0.19       | D   0.15       | N   0.20       | IH  0.14      
ZH        68 | T   0.13       | N   0.22       | R   0.11       | M   0.14       | N   0.15       | S   0.22      
(* = modal hypothesis phone equals the gold phone)

```

```
mean modal share (weighted by aligned count) and #gold phones whose modal hyp == itself:
  sf_20        mean_modal_share=0.1933  n_gold_phones_with_self_modal=16/39  mass_of_those=63.9%
  soft_20      mean_modal_share=0.2013  n_gold_phones_with_self_modal=18/39  mass_of_those=72.3%
  soft_20_s1   mean_modal_share=0.2025  n_gold_phones_with_self_modal=20/39  mass_of_those=75.9%
  softshuf_20  mean_modal_share=0.2074  n_gold_phones_with_self_modal=19/39  mass_of_those=71.9%
  ctrl_20      mean_modal_share=0.2273  n_gold_phones_with_self_modal=17/39  mass_of_those=71.0%
  ctrl_20_s1   mean_modal_share=0.2038  n_gold_phones_with_self_modal=19/39  mass_of_those=73.9%
```

Chance baseline: under independence every gold phone's modal hypothesis would be the globally most
frequent hypothesis phone, giving a modal share equal to max_h p_hyp(h) = sf_20 0.091 (R),
soft_20 0.079 (AH), soft_20_s1 0.122 (AH), softshuf_20 0.125 (AH), ctrl_20 0.096 (N),
ctrl_20_s1 0.098 (AH). Observed weighted modal shares are 0.193-0.227, i.e. roughly twice chance but
nowhere near a relabelling. **This is not a consistent private code**: at best ~20% of a gold phone's
aligned tokens go to one hypothesis symbol, and for 19-23 of 39 gold phones the modal hypothesis is
not even the phone itself. The frequent phones do map to themselves (AH->AH 0.29-0.49, N->N
0.23-0.40, T->T 0.14-0.27, IH->IH 0.19-0.28) in every arm including both controls; the code is a
blurred identity map with ~20-30% hit rate on the top phones, not a permutation. ctrl_20 has the
highest mean modal share (0.227) of all six arms; ctrl_20_s1 (0.204) sits in the middle of the
scorer arms (0.193-0.207), so the scorer does not change this either way beyond seed noise.

## 5. Paired per-utterance deltas against ctrl_20

delta = per-utt PER(arm) - per-utt PER(ctrl_20), positive = arm worse, over all 2864 utterances.
r_len / rho_len are Pearson and Spearman correlations of the delta with gold utterance length.

```
arm            mean_d   median       sd       p5      p25      p75      p95   frac>0     wins     r_len   rho_len
sf_20          0.0369   0.0308   0.0741  -0.0625   0.0000   0.0698   0.1552    0.655      592   -0.1241   -0.1219
soft_20        0.0117   0.0034   0.0725  -0.0952  -0.0250   0.0464   0.1250    0.500      975   -0.0333   -0.0354
soft_20_s1     0.0292   0.0217   0.0806  -0.0714  -0.0132   0.0625   0.1538    0.589      791   -0.1420   -0.1442
softshuf_20    0.0202   0.0127   0.0771  -0.0789  -0.0206   0.0508   0.1481    0.529      945   -0.1533   -0.1708
ctrl_20_s1     0.0142   0.0090   0.0746  -0.0882  -0.0238   0.0455   0.1290    0.512      989   -0.0552   -0.0521

(utt-level PER = (S+D+I)/len(gold) from my own Levenshtein; 2864 utts; corpus-level PER of ctrl_20=0.8689)
macro-mean per-utt PER: sf_20=0.9283, soft_20=0.9031, soft_20_s1=0.9205, softshuf_20=0.9116, ctrl_20=0.8913, ctrl_20_s1=0.9055

```

Every arm is worse than ctrl_20 on average, including the other control seed: ctrl_20_s1 +0.0142,
soft_20 +0.0117, softshuf_20 +0.0202, soft_20_s1 +0.0292, sf_20 +0.0369. **soft_20's deficit
(+0.0117) is smaller than the control seed's own deficit (+0.0142)**, i.e. soft_20 is inside the
replicate band; soft_20_s1 (+0.0292) is outside it, as is sf_20 (+0.0369) -- but note soft_20_s1 is
compared to the wrong-seed control here; against its own seed's control (ctrl_20_s1) the soft seed
difference is +0.0174 (soft_20_s1 - soft_20) versus +0.0142 (ctrl_20_s1 - ctrl_20). Per-utterance
win counts are near even for soft_20 (975 of 2864 utterances beat ctrl_20; median delta +0.0034,
sd 0.0725, p5 -0.095 / p95 +0.125), while sf_20 loses on 65.5% of utterances.

The deltas correlate *negatively* with utterance length (sf_20 r = -0.124, soft_20_s1 r = -0.142,
softshuf_20 r = -0.153; soft_20 only -0.033), so the scorer arms are relatively worse on short
utterances.

### 5a. The worst 20 utterances

```
-- soft_20 worst-20: mean gold len=14.8 (corpus mean 61.9, median 50.0); mean delta=0.329
   ids: 1630-73710-0004 1686-142278-0068 4323-55228-0038 2506-169427-0006 700-122868-0033 2506-13150-0003 1255-90413-0018 1651-136854-0009 6467-97061-0020 8288-274150-0005 4515-11057-0070 1686-142278-0008 8288-274162-0032 1630-73710-0011 5543-27761-0019 6467-94831-0021 1701-141759-0012 8254-84205-0026 1686-142278-0010 6467-94831-0042
   gold lens: 5 2 13 12 5 6 6 33 34 14 21 7 7 18 15 23 27 16 16 17
   hyp/ref len ratio on those: soft_20=1.333 ctrl_20=1.108
   per-utt PER on those: soft_20=1.397 ctrl_20=1.067
   [1630-73710-0004] d=+0.600 len=5
      GOLD : M AY N OW N
      CTRL : HH R N T AA N OY T
      soft_: B AH L Z TH R ER Z AY
   [1686-142278-0068] d=+0.500 len=2
      GOLD : N OW
      CTRL : HH R EH N OY
      soft_: AA DH ER D AY
   [4323-55228-0038] d=+0.462 len=13
      GOLD : F AY V D EY Z IH N D IY D S ER
      CTRL : K AE N R IH N IY S T R IH N D M AE N
      soft_: S T AH L Z M IH NG AH P F TH IH NG AH L M AE N AH D
   [2506-169427-0006] d=+0.417 len=12
      GOLD : AH L AE S P UW R Y AO R IH K
      CTRL : HH S T AE L Z TH R IH EY R EY AH D Z
      soft_: AA IY L K L IY L IY AA IH S T L NG AH L IY M
   [700-122868-0033] d=+0.400 len=5
      GOLD : AO L R AY T
      CTRL : HH CH AO N D
      soft_: HH AH L B AH L EY Z
   [2506-13150-0003] d=+0.333 len=6
      GOLD : S IH K S IH K
      CTRL : M AE L D TH M UW AE L R D TH
      soft_: AE N IY M L AE OW AE N IY M L M L
-- sf_20 worst-20: mean gold len=13.4 (corpus mean 61.9, median 50.0); mean delta=0.435
   ids: 1255-138279-0008 1686-142278-0041 1630-73710-0004 4153-186222-0001 1686-142278-0068 1255-90413-0018 8288-274162-0048 4323-55228-0045 1701-141759-0020 2506-169427-0006 700-122868-0033 6467-94831-0037 8288-274150-0005 5543-27761-0106 8254-84205-0014 5849-50963-0004 5543-27761-0049 4153-186222-0010 4831-29134-0005 1650-167613-0043
   gold lens: 5 3 5 6 2 6 11 9 7 12 5 26 14 14 20 18 19 32 20 35
   hyp/ref len ratio on those: sf_20=1.457 ctrl_20=1.201
   per-utt PER on those: sf_20=1.536 ctrl_20=1.101
   [1255-138279-0008] d=+0.800 len=5
      GOLD : T UW TH R IY
      CTRL : HH R IH N OY R HH AE N D
      sf_20: HH W R OW N Z SH M W M AH N Z
   [1686-142278-0041] d=+0.667 len=3
      GOLD : Y EH S
      CTRL : HH R L Z
      sf_20: HH W AH N TH AA
   [1630-73710-0004] d=+0.600 len=5
      GOLD : M AY N OW N
      CTRL : HH R N T AA N OY T
      sf_20: HH W IH N D DH AH V D Z
   [4153-186222-0001] d=+0.500 len=6
      GOLD : AA W AY N AA T
      CTRL : HH AW AY CH HH W AY AH N R
      sf_20: HH IH Z SH HH Y UW NG AE N D
   [1686-142278-0068] d=+0.500 len=2
      GOLD : N OW
      CTRL : HH R EH N OY
      sf_20: HH W AH N Z SH
   [1255-90413-0018] d=+0.500 len=6
      GOLD : JH AO S L IH N
      CTRL : HH AE Z S AH S T IH AH Z AH T
      sf_20: HH W AH P AA S EY D R AH L P R IY Z

```

What the worst 20 have in common, read directly: they are **short** (mean gold length 14.8 for
soft_20 and 13.4 for sf_20 against a corpus mean of 61.9 and median 50), and on exactly those
utterances the scorer arm **over-generates**: hyp/ref length ratio 1.333 (soft_20) and 1.457 (sf_20)
against 1.108 / 1.201 for ctrl_20 on the same utterances, with per-utt PER above 1.0 (1.397 and
1.536) because insertions alone exceed the tiny reference. The mechanism visible in the examples is
that the decoder emits a fixed-ish minimum number of phones (shortest output 3-5 phones across arms) regardless of
how short the utterance is: `N OW` (2 phones) draws 4-6 phones from every arm. This is a
short-utterance floor effect shared by all arms, amplified in the arms that are slightly longer.

## 6. Cross-arm structure

```
pairwise Pearson r of per-utterance PER between arms:
             sf_20       soft_20     soft_20_s1  softshuf_20 ctrl_20     ctrl_20_s1 
sf_20        1.000       0.846       0.875       0.870       0.862       0.858      
soft_20      0.846       1.000       0.829       0.842       0.849       0.861      
soft_20_s1   0.875       0.829       1.000       0.863       0.839       0.849      
softshuf_20  0.870       0.842       0.863       1.000       0.848       0.862      
ctrl_20      0.862       0.849       0.839       0.848       1.000       0.853      
ctrl_20_s1   0.858       0.861       0.849       0.862       0.853       1.000      

Jaccard/token overlap of hypothesis 4-gram types between arms (how similar are the codes?):
             sf_20       soft_20     soft_20_s1  softshuf_20 ctrl_20     ctrl_20_s1 
sf_20        1.000       0.072       0.072       0.065       0.081       0.075      
soft_20      0.072       1.000       0.064       0.065       0.072       0.083      
soft_20_s1   0.072       0.064       1.000       0.068       0.067       0.074      
softshuf_20  0.065       0.065       0.068       1.000       0.077       0.097      
ctrl_20      0.081       0.072       0.067       0.077       1.000       0.091      
ctrl_20_s1   0.075       0.083       0.074       0.097       0.091       1.000      
```

Per-utterance PER is highly correlated between *all* pairs of arms (r 0.83-0.88), with the
same-condition pairs (ctrl_20 vs ctrl_20_s1 0.853, soft_20 vs soft_20_s1 0.829) no higher than the
cross-condition pairs -- the arms find the same utterances hard, and that agreement is entirely
about the data, not the condition. Conversely the *codes* are almost disjoint: 4-gram type Jaccard
is 0.064-0.097 between every pair, and again same-condition pairs (ctrl/ctrl 0.091, soft/soft 0.064)
are not more similar than cross-condition pairs (softshuf/ctrl_20_s1 0.097 is the highest of all).
Each run invents its own vocabulary of nonsense words, independent of the scorer.

## 7. Error-type mix

```
arm               S%      D%      I%      PER
sf_20          74.06   10.04    5.85   0.8995
soft_20        72.50   10.71    4.69   0.8789
soft_20_s1     72.39   10.60    6.03   0.8901
softshuf_20    71.78   10.90    5.41   0.8810
ctrl_20        70.96   10.43    5.50   0.8689
ctrl_20_s1     72.19   10.43    5.41   0.8802

```

The sub/del/ins composition is nearly identical across arms: deletions 10.0-10.9% of N,
insertions 4.7-6.0%, everything else substitutions. The control seed spread (S 70.96 vs 72.19,
I 5.50 vs 5.41) covers every scorer arm except sf_20's substitution rate (74.06), which is the one
value outside the control band -- and sf_20 is also the worst arm overall, so this is a restatement
of its higher PER, not a distinct pattern.

## 8. Answer

1. **The decodes are fluent English-shaped phone nonsense, in every arm including both controls.**
   Right length (94-96% of reference), right inventory, word-like 4-grams, no degenerate loops, and a
   strong utterance-initial HH artefact (77-91% in four of six arms). The strings carry no visible
   relation to the reference (see section 2).
2. **It is not a consistent relabelling.** Weighted modal share 0.193-0.227 against a chance baseline
   of 0.08-0.12; NMI(gold, hyp) 0.066-0.093 of an available H(gold) = 3.34 nats. The frequent phones
   map weakly to themselves; 19-23 of 39 gold phones have a modal hypothesis other than themselves.
   The code is a heavily blurred identity map, not a permutation; a decipherment oracle would not
   recover much.
3. **No error-pattern statistic separates the scorer arms from the controls beyond the seed spread.**
   On every statistic computed -- length ratio, 4-gram diversity, 4-gram-in-gold share, unigram JS,
   NMI, modal share, sub/del/ins mix, paired per-utterance delta -- the ctrl_20 vs ctrl_20_s1 (and
   soft_20 vs soft_20_s1) spread is as large as or larger than the scorer-vs-control gap, and in
   several cases the two controls sit at the *two ends* of the whole six-arm range (4-gram diversity
   2749 vs 3148; unigram N production +2.38pp vs -1.45pp; initial-HH 90.7% vs 81.1%). The single
   exception is sf_20, which is consistently the worst arm (PER 0.899, substitution rate 74.06%,
   worse on 65.5% of utterances, highest unigram JS 0.030, largest R flood +5.0pp) -- the
   score-function scorer makes the decode measurably worse, outside the control band, while the
   straight-through soft scorer changes nothing that the seed does not change more.
4. **The destroyed-structure null behaves like the real scorer.** softshuf_20 (PER 0.881, NMI 0.0706,
   +0.0202 paired) sits between soft_20 and soft_20_s1 on essentially every statistic here, which is
   what "the scorer's phone identities are not being used" looks like.

## 9. Caveats

* The alignment is mine, not the job's; the job stores no confusion matrix. My S/D/I reproduce the
  banked counts exactly for all six arms, but the *pair-level* attribution (which gold phone aligns
  to which hypothesis phone) depends on tie-breaking among equal-cost paths and is therefore not
  unique. The NMI and modal-share numbers should be read as one consistent convention applied
  identically to six arms, not as absolute quantities. Section 4's conclusions rest on the ranking,
  which is stable in the sense that the same convention was used everywhere.
* "Beyond the seed spread" here means "a one-replicate range", n = 2 per condition. That is the
  spread the pack registered, and it is what I used; it is a weak null and I did not construct a
  bootstrap or permutation null over utterances.
* The per-utterance PER used for the selection in section 2 and the deltas in section 5 uses
  len(gold) as denominator per utterance, so utterances of 2-13 phones dominate the |delta| tail.
  The corpus-level PER is the pooled (S+D+I)/N, which is what the banked per.txt reports; the
  macro-mean of per-utterance PER differs (ctrl_20 0.8913 vs pooled 0.8689) and is not used for any
  claim.
* The training-side identity of the arms (same streams, prior, quantizer, schedule, seeds) is taken
  from `config_sae_4a_soft_pack_v1.py`'s docstring, not re-verified from the training job hashes.
