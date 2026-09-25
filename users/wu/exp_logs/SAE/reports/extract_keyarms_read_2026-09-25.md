# extract_keyarms_read_2026-09-25

Job dirs:
- work/speech_llm/sae/emc/keyarms_read_jobs/KeyArmsReadJob.STcxhF0w4kpq/output/report.txt (log.run.1 NOT FOUND in that dir)
- work/speech_llm/sae/emc/blankfree_a13_disjoint/PhiFirstA10DiagnosticsDisjointJob.nyCwi94aepC0/output/report.txt

## 1. Rule text and verdict (verbatim, report.txt)
"Key arms: the stage-1 top 4, as one four-GPU pack.
    **KEY BASIN** if the best key arm by S at 48 (260 set, paired as in A14 (ii)) has S < 3.289; else
    **NO KEY BASIN**.
    Reported, never gating: direct and Hungarian PER, NMI, and A15-F's measures at 0-48.
  KEY BASIN sends the selected phi to the lift test (A14 (i)'s form) and to L2-2."

sub-epoch 48, 260 set, paired over 260 of 260 (excluded []):
  arms: rank1 3.27275, rank2 3.29550, rank3 3.28529, rank4 3.33540
  best arm rank1: S 3.27275 against the bar 3.289
  paired S_min (report only) 3.29903 [durinit_s01]; registered 3.29903 [durinit_s01, 260 tags]
VERDICT: KEY BASIN

## 2. Per-arm S at ep0/4/12/48 (260 set; 285 SECONDARY in parens), PER direct/Hungarian, NMI(symbol,phone) — from KeyArmsReadJob report.txt
- rank1: ep0 S 4.39305 (4.39871), PER dir 0.9155, Hung 0.8953, NMI 0.0757 | ep4 S 3.41928 (3.42430), PER dir 0.8619, Hung 0.8491, NMI 0.0762 | ep12 S 3.32537 (3.33010), PER dir 0.8585, Hung 0.8362, NMI 0.0792 | ep48 S 3.27275 (3.27785), PER dir 0.8582, Hung 0.8613, NMI 0.0774
- rank2: ep0 S 4.42365 (4.42910), PER dir 0.9250, Hung 0.8108, NMI 0.1121 | ep4 S 3.45813 (3.46073), PER dir 0.8454, Hung 0.7923, NMI 0.1065 | ep12 S 3.35708 (3.35904), PER dir 0.8451, Hung 0.7978, NMI 0.1016 | ep48 S 3.29550 (3.29783), PER dir 0.8469, Hung 0.7975, NMI 0.0986
- rank3: ep0 S 4.41255 (4.41740), PER dir 0.9358, Hung 0.9335, NMI 0.0825 | ep4 S 3.43881 (3.44367), PER dir 0.8563, Hung 0.8427, NMI 0.0914 | ep12 S 3.33122 (3.33632), PER dir 0.8529, Hung 0.8399, NMI 0.0925 | ep48 S 3.28529 (3.29080), PER dir 0.8490, Hung 0.8416, NMI 0.0962
- rank4: ep0 S 4.40882 (4.41441), PER dir 0.9499, Hung 0.9654, NMI 0.0661 | ep4 S 3.49792 (3.50213), PER dir 0.8777, Hung 0.8805, NMI 0.0688 | ep12 S 3.39762 (3.40157), PER dir 0.8749, Hung 0.8763, NMI 0.0651 | ep48 S 3.33540 (3.34045), PER dir 0.8715, Hung 0.8734, NMI 0.0677

MISSING: a "gold-key control" arm — no such arm appears in either report.txt (searched: KeyArmsReadJob report.txt with grep "gold", A10 disjoint report.txt). Only reference points printed: reference S (CV holdout) 260 PRIMARY: gold 3.47351, r100 4.69008 | 285 SECONDARY: gold 3.47024, r100 4.68612 (gold and r100 are reference nulls, not a fifth key arm).
MISSING: paired differences with CIs against the reference — KeyArmsReadJob report.txt prints only the paired S_min (report only) value, no CI figures; no CI numbers found in report.txt.
MISSING: SIL share per key arm at 0/4/12/48 in the KeyArmsReadJob report — not printed there (only rate Hz and PER/NMI); SIL share (E[d] SIL) IS printed per sub-epoch in the A10 disjoint report (job PhiFirstA10DiagnosticsDisjointJob.nyCwi94aepC0), e.g. rank1 ep0 SIL 26.000, ep4 SIL 18.241, ep12 SIL 10.438, ep48 SIL 6.862 (and similarly for rank2/3/4, see full report.txt).
MISSING: log.run.1 for KeyArmsReadJob.STcxhF0w4kpq (searched: work/speech_llm/sae/emc/keyarms_read_jobs/KeyArmsReadJob.STcxhF0w4kpq/ — no log.run.1 present under that dir).

## 3. S-best arm at 48
rank1, S = 3.27275 (260 set) — goes to L2-2 under A18 (c).

## 4. Warnings/exclusions/substitutions
"paired over 260 of 260 (excluded [])" — no exclusions. No warning text found (log.run.1 missing to search further). A10 disjoint report notes "0 impossible" utterances for each sub-epoch/arm dev-other read.
