# Extract: Neural Phone LM and Falsifier (ii) Inputs

## 1. Neural Phone LM Instance (a): 4L / w256, 30 ep, 3.3 M params, ep 10 selected, ppl 5.091

**Training job class + hash:**
- First-pass training: `NeuralPhoneLmTrainJob.Iv6P6YVPNWmB`
- Rerun (a) training job: Module is `speech_llm/sae/emc/neural_phone_lm_v2.py` with class `NeuralPhoneLmTrainJobV2` (line 55)
- Scored in: `PriorGapAnalysisJob.5wNIQs2lpC5P` (SAE_4A_prior.md:414)

**Selected checkpoint file path:** 
- From first-pass job `Iv6P6YVPNWmB`: `model.pt` in the work directory
- From rerun at `PriorGapAnalysisJob.5wNIQs2lpC5P`: epoch 10 checkpoint

**Model config:**
- Layers: 4, Width: 256, Params: 3.3 M
- Vocab: 39 ARPAbet phones + SIL (40 symbols)
- Max positions: 512 (truncation applied in encode_lines)
- Causal transformer phone LM

**Scoring module/class file:line:**
- Module: `speech_llm/sae/emc/prior_gap.py`
- Class: `PriorGapAnalysisJob` (line 1336)
- Scoring happens in `run()` method (line 1487)
- Neural LM scoring at lines 1588+ (reads the neural_lm path and scores via kenlm wrapper or direct forward)

## 2. Unigram Baseline (Falsifier ii Amendment A1)

**Where fitted/stored file:line:**
- Module: `speech_llm/sae/emc/prior_gap.py`
- PriorGapAnalysisJob.run() lines 1562-1569
- Fitted on PhoneNgramPrior loaded at line 1548 from the arm's banked `prior.npz`
- Function: `witten_bell_utt_log_prob(prior, sets[s][t], order=1)` 

**"1 M-line window" text path:**
- Window is the banked uniform sample from `SampleLinesJob` (1,010,000 lines)
- Counted lines (1,000,000) selected by `window_line_flags()` function (line 1536)
- Phonemized text from `PhonemizeWithSilJob.DbFgvZOGZQ8F` (SAE_4A_prior.md:66)
- Word corpus: `librispeech-lm-norm.txt.gz`

## 3. Falsifier (ii): SampledRewardProbeJob

**Module path:** `speech_llm/sae/emc/blankfree_probe_jobs.py`

**Class location:** Line 319

**Functions NOT YET LOCATED:**
- (a) Loading banked blank-free checkpoint and rebuilding model: NOT FOUND
- (b) Blank-free lattice posterior / FFBS sampler: NOT FOUND
- (c) Neural scoring on GPU: NOT FOUND
- Whether scorer forward runs on GPU with gradients: NOT FOUND

## 4. Blank-Free Training Step and Model Definition

**Train step file:** `speech_llm/prefix_lm/model/train_steps/sae_blankfree.py`

**File:line of default-off lexlat_* block added in round 2:** NOT FOUND

**File:line where lattice term and rate term are formed:** NOT FOUND

## 5. Packed-Arm Job and Configs

**blankfree_pack_jobs module path:** NOT FOUND

**PackedBlankfreeTrainJob signature:** NOT FOUND (mentioned in SAE_4A_prepro.md as `PackedBlankfreeTrainJob.5EIGJJ1MkcO9`)

**Prepro pack config with N=20 schedule:**
- File: `speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/config_sae_4a_prepro_pack*.py`
- Schedule constants: N=20, sub-epochs, kept epochs 1/4/10/20, 601 s/sub-epoch, batch 88,000/max_seqs 128, tau schedule (8→2 over 4 sub-epochs)

**Frozen paths of ctrl_20 and ctrl_20_s1:**
- From SAE_4A_prepro.md: PackedBlankfreeTrainJob.5EIGJJ1MkcO9 outputs under `output/exp2025_11_06_speech_llms/librispeech/sae_4a_prepro/{arm}/ep{1,4,10,20}/dev-other/`
- Checkpoints ep1/4/10/20: NOT FOUND (exact paths)

## 6. Paired-Delta and Derangement-Gap Jobs

**Paired-delta job:** 
- `PairedPerDeltaJob` mentioned in multiple md files
- Module path: NOT FOUND

**Derangement-gap read job:**
- Used in gate checks across phases
- Module path: NOT FOUND

## 7. Greedy PER Scoring Job

**Class + module:** NOT FOUND

---

Status: PARTIAL - Core neural LM and scoring locations identified; many detailed paths and function locations remain unresolved due to time constraint.
