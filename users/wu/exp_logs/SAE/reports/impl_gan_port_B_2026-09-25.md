# Implementer B report: the GAN generator forward in RETURNN (2026-09-25)

Status: DONE_WITH_CONCERNS

## Outcome

The §1c wav2vec-U 2.0 generator now converts from its fairseq checkpoint to a RETURNN checkpoint and
decodes through `ReturnnForwardJobV2`. It then feeds an in-process PER job and an in-process
pseudo-label job. The port reads the checkpoint without fairseq.

On CPU through RETURNN, the forward matches the banked results as follows:

- dev-clean reproduces the banked PER exactly: 33487 errors out of 193644 reference phones.
- dev-other gives 37952 errors against 37955 banked. The PER is 0.214085 against 0.214102, the same to
  three digits.
- The pseudo-labels on 200 train utterances equal the banked strings (200 of 200).

There are two concerns:

1. In float32 the logit agreement is 6.2e-5, not the 1e-5 the dispatch asked for. In float64 the two
   sides agree to 1.4e-14.
2. dev-other misses the banked error count by 3 errors.

Both are explained below.

Worktree: `/e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency` (branch `haotian_cycle_consistency_unsupervised`). Nothing is committed.

## Files (all new; package `users/wu/experiments/unsupervised_asr/`)

- **`model/w2vu2_generator.py`** holds the RETURNN-side code. Nothing is imported at module time.
  - `W2VU2_NET_ARGS` is the s0 row: in_dim 1024, n_out 44, kernel 9, stride 3, 1 layer, dropout 0.1, BN, residual, no bias.
  - `generator_logits(model, feats, lens)` returns the raw logits.
  - `greedy_decode(logits, vocab, sil_symbol="<SIL>")`.
  - `w2vu2_forward_step`.
  - `W2vu2GreedyDecodeCallback(vocab_file, expected_num_seqs)` writes `hyps.json` and `hyps.stats.txt`.
- **`analysis/w2vu2_gan_eval.py`** holds the converter, the forward builders, the PER job and the pseudo-label job.
- **`tests/test_w2vu2_gan_eval.py`** has 8 fast tests and 3 artefact tests (marked `artefact` and `slow`).

`git status --porcelain` shows only `??` lines. The other `??` files belong to A and C.

## Design

- **Model.** The generator is the port's existing `ConvRecognizer`, built with `recognizer_only.get_model(**W2VU2_NET_ARGS)`. It already mirrors fairseq's `Generator` (wav2vec_u.py:288-367) layer for layer. No layer is reimplemented.
  - The source's eval used `no_softmax=True` and `segmentation NONE`.
  - `ConvRecognizer.forward` returns a log-softmax, so the raw logits are read with a forward hook on `model.conv`.
- **Decode.** This is the production convention from eval_per.py: argmax, `groupby` collapse, drop `<SIL>`, then map through the dictionary. The fairseq specials are kept, as in the source.
- **Checkpoint conversion.** The converter unpickles with `torch.load(weights_only=True)`.
  - This is torch's restricted unpickler. It rebuilds only tensors, storages, OrderedDict and plain containers, and raises on any other global.
  - A fairseq 0.12 checkpoint stores its cfg as a plain container, so no fairseq or omegaconf class is imported.
  - It keeps the `generator.*` keys and renames `proj.1.` to `conv.`, then does a strict `load_state_dict`.
  - It checks the checkpoint's `cfg["model"]` against the net args.
  - It saves `{"model", "epoch", "step"}`. For s0 that is epoch 823 and step 148000, with 1459201 parameters.
- **Gold.** Gold comes from the port's existing `data.gold.GoldPhonesJob` on the pinned MFA snapshot. Its output equals the banked `GoldPhonesJob.ZGSp0hxyd2YP` json exactly; test (b) asserts this.

## Interface

```
w2vu2_generator_checkpoint(*, fairseq_checkpoint: tk.Path, text_dict: tk.Path, alias=None, net_args=None)
    -> (W2vu2GeneratorCheckpointJob, PtCheckpoint)      # job.out_checkpoint, job.out_vocab (vocab.json)
build_w2vu2_forward_config(*, feature_hdfs, vocab, net_args=None, expected_num_seqs=0,
                           batch_size_frames=None, max_seqs=None) -> ReturnnConfig
w2vu2_forward_job(*, name, checkpoint, vocab, feature_hdfs, expected_num_seqs=0, returnn_exe=None,
                  returnn_root=None, net_args=None, device="gpu", time_rqmt=2.0, mem_rqmt=24,
                  cpu_rqmt=4, gpu_mem=24) -> ReturnnForwardJobV2   # out_files["hyps.json"], ["hyps.stats.txt"]
W2vu2GanPerJob(*, hyps: tk.Path, gold: tk.Path, split: str)         # out_per (per.json), out_report
W2vu2GanPseudoLabelJob(*, hyps: Sequence[tk.Path], expected_num_seqs=None)   # out_labels (labels.json)
```

- The batching defaults come from the posterior dump: 1.6M frames and 200 sequences per batch.
- The alias is `sae/1c/gan/{name}/forward`.

### What B needs from A

1. The fairseq `checkpoint_best.pt` of the GAN run. The banked s0 is `FairseqW2vu2TrainJob.HOb2GgtYT7Bc/output/train/checkpoint_best.pt`.
2. The text side's `dict.txt` (`FairseqPreprocessTextJob.../output/text_data/dict.txt`). The checkpoint does not store the vocabulary, and dict.txt fixes the 44 output symbols.
3. The VAD-trimmed L15 feature HDFs for each split: `BlankfreeVadHdfJob` `feats.{split}.shard{k}.hdf`, float16, 1024-d, tags equal to utterance ids.
   - All tests used the production job `BlankfreeVadHdfJob.SAjz8y1cT06g` under the speech_llm work dir.
   - A's port-built equivalent should be checked for identity against it.

### What C consumes

`W2vu2GanPseudoLabelJob(hyps=[fwd.out_files["hyps.json"] for each train shard], expected_num_seqs=28539).out_labels`.

- The format is the source's own: `{"labels": {id: "p1 p2 ..."}, "utts": n, "empty": k}`. The strings are collapsed and SIL-free.
- C reads it with `data.gold.PhoneTargetHdfJob(labels_json=..., label_key="labels")`. Test (c) and a fast test run this path.

## Checks (CPU, login node, speech_llm env, PYTHONDONTWRITEBYTECODE)

- **(a) Equivalence with fairseq.** 20 dev-other utterances (every 143rd of the port HDF), 2171 output frames. The fairseq model forward ran in the w2vu python on CPU in eval mode, with the source's overrides, as a subprocess.
  - The vocabulary equals fairseq's `target_dictionary`.
  - In float64 on both sides, the max |logit diff| is 1.421e-14.
  - In float32, the max |diff| is 6.199e-05. For comparison, fairseq's own float32 output deviates from its own float64 output by 6.368e-05. The largest logit magnitude is 15.42.
  - The argmax and the decoded strings are identical on every frame.
  - The float32 1e-5 criterion is therefore not met. The cause is float32 accumulation-order rounding in the 1024-d in_proj and the 9x1024 conv, which differs between torch 2.6 (w2vu) and 2.7 (port). The implementation itself matches (1.4e-14 in float64).
  - The test asserts the float64 1e-5 bound and float32 argmax identity. It prints the float32 deviation without asserting a bound on it.
- **(b) PER.** Full chain through RETURNN (`ReturnnForwardJobV2.create_files` and `run` in-process, `rnn.py` on CPU, 16 threads, 200k-frame batches), then `W2vu2GanPerJob`.
  - dev-clean: PER 0.172931, 33487 errors, 193644 reference phones, 2703 utterances. This equals the banked record exactly.
  - dev-other: PER 0.214085, 37952 errors, 177275 reference phones, 2864 utterances. The banked record is 0.214102 with 37955 errors.
  - The test asserts three-digit equality plus equal utterance and reference counts.
  - A scratch diagnostic (deleted) ran the same generator in-process, one utterance at a time, on the GAN's own `MergeW2vu2DataJob` valid.npy features. It gave the identical counts: 33487 and 37952. So the 3-error gap is not caused by the feature files. It is consistent with near-tie argmax flips from the source's GPU run (batch 1, possibly TF32). No GPU run was allowed here, so this is not verified.
- **(c) Pseudo-labels.** The first 200 utterances of train shard0 went through RETURNN on CPU and `W2vu2GanPseudoLabelJob`. All 200 of 200 strings equal the banked `GanPseudoLabelJob.xjn6QnNqwEEH` strings, and `PhoneTargetHdfJob` loads them.
- **(d) Forward job.** The job from `w2vu2_forward_job(device="cpu")` ran its own `create_files` and `run` on 57 tiny utterances. The config serialized, loaded in RETURNN and ran, and every hypothesis equals the direct decode.
  - The production-shaped default config (GPU, 1024-d) also serializes, and its python part executes. `get_model` there builds the 1024-to-44 stride-3 model.
- **(e) Full suite.** 478 passed, 1 failed, 17 skipped, 7 xfailed.
  - The failure, `test_model_reverse.py::test_segment_scores_explicit_sum[3]`, was already failing before this change (tolerance 1.46e-5 against 1e-5). It is unrelated to this work.
  - The baseline also failed `test_t2_6`, only because `black` was not on PATH. With the env's bin dir on PATH it passes.
  - The suite count includes C's concurrently added test file.

## Undetermined / notes

- Whether dev-other reproduces 37955 exactly needs a GPU forward, which this dispatch did not permit.
- The float32 1e-5 criterion is unattainable on CPU across the two torch builds (fairseq disagrees with itself by 6.4e-5). Whether the float64 reading is acceptable is the caller's decision.
- The GPU batching defaults (1.6M frames) come from the posterior dump. Only CPU with 200k-frame batches was run.
- The scratch dir `worktrees/port_checks_gan/B/` is deleted. The shared `__pycache__` dirs in the worktree are gitignored. I removed my modules' `.pyc` files; other sessions' runs write their own.

## Follow-up: the 3-error dev-other gap, checked per utterance (2026-09-25)

Verdict: **VERIFIED_NEAR_TIE**. The source ran on GPU and our forward runs on CPU. Each differing utterance
differs in exactly one output frame, and that frame is the one where the top two phones are closest.
fairseq's own CPU forward agrees with ours on these frames, not with the banked hypotheses.

### Which banked hypotheses were used

- `W2vu2PerEvalJob.ptwMk3TuPPYb` kept no per-utterance hypotheses. Its outputs are only `per.json`.
- `W2vu2PerEvalJob.hrAAsQPKWM3G` is a re-run of the same eval. It has the same checkpoint, the same
  `MergeW2vu2DataJob.wxBxCIaJpqS2` features and the same gold, and it adds `hyps_splits`. It reproduces
  the banked counts exactly: 33487 / 37955 errors.
- It saved `greedy_phones.{dev-clean,dev-other}.json`, and those are what I diffed against.

### Our generator on the banked npy features

These are the features the source's eval read, so this comparison isolates the forward from the features.
The runs were per utterance, on CPU, in float32.

- **Utterances that differ.** dev-clean has 1: 2803-154328-0018. dev-other has 3: 3660-6517-0012,
  5849-50962-0009 and 7697-105817-0009.
- **How each one differs.** In each utterance a single frame flips to fairseq's runner-up phone. That
  frame has the smallest top-2 margin in its utterance. The median margin within these utterances is
  4.4 to 5.1.
- **fairseq on CPU.** Run in the w2vu python in eval mode with the source's overrides, fairseq decodes
  these 4 utterances exactly as we do, in float32. Its float64 argmax also gives our decode, not the
  banked one.

fairseq's top-2 margin on each flipped frame:

| utterance | frame | ours -> banked | margin float32 | margin float64 |
|---|---|---|---|---|
| 2803-154328-0018 | 225 | M -> Y | 4.244e-04 | 4.247e-04 |
| 3660-6517-0012 | 55 | T -> AE | 3.691e-04 | 3.670e-04 |
| 5849-50962-0009 | 136 | D -> `<SIL>` | 5.102e-05 | 6.413e-05 |
| 7697-105817-0009 | 130 | T -> S | 8.583e-06 | 2.449e-05 |

### Our generator on the port feature HDF (`BlankfreeVadHdfJob.SAjz8y1cT06g`)

This is the path used in test (b).

- **Utterances that differ.** dev-clean has 6, dev-other has 4. The rest of this section compares our
  HDF decode against fairseq's decode on the npy features.
- **The 8 utterances fairseq's CPU forward decodes as banked.** Each differs by exactly 1 frame. fairseq's
  top-2 margins on those frames are:
  - float32: 2.0e-4 to 3.6e-3;
  - float64: 1.95e-4 to 3.6e-3.

  These flips come from the small drift in the HDF feature values, not from the forward.
- **The other 2 utterances.** These are 2803-154328-0018 and 5849-50962-0009. On them our HDF decode
  matches fairseq's CPU decode on the npy features, not the banked one.
- **dev-clean totals.** The flips happen to cancel, so the total error count still matches
  (33487 / 33487).

**Consequence for A:** the port HDF features are near-equivalent to the GAN's own features but not
bit-identical. At the utterance level this flips about 0.1-0.2% of utterances by one phone.

The scratch directory `port_checks_gan/B2` is deleted.
