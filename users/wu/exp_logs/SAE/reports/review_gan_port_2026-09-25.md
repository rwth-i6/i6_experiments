# Review: GAN reproduction port (1c wav2vec-U 2.0 + 1d CTC self-training), 2026-09-25

Verdict: APPROVE_WITH_FIXES. The change is a pure addition. Configs, seeds, selection, the PER, pseudo-label
and WER conventions, and the decoder parameters match production. The fix to make before i6 launches is the
GAN host-memory request (F1). The other items are cheap to correct, or they fail loudly before any GPU time
is spent.

Worktree: /e/project1/spell/wu24/worktrees/i6_experiments_cycle_consistency (branch
haotian_cycle_consistency_unsupervised), package users/wu/experiments/unsupervised_asr/ (PKG below).
Inputs: reports/impl_gan_port_{A,B,C,wire}_2026-09-25.md.

## Findings

### F1 (fix before launch): GAN host memory. The 60 GB request is backed by no evidence.
Location: PKG/training/w2vu2_gan.py:245 (`GAN_RQMT` mem 60). The docstring (PKG/config/w2vu2.py:54-62)
states no RAM need.

**The JUPITER run enforced no 60 GB limit.** Job 968302 (GAN s0) shows `mem=0` in sacct AllocTRES.
- JUPITER runs SelectTypeParameters=CR_CORE (memory is not consumable) with --exclusive, so the cgroup
  limit was 95% of the node.
- "It succeeded with mem 60" is therefore no evidence that 60 GB is enough.

**What the 307.67 GiB peak (sisyphus RSS; sacct MaxRSS 297.6 GiB) is.** It is a per-process sum.
- sacct MaxVMSize 3.59 TiB equals the per-process VMS sum. The CTC job 976159 shows the same pattern
  (41.0 GiB vs 49.8 GiB sisyphus).
- The sum counts shared file pages once per process.

**Evidence that the per-process memory is the mmap'd inputs.**
- fairseq maps its inputs: `np.load(mmap_mode="r")` and MMapIndexedDataset.
- Its DataLoader workers are persistent (fairseq/data/iterators.py:221).
- The 6 train workers each grow to the size of the mapped files:
  - train.npy 31.60 GB + text train.bin 6.54 GB + train.idx 0.48 GB = 38.6 GB;
  - at exit, RSS drops in steps of 38 GiB, one per worker;
  - VMS drops by about 265 GB per worker.
- The 6 valid workers each map valid.npy (3.3 GB).
- Measured on the login node (scripts/mmap_share.py): two forked readers of the same GPFS
  np.load(mmap_mode="r") region each showed 2.0 GiB RSS, all of it Shared_Clean with Pss 1.0. The memcg
  file charge grew by 2.0 GiB, not 4. So the pages are shared, charged once, and reclaimable.

**The real need.**
- The file working set is about 42 GB (train+valid npy and text).
- Anonymous memory is not measured exactly. Upper bound: the main process ends at about 45 GiB (83.55 GiB
  minus one 38.6 GiB worker). Workers start at about 2 GiB each (the jump at fork).
- The total, about 42 GB plus up to about 45-57 GB, exceeds 60.

**What breaks at mem 60.** This applies under a cgroup-enforced 60 GB, which the i6 settings may or may
not impose.
- The randomly sampled 31.6 GB train.npy cannot stay cached, so it is re-read from the network FS on
  every epoch: about 840 epochs per seed, times 5 seeds.
- If anonymous memory exceeds 60 GB, the job is OOM-killed.

**Fix.** Set GAN mem to about 100. rqmt is not hashed, so this costs nothing. Alternatively, run one seed
and read `sstat`/memory.stat before fanning out to 5 seeds. State the requirement in the docstring.

### F2: the CUDA_ARCH=89 instruction makes the env build abort
Location: PKG/config/w2vu2.py:68 tells the builder to set CUDA_ARCH=89 for L40S.
- The gate at PKG/env/build_w2vu_env.sh:158-160 asserts `sm_$CUDA_ARCH in torch.cuda.get_arch_list()`.
- Official torch wheels do not list sm_89 (sm_86 SASS runs on Ada). See the PyTorch forum thread
  "sm_89 not listed in torch.cuda.get_arch_list()" and pytorch issue #95648.
- One wrapper serves both trainings and decodes, so only one arch can be asserted anyway.
- Failure mode: loud, at build time, no GPU time lost.
- Fix: document CUDA_ARCH=70 (or 86), or GATE_CUDA=0.

### F3: gpu_mem routing on i6 is decided by i6 settings, not by the docstring
Location: PKG/config/w2vu2.py:57-60, 138, 140.
- The i6 settings in this repo map `gpu_mem > 24` to `-p gpu_48gb`:
  - users/vieting/settings/settings_v1.py:43-44;
  - users/azim_javed/memristor/model_pipelines/settings.py:34-35.
- Under such settings, I6_TRAIN_GPU_MEM=32 sends the trainings to the 48 GB (L40S) partition, not V100,
  and so do the 40 GB forwards and decodes.
- Sisyphus itself ignores gpu_mem, and JUPITER's settings.py never read it, so production's 40 and 80 were
  inert labels.
- Nothing crashes, and the actual need is small on any of these GPUs:
  - GAN about 6.4 GB and CTC about 11 GB per GPU (production gb_free);
  - the RETURNN forward is bounded by max_seqs 200 (PKG/analysis/w2vu2_gan_eval.py:288-289);
  - the decodes use max_tokens 1.1M.
- The docstring's "trainings on V100" should say that the partition comes from the i6 settings.

### F4: silent CPU training is possible if CUDA is hidden
Location: PKG/env/build_w2vu_env.sh:139 replaces LD_LIBRARY_PATH.
- The gate prints `torch.cuda.is_available()` but does not assert it (line 173).
- fairseq 0.12.2 sets `self.cuda = torch.cuda.is_available() and not cfg.common.cpu`
  (fairseq/trainer.py:61). A 1-GPU GAN would therefore train on CPU without an error if libcuda.so were
  reachable only through LD_LIBRARY_PATH on i6 nodes.
- Production made the same replacement and worked on JUPITER, so this is a portability check, not a
  defect.
- Check: the first GAN log must contain fairseq's "CUDA enviroments for all 1 workers" banner. The banked
  s0 log has it.

### Informational (no failure path)
- The artefact tests skip unless SAE_ARTEFACT_DIR is set and the banked paths exist, so on i6 they pass
  vacuously. This is documented.
- The GAN resolved-config test roots fairseq at site-packages symlinks, not at the sparse CloneGitRepositoryJob.
  A's report states that the tag's fairseq_cli, fairseq/config and examples/wav2vec/unsupervised equal the
  wheel's. I checked that the tag's (4a388e64) examples/speech_recognition/new/conf/infer.yaml equals the
  wheel's byte for byte.
- In the decodes, the wrapper's shim comes first on PYTHONPATH (build_w2vu_env.sh:140). It shadows the
  job's own shim (PKG/analysis/w2vu2_ctc_decode.py:216), so `examples` resolves to the wheel's copy. This is
  harmless because the conf and the module are equal.
- Disk, identical to production (keep_interval_updates -1): 4.4 GB of checkpoints per GAN seed, and 60 GB
  for the CTC student.
- CTC data: `labels.get(uid, "")` (PKG/training/w2vu2_ctc.py:169) drops unmatched ids silently. Only a total
  mismatch trips `assert kept`. Id forms were checked consistent (B test c and C decode test), so there is
  no failure path now.

## Checks with evidence

**1. Pure addition and allowlist.** `git status --porcelain` shows only 14 untracked new files. No tracked
file is modified and no stray file is present.
- Third-party imports outside stdlib and the allowlist are fairseq, omegaconf and datasets, and only in
  tests.
- No /e/project1 or HF-cache path appears in package code.
- A name-resolution script found 0 unresolved imports.
- test_graph_build_imports_only_the_allowlist passes.

**2. Faithfulness to production.**
- GAN hydra config: 313/313 keys, and the diff is exactly {checkpoint.save_dir, common.user_dir}. This is
  test_resolved_config_equals_banked_s0_except_paths, which I ran; it passed.
- CTC config: 358/358 settings. 7 differ: 4 paths and 3 runtime DDP fields. This is
  test_resolved_config_equals_production, which I ran; it passed.
- Seeds 0-4 are asserted per job (test_production_seeds).
- Selection is argmin weighted_lm_ppl, ties broken by seed. It reproduces s0 at 15.85, update 148000
  (test_select_reproduces_banked_seed0, which I ran; it passed).
- PER: summed errors over summed gold phones (B test b: 0.173 / 0.214 to three digits; dev-other 37952 vs
  37955 errors, documented). Pseudo-labels: space-joined, collapsed, SIL-free strings (B test c, 200/200).
  I did not rerun these slow tests.
- Decoder parameters are asserted in test_selftrain_wiring: beam 500, lmweight 2.0, wordscore -1.0,
  max_tokens 1.1M.
- WER: upper().split() Levenshtein (test_word_collect...). C reports decode equality of 20/20 phone and
  20/20 word hypotheses (slow; not rerun).

**3. Job idiom.**
- The only subprocess is fairseq's own CLI (`-m examples.speech_recognition.new.infer`,
  PKG/analysis/w2vu2_ctc_decode.py:265). Both trainings are i6_core FairseqHydraTrainingJob.
- The converter uses torch.load(weights_only=True) and imports no fairseq.
- The RETURNN forward loads the converted checkpoint. Test (d) is a fast test and passed in my run. It
  uses a model with non-default BN statistics and conv weights scaled by 20, and every hypothesis must equal
  the direct decode, so a random init would fail it. B's test (b) reproduces the banked PER through
  ReturnnForwardJobV2.

**4. Launch feasibility.**
- **Memory:** see F1.
- **Resume:**
  - The run task is `Task("run", resume="run")`. Upstream i6_core main is identical in signature,
    rqmt, tasks, run and hash.
  - restore_file checkpoint_last.pt resolves against the CLI save_dir.
  - reset_* are all False, so the composite optimizer groups, lr scheduler, num_updates, the best value
    and the iterator are restored.
  - The GAN saves every 1000 updates and the CTC student every 2500. A resumed run is not
    bit-reproducible.
- **gpu_mem 40:** see F3.
- **Env propagation:** FairseqHydraTrainingJob executes fairseq_python_exe, which is the wrapper, directly
  with os.environ. The wrapper exports TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, PYTHONNOUSERSITE, the shim and
  LD_LIBRARY_PATH, and DDP children inherit them. The decode jobs call the same wrapper and also set the
  flag themselves.
- **CTC resources:** i6_core multiplies cpu and mem by the 4 GPUs, giving 16 cpu and 60 GB, as in
  production. The job needs one 4-GPU node. The production /proc-sum peak was 49.8 GiB, under 60.
- **torchaudio:** the MFCC jobs need torchaudio in the main env. env/environment.yml does not list it, and
  its comment at line 29 says it is unused. This is documented at PKG/config/w2vu2.py:77-79. It fails
  loudly on a CPU job if absent.

**5. Test runs (login node CPU, speech_llm env).**
- Fast tests (`-m "not slow"`): 33 passed, 7 skipped, 7 deselected.
- With SAE_ARTEFACT_DIR=1 (`-m "artefact and not slow"`): 7 passed, 0 skipped.
- C's slow resolved-config test: passed.
- Not rerun: B's (a)/(b)/(c), A's converter and text byte-equality, C's decode equivalence. These results
  are taken from the implementer reports.

Scratch (/e/project1/spell/wu24/worktrees/port_checks_gan/review/) deleted after the checks.
