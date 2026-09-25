# GAN (w2vu2) port inventory — 2026-09-25

## 1. w2vu2/ package
Path: recipe/i6_experiments/users/wu/experiments/unsupervised_asr/w2vu2/
Live .py files (non-cache, excl. tests): __init__.py(14), packed_audio.py(41), rev_eval.py(81),
rev_diag.py(95), rev_data.py(108), silence_stats.py(115), repr_audit_w2v2.py(126),
packed_word_decode_worker.py(140), rev_dev.py(152), sparse_autoencoder.py(173),
text.py(191), features.py(222), eval.py(233), sparse_autoencoder_l1.py(257),
dump_w2vu2_data.py(261), pipeline.py(261), eval_per.py(324), sparse_autoencoder_probe.py(388),
gan.py(422), selftrain.py(547), word_decode.py(603), rev_diag_worker.py(821).
Plus userdir/w2vu_rev/ (fairseq user-dir plugin): __init__.py(22), rev_dataset.py(86),
rev_term.py(212), models/wav2vec_u_rev.py(180), tasks/unpaired_audio_text_rev.py(86).
Tests: test_word_decode.py(59), test_text.py(77), test_rev_diag.py(228).
build_w2vu_env.sh(110). Total incl. tests ~5900 lines of code (full dir incl. pycache 11695 wc -l, cache excluded above).

Foreign imports (not i6_core/i6_experiments.common/returnn/sisyphus/stdlib/3rd-party):
- repr_audit_w2v2.py:27 `i6_experiments.users.wu.experiments.ssl.analysis.real_repr_probe`
- repr_audit_w2v2.py:28 `i6_experiments.users.wu.experiments.unsupervised_asr.vad_port` (sibling module in same unsupervised_asr package, present in production tree)
- text.py:30 `i6_experiments.users.wu.experiments.posterior_hmm.data.phon_lm` (`_load_g2p_lexicon`, `_load_lexicon_word_to_phon`)
- test_rev_diag.py:29,36 `speech_llm.sae.emc.lattice`, `speech_llm.sae.emc.reverse`
- rev_diag_worker.py:91-94 `speech_llm.sae.emc.lattice`, `speech_llm.sae.emc.prior`, `speech_llm.sae.emc.reverse`, `speech_llm.sae.psi_align`
- userdir/w2vu_rev/rev_term.py:37-39 `speech_llm.sae.emc.prior`, `.lattice`, `.reverse`
- userdir/w2vu_rev/models/wav2vec_u_rev.py:36,39 `fairseq.models`, `unsupervised.models.wav2vec_u` (fairseq examples/wav2vec/unsupervised, shipped in the fairseq wheel, not this repo)
- userdir/w2vu_rev/rev_dataset.py:23 `unsupervised.data.ExtractedFeaturesDataset` (same fairseq examples pkg)
- userdir/w2vu_rev/tasks/unpaired_audio_text_rev.py:24-27 `fairseq.data`, `fairseq.dataclass`, `fairseq.tasks`, `unsupervised.tasks.unpaired_audio_text`
- packed_word_decode_worker.py:19,21-22 `fairseq.checkpoint_utils`, `examples.speech_recognition.new.decoders.decoder`, `examples.speech_recognition.new.infer` (fairseq wheel's `examples` shim, see build_w2vu_env.sh)

Hardcoded absolute paths / cluster settings:
- features.py:20, test_rev_diag.py:7, test_text.py:8 — `/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python`
- text.py:60 default `/e/project1/spell/wu24/env/conda/envs/w2vu/bin/python` (overridable via `gs.W2VU_PYTHON_EXE`, `tk.Path(hash_overwrite="W2VU_ENV_PYTHON")`)
- eval_per.py:90, features.py:43, sparse_autoencoder_probe.py:14 — `/e/project1/spell/wu24/common_hf_home`-style HF cache path
- pipeline.py:92 — `/e/project1/spell/wu24/2026-07-13_unsupervised/output/sae/1a/phoneme_lm_o4.bin`
- test_text.py:20 — `/e/project1/spell/wu24/2026-07-13_unsupervised/tools/w2vu_reference/fs_phonemize_with_sil.py`
- build_w2vu_env.sh: default PREFIX `/e/project1/spell/wu24/env/conda/envs/w2vu`
- Jobs declare `requires_env = "w2vu"`, wired via this setup's (non-version-controlled) settings.py `worker_wrapper`/`_w2vu_env_overrides`, and `W2VU_SHIM_DIR` for the fairseq `examples` shim.

## 2. External dependencies
- fairseq 0.12.2 (pinned, dedicated `w2vu` conda env, py3.9 + CUDA torch 2.6.0/cu126). Reference impl run in-place from the wheel's `examples/wav2vec/unsupervised` (via `unsupervised.*`/`examples.*` imports, not reimplemented). Two Cython extensions (`data_utils_fast`, `token_block_utils_fast`) locally rebuilt against numpy 1.23.5 (build_w2vu_env.sh); numpy pinned <1.24 (fairseq/data/data_utils.py:488 `np.int`). No local source patch/diff of fairseq itself found (userdir/w2vu_rev is an external fairseq user-dir plugin, not a fairseq source patch).
- kenlm==0.3.0, pip-installed in the same `w2vu` env (build_w2vu_env.sh:41); gan.py passes `kenlm_path` (tk.Path) to the job via `task.kenlm_path`.
- No flashlight/wav2letter, rVAD, or faiss import found in w2vu2/*.py or userdir/ (rVAD is referenced only in text.py's docstring as external prior knowledge about measured VAD recall, not as a code dependency in this package).
- REV_USER_DIR = os.path.join(w2vu2 dir, "userdir", "w2vu_rev") (gan.py:44) — points fairseq's `--user-dir` at the plugin.
- W2VU_PYTHON located via `tk.Path(getattr(gs, "W2VU_PYTHON_EXE", <hardcoded path>), hash_overwrite=...)` (text.py:59); jobs must call it explicitly since settings.py rewrites default worker argv[0] to the speech_llm python.

## 3. Production job graph
- §1c GAN entry: `w2vu2/pipeline.py` (PREFIX="sae/1c"), functions building `gan.py`'s `FairseqW2vu2TrainJob`. Selected wav2vec2-L15 seed-0 GAN (the §1d teacher): `FairseqW2vu2TrainJob.HOb2GgtYT7Bc`, alias `sae/1c/w2v2_lv60_l15/gan_l15_sil0.5/s0`, text side `FairseqPreprocessTextJob.bi2B89fES77z` (p_sil=0.5). PER eval `W2vu2PerEvalJob.ptwMk3TuPPYb` = 0.173 dev-clean / 0.214 dev-other (SAE_1c.md:105, and correction at SAE_1c.md:129-133 — earlier Catalog row cited a different, BEST-RQ layer-5 arm `FairseqW2vu2TrainJob.5IPlLRrakdij`, alias `sae/1c/gan_l5_sil0.5/bn40`, flagged as an error 2026-08-17, "numbers and conclusions untouched"). Per-seed table SAE_1c.md:60 shows s0 dev-other=0.214 selected at 15.85 (min perplexity-selected), other seeds 0.205/0.215/0.168 at 16.24/17.45/17.76.
- §1d self-training + decode entry: `w2vu2/selftrain.py`, `config/sae_1d_selftrain.py` (SAE_1d.md Catalog). `GanPseudoLabelJob.xjn6QnNqwEEH` (28,539 utts). CTC student `Wav2Vec2CtcFinetuneJob.BI1uYgPyTeQ0`. Phone decodes `Wav2Vec2CtcDecodeJob.{1C2fmWJR1mcM,qqKPLPBEt1K3,YwTdIJTgPaaF}`. Word decode `Wav2Vec2KenlmDecodeJob.AQw3EcUo6rks` — WER 0.179607/0.218654 (SAE_1d.md:83-85).
- GAN960 / G-track: `ReturnnTrainingJob.HuSkdbuVRg6d` (theta_0^G960, §3d.A scale arm, one 960h pass, 10 sub-epochs); pseudo-label decode `w2vu2/word_decode/PackedWav2Vec2KenlmDecodeJob.{523Wyir5Weg2,K71ryNx3XFZH}`, agreement check `PackedDecodeAgreementJob.FATi7mwI43o7`; downstream AV SFT `ReturnnTrainingJob.1OALJ3Yaa9UL`. Result at sub-ep 10: 13.11 / 16.82 dev-clean/dev-other (SAE_3D_GTRACK.md:113,198-200,7).

## 4. Inputs and port overlap
- Frozen encoder features (BEST-RQ / w2v2-L15) — port has `w2v2/` (RFJv2 L15 dump, per Design decision 3) but built off i6_experiments.common librispeech getters + pinned ffmpeg audio; not verified identical to the GAN's HF-Ogg-derived features (features.py's HF cache path differs from the port's data pipeline).
- SIL-augmented phoneme text (w2vu2/text.py, its own job, explicitly NOT reusing §0b's TextToPhonemeJob because that job's join destroys word boundaries) — port's `lm/` has phone text / lexicon / g2p jobs (Design decision 4) but no SIL-augmented variant found in this search; not confirmed present.
- Lexicon/G2P — text.py reuses `posterior_hmm.data.phon_lm` lexicon lookups verbatim; port uses "i6 common lexicon + i6_core g2p" per Design decision 4 — different construction, not verified equivalent.
- KenLM phone LM (pipeline.py:92 `.../output/sae/1a/phoneme_lm_o4.bin`) — port's Design decision 4 states "KenLM comes from CompileKenLMJob"; a matching KenLM build job may already exist in the port but was not located/confirmed identical to sae/1a's.
- VAD (rVAD, mentioned as external prior in text.py docstring; used by `silence_stats.py`) — port has `data/vad_port.py`, `data/vad.py`; repr_audit_w2v2.py in w2vu2 itself imports the production `unsupervised_asr.vad_port` (not w2vu2-local), so overlap exists but content-identity not verified here.

## 5. i6 restrictions (quoted)
- PORT_WORK.md:13-14: "Port only the phase 4a core. GRPO / speech-LLM, the retired scorer, GAN / wav2vec-U and detailed analysis configs are out. Remove everything from `unsupervised_asr/` that 4a does not use." (This is a standing exclusion of GAN/wav2vec-U from the port scope, stated as a user constraint dated 2026-09-23.)
- PORT_WORK.md:56-57 (Design decision 9): "The port imports only i6_core, `i6_experiments.common`, returnn, sisyphus and third-party code: no other `users/` dirs and no speech-llm."
- SAE_i6.md:10,18: "Pure unsupervised phone recognition without GANs... `SAE_i6_ref.md` section 2 governs: label quarantine, no GAN, final or label-free checkpoints only..."
- SAE_i6.md:20-22 hardware: "i6: L40S 46 GB GPUs, 5 per user; V100 32 GB, 32 of them with no per-user cap... Trainings run on V100 from 2026-09-25 (user decision)."
- MEMORY.md rule "no-gan-pure-unsupervised.md": GAN and supervised inits are analysis-only, never an option or fallback (repo-wide standing rule, not i6-doc-specific but directly applicable).

## 6. Collisions
- Port branch `users/wu/experiments/unsupervised_asr/` currently has NO `w2vu2/` subpackage or `config/w2vu2.py`; `find` for `*w2vu*`/`*gan*` under the worktree only hit an unrelated `users/enrique/jobs/gan`, `users/enrique/experiments/gan`, and `users/enrique/jobs/fairseq/wav2vec/{w2vu_generate_job.py,wav2vec_u_GAN.py}` — a different user's own GAN work, not in `users/wu/...unsupervised_asr/`, no name overlap with a `w2vu2/` addition.
- Port `unsupervised_asr/` top-level names in use: `phones.py`, `model/`, `default_tools.py`, `training/`, `data/`, `w2v2/` (note: singular "w2v2", not "w2vu2" — no direct name collision), `lm/`, `reverse_model/`, `analysis/`, `inputs.py`, `config/`, `README.md`, `env/`, `vad_port.py`, `tests/`. No existing `w2vu2/` or GAN-named file to collide with a new `w2vu2/` addition.
