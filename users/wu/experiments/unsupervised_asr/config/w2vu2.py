"""Entry point: the wav2vec-U 2.0 GAN (section 1c) and its self-training student (section 1d), as a
REPRODUCTION REFERENCE of the reference setup's production runs.

This line is outside the GAN-free research line of this package (``config.base`` and the other
entries): it reproduces a published baseline with its own framework (fairseq 0.12.2) through i6_core
jobs, and nothing of the EMC line depends on it.

What it reproduces (the banked production numbers)
--------------------------------------------------
* 1c, the GAN on wav2vec 2.0 large-lv60 layer-15 features, p_sil 0.5, five seeds
  (``config/sae_1c_w2v2_pilot.py``: ``seed_grid(5)``), the best seed selected by fairseq's best valid
  ``weighted_lm_ppl``; no label is read.  Greedy phone PER (dev-clean / dev-other):

  ==== =============== =========== ======== =========
  seed weighted_lm_ppl best update dev-clean dev-other
  ==== =============== =========== ======== =========
  s0   15.85 SELECTED  148000      0.173    0.214
  s1   16.24           77000       0.162    0.205
  s2   17.45           140000      0.175    0.215
  s3   17.76           66000       0.137    0.168
  s4   63.52           29000       0.851    0.862 (collapsed)
  ==== =============== =========== ======== =========

  The reported system is the selected s0 (0.173 / 0.214); s3's 0.168 is the oracle-best seed and is
  not reportable.  Production trained and evaluated all five seeds (banked
  ``FairseqW2vu2TrainJob.{HOb2GgtYT7Bc, KPeBBDiEJMxT, MbO9o9hBZs2G, zygHGPvCrQZn, otFs6lCBF3wX}``,
  ``common.seed`` 0..4); s4 collapsed and is usually left out of the summary.
* 1d, a wav2vec 2.0 large-lv60 CTC student fine-tuned by fairseq on the selected GAN's greedy
  pseudo-labels of train-clean-100 (40k updates, 4 GPUs, the last checkpoint): viterbi phone PER
  0.1384 / 0.1716, and lexicon + official word 4-gram KenLM decode (beam 500, LM weight 2.0, word
  score -1.0) word WER 0.1796 / 0.2187 (dev-clean / dev-other).

The graph (:func:`py`)
----------------------
1c (``training.w2vu2_gan.get_w2vu2_gan``): the p_sil 0.5 phone text (``lm.w2vu2_text``), the phone
4-gram KenLM (i6_core ``KenLMplzJob`` + ``CreateBinaryLMJob``), the fairseq feature data from the
port's VAD-trimmed L15 stream (``data.w2vu2_features``), one i6_core ``FairseqHydraTrainingJob`` per
seed and ``W2vu2GanSelectJob``.  Then per seed (production evaluated every seed) the generator is
converted to a RETURNN checkpoint and decoded greedily by ``ReturnnForwardJobV2`` (on CPU) on
dev-clean and dev-other, and scored by ``W2vu2GanPerJob`` (``analysis.w2vu2_gan_eval``).  The selected
seed's generator decodes train-clean-100 into the pseudo-labels (``W2vu2GanPseudoLabelJob``).

Intermediate evaluation (not in production): the same conversion, CPU forward and PER chain on each
seed's update checkpoint ``checkpoint_<E>_<U>.pt`` every :data:`INTERMEDIATE_EVAL_INTERVAL` updates, a
point on an epoch end moved one save earlier (:func:`intermediate_eval_updates`,
:func:`_intermediate_checkpoint`).  Each point waits only for its own checkpoint file (a custom
``available`` check, not the training's completion), so it runs while the training runs.  fairseq's
in-training ``uer`` is no substitute: the GAN's ``valid`` split has no labels, so it logs 0.0.
Outputs: ``<GAN_ARM>/intermediate/s<seed>/u<U>/<split>/per.{json,txt}``.

1d (``training.w2vu2_ctc``, ``analysis.w2vu2_ctc_decode``): fairseq manifests of the ogg zips, the
CTC data dir, the LV-60 fairseq checkpoint, the CTC fine-tune as i6_core ``FairseqHydraTrainingJob``,
and fairseq's own ``infer.py`` for the viterbi phone PER and the lexicon + KenLM word WER on
dev-clean and dev-other.

Aliases and outputs mirror production's (``sae/1c/...``, ``sae/1d/...``) and are scoped under
:data:`ALIAS_AND_OUTPUT_PREFIX`: the jobs this entry creates carry the prefix
(``gs.ALIAS_AND_OUTPUT_SUBDIR`` while they are built), so the builders' own aliases land under
``alias/w2vu2/sae/...`` and every result under ``output/w2vu2/sae/...``.  The shared port inputs
(``inputs.get_inputs()``) are built before the prefix is set and keep their ``sae/4a/...`` aliases.

Resources
---------
The rqmt of every GPU job is production's (read off the banked jobs' ``job.save``), with three
unhashed exceptions:

* GAN host memory: ``mem`` 100, not 60 (``training.w2vu2_gan.GAN_RQMT`` states the measured need and
  its source).  The CTC student keeps production's 60 (4 x 15): its measured peaks, 49.8 GiB
  (sisyphus, per-process RSS sum) and 41.0 GiB (sacct MaxRSS, job 976159), are both upper bounds on
  its charged memory and both lie below 60.
* The ``gpu_mem`` of the two trainings: :data:`I6_TRAIN_GPU_MEM` (32, V100) instead of production's 80
  (GH200).  Production's own logs show the need is far below that (fairseq's ``gb_free`` on a 95 GB
  GH200: GAN s0 at least 88.6 free, i.e. about 6.4 GB used; the CTC student at least 84.1 free, about
  11 GB used per GPU).
* The partition of the two trainings.  On i6 the partition comes from ``settings.py``
  ``check_engine_limits``, not from this file, and ``gpu_mem`` cannot reach V100 there: a
  ``ReturnnTrainingJob`` ``run`` task goes to ``GPU_ROUTE_TRAIN`` (gpu_32gb, V100 32 GB), but that
  rule matches the exact class, so a ``FairseqHydraTrainingJob`` is routed like any other GPU job:
  ``gpu_mem`` <= 24 to gpu_24gb (A10) or gpu_48gb (L40S), whichever has more free GPUs, and
  ``gpu_mem`` > 24 to gpu_48gb (L40S).  An explicit ``-p`` in the rqmt's ``sbatch_args`` wins over all
  of these rules.  So the trainings carry ``sbatch_args = ["-p", gs.GPU_ROUTE_TRAIN]`` when the loaded
  ``settings.py`` defines :data:`I6_TRAIN_PARTITION_SETTING` (i6: V100, the user's decision of
  2026-09-25), and no ``sbatch_args`` otherwise (JUPITER; the cluster's own rules apply).  If i6
  renames that constant, the trainings fall back silently to gpu_48gb (L40S): check that the first
  training's ``submit_log.run`` shows ``-p gpu_32gb``.

The train-clean-100 pseudo-label forward and the two CTC decodes keep production's ``gpu_mem`` 40
(:data:`FORWARD_RQMT`; the two CTC decodes' own default), which the i6 rule ``gpu_mem > 24`` sends to
gpu_48gb (L40S 46 GB), as intended.  The dev-clean / dev-other generator forwards (per-seed
``checkpoint_best.pt`` and intermediate) run on CPU (:data:`FORWARD_CPU_RQMT`: production's time, mem
and cpu without a GPU; ``check_engine_limits`` leaves them on the default CPU partition), so every
per-seed PER comes from the same device; ``device`` sits in the RETURNN ``post_config`` and is not
hashed.  The
GAN took 10.5 h and the CTC student 4.25 h on GH200 under an 11.5 h limit (i6's 72 h training floor
also matches only ``ReturnnTrainingJob``); on V100 both will take longer and rely on
``FairseqHydraTrainingJob``'s resumable ``run`` task (fairseq resumes from ``checkpoint_last.pt``).

Environment
-----------
* The fairseq env: ``env/build_w2vu_env.sh <prefix>`` (py3.9, torch 2.6.0, fairseq 0.12.2 with its
  rebuilt Cython extensions, hydra-core 1.0.7, omegaconf 2.0.6, kenlm, flashlight-text 0.0.7).  It
  writes the wrapper ``<prefix>/bin/w2vu-python``.  On i6 build it with ``CUDA_ARCH=70`` on a V100
  node (gpu_32gb): the gate then asserts ``sm_70`` in torch's arch list and
  ``torch.cuda.is_available()`` through the wrapper, i.e. on the GPU type of the trainings, the jobs
  that must not fall back to CPU.  Do not use 89 for the L40S: the x86_64 torch 2.6.0 cu126 wheel is
  built for sm_50, 60, 70, 75, 80, 86 and 90 (``TORCH_CUDA_ARCH_LIST`` in pytorch v2.6.0
  ``.ci/manywheel/build_cuda.sh``), not sm_89 (Ada runs the sm_86 code), so the gate would abort.
  One env serves both GPU types; the same wheel covers V100 (sm_70) and L40S (sm_86), and
  rerunning the gate on an L40S node with ``CUDA_ARCH=86`` is an optional extra check.  On a host
  without a GPU only ``GATE_CUDA=0`` works; that skips both CUDA asserts, so the first-training-log
  check below becomes the only CUDA evidence.
* ``settings.py`` keys this line reads:
    - ``W2VU_PYTHON`` (required) = ``<prefix>/bin/w2vu-python``, the wrapper, not ``bin/python``
      (``w2vu2_tools.get_w2vu_python``);
    - ``W2VU_FAIRSEQ_ROOT`` (optional) = a fairseq v0.12.2 tree; unset, a sparse
      ``CloneGitRepositoryJob`` of the tag is used (``w2vu2_tools.get_fairseq_root``);
    - the port's own keys (``default_tools``): ``FFMPEG_BINARY`` (required), ``FFMPEG_PIN_ACCEPT``,
      ``KENLM_BINARY_PATH``, ``SAE_PYTHON``, ``HF_HOME``.
* The main env (``env/environment.yml``) runs the RETURNN forwards and the in-process jobs; the MFCC
  k-means and the feature-data job import ``torchaudio`` and ``sklearn``.  ``env/environment.yml``
  does not list torchaudio, so run ``env/w2vu2_port_extras.sh <main env prefix>`` once after
  ``install_env.sh``: it pip-installs the torchaudio matching the env's torch (2.7.1, as in the
  reference env) and ensures scikit-learn 1.8.0 (the reference env's), then checks both imports and
  one MFCC call.

Checks on i6 (the evidence available there)
-------------------------------------------
* The artefact tests (``SAE_ARTEFACT_DIR``) need the banked JUPITER outputs and are skipped on i6, so
  a green i6 test run says nothing about equivalence to production.  The evidence for that is the
  JUPITER-run implementer and review reports of 2026-09-25 (``impl_gan_port_{A,B,C,wire}``,
  ``review_gan_port``, ``impl_gan_port_fixes``).
* The first GAN training log (``<job>/work/outputs/*/*/hydra_train.log``) must contain fairseq's
  CUDA banner (``CUDA enviroments for all 1 workers``, fairseq's spelling); without it fairseq has
  silently fallen back to CPU training (``fairseq/trainer.py``: ``cuda = torch.cuda.is_available()
  and not cfg.common.cpu``).  The CTC student's log must show it for all 4 workers.
* No other ``fairseq`` may shadow the env's: the worker's ``PYTHONPATH`` reaches the fairseq jobs
  (i6_core prepends the fairseq root, the wrapper the shim), so a regular ``fairseq`` package on it
  (e.g. a recipe checkout of fairseq) would be imported instead of the env's 0.12.2.  Check that
  ``PYTHONPATH=<fairseq root>:<worker's PYTHONPATH> <W2VU_PYTHON> -c 'import fairseq;
  print(fairseq.__file__, fairseq.__version__)'`` prints the env's site-packages and 0.12.2.

Known deviations from production (implementer reports A, B and C of 2026-09-25)
-------------------------------------------------------------------------------
* Utterance order: the fairseq feature data follow the VAD stream's order (train by shard, dev by
  sorted id), production followed the HF row order; the utterance sets and frame counts are equal.
* Features: the port's L15 features differ from the banked dump by about one fp16 step (max abs diff
  3-4 at values near 2500, min frame cosine 0.99997); lengths and k-means ids are equal.
* GAN decode (report B, s0): on the port features 10 utterances (6 dev-clean, 4 dev-other) differ
  from the banked hypotheses in exactly one frame each; 8 of them follow the feature drift above, 2
  are CPU/GPU near-ties.  On the banked features themselves 4 utterances (1 dev-clean, 3 dev-other)
  flip one near-tie frame (top-2 margin 9e-6 to 4e-4) between the port's CPU forward and production's
  GPU forward.  Dev-other PER 0.214085 against the banked 0.214102 (37952 vs 37955 errors);
  dev-clean equal (33487).
* 1d audio: FLAC from the pinned-ffmpeg ogg zips, at most 1 LSB from production's (0/20 checked
  utterances bit-identical).
* 1d manifest order: zip order, production used the HF order; fairseq's seeded batching therefore
  differs, and the student is not bit-reproducible.
* The MFCC centroids are not bit-reproducible (the fit pool follows the ogg-zip order); the GAN
  trainings themselves are not bit-reproducible either (utterance order).

Run from the setup dir (sisyphus imports the module and calls :func:`py`)::

    tools/sisyphus/sis manager recipe/i6_experiments/users/wu/experiments/unsupervised_asr/config/w2vu2.py
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Dict, Iterator

from sisyphus import gs, tk

__all__ = [
    "PRODUCTION_SEEDS",
    "ALIAS_AND_OUTPUT_PREFIX",
    "GAN_ARM",
    "SELFTRAIN_PREFIX",
    "EVAL_SPLITS",
    "I6_TRAIN_GPU_MEM",
    "I6_TRAIN_PARTITION_SETTING",
    "FORWARD_RQMT",
    "FORWARD_CPU_RQMT",
    "PHONE_DECODE_TIME",
    "INTERMEDIATE_EVAL_INTERVAL",
    "GAN_TRAIN_UTTS",
    "GAN_BATCH_SIZE",
    "GAN_BATCH_SIZE_MULTIPLE",
    "GAN_UPDATES_PER_EPOCH",
    "INTERMEDIATE_EPOCH_END_SHIFT",
    "intermediate_eval_updates",
    "gan_update_checkpoint_name",
    "gan_1c",
    "selftrain_1d",
    "py",
]

#: the seeds production trained and evaluated (``seed_grid(5)``; banked seeds 0..4, see the docstring)
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)
#: ``gs.ALIAS_AND_OUTPUT_SUBDIR`` while this entry builds its jobs
ALIAS_AND_OUTPUT_PREFIX = "w2vu2/"
#: production's 1c arm (``pipeline._prefix(W2V2_LV60_L15)`` + ``gan_l15_sil0.5``); the GAN builder
#: aliases its trainings ``<GAN_ARM>/s<seed>`` and the selection ``<GAN_ARM>/select``
GAN_ARM = "sae/1c/w2v2_lv60_l15/gan_l15_sil0.5"
#: production's 1d prefix (``config/sae_1d_selftrain.py``, ``config/sae_1d_word_decode.py``)
SELFTRAIN_PREFIX = "sae/1d"
#: the evaluated splits (production's GAN PER and 1d decodes)
EVAL_SPLITS = ("dev-clean", "dev-other")
#: GPU memory of the two trainings (V100 32 GB); production asked for 80 (GH200).  It does not pick
#: the partition on i6 (their settings send any non-ReturnnTrainingJob with gpu_mem > 24 to L40S);
#: :data:`I6_TRAIN_PARTITION_SETTING` does (module docstring, Resources)
I6_TRAIN_GPU_MEM = 32
#: the ``settings.py`` constant naming the trainings' partition (i6: ``GPU_ROUTE_TRAIN = "gpu_32gb"``,
#: V100); when the loaded settings define it, the trainings get ``sbatch_args = ["-p", <it>]`` (unhashed)
I6_TRAIN_PARTITION_SETTING = "GPU_ROUTE_TRAIN"
#: production's ``W2vu2PerEvalJob`` and ``GanPseudoLabelJob`` rqmt, for the RETURNN generator forwards
FORWARD_RQMT = {"time_rqmt": 2, "mem_rqmt": 24, "cpu_rqmt": 4, "gpu_mem": 40}
#: :data:`FORWARD_RQMT` without the GPU, for the dev-clean / dev-other generator forwards (``device="cpu"``)
FORWARD_CPU_RQMT = {k: v for k, v in FORWARD_RQMT.items() if k != "gpu_mem"}
#: production's dev-only ``Wav2Vec2CtcDecodeJob.qqKPLPBEt1K3`` time (``CtcPhoneDecodeJob`` defaults to
#: 3 h, the train-including ``decode_all``'s)
PHONE_DECODE_TIME = 2
#: updates between two intermediate evaluations (K = 5000: 30 points per seed up to max_update 150000)
INTERMEDIATE_EVAL_INTERVAL = 5000
#: a grid point on an epoch end moves this many updates earlier (one ``save_interval_updates``): fairseq
#: writes no ``checkpoint_<E>_<U>.pt`` at an epoch end
INTERMEDIATE_EPOCH_END_SHIFT = 1000
#: GAN training utterances (train-clean-100, ``data.librispeech.EXPECTED_UTTS``), the yaml's
#: ``dataset.batch_size`` and fairseq's ``dataset.required_batch_size_multiple`` (default 8, not set in
#: the yaml); all three are asserted at graph time in :func:`gan_1c`
GAN_TRAIN_UTTS = 28_539
GAN_BATCH_SIZE = 160
GAN_BATCH_SIZE_MULTIPLE = 8
#: fairseq updates per GAN epoch.  ``batch_by_size`` (max_sentences 160, multiple 8) gives 178 full
#: batches of 160 and splits the 59 left-over utterances into 56 + 3, so 180 = full batches
#: + (rem >= 8) + (rem % 8 > 0) (checked by the review with fairseq 0.12.2's batch_by_size; production s0
#: has ``checkpoint_823_148000.pt`` = ceil(148000 / 180)).  It fixes E = ceil(U / 180) of
#: ``checkpoint_<E>_<U>.pt``.  180 holds for N_train 28,537 .. 28,543 around 28,539 (overall for 160
#: values in [28,489, 28,800], not a contiguous range: e.g. 28,536 and 28,544 give 179).  If it were
#: wrong, the named files would never appear and the intermediate points would wait, not evaluate a wrong
#: checkpoint.
GAN_UPDATES_PER_EPOCH = (GAN_TRAIN_UTTS // GAN_BATCH_SIZE
                         + int(GAN_TRAIN_UTTS % GAN_BATCH_SIZE >= GAN_BATCH_SIZE_MULTIPLE)
                         + int(GAN_TRAIN_UTTS % GAN_BATCH_SIZE % GAN_BATCH_SIZE_MULTIPLE > 0))
assert GAN_UPDATES_PER_EPOCH == 180, GAN_UPDATES_PER_EPOCH


def intermediate_eval_updates(max_update: int) -> tuple:
    """The evaluated updates: every :data:`INTERMEDIATE_EVAL_INTERVAL` up to ``max_update``, a point on an
    epoch end moved :data:`INTERMEDIATE_EPOCH_END_SHIFT` earlier (150000: 45000, 90000, 135000 -> 44000,
    89000, 134000; 30 points)."""
    out = tuple(u - INTERMEDIATE_EPOCH_END_SHIFT if u % GAN_UPDATES_PER_EPOCH == 0 else u
                for u in range(INTERMEDIATE_EVAL_INTERVAL, max_update + 1, INTERMEDIATE_EVAL_INTERVAL))
    assert len(set(out)) == len(out), out
    return out


def gan_update_checkpoint_name(update: int) -> str:
    """fairseq's name of the save at ``update`` (not an epoch end): ``checkpoint_<E>_<U>.pt``,
    E = ceil(U / :data:`GAN_UPDATES_PER_EPOCH`)."""
    assert update % GAN_UPDATES_PER_EPOCH != 0, (update, GAN_UPDATES_PER_EPOCH)
    return f"checkpoint_{math.ceil(update / GAN_UPDATES_PER_EPOCH)}_{update}.pt"


def _i6_train_rqmt(rqmt: Dict[str, Any]) -> Dict[str, Any]:
    """``rqmt`` with :data:`I6_TRAIN_GPU_MEM` and, if the settings name one, the explicit training
    partition (module docstring, Resources).  Both are unhashed."""
    rqmt = dict(rqmt, gpu_mem=I6_TRAIN_GPU_MEM)
    partition = getattr(gs, I6_TRAIN_PARTITION_SETTING, None)
    if partition:
        rqmt["sbatch_args"] = ["-p", partition]
    return rqmt


def _checkpoint_file_exists(path: tk.Path) -> bool:
    """``available`` check of an intermediate checkpoint Path: the file exists.  fairseq writes
    ``checkpoint_<E>_<U>.pt`` to ``.tmp`` and renames it, so the name never shows a partial file.
    Module level: sisyphus pickles it with the Path (job.save); moving or renaming it breaks the
    unpickling of existing job.save files."""
    return os.path.isfile(path.get_path())


def _intermediate_checkpoint(train, update: int) -> tk.Path:
    """``checkpoints/<gan_update_checkpoint_name(update)>`` of the fairseq GAN ``train``, available as soon
    as the file exists.

    The Path hash is (creator, path), as for ``out_checkpoint_dir.join_right(...)``; the ``available``
    callable is not hashed.  fairseq names a save ``checkpoint_<E>_<U>.pt`` only when it is not at an
    epoch end (asserted in :func:`gan_update_checkpoint_name`)."""
    return tk.Path(f"checkpoints/{gan_update_checkpoint_name(update)}", creator=train,
                   available=_checkpoint_file_exists)


@contextlib.contextmanager
def _scoped() -> Iterator[None]:
    """Build jobs and register outputs under :data:`ALIAS_AND_OUTPUT_PREFIX`; restore afterwards."""
    before = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = ALIAS_AND_OUTPUT_PREFIX
    try:
        yield
    finally:
        gs.ALIAS_AND_OUTPUT_SUBDIR = before


def gan_1c(inputs) -> Dict[str, Any]:
    """Section 1c: the GAN per seed, the selection, the per-seed PER, the pseudo-labels.

    :param inputs: ``inputs.get_inputs()``.
    :return: ``{"gan": W2vu2Gan, "eval": {seed: {"generator", split: {"forward", "per"}}},
        "intermediate": {seed: {update: {"generator", split: {"forward", "per"}}}},
        "select": {"generator", "forward", "labels"}}``.
    """
    from ..analysis.w2vu2_gan_eval import W2vu2GanPerJob, W2vu2GanPseudoLabelJob, w2vu2_forward_job, \
        w2vu2_generator_checkpoint
    from ..data.librispeech import EXPECTED_UTTS
    from ..training.w2vu2_gan import GAN_MAX_UPDATE, get_w2vu2_gan, w2vu2_base_config

    # the constants behind GAN_UPDATES_PER_EPOCH (intermediate checkpoint names)
    assert EXPECTED_UTTS["train-clean-100"] == GAN_TRAIN_UTTS, (EXPECTED_UTTS["train-clean-100"], GAN_TRAIN_UTTS)
    dataset_cfg = w2vu2_base_config()["dataset"]
    assert dataset_cfg["batch_size"] == GAN_BATCH_SIZE, GAN_BATCH_SIZE
    # fairseq 0.12.2 DatasetConfig.required_batch_size_multiple defaults to 8
    assert dataset_cfg.get("required_batch_size_multiple", 8) == GAN_BATCH_SIZE_MULTIPLE, dataset_cfg

    gan = get_w2vu2_gan(seeds=PRODUCTION_SEEDS)
    text_dict = gan.text_data.out_dict
    arm = GAN_ARM.split("/", 2)[2]  # w2vu2_forward_job prefixes its alias with sae/1c/gan

    def dev_eval(fairseq_checkpoint: tk.Path, prefix: str) -> Dict[str, Any]:
        """conversion, CPU forward and PER on :data:`EVAL_SPLITS`; aliases and outputs under ``prefix``
        (relative to :data:`GAN_ARM`)"""
        conv, ckpt = w2vu2_generator_checkpoint(
            fairseq_checkpoint=fairseq_checkpoint, text_dict=text_dict, alias=f"{GAN_ARM}/{prefix}/generator")
        res = {"generator": conv}
        for split in EVAL_SPLITS:
            fwd = w2vu2_forward_job(
                name=f"{arm}/{prefix}/{split}", checkpoint=ckpt, vocab=conv.out_vocab,
                feature_hdfs=list(inputs.vad.out_feature_hdfs[split]), expected_num_seqs=EXPECTED_UTTS[split],
                device="cpu", **FORWARD_CPU_RQMT)
            per = W2vu2GanPerJob(hyps=fwd.out_files["hyps.json"], gold=inputs.gold, split=split)
            per.add_alias(f"{GAN_ARM}/{prefix}/per/{split}")
            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.json", per.out_per)
            tk.register_output(f"{GAN_ARM}/{prefix}/{split}/per.txt", per.out_report)
            res[split] = {"forward": fwd, "per": per}
        return res

    evals: Dict[int, Dict[str, Any]] = {}
    intermediate: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for seed, train in gan.trainings.items():
        train.rqmt.update(_i6_train_rqmt(train.rqmt))  # unhashed (module docstring, Resources)
        evals[seed] = dev_eval(train.out_checkpoint_dir.join_right("checkpoint_best.pt"), f"s{seed}")
        intermediate[seed] = {
            update: dev_eval(_intermediate_checkpoint(train, update), f"intermediate/s{seed}/u{update}")
            for update in intermediate_eval_updates(GAN_MAX_UPDATE)}
    tk.register_output(f"{GAN_ARM}/select/selection.json", gan.selection.out_selection)

    # the selected seed's generator labels train-clean-100 (production: GanPseudoLabelJob on s0)
    conv, ckpt = w2vu2_generator_checkpoint(
        fairseq_checkpoint=gan.selection.out_checkpoint, text_dict=text_dict, alias=f"{GAN_ARM}/select/generator")
    n_train = EXPECTED_UTTS["train-clean-100"]
    fwd = w2vu2_forward_job(
        name=f"{arm}/select/train", checkpoint=ckpt, vocab=conv.out_vocab,
        feature_hdfs=list(inputs.vad.out_feature_hdfs["train"]), expected_num_seqs=n_train, **FORWARD_RQMT)
    labels = W2vu2GanPseudoLabelJob(hyps=[fwd.out_files["hyps.json"]], expected_num_seqs=n_train)
    labels.add_alias(f"{SELFTRAIN_PREFIX}/pseudo_labels")
    tk.register_output(f"{SELFTRAIN_PREFIX}/pseudo_labels.json", labels.out_labels)
    return {"gan": gan, "eval": evals, "intermediate": intermediate,
            "select": {"generator": conv, "forward": fwd, "labels": labels}}


def selftrain_1d(inputs, pseudo_labels: tk.Path) -> Dict[str, Any]:
    """Section 1d: the fairseq CTC student on ``pseudo_labels`` and its decodes on :data:`EVAL_SPLITS`.

    :param inputs: ``inputs.get_inputs()``.
    :param pseudo_labels: ``W2vu2GanPseudoLabelJob.out_labels``.
    :return: ``{"manifests", "data", "train", "phone", "lexicon", "refs", "word"}``.
    """
    from ..analysis.w2vu2_ctc_decode import CtcPhoneDecodeJob, CtcWordDecodeJob, FlashlightLexiconJob, \
        OggZipWordRefsJob
    from ..lm.word_lm import official_4gram_arpa, official_lexicon
    from ..training.w2vu2_ctc import TRAIN_RQMT, FairseqAudioManifestJob, FairseqCtcDataJob, \
        build_ctc_training_job, get_fairseq_w2v2_lv60_checkpoint, get_last_checkpoint
    from ..w2vu2_tools import get_fairseq_root, get_w2vu_python

    p = SELFTRAIN_PREFIX
    python, root = get_w2vu_python(), get_fairseq_root()
    subsets = {"train": "train-clean-100", **{s: s for s in EVAL_SPLITS}}
    manifests = {}
    for name, subset in subsets.items():
        job = FairseqAudioManifestJob(ogg_zip=inputs.ogg_zips[subset], name=name)
        job.add_alias(f"{p}/audio/{name}")
        manifests[name] = job
    dev = {s: manifests[s].out_dir for s in EVAL_SPLITS}

    data = FairseqCtcDataJob(manifest_dir=manifests["train"].out_dir, labels=pseudo_labels, gold=inputs.gold)
    data.add_alias(f"{p}/ctc_data")
    tk.register_output(f"{p}/dict.phn.txt", data.out_dict_phn)

    w2v = get_fairseq_w2v2_lv60_checkpoint()
    w2v.creator.add_alias(f"{p}/w2v2_lv60_ckpt")
    train = build_ctc_training_job(data_dir=data.out_data_dir, w2v_path=w2v, fairseq_python_exe=python,
                                   fairseq_root=root, rqmt=_i6_train_rqmt(TRAIN_RQMT))
    train.add_alias(f"{p}/ctc_finetune")
    ckpt = get_last_checkpoint(train)

    phone = CtcPhoneDecodeJob(manifests=dev, checkpoint=ckpt, dict_phn=data.out_dict_phn, gold=inputs.gold,
                              fairseq_python_exe=python, fairseq_root=root)
    phone.rqmt["time"] = PHONE_DECODE_TIME  # unhashed; production's dev-only decode
    phone.add_alias(f"{p}/decode")
    tk.register_output(f"{p}/per.json", phone.out_per)
    tk.register_output(f"{p}/hyps.json", phone.out_hyps)

    lexicon = FlashlightLexiconJob(lexicon=official_lexicon(), dict_phn=data.out_dict_phn)
    lexicon.add_alias(f"{p}/flashlight_lexicon")
    tk.register_output(f"{p}/lm/lexicon.phn.txt", lexicon.out_lexicon)
    refs = OggZipWordRefsJob(ogg_zips={s: inputs.ogg_zips[s] for s in EVAL_SPLITS})
    refs.add_alias(f"{p}/word_refs")
    tk.register_output(f"{p}/word_refs.json", refs.out_refs)
    word = CtcWordDecodeJob(manifests=dev, checkpoint=ckpt, dict_phn=data.out_dict_phn, lexicon=lexicon.out_lexicon,
                            lm=official_4gram_arpa(), fairseq_python_exe=python, fairseq_root=root,
                            word_refs=refs.out_refs)
    word.add_alias(f"{p}/word_decode")
    tk.register_output(f"{p}/word_wer.json", word.out_wer)
    tk.register_output(f"{p}/word_hyps.json", word.out_hyps)
    return {"manifests": manifests, "data": data, "train": train, "phone": phone, "lexicon": lexicon,
            "refs": refs, "word": word}


def py() -> Dict[str, Any]:
    from ..inputs import get_inputs

    inputs = get_inputs()  # built unscoped: the shared inputs keep their own aliases
    with _scoped():
        c = gan_1c(inputs)
        d = selftrain_1d(inputs, c["select"]["labels"].out_labels)
    return {"inputs": inputs, "1c": c, "1d": d}
