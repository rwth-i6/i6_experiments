"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/default_tools.py
(plus the tool pins of configs/config_sae_4a_lexlat_k2_pack_v1.py and unsupervised_asr/phoneme_lm.py).

The external software of the phase-4a (EMC) setup.  Every interpreter, tool binary and cache
directory the package uses is named here, and every one carries a FIXED ``hash_overwrite`` string,
so moving a tool to another server (editing a path below) does not move any job hash.

What a new server edits
-----------------------
In ``settings.py`` the package reads exactly:

* ``FFMPEG_BINARY`` -- REQUIRED (:func:`get_ffmpeg_binary` raises without it); the ffmpeg that
  encodes the LibriSpeech audio.  ``data.ffmpeg_pin.FfmpegPinCheckJob`` checks at run time that it
  reproduces the banked audio before any encode job can run;
* ``FFMPEG_PIN_ACCEPT`` -- OPTIONAL, unset by default; a label that lets an ffmpeg which fails the
  pin check run anyway, as a visibly different audio generation (the label moves every downstream
  hash; :func:`get_ffmpeg_pin_accept`, ``data.ffmpeg_pin``);
* ``SAE_PYTHON``, ``HF_HOME`` -- OPTIONAL overrides of the conda env's python (also ``RETURNN_EXE``)
  and the Hugging Face cache; the defaults below are the reference cluster's paths.  The env is the
  one ``env/install_env.sh`` creates (one env, k2 included);
* ``K2_PYTHON`` -- OPTIONAL; the python of a SEPARATE k2 env.  Without it the k2 jobs use
  ``SAE_PYTHON`` (one env).  The reference cluster (JUPITER) must set it: its main env has no k2,
  which was built from source into a clone of the main env
  (``/e/project1/spell/wu24/envs/sae_k2/bin/python``);
* ``KENLM_BINARY_PATH`` -- OPTIONAL; a prebuilt KenLM ``bin/`` dir (built from ``KENLM_COMMIT``)
  used instead of compiling KenLM (:func:`get_kenlm_binary_path`).  Set it where the job env cannot
  compile KenLM (the reference cluster: ``<setup>/tools/kenlm/build/bin``);
* ``G2P_PATH`` and ``G2P_PYTHON`` -- read directly by i6_core's g2p jobs (sequitur ``g2p.py`` and
  its python); not pinned here.

Every one of these but ``FFMPEG_PIN_ACCEPT`` carries a fixed hash (or none), so setting them moves
no job hash.  In THIS file
only ``RETURNN_COMMIT`` / ``KENLM_COMMIT`` remain, edited only if moving off the pins.

``RETURNN_PYTHON_EXE`` / ``RETURNN_ROOT`` in ``settings.py`` are NOT used: every RETURNN job of the
package states ``returnn_python_exe`` (``SAE_PYTHON_EXE`` or ``K2_PYTHON_EXE``) and ``returnn_root``
(``RETURNN_ROOT``) explicitly.  (sisyphus' own settings -- engine, work dir -- are outside this list.)

Names
-----

* ``RETURNN_ROOT`` -- rwth-i6/returnn at 00171dfe (2026-05-18), the commit every banked run used,
  plus the three local fixes the banked runs ran with (shipped as
  ``training/returnn_local_fixes.patch`` and applied by i6_core's ``CloneGitRepositoryJob(patches=...)``).
  No upstream commit carries all three: rwth-i6/returnn master 5f752be4 (2026-09-22) still loads the
  optimizer state without ``weights_only=False`` and still calls ``numpy.fromstring``.

    - ``returnn/torch/engine.py``: the padding-ratio statistic skips extern_data keys with
      packed == padded == 0 (it divided 0/0 after the checkpoint was written);
    - ``returnn/torch/updater.py``: ``torch.load(..., weights_only=False)`` for the optimizer state
      (torch >= 2.6 otherwise refuses ``functools.partial`` and breaks every RESUME, e.g. a
      60-sub-epoch run over the 11.5 h wall clock);
    - ``returnn/util/task_system.py``: ``numpy.frombuffer(...).copy()`` instead of
      ``numpy.fromstring`` (numpy >= 1.22).

* ``SAE_PYTHON_EXE`` -- the main conda env (``speech_llm`` on the reference cluster): the RETURNN
  interpreter of every non-k2 job and the python of jobs that run a child python of their own.
* ``RETURNN_EXE`` -- the same object as ``SAE_PYTHON_EXE`` (kept as the name other modules import).
* ``K2_PYTHON_EXE`` -- the python with k2 (CUDA, torch 2.7.1): ``settings.py``'s ``K2_PYTHON`` if
  set, else ``SAE_PYTHON`` (one env).  The k2 training arms run RETURNN under it, and the HLG build
  jobs run their child scripts under it.  On the reference cluster it is a separate env (a conda
  clone of the main env with k2 built from source), so ``settings.py`` there sets ``K2_PYTHON``.
* ``get_kenlm_binary_path()`` -- KenLM ``bin/`` (``lmplz``, ``build_binary``, ``query``), kpu/kenlm
  at ``KENLM_COMMIT`` (the commit of the reference cluster's prebuilt ``tools/kenlm``).
* ``FFMPEG_BINARY`` -- the ffmpeg that transcodes the openslr LibriSpeech FLAC to 16 kHz Ogg Vorbis
  (``data.librispeech.get_bliss_corpus``: ``BlissChangeEncodingJob`` with the checked wrapper of this
  binary, ``data.librispeech.get_checked_ffmpeg_binary``).  It must reproduce the banked audio (the
  reference is the conda build of ffmpeg 7.1.1): a different build changes the audio, and with it
  every feature, while no hash changes, so ``data.ffmpeg_pin.FfmpegPinCheckJob`` checks it first.
* ``HF_HOME`` -- the (offline) Hugging Face cache.
"""

import os

from sisyphus import gs, tk  # gs: the settings keys listed in the module docstring

from i6_core.lm.kenlm import CompileKenLMJob
from i6_core.tools.git import CloneGitRepositoryJob

# -------------------------------------------------------------------------------------------------
# RETURNN
# -------------------------------------------------------------------------------------------------
#: the RETURNN commit of every banked phase-4a run (``returnn --version``:
#: ``1.20260518.222345+git.00171dfe.dirty``)
RETURNN_COMMIT = "00171dfe252eaabc71f6c1fa5a5a910c11a14788"
#: the three uncommitted fixes the banked runs ran with (``git diff`` of the reference checkout)
RETURNN_PATCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "returnn_local_fixes.patch")
with open(RETURNN_PATCH_FILE, "rt") as _fh:
    RETURNN_LOCAL_FIXES_PATCH = _fh.read()

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    commit=RETURNN_COMMIT,
    checkout_folder_name="returnn",
    patches=[RETURNN_LOCAL_FIXES_PATCH],
).out_repository.copy()
RETURNN_ROOT.hash_overwrite = "UNSUPERVISED_ASR_RETURNN_ROOT_00171dfe_LOCAL_FIXES"

# -------------------------------------------------------------------------------------------------
# Interpreters (``settings.py`` SAE_PYTHON overrides the reference cluster's default; K2_PYTHON
# names a separate k2 env and defaults to SAE_PYTHON)
# -------------------------------------------------------------------------------------------------
#: the conda env (torch 2.7.1, transformers, datasets, h5py, rVADfast, ffmpeg, sequitur; k2 in the
#: env ``env/install_env.sh`` creates, not in the reference cluster's main env)
SAE_PYTHON = str(getattr(gs, "SAE_PYTHON", None) or "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python")
SAE_PYTHON_EXE = tk.Path(SAE_PYTHON, hash_overwrite="UNSUPERVISED_ASR_SAE_PYTHON_EXE")
#: the RETURNN interpreter of every non-k2 job (no ``settings.py`` RETURNN_PYTHON_EXE is read)
RETURNN_EXE = SAE_PYTHON_EXE

#: the python with k2 (k2 ec31d2c9, CUDA 12, torch 2.7.1): ``settings.py`` K2_PYTHON if set, else
#: :data:`SAE_PYTHON` (one env).  The reference cluster sets K2_PYTHON to its separate k2 env
#: (``/e/project1/spell/wu24/envs/sae_k2/bin/python``), because its main env has no k2.  A fixed
#: ``hash_overwrite``: the path may differ per server, the hash does not.
K2_PYTHON = str(getattr(gs, "K2_PYTHON", None) or SAE_PYTHON)
K2_PYTHON_EXE = tk.Path(K2_PYTHON, hash_overwrite="UNSUPERVISED_ASR_K2_PYTHON_EXE")

# -------------------------------------------------------------------------------------------------
# KenLM
# -------------------------------------------------------------------------------------------------
#: the kpu/kenlm commit ``CompileKenLMJob`` builds: the HEAD of the reference cluster's prebuilt
#: ``tools/kenlm`` checkout (2025-03-30)
KENLM_COMMIT = "4cb443e60b7bf2c0ddf3c745378f76cb59e254e5"
#: the fixed hash of the KenLM binaries, prebuilt or compiled (both are ``KENLM_COMMIT``)
KENLM_HASH_OVERWRITE = f"UNSUPERVISED_ASR_KENLM_BINARIES_{KENLM_COMMIT[:8]}"


def get_kenlm_binary_path() -> tk.Path:
    """KenLM ``bin/``: ``settings.py``'s ``KENLM_BINARY_PATH`` (a prebuilt ``bin/`` of
    :data:`KENLM_COMMIT`) if set, else ``CompileKenLMJob`` on kpu/kenlm at :data:`KENLM_COMMIT`.
    Both carry :data:`KENLM_HASH_OVERWRITE`, so the choice does not move a downstream hash."""
    prebuilt = getattr(gs, "KENLM_BINARY_PATH", None)
    if prebuilt:
        return tk.Path(str(prebuilt), hash_overwrite=KENLM_HASH_OVERWRITE)
    kenlm_repo = CloneGitRepositoryJob("https://github.com/kpu/kenlm", commit=KENLM_COMMIT).out_repository.copy()
    path = CompileKenLMJob(repository=kenlm_repo).out_binaries.copy()
    path.hash_overwrite = KENLM_HASH_OVERWRITE
    return path


# -------------------------------------------------------------------------------------------------
# ffmpeg and the HF cache
# -------------------------------------------------------------------------------------------------
#: the fixed hash of ``FFMPEG_BINARY``: names the ffmpeg version the audio was verified with
FFMPEG_HASH_OVERWRITE = "UNSUPERVISED_ASR_FFMPEG_BINARY_ffmpeg-7.1.1"


def get_ffmpeg_binary() -> tk.Path:
    """``settings.py``'s ``FFMPEG_BINARY`` (required; there is no fallback to the ``ffmpeg`` on PATH).

    Must reproduce the banked Ogg audio: the reference is ffmpeg 7.1.1, the reference cluster's conda
    build (``speech_llm/bin/ffmpeg``, sha256
    ``c5eee15fde4da2bc63f783358a30874c272c5ecac4cf7232fe7312b14274f558``).  A different ffmpeg
    changes the audio -- another build (Lavc61.3.100) moved most dev-other waveforms to 20-100 dB
    SNR -- and with it every feature, while the job hashes stay the same (the hash is the fixed
    :data:`FFMPEG_HASH_OVERWRITE`).  So the encode jobs never take this path directly: they take
    ``data.librispeech.get_checked_ffmpeg_binary()``, the output of the run-time check
    ``data.ffmpeg_pin.FfmpegPinCheckJob``.
    """
    path = getattr(gs, "FFMPEG_BINARY", None)
    if not path:
        raise RuntimeError(
            "FFMPEG_BINARY is not set in settings.py.  It is required: set it to the ffmpeg that encodes "
            "the LibriSpeech audio (reference: conda-forge ffmpeg 7.1.1); data.ffmpeg_pin.FfmpegPinCheckJob "
            "checks at run time that it reproduces the banked audio.")
    return tk.Path(str(path), hash_overwrite=FFMPEG_HASH_OVERWRITE)


def get_ffmpeg_pin_accept():
    """``settings.py``'s ``FFMPEG_PIN_ACCEPT`` label, or ``None`` (unset or empty: the strict check).

    Set only to run on audio that is knowingly not the banked audio (``data.ffmpeg_pin`` module
    docstring, README): the label is hashed into ``FfmpegPinCheckJob`` and moves every downstream hash.
    """
    label = getattr(gs, "FFMPEG_PIN_ACCEPT", None)
    return str(label) if label else None


def __getattr__(name):
    # ``FFMPEG_BINARY`` is resolved on first use (PEP 562), so a setup without the settings key can
    # still import this module for everything that does not transcode audio.
    if name == "FFMPEG_BINARY":
        return get_ffmpeg_binary()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: the Hugging Face cache (``HF_HOME``; ``settings.py`` HF_HOME overrides the reference cluster's
#: offline, shared cache).  Holds the pinned LibriSpeech parquet revisions, wav2vec2-large-lv60 and the
#: MFA alignment dataset.
HF_HOME = tk.Path(str(getattr(gs, "HF_HOME", None) or "/e/project1/spell/common_hf_home"),
                  hash_overwrite="UNSUPERVISED_ASR_HF_HOME")
