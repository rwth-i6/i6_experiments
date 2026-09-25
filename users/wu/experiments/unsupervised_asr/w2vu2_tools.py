"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/text.py
(``W2VU_PYTHON``) and gan.py (``_fairseq_dir``, ``common.user_dir``), with the settings.py
``_w2vu_env_overrides`` / ``W2VU_SHIM_DIR`` of the reference setup moved into the env
(``env/build_w2vu_env.sh``).

The external software of the wav2vec-U 2.0 GAN (section 1c) and of every job that runs fairseq
0.12.2.  Same pattern as ``default_tools.py``: a ``settings.py`` key per tool, and a FIXED
``hash_overwrite`` on every path, so moving the tools to another server moves no job hash.

What a new server edits
-----------------------
In ``settings.py`` the GAN reads:

* ``W2VU_PYTHON`` -- REQUIRED for the GAN jobs (:func:`get_w2vu_python` raises without it).  The
  ``w2vu`` env's python WRAPPER ``<prefix>/bin/w2vu-python`` that ``env/build_w2vu_env.sh`` writes
  (py3.9, torch 2.6.0, fairseq 0.12.2 with the two rebuilt Cython extensions, numpy 1.23.5,
  hydra-core 1.0.7, omegaconf 2.0.6, kenlm 0.3.0, flashlight-text 0.0.7).  The wrapper, not
  ``<prefix>/bin/python``: it sets what the reference setup set for its w2vu jobs in ``settings.py``'s
  ``worker_wrapper`` (``_w2vu_env_overrides``, ``requires_env = "w2vu"``) and in its 1d fine-tuning
  job (``selftrain.py:295``): ``LD_LIBRARY_PATH`` replaced by the env's own ``lib`` and
  ``torch/lib``; ``<prefix>/fairseq_shim`` (the wheel's ``examples`` as a top-level package)
  prepended to ``PYTHONPATH``; ``PYTHONNOUSERSITE=1``; ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1``.
  Without it, a job inherits the main env's ``LD_LIBRARY_PATH`` and the py3.9 interpreter dies with
  ``undefined symbol: PyObject_CallOneArg`` (it dlopens the py3.11 ``libtorch_python``).  There is
  no fallback path: the package names no cluster path.
* ``W2VU_FAIRSEQ_ROOT`` -- OPTIONAL; an existing fairseq v0.12.2 source tree whose
  ``fairseq_cli/``, ``examples/`` and ``fairseq/config/`` are those of the tag.  Without it,
  :func:`get_fairseq_root` checks those three dirs out of the tag with i6_core's
  ``CloneGitRepositoryJob`` (a sparse checkout).

The fairseq root and the library
--------------------------------
i6_core's ``FairseqHydraTrainingJob`` runs ``<python> <fairseq_root>/fairseq_cli/hydra_train.py``
with ``fairseq_root`` prepended to ``PYTHONPATH``.  The sparse checkout has ``fairseq_cli/``,
``examples/`` and ``fairseq/config/``, but no ``fairseq/__init__.py``: its ``fairseq/`` dir is only a
namespace portion, and Python imports the regular package ``fairseq`` of the env's site-packages
(PEP 420: a regular package found later on ``sys.path`` wins over a namespace portion).  So the
library that runs is the env's fairseq 0.12.2 with its rebuilt Cython extensions -- the one the
reference runs used -- while ``fairseq_cli``, ``common.user_dir`` (``examples/wav2vec/unsupervised``)
and hydra's ``config_path`` (``fairseq_cli/../fairseq/config``) come from the tag.  The top-level
``examples`` package (the unsupervised task imports ``examples.speech_recognition.kaldi.
kaldi_decoder``) resolves to the wrapper's shim, i.e. the wheel's copy, as in the reference runs; the
root's ``examples/`` is the fallback.  A full checkout would shadow the env's
fairseq with an unbuilt source tree (no ``data_utils_fast``: ``batch_by_size`` raises).

Checked against the reference env (2026-09-25): the tag's ``fairseq_cli/`` and ``fairseq/config/``
are file-identical to the env's ``site-packages/fairseq_cli`` and ``site-packages/fairseq/config``,
and its ``examples/wav2vec/unsupervised`` to the env's ``site-packages/fairseq/examples/wav2vec/
unsupervised`` (the tag has two more entries, the dangling kaldi symlinks ``kaldi_self_train/st/
{steps,utils}``, which the GAN never reads).
"""

from __future__ import annotations

from sisyphus import gs, tk  # gs: the settings keys listed in the module docstring

__all__ = [
    "FAIRSEQ_URL",
    "FAIRSEQ_VERSION",
    "FAIRSEQ_COMMIT",
    "FAIRSEQ_CHECKOUT_FILES",
    "W2VU_PYTHON_HASH_OVERWRITE",
    "FAIRSEQ_ROOT_HASH_OVERWRITE",
    "get_w2vu_python",
    "get_fairseq_root",
    "get_fairseq_user_dir",
]

FAIRSEQ_URL = "https://github.com/facebookresearch/fairseq"
FAIRSEQ_VERSION = "0.12.2"
#: the commit of the tag ``v0.12.2`` (``git ls-remote``, refs/tags/v0.12.2)
FAIRSEQ_COMMIT = "4a388e64cd646ed7d7ad8de8fae55df2b8eea91d"
#: the sparse checkout (module docstring): the CLI, the examples and hydra's config dir, not the library
FAIRSEQ_CHECKOUT_FILES = ["fairseq_cli", "examples", "fairseq/config"]

#: the fixed hash of ``W2VU_PYTHON``: names the env the reference runs used
W2VU_PYTHON_HASH_OVERWRITE = "UNSUPERVISED_ASR_W2VU_PYTHON_EXE_fairseq-0.12.2"
#: the fixed hash of the fairseq root, cloned or given (both are the tag)
FAIRSEQ_ROOT_HASH_OVERWRITE = f"UNSUPERVISED_ASR_FAIRSEQ_ROOT_v{FAIRSEQ_VERSION}"


def get_w2vu_python() -> tk.Path:
    """``settings.py``'s ``W2VU_PYTHON`` (required; no fallback), the ``w2vu-python`` wrapper."""
    path = getattr(gs, "W2VU_PYTHON", None)
    if not path:
        raise RuntimeError(
            "W2VU_PYTHON is not set in settings.py.  The GAN jobs need it: create the fairseq 0.12.2 env "
            "with env/build_w2vu_env.sh <prefix> and set W2VU_PYTHON = '<prefix>/bin/w2vu-python' (the "
            "wrapper that isolates the env's LD_LIBRARY_PATH; see w2vu2_tools.py).")
    return tk.Path(str(path), hash_overwrite=W2VU_PYTHON_HASH_OVERWRITE)


def get_fairseq_root() -> tk.Path:
    """The fairseq v0.12.2 root: ``settings.py``'s ``W2VU_FAIRSEQ_ROOT`` if set, else the sparse
    ``CloneGitRepositoryJob`` of :data:`FAIRSEQ_CHECKOUT_FILES` at :data:`FAIRSEQ_COMMIT`.  Both carry
    :data:`FAIRSEQ_ROOT_HASH_OVERWRITE`, so the choice moves no downstream hash."""
    given = getattr(gs, "W2VU_FAIRSEQ_ROOT", None)
    if given:
        return tk.Path(str(given), hash_overwrite=FAIRSEQ_ROOT_HASH_OVERWRITE)
    from i6_core.tools.git import CloneGitRepositoryJob

    root = CloneGitRepositoryJob(
        FAIRSEQ_URL,
        commit=FAIRSEQ_COMMIT,
        checkout_folder_name="fairseq",
        files_to_checkout=list(FAIRSEQ_CHECKOUT_FILES),
    ).out_repository.copy()
    root.hash_overwrite = FAIRSEQ_ROOT_HASH_OVERWRITE
    return root


def get_fairseq_user_dir() -> tk.Path:
    """``<fairseq_root>/examples/wav2vec/unsupervised``, the GAN's ``common.user_dir`` (its hash is the
    root's fixed hash plus the relative path)."""
    return get_fairseq_root().join_right("examples/wav2vec/unsupervised")
