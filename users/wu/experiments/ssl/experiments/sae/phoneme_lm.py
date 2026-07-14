"""SAE §1.0 / §1a — phoneme n-gram LM over 𝒯_φ (KenLM).

A 4-gram KenLM on the boundary-free phoneme stream 𝒯_φ (from phonemize.py). Serves as (i) the
language-model term of the §1.0 unsupervised metric and (ii) the target-side LM of the §1a
decipherment bootstrap.

KenLM binaries are pre-built once against the cluster's CMake/Boost/Eigen stack and consumed as a
frozen folder (see build note below); CompileKenLMJob is *not* used because the sisyphus job env has
no cmake/boost on PATH — only the login shell's module stack does. Kneser-Ney fails on this ~40-symbol
vocab (no singleton unigrams: n[1]=0), so lmplz must run with --discount_fallback.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from sisyphus import tk

from i6_core.lm.kenlm import KenLMplzJob, CreateBinaryLMJob
from i6_experiments.users.wu.experiments.ssl.experiments.sae.phonemize import phonemize_lm_corpus

LM_PREFIX = "sae/1a"

# Pre-built KenLM binaries (lmplz/build_binary/query). Built once via:
#   cmake .. -DCMAKE_PREFIX_PATH="<Boost>;<Eigen>;<conda>" -DZLIB_ROOT=<conda> -DBZIP2_ROOT=<conda>
#   make -j8
# with CMake 3.31.8 / Boost 1.88.0 / Eigen 3.4.0 from the 2026 EasyBuild stage. Boost resolves via
# the binaries' baked-in RPATH, so no LD_LIBRARY_PATH tweak is needed at run time.
KENLM_BINARY_FOLDER = "/e/project1/spell/wu24/2026-07-13_unsupervised/tools/kenlm/build/bin"


def kenlm_binaries() -> tk.Path:
    return tk.Path(KENLM_BINARY_FOLDER, hash_overwrite="sae_kenlm_binaries")


def phoneme_ngram_lm(order: int = 4, discount_fallback: Sequence[float] = (0.5, 1.0, 1.5)) -> Tuple[tk.Path, tk.Path]:
    """Wire the 𝒯_φ phoneme n-gram LM. Returns (arpa.gz, binary.bin) tk.Paths."""
    tphi = phonemize_lm_corpus()  # same job hash -> reuses the computed 𝒯_φ

    plz = KenLMplzJob(
        text=[tphi],
        order=order,
        interpolate_unigrams=True,
        pruning=None,  # phoneme vocab is tiny -> keep every n-gram
        vocabulary=None,
        discount_fallback=list(discount_fallback),
        kenlm_binary_folder=kenlm_binaries(),
        mem=16,
        time=2,
    )
    plz.add_alias(f"{LM_PREFIX}/kenlm_plz_o{order}")
    tk.register_output(f"{LM_PREFIX}/phoneme_lm_o{order}.arpa.gz", plz.out_lm)

    binlm = CreateBinaryLMJob(arpa_lm=plz.out_lm, kenlm_binary_folder=kenlm_binaries())
    binlm.add_alias(f"{LM_PREFIX}/kenlm_binary_o{order}")
    tk.register_output(f"{LM_PREFIX}/phoneme_lm_o{order}.bin", binlm.out_lm)

    return plz.out_lm, binlm.out_lm
