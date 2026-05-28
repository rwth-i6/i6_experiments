"""
TTS-encoder project, FZJ (Juelich) variant -- for larger-scale runs.

Mirrors the RZ recipe ``i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder``
and reuses its ``_train_ls_base`` / ``DbMelFeatureExtractor`` directly, so Job hashes are
identical across clusters (an RZ-trained model imports here unchanged); only the output
prefix differs (``exp2026_05_28_tts_encoder_fzj``).

For now: the standard ``base-ls`` baseline only -- it has the same hash as RZ / the FZJ base
setup, so it imports rather than re-trains. To be extended with:
  - ``base-ls-dbmel``: import the finished RZ training (do NOT re-train on FZJ).
  - the TTS-encoder text-util training: multi-GPU, large-scale on the LibriSpeech LM text corpus.

Run from an FZJ setup:
``py7 ./sis m recipe/i6_experiments/users/zeyer/experiments/exp2026_05_28_tts_encoder_fzj.py``
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import _train_ls_base

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_fzj"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # Standard log-mel baseline. Same Job hash as RZ base-ls / the FZJ base setup -> imports, not re-trained.
    _train_ls_base("base-ls", prefix=prefix)

    # TODO next, once the RZ base-ls-dbmel (ReturnnTrainingJob.8mdaueLDfiGP) finishes:
    #   import it here (rsync RZ work/ + import_work_directory), do NOT re-train on FZJ. Add:
    #     import returnn.frontend as rf
    #     from returnn.util.basic import BehaviorVersion
    #     from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import DbMelFeatureExtractor
    #     _train_ls_base(
    #         "base-ls-dbmel",
    #         feature_extraction=rf.build_dict(DbMelFeatureExtractor),
    #         behavior_version=BehaviorVersion._latest_behavior_version,
    #         prefix=prefix,
    #     )
    # TODO: TTS-encoder text-util training -- multi-GPU, large-scale on the LS LM text corpus.
