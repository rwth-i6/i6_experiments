"""Consolidate completed frozen results only; no recognition runs or downloads."""


def py():
    import os
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.study_summary_job import register_public_study

    return register_public_study(os.environ.get("ASR_STUDY_OUTPUT_ROOT", gs.OUTPUT_DIR))
