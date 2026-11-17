"""Audit frozen public transcripts; one CPU job, no recognition."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.public_pilots import register_convention_audit

    return register_convention_audit(gs.OUTPUT_DIR)
